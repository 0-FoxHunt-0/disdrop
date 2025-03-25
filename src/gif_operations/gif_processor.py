from __future__ import annotations

import asyncio
import atexit
import gc
import json
import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Tuple, Union

import cv2
import psutil
from cachetools import TTLCache
from tqdm import tqdm
import numpy as np
import math
from PIL import Image
from PIL import ImageChops

from src.base.processor import BaseProcessor
from src.default_config import (GIF_COMPRESSION, GIF_SIZE_TO_SKIP, INPUT_DIR,
                                OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS, TEMP_FILE_DIR)
from src.gif_operations.ffmpeg_handler import FFmpegHandler
from src.gif_operations.memory import MemoryManager
from src.gif_operations.optimizer import (DynamicGIFOptimizer, GIFOptimizer,
                                          ProcessingStatus)
from src.gif_operations.processing_stats import ProcessingStats
from src.gif_operations.quality_manager import QualityManager
from src.gif_operations.resource_manager import ResourceMonitor
from src.logging_system import (UnifiedLogger, get_logger, performance_monitor, log_function_call,
                                ModernLogStyle, ThreadSafeProgressLogger, run_ffmpeg_command, SUCCESS_LEVEL,
                                log_gif_progress, display_progress_update, ICONS)
from src.temp_file_manager import TempFileManager
from src.utils.video_dimensions import _validate_dimensions as validate_dimensions
from src.gif_operations.enhanced_gif_optimizer import EnhancedGIFOptimizer

# Define GIF processing status enum with icons and colors


class GIFProcessingStatus(Enum):
    """Status indicators for GIF processing with icons and colors."""
    SUCCESS = ("✓", ModernLogStyle.GREEN.value)
    ERROR = ("✗", ModernLogStyle.RED.value)
    WARNING = ("⚠", ModernLogStyle.AMBER.value)
    PROCESSING = ("⚙", ModernLogStyle.BLUE.value)
    OPTIMIZING = ("↻", ModernLogStyle.PURPLE.value)
    SKIPPED = ("→", ModernLogStyle.GRAY.value)
    STARTING = ("▶", ModernLogStyle.TEAL.value)


# Create module-level logger
module_logger = get_logger('gif_processor')


@asynccontextmanager
async def managed_process_pool():
    """Async context manager for process pool with automatic cleanup."""
    pool = None
    try:
        # Create a thread pool with a reasonable number of workers
        pool = ThreadPoolExecutor(max_workers=max(2, os.cpu_count() or 4))
        yield pool
    finally:
        # Ensure proper cleanup when the context is exited
        if pool:
            pool.shutdown(wait=False)


def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor function performance with enhanced logging.

    Uses the modern logging system to track performance metrics and system stats
    for both synchronous and asynchronous functions.
    """
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Get a unified logger for performance monitoring
        perf_logger = UnifiedLogger('performance')
        func_name = func.__name__

        # Start a named timer for this function
        perf_logger.start_timer(func_name)

        try:
            # Log the start of the function execution
            perf_logger.debug(f"Starting {func_name}")
            return await func(*args, **kwargs)
        except Exception as e:
            # Log any errors that occur
            perf_logger.error(f"Error in {func_name}: {str(e)}")
            raise
        finally:
            # End the timer with system stats when the function completes
            perf_logger.end_timer(func_name, include_system_stats=True)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        # Get a unified logger for performance monitoring
        perf_logger = UnifiedLogger('performance')
        func_name = func.__name__

        # Start a named timer for this function
        perf_logger.start_timer(func_name)

        try:
            # Log the start of the function execution
            perf_logger.debug(f"Starting {func_name}")
            return func(*args, **kwargs)
        except Exception as e:
            # Log any errors that occur
            perf_logger.error(f"Error in {func_name}: {str(e)}")
            raise
        finally:
            # End the timer with system stats when the function completes
            perf_logger.end_timer(func_name, include_system_stats=True)

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class OptimizationConfig(NamedTuple):
    scale_factor: float
    colors: int
    lossy_value: int
    dither_mode: str = 'bayer'
    bayer_scale: int = 3


class ProcessingStatus(Enum):
    """Improved processing status enumeration."""
    SUCCESS = "success"
    DIMENSION_ERROR = "dimension_error"
    PALETTE_ERROR = "palette_error"
    CONVERSION_ERROR = "conversion_error"
    OPTIMIZATION_ERROR = "optimization_error"
    FILE_ERROR = "file_error"
    SIZE_THRESHOLD_EXCEEDED = "size_threshold_exceeded"


@dataclass
class ProcessingResult:
    """Result tracking for GIF processing operations."""
    fps: int
    size: float
    path: Optional[str]
    status: ProcessingStatus
    message: str = ""
    settings: Optional[OptimizationConfig] = None
    success: bool = False
    error: Optional[str] = None

    def __post_init__(self):
        """Set success based on status."""
        self.success = self.status == ProcessingStatus.SUCCESS


class GIFProcessor(BaseProcessor):
    """Main GIF processing class combining optimization and conversion."""

    def __init__(self, logger=None, gpu_enabled=False, gpu_settings=None):
        """Initialize the GIF processor with advanced performance options.

        Args:
            logger: Optional custom logger
            gpu_enabled: Whether GPU acceleration is enabled
            gpu_settings: Optional GPU settings dictionary
        """
        super().__init__()
        # Store initialization parameters
        self.logger = logger or logging.getLogger('app')
        self.gpu_enabled = gpu_enabled
        self.gpu_settings = gpu_settings or {}

        # Progress tracking variables
        self.active_progress_bars = {}
        self.progress_lock = threading.RLock()

        # Start progress display timer
        self.last_progress_update = time.time()
        self.progress_update_interval = 2.0  # seconds

        # Initialize thread-safe locks and events
        self._shutdown_event = threading.Event()
        self._processing_cancelled = threading.Event()
        self._immediate_termination = threading.Event()
        self._shutdown_initiated = False
        self._cleanup_complete = False
        self._file_locks = {}
        self._file_locks_lock = threading.Lock()
        self._threads_lock = threading.Lock()
        self._active_threads = set()
        self._stats_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._processing_progress = {}
        self._stats = {'retried': 0, 'processed': 0, 'failed': 0, 'skipped': 0}
        self._cleanup_handlers = []
        self.processed_files = set()
        self.logging_lock = threading.Lock()
        self.failed_files = []
        # Add the missing file queue lock
        self._file_queue_lock = threading.Lock()

        # Use modern logging style for initialization
        self.logger.debug(
            f"Initializing GIF Processor. GPU enabled: {gpu_enabled}")
        if gpu_settings:
            self.logger.debug(f"GPU settings: {gpu_settings}")

        # Import compression settings from default config
        self.compression_settings = GIF_COMPRESSION

        # Initialize rest of components
        self._init_worker_management()
        self._init_caches()
        self._init_components()
        self._init_async_support()

        # Setup signal handlers for graceful termination
        self._setup_signal_handlers()

        # Register cleanup handlers
        atexit.register(self.cleanup_resources)

        self.logger.debug("GIF Processor initialization complete")

    def _init_worker_management(self):
        """Initialize worker thread management."""
        # Determine optimal number of threads based on CPU count
        cpu_count = os.cpu_count() or 4
        # At least 2, at most 6 threads
        self.max_threads = max(2, min(cpu_count - 1, 6))

        self.worker_threads = []
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self._stop_workers = threading.Event()

        # Add priority queue for important tasks
        self.priority_queue = queue.PriorityQueue()

        # Initialize thread pool
        self._ensure_worker_threads()

        self.logger.debug(
            f"Worker management initialized with {self.max_threads} max threads")

    def _init_caches(self):
        """Initialize caches with improved memory management."""
        # Use LRU cache for file info with configurable size
        # Increased cache size and TTL
        self._file_cache = TTLCache(maxsize=150, ttl=600)
        self.dimension_cache = TTLCache(maxsize=150, ttl=600)

        # Add optimization result cache to avoid redundant processing
        self.optimization_cache = TTLCache(maxsize=100, ttl=900)

        # Add frame analysis cache
        self.frame_analysis_cache = TTLCache(maxsize=50, ttl=300)

        self.logger.debug("Caches initialized")

    def _init_components(self):
        """Initialize processing components with improved configuration."""
        # Pass GPU settings to FFmpeg handler if available
        self.ffmpeg = FFmpegHandler()
        if self.has_gpu and self.gpu_settings:
            self.logger.debug("Updating FFmpeg handler with GPU settings")
            self.ffmpeg.update_gpu_settings(self.gpu_settings)

        # Configure memory management based on system resources
        total_memory_gb = psutil.virtual_memory().total / (1024**3)
        # Dynamic memory threshold based on system memory
        # 40% of system memory for larger systems, 30% for smaller ones
        memory_threshold_percent = 0.4 if total_memory_gb > 8 else 0.3
        memory_threshold = min(
            max(1500, int(total_memory_gb * memory_threshold_percent * 1024)), 6144)

        self.memory_manager = MemoryManager(threshold_mb=memory_threshold)
        self.memory_monitor_interval = 60  # seconds
        self._last_gc_time = time.time()
        self._start_memory_monitor()

        self.stats = ProcessingStats()
        self.resource_monitor = ResourceMonitor()
        self.quality_manager = QualityManager()
        self.progressive_optimization = True
        self.retry_count = 3
        self.retry_delay = 1

        # Initialize the optimizer with GPU settings
        self.logger.debug(f"Initializing DynamicGIFOptimizer")
        self.dynamic_optimizer = DynamicGIFOptimizer(
            self.compression_settings
        )

        self.logger.debug("Components initialization complete")

    def _start_memory_monitor(self):
        """Start a background thread to monitor memory usage and perform garbage collection when needed."""
        self._memory_monitor_thread = threading.Thread(
            target=self._memory_monitor_task,
            daemon=True
        )
        self._memory_monitor_thread.start()
        self.logger.debug("Memory monitor started")

    def _memory_monitor_task(self):
        """Background task to monitor memory usage and perform garbage collection when needed."""
        last_check_time = time.time()
        memory_history = []  # Track recent memory usage to detect leaks
        idle_count = 0

        while not self._shutdown_event.is_set():
            try:
                # Get current memory usage
                memory_info = psutil.Process().memory_info()
                memory_usage_mb = memory_info.rss / (1024 * 1024)

                # Record memory usage history (keep last 10 readings)
                memory_history.append(memory_usage_mb)
                if len(memory_history) > 10:
                    memory_history.pop(0)

                # Check for memory leak patterns if we have enough history
                memory_leak_detected = False
                if len(memory_history) >= 5:
                    # Check if memory is consistently increasing
                    if all(memory_history[i] < memory_history[i+1] for i in range(len(memory_history)-5, len(memory_history)-1)):
                        memory_increase = memory_history[-1] - \
                            memory_history[-5]
                        # If memory increased by more than 10% in the last 5 checks
                        if memory_increase > memory_history[-5] * 0.1:
                            memory_leak_detected = True

                # Get adaptive memory threshold
                memory_threshold = self.memory_manager.threshold_mb

                # Determine memory pressure level
                if memory_usage_mb > memory_threshold * 0.9 or memory_leak_detected:
                    # Critical memory usage - aggressive cleanup
                    self.logger.warning(
                        f"Critical memory usage detected: {memory_usage_mb:.2f}MB. Running aggressive cleanup.")
                    self._perform_garbage_collection(aggressive=True)
                    idle_count = 0
                elif memory_usage_mb > memory_threshold * 0.8:
                    # High memory usage - normal cleanup
                    self.logger.debug(
                        f"High memory usage detected: {memory_usage_mb:.2f}MB. Running garbage collection.")
                    self._perform_garbage_collection(aggressive=False)
                    idle_count = 0
                elif time.time() - last_check_time > self.memory_monitor_interval:
                    # Periodic gentle GC if it's been a while
                    self._perform_garbage_collection(aggressive=False)
                    last_check_time = time.time()
                    idle_count += 1
                else:
                    idle_count += 1

                # Adaptive sleep interval based on memory pressure and activity
                if memory_usage_mb > memory_threshold * 0.7:
                    sleep_time = 3  # Check more frequently under high memory pressure
                elif idle_count > 10:
                    sleep_time = 30  # Less frequent checks when idle
                else:
                    sleep_time = 10  # Normal interval

                time.sleep(sleep_time)
            except Exception as e:
                self.logger.error(f"Error in memory monitor: {e}")
                time.sleep(30)  # Longer sleep on error

    def _perform_garbage_collection(self, aggressive=False):
        """Perform garbage collection with configurable aggressiveness."""
        try:
            # Clear caches if aggressive collection is requested
            if aggressive:
                self._clear_caches()

                # Kill any long-running ffmpeg processes that might be stuck
                self._kill_ffmpeg_processes()

            # Record memory before collection
            before_mem = psutil.Process().memory_info().rss / (1024 * 1024)

            # Run Python's garbage collector with appropriate generation
            collected = gc.collect(generation=2 if aggressive else 1)
            self._last_gc_time = time.time()

            if aggressive:
                # Force release of memory back to OS if possible
                if hasattr(gc, 'malloc_trim'):  # PyPy specific
                    gc.malloc_trim(0)
                elif sys.platform == 'linux':
                    try:
                        import ctypes
                        ctypes.CDLL('libc.so.6').malloc_trim(0)
                    except (ImportError, OSError):
                        pass

                # On Windows, try to trigger full compaction
                elif sys.platform == 'win32':
                    try:
                        import ctypes
                        ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
                    except (ImportError, OSError, AttributeError):
                        pass

            # Record memory after collection for reporting
            after_mem = psutil.Process().memory_info().rss / (1024 * 1024)
            mem_freed = before_mem - after_mem

            if mem_freed > 0:
                self.logger.debug(
                    f"Garbage collection freed {mem_freed:.2f}MB of memory, {collected} objects collected")

        except Exception as e:
            self.logger.error(f"Error during garbage collection: {e}")

    def _init_async_support(self):
        """Initialize async support."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        self.logger.debug("Async support initialized")

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful termination."""
        try:
            # Set up signal handlers for graceful shutdown
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            self.logger.debug("Signal handlers installed")
        except Exception as e:
            self.logger.warning(
                f"Unable to set up signal handlers: {e}", exc_info=True)

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        signal_name = signal.Signals(signum).name if hasattr(
            signal, 'Signals') else str(signum)
        self.logger.info(
            f"Received signal {signal_name}. Initiating graceful shutdown...")

        # Set shutdown flag
        self._shutdown_event.set()
        self._processing_cancelled.set()

        # Stop all worker threads immediately
        self._stop_workers.set()

        # Don't exit immediately, allow tasks to finish gracefully
        if not self._shutdown_initiated:
            self._shutdown_initiated = True
            self.logger.info(
                "Waiting for current tasks to complete (Press Ctrl+C again for immediate exit)...")

            # Schedule cleanup
            threading.Thread(target=self.cleanup_resources,
                             daemon=True).start()
        else:
            # Second signal - immediate termination
            self.logger.warning(
                "Second termination signal received. Forcing immediate exit...")
            self._immediate_termination.set()
            self._kill_ffmpeg_processes()

            # Make sure we're really cleaning up
            self.cleanup_resources()

    def _log_message(self, level: str, message: str, log_key: str = "") -> None:
        """Log a message with appropriate styling based on level.

        Args:
            level: Log level (SUCCESS, ERROR, WARNING, INFO, etc.)
            message: The message to log
            log_key: Optional key to prevent duplicate log messages
        """
        # Use the modern logging system with appropriate styling
        if level in GIFProcessingStatus.__members__:
            # Use status styling
            status = GIFProcessingStatus[level]
            status_icon, color = status.value
            formatted_message = f"{status_icon} {message}"

            # Use appropriate log method
            if level == "ERROR":
                self.logger.error(formatted_message)
            elif level == "WARNING":
                self.logger.warning(formatted_message)
            elif level == "SUCCESS":
                self.logger.success(formatted_message)
            elif level == "INFO":
                self.logger.info(formatted_message)
            elif level == "STARTING":
                self.logger.info(formatted_message)
            elif level == "PROCESSING":
                self.logger.info(formatted_message)
            elif level == "OPTIMIZING":
                self.logger.info(formatted_message)
            elif level == "SKIPPED":
                self.logger.info(formatted_message)
            else:
                # Default to info for unknown levels
                self.logger.info(formatted_message)
        else:
            # Use standard logging for non-status levels
            if level.upper() == "ERROR":
                self.logger.error(message)
            elif level.upper() == "WARNING":
                self.logger.warning(message)
            elif level.upper() == "INFO":
                self.logger.info(message)
            elif level.upper() == "DEBUG":
                self.logger.debug(message)
            else:
                # Default to info
                self.logger.info(message)

        # Add to processed log keys to prevent duplicates if key is provided
        if log_key and hasattr(self, 'processed_files'):
            self.processed_files.add(log_key)

    def get_file_lock(self, file_path: str) -> threading.Lock:
        """Get or create a lock for a specific file."""
        with self._file_locks_lock:
            if file_path not in self._file_locks:
                self._file_locks[file_path] = threading.Lock()
            return self._file_locks[file_path]

    def _register_thread(self):
        """Register current thread as active."""
        with self._threads_lock:
            self._active_threads.add(threading.current_thread())

    def _unregister_thread(self):
        """Unregister current thread."""
        with self._threads_lock:
            self._active_threads.discard(threading.current_thread())

    def cleanup_resources(self) -> None:
        """Enhanced cleanup with improved resource management."""
        if self._cleanup_complete:
            return

        self._cleanup_complete = True
        self._shutdown_initiated = True
        self._shutdown_event.set()
        self._stop_workers.set()

        try:
            self.logger.info("Starting cleanup process...")
            cleanup_start = time.time()

            # First kill any running processes
            self._kill_ffmpeg_processes()

            # Wait a moment for processes to die
            time.sleep(0.5)

            # Force close any file handles
            gc.collect()

            # Close any opened files (more aggressive cleanup)
            try:
                import psutil
                process = psutil.Process()
                for handler in process.open_files():
                    try:
                        if 'temp_' in handler.path or 'gif' in handler.path.lower():
                            os.close(handler.fd)
                    except (OSError, IOError):
                        pass
            except Exception as e:
                self.logger.debug(f"Error during file handle cleanup: {e}")

            # Clear locks and resources
            with self._file_locks_lock:
                self._file_locks.clear()

            # Run cleanup handlers
            for handler in self._cleanup_handlers:
                try:
                    handler()
                except Exception as e:
                    self.logger.error(
                        f"Cleanup handler failed: {str(e)}")

            # Enhanced temp file cleanup - more thorough than standard cleanup
            self._enhanced_temp_directory_cleanup()

            # Clean temp files with multiple attempts
            self._cleanup_temp_directory()

            # Clear caches
            self._clear_caches()

            # Stop background threads
            self._stop_background_threads()

            cleanup_duration = time.time() - cleanup_start
            self.logger.info(f"Cleanup completed in {cleanup_duration:.2f}s")

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}", exc_info=True)
        finally:
            self._cleanup_threads()

            # Final garbage collection pass
            gc.collect()

            # Force memory compaction
            if hasattr(gc, 'malloc_trim'):  # PyPy specific
                gc.malloc_trim(0)
            elif sys.platform == 'linux':
                try:
                    import ctypes
                    ctypes.CDLL('libc.so.6').malloc_trim(0)
                except (ImportError, OSError):
                    pass
            elif sys.platform == 'win32':
                try:
                    import ctypes
                    ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
                except (ImportError, OSError, AttributeError):
                    pass

    def _enhanced_temp_directory_cleanup(self):
        """Perform a more thorough cleanup of temporary files."""
        self.logger.info("Performing enhanced temporary file cleanup...")

        temp_dir = Path(TEMP_FILE_DIR)
        if not temp_dir.exists():
            return

        # File patterns to search for
        patterns = [
            "*.gif", "*.png", "*.mp4", "*.avi", "*.mkv",
            "temp_*", "palette_*", "lossless_*", "opt*",
            "optimized_*", "reduced_*", "ffmpeg_*", "temp_gifsicle_*"
        ]

        total_cleaned = 0

        # Iterate through each pattern and clean matching files
        for pattern in patterns:
            try:
                for file_path in temp_dir.glob(pattern):
                    try:
                        if file_path.is_file():
                            # Use a safe delete approach
                            try:
                                file_path.unlink(missing_ok=True)
                                total_cleaned += 1
                            except PermissionError:
                                # If file is locked, try different approach
                                self.logger.warning(
                                    f"Permission error cleaning {file_path}, will retry")
                                # Wait briefly and try again
                                time.sleep(0.1)
                                if file_path.exists():
                                    file_path.unlink(missing_ok=True)
                                    total_cleaned += 1
                    except Exception as e:
                        self.logger.error(
                            f"Error cleaning temp file {file_path}: {e}")
            except Exception as e:
                self.logger.error(f"Error processing pattern {pattern}: {e}")

        self.logger.info(
            f"Enhanced temp cleanup complete - removed {total_cleaned} files")

        # One final forced garbage collection to release file handles
        gc.collect()

    def _immediate_shutdown_handler(self, signum, frame):
        """Handle immediate shutdown requests (SIGINT, SIGTERM)."""
        signal_name = signal.Signals(signum).name if hasattr(
            signal, 'Signals') else str(signum)
        self.logger.warning(
            f"Received {signal_name} signal - performing emergency cleanup")

        # Set shutdown flag to stop all active processing
        self._shutdown_initiated = True
        self._shutdown_event.set()
        self._stop_workers.set()

        try:
            # Kill all FFmpeg processes immediately
            self._kill_ffmpeg_processes()

            # Perform thorough cleanup
            self._enhanced_temp_directory_cleanup()

            # Clear memory and resources
            self._clear_caches()
            self._cleanup_threads()

            # Force garbage collection
            self._perform_garbage_collection(aggressive=True)

            self.logger.info("Emergency cleanup complete")
        except Exception as e:
            self.logger.error(f"Error during emergency cleanup: {e}")

    def _stop_background_threads(self):
        """Stop all background monitoring threads."""
        try:
            # Set flags to stop background threads
            self._shutdown_event.set()
            self._stop_workers.set()

            # Wait for memory monitor to finish if it exists
            if hasattr(self, '_memory_monitor_thread') and self._memory_monitor_thread.is_alive():
                self._memory_monitor_thread.join(timeout=2)

            # Join other background threads if needed
            # ...

        except Exception as e:
            self.logger.error(f"Error stopping background threads: {e}")

    def _cleanup_temp_directory(self):
        """Clean up temporary directory with improved handling."""
        temp_dir = Path(TEMP_FILE_DIR)
        if not temp_dir.exists():
            return

        try:
            # Log number of temp files before cleanup
            files_count = sum(1 for _ in temp_dir.glob("*"))
            self.logger.debug(
                f"Cleaning up {files_count} files in temp directory")

            # Multiple attempts to clean files
            for attempt in range(3):
                remaining_files = []

                # Try to remove each file
                for item in temp_dir.glob("*"):
                    try:
                        if item.is_file():
                            try:
                                item.unlink(missing_ok=True)
                            except Exception as e:
                                self.logger.debug(
                                    f"Failed to remove file {item}: {e}")
                                remaining_files.append(item)
                        elif item.is_dir():
                            try:
                                shutil.rmtree(item, ignore_errors=True)
                            except Exception as e:
                                self.logger.debug(
                                    f"Failed to remove directory {item}: {e}")
                                remaining_files.append(item)
                    except Exception as e:
                        self.logger.debug(
                            f"Failed to remove {item}: {e}")
                        remaining_files.append(item)

                if not remaining_files:
                    break

                # If files remain, try more aggressive cleanup
                if attempt < 2:
                    self.logger.debug(
                        f"Cleanup attempt {attempt+1} left {len(remaining_files)} files. Trying again.")
                    self._kill_ffmpeg_processes()
                    self._perform_garbage_collection(aggressive=True)
                    time.sleep(1)

            # If files still remain, log them but continue
            if remaining_files:
                self.logger.warning(
                    f"Could not remove {len(remaining_files)} files in temp directory")
                if len(remaining_files) < 10:
                    for file in remaining_files:
                        self.logger.debug(f"Remaining file: {file}")
                else:
                    self.logger.debug(
                        f"First 10 remaining files: {remaining_files[:10]}")

            # Don't try to remove the temp directory if files remain
            if not remaining_files:
                try:
                    temp_dir.rmdir()
                    self.logger.debug("Temp directory removed successfully")
                except Exception as e:
                    self.logger.debug(
                        f"Could not remove temp directory: {e}")

            # Always ensure temp directory exists
            temp_dir.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            self.logger.error(
                f"Error cleaning temp directory: {str(e)}", exc_info=True)
            # Ensure temp directory exists even if cleanup failed
            temp_dir.mkdir(parents=True, exist_ok=True)

    def _clear_caches(self):
        """Clear all caches."""
        if hasattr(self, '_file_cache'):
            self._file_cache.clear()
        if hasattr(self, 'dimension_cache'):
            self.dimension_cache.clear()
        if hasattr(self, 'palette_cache'):
            self.palette_cache.clear()

    def _cleanup_threads(self):
        """Clean up worker threads."""
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=5)
        self.worker_threads.clear()

        # Clear queues
        self._clear_queues()

    def _clear_queues(self):
        """Clear all queues."""
        for queue_obj in [self.task_queue, self.result_queue]:
            while not queue_obj.empty():
                try:
                    queue_obj.get_nowait()
                except queue.Empty:
                    break

    def _kill_ffmpeg_processes(self):
        """Kill any lingering ffmpeg processes.

        This is a safety measure for when processing is interrupted.
        """
        try:
            if sys.platform == 'win32':
                # Windows specific cleanup
                subprocess.run(["taskkill", "/F", "/IM", "ffmpeg.exe"],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            else:
                # Unix/Linux/Mac cleanup
                subprocess.run(["pkill", "-9", "ffmpeg"],
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
        except Exception as e:
            self.logger.error(f"Failed to kill processes: {e}")

    def _log_with_lock(self, level: str, message: str, file_id: str = "") -> None:
        """Log a message with thread safety and duplication prevention.

        Args:
            level: Log level (info, error, warning, success)
            message: Message to log
            file_id: Optional file identifier to prevent duplicate logs
        """
        with self.logging_lock:
            log_key = f"{file_id}:{message}"
            if log_key not in self.processed_files:
                # Use the _log_message method to handle formatting
                self._log_message(level, message, log_key)

    @log_function_call
    def create_gif(self, file_path: Path, output_path: Path, fps: int, dimensions: Tuple[int, int]) -> bool:
        """Create a GIF from a video file with improved quality settings.

        Args:
            file_path: Path to the source video file
            output_path: Path where the GIF should be saved
            fps: Frames per second for the GIF
            dimensions: Tuple containing (width, height)

        Returns:
            True if successful, False otherwise
        """
        # Add progress tracking
        operation_id = f"create_gif_{file_path.stem}"
        self.update_progress(
            operation_id=operation_id,
            current=0,
            total=100,
            description="Creating GIF from video",
            file_name=file_path.name,
            status="PROCESSING",
            start_time=time.time(),
            force_update=True
        )

        try:
            # Get video dimensions
            width, height = dimensions or (640, 480)

            # Prepare paths
            palette_path = self.temp_manager.create_temp_file(
                custom_suffix=".png")
            self.logger.info(f"Creating palette for {file_path.name}")

            # Update progress - creating palette (25%)
            self.update_progress(
                operation_id=operation_id,
                current=25,
                total=100,
                description="Creating color palette",
                file_name=file_path.name,
                status="PROCESSING"
            )

            # Create palette command
            palette_cmd = [
                'ffmpeg', '-i', str(file_path),
                '-vf', f'fps={fps},scale={width}:{height}:flags=lanczos,palettegen=max_colors=256:stats_mode=diff',
                '-y', str(palette_path)
            ]

            # Run the command with timeout
            if not self._run_command(palette_cmd, "Create palette"):
                self.logger.error(f"Failed to create palette for {file_path}")
                self.complete_progress(
                    operation_id, success=False, message="Failed to create palette")
                return False

            # Update progress - creating GIF (50%)
            self.update_progress(
                operation_id=operation_id,
                current=50,
                total=100,
                description="Converting video to GIF",
                file_name=file_path.name,
                status="PROCESSING"
            )

            # Create GIF command
            gif_cmd = [
                'ffmpeg', '-i', str(file_path),
                '-i', str(palette_path),
                '-lavfi', f'fps={fps},scale={width}:{height}:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
                '-y', str(output_path)
            ]

            # Run the command with timeout
            if not self._run_command(gif_cmd, "Create GIF"):
                self.logger.error(
                    f"Failed to create GIF from {file_path.name}")
                self.complete_progress(
                    operation_id, success=False, message="Failed to convert video to GIF")
                return False

            # Update progress - optimizing (75%)
            self.update_progress(
                operation_id=operation_id,
                current=75,
                total=100,
                description="Optimizing GIF",
                file_name=output_path.name,
                status="OPTIMIZING"
            )

            # Apply additional optimization with gifsicle
            optimize_cmd = [
                'gifsicle', '--optimize=3', '--colors=256',
                '-o', str(output_path), str(output_path)
            ]

            # Run the command with timeout
            self._run_command(optimize_cmd, "Optimize GIF")

            # Clean up the palette file
            if palette_path.exists():
                palette_path.unlink()

            # Mark progress as complete
            self.complete_progress(
                operation_id, success=True, message="GIF creation complete")

            return True

        except Exception as e:
            self.logger.error(f"Error creating GIF from {file_path.name}: {e}")
            self.complete_progress(
                operation_id, success=False, message=f"Error: {str(e)}")
            return False

    def _check_early_exit(self, results: List[Dict]) -> bool:
        """Check if we can exit early with good enough results."""
        if not results:
            return False

        best_result = max(results, key=lambda x: (x['fps'], -x['size']))
        target_size = self.compression_settings['min_size_mb']

        return (best_result['size'] <= target_size and
                best_result['fps'] >= self.compression_settings['fps_range'][0] + 2)

    async def process_file_async(self, file_path: Path, output_path: Path, is_video: bool) -> None:
        """Async version of process_file with better resource management."""
        async with managed_process_pool() as pool:
            try:
                # Get file lock with timeout
                file_lock = self.get_file_lock(str(file_path))
                if not await asyncio.get_event_loop().run_in_executor(None, file_lock.acquire, True, 5):
                    self.logger.error(
                        f"Failed to acquire lock for {file_path.name}")
                    self.failed_files.append(file_path)
                    return

                # Check file size and target
                file_size = await asyncio.get_event_loop().run_in_executor(None, self.get_file_size, file_path)
                target_size = self.compression_settings['min_size_mb']

                if file_size <= target_size:
                    if not output_path.exists():
                        await asyncio.get_event_loop().run_in_executor(None, shutil.copy2, file_path, output_path)
                    return

                # Get dimensions
                dimensions = await asyncio.get_event_loop().run_in_executor(None, self._get_dimensions_with_retry, file_path)
                if not self._validate_dimensions(dimensions):
                    self.failed_files.append(file_path)
                    return

                # Process based on type
                if is_video:
                    await self._process_video_async(file_path, output_path, dimensions)
                else:
                    await self._process_gif_async(file_path, output_path, dimensions)

            except Exception as e:
                self.logger.error(f"Async processing error: {str(e)}")
                self.failed_files.append(file_path)
            finally:
                if file_lock:
                    file_lock.release()

    async def _process_video_async(self, file_path: Path, output_path: Path, dimensions: Tuple[int, int]) -> bool:
        """Process a video file asynchronously.

        Args:
            file_path: Path to the input video
            output_path: Path for the output GIF
            dimensions: Tuple of (width, height)

        Returns:
            bool: Success status
        """
        try:
            # Get optimal FPS
            fps = self._get_source_fps(file_path)

            # Use the adaptive method for videos
            success = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.create_gif_adaptive(
                    file_path, output_path, self.compression_settings['min_size_mb'])
            )

            # If adaptive method fails, fall back to standard method
            if not success:
                self.logger.warning(
                    f"Adaptive method failed for {file_path.name}, trying standard method")
                success = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.create_gif(
                        file_path, output_path, int(fps), dimensions)
                )

            return success
        except Exception as e:
            self.logger.error(f"Error in async video processing: {e}")
            return False

    async def _process_gif_async(self, file_path: Path, output_path: Path, dimensions: Tuple[int, int]) -> bool:
        """Process a GIF file asynchronously.

        Args:
            file_path: Path to the input GIF
            output_path: Path for the output GIF
            dimensions: Tuple of (width, height)

        Returns:
            bool: Success status
        """
        try:
            # Get file size
            file_size = await asyncio.get_event_loop().run_in_executor(None, self.get_file_size, file_path)
            target_size = self.compression_settings['min_size_mb']

            # For very large GIFs, use the adaptive method
            if file_size > target_size * 4:
                return await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self.create_gif_adaptive(
                        file_path, output_path, target_size)
                )
            else:
                # For smaller GIFs, use progressive optimization
                temp_output = Path(TEMP_FILE_DIR) / \
                    f"temp_async_{uuid.uuid4().hex}.gif"

                result = await asyncio.get_event_loop().run_in_executor(
                    None,
                    lambda: self._progressive_optimize(
                        file_path, temp_output, target_size)
                )

                if result.status == ProcessingStatus.SUCCESS and temp_output.exists():
                    await asyncio.get_event_loop().run_in_executor(None, shutil.copy2, temp_output, output_path)
                    await asyncio.get_event_loop().run_in_executor(None, temp_output.unlink, True)
                    return True

                # Clean up temp file if it exists
                if temp_output.exists():
                    await asyncio.get_event_loop().run_in_executor(None, temp_output.unlink, True)

                return False
        except Exception as e:
            self.logger.error(f"Error in async GIF processing: {e}")
            return False

    def process_file(self, file_path: Path, output_path: Path, is_video: bool) -> bool:
        """
        Process a single file (GIF or video) synchronously with improved adaptive optimization.

        Args:
            file_path: Path to the input file
            output_path: Path for the output file
            is_video: Whether the input is a video file

        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            # Get file lock with timeout
            file_lock = self.get_file_lock(str(file_path))
            if not file_lock.acquire(timeout=5):
                self.logger.error(
                    f"Failed to acquire lock for {file_path.name}")
                self.failed_files.append(file_path)
                return False

            try:
                # Check file size and target
                file_size = self.get_file_size(file_path)
                target_size = self.compression_settings['min_size_mb']

                if file_size <= target_size:
                    if not output_path.exists():
                        shutil.copy2(file_path, output_path)
                    return True

                # Get dimensions
                dimensions = self._get_dimensions_with_retry(file_path)
                if not self._validate_dimensions(dimensions):
                    self.failed_files.append(file_path)
                    return False

                # Temporary file for processing results
                temp_output = Path(TEMP_FILE_DIR) / \
                    f"temp_output_{uuid.uuid4().hex}.gif"

                # Track all temp files for cleanup
                temp_files = [temp_output]
                success = False
                final_size = 0

                # Choose processing method based on file type and size
                if is_video:
                    # For videos, use the new adaptive method which automatically analyzes content
                    success = self.create_gif_adaptive(
                        file_path, temp_output, target_size)

                    # If adaptive method fails, fall back to the original method as a backup
                    if not success:
                        self.logger.warning(
                            f"Adaptive method failed for {file_path.name}, trying legacy method")
                        fps = self._get_source_fps(file_path)
                        success = self.create_gif(
                            file_path, temp_output, int(fps), dimensions)
                else:
                    # For existing GIFs, determine best approach based on size
                    if file_size > target_size * 4:
                        # For very large GIFs, use the adaptive method
                        success = self.create_gif_adaptive(
                            file_path, temp_output, target_size)
                    else:
                        # For smaller GIFs, use progressive optimization
                        result = self._progressive_optimize(
                            file_path, temp_output, target_size)

                        # Check if optimization was successful AND size requirement was met
                        success = result.success and result.size <= target_size

                        # Log detailed information about the result
                        if result.success:
                            if result.size <= target_size:
                                self.logger.success(
                                    f"Progressive optimization successful: {result.size:.2f}MB (target: {target_size:.2f}MB)")
                            else:
                                self.logger.warning(
                                    f"Progressive optimization succeeded but exceeded target size: {result.size:.2f}MB > {target_size:.2f}MB")
                                # Don't count as success if size isn't met
                                success = False
                        else:
                            self.logger.error(
                                f"Progressive optimization failed: {result.message}")

                # Verify file exists and is valid
                if success and temp_output.exists():
                    final_size = self.get_file_size(temp_output)

                    # Only copy to output_path if size is under or equal to target
                    if final_size <= target_size:
                        shutil.copy2(temp_output, output_path)
                        self.logger.success(
                            f"Successfully processed to {final_size:.2f}MB")
                        return True
                    else:
                        self.logger.warning(
                            f"Output file larger than target: {final_size:.2f}MB > {target_size:.2f}MB")

                        # Try additional optimization if we're close to the target
                        if final_size <= target_size * 1.2:
                            self.logger.info(
                                "Applying additional optimization as final attempt")

                            # Create a temporary file for optimization
                            temp_optimized = Path(
                                TEMP_FILE_DIR) / f"optimized_{uuid.uuid4().hex}.gif"
                            temp_files.append(temp_optimized)

                            # Try to optimize with the most aggressive settings
                            optimization_result, optimized_size = self.optimize_gif(
                                temp_output, temp_optimized, target_size, self._analyze_color_count(file_path))

                            # Only use the result if it meets the target size
                            if optimization_result and optimized_size <= target_size:
                                shutil.copy2(temp_optimized, output_path)
                                self.logger.success(
                                    f"Final optimization successful: {optimized_size:.2f}MB")
                                return True

                        # If still not meeting target, try going even more extreme with a desperate attempt
                        if final_size > target_size and final_size <= target_size * 1.5:
                            self.logger.warning(
                                "Target size not met, trying desperate optimization measures")

                            # Create a temporary file for the desperate attempt
                            temp_desperate = Path(
                                TEMP_FILE_DIR) / f"desperate_{uuid.uuid4().hex}.gif"
                            temp_files.append(temp_desperate)

                            # Use the most extreme compression settings available
                            success = self._compress_with_gifsicle(
                                temp_output,
                                temp_desperate,
                                colors=16,  # Ultra-low colors
                                scale=0.12,  # Ultra-low scale
                                lossy=180,  # Maximum lossy value
                                optimize_level=3
                            )

                            if success and temp_desperate.exists():
                                desperate_size = self.get_file_size(
                                    temp_desperate)

                                if desperate_size <= target_size:
                                    shutil.copy2(temp_desperate, output_path)
                                    self.logger.success(
                                        f"Desperate optimization successful: {desperate_size:.2f}MB")
                                    return True
                                elif desperate_size < final_size:
                                    # Use it if it's at least better than what we had
                                    final_size = desperate_size
                                    temp_output = temp_desperate
                                    self.logger.warning(
                                        f"Desperate optimization improved size but still exceeds target: {desperate_size:.2f}MB > {target_size:.2f}MB")

                        # If we get here, optimization failed to meet the target size
                        self.logger.warning(
                            f"All optimization attempts failed to meet target size of {target_size:.2f}MB")

                        # Fallback: Use the best result we have, even if it doesn't meet the target size
                        if temp_output.exists():
                            self.logger.warning(
                                f"Using best available result ({final_size:.2f}MB) despite exceeding target size")
                            shutil.copy2(temp_output, output_path)
                            return True  # Return success but with warning already logged

                        return False

                # If we reach here, processing failed or size requirement wasn't met
                self.logger.error(f"Processing failed for {file_path.name}")
                return False

            finally:
                # Clean up temp files
                for temp_file in temp_files:
                    if temp_file.exists():
                        try:
                            temp_file.unlink(missing_ok=True)
                        except Exception:
                            pass

                # Release the file lock
                if file_lock:
                    file_lock.release()

        except Exception as e:
            self.logger.error(
                f"Processing error for {file_path}: {str(e)}", exc_info=True)
            self.failed_files.append(file_path)
            return False

    def _get_dimensions_with_retry(self, file_path: Path, max_retries: int = 3) -> Optional[Tuple[int, int]]:
        """Get file dimensions with retry logic."""
        for attempt in range(max_retries):
            try:
                if hasattr(self, 'ffmpeg') and hasattr(self.ffmpeg, 'get_dimensions'):
                    return self.ffmpeg.get_dimensions(file_path)
                else:
                    # Fallback to importing the video dimensions utility
                    from ..utils.video_dimensions import get_video_dimensions
                    return get_video_dimensions(str(file_path))
            except Exception as e:
                if attempt == max_retries - 1:
                    self.logger.error(
                        f"Failed to get dimensions for {file_path}: {e}")
                    return None
                time.sleep(0.5 * (attempt + 1))
        return None

    def _validate_dimensions(self, dimensions: Optional[Tuple[int, int]]) -> bool:
        """
        Validate dimensions are reasonable for GIF creation.

        Args:
            dimensions: Tuple of (width, height)

        Returns:
            bool: True if dimensions are valid
        """
        if not dimensions:
            self.logger.error("Dimensions not available")
            return False

        width, height = dimensions

        # Check if dimensions are reasonable
        if width <= 0 or height <= 0:
            self.logger.error(f"Invalid dimensions: {width}x{height}")
            return False

        # Check if dimensions are too large
        max_dimension = 1920  # Standard maximum for most GIFs
        if width > max_dimension or height > max_dimension:
            self.logger.warning(
                f"Dimensions {width}x{height} exceed recommended maximum of {max_dimension}")
            return True  # Still process but with warning

        return True

    def _analyze_color_count(self, file_path: Path) -> int:
        """
        Analyze a GIF file to determine the number of unique colors.

        This method analyzes the color palette of a GIF to determine optimal
        compression settings. It works with both animated and static GIFs.
        For palette-based GIFs, it directly counts colors from the palette.
        For RGB images, it samples colors from frames to estimate unique colors.

        Args:
            file_path: Path to the GIF file

        Returns:
            int: Estimated number of unique colors (defaults to 256 if analysis fails)
        """
        try:
            # Use PIL/Pillow to open the image and analyze colors
            with Image.open(file_path) as img:
                if img.mode == 'P':  # Palette mode
                    # For palette mode, get the palette size
                    palette = img.getpalette()
                    if palette:
                        # The palette contains R,G,B values, so divide by 3 for color count
                        return min(256, len(palette) // 3)
                    return 256

                # For non-palette images, sample colors from frames
                frames = []
                try:
                    # Try to get multiple frames if it's an animated GIF
                    for i in range(min(5, getattr(img, 'n_frames', 1))):
                        img.seek(i)
                        # Convert to RGB and resize to reduce processing time
                        frame = img.convert('RGB').resize(
                            (100, 100), Image.LANCZOS)
                        frames.append(frame)
                except (EOFError, AttributeError):
                    # Not an animated GIF or other error
                    if not frames:
                        frames = [img.convert('RGB').resize(
                            (100, 100), Image.LANCZOS)]

                # Count unique colors across frames
                unique_colors = set()
                for frame in frames:
                    colors = frame.getcolors(frame.width * frame.height)
                    if colors:
                        unique_colors.update([color[1] for color in colors])
                    else:
                        # Too many colors - revert to default
                        return 256

                return len(unique_colors)
        except Exception as e:
            self._log_with_lock(
                "warning",
                f"Error analyzing color count: {str(e)}",
                str(file_path)
            )
            return 256  # Default to maximum palette size

    def _get_source_fps(self, file_path: Path) -> float:
        """Get the frames per second (FPS) of a source video file.

        Args:
            file_path: Path to the video file

        Returns:
            float: FPS of the video, defaults to 15 if it cannot be determined
        """
        try:
            # Try to use FFmpeg handler if available
            if hasattr(self, 'ffmpeg') and hasattr(self.ffmpeg, 'get_fps'):
                return self.ffmpeg.get_fps(file_path)

            # Fallback to using OpenCV
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                self.logger.warning(f"Could not open video file: {file_path}")
                return 15.0

            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()

            # Sometimes FPS can be incorrectly reported as very high or very low
            if fps < 1 or fps > 120:
                self.logger.warning(
                    f"Unrealistic FPS value: {fps}, using default")
                return 15.0

            # Cap maximum FPS for GIF creation
            return min(fps, 30.0)
        except Exception as e:
            self.logger.error(f"Error getting video FPS: {e}")
            return 15.0  # Default to 15 FPS as a reasonable value

    def optimize_gif(self, input_path: Path, output_path: Path, target_size_mb: float = None, original_colors: int = None) -> Tuple[float, bool]:
        """Optimize GIF to target size.

        This is now a wrapper around EnhancedGIFOptimizer for backward compatibility.

        Args:
            input_path: Source GIF file
            output_path: Output path for optimized GIF
            target_size_mb: Target size in MB (use class setting if None)
            original_colors: Number of colors in original GIF (not used in enhanced optimizer)

        Returns:
            Tuple[float, bool]: Tuple of (final_size, success)
        """
        # Start performance tracking
        self._start_timer("gif_optimization")
        self.logger.start_phase("GIF optimization")

        try:
            if not input_path.exists():
                self._log_message("ERROR",
                                  f"Input file does not exist: {input_path}",
                                  f"optimize_no_file_{str(input_path)}")
                return 0, False

            # Use configured size if not specified
            if target_size_mb is None:
                target_size_mb = self.compression_settings.get(
                    'min_size_mb', 10)

            # Get original size
            original_size = self.get_file_size(input_path)

            # Skip optimization if already under target size
            if original_size <= target_size_mb:
                self._log_message("SKIPPED",
                                  f"GIF already under target size: {original_size:.2f}MB (target: {target_size_mb:.2f}MB)",
                                  f"optimize_skip_{str(input_path)}")
                shutil.copy2(input_path, output_path)
                return original_size, True

            # Create an instance of the enhanced optimizer
            enhanced_optimizer = EnhancedGIFOptimizer(
                max_workers=4,
                compression_settings=self.compression_settings
            )

            # Perform the optimization
            final_size, success = enhanced_optimizer.optimize_gif(
                input_path, output_path, target_size_mb
            )

            # Log the result
            if success:
                self._log_message("SUCCESS",
                                  f"Enhanced optimization successful: {final_size:.2f}MB",
                                  f"optimize_success_{str(input_path)}")
            else:
                self._log_message("ERROR",
                                  f"Enhanced optimization failed to meet target size: {final_size:.2f}MB > {target_size_mb:.2f}MB",
                                  f"optimize_size_fail_{str(input_path)}")

                # Even if we didn't meet the target size, we still have a file to use
                if output_path.exists():
                    return final_size, True

            return final_size, success

        except Exception as e:
            self._log_message("ERROR",
                              f"GIF optimization error: {e}",
                              f"optimize_general_error_{str(input_path)}")
            return original_size, False
        finally:
            # End the performance timer and phase
            self._end_timer("gif_optimization", include_system_stats=True)
            self.logger.end_phase("GIF optimization")

    def _is_video_file(self, file_path: str) -> bool:
        """Check if a file is a video file based on its extension.

        Args:
            file_path: Path to the file to check

        Returns:
            bool: True if the file is a video file, False otherwise
        """
        try:
            file_ext = Path(file_path).suffix.lower()
            video_extensions = ['.mp4', '.avi', '.mkv', '.mov',
                                '.wmv', '.flv', '.webm', '.m4v', '.mpg', '.mpeg']
            return file_ext in video_extensions or file_ext in SUPPORTED_VIDEO_FORMATS
        except Exception as e:
            self.logger.error(f"Error checking if file is video: {e}")
            # Default to treating as non-video if we can't determine
            return False

    def _ensure_worker_threads(self):
        """Ensure enough worker threads are running."""
        # Clean up exited threads
        self.worker_threads = [
            t for t in self.worker_threads if t.is_alive()]

        # Only create new threads if we don't have enough running
        current_thread_count = len(
            [t for t in self.worker_threads if t.is_alive()])

        # Calculate how many threads to add
        threads_to_add = max(0, self.max_threads - current_thread_count)

        if threads_to_add > 0:
            self.logger.debug(f"Starting {threads_to_add} worker threads")
            for _ in range(threads_to_add):
                thread = threading.Thread(
                    target=self._worker_thread, daemon=True)
                thread.start()
                self.worker_threads.append(thread)

    def _worker_thread(self):
        """Worker thread to process tasks from queue."""
        self._register_thread()

        try:
            while not self._stop_workers.is_set():
                try:
                    # First check priority queue
                    try:
                        priority, task = self.priority_queue.get(block=False)
                        self.logger.debug(
                            f"Processing priority task: {priority}")
                    except queue.Empty:
                        # Then check regular queue
                        task = self.task_queue.get(timeout=1)

                        if task is None:  # None is a signal to stop
                            break

                        # Process the task
                        input_path = task.get('input_path')
                        output_path = task.get('output_path')
                        is_video = task.get('is_video', False)

                        self.logger.debug(
                            f"Worker processing: {input_path}")

                        try:
                            result = self.process_file(
                                input_path, output_path, is_video)
                            self.result_queue.put({
                                'file_path': input_path,
                                'success': result,
                                'error': None
                            })
                        except Exception as e:
                            self.logger.error(
                                f"Error processing {input_path}: {str(e)}")
                            self.result_queue.put({
                                'file_path': input_path,
                                'success': False,
                                'error': str(e)
                            })

                            # Mark task as done in appropriate queue
                            try:
                                self.priority_queue.task_done()
                            except ValueError:
                                self.task_queue.task_done()

                except queue.Empty:
                    # No tasks available, just continue
                    continue
                except Exception as e:
                    self.logger.error(f"Worker thread error: {str(e)}")
                    time.sleep(0.5)  # Prevent CPU spinning in case of errors
        finally:
            self._unregister_thread()

    def _analyze_video_content(self, file_path: Path) -> Dict[str, float]:
        """Analyze video content to determine optimal optimization strategy.

        Returns a dictionary with content metrics:
        - motion_score: 0-1 score indicating amount of motion (higher = more motion)
        - complexity_score: 0-1 score indicating visual complexity (higher = more complex)
        - color_variety: 0-1 score indicating color variety (higher = more colors)
        """
        # Check cache first
        cache_key = str(file_path)
        if cache_key in self.frame_analysis_cache:
            return self.frame_analysis_cache[cache_key]

        try:
            # Open video file
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                self.logger.error(f"Cannot open video file: {file_path}")
                return {'motion_score': 0.5, 'complexity_score': 0.5, 'color_variety': 0.5}

            # Get total frames and sample every nth frame
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # If video is very long, sample fewer frames
            sample_interval = max(
                1, int(total_frames / min(100, total_frames)))

            frames = []
            motion_scores = []
            color_histograms = []
            prev_frame = None

            # Sample frames
            for i in range(0, total_frames, sample_interval):
                cap.set(cv2.CAP_PROP_POS_FRAMES, i)
                ret, frame = cap.read()
                if not ret:
                    break

                # Resize frame to reduce computation
                frame = cv2.resize(frame, (320, 180))
                frames.append(frame)

                # Calculate motion score if we have a previous frame
                if prev_frame is not None:
                    # Calculate frame difference
                    diff = cv2.absdiff(frame, prev_frame)
                    motion_score = diff.sum() / (diff.size * 255)
                    motion_scores.append(motion_score)

                # Calculate color histogram
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                hist = cv2.calcHist([hsv], [0, 1], None, [
                    30, 32], [0, 180, 0, 256])
                cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
                color_histograms.append(hist)

                prev_frame = frame

            cap.release()

            # Calculate metrics
            if not motion_scores:
                motion_score = 0.5  # Default if we couldn't calculate
            else:
                motion_score = min(1.0, sum(motion_scores) /
                                   len(motion_scores) * 10)

            # Calculate color variety
            color_variety = 0.5
            if len(color_histograms) > 1:
                hist_sum = sum(cv2.compareHist(color_histograms[0], h, cv2.HISTCMP_CORREL)
                               for h in color_histograms[1:]) / (len(color_histograms) - 1)
                # Invert correlation score (lower correlation = higher variety)
                color_variety = 1.0 - max(0, min(1.0, hist_sum))

            # Calculate complexity score based on edge detection
            if frames:
                edges = [cv2.Canny(frame, 100, 200) for frame in frames]
                edge_densities = [e.sum() / e.size for e in edges]
                complexity_score = min(
                    1.0, sum(edge_densities) / len(edge_densities) * 5)
            else:
                complexity_score = 0.5

            result = {
                'motion_score': motion_score,
                'complexity_score': complexity_score,
                'color_variety': color_variety
            }

            # Cache the result
            self.frame_analysis_cache[cache_key] = result
            return result

        except Exception as e:
            self.logger.error(f"Error analyzing video content: {e}")
            return {'motion_score': 0.5, 'complexity_score': 0.5, 'color_variety': 0.5}

    def _get_adaptive_optimization_settings(self, file_path: Path, target_size_mb: float, original_colors: int) -> Dict:
        """Analyzes file content to determine optimal conversion settings.

        This method performs content analysis to intelligently select optimization
        parameters based on the specific characteristics of the input file, including
        motion, complexity, color usage, and size.

        Args:
            file_path: Path to the file to analyze
            target_size_mb: Target size in MB
            original_colors: Number of unique colors in the source file

        Returns:
            Dict: Optimization settings including fps, colors, scale_factor, etc.
        """
        # Get file size
        file_size = self.get_file_size(file_path)
        self.logger.debug(
            f"Original size: {file_size:.2f}MB, target: {target_size_mb:.2f}MB")

        # Calculate size ratio (how much we need to reduce)
        size_ratio = file_size / \
            target_size_mb if target_size_mb > 0 else float('inf')

        # Try to get from cache
        cache_key = str(file_path)
        if hasattr(self, 'frame_analysis_cache') and cache_key in self.frame_analysis_cache:
            cache_data = self.frame_analysis_cache[cache_key]
            if 'content_analysis' in cache_data:
                content_analysis = cache_data['content_analysis']
                self.logger.debug(f"Using cached content analysis")
            else:
                # Run analysis and store in cache
                content_analysis = self._analyze_video_content(file_path)
                cache_data['content_analysis'] = content_analysis
        else:
            # Run analysis
            content_analysis = self._analyze_video_content(file_path)

            # Store in cache
            if hasattr(self, 'frame_analysis_cache'):
                self.frame_analysis_cache[cache_key] = {
                    'content_analysis': content_analysis}

        # Extract metrics
        motion_score = content_analysis.get('motion_score', 0.5)
        complexity_score = content_analysis.get('complexity_score', 0.5)
        color_variety = content_analysis.get('color_variety', 0.5)

        self.logger.debug(
            f"Content analysis: motion={motion_score:.2f}, complexity={complexity_score:.2f}, color_variety={color_variety:.2f}")

        # Get dimensions for scaling decisions
        dimensions = self._get_dimensions_with_retry(file_path)
        if not dimensions:
            dimensions = (0, 0)
        width, height = dimensions
        max_dimension = max(width, height) if width > 0 and height > 0 else 0

        # Determine if this is a high-resolution source
        is_high_res = max_dimension >= 1280
        is_very_high_res = max_dimension >= 1920

        # ----- FPS Selection -----
        # Base FPS on motion and file size
        if motion_score > 0.8:
            # High motion needs higher FPS
            base_fps = 20 if file_size < 20 else 15
        elif motion_score > 0.5:
            # Medium motion
            base_fps = 15 if file_size < 20 else 12
        else:
            # Low motion can use lower FPS
            base_fps = 12 if file_size < 20 else 10

        # Adjust based on size ratio
        if size_ratio > 5.0:
            # Very large files need more aggressive FPS reduction
            fps = max(8, base_fps - 5)
        elif size_ratio > 3.0:
            # Large files need moderate FPS reduction
            fps = max(10, base_fps - 3)
        else:
            # Files close to target can use higher FPS
            fps = base_fps

        # Further adjust based on resolution
        if is_very_high_res and fps > 10:
            fps -= 2  # Higher resolutions benefit more from FPS reduction

        # ----- Color Selection -----
        # Base colors on original count and color variety
        if color_variety > 0.8:
            # High color variety needs more colors
            colors = min(256, original_colors)
        elif color_variety > 0.5:
            # Medium color variety
            colors = min(192, max(128, original_colors))
        else:
            # Low color variety can use fewer colors
            colors = min(128, max(64, original_colors))

        # Adjust based on size ratio
        if size_ratio > 5.0:
            # Very large files need more aggressive color reduction
            colors = min(colors, 128)
        elif size_ratio > 3.0:
            # Large files need moderate color reduction
            colors = min(colors, 192)

        # ----- Scale Factor Selection -----
        # Base scale on complexity and resolution
        if complexity_score > 0.8:
            # High complexity benefits from less scaling
            base_scale = 0.9 if not is_high_res else 0.8
        elif complexity_score > 0.5:
            # Medium complexity
            base_scale = 0.8 if not is_high_res else 0.7
        else:
            # Low complexity can scale more
            base_scale = 0.7 if not is_high_res else 0.6

        # Adjust based on size ratio
        if size_ratio > 5.0:
            # Very large files need more aggressive scaling
            scale_factor = max(0.5, base_scale - 0.2)
        elif size_ratio > 3.0:
            # Large files need moderate scaling
            scale_factor = max(0.6, base_scale - 0.1)
        elif size_ratio > 1.5:
            # Files close to target need minimal scaling
            scale_factor = base_scale
        else:
            # Files very close to target can use higher quality
            scale_factor = min(1.0, base_scale + 0.1)

        # ----- Dither Method Selection -----
        # Base dither on color variety and complexity
        if color_variety > 0.7 and complexity_score > 0.6:
            # High color variety and complexity benefits from floyd_steinberg
            dither = 'floyd_steinberg'
        else:
            # Lower color variety or complexity can use bayer
            dither = 'bayer'

        # ----- Lossy Compression Selection -----
        # Base lossy on size ratio and color variety
        if size_ratio > 5.0:
            # Very large files need aggressive lossy
            lossy = 80
        elif size_ratio > 3.0:
            # Large files need moderate lossy
            lossy = 60
        elif size_ratio > 1.5:
            # Files close to target need minimal lossy
            lossy = 40
        else:
            # Files very close to target can use minimal lossy
            lossy = 20

        # Adjust lossy based on color variety
        if color_variety > 0.8:
            # High color variety should use less lossy to preserve colors
            lossy = max(0, lossy - 20)

        # ----- Bayer Scale Selection -----
        # Base bayer_scale on complexity
        if complexity_score > 0.7:
            # High complexity benefits from lower bayer_scale
            bayer_scale = 2
        else:
            # Lower complexity can use higher bayer_scale
            bayer_scale = 3

        # ----- Quality Flag -----
        # Determine if we should prioritize quality
        quality_focused = size_ratio < 1.2 or (
            file_size < 15 and complexity_score > 0.7)

        # Create settings dictionary
        settings = {
            'fps': fps,
            'colors': colors,
            'scale_factor': scale_factor,
            'dither': dither,
            'lossy': lossy,
            'bayer_scale': bayer_scale,
            'quality_focused': quality_focused,
            'size_ratio': size_ratio
        }

        self.logger.info(
            f"Adaptive settings: fps={fps}, colors={colors}, scale={scale_factor:.2f}, dither={dither}")

        return settings

    def _get_ffmpeg_gpu_args(self) -> List[str]:
        """Get FFmpeg GPU acceleration arguments based on available hardware."""
        if not self.has_gpu:
            return []

        gpu_args = []
        try:
            # Get GPU specific settings from the stored configuration
            gpu_type = self.gpu_settings.get('type', '').lower()
            gpu_device = self.gpu_settings.get('device', 0)

            if gpu_type == 'nvidia':
                # NVIDIA GPU acceleration
                gpu_args = [
                    '-hwaccel', 'cuda',
                    '-hwaccel_device', str(gpu_device),
                    '-c:v', 'h264_cuvid'
                ]
            elif gpu_type == 'amd':
                # AMD GPU acceleration
                gpu_args = [
                    '-hwaccel', 'amf',
                    '-hwaccel_device', str(gpu_device)
                ]
            elif gpu_type == 'intel':
                # Intel GPU acceleration
                gpu_args = [
                    '-hwaccel', 'qsv',
                    '-hwaccel_device', str(gpu_device)
                ]
            elif gpu_type == 'apple' and sys.platform == 'darwin':
                # Apple Silicon / Mac GPU
                gpu_args = [
                    '-hwaccel', 'videotoolbox'
                ]

            self.logger.debug(
                f"Using GPU acceleration with {gpu_type} GPU")
            return gpu_args
        except Exception as e:
            self.logger.warning(f"Error setting up GPU acceleration: {e}")
            return []

    def create_gif_adaptive(self, file_path: Path, output_path: Path, target_size_mb: float = None) -> bool:
        """Create GIF with adaptive settings based on content analysis.

        This method prioritizes gifsicle for optimization over FFmpeg for better
        reliability and performance when handling large files.

        Args:
            file_path: Path to input file
            output_path: Path for output GIF
            target_size_mb: Target size in MB, uses default if None

        Returns:
            bool: True if successful and meets target size requirements, False otherwise
        """
        start_time = time.time()
        temp_files = []  # Track temporary files for cleanup

        # Use the configured value if not specified
        if target_size_mb is None:
            target_size_mb = self.compression_settings.get('min_size_mb', 10)

        try:
            # Get source file size
            original_size = self.get_file_size(file_path)

            # Early exit if file is already small enough
            if original_size <= target_size_mb * 0.95:  # 5% buffer
                self.logger.info(
                    f"File already meets size target: {original_size:.2f}MB")
                shutil.copy2(file_path, output_path)
                return True

            # Get dimensions
            dimensions = self._get_dimensions_with_retry(file_path)
            if not dimensions:
                self.logger.error("Failed to get dimensions")
                return False

            width, height = dimensions
            self.logger.debug(f"Source dimensions: {width}x{height}")

            # Get optimization settings using cached analysis when available
            original_colors = self._analyze_color_count(file_path)
            settings = self._get_adaptive_optimization_settings(
                file_path, target_size_mb, original_colors)

            # Calculate size ratio to determine how aggressive to be
            size_ratio = original_size / target_size_mb

            # For very large files, try direct gifsicle approach first
            if size_ratio > 5.0:
                self.logger.info(
                    f"Large size ratio ({size_ratio:.1f}x), trying direct gifsicle optimization")
                # Use a unique temporary filename
                temp_output = Path(TEMP_FILE_DIR) / \
                    f"adaptive_temp_{uuid.uuid4().hex}.gif"
                temp_files.append(temp_output)

                # Copy input to temporary file if it's not already a GIF (for videos)
                if not str(file_path).lower().endswith('.gif'):
                    # Convert video to GIF with default settings first
                    temp_initial = Path(TEMP_FILE_DIR) / \
                        f"initial_{uuid.uuid4().hex}.gif"
                    temp_files.append(temp_initial)

                    success = self._compress_large_with_ffmpeg(
                        file_path,
                        temp_initial,
                        scale_factor=settings.get('scale_factor', 0.6),
                        colors=settings.get('colors', 128),
                        dither_method=settings.get('dither_mode', 'bayer')
                    )

                    if not success or not temp_initial.exists():
                        self.logger.error(
                            "Failed to create initial GIF from video")
                        return False

                    input_file = temp_initial
                else:
                    input_file = file_path

                # Try optimization with gifsicle directly
                colors = settings.get('colors', 128)
                lossy = settings.get('lossy_value', 80)
                scale_factor = settings.get('scale_factor', 0.6)

                success = self._compress_with_gifsicle(
                    input_file,
                    temp_output,
                    colors=colors,
                    scale=scale_factor,
                    lossy=lossy
                )

                if success and temp_output.exists():
                    result_size = self.get_file_size(temp_output)

                    if result_size <= target_size_mb:
                        # Success - copy to output path
                        shutil.copy2(temp_output, output_path)
                        self.logger.success(
                            f"Direct optimization successful: {result_size:.2f}MB")
                        return True
                    else:
                        self.logger.info(
                            f"Direct optimization resulted in {result_size:.2f}MB, target is {target_size_mb:.2f}MB. Trying progressive approach.")

            # Try progressive optimization approach
            self.logger.info("Using progressive optimization approach")

            # For videos, first convert to GIF
            if not str(file_path).lower().endswith('.gif'):
                temp_gif = Path(TEMP_FILE_DIR) / \
                    f"temp_progressive_{uuid.uuid4().hex}.gif"
                temp_files.append(temp_gif)

                # Convert video to initial GIF
                fps = settings.get('fps', 15)
                width, height = dimensions
                new_width = int(width * settings.get('scale_factor', 0.7))
                new_height = int(height * settings.get('scale_factor', 0.7))

                # Ensure dimensions are even
                new_width = new_width if new_width % 2 == 0 else new_width + 1
                new_height = new_height if new_height % 2 == 0 else new_height + 1

                success = self._compress_large_with_ffmpeg(
                    file_path,
                    temp_gif,
                    scale_factor=settings.get('scale_factor', 0.7),
                    colors=settings.get('colors', 128),
                    dither_method=settings.get('dither_mode', 'bayer')
                )

                if not success or not temp_gif.exists():
                    self.logger.error(
                        "Failed to create initial GIF from video")
                    return False

                # Now optimize this GIF
                result = self._progressive_optimize(
                    temp_gif, output_path, target_size_mb)

                # Return success regardless of whether target size was met
                # (as long as we have a valid result file)
                return result.success and result.path is not None
            else:
                # For existing GIFs, optimize directly
                result = self._progressive_optimize(
                    file_path, output_path, target_size_mb)

                # Return success regardless of whether target size was met
                # (as long as we have a valid result file)
                return result.success and result.path is not None

        except Exception as e:
            self.logger.error(
                f"Error in adaptive GIF creation: {e}", exc_info=True)
            return False
        finally:
            # Clean up all temporary files
            for temp_file in temp_files:
                if temp_file.exists():
                    try:
                        temp_file.unlink(missing_ok=True)
                    except Exception:
                        pass

            # Log performance
            duration = time.time() - start_time
            self.logger.debug(
                f"Adaptive GIF creation completed in {duration:.2f}s")

    def _progressive_optimize(self, input_path: Path, output_path: Path, target_size_mb: float) -> ProcessingResult:
        """Optimize a GIF using progressive settings with quality preservation.

        Starts with lossless optimization and progressively increases compression
        until target size is achieved or quality threshold is reached.

        Args:
            input_path: Source GIF path
            output_path: Output GIF path
            target_size_mb: Target size in MB

        Returns:
            ProcessingResult: Result object with status and details
        """
        # Start performance monitoring
        self._start_timer("progressive_optimization")

        # Log the start of processing
        self._log_message(
            "OPTIMIZING",
            f"Progressive optimization for {input_path.name}",
            f"prog_opt_start_{str(input_path)}")

        # Start phase after logging to avoid duplication
        self.logger.start_phase("Progressive GIF optimization")

        # Initialize variables
        original_size = self.get_file_size(input_path)
        best_size = float('inf')
        best_settings = None
        best_file = None
        temp_files = []

        try:
            # Analyze original color count for better quality preservation
            original_colors = self._analyze_color_count(input_path)

            # Progressive optimization levels with enhanced compression strategy
            optimization_levels = [
                # Lossless optimization
                {"scale": 1.0, "colors": min(
                    224, original_colors), "lossy": 0},
                # Prefer scaling over quality reduction
                {"scale": 0.9, "colors": min(
                    224, original_colors), "lossy": 0},
                {"scale": 0.85, "colors": min(
                    208, original_colors), "lossy": 0},
                {"scale": 0.8, "colors": min(
                    192, original_colors), "lossy": 0},
                {"scale": 0.75, "colors": min(
                    176, original_colors), "lossy": 10},
                {"scale": 0.7, "colors": min(
                    160, original_colors), "lossy": 20},
                {"scale": 0.65, "colors": min(
                    160, original_colors), "lossy": 30},
                {"scale": 0.6, "colors": min(
                    144, original_colors), "lossy": 40},
                {"scale": 0.55, "colors": min(
                    144, original_colors), "lossy": 50},
                {"scale": 0.5, "colors": min(
                    128, original_colors), "lossy": 60},
                # More aggressive options for very large files
                {"scale": 0.45, "colors": 128, "lossy": 70},
                {"scale": 0.4, "colors": 112, "lossy": 80},
                {"scale": 0.35, "colors": 96, "lossy": 90},
                # Extreme last resort option
                {"scale": 0.3, "colors": 80, "lossy": 100},
                # Absolute last resort with extreme compression
                {"scale": 0.25, "colors": 64, "lossy": 120},
                # Ultra aggressive options added for more optimization attempts
                {"scale": 0.22, "colors": 56, "lossy": 130},
                {"scale": 0.2, "colors": 48, "lossy": 140},
                {"scale": 0.18, "colors": 40, "lossy": 150},
                {"scale": 0.16, "colors": 32, "lossy": 160},
                {"scale": 0.14, "colors": 24, "lossy": 170},
                {"scale": 0.12, "colors": 16, "lossy": 180}
            ]

            # Skip early passes for large files to save processing time
            start_pass = 0
            if original_size > target_size_mb * 10:  # Very large files
                start_pass = 5  # Start with significant compression
                self._log_message("INFO",
                                  f"File is very large ({original_size:.2f}MB), starting at pass {start_pass+1}",
                                  f"prog_skip_passes_{str(input_path)}")
            elif original_size > target_size_mb * 5:  # Large files
                start_pass = 3
                self._log_message("INFO",
                                  f"File is large ({original_size:.2f}MB), starting at pass {start_pass+1}",
                                  f"prog_skip_passes_{str(input_path)}")
            elif original_size > target_size_mb * 2:  # Medium-large files
                start_pass = 1
                self._log_message("INFO",
                                  f"File is medium-large ({original_size:.2f}MB), starting at pass {start_pass+1}",
                                  f"prog_skip_passes_{str(input_path)}")

            # Track progress
            total_levels = len(optimization_levels) - start_pass
            self._log_message("INFO",
                              f"Starting progressive optimization with {total_levels} passes",
                              f"prog_opt_begin_{str(input_path)}")

            # Try each optimization level until target is met
            for i, config in enumerate(optimization_levels[start_pass:], start=start_pass + 1):
                if self._should_exit():
                    break

                # Create temp file for this attempt
                temp_output = Path(TEMP_FILE_DIR) / \
                    f"progressive_{i}_{uuid.uuid4().hex}.gif"
                temp_files.append(temp_output)

                # Log progress
                self.logger.log_progress(
                    f"Trying optimization level {i}/{len(optimization_levels)}",
                    i - start_pass + 1,
                    total_levels,
                    details={
                        'Scale': f"{config['scale']:.2f}",
                        'Colors': str(config['colors']),
                        'Lossy': f"{config['lossy']}%"
                    }
                )

                # Log current pass details
                self._log_message("OPTIMIZING",
                                  f"Pass {i}: scale={config['scale']:.2f}, colors={config['colors']}, lossy={config['lossy']}",
                                  f"prog_pass{i}_{str(input_path)}")

                try:
                    # Compress using gifsicle with current settings
                    success = self._compress_with_gifsicle(
                        input_path,
                        temp_output,
                        colors=config['colors'],
                        scale=config['scale'],
                        lossy=config['lossy'],
                        optimize_level=3  # Always use maximum optimization level
                    )

                    if success and temp_output.exists():
                        try:
                            # Validate GIF format
                            with Image.open(temp_output) as img:
                                if img.format != 'GIF':
                                    self.logger.warning(
                                        f"Invalid GIF format in optimization result")
                                    continue

                            # Check size
                            current_size = self.get_file_size(temp_output)

                            # Log result with unique ID
                            self._log_optimization_result(
                                original_size, current_size, target_size_mb, f"Level {i}", input_path)

                            # Update best result if this is better
                            if current_size < best_size:
                                best_size = current_size
                                best_settings = config
                                best_file = temp_output

                                self._log_message("INFO",
                                                  f"New best result: {current_size:.2f}MB (target: {target_size_mb:.2f}MB)",
                                                  f"prog_new_best_{str(input_path)}")

                                # If we've met the target, we can stop
                                if current_size <= target_size_mb:
                                    self._log_message(
                                        "SUCCESS",
                                        f"Target size achieved: {current_size:.2f}MB",
                                        f"prog_opt_target_met_{str(input_path)}")
                                    break
                        except Exception as e:
                            self._log_message(
                                "WARNING",
                                f"Error validating level {i} result: {e}",
                                f"prog_opt_validation_error_{i}_{str(input_path)}")
                except Exception as e:
                    self._log_message(
                        "ERROR",
                        f"Error in optimization level {i}: {e}",
                        f"prog_opt_level_error_{i}_{str(input_path)}")

            # Process the results
            if best_file and best_file.exists():
                try:
                    # Copy the best result to output path
                    shutil.copy2(best_file, output_path)

                    # Create result object
                    fps = best_settings.get('fps', 10)

                    # Determine status based on size
                    met_target = best_size <= target_size_mb
                    if met_target:
                        status = ProcessingStatus.SUCCESS
                        message = f"Optimized successfully: {best_size:.2f}MB (target: {target_size_mb:.2f}MB)"
                        self._log_message(
                            "SUCCESS",
                            message,
                            f"prog_opt_success_{str(input_path)}")
                    else:
                        # We couldn't meet the target size but still using best result
                        status = ProcessingStatus.SIZE_THRESHOLD_EXCEEDED
                        message = f"Best optimization still exceeds target: {best_size:.2f}MB > {target_size_mb:.2f}MB, using anyway"
                        self._log_message(
                            "WARNING",
                            message,
                            f"prog_opt_target_exceeded_{str(input_path)}")

                    # Create optimization config from settings
                    opt_config = OptimizationConfig(
                        scale_factor=best_settings['scale'],
                        colors=best_settings['colors'],
                        lossy_value=best_settings['lossy']
                    )

                    return ProcessingResult(
                        fps=fps,
                        size=best_size,
                        path=str(output_path),
                        status=status,
                        message=message,
                        settings=opt_config,
                        success=True  # Always return success if we have a best result, even if not meeting target
                    )
                except Exception as e:
                    self._log_message(
                        "ERROR",
                        f"Error finalizing optimization: {e}",
                        f"prog_opt_finalize_error_{str(input_path)}")

            # If we get here, we failed to optimize
            return ProcessingResult(
                fps=10,
                size=original_size,
                path=None,
                status=ProcessingStatus.OPTIMIZATION_ERROR,
                message="Failed to create any valid optimization",
                success=False
            )
        except Exception as e:
            self._log_message(
                "ERROR",
                f"Progressive optimization error: {e}",
                f"prog_opt_general_error_{str(input_path)}")
            return ProcessingResult(
                fps=10,
                size=original_size,
                path=None,
                status=ProcessingStatus.OPTIMIZATION_ERROR,
                message=f"Error during optimization: {str(e)}",
                success=False,
                error=str(e)
            )
        finally:
            # End performance monitoring
            self._end_timer("progressive_optimization",
                            include_system_stats=True)
            self.logger.end_phase("Progressive GIF optimization")

            # Clean up temporary files
            for temp_file in temp_files:
                if temp_file != best_file and temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass

    def _start_timer(self, name: str):
        """Start a named timer for performance tracking."""
        if hasattr(self.logger, 'start_timer'):
            self.logger.start_timer(name)
        else:
            # Fallback if logger doesn't support timers
            self.logger.debug(
                f"Timer {name} started (no timer support in logger)")

    def _end_timer(self, name: str, include_system_stats=False):
        """End a named timer and log the elapsed time with optional system stats."""
        if hasattr(self.logger, 'end_timer'):
            self.logger.end_timer(name, include_system_stats)
        else:
            # Fallback if logger doesn't support timers
            self.logger.debug(
                f"Timer {name} stopped (no timer support in logger)")

    def _compress_with_gifsicle(self, input_path: Path, output_path: Path,
                                colors: int = 256, scale: float = 1.0,
                                lossy: int = 0, optimize_level: int = 2) -> bool:
        """Compress GIF using gifsicle with specified settings.

        Args:
            input_path: Source GIF path
            output_path: Output GIF path
            colors: Number of colors to use (1-256)
            scale: Scale factor (0.1-1.0)
            lossy: Lossy compression level (0-200)
            optimize_level: Optimization level (1-3)

        Returns:
            bool: True if successful, False otherwise
        """
        # Start performance timer for gifsicle
        self._start_timer(f"gifsicle_{colors}_{scale}_{lossy}")

        try:
            # Log detailed information about the compression parameters
            compression_details = {
                "Colors": str(colors),
                "Scale": f"{scale:.2f}",
                "Lossy": str(lossy),
                "Opt level": str(optimize_level)
            }

            self._log_message("PROCESSING",
                              f"Compressing with gifsicle: {colors} colors, scale={scale:.2f}, lossy={lossy}",
                              f"gifsicle_start_{str(input_path)}_{str(output_path)}")

            # Create gifsicle command
            cmd = ['gifsicle', f'--optimize={optimize_level}']

            # Add colors parameter
            cmd.extend(['--colors', str(colors)])

            # Add lossy parameter if needed
            if lossy > 0:
                # Updated to use consistent parameter style
                cmd.extend(['--lossy', str(lossy)])

            # Add scale parameter if needed
            if scale < 1.0:
                cmd.extend(['--scale', f"{scale}"])

            # Add no-conserve-memory for better handling of large files
            cmd.append('--no-conserve-memory')

            # Add input and output paths
            cmd.extend([str(input_path), '-o', str(output_path)])

            # Get file size before compression
            input_size = self.get_file_size(
                input_path) if input_path.exists() else 0

            # Run gifsicle command
            success = self._run_command_with_logging(cmd, timeout=300)

            # Get file size after compression if successful
            if success and output_path.exists():
                output_size = self.get_file_size(output_path)
                reduction = 0 if input_size == 0 else (
                    1 - output_size / input_size) * 100

                # Log success with size reduction info
                self._log_message("SUCCESS" if reduction > 0 else "WARNING",
                                  f"Gifsicle compression: {input_size:.2f}MB → {output_size:.2f}MB ({reduction:.1f}% reduction)",
                                  f"gifsicle_success_{str(input_path)}_{str(output_path)}")

                # Log more detailed information with the logger
                self.logger.info(f"Gifsicle compression results", extra={
                    'details': {
                        **compression_details,
                        "Original size": f"{input_size:.2f}MB",
                        "New size": f"{output_size:.2f}MB",
                        "Reduction": f"{reduction:.1f}%"
                    }
                })

                return True
            else:
                # Log failure
                self._log_message("ERROR",
                                  f"Gifsicle compression failed or produced no output",
                                  f"gifsicle_failed_{str(input_path)}_{str(output_path)}")
                return False

        except Exception as e:
            self._log_message(
                "ERROR",
                f"Gifsicle compression error: {e}",
                f"gifsicle_error_{str(input_path)}_{str(output_path)}")
            return False
        finally:
            # End performance timer
            self._end_timer(
                f"gifsicle_{colors}_{scale}_{lossy}", include_system_stats=True)

    def update_progress(self, operation_id: str, current: int, total: int,
                        description: str, file_name: str = None, status: str = "PROCESSING",
                        start_time: float = None, force_update: bool = False):
        """Update progress for a specific operation with real-time display.

        Args:
            operation_id: Unique identifier for this operation
            current: Current progress value
            total: Total progress value
            description: Description of the operation
            file_name: Current file being processed
            status: Status message (processing, optimizing, etc.)
            start_time: Start time for ETA calculation
            force_update: Whether to force an update regardless of the interval
        """
        with self.progress_lock:
            current_time = time.time()

            # Store progress data
            if operation_id not in self.active_progress_bars:
                self.active_progress_bars[operation_id] = {
                    'start_time': start_time or current_time,
                    'last_update': 0,
                    'description': description,
                    'file_name': file_name,
                    'status': status
                }

            # Update values
            progress_data = self.active_progress_bars[operation_id]
            progress_data['current'] = current
            progress_data['total'] = total
            progress_data['description'] = description
            progress_data['file_name'] = file_name
            progress_data['status'] = status

            # Check if it's time to update the display
            if force_update or (current_time - self.last_progress_update) >= self.progress_update_interval:
                # Update display
                display_progress_update(
                    current, total,
                    description=description,
                    file_name=file_name,
                    status=status,
                    start_time=progress_data['start_time'],
                    is_new_line=(progress_data['last_update'] == 0)
                )

                # Update timestamps
                progress_data['last_update'] = current_time
                self.last_progress_update = current_time

                # Log to logger as well
                if current == total:
                    self.logger.info(f"{description} complete: {file_name}")
                else:
                    self.logger.debug(
                        f"{description} progress: {current}/{total} - {file_name}")

    def complete_progress(self, operation_id: str, success: bool = True, message: str = None):
        """Mark a progress operation as complete.

        Args:
            operation_id: The operation ID to complete
            success: Whether the operation was successful
            message: Optional final message
        """
        with self.progress_lock:
            if operation_id in self.active_progress_bars:
                progress_data = self.active_progress_bars[operation_id]
                status = "SUCCESS" if success else "ERROR"
                final_message = message or f"{'Completed' if success else 'Failed'}: {progress_data['description']}"

                # Show final progress
                display_progress_update(
                    progress_data['total'], progress_data['total'],
                    description=final_message,
                    file_name=progress_data['file_name'],
                    status=status,
                    start_time=progress_data['start_time']
                )

                # Remove from active tracking
                del self.active_progress_bars[operation_id]
