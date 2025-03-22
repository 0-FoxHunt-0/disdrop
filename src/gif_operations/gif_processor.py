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
                                ModernLogStyle, ThreadSafeProgressLogger, run_ffmpeg_command, SUCCESS_LEVEL)
from src.temp_file_manager import TempFileManager
from src.utils.video_dimensions import _validate_dimensions as validate_dimensions

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
        """Initialize GIF processor with GPU settings.

        Args:
            logger: Optional logger to use. If None, a modern unified logger is created.
            gpu_enabled: Whether GPU acceleration is enabled.
            gpu_settings: Dictionary of GPU settings to use.
        """
        # Initialize with modern unified logger if not provided
        if logger is None:
            logger = UnifiedLogger('gif_processor')
        else:
            # Ensure logger is a UnifiedLogger to have access to methods like start_timer
            if not isinstance(logger, UnifiedLogger):
                logger = UnifiedLogger('gif_processor')

        # Initialize base class with logger
        super().__init__(logger=logger)

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

        # Store GPU status from parameters
        self.has_gpu = gpu_enabled
        self.gpu_settings = gpu_settings or {}

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
        """Create an optimized GIF from video with enhanced quality focus.

        Args:
            file_path: Path to input video file
            output_path: Path for output GIF file
            fps: Frames per second (will be adaptively adjusted)
            dimensions: Tuple of (width, height)

        Returns:
            bool: Success status
        """
        # Initialize temp file variables outside the try block to ensure they exist in the finally block
        temp_palette = None
        temp_output = None

        try:
            # Start a performance timer for GIF creation
            self._start_timer("gif_creation")

            # Log the start of processing with modern styling
            self._log_message(
                "STARTING", f"Starting GIF creation for {file_path.name}", str(file_path))

            # Use the configured target size instead of hardcoded default
            target_size_mb = self.compression_settings['min_size_mb']
            MAX_INTERMEDIATE_SIZE_MB = max(100, target_size_mb * 2)

            original_size = self.get_file_size(file_path)
            if original_size > 1000:
                self._log_message(
                    "ERROR", f"Source file too large: {original_size:.2f}MB", str(file_path))
                return False

            # Check if file is extremely large compared to target (>700% of target)
            size_ratio = original_size / target_size_mb
            if size_ratio > 7.0:
                self._log_message(
                    "WARNING",
                    f"Source file is {size_ratio:.1f}x larger than target. Using aggressive optimization approach.",
                    str(file_path)
                )
                return self._create_gif_aggressive(file_path, output_path, fps, dimensions)

            # Adjust FPS based on file size
            adjusted_fps = fps
            if original_size > 200:
                adjusted_fps = min(fps, 12)
            elif original_size > 100:
                adjusted_fps = min(fps, 15)
            elif original_size > 50:
                adjusted_fps = min(fps, 20)

            # Calculate dimensions with adaptive scaling
            max_dimension = max(dimensions[0], dimensions[1])
            scale_factor = 1.0

            # Scale based on file size and dimensions
            if original_size > 200:
                scale_factor = min(scale_factor, 0.4)
            elif original_size > 100:
                scale_factor = min(scale_factor, 0.6)
            elif original_size > 50:
                scale_factor = min(scale_factor, 0.8)

            if max_dimension > 1920:
                scale_factor = min(scale_factor, 0.4)
            elif max_dimension > 1280:
                scale_factor = min(scale_factor, 0.6)
            elif max_dimension > 720:
                scale_factor = min(scale_factor, 0.8)

            new_width = int(dimensions[0] * scale_factor // 2 * 2)
            new_height = int(dimensions[1] * scale_factor // 2 * 2)
            new_width = max(new_width, 320)
            new_height = max(new_height, 180)

            # Determine optimal color count
            color_count = 128
            if original_size > 100:
                color_count = 64

            # Generate temp file paths with UUIDs to avoid FFmpeg image sequence pattern warnings
            unique_id = str(uuid.uuid4())
            temp_palette = Path(TEMP_FILE_DIR) / f"palette_{unique_id}.png"
            temp_output = Path(TEMP_FILE_DIR) / f"temp_{unique_id}.gif"

            # Create an optimized palette
            palette_cmd = [
                'ffmpeg', '-i', str(file_path),
                '-vf', f'fps={adjusted_fps},scale={new_width}:{new_height}:flags=lanczos,palettegen=max_colors={color_count}:stats_mode=diff:reserve_transparent=0',
                '-y', str(temp_palette)
            ]

            if not run_ffmpeg_command(palette_cmd):
                self.logger.error("Palette generation failed")
                return False

            # Create the GIF with the generated palette - always favor quality
            # Always use high quality dithering for better gradients and visual quality
            dither_method = "floyd_steinberg"
            bayer_scale = 3

            gif_cmd = [
                'ffmpeg', '-i', str(file_path),
                '-i', str(temp_palette),
                '-lavfi',
                f'fps={adjusted_fps},scale={new_width}:{new_height}:flags=lanczos[x];[x][1:v]paletteuse=dither={dither_method}:diff_mode=rectangle',
                '-y', str(temp_output)
            ]

            if not run_ffmpeg_command(gif_cmd):
                self.logger.error("Initial GIF creation failed")
                return False

            # Check and optimize the initial GIF
            if temp_output.exists():
                initial_size = self.get_file_size(temp_output)

                # Check if initial GIF is still very large compared to target (>700% of target)
                size_ratio = initial_size / target_size_mb
                if size_ratio > 7.0:
                    self.logger.warning(
                        f"Initial GIF is {size_ratio:.1f}x larger than target. Applying more aggressive optimization.")

                    # Try progressive optimization to reduce size before gifsicle
                    attempt_settings = [
                        # Prefer scaling down over quality loss
                        {'scale': 0.8, 'fps': adjusted_fps, 'colors': color_count},
                        {'scale': 0.7, 'fps': adjusted_fps, 'colors': color_count},
                        {'scale': 0.6, 'fps': adjusted_fps, 'colors': color_count},
                        {'scale': 0.5, 'fps': adjusted_fps, 'colors': color_count},
                        # Only as last resort reduce colors and FPS
                        {'scale': 0.5, 'fps': max(
                            12, adjusted_fps-2), 'colors': min(192, color_count)},
                        {'scale': 0.45, 'fps': max(
                            10, adjusted_fps-4), 'colors': min(160, color_count)}
                    ]

                    for idx, settings in enumerate(attempt_settings):
                        if initial_size <= target_size_mb * 2:
                            break

                        try:
                            attempt_id = f"attempt_{idx}_{unique_id}"
                            attempt_palette = Path(
                                TEMP_FILE_DIR) / f"palette_{attempt_id}.png"
                            attempt_output = Path(
                                TEMP_FILE_DIR) / f"temp_{attempt_id}.gif"

                            # Create new palette with reduced settings
                            attempt_width = int(
                                dimensions[0] * settings['scale'] // 2 * 2)
                            attempt_height = int(
                                dimensions[1] * settings['scale'] // 2 * 2)

                            attempt_palette_cmd = [
                                'ffmpeg', '-i', str(file_path),
                                '-vf', f"fps={settings['fps']},scale={attempt_width}:{attempt_height}:flags=lanczos,palettegen=max_colors={settings['colors']}:stats_mode=diff:reserve_transparent=0",
                                '-y', str(attempt_palette)
                            ]

                            if run_ffmpeg_command(attempt_palette_cmd):
                                attempt_gif_cmd = [
                                    'ffmpeg', '-i', str(file_path),
                                    '-i', str(attempt_palette),
                                    '-lavfi', f"fps={settings['fps']},scale={attempt_width}:{attempt_height}:flags=lanczos[x];[x][1:v]paletteuse=dither=floyd_steinberg:diff_mode=rectangle",
                                    '-y', str(attempt_output)
                                ]

                                if run_ffmpeg_command(attempt_gif_cmd) and attempt_output.exists():
                                    attempt_size = self.get_file_size(
                                        attempt_output)

                                    if attempt_size < initial_size:
                                        # Use this better result
                                        self.logger.info(
                                            f"Using improved size: {attempt_size:.2f}MB (was {initial_size:.2f}MB)")
                                        temp_output.unlink(missing_ok=True)
                                        shutil.move(
                                            str(attempt_output), str(temp_output))
                                        initial_size = attempt_size
                                    else:
                                        attempt_output.unlink(missing_ok=True)

                            # Clean up
                            attempt_palette.unlink(missing_ok=True)
                        except Exception as e:
                            self.logger.error(
                                f"Error in progressive optimization attempt {idx}: {e}")

                # If size is already good, copy directly to output
                if initial_size <= target_size_mb:
                    shutil.copy2(temp_output, output_path)
                    self.logger.success(
                        f"Initial GIF meets size target: {initial_size:.2f}MB")
                    return True

                # If initial GIF is too large, try aggressive optimization
                if initial_size > MAX_INTERMEDIATE_SIZE_MB:
                    self.logger.warning(
                        f"Initial GIF too large, trying aggressive compression")
                    return self._create_gif_aggressive(file_path, output_path, adjusted_fps, dimensions)

                # Apply standard optimization
                gifsicle_cmd = [
                    'gifsicle',
                    '--optimize=3',
                    '--colors', str(color_count),
                    '--lossy=30',
                    '--no-conserve-memory',
                    str(temp_output),
                    '-o', str(output_path)
                ]

                if not run_ffmpeg_command(gifsicle_cmd):
                    shutil.copy2(temp_output, output_path)
                    self.logger.warning(
                        "Gifsicle optimization failed, using unoptimized GIF")

                # Check final size and apply more optimization if needed
                final_size = self.get_file_size(output_path)
                if final_size > target_size_mb:
                    self.logger.info(
                        f"Further optimization needed: {final_size:.2f}MB > {target_size_mb}MB")

                    # Try increasingly aggressive optimization until target is reached
                    optimization_attempts = [
                        {'colors': color_count//2, 'lossy': 50},
                        {'colors': min(64, color_count//2), 'lossy': 70},
                        {'colors': min(32, color_count//4), 'lossy': 90}
                    ]

                    for idx, settings in enumerate(optimization_attempts):
                        if final_size <= target_size_mb:
                            break

                        try:
                            opt_id = f"opt_{idx}_{unique_id}"
                            opt_output = Path(TEMP_FILE_DIR) / \
                                f"opt_{opt_id}.gif"

                            opt_cmd = [
                                'gifsicle',
                                '--optimize=3',
                                '--colors', str(settings['colors']),
                                f"--lossy={settings['lossy']}",
                                '--no-conserve-memory',
                                str(output_path),
                                '-o', str(opt_output)
                            ]

                            if run_ffmpeg_command(opt_cmd) and opt_output.exists():
                                opt_size = self.get_file_size(opt_output)
                                self.logger.info(
                                    f"Optimization attempt {idx+1}: {opt_size:.2f}MB")

                                if opt_size < final_size:
                                    # Replace with optimized version
                                    shutil.move(str(opt_output),
                                                str(output_path))
                                    final_size = opt_size
                                else:
                                    opt_output.unlink(missing_ok=True)
                        except Exception as e:
                            self.logger.error(
                                f"Error in optimization attempt {idx}: {e}")
                else:
                    self.logger.success(
                        f"Initial optimization successful: {final_size:.2f}MB")

                return True
            else:
                self.logger.error(
                    "GIF creation failed - no output file generated")
                return False

        except Exception as e:
            self.logger.error(f"GIF creation error: {str(e)}")
            traceback.print_exc()
            return False
        finally:
            # End the performance timer
            try:
                self._end_timer("gif_creation", include_system_stats=True)
            except Exception:
                pass  # Ignore timer errors

            # Cleanup temp files
            try:
                if temp_palette and temp_palette.exists():
                    temp_palette.unlink(missing_ok=True)
                if temp_output and temp_output.exists():
                    temp_output.unlink(missing_ok=True)
            except Exception as cleanup_e:
                self.logger.error(f"Cleanup error: {cleanup_e}")

        # Try gifsicle optimization to reduce size
        original_size = self.get_file_size(temp_output)
        if original_size > target_size_mb:
            self.logger.info(
                f"Initial GIF exceeds target size. Trying gifsicle optimization.")
            optimized = self.optimize_gif(
                temp_output, output_path, target_size_mb)
            if optimized[0] <= target_size_mb:
                self.logger.success(
                    f"Successfully optimized GIF: {optimized[0]:.2f}MB")
                return True
            else:
                # If we couldn't reach the target size, still use the best result we have
                self.logger.warning(
                    f"Optimized GIF still exceeds target size ({optimized[0]:.2f}MB > {target_size_mb:.2f}MB)")
                # Proceed with the best result we have
                return True
        else:
            # Copy directly to output
            shutil.copy2(temp_output, output_path)
            self.logger.success(
                f"Created GIF successfully: {original_size:.2f}MB")
            return True

    def _create_gif_aggressive(self, file_path: Path, output_path: Path, fps: int, dimensions: Tuple[int, int]) -> bool:
        """Create GIF with aggressive compression to ensure size constraints are met."""
        original_size = self.get_file_size(file_path)
        target_size = self.compression_settings['min_size_mb']

        try:
            # Create a temp directory for this process with UUID
            unique_id = str(uuid.uuid4())
            temp_dir = Path(TEMP_FILE_DIR) / f"aggressive_{unique_id}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            # Define a series of increasingly aggressive configs
            configs = [
                {'scale': 0.7, 'fps': max(
                    10, fps-10), 'colors': 128, 'lossy': 40},
                {'scale': 0.6, 'fps': max(8, fps-15),
                 'colors': 96, 'lossy': 60},
                {'scale': 0.5, 'fps': max(6, fps-20),
                 'colors': 64, 'lossy': 80},
                {'scale': 0.4, 'fps': max(5, fps-22),
                 'colors': 32, 'lossy': 90}
            ]

            # For very large files (>700% of target), add even more aggressive configs
            size_ratio = original_size / target_size
            if size_ratio > 7.0:
                configs.extend([
                    {'scale': 0.3, 'fps': max(
                        4, fps-25), 'colors': 24, 'lossy': 100},
                    {'scale': 0.25, 'fps': max(
                        3, fps-25), 'colors': 16, 'lossy': 100}
                ])

            # Check if this is a video file that needs keyframe handling
            is_video = self._is_video_file(str(file_path))
            keyframe_flags = ['-ignore_loop', '0',
                              '-vsync', '0'] if is_video else []

            # Calculate aspect ratio for scaling
            width, height = dimensions
            aspect_ratio = width / height
            self.logger.info(
                f"Original dimensions: {width}x{height}, aspect ratio: {aspect_ratio:.3f}")

            # Try each configuration until we get one that works
            best_result = None
            for i, config in enumerate(configs):
                try:
                    # Calculate dimensions while preserving aspect ratio
                    scale_factor = config['scale']

                    if width > height:  # Landscape orientation
                        new_width = int(width * scale_factor // 2 * 2)
                        new_height = int(
                            new_width / aspect_ratio + 0.5) // 2 * 2
                    else:  # Portrait orientation
                        new_height = int(height * scale_factor // 2 * 2)
                        new_width = int(
                            new_height * aspect_ratio + 0.5) // 2 * 2

                    # Ensure minimum dimensions
                    new_width = max(new_width, 160)
                    new_height = max(new_height, 120)

                    # Verify aspect ratio is preserved
                    new_aspect = new_width / new_height
                    if abs(new_aspect - aspect_ratio) / aspect_ratio > 0.05:
                        # Recalculate if aspect ratio is off by more than 5%
                        if width > height:
                            new_width = max(
                                int(width * scale_factor) // 2 * 2, 160)
                            new_height = max(
                                int(new_width / aspect_ratio + 0.5) // 2 * 2, 120)
                        else:
                            new_height = max(
                                int(height * scale_factor) // 2 * 2, 120)
                            new_width = max(
                                int(new_height * aspect_ratio + 0.5) // 2 * 2, 160)

                    self.logger.info(
                        f"Config {i+1}: Using dimensions {new_width}x{new_height}")

                    # Use UUID-based filenames to avoid FFmpeg warnings
                    config_id = f"{unique_id}_{i}"
                    temp_palette = temp_dir / f"palette_{config_id}.png"
                    temp_output = temp_dir / f"output_{config_id}.gif"

                    # Create palette
                    palette_cmd = ['ffmpeg']
                    if is_video:
                        palette_cmd.extend(keyframe_flags)

                    palette_cmd.extend([
                        '-i', str(file_path),
                        '-vf', f"fps={config['fps']},scale={new_width}:{new_height}:flags=lanczos,palettegen=max_colors={config['colors']}:stats_mode=diff",
                        '-y', str(temp_palette)
                    ])

                    if not run_ffmpeg_command(palette_cmd):
                        continue

                    # Create GIF
                    gif_cmd = ['ffmpeg']
                    if is_video:
                        gif_cmd.extend(keyframe_flags)

                    gif_cmd.extend([
                        '-i', str(file_path),
                        '-i', str(temp_palette),
                        '-lavfi', f"fps={config['fps']},scale={new_width}:{new_height}:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
                        '-y', str(temp_output)
                    ])

                    if not run_ffmpeg_command(gif_cmd):
                        continue

                    # Optimize with gifsicle
                    if temp_output.exists():
                        # Verify the file is valid
                        try:
                            with Image.open(temp_output) as img:
                                if img.format != 'GIF':
                                    self.logger.warning(
                                        f"Output is not a valid GIF: {img.format}")
                                    continue
                        except Exception as e:
                            self.logger.warning(f"Invalid GIF created: {e}")
                            continue

                        optimized_output = temp_dir / \
                            f"optimized_{config['colors']}.gif"

                        gifsicle_cmd = [
                            'gifsicle',
                            '--optimize=3',
                            '--colors', str(config['colors']),
                            f"--lossy={config['lossy']}",
                            '--no-conserve-memory',
                            str(temp_output),
                            '-o', str(optimized_output)
                        ]

                        if run_ffmpeg_command(gifsicle_cmd) and optimized_output.exists():
                            # Check size
                            size = self.get_file_size(optimized_output)
                            self.logger.info(
                                f"Config {i+1}: {size:.2f}MB (target: {target_size:.2f}MB)")

                            # Store result if it's the first valid one or better than previous
                            if (size <= target_size and
                                (best_result is None or
                                 # prefer larger file within target size
                                 size > best_result['size'])):

                                best_result = {
                                    'path': optimized_output,
                                    'size': size,
                                    'config': config
                                }

                                # If we're close enough to target size, stop
                                if target_size - size < 0.5:
                                    break
                            elif best_result is None or size < best_result['size']:
                                # Keep track of the best result even if it's not under target
                                best_result = {
                                    'path': optimized_output,
                                    'size': size,
                                    'config': config
                                }

                except Exception as e:
                    self.logger.error(f"Error with config {config}: {e}")
                    continue

            # Use best result or most aggressive as fallback
            if best_result is not None:
                shutil.copy2(best_result['path'], output_path)
                success_message = "Successfully optimized"
                if best_result['size'] <= target_size:
                    success_message = f"Aggressive optimization succeeded"
                else:
                    success_message = f"Couldn't reach target size but optimized"

                self.logger.success(
                    f"{success_message}: {best_result['size']:.2f}MB (Target: {target_size}MB)"
                )
                return True
            else:
                # Last attempt with maximum compression
                try:
                    final_id = str(uuid.uuid4())
                    last_chance = temp_dir / f"last_attempt_{final_id}.gif"
                    final_palette = temp_dir / f"final_palette_{final_id}.png"

                    # Extremely reduced settings while preserving aspect ratio
                    scale_factor = 0.2
                    if width > height:  # Landscape
                        min_width = max(
                            160, int(width * scale_factor) // 2 * 2)
                        min_height = max(
                            120, int(min_width / aspect_ratio + 0.5) // 2 * 2)
                    else:  # Portrait
                        min_height = max(
                            120, int(height * scale_factor) // 2 * 2)
                        min_width = max(
                            160, int(min_height * aspect_ratio + 0.5) // 2 * 2)

                    min_fps = max(3, fps // 5)

                    # Create palette
                    palette_cmd = ['ffmpeg']
                    if is_video:
                        palette_cmd.extend(keyframe_flags)

                    palette_cmd.extend([
                        '-i', str(file_path),
                        '-vf', f"fps={min_fps},scale={min_width}:{min_height}:flags=fast_bilinear,palettegen=max_colors=16",
                        '-y', str(final_palette)
                    ])

                    if run_ffmpeg_command(palette_cmd):
                        # Create GIF
                        gif_cmd = ['ffmpeg']
                        if is_video:
                            gif_cmd.extend(keyframe_flags)

                        gif_cmd.extend([
                            '-i', str(file_path),
                            '-i', str(final_palette),
                            '-lavfi', f"fps={min_fps},scale={min_width}:{min_height}:flags=fast_bilinear[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=1",
                            '-y', str(last_chance)
                        ])

                        if run_ffmpeg_command(gif_cmd) and last_chance.exists():
                            # Final gifsicle pass with maximum compression
                            final_output = temp_dir / \
                                f"final_output_{final_id}.gif"
                            gifsicle_cmd = [
                                'gifsicle',
                                '--optimize=3',
                                '--colors', '16',
                                '--lossy=100',
                                '--scale', '0.5',
                                '--no-conserve-memory',
                                str(last_chance),
                                '-o', str(final_output)
                            ]

                            if run_ffmpeg_command(gifsicle_cmd) and final_output.exists():
                                final_size = self.get_file_size(final_output)

                                if final_size <= target_size:
                                    shutil.copy2(final_output, output_path)
                                    self.logger.success(
                                        f"Final aggressive attempt succeeded: {final_size:.2f}MB"
                                    )
                                    return True
                                else:
                                    # Use this result anyway as our best effort
                                    shutil.copy2(final_output, output_path)
                                    self.logger.warning(
                                        f"Used best effort result: {final_size:.2f}MB (target: {target_size:.2f}MB)"
                                    )
                                    return True

                except Exception as e:
                    self.logger.error(f"Final compression attempt failed: {e}")

            self.logger.error(
                f"Failed to compress GIF within target size: {target_size}MB")
            return False

        except Exception as e:
            self.logger.error(f"Aggressive GIF creation error: {str(e)}")
            return False
        finally:
            # Clean up temp directory and files
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                self.logger.error(f"Failed to clean up temp directory: {e}")

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

                        # If we get here, optimization failed to meet the target size
                        self.logger.warning(
                            f"All optimization attempts failed to meet target size of {target_size:.2f}MB")
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
        """Optimize GIF to target size with quality preservation strategy.

        Args:
            input_path: Source GIF file
            output_path: Output path for optimized GIF
            target_size_mb: Target size in MB (use class setting if None)
            original_colors: Number of colors in original GIF (will be analyzed if None)

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

            # Use smart optimization approach
            self._log_message("STARTING",
                              f"Starting optimization with target: {target_size_mb:.2f}MB",
                              f"optimize_start_{str(input_path)}")

            # Get result from smart optimization
            result = self._smart_gif_optimization(
                input_path, output_path, target_size_mb, original_colors)

            # Check result
            final_size = result.size

            if result.success and final_size <= target_size_mb:
                self._log_message("SUCCESS",
                                  f"Optimization successful: {final_size:.2f}MB",
                                  f"optimize_success_{str(input_path)}")
                return final_size, True
            else:
                # Don't copy to output if size exceeds target
                if result.path and Path(result.path).exists() and final_size <= target_size_mb:
                    # Only copy if we're within the target size
                    shutil.copy2(Path(result.path), output_path)
                    self._log_message("SUCCESS",
                                      f"Optimization met target: {final_size:.2f}MB <= {target_size_mb:.2f}MB",
                                      f"optimize_success_copy_{str(input_path)}")
                    return final_size, True
                else:
                    self._log_message("ERROR",
                                      f"Optimization failed to meet target size: {final_size:.2f}MB > {target_size_mb:.2f}MB",
                                      f"optimize_size_fail_{str(input_path)}")
                    return original_size, False

        except Exception as e:
            self._log_message("ERROR",
                              f"GIF optimization error: {e}",
                              f"optimize_general_error_{str(input_path)}")
            return original_size, False
        finally:
            # End the performance timer and phase
            self._end_timer("gif_optimization", include_system_stats=True)
            self.logger.end_phase("GIF optimization")

    def _smart_gif_optimization(self, input_path: Path, output_path: Path, target_size_mb: float, original_colors: int = None) -> ProcessingResult:
        """Smart GIF optimization with content-adaptive approach.

        Uses a combination of techniques to efficiently optimize GIFs:
        1. Frame optimization - removes duplicate pixels between frames
        2. Smart color quantization - uses perceptual quality metrics
        3. Adaptive frame rate - adjusts based on content motion
        4. Selective dithering - applies only where visually beneficial

        Args:
            input_path: Source GIF path
            output_path: Destination path for optimized GIF
            target_size_mb: Target size in MB
            original_colors: Original color count (will be analyzed if None)

        Returns:
            ProcessingResult: Result of the optimization
        """
        # Start performance tracking
        self._start_timer("smart_gif_optimization")
        self.logger.start_phase("Smart GIF Optimization")

        temp_files = []

        try:
            self._log_message("STARTING",
                              f"Starting smart optimization for {input_path.name}",
                              f"smart_opt_start_{str(input_path)}")

            # Analyze content if needed
            if original_colors is None:
                self._log_message("PROCESSING",
                                  "Analyzing color palette",
                                  f"color_analysis_{str(input_path)}")
                original_colors = self._analyze_color_count(input_path)
                self._log_message("INFO",
                                  f"Detected {original_colors} colors in original GIF",
                                  f"color_count_{str(input_path)}")

            # Get original dimensions
            dimensions = self._get_dimensions_with_retry(input_path)
            if not dimensions:
                self._log_message("ERROR",
                                  "Failed to get GIF dimensions",
                                  f"dimension_error_{str(input_path)}")
                return ProcessingResult(
                    fps=0,
                    size=0,
                    path=None,
                    status=ProcessingStatus.DIMENSION_ERROR,
                    message="Failed to get dimensions"
                )
            width, height = dimensions
            self._log_message("INFO",
                              f"GIF dimensions: {width}x{height}",
                              f"dimensions_{str(input_path)}")

            # Analyze motion in the GIF
            self._log_message("PROCESSING",
                              "Analyzing motion patterns",
                              f"motion_start_{str(input_path)}")
            motion_score = self._analyze_gif_motion(input_path)

            # Begin optimization process - first with lossless pass
            self._log_message("OPTIMIZING",
                              "Starting initial lossless optimization",
                              f"lossless_start_{str(input_path)}")

            # Create intermediate optimized GIF using gifsicle's built-in optimization
            temp_optimized = Path(TEMP_FILE_DIR) / \
                f"opt_base_{uuid.uuid4().hex}.gif"
            temp_files.append(temp_optimized)

            # First pass - lossless optimization with gifsicle
            if not self._compress_with_gifsicle(
                input_path,
                temp_optimized,
                # Already using 256 colors maximum - use a lower color count to avoid size increase
                colors=min(original_colors, 224),
                optimize_level=3,
                lossy=0
            ):
                self._log_message("ERROR",
                                  "Initial lossless optimization failed",
                                  f"lossless_fail_{str(input_path)}")
                return ProcessingResult(
                    fps=0,
                    size=0,
                    path=None,
                    status=ProcessingStatus.OPTIMIZATION_ERROR,
                    message="Initial optimization failed"
                )

            # Check if first pass is enough
            optimized_size = self.get_file_size(temp_optimized)
            original_size = self.get_file_size(input_path)
            savings_percent = (1 - optimized_size / original_size) * 100

            self._log_message("INFO",
                              f"Lossless optimization: {original_size:.2f}MB → {optimized_size:.2f}MB ({savings_percent:.1f}% reduction)",
                              f"lossless_result_{str(input_path)}")

            if optimized_size <= target_size_mb:
                self._log_message("SUCCESS",
                                  f"Target size achieved with lossless optimization: {optimized_size:.2f}MB (target: {target_size_mb:.2f}MB)",
                                  f"lossless_success_{str(input_path)}")
                shutil.copy2(temp_optimized, output_path)
                return ProcessingResult(
                    fps=0,
                    size=optimized_size,
                    path=str(output_path),
                    status=ProcessingStatus.SUCCESS,
                    message=f"Lossless optimization achieved target size"
                )

            # Determine adaptive settings based on content
            # Low motion = can reduce FPS, high detail = need more colors
            # Increased thresholds to classify more content as high quality
            is_high_motion = motion_score > 0.2  # Lower threshold for better quality
            is_high_detail = original_colors > 32  # Lower threshold to preserve colors

            # Log the content classification for better diagnostics
            content_type = []
            if is_high_motion:
                content_type.append("high motion")
            else:
                content_type.append("low motion")

            if is_high_detail:
                content_type.append("high detail")
            else:
                content_type.append("low detail")

            content_description = " + ".join(content_type)

            self._log_message("INFO",
                              f"Content classified as: {content_description}",
                              f"content_class_{str(input_path)}")

            # Log progress before starting the appropriate optimization strategy
            self.logger.log_progress(
                "GIF optimization progress",
                1,  # Completed lossless phase
                3,  # Total phases (lossless, content-specific, validation)
                details={
                    "Content type": content_description,
                    "Target size": f"{target_size_mb:.2f}MB",
                    "Current size": f"{optimized_size:.2f}MB"
                }
            )

            # Choose appropriate optimization path based on content type
            self._log_message("PROCESSING",
                              f"Starting content-adaptive optimization for {content_description} content",
                              f"adaptive_start_{str(input_path)}")

            if is_high_motion and is_high_detail:
                # High motion, high detail - prioritize frame rate, reduce colors moderately
                self._log_message("PROCESSING",
                                  "Using high motion + high detail optimization strategy",
                                  f"hmhd_start_{str(input_path)}")
                result = self._optimize_high_motion_high_detail(
                    temp_optimized, output_path, target_size_mb)
            elif is_high_motion and not is_high_detail:
                # High motion, low detail - maintain frame rate, reduce colors aggressively
                self._log_message("PROCESSING",
                                  "Using high motion + low detail optimization strategy",
                                  f"hmld_start_{str(input_path)}")
                result = self._optimize_high_motion_low_detail(
                    temp_optimized, output_path, target_size_mb)
            elif not is_high_motion and is_high_detail:
                # Low motion, high detail - reduce frame rate, maintain colors
                self._log_message("PROCESSING",
                                  "Using low motion + high detail optimization strategy",
                                  f"lmhd_start_{str(input_path)}")
                result = self._optimize_low_motion_high_detail(
                    temp_optimized, output_path, target_size_mb)
            else:
                # Low motion, low detail - aggressive optimization on both
                self._log_message("PROCESSING",
                                  "Using low motion + low detail optimization strategy",
                                  f"lmld_start_{str(input_path)}")
                result = self._optimize_low_motion_low_detail(
                    temp_optimized, output_path, target_size_mb)

            # Log progress after content-specific optimization
            if result.success:
                status = GIFProcessingStatus.SUCCESS
                log_level = "SUCCESS"
            else:
                status = GIFProcessingStatus.WARNING if result.path else GIFProcessingStatus.ERROR
                log_level = "WARNING" if result.path else "ERROR"

            # Log the optimization result
            if result.path:
                result_size = self.get_file_size(Path(result.path))
                size_desc = f"{result_size:.2f}MB"
                size_ratio = result_size / original_size
                percent_reduction = (1 - size_ratio) * 100

                self._log_message(log_level,
                                  f"Optimization result: {original_size:.2f}MB → {size_desc} ({percent_reduction:.1f}% reduction)",
                                  f"opt_result_{str(input_path)}")

                # Update progress with final result
                self.logger.log_progress(
                    "GIF optimization complete",
                    3,  # Final phase
                    3,  # Total phases
                    details={
                        "Original size": f"{original_size:.2f}MB",
                        "Final size": size_desc,
                        "Reduction": f"{percent_reduction:.1f}%",
                        "Target met": "Yes" if result_size <= target_size_mb else "No"
                    }
                )
            else:
                self._log_message("ERROR",
                                  "Optimization failed to produce a valid file",
                                  f"opt_result_fail_{str(input_path)}")

            # Final validation
            if result.success and Path(result.path).exists():
                # Validate the optimized GIF
                self._log_message("PROCESSING",
                                  "Validating optimized GIF",
                                  f"validation_start_{str(input_path)}")
                try:
                    with Image.open(result.path) as img:
                        if img.format != 'GIF':
                            self._log_message("ERROR",
                                              "Output is not a valid GIF",
                                              f"validation_format_{str(input_path)}")
                            return ProcessingResult(
                                fps=0,
                                size=0,
                                path=None,
                                status=ProcessingStatus.CONVERSION_ERROR,
                                message="Output is not a valid GIF"
                            )

                    self._log_message("SUCCESS",
                                      "GIF validation successful",
                                      f"validation_success_{str(input_path)}")
                except Exception as e:
                    self._log_message("ERROR",
                                      f"Invalid output file: {e}",
                                      f"validation_error_{str(input_path)}")
                    return ProcessingResult(
                        fps=0,
                        size=0,
                        path=None,
                        status=ProcessingStatus.FILE_ERROR,
                        message=f"Invalid output file: {e}"
                    )

                # Copy to output if not already there
                if result.path != str(output_path):
                    self._log_message("INFO",
                                      "Copying optimized GIF to final location",
                                      f"copy_start_{str(input_path)}")
                    shutil.copy2(result.path, output_path)
                    self._log_message("SUCCESS",
                                      f"Copied optimized GIF to {output_path.name}",
                                      f"copy_success_{str(input_path)}")

            return result

        except Exception as e:
            self._log_message("ERROR",
                              f"Smart optimization error: {e}",
                              f"smart_opt_error_{str(input_path)}")
            return ProcessingResult(
                fps=0,
                size=0,
                path=None,
                status=ProcessingStatus.OPTIMIZATION_ERROR,
                message=f"Optimization error: {e}"
            )
        finally:
            # Clean up temp files
            self._log_message("INFO",
                              f"Cleaning up {len(temp_files)} temporary files",
                              f"cleanup_{str(input_path)}")
            self._safe_cleanup_temp_files([Path(f) for f in temp_files])

            # End performance monitoring
            self._end_timer("smart_gif_optimization",
                            include_system_stats=True)
            self.logger.end_phase("Smart GIF Optimization")

    def _analyze_gif_motion(self, gif_path: Path) -> float:
        """Analyze motion intensity in a GIF.

        Returns a score from 0.0 (static) to 1.0 (high motion)
        """
        # Start performance timer for motion analysis
        self._start_timer("gif_motion_analysis")
        self.logger.start_phase("GIF motion analysis")

        try:
            self._log_message("PROCESSING",
                              f"Analyzing motion in {gif_path.name}",
                              f"motion_analysis_start_{str(gif_path)}")

            frames = []
            frame_count = 0
            with Image.open(gif_path) as img:
                # Extract frames
                try:
                    while True:
                        # Convert to grayscale
                        frames.append(img.copy().convert('L'))
                        frame_count += 1
                        img.seek(img.tell() + 1)
                except EOFError:
                    pass

            # If fewer than 2 frames, return 0 motion
            if len(frames) < 2:
                self._log_message("INFO",
                                  f"GIF has only {len(frames)} frame(s) - considering as static",
                                  f"motion_static_{str(gif_path)}")
                return 0.0

            # Log frame count for diagnostics
            self._log_message("INFO",
                              f"Analyzing {len(frames)} frames for motion detection",
                              f"motion_frames_{str(gif_path)}")

            # Calculate motion score based on frame differences
            total_diff = 0
            total_pixels = frames[0].width * frames[0].height

            # Track progress for large GIFs
            if len(frames) > 10:
                self.logger.log_progress(
                    "Calculating frame differences",
                    0,
                    len(frames) - 1
                )

            for i in range(len(frames) - 1):
                # Calculate frame difference
                diff = ImageChops.difference(frames[i], frames[i+1])
                diff_sum = sum(diff.getdata())
                avg_diff = diff_sum / (total_pixels * 255)  # Normalize
                total_diff += avg_diff

                # Update progress for large GIFs
                if len(frames) > 10 and i % 5 == 0:
                    self.logger.log_progress(
                        "Calculating frame differences",
                        i + 1,
                        len(frames) - 1
                    )

            motion_score = min(1.0, total_diff / (len(frames) - 1))

            # Log the result with appropriate status icon based on motion level
            if motion_score < 0.3:
                log_level = "INFO"
                motion_desc = "low"
            elif motion_score < 0.7:
                log_level = "INFO"
                motion_desc = "medium"
            else:
                log_level = "INFO"
                motion_desc = "high"

            self._log_message(log_level,
                              f"GIF motion score: {motion_score:.3f} ({motion_desc} motion)",
                              f"motion_result_{str(gif_path)}")

            return motion_score

        except Exception as e:
            self._log_message("ERROR",
                              f"Error analyzing GIF motion: {e}",
                              f"motion_error_{str(gif_path)}")
            return 0.5  # Default to medium motion on error
        finally:
            # End performance monitoring
            self._end_timer("gif_motion_analysis", include_system_stats=True)
            self.logger.end_phase("GIF motion analysis")

    def _optimize_high_motion_high_detail(self, input_path: Path, output_path: Path, target_size_mb: float) -> ProcessingResult:
        """Optimize GIF with high motion and high detail.

        Strategy: Preserve frame rate, use perceptual color reduction, apply minimal lossy compression.
        Multi-pass encoding with progressive quality reduction until target size is met.
        """
        # Start performance timer for this strategy
        self._start_timer("optimize_high_motion_high_detail")
        self.logger.start_phase("High motion + high detail optimization")

        # Create a unique temp file for each attempt
        temp_files = []
        best_size = float('inf')
        best_file = None
        best_settings = None

        try:
            # Get dimensions
            dimensions = self._get_dimensions_with_retry(input_path)
            if not dimensions:
                self._log_message("ERROR",
                                  "Failed to get dimensions for optimization",
                                  f"hmhd_dim_error_{str(input_path)}")
                return ProcessingResult(0, 0, None, ProcessingStatus.DIMENSION_ERROR)

            width, height = dimensions

            # Get original size
            original_size = self.get_file_size(input_path)
            size_ratio = original_size / target_size_mb

            self._log_message("INFO",
                              f"Size ratio: {size_ratio:.2f}x target size ({original_size:.2f}MB / {target_size_mb:.2f}MB)",
                              f"hmhd_size_ratio_{str(input_path)}")

            # Define multi-pass settings with progressively stronger compression
            # Start with less aggressive settings and gradually increase compression
            optimization_passes = [
                # Pass 1: Minimal compression, preserve quality
                {"scale": 0.95, "colors": 256, "lossy": 0},
                # Pass 2: Light compression
                {"scale": 0.9, "colors": 240, "lossy": 0},
                # Pass 3: Moderate compression
                {"scale": 0.85, "colors": 224, "lossy": 10},
                # Pass 4: Medium compression
                {"scale": 0.8, "colors": 208, "lossy": 20},
                # Pass 5: Higher compression
                {"scale": 0.75, "colors": 192, "lossy": 30},
                # Pass 6: Strong compression
                {"scale": 0.7, "colors": 176, "lossy": 40},
                # Pass 7: Very strong compression
                {"scale": 0.65, "colors": 160, "lossy": 50},
                # Pass 8: Extreme compression
                {"scale": 0.6, "colors": 144, "lossy": 60},
                # Pass 9: Maximum compression
                {"scale": 0.55, "colors": 128, "lossy": 80}
            ]

            # Skip early passes if file is much larger than target
            start_pass = 0
            if size_ratio > 5:
                start_pass = 4  # Start with higher compression for large files
            elif size_ratio > 3:
                start_pass = 2  # Start with moderate compression

            # Log starting strategy
            self._log_message("INFO",
                              f"Starting at optimization pass {start_pass+1}/{len(optimization_passes)} for {size_ratio:.1f}x target size",
                              f"hmhd_strategy_{str(input_path)}")

            # Process each pass until we reach target size
            for i, settings in enumerate(optimization_passes[start_pass:], start=start_pass + 1):
                # Create temp file for this pass
                temp_output = Path(TEMP_FILE_DIR) / \
                    f"hmhd_pass{i}_{uuid.uuid4().hex}.gif"
                temp_files.append(temp_output)

                # Log current pass
                self._log_message("OPTIMIZING",
                                  f"Pass {i}/{len(optimization_passes)}: scale={settings['scale']:.2f}, colors={settings['colors']}, lossy={settings['lossy']}",
                                  f"hmhd_pass{i}_{str(input_path)}")

                # Run optimization
                success = self._compress_with_gifsicle(
                    input_path,
                    temp_output,
                    colors=settings['colors'],
                    scale=settings['scale'],
                    lossy=settings['lossy'],
                    optimize_level=3  # Use maximum optimization level
                )

                if success and temp_output.exists():
                    # Check result size
                    result_size = self.get_file_size(temp_output)

                    # Log result
                    self._log_message("INFO",
                                      f"Pass {i} result: {result_size:.2f}MB (target: {target_size_mb:.2f}MB)",
                                      f"hmhd_pass{i}_result_{str(input_path)}")

                    # Validate the file is a valid GIF
                    try:
                        with Image.open(temp_output) as img:
                            if img.format != 'GIF':
                                continue  # Skip invalid files
                    except Exception:
                        continue  # Skip on error

                    # Update best result if this is better
                    if result_size < best_size:
                        best_size = result_size
                        best_file = temp_output
                        best_settings = settings

                    # Stop if we've reached target size
                    if result_size <= target_size_mb:
                        self._log_message("SUCCESS",
                                          f"Target size achieved in pass {i}: {result_size:.2f}MB",
                                          f"hmhd_target_met_{str(input_path)}")
                        break

            # Process results
            if best_file and best_file.exists():
                # Copy best result to output
                shutil.copy2(best_file, output_path)

                # Determine success based on size
                success = best_size <= target_size_mb
                status = ProcessingStatus.SUCCESS if success else ProcessingStatus.SIZE_THRESHOLD_EXCEEDED

                message = (f"Optimized to {best_size:.2f}MB (target: {target_size_mb:.2f}MB)"
                           f" with scale={best_settings['scale']:.2f}, colors={best_settings['colors']}, lossy={best_settings['lossy']}")

                # Create optimization config from settings
                settings = OptimizationConfig(
                    scale_factor=best_settings['scale'],
                    colors=best_settings['colors'],
                    lossy_value=best_settings['lossy']
                )

                return ProcessingResult(
                    fps=0,  # FPS is preserved in GIF
                    size=best_size,
                    path=str(output_path),
                    status=status,
                    message=message,
                    settings=settings,
                    success=success
                )
            else:
                # No valid result found
                self._log_message("ERROR",
                                  "Failed to produce valid output in any pass",
                                  f"hmhd_all_failed_{str(input_path)}")

                return ProcessingResult(
                    fps=0,
                    size=0,
                    path=None,
                    status=ProcessingStatus.OPTIMIZATION_ERROR,
                    message="All optimization passes failed"
                )
        except Exception as e:
            self._log_message("ERROR",
                              f"High motion optimization error: {e}",
                              f"hmhd_error_{str(input_path)}")

            return ProcessingResult(
                fps=0,
                size=0,
                path=None,
                status=ProcessingStatus.OPTIMIZATION_ERROR,
                message=f"Optimization error: {str(e)}"
            )
        finally:
            # Clean up temp files
            for temp_file in temp_files:
                if temp_file != best_file and temp_file.exists():
                    try:
                        temp_file.unlink()
                    except:
                        pass

            # End performance monitoring
            self._end_timer("optimize_high_motion_high_detail",
                            include_system_stats=True)
            self.logger.end_phase("High motion + high detail optimization")

    def _optimize_high_motion_low_detail(self, input_path: Path, output_path: Path, target_size_mb: float) -> ProcessingResult:
        """Optimize GIF with high motion but low detail.

        Strategy: Maintain frame rate, aggressively reduce colors, apply moderate scaling.
        """
        temp_output = Path(TEMP_FILE_DIR) / f"hmld_{uuid.uuid4().hex}.gif"

        try:
            # Get dimensions
            dimensions = self._get_dimensions_with_retry(input_path)
            if not dimensions:
                return ProcessingResult(0, 0, None, ProcessingStatus.DIMENSION_ERROR)

            width, height = dimensions

            # Calculate compression parameters
            original_size = self.get_file_size(input_path)
            size_ratio = original_size / target_size_mb

            # Adaptive scaling - can be more aggressive since low detail
            scale = 1.0
            if size_ratio > 4:
                scale = 0.6
            elif size_ratio > 2.5:
                scale = 0.7
            elif size_ratio > 1.5:
                scale = 0.8

            # Aggressively reduce colors for low detail content
            colors = 256
            if size_ratio > 4:
                colors = 196  # Changed from 64 to 196 (75.56% of 256)
            elif size_ratio > 2:
                colors = 216  # Changed from 96

            # Higher lossy value since detail is less important
            lossy = int(min(70, size_ratio * 15))

            # For low detail, Bayer dithering often works better than Floyd-Steinberg
            success = self._compress_with_gifsicle(
                input_path,
                temp_output,
                colors=colors,
                scale=scale,
                lossy=lossy,
                optimize_level=3
            )

            if success and temp_output.exists():
                result_size = self.get_file_size(temp_output)

                # Try more aggressive approach if needed
                if result_size > target_size_mb:
                    second_temp = Path(TEMP_FILE_DIR) / \
                        f"hmld_2_{uuid.uuid4().hex}.gif"

                    # More aggressive settings
                    new_scale = max(0.4, scale - 0.15)
                    new_colors = max(32, colors // 2)
                    new_lossy = min(90, lossy + 30)

                    if self._compress_with_gifsicle(
                        input_path,
                        second_temp,
                        colors=new_colors,
                        scale=new_scale,
                        lossy=new_lossy,
                        optimize_level=3
                    ) and second_temp.exists():
                        second_size = self.get_file_size(second_temp)

                        if second_size < result_size:
                            temp_output = second_temp
                            result_size = second_size

                # Copy to output path
                shutil.copy2(temp_output, output_path)

                return ProcessingResult(
                    fps=0,
                    size=result_size,
                    path=str(output_path),
                    status=ProcessingStatus.SUCCESS if result_size <= target_size_mb else ProcessingStatus.SIZE_THRESHOLD_EXCEEDED,
                    message=f"Optimized to {result_size:.2f}MB (target: {target_size_mb:.2f}MB)",
                    settings=OptimizationConfig(
                        scale_factor=scale,
                        colors=colors,
                        lossy_value=lossy,
                        dither_mode='bayer',
                        bayer_scale=3
                    )
                )
            else:
                return ProcessingResult(0, 0, None, ProcessingStatus.OPTIMIZATION_ERROR, "Optimization failed")

        except Exception as e:
            return ProcessingResult(0, 0, None, ProcessingStatus.OPTIMIZATION_ERROR, f"Error: {e}")

    def _optimize_low_motion_high_detail(self, input_path: Path, output_path: Path, target_size_mb: float) -> ProcessingResult:
        """Optimize GIF with low motion but high detail.

        Strategy: Reduce frame rate, preserve colors and detail, minimal scaling.
        """
        # For low motion, we can use ffmpeg to reduce the frame rate
        temp_reduced_fps = Path(TEMP_FILE_DIR) / \
            f"lmhd_fps_{uuid.uuid4().hex}.gif"
        temp_output = Path(TEMP_FILE_DIR) / f"lmhd_{uuid.uuid4().hex}.gif"

        try:
            # Calculate compression parameters
            original_size = self.get_file_size(input_path)
            size_ratio = original_size / target_size_mb

            # First reduce the frame rate based on motion and size ratio
            target_fps = 10
            if size_ratio > 3:
                target_fps = 8
            elif size_ratio > 2:
                target_fps = 10
            elif size_ratio > 1.5:
                target_fps = 12

            # Generate lower FPS GIF using FFmpeg
            fps_cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-vf', f'fps={target_fps}',
                '-y', str(temp_reduced_fps)
            ]

            if not run_ffmpeg_command(fps_cmd, timeout=180, capture_output=True) or not temp_reduced_fps.exists():
                # Fall back to the original if FPS reduction fails
                shutil.copy2(input_path, temp_reduced_fps)

            # Now optimize with emphasis on color preservation
            scale = 1.0
            if size_ratio > 4:
                scale = 0.8
            elif size_ratio > 2:
                scale = 0.9

            # Preserve more colors for high detail
            colors = 256  # Changed from 192 to 256
            if size_ratio > 3:
                colors = 196  # Changed from 160 to 196 (75.56% of 256)

            # Less lossy compression to preserve detail
            lossy = int(min(40, size_ratio * 8))

            # Run optimization with focus on detail preservation
            success = self._compress_with_gifsicle(
                temp_reduced_fps,
                temp_output,
                colors=colors,
                scale=scale,
                lossy=lossy,
                optimize_level=3
            )

            if success and temp_output.exists():
                result_size = self.get_file_size(temp_output)

                # If still too large, try more aggressive scaling
                if result_size > target_size_mb:
                    second_temp = Path(TEMP_FILE_DIR) / \
                        f"lmhd_2_{uuid.uuid4().hex}.gif"

                    # Focus more on scaling than color reduction
                    new_scale = max(0.6, scale - 0.15)
                    new_colors = max(128, colors - 32)
                    new_lossy = min(50, lossy + 15)

                    if self._compress_with_gifsicle(
                        temp_reduced_fps,
                        second_temp,
                        colors=new_colors,
                        scale=new_scale,
                        lossy=new_lossy,
                        optimize_level=3
                    ) and second_temp.exists():
                        second_size = self.get_file_size(second_temp)

                        if second_size < result_size:
                            temp_output = second_temp
                            result_size = second_size

                # Copy to output path
                shutil.copy2(temp_output, output_path)

                return ProcessingResult(
                    fps=target_fps,
                    size=result_size,
                    path=str(output_path),
                    status=ProcessingStatus.SUCCESS if result_size <= target_size_mb else ProcessingStatus.SIZE_THRESHOLD_EXCEEDED,
                    message=f"Optimized to {result_size:.2f}MB (target: {target_size_mb:.2f}MB)",
                    settings=OptimizationConfig(
                        scale_factor=scale,
                        colors=colors,
                        lossy_value=lossy,
                        dither_mode='floyd_steinberg'
                    )
                )
            else:
                return ProcessingResult(0, 0, None, ProcessingStatus.OPTIMIZATION_ERROR, "Optimization failed")

        except Exception as e:
            return ProcessingResult(0, 0, None, ProcessingStatus.OPTIMIZATION_ERROR, f"Error: {e}")

    def _optimize_low_motion_low_detail(self, input_path: Path, output_path: Path, target_size_mb: float) -> ProcessingResult:
        """Optimize GIF with low motion and low detail.

        Strategy: Aggressive optimization on all parameters (FPS, colors, scale, lossy).
        """
        # We can be very aggressive with optimization here
        temp_reduced_fps = Path(TEMP_FILE_DIR) / \
            f"lmld_fps_{uuid.uuid4().hex}.gif"
        temp_output = Path(TEMP_FILE_DIR) / f"lmld_{uuid.uuid4().hex}.gif"

        try:
            # Calculate compression parameters
            original_size = self.get_file_size(input_path)
            size_ratio = original_size / target_size_mb

            # Aggressive FPS reduction
            target_fps = 8
            if size_ratio > 3:
                target_fps = 5
            elif size_ratio > 2:
                target_fps = 6
            elif size_ratio > 1.5:
                target_fps = 8

            # Generate lower FPS GIF
            fps_cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-vf', f'fps={target_fps}',
                '-y', str(temp_reduced_fps)
            ]

            if not run_ffmpeg_command(fps_cmd, timeout=180, capture_output=True) or not temp_reduced_fps.exists():
                # Fall back to original
                shutil.copy2(input_path, temp_reduced_fps)

            # Very aggressive scaling
            scale = 0.8
            if size_ratio > 4:
                scale = 0.5
            elif size_ratio > 2.5:
                scale = 0.6
            elif size_ratio > 1.5:
                scale = 0.7

            # Minimal colors for low detail
            colors = 256  # Changed from 96 to preserve more colors
            if size_ratio > 3:
                colors = 196  # Changed from 64 to 196 (75.56% of 256)
            elif size_ratio > 2:
                colors = 216  # Changed from 80 to higher value

            # Heavy lossy compression
            # Reduced from min(100, size_ratio * 20)
            lossy = int(min(70, size_ratio * 12))

            # Run aggressive optimization
            success = self._compress_with_gifsicle(
                temp_reduced_fps,
                temp_output,
                colors=colors,
                scale=scale,
                lossy=lossy,
                optimize_level=3
            )

            if success and temp_output.exists():
                result_size = self.get_file_size(temp_output)

                # If incredibly still too large, one last extreme attempt
                if result_size > target_size_mb:
                    second_temp = Path(TEMP_FILE_DIR) / \
                        f"lmld_extreme_{uuid.uuid4().hex}.gif"

                    # Maximum compression settings
                    if self._compress_with_gifsicle(
                        temp_reduced_fps,
                        second_temp,
                        # Changed from 128 to 196 (75.56% of 256, rounded to whole number)
                        colors=196,
                        scale=0.5,  # Kept as is
                        lossy=80,   # Kept as is
                        optimize_level=3
                    ) and second_temp.exists():
                        second_size = self.get_file_size(second_temp)

                        if second_size < result_size:
                            temp_output = second_temp
                            result_size = second_size

                # Copy to output path
                shutil.copy2(temp_output, output_path)

                return ProcessingResult(
                    fps=target_fps,
                    size=result_size,
                    path=str(output_path),
                    status=ProcessingStatus.SUCCESS if result_size <= target_size_mb else ProcessingStatus.SIZE_THRESHOLD_EXCEEDED,
                    message=f"Optimized to {result_size:.2f}MB (target: {target_size_mb:.2f}MB)",
                    settings=OptimizationConfig(
                        scale_factor=scale,
                        colors=colors,
                        lossy_value=lossy,
                        dither_mode='bayer',
                        bayer_scale=5
                    )
                )
            else:
                return ProcessingResult(0, 0, None, ProcessingStatus.OPTIMIZATION_ERROR, "Optimization failed")

        except Exception as e:
            return ProcessingResult(0, 0, None, ProcessingStatus.OPTIMIZATION_ERROR, f"Error: {e}")

    def _compress_large_with_ffmpeg(self, input_path: Path, output_path: Path, scale_factor: float = 0.7, colors: int = 192, dither_method: str = 'floyd_steinberg') -> bool:
        """Compress large GIF using FFmpeg with enhanced quality settings.

        Args:
            input_path: Path to input GIF
            output_path: Path to output GIF
            scale_factor: How much to scale dimensions (between 0.0-1.0)
            colors: Number of colors in palette (higher = better quality but larger file)
            dither_method: Dithering method ('floyd_steinberg' or 'bayer')

        Returns:
            bool: Success status
        """
        try:
            # Check if we need to process at all - skip if already small enough
            file_size_mb = self.get_file_size(input_path)
            target_size_mb = self.compression_settings.get('min_size_mb', 10)

            if file_size_mb <= target_size_mb * 0.95:  # Leave 5% margin
                self.logger.info(
                    f"Skipping FFmpeg - already optimized: {file_size_mb:.2f}MB < {target_size_mb:.2f}MB")
                if input_path != output_path:
                    shutil.copy2(input_path, output_path)
                return True

            # Get dimensions
            dimensions = self._get_dimensions_with_retry(input_path)
            if not dimensions:
                self.logger.error(
                    "Failed to get dimensions, cannot process file")
                return False

            width, height = dimensions
            self.logger.info(
                f"Got dimensions: ({width}, {height})")

            # Validate dimensions to avoid invalid FFmpeg input
            if width <= 0 or height <= 0:
                self.logger.error(f"Invalid dimensions: {width}x{height}")
                return False

            # Calculate scaled dimensions while preserving aspect ratio
            aspect_ratio = width / height

            # Recalculate dimensions more accurately to preserve aspect ratio
            if width > height:  # Landscape orientation
                new_width = max(320, int(width * scale_factor) // 2 * 2)
                new_height = max(
                    180, int(new_width / aspect_ratio + 0.5) // 2 * 2)
            else:  # Portrait orientation
                new_height = max(180, int(height * scale_factor) // 2 * 2)
                new_width = max(
                    320, int(new_height * aspect_ratio + 0.5) // 2 * 2)

            # Double-check aspect ratio
            new_aspect = new_width / new_height
            aspect_error = abs(new_aspect - aspect_ratio) / aspect_ratio
            if aspect_error > 0.05:  # More than 5% error
                self.logger.warning(
                    f"Aspect ratio deviation detected ({aspect_error:.2%}). Recalculating dimensions...")
                # Recalculate to preserve aspect ratio
                if width > height:  # Landscape
                    new_width = max(320, int(width * scale_factor) // 2 * 2)
                    new_height = max(
                        180, int(round(new_width / aspect_ratio)) // 2 * 2)
                else:  # Portrait
                    new_height = max(180, int(height * scale_factor) // 2 * 2)
                    new_width = max(
                        320, int(round(new_height * aspect_ratio)) // 2 * 2)

            self.logger.info(
                f"Using dimensions {new_width}x{new_height} (original: {width}x{height}, aspect ratio: {aspect_ratio:.3f})")

            # Use appropriate frames per second - maintain higher FPS for better quality
            fps = 15
            if file_size_mb > target_size_mb * 5:
                fps = 12
            elif file_size_mb > target_size_mb * 10:
                fps = 10

            # Skip direct FFmpeg conversion and only use the two-pass method
            self.logger.info(
                f"File is {file_size_mb/target_size_mb:.1f}x too large, using two-pass FFmpeg conversion")

            # Create a unique temporary filename for intermediate GIF
            temp_palette = Path(TEMP_FILE_DIR) / \
                f"palette_{uuid.uuid4().hex}.png"
            temp_gif = Path(TEMP_FILE_DIR) / \
                f"temp_twopass_{uuid.uuid4().hex}.gif"

            try:
                # Check if this is a video file that needs keyframe handling
                is_video = self._is_video_file(str(input_path))
                keyframe_flags = ['-ignore_loop', '0',
                                  '-vsync', '0'] if is_video else []

                # Create palette with enhanced settings for better quality
                stats_mode = 'full'  # Always use full mode for better quality

                # Create palette command
                palette_cmd = ['ffmpeg']
                if is_video:
                    palette_cmd.extend(keyframe_flags)

                palette_cmd.extend([
                    '-i', str(input_path),
                    '-vf', f'fps={fps},scale={new_width}:{new_height}:flags=lanczos,palettegen=max_colors={colors}:reserve_transparent=0:stats_mode={stats_mode}',
                    '-y', str(temp_palette)
                ])

                # Run palette command with timeout
                if not run_ffmpeg_command(palette_cmd, timeout=180, capture_output=True) or not temp_palette.exists():
                    self.logger.error("Failed to generate palette")
                    return False

                # Setup dithering parameters - always use floyd_steinberg for best quality
                dither_params = 'dither=floyd_steinberg:diff_mode=rectangle'

                # Create GIF command
                gif_cmd = ['ffmpeg']
                if is_video:
                    gif_cmd.extend(keyframe_flags)

                gif_cmd.extend([
                    '-i', str(input_path),
                    '-i', str(temp_palette),
                    '-lavfi', f'fps={fps},scale={new_width}:{new_height}:flags=lanczos[x];[x][1:v]paletteuse={dither_params}',
                    '-y', str(temp_gif)
                ])

                # Run GIF command with timeout
                if not run_ffmpeg_command(gif_cmd, timeout=180, capture_output=True) or not temp_gif.exists():
                    self.logger.error("Failed to create GIF")
                    return False

                # Verify the file is valid
                try:
                    with Image.open(temp_gif) as img:
                        if img.format != 'GIF':
                            self.logger.warning(
                                f"Output is not a valid GIF: {img.format}")
                            return False
                except Exception as e:
                    self.logger.warning(f"Invalid GIF created: {e}")
                    return False

                # Copy to the output path
                shutil.copy2(temp_gif, output_path)
                return True

            finally:
                # Clean up temporary files safely
                try:
                    if temp_palette.exists():
                        temp_palette.unlink(missing_ok=True)
                    if temp_gif.exists():
                        temp_gif.unlink(missing_ok=True)
                except Exception as e:
                    self.logger.error(
                        f"Failed to clean up temporary files: {e}")

                # Force garbage collection to release file handles
                gc.collect()

        except Exception as e:
            self.logger.error(f"FFmpeg compression error: {e}", exc_info=True)
            return False

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

                # Only return success if target size was met
                return result.success and result.size <= target_size_mb
            else:
                # For existing GIFs, optimize directly
                result = self._progressive_optimize(
                    file_path, output_path, target_size_mb)

                # Only return success if target size was met
                return result.success and result.size <= target_size_mb

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
                {"scale": 0.25, "colors": 64, "lossy": 120}
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
                        # We couldn't meet the target size
                        status = ProcessingStatus.SIZE_THRESHOLD_EXCEEDED
                        message = f"Best optimization still exceeds target: {best_size:.2f}MB > {target_size_mb:.2f}MB"
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
                        success=met_target  # Only count as success if target size is met
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
                cmd.append(f"--lossy={lossy}")

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

    def _run_command_with_logging(self, command: list, timeout=None) -> bool:
        """Run a command with proper logging.

        Args:
            command: Command list to run
            timeout: Optional timeout in seconds

        Returns:
            bool: Success status
        """
        # Start performance timer
        cmd_name = command[0] if command else "unknown"
        timer_name = f"cmd_{cmd_name}_{uuid.uuid4().hex[:6]}"
        self._start_timer(timer_name)

        try:
            # Log the command details
            cmd_str = ' '.join(command)
            truncated_cmd = cmd_str[:100] + \
                "..." if len(cmd_str) > 100 else cmd_str

            self._log_message("INFO",
                              f"Running command: {truncated_cmd}",
                              f"cmd_start_{cmd_name}")

            # Log detailed command information
            self.logger.debug(f"Executing command with timeout {timeout}s", extra={
                'details': {
                    "Command": truncated_cmd,
                    "Timeout": f"{timeout}s" if timeout else "None"
                }
            })

            # Run the command
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=timeout,
                check=False,  # Don't raise exception on non-zero exit
                text=True
            )

            # Process the results
            exit_code = process.returncode
            success = exit_code == 0

            # Log command outcome
            if success:
                self._log_message("SUCCESS",
                                  f"Command completed successfully (exit code: {exit_code})",
                                  f"cmd_success_{cmd_name}")
            else:
                # Log error with details
                stderr_preview = process.stderr[:200] + "..." if len(
                    process.stderr) > 200 else process.stderr
                self._log_message("ERROR",
                                  f"Command failed with exit code {exit_code}",
                                  f"cmd_error_{cmd_name}")

                # Log stderr to help with debugging
                if stderr_preview.strip():
                    self.logger.error(f"Command error output", extra={
                        'details': {
                            "Exit code": str(exit_code),
                            "Command": truncated_cmd,
                            "Error": stderr_preview
                        }
                    })

            return success
        except subprocess.TimeoutExpired:
            self._log_message("ERROR",
                              f"Command timed out after {timeout} seconds",
                              f"cmd_timeout_{cmd_name}")
            return False
        except Exception as e:
            self._log_message("ERROR",
                              f"Error running command: {e}",
                              f"cmd_exception_{cmd_name}")

            # Log exception details
            self.logger.error(f"Command execution error", extra={
                'details': {
                    "Exception": str(e),
                    "Command": truncated_cmd
                }
            }, exc_info=True)

            return False
        finally:
            # End performance timer
            self._end_timer(timer_name)

    def _safe_cleanup_temp_files(self, temp_files: List[Path]) -> None:
        """Safely clean up temporary files with proper error handling.

        Args:
            temp_files: List of Path objects to clean up
        """
        for file_path in temp_files:
            try:
                if file_path and file_path.exists():
                    file_path.unlink(missing_ok=True)
            except Exception as e:
                self.logger.debug(
                    f"Failed to clean up temp file {file_path}: {e}")

        # Force garbage collection to help release file handles
        gc.collect()

    def _log_optimization_result(self, original_size: float, result_size: float,
                                 target_size_mb: float, strategy_name: str, file_path: Path) -> None:
        """Log the results of an optimization strategy with consistent formatting.

        Args:
            original_size: Size in MB before optimization
            result_size: Size in MB after optimization
            target_size_mb: Target size in MB
            strategy_name: Name of the optimization strategy used
            file_path: Path to the optimized file
        """
        # Calculate reduction statistics
        reduction = (1 - result_size / original_size) * 100
        target_met = result_size <= target_size_mb

        # Determine status based on whether target was met
        if target_met:
            status = GIFProcessingStatus.SUCCESS
            log_level = "SUCCESS"
        else:
            status = GIFProcessingStatus.WARNING
            log_level = "WARNING"

        # Create details dictionary for extended logging
        details = {
            "Original size": f"{original_size:.2f}MB",
            "Final size": f"{result_size:.2f}MB",
            "Reduction": f"{reduction:.1f}%",
            "Target": f"{target_size_mb:.2f}MB",
            "Target met": "Yes" if target_met else "No",
            "Strategy": strategy_name
        }

        # Log the main message with appropriate level and icon
        icon, color = status.value
        message = f"{strategy_name} optimization: {original_size:.2f}MB → {result_size:.2f}MB ({reduction:.1f}% reduction)"

        if target_met:
            message += f" - Target met ({target_size_mb:.2f}MB)"
        else:
            message += f" - Target not met ({result_size:.2f}MB > {target_size_mb:.2f}MB)"

        self._log_message(log_level, message,
                          f"opt_result_{strategy_name}_{str(file_path)}")

        # Log extended details with the logger
        self.logger.log(
            SUCCESS_LEVEL if target_met else logging.WARNING,
            f"Optimization result [{strategy_name}]",
            extra={'details': details}
        )
