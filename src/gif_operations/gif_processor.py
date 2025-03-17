import asyncio
from asyncio.log import logger
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import gc
import json
import logging
import os
from pathlib import Path
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from tkinter import Image
import traceback
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple, Any, Set

from cachetools import TTLCache
import cv2
import psutil
from tqdm import tqdm
import atexit
import uuid

from src.default_config import GIF_SIZE_TO_SKIP, INPUT_DIR, OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS, TEMP_FILE_DIR, GIF_COMPRESSION
from src.gif_operations.ffmpeg_handler import FFmpegHandler
from src.gif_operations.memory import MemoryManager
from src.gif_operations.optimizer import GIFOptimizer, ProcessingStatus
from src.gif_operations.processing_stats import ProcessingStats
from src.gif_operations.quality_manager import QualityManager
from src.gif_operations.resource_manager import ResourceMonitor
from src.logging_system import get_logger, log_function_call, run_ffmpeg_command, performance_monitor, ModernLogStyle, UnifiedLogger
from src.temp_file_manager import TempFileManager
# Add DynamicGIFOptimizer to import
from src.gif_operations.optimizer import GIFOptimizer, DynamicGIFOptimizer
from src.base.processor import BaseProcessor
from ..utils.video_dimensions import _validate_dimensions as validate_dimensions


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
    """Decorator to monitor function performance."""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            return await func(*args, **kwargs)
        finally:
            end_time = time.time()
            logger.debug(f"{func.__name__} took {end_time - start_time:.2f}s")

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end_time = time.time()
            logger.debug(f"{func.__name__} took {end_time - start_time:.2f}s")

    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper


class OptimizationConfig(NamedTuple):
    scale_factor: float
    colors: int
    lossy_value: int
    dither_mode: str = 'bayer'
    bayer_scale: int = 3


class ProcessingResult(NamedTuple):
    """Improved result tracking."""
    success: bool
    size: float
    settings: OptimizationConfig
    error: Optional[str] = None


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
    fps: int
    size: float
    path: Optional[str]
    status: ProcessingStatus
    message: str = ""
    settings: Optional[OptimizationConfig] = None


class GIFProcessingStatus(Enum):
    """Enhanced GIF processing status with visual indicators"""
    STARTING = ('⚡', ModernLogStyle.AZURE)
    PROCESSING = ('↻', ModernLogStyle.CYAN)
    OPTIMIZING = ('⚙', ModernLogStyle.SLATE)
    SUCCESS = ('✓', ModernLogStyle.EMERALD)
    WARNING = ('⚠', ModernLogStyle.AMBER)
    ERROR = ('✖', ModernLogStyle.ROSE)
    SKIPPED = ('→', ModernLogStyle.SLATE)


class GIFProcessor(BaseProcessor):
    """Main GIF processing class combining optimization and conversion."""

    def __init__(self, logger=None, gpu_enabled=False, gpu_settings=None):
        """Initialize GIF processor with GPU settings.

        Args:
            logger: Optional logger to use. If None, a module-specific logger is created.
            gpu_enabled: Whether GPU acceleration is enabled.
            gpu_settings: Dictionary of GPU settings to use.
        """
        # Initialize base class with logger
        super().__init__(logger=logger or get_logger('gif_processor'))

        # Initialize thread-safe locks and events
        self._shutdown_event = threading.Event()
        self._processing_cancelled = threading.Event()
        self._immediate_termination = threading.Event()
        self._shutdown_initiated = False
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

        # Increase memory threshold based on system memory
        system_memory_gb = psutil.virtual_memory().total / (1024**3)
        # 30% of system memory, min 1.5GB, max 4GB
        memory_threshold = min(
            max(1500, int(system_memory_gb * 0.3 * 1024)), 4096)

        self.memory_manager = MemoryManager(threshold_mb=memory_threshold)
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
        if self._shutdown_initiated:
            return

        self._shutdown_initiated = True
        self._shutdown_event.set()

        try:
            # First kill any running processes
            self._kill_ffmpeg_processes()

            # Wait a moment for processes to die
            time.sleep(0.5)

            # Force close any file handles
            gc.collect()

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

            # Clean temp files with multiple attempts
            self._cleanup_temp_directory()

            # Clear caches
            self._clear_caches()

        except Exception as e:
            self.logger.error(f"Error during cleanup: {str(e)}")
        finally:
            self._stop_workers.set()
            self._cleanup_threads()

    def _cleanup_temp_directory(self):
        """Clean up temporary directory with improved handling."""
        temp_dir = Path(TEMP_FILE_DIR)
        if not temp_dir.exists():
            return

        try:
            # Multiple attempts to clean files
            for attempt in range(3):
                remaining_files = []

                # Try to remove each file
                for item in temp_dir.glob("*"):
                    try:
                        if item.is_file():
                            try:
                                item.unlink(missing_ok=True)
                            except Exception:
                                remaining_files.append(item)
                        elif item.is_dir():
                            try:
                                shutil.rmtree(item, ignore_errors=True)
                            except Exception:
                                remaining_files.append(item)
                    except Exception as e:
                        self.logger.debug(
                            f"Failed to remove {item}: {e}")
                        remaining_files.append(item)

                if not remaining_files:
                    break

                # If files remain, try more aggressive cleanup
                if attempt < 2:
                    self._kill_ffmpeg_processes()
                    gc.collect()
                    time.sleep(1)

            # If files still remain, log them but continue
            if remaining_files:
                self.logger.warning(
                    f"Could not remove {len(remaining_files)} files in temp directory")
                for file in remaining_files:
                    self.logger.debug(f"Remaining file: {file}")

            # Don't try to remove the temp directory if files remain
            if not remaining_files:
                try:
                    temp_dir.rmdir()
                except Exception as e:
                    self.logger.debug(
                        f"Could not remove temp directory: {e}")

            # Always ensure temp directory exists
            temp_dir.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            self.logger.error(
                f"Error cleaning temp directory: {str(e)}")
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
        """Kill any running FFmpeg processes."""
        try:
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/F', '/IM', 'ffmpeg.exe'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                subprocess.run(['taskkill', '/F', '/IM', 'gifsicle.exe'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                os.system("pkill -9 ffmpeg")
                os.system("pkill -9 gifsicle")
        except Exception as e:
            self.logger.error(f"Failed to kill processes: {e}")

    def _log_with_lock(self, level: str, message: str, file_id: str = "") -> None:
        """Thread-safe logging with deduplication."""
        with self.logging_lock:
            log_key = f"{file_id}:{message}"
            if log_key not in self.processed_files:
                if level == "info":
                    self.logger.info(message)
                elif level == "error":
                    self.logger.error(message)
                elif level == "warning":
                    self.logger.warning(message)
                elif level == "success":
                    self.logger.success(message)
                self.processed_files.add(log_key)

    @log_function_call
    def create_gif(self, file_path: Path, output_path: Path, fps: int, dimensions: Tuple[int, int]) -> bool:
        """Create optimized GIF with enhanced quality and size control."""
        # Initialize temp file variables outside the try block to ensure they exist in the finally block
        temp_palette = None
        temp_output = None

        try:
            # Use the configured target size instead of hardcoded default
            target_size_mb = self.compression_settings['min_size_mb']
            MAX_INTERMEDIATE_SIZE_MB = max(100, target_size_mb * 2)

            original_size = self.get_file_size(file_path)
            if original_size > 1000:
                self.logger.error(
                    f"Source file too large: {original_size:.2f}MB")
                return False

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
                '-vf', f'fps={adjusted_fps},scale={new_width}:{new_height}:flags=lanczos,palettegen=max_colors={color_count}:stats_mode=diff',
                '-y', str(temp_palette)
            ]

            if not run_ffmpeg_command(palette_cmd):
                self.logger.error("Palette generation failed")
                return False

            # Create the GIF with the generated palette
            dither_method = "bayer" if original_size > 100 else "floyd_steinberg"
            bayer_scale = 5 if original_size > 200 else 3

            gif_cmd = [
                'ffmpeg', '-i', str(file_path),
                '-i', str(temp_palette),
                '-lavfi',
                f'fps={adjusted_fps},scale={new_width}:{new_height}:flags=lanczos[x];[x][1:v]paletteuse=dither={dither_method}:bayer_scale={bayer_scale}:diff_mode=rectangle',
                '-y', str(temp_output)
            ]

            if not run_ffmpeg_command(gif_cmd):
                self.logger.error("Initial GIF creation failed")
                return False

            # Check and optimize the initial GIF
            if temp_output.exists():
                initial_size = self.get_file_size(temp_output)

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

                final_size = self.get_file_size(output_path)
                if final_size > target_size_mb:
                    self.logger.info(
                        f"Further optimization needed: {final_size:.2f}MB > {target_size_mb}MB")
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
            # Cleanup temp files
            try:
                if temp_palette and temp_palette.exists():
                    temp_palette.unlink(missing_ok=True)
                if temp_output and temp_output.exists():
                    temp_output.unlink(missing_ok=True)
            except Exception as cleanup_e:
                self.logger.error(f"Cleanup error: {cleanup_e}")

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

    def process_file(self, file_path: Path, output_path: Path, is_video: bool) -> bool:
        """
        Process a single file (GIF or video) synchronously.

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

                # Process based on type
                if is_video:
                    fps = self._get_source_fps(file_path)
                    success = self.create_gif(
                        file_path, output_path, int(fps), dimensions)

                    # Verify final file size after creation
                    if success and output_path.exists():
                        final_size = self.get_file_size(output_path)
                        if final_size > target_size:
                            self.logger.warning(
                                f"Video-to-GIF conversion resulted in a file larger than target size: {final_size:.2f}MB > {target_size:.2f}MB")
                            # Try to optimize the created GIF further
                            optimized_size, opt_success = self.optimize_gif(
                                output_path, output_path, target_size)
                            if not opt_success:
                                self.logger.error(
                                    f"Failed to optimize GIF to target size. Final size: {optimized_size:.2f}MB")
                                # We'll still return True since we have a valid GIF, just oversized

                    return success
                else:
                    # For existing GIFs, use the progressive optimization directly
                    result = self._progressive_optimize(
                        file_path, output_path, target_size)

                    # If optimization succeeded but file is still over size limit, try more aggressive optimization
                    if result.status == ProcessingStatus.SUCCESS and output_path.exists():
                        final_size = self.get_file_size(output_path)
                        if final_size > target_size:
                            self.logger.warning(
                                f"GIF optimization resulted in a file larger than target: {final_size:.2f}MB > {target_size:.2f}MB")
                            # Apply more aggressive optimization
                            optimized_size, opt_success = self.optimize_gif(
                                output_path, output_path, target_size)
                            if not opt_success:
                                self.logger.error(
                                    f"Failed to optimize GIF to target size. Final size: {optimized_size:.2f}MB")

                    return result.status == ProcessingStatus.SUCCESS

            finally:
                if file_lock:
                    file_lock.release()

        except Exception as e:
            self.logger.error(f"Processing error for {file_path}: {str(e)}")
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
        """Validate that dimensions are reasonable."""
        if dimensions is None:
            return False

        try:
            # Use the utility function if available
            from ..utils.video_dimensions import _validate_dimensions
            return _validate_dimensions(dimensions)
        except ImportError:
            # Fallback implementation
            if not dimensions or len(dimensions) != 2:
                return False

            width, height = dimensions
            if not isinstance(width, int) or not isinstance(height, int):
                return False

            # Check dimension limits
            if width <= 0 or height <= 0:
                return False
            if width > 7680 or height > 4320:  # 8K resolution limit
                return False
            if width < 16 or height < 16:  # Minimum reasonable dimensions
                return False

            return True

    def optimize_gif(self, input_path: Path, output_path: Path, target_size_mb: float = None) -> Tuple[float, bool]:
        """Optimize GIF until it reaches target size or max attempts are reached."""
        # Initialize these variables outside the try block to ensure they exist in the finally block
        temp_output = None
        current_input = input_path
        best_output = None
        original_size = 0

        try:
            original_size = self.get_file_size(input_path)

            # Use the configured value if no target size is specified
            if target_size_mb is None:
                target_size_mb = self.compression_settings['min_size_mb']

            # Ensure we never exceed the configured limit
            target_size_mb = min(
                target_size_mb, self.compression_settings['min_size_mb'])

            # Copy if already small enough
            if original_size <= target_size_mb:
                shutil.copy2(input_path, output_path)
                return original_size, True

            max_attempts = 8  # Increase maximum optimization attempts for better size reduction
            current_size = original_size
            best_size = float('inf')

            for attempt in range(max_attempts):
                if self._should_exit():
                    break

                temp_output = Path(TEMP_FILE_DIR) / \
                    f"temp_opt_{attempt}_{output_path.name}"

                # For larger files, use progressive reduction
                if current_size > 50:
                    # Reduce scale factor progressively based on attempt
                    scale_factor = max(0.3, 1.0 - (attempt * 0.1))
                    colors = max(32, 256 - (attempt * 32))
                    lossy_value = min(95, 60 + (attempt * 5))

                    # For really large files, use FFmpeg with more aggressive scaling
                    if current_size > 100:
                        success = self._compress_large_with_ffmpeg(
                            current_input, temp_output, scale_factor=scale_factor)
                    else:
                        success = self._compress_with_gifsicle(
                            current_input, temp_output, colors=colors, lossy=lossy_value)
                else:
                    # For smaller files, use gifsicle with gradually increasing compression
                    lossy_value = min(95, 50 + (attempt * 10))
                    colors = max(32, 256 - (attempt * 32))
                    success = self._compress_with_gifsicle(
                        current_input, temp_output, colors=colors, lossy=lossy_value)

                if not success or not temp_output.exists():
                    self.logger.error(
                        f"Optimization attempt {attempt + 1} failed")
                    continue

                new_size = self.get_file_size(temp_output)
                self.logger.info(
                    f"Attempt {attempt + 1}: {new_size:.2f}MB (Target: {target_size_mb:.2f}MB)")

                # Keep track of the best result so far
                if new_size <= target_size_mb and (best_output is None or new_size > best_size):
                    # Best result that meets target size criteria
                    if best_output and best_output != current_input and best_output != input_path:
                        best_output.unlink(missing_ok=True)
                    best_size = new_size
                    best_output = temp_output

                    # If we're very close to target size, no need to optimize further
                    if target_size_mb - new_size < 0.5:
                        break

                    # Continue with optimization to find even better result
                    if current_input != input_path:
                        current_input.unlink(missing_ok=True)
                    current_input = temp_output
                    continue
                elif new_size <= target_size_mb:
                    # Target reached but smaller than previous best
                    if best_output and best_output != current_input and best_output != input_path:
                        best_output.unlink(missing_ok=True)
                    best_size = new_size
                    best_output = temp_output
                    break
                elif new_size < current_size:
                    # Better but not enough - continue optimizing
                    if best_output and best_output != current_input and best_output != input_path:
                        best_output.unlink(missing_ok=True)
                    current_size = new_size
                    if current_input != input_path:
                        current_input.unlink(missing_ok=True)
                    current_input = temp_output
                    continue
                else:
                    # No improvement
                    temp_output.unlink(missing_ok=True)
                    temp_output = None
            # Use best result if we found one that meets target size
            if best_output and best_size <= target_size_mb:
                shutil.move(str(best_output), str(output_path))
                return best_size, True

            # If we have a better result but didn't reach target, use most aggressive compression
            if current_size < original_size:
                # Last-ditch effort with maximum compression
                temp_final = Path(TEMP_FILE_DIR) / \
                    f"temp_final_{output_path.name}"
                success = self._compress_with_gifsicle(
                    current_input, temp_final, colors=32, lossy=95, scale=0.5)

                if success and temp_final.exists():
                    final_size = self.get_file_size(temp_final)
                    if final_size <= target_size_mb:
                        shutil.move(str(temp_final), str(output_path))
                        return final_size, True
                    elif final_size < current_size:
                        shutil.move(str(temp_final), str(output_path))
                        # Still failed to meet target
                        return final_size, False
                    else:
                        temp_final.unlink(missing_ok=True)
                        if current_input != input_path:
                            shutil.move(str(current_input), str(output_path))
                        return current_size, False
                else:
                    if current_input != input_path:
                        shutil.move(str(current_input), str(output_path))
                    return current_size, False

            # If we reach here, optimization failed to meet target size
            self.logger.error(
                f"Failed to optimize GIF to target size {target_size_mb}MB")
            return original_size, False

        except Exception as e:
            self.logger.error(f"Optimization error: {e}")
            return original_size, False
        finally:
            # Cleanup any remaining temp files
            try:
                if temp_output and temp_output.exists():
                    temp_output.unlink(missing_ok=True)
                if current_input != input_path and current_input.exists():
                    current_input.unlink(missing_ok=True)
                if best_output and best_output.exists() and best_output != current_input:
                    best_output.unlink(missing_ok=True)
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")

    def _compress_with_gifsicle(self, input_path: Path, output_path: Path, colors: int = 128, lossy: int = 80, scale: float = 1.0) -> bool:
        """Compress GIF using gifsicle with configurable settings."""
        try:
            cmd = [
                'gifsicle',
                '--optimize=3',
                '--colors', str(colors),
                '--lossy=' + str(lossy),
                '--no-conserve-memory',
            ]

            # Add scale if less than 1.0
            if scale < 1.0:
                cmd.extend(['--scale', str(scale)])

            # Add input and output paths
            cmd.extend([str(input_path), '-o', str(output_path)])

            return run_ffmpeg_command(cmd)
        except Exception as e:
            self.logger.error(f"Gifsicle compression failed: {e}")
            return False

    def _compress_large_with_ffmpeg(self, input_path: Path, output_path: Path, scale_factor: float = 0.7) -> bool:
        """Compress large GIF using FFmpeg with configurable settings."""
        try:
            # Get dimensions
            dimensions = self._get_dimensions_with_retry(input_path)
            if not dimensions:
                return False

            width, height = dimensions
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-vf', f'scale={new_width}:{new_height}:flags=lanczos,fps=15',
                '-c:v', 'gif',
                '-f', 'gif',
                '-y', str(output_path)
            ]

            return run_ffmpeg_command(cmd)
        except Exception as e:
            self.logger.error(f"FFmpeg compression failed: {e}")
            return False

    def _get_optimized_configs(self, input_size: float, target_size: float,
                               source_fps: float, dimensions: Tuple[int, int],
                               source_colors: int) -> List[Dict]:
        """Generate optimized configurations based on input characteristics."""
        configs = []
        size_ratio = target_size / input_size
        max_dimension = max(dimensions)

        # Scale factors based on dimensions and ratio
        if max_dimension > 1920:
            scale_factors = [0.75, 0.5, 0.35]
        elif max_dimension > 1280:
            scale_factors = [0.85, 0.75, 0.5]
        else:
            scale_factors = [1.0, 0.85, 0.75]

        # Use source colors as maximum
        color_configs = [min(256, source_colors)]
        if size_ratio < 0.3:
            color_configs.extend([
                min(192, int(source_colors * 0.75)),
                min(128, int(source_colors * 0.5))
            ])

        # Create configs prioritizing quality
        for scale in scale_factors:
            for colors in color_configs:
                configs.append({
                    'scale_factor': scale,
                    'colors': colors,
                    'lossy_value': min(100, int(40 * (1/scale))),
                    'priority': scale * (colors / source_colors)
                })

        return sorted(configs, key=lambda x: x['priority'], reverse=True)

    def _get_source_fps(self, file_path: Path) -> float:
        """Get the original framerate of the source file."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'json',
                str(file_path)
            ]

            output = subprocess.check_output(cmd, text=True)
            data = json.loads(output)

            if 'streams' in data and data['streams']:
                # Parse fractional framerate (e.g., "24000/1001")
                rate = data['streams'][0].get('r_frame_rate', '')
                if rate and '/' in rate:
                    num, den = map(float, rate.split('/'))
                    if den != 0:
                        return round(num / den, 2)
                elif rate:
                    return float(rate)

            # Fallback to manual frame counting
            frame_info = self._get_video_frame_info(file_path)
            if frame_info:
                frames, duration = frame_info
                if duration > 0:
                    return round(frames / duration, 2)

            return 30.0  # Default fallback

        except Exception as e:
            self.logger.warning(f"Failed to get source FPS: {e}")
            return 30.0  # Safe default

    def _progressive_optimize(self, input_path: Path, output_path: Path, target_size_mb: float) -> ProcessingResult:
        """Progressive optimization with quality reduction."""
        # Initialize temp_output outside the try block to ensure it exists in the finally block
        temp_output = None

        try:
            # Instead of creating a temp directory, just use temp files directly in TEMP_FILE_DIR
            timestamp = int(time.time() * 1000)
            temp_output = Path(TEMP_FILE_DIR) / \
                f"temp_opt_{timestamp}_{output_path.name}"

            original_size = self.get_file_size(input_path)
            dimensions = self._get_dimensions_with_retry(input_path)
            if not dimensions:
                return ProcessingResult(0, original_size, None, ProcessingStatus.DIMENSION_ERROR)

            # Start with best quality settings
            config = OptimizationConfig(
                scale_factor=1.0,
                colors=256,
                lossy_value=20,
                dither_mode='floyd_steinberg'
            )

            best_result = None
            for attempt in range(3):
                if self._should_exit():
                    break

                result = self._process_single_config(
                    input_path,
                    temp_output,
                    self.compression_settings['fps_range'][0],
                    dimensions[0],
                    dimensions[1],
                    config,
                    target_size_mb
                )

                # Add null check for result before accessing
                if result is None:
                    continue

                # Now safely access result dictionary
                if result.get('success'):
                    if result.get('size', float('inf')) <= target_size_mb:
                        return ProcessingResult(
                            result.get('fps', 0),
                            result.get('size', 0),
                            result.get('path'),
                            ProcessingStatus.SUCCESS,
                            settings=config
                        )

                    # Update best_result if we have a better size
                    if not best_result or result.get('size', float('inf')) < best_result.get('size', float('inf')):
                        best_result = result

                # Adjust settings for next attempt - safely handle None case
                config = self._adjust_config(config, result.get(
                    'size', float('inf')), target_size_mb)

            # Use best result if we have one
            if best_result:
                return ProcessingResult(
                    best_result.get('fps', 0),
                    best_result.get('size', 0),
                    best_result.get('path'),
                    ProcessingStatus.SUCCESS,
                    settings=config
                )

            return ProcessingResult(0, original_size, None, ProcessingStatus.OPTIMIZATION_ERROR)

        except Exception as e:
            self.logger.error(
                f"Progressive optimization failed: {str(e)}")
            return ProcessingResult(0, original_size, None, ProcessingStatus.OPTIMIZATION_ERROR, str(e))
        finally:
            # Clean up temp_output if it exists
            try:
                if temp_output and temp_output.exists():
                    temp_output.unlink(missing_ok=True)
            except Exception as cleanup_e:
                self.logger.error(f"Cleanup error: {cleanup_e}")

    def _analyze_source_colors(self, input_path: Path) -> int:
        """Analyze color distribution in the GIF with better preservation."""
        try:
            from PIL import Image
            with Image.open(input_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                colors = img.getcolors(maxcolors=257)
                if not colors:  # More than 256 colors
                    return 256  # Always try to preserve maximum colors

                total_pixels = img.size[0] * img.size[1]
                threshold = total_pixels * 0.0005  # Reduced threshold for better preservation
                significant_colors = sum(
                    1 for count, _ in colors if count > threshold)

                # More generous color preservation
                if significant_colors > 128:
                    return 256
                elif significant_colors > 64:
                    return 192
                else:
                    return 128  # Minimum color count increased

        except Exception as e:
            self.logger.warning(
                f"Color analysis failed: {e}, using maximum colors")
            return 256  # Default to maximum colors on error

    def _adjust_config(self, config: OptimizationConfig, current_size: float, target_size: float) -> OptimizationConfig:
        """Adjust optimization config based on results."""
        if current_size == float('inf'):
            # If previous attempt failed, be more aggressive
            return OptimizationConfig(
                scale_factor=max(0.5, config.scale_factor * 0.8),
                colors=min(config.colors, 128),
                lossy_value=min(100, config.lossy_value + 20)
            )

        size_ratio = current_size / target_size
        if size_ratio > 2:
            # Far from target, be aggressive
            return OptimizationConfig(
                scale_factor=max(0.4, config.scale_factor * 0.7),
                colors=max(64, config.colors - 64),
                lossy_value=min(100, config.lossy_value + 30)
            )
        elif size_ratio > 1.5:
            # Getting closer, be more careful
            return OptimizationConfig(
                scale_factor=max(0.6, config.scale_factor * 0.85),
                colors=max(128, config.colors - 32),
                lossy_value=min(100, config.lossy_value + 15)
            )
        else:
            # Fine tuning
            return OptimizationConfig(
                scale_factor=max(0.7, config.scale_factor * 0.9),
                colors=max(128, config.colors - 16),
                lossy_value=min(100, config.lossy_value + 10)
            )

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

            # For very large files, add even more aggressive configs
            if original_size > 100:
                configs.extend([
                    {'scale': 0.3, 'fps': 5, 'colors': 24, 'lossy': 95},
                    {'scale': 0.25, 'fps': 4, 'colors': 16, 'lossy': 100}
                ])

            best_result = None

            for i, config in enumerate(configs):
                if self._should_exit():
                    break

                try:
                    # Calculate dimensions
                    new_width = int(dimensions[0] * config['scale'] // 2 * 2)
                    new_height = int(dimensions[1] * config['scale'] // 2 * 2)

                    # Ensure minimum dimensions
                    new_width = max(new_width, 200)
                    new_height = max(new_height, 150)

                    # Use UUID-based filenames to avoid FFmpeg warnings
                    config_id = f"{unique_id}_{i}"
                    temp_palette = temp_dir / f"palette_{config_id}.png"
                    temp_output = temp_dir / f"output_{config_id}.gif"

                    # Create palette
                    palette_cmd = [
                        'ffmpeg', '-i', str(file_path),
                        '-vf', f"fps={config['fps']},scale={new_width}:{new_height}:flags=lanczos,palettegen=max_colors={config['colors']}:stats_mode=diff",
                        '-y', str(temp_palette)
                    ]

                    if not run_ffmpeg_command(palette_cmd):
                        continue

                    # Create GIF
                    gif_cmd = [
                        'ffmpeg', '-i', str(file_path),
                        '-i', str(temp_palette),
                        '-lavfi', f"fps={config['fps']},scale={new_width}:{new_height}:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle",
                        '-y', str(temp_output)
                    ]

                    if not run_ffmpeg_command(gif_cmd):
                        continue

                    # Optimize with gifsicle
                    if temp_output.exists():
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

                except Exception as e:
                    self.logger.error(f"Error with config {config}: {e}")
                    continue

            # Use best result or most aggressive as fallback
            if best_result is not None:
                shutil.copy2(best_result['path'], output_path)
                self.logger.success(
                    f"Aggressive optimization succeeded: {best_result['size']:.2f}MB (Target: {target_size}MB)"
                )
                return True
            else:
                # Last attempt with maximum compression
                try:
                    final_id = str(uuid.uuid4())
                    last_chance = temp_dir / f"last_attempt_{final_id}.gif"
                    final_palette = temp_dir / f"final_palette_{final_id}.png"

                    # Extremely reduced settings
                    min_width = max(160, int(dimensions[0] * 0.2))
                    min_height = max(120, int(dimensions[1] * 0.2))
                    min_fps = max(3, fps // 5)

                    # Create palette
                    palette_cmd = [
                        'ffmpeg', '-i', str(file_path),
                        '-vf', f"fps={min_fps},scale={min_width}:{min_height}:flags=fast_bilinear,palettegen=max_colors=16",
                        '-y', str(final_palette)
                    ]

                    if run_ffmpeg_command(palette_cmd):
                        # Create GIF
                        gif_cmd = [
                            'ffmpeg', '-i', str(file_path),
                            '-i', str(final_palette),
                            '-lavfi', f"fps={min_fps},scale={min_width}:{min_height}:flags=fast_bilinear[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=1",
                            '-y', str(last_chance)
                        ]

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

    def _signal_handler(self, signum, frame):
        """Handle termination signals gracefully."""
        signal_name = signal.Signals(signum).name if hasattr(
            signal, 'Signals') else str(signum)
        self.logger.info(
            f"Received signal {signal_name}. Initiating graceful shutdown...")

        # Set shutdown flag
        self._shutdown_event.set()
        self._processing_cancelled.set()

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

    def _should_exit(self) -> bool:
        """Check if processing should be stopped."""
        return (self._shutdown_event.is_set() or
                self._processing_cancelled.is_set() or
                self._immediate_termination.is_set())

    @performance_monitor
    def process_all(self) -> List[Path]:
        """Process all files in the queue with sequential processing - one at a time."""
        start_time = time.time()
        processed_files = []

        # First, find all GIF files in the input directory if queue is empty
        if self.task_queue.qsize() == 0:
            self.logger.info("Scanning for files to process...")
            self._find_gif_files()

        total_files = self.task_queue.qsize()

        if total_files == 0:
            self.logger.warning("No files in queue to process")
            return []

        self.logger.info(
            f"Starting sequential processing of {total_files} files")
        self.logger.info(
            "Processing one file at a time - convert, optimize, then move to next file")

        # Create a list of tasks from the queue
        tasks = []
        with self._file_queue_lock:
            while not self.task_queue.empty():
                try:
                    task = self.task_queue.get(block=False)
                    tasks.append(task)
                    self.task_queue.task_done()
                except Exception as e:
                    self.logger.error(f"Error retrieving tasks: {e}")
                    break

        # Sort tasks - process videos first
        tasks.sort(key=lambda x: not x.get('is_video', False))

        # Process files one by one with progress tracking
        with tqdm(total=len(tasks), desc="Processing Files", unit="file") as progress_bar:
            for task in tasks:
                if self._should_exit():
                    break

                file_path = task.get('input_path')
                output_path = task.get('output_path')
                is_video = task.get('is_video', False)
                final_path = task.get('final_path')

                try:
                    # Skip if already processed
                    if str(file_path) in self.processed_files:
                        self.logger.debug(
                            f"Skipping already processed file: {file_path}")
                        progress_bar.update(1)
                        self._stats['skipped'] += 1
                        continue

                    self.logger.info(f"Processing: {file_path}")

                    # STEP 1: For videos, convert to GIF
                    success = False
                    if is_video:
                        # First convert MP4 to temporary GIF
                        self.logger.info(
                            f"Converting video to GIF: {file_path} → {output_path}")
                        success = self.process_file(
                            file_path, output_path, True)

                        if not success:
                            self.logger.error(
                                f"Failed to convert video to GIF: {file_path}")
                            self._stats['failed'] += 1
                            self.failed_files.append(str(file_path))
                            progress_bar.update(1)
                            continue

                        # Verify the temporary GIF exists and has content
                        if not output_path.exists() or output_path.stat().st_size == 0:
                            self.logger.error(
                                f"Converted GIF is missing or empty: {output_path}")
                            self._stats['failed'] += 1
                            self.failed_files.append(str(file_path))
                            progress_bar.update(1)
                            continue

                        # Now optimize the temporary GIF to final location
                        self.logger.info(
                            f"Optimizing converted GIF: {output_path} → {final_path}")
                        target_size = self.compression_settings['min_size_mb']
                        try:
                            # Use our own optimize_gif method directly
                            final_size, optimization_success = self.optimize_gif(
                                output_path, final_path, target_size)

                            if optimization_success:
                                self.logger.success(
                                    f"Successfully optimized to {final_size:.2f}MB")
                                processed_files.append(final_path)
                                self._stats['processed'] += 1
                            else:
                                self.logger.warning(
                                    f"Optimization didn't meet target size, using direct conversion")
                                # Still use the file but log it as a less-than-ideal result
                                shutil.copy2(output_path, final_path)
                                processed_files.append(final_path)
                                self._stats['processed'] += 1
                        except Exception as e:
                            self.logger.error(
                                f"Error during GIF optimization: {e}")
                            # Try a direct copy as fallback
                            try:
                                shutil.copy2(output_path, final_path)
                                self.logger.warning(
                                    "Used direct copy due to optimization error")
                                processed_files.append(final_path)
                                self._stats['processed'] += 1
                            except Exception as copy_error:
                                self.logger.error(
                                    f"Failed to copy GIF: {copy_error}")
                                self._stats['failed'] += 1
                                self.failed_files.append(str(file_path))
                        finally:
                            # Clean up the temporary file regardless of outcome
                            try:
                                if output_path.exists():
                                    output_path.unlink(missing_ok=True)
                            except Exception as cleanup_error:
                                self.logger.warning(
                                    f"Failed to remove temp file: {cleanup_error}")

                    # STEP 2: For GIFs, just optimize them
                    else:
                        self.logger.info(
                            f"Optimizing GIF: {file_path} → {output_path}")
                        target_size = self.compression_settings['min_size_mb']
                        try:
                            final_size, success = self.optimize_gif(
                                file_path, output_path, target_size)

                            if success:
                                self.logger.success(
                                    f"Successfully optimized to {final_size:.2f}MB")
                                processed_files.append(output_path)
                                self._stats['processed'] += 1
                            else:
                                self.logger.error(
                                    f"Failed to optimize GIF to target size: {file_path}")
                                self._stats['failed'] += 1
                                self.failed_files.append(str(file_path))
                        except Exception as e:
                            self.logger.error(
                                f"Error optimizing GIF {file_path}: {e}")
                            self._stats['failed'] += 1
                            self.failed_files.append(str(file_path))

                except Exception as e:
                    self.logger.error(
                        f"Processing error for {file_path}: {str(e)}")
                    self._stats['failed'] += 1
                    self.failed_files.append(str(file_path))

                # Mark as processed and update progress
                self.processed_files.add(str(file_path))
                progress_bar.update(1)

                # Check memory and cleanup
                if hasattr(self, 'memory_manager'):
                    self.memory_manager.check_memory()

                # Small delay between files to ensure resources are freed
                gc.collect()
                time.sleep(0.2)

        # Log summary
        elapsed_time = time.time() - start_time
        self.logger.info(
            f"Sequential processing completed in {elapsed_time:.2f} seconds")
        self.logger.info(
            f"Processed: {self._stats['processed']}, Failed: {self._stats['failed']}, Skipped: {self._stats['skipped']}"
        )

        if self.failed_files:
            self.logger.warning(
                f"Failed to process {len(self.failed_files)} files:")
            # Only show first 10 for brevity
            for file in self.failed_files[:10]:
                self.logger.warning(f"  - {file}")
            if len(self.failed_files) > 10:
                self.logger.warning(
                    f"  ... and {len(self.failed_files) - 10} more")

        return processed_files

    def _process_with_timeout(self, file_path: Path, output_path: Path, is_video: bool):
        """Process a file with timeout monitoring and result tracking."""
        try:
            # Get final output path if it exists in the task data
            final_path = None
            if is_video:
                for item in list(self.task_queue.queue):
                    if isinstance(item, dict) and item.get('input_path') == file_path:
                        final_path = item.get('final_path')
                        break

            # First process to the output path (temp directory for videos)
            success = self.process_file(file_path, output_path, is_video)

            # For videos, after converting to temp GIF, optimize it to the final location
            if success and is_video and final_path:
                self.logger.info(
                    f"Optimizing converted GIF from {output_path} to {final_path}")
                # Use DynamicGIFOptimizer for the optimization step
                optimizer = DynamicGIFOptimizer()
                target_size = self.compression_settings['min_size_mb']
                _, optimization_success = optimizer.optimize_gif(
                    output_path, final_path, target_size)

                if not optimization_success:
                    self.logger.warning(
                        f"Failed to optimize converted GIF {output_path}, using direct conversion")
                    # Fall back to a copy if optimization fails
                    shutil.copy2(output_path, final_path)

                # Clean up the temporary file
                try:
                    if output_path.exists():
                        output_path.unlink()
                except Exception as e:
                    self.logger.warning(
                        f"Failed to remove temp file {output_path}: {e}")

            # Put result in the queue
            self.result_queue.put({
                'file_path': file_path,
                'success': success,
                'error': None if success else "Failed to process file"
            })
        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {str(e)}")

            # Put error result in the queue
            self.result_queue.put({
                'file_path': file_path,
                'success': False,
                'error': str(e)
            })

    def _immediate_shutdown_handler(self, signum, frame):
        """Handle immediate shutdown signal."""
        self.logger.warning(f"Immediate shutdown initiated (signal: {signum})")
        self._immediate_termination.set()

        # Force kill all processes
        self._kill_ffmpeg_processes()

        # Force exit
        sys.exit(1)

    def _find_gif_files(self, max_files_to_process=5):
        """Find GIF and MP4 files to process based on configuration.

        Args:
            max_files_to_process: Maximum number of files to queue at once
        """
        # Get directories from imported constants
        input_dir = Path(INPUT_DIR)
        output_dir = Path(OUTPUT_DIR)
        temp_dir = Path(TEMP_FILE_DIR)

        # Ensure directories exist
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        temp_dir.mkdir(exist_ok=True)

        # Clear the temp directory before starting
        for temp_file in temp_dir.glob('*'):
            try:
                if temp_file.is_file():
                    temp_file.unlink()
            except Exception as e:
                self.logger.warning(
                    f"Failed to remove temp file {temp_file}: {e}")

        # First find MP4 files in output directory to convert to GIFs
        mp4_files_to_convert = list(output_dir.glob('*.mp4'))
        self.logger.info(
            f"Found {len(mp4_files_to_convert)} MP4 files to convert to GIFs")

        # Then find GIF files in input directory to optimize
        gif_files_to_optimize = list(input_dir.glob('*.gif'))
        self.logger.info(
            f"Found {len(gif_files_to_optimize)} GIFs to optimize")

        # Only queue a limited number of files at once
        mp4_files_to_process = mp4_files_to_convert[:max_files_to_process]
        remaining_slots = max(0, max_files_to_process -
                              len(mp4_files_to_process))
        gif_files_to_process = gif_files_to_optimize[:remaining_slots]

        self.logger.info(
            f"Queueing {len(mp4_files_to_process)} MP4 files and {len(gif_files_to_process)} GIF files")

        # Reset queues and processed files
        with self._file_queue_lock:
            # Clear the queue
            while not self.task_queue.empty():
                try:
                    self.task_queue.get(block=False)
                    self.task_queue.task_done()
                except Exception:
                    pass

            # Clear the processed files set if it's a new processing run
            if not hasattr(self, '_partial_processing') or not self._partial_processing:
                self.processed_files.clear()
                self.failed_files.clear()
                self._stats = {'retried': 0, 'processed': 0,
                               'failed': 0, 'skipped': 0}
                self._partial_processing = True

            # First add MP4 files to convert
            for file_path in mp4_files_to_process:
                # Skip if already processed
                if str(file_path) in self.processed_files:
                    continue

                # Generate a temporary path for the converted GIF
                temp_gif_path = temp_dir / f"{file_path.stem}.gif"
                final_output_path = output_dir / f"{file_path.stem}.gif"

                # Add to task queue
                self.task_queue.put({
                    'input_path': file_path,
                    'output_path': temp_gif_path,  # First convert to temp
                    'final_path': final_output_path,  # Then optimize to final
                    'is_video': True
                })

            # Then add GIF files to optimize
            for file_path in gif_files_to_process:
                # Skip if already processed
                if str(file_path) in self.processed_files:
                    continue

                output_path = output_dir / f"{file_path.stem}_optimized.gif"

                # Add to task queue
                self.task_queue.put({
                    'input_path': file_path,
                    'output_path': output_path,
                    'is_video': False
                })

        # Return count of total files to process
        count = self.task_queue.qsize()

        # If no files were queued but there are still files to process,
        # we've processed a batch and there are more remaining
        if count == 0 and (len(mp4_files_to_convert) > len(mp4_files_to_process) or
                           len(gif_files_to_optimize) > len(gif_files_to_process)):
            self.logger.info(
                "Batch complete, more files remain to be processed")
            # Signal that there are more files to process
            return -1

        return count
