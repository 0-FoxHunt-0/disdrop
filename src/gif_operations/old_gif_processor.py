import asyncio
from asyncio.log import logger
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, contextmanager
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
import tempfile

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
            elapsed = end_time - start_time
            logger.debug(f"{func.__name__} took {elapsed:.2f}s")
            # Update performance stats if this is a method of a class with update_performance_stats
            if args and hasattr(args[0], 'update_performance_stats'):
                args[0].update_performance_stats(func.__name__, elapsed)

    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            return func(*args, **kwargs)
        finally:
            end_time = time.time()
            elapsed = end_time - start_time
            logger.debug(f"{func.__name__} took {elapsed:.2f}s")
            # Update performance stats if this is a method of a class with update_performance_stats
            if args and hasattr(args[0], 'update_performance_stats'):
                args[0].update_performance_stats(func.__name__, elapsed)

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

        # Initialize thread and worker management
        self._file_queue_lock = threading.Lock()
        self._file_locks_lock = threading.Lock()
        self._file_locks = {}
        self._worker_threads = []
        self._max_threads = 1  # Default to single-threaded operation
        self._stats_lock = threading.Lock()
        self._stats = {'processed': 0, 'failed': 0, 'skipped': 0, 'retried': 0}

        # Initialize file tracking
        self.processed_files = set()
        self.failed_files = []

        # Set up GPU and processing settings
        self.gpu_enabled = gpu_enabled
        self.gpu_settings = gpu_settings or {}

        # Set up performance monitoring
        self.performance_stats = {}
        self.logging_lock = threading.Lock()

        # Load configuration
        self.compression_settings = {
            'fps_range': GIF_COMPRESSION['fps_range'],
            'colors': GIF_COMPRESSION['colors'],
            'lossy_value': GIF_COMPRESSION['lossy_value'],
            'min_size_mb': GIF_COMPRESSION['min_size_mb'],
            'min_width': GIF_COMPRESSION['min_width'],
            'min_height': GIF_COMPRESSION['min_height'],
            'quality': GIF_COMPRESSION['quality']
        }

        # Initialize components
        self._init_worker_management()
        self._init_caches()
        self._init_components()
        self._init_async_support()
        self._setup_signal_handlers()

        # Log GPU status
        if self.gpu_enabled:
            self.logger.info(
                f"GPU acceleration enabled with settings: {self.gpu_settings}")
        else:
            self.logger.info("GPU acceleration disabled")

        # Register cleanup handlers
        atexit.register(self.cleanup_resources)

        self.logger.debug("GIF Processor initialization complete")

    def _init_worker_management(self):
        """Initialize worker management components."""
        # Set up thread and task management
        self.worker_threads = []
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()

        # Set to sequential processing (one file at a time)
        self._max_workers = 1
        self._partial_processing = False
        self._processing_in_progress = False

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
        if self.gpu_enabled and self.gpu_settings:
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
        """Get or create a file lock for the given path.

        This returns a context manager for use with 'with' statements.

        Args:
            file_path: The path to get a lock for

        Returns:
            A context manager that acquires and releases the lock
        """
        from contextlib import contextmanager

        # Get or create lock
        with self._file_locks_lock:
            if file_path not in self._file_locks:
                self._file_locks[file_path] = threading.Lock()
            lock = self._file_locks[file_path]

        @contextmanager
        def lock_context():
            try:
                # Try to acquire lock with timeout
                if not lock.acquire(timeout=5):
                    self.logger.error(
                        f"Failed to acquire lock for {file_path}")
                    raise TimeoutError(
                        f"Failed to acquire lock for {file_path}")
                yield lock
            finally:
                if lock.locked():
                    lock.release()

        return lock_context()

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
    @performance_monitor
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

            # Determine if we're in ultra-aggressive mode based on file size ratio
            size_ratio = original_size / target_size_mb
            ultra_aggressive = size_ratio > 7.0
            very_aggressive = size_ratio > 5.0

            if ultra_aggressive:
                self.logger.warning(
                    f"Using ultra-aggressive conversion: size ratio {size_ratio:.1f}x exceeds 7.0x threshold")
            elif very_aggressive:
                self.logger.warning(
                    f"Using very aggressive conversion: size ratio {size_ratio:.1f}x exceeds 5.0x threshold")

            # Adjust FPS based on file size and aggressiveness
            adjusted_fps = fps
            if ultra_aggressive:
                adjusted_fps = min(fps, 10)
            elif very_aggressive:
                adjusted_fps = min(fps, 12)
            elif original_size > 200:
                adjusted_fps = min(fps, 15)
            elif original_size > 100:
                adjusted_fps = min(fps, 20)
            elif original_size > 50:
                adjusted_fps = min(fps, 24)

            # Calculate dimensions with adaptive scaling
            max_dimension = max(dimensions[0], dimensions[1])
            scale_factor = 1.0

            # Scale based on file size, dimensions, and aggressiveness
            if ultra_aggressive:
                scale_factor = min(scale_factor, 0.3)
            elif very_aggressive:
                scale_factor = min(scale_factor, 0.4)
            elif original_size > 200:
                scale_factor = min(scale_factor, 0.5)
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

            # Ensure minimum reasonable dimensions
            new_width = max(new_width, 320)
            new_height = max(new_height, 180)

            self.logger.info(
                f"Input dimensions: {dimensions[0]}x{dimensions[1]}, output: {new_width}x{new_height}, scale: {scale_factor:.2f}")
            self.logger.info(f"Input FPS: {fps}, adjusted FPS: {adjusted_fps}")

            # Start with a reasonable default for color count
            color_count = 256

            # Reduce colors based on file size ratio and dimensions
            if ultra_aggressive:
                color_count = 64
            elif very_aggressive:
                color_count = 96
            elif original_size > 100:
                color_count = 128

            # For palette generation, use the selected color count
            palette_max_colors = color_count

            self.logger.info(
                f"Using color count: {color_count}, palette max colors: {palette_max_colors}")

            # Generate temp file paths with UUIDs to avoid FFmpeg image sequence pattern warnings
            unique_id = str(uuid.uuid4())
            temp_palette = Path(TEMP_FILE_DIR) / f"palette_{unique_id}.png"
            temp_output = Path(TEMP_FILE_DIR) / f"temp_{unique_id}.gif"

            # Create an optimized palette
            palette_cmd = [
                'ffmpeg', '-i', str(file_path),
                '-vf', f'fps={adjusted_fps},scale={new_width}:{new_height}:flags=lanczos,palettegen=max_colors={palette_max_colors}:stats_mode=diff',
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
                self.logger.info(
                    f"Initial GIF size: {initial_size:.2f}MB (target: {target_size_mb:.2f}MB)")

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

                # Check size ratio before using gifsicle
                size_ratio = initial_size / target_size_mb
                skip_gifsicle = size_ratio > 7.0

                if skip_gifsicle:
                    self.logger.warning(
                        f"Initial GIF too large ({size_ratio:.1f}x target size), skipping gifsicle and using FFmpeg directly")
                    # Use FFmpeg compression instead of gifsicle
                    new_width = int(dimensions[0] * 0.5)
                    new_height = int(dimensions[1] * 0.5)
                    ffmpeg_cmd = [
                        'ffmpeg', '-i', str(temp_output),
                        '-vf', f'scale={new_width}:{new_height}:flags=lanczos,fps=15',
                        '-c:v', 'gif',
                        '-f', 'gif',
                        '-y', str(output_path)
                    ]
                    success = run_ffmpeg_command(ffmpeg_cmd)
                    if success:
                        self.logger.info(f"Used FFmpeg to compress large GIF")
                        final_size = self.get_file_size(output_path)
                        self.logger.info(f"Final size: {final_size:.2f}MB")
                        return True
                    else:
                        self.logger.error("FFmpeg compression failed")
                        return False

                # Analyze the temp GIF to determine its actual color count
                actual_colors = color_count  # Default
                try:
                    analyze_cmd = [
                        'gifsicle',
                        '--info',
                        str(temp_output)
                    ]

                    result = subprocess.run(
                        analyze_cmd, capture_output=True, text=True)
                    color_info = result.stdout

                    # Try to extract color count from output
                    color_match = re.search(r'(\d+) colors', color_info)
                    if color_match:
                        detected_colors = int(color_match.group(1))
                        self.logger.info(
                            f"Detected {detected_colors} colors in generated GIF")
                        # Use the detected colors, never more
                        actual_colors = detected_colors
                except Exception as e:
                    self.logger.warning(f"Could not analyze GIF colors: {e}")

                self.logger.info(
                    f"Using {actual_colors} colors for gifsicle optimization")

                gifsicle_cmd = [
                    'gifsicle',
                    '--optimize=3',
                    '--colors', str(actual_colors),
                    '--lossy=30',
                    '--no-conserve-memory',
                    str(temp_output),
                    '-o', str(output_path)
                ]

                # Run with output capture to check for warnings
                gifsicle_process = subprocess.run(
                    gifsicle_cmd, capture_output=True, text=True)
                gifsicle_stderr = gifsicle_process.stderr if hasattr(
                    gifsicle_process, 'stderr') else ""

                # Check for the "huge GIF" or "trivial palette" warnings
                if "huge GIF" in gifsicle_stderr:
                    self.logger.warning(
                        "Detected 'huge GIF' warning from gifsicle, falling back to FFmpeg")
                    # Fall back to FFmpeg
                    ffmpeg_cmd = [
                        'ffmpeg', '-i', str(temp_output),
                        '-vf', f'scale={new_width}:{new_height}:flags=lanczos,fps=15',
                        '-c:v', 'gif',
                        '-f', 'gif',
                        '-y', str(output_path)
                    ]
                    if not run_ffmpeg_command(ffmpeg_cmd):
                        shutil.copy2(temp_output, output_path)
                        self.logger.warning(
                            "FFmpeg fallback failed, using unoptimized GIF")
                elif gifsicle_process.returncode != 0:
                    shutil.copy2(temp_output, output_path)
                    self.logger.warning(
                        f"Gifsicle optimization failed, using unoptimized GIF. Error: {gifsicle_stderr}")
                elif "trivial adaptive palette" in gifsicle_stderr:
                    # If we're seeing a palette warning, extract the actual color count and retry
                    color_match = re.search(
                        r'trivial adaptive palette \(only (\d+) colors', gifsicle_stderr)
                    if color_match:
                        detected_colors = int(color_match.group(1))
                        self.logger.info(
                            f"Detected trivial palette - re-running with {detected_colors} colors")

                        # Re-run with corrected color count
                        updated_cmd = [
                            'gifsicle',
                            '--optimize=3',
                            '--colors', str(detected_colors),
                            '--lossy=30',
                            '--no-conserve-memory',
                            str(temp_output),
                            '-o', str(output_path)
                        ]
                        updated_process = subprocess.run(
                            updated_cmd, capture_output=True, text=True)
                        if updated_process.returncode != 0:
                            self.logger.warning(
                                f"Retry with correct color count failed: {updated_process.stderr}")
                            shutil.copy2(temp_output, output_path)

                # Check final size
                final_size = self.get_file_size(output_path)
                if final_size > target_size_mb:
                    self.logger.info(
                        f"Further optimization needed: {final_size:.2f}MB > {target_size_mb:.2f}MB")
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

    @performance_monitor
    def process_file(self, file_path: Path, output_path: Path, is_video: bool, convert_to_mp4: bool = False) -> bool:
        """Process a single file, ensuring it reaches target size or maximum attempts are exhausted.

        Args:
            file_path: Path to the input file
            output_path: Path to save the output file
            is_video: Whether the input is a video file
            convert_to_mp4: Whether to convert to MP4 format

        Returns:
            bool: True if processing was successful, False otherwise
        """
        try:
            target_size_mb = self.compression_settings['min_size_mb']
            self.logger.info(
                f"Processing {file_path.name} (target size: {target_size_mb:.2f}MB)")

            # For video files - handle MP4 conversion if needed
            if is_video:
                # Check if MP4 conversion is required
                if convert_to_mp4:
                    mp4_path = output_path.with_suffix('.mp4')
                    self.logger.info(
                        f"Converting video to MP4: {mp4_path.name}")

                    # Convert to MP4
                    success = self.convert_to_mp4(file_path, mp4_path)
                    if not success:
                        self.logger.error(
                            f"Failed to convert video to MP4: {file_path.name}")
                        return False

                    self.logger.success(
                        f"Successfully converted video to MP4: {mp4_path.name}")
                    return True

                # If we're here, we need to create a GIF from the video
                # Determine the output GIF path
                gif_path = output_path.with_suffix('.gif')
                self.logger.info(f"Converting video to GIF: {gif_path.name}")

                # Get video file properties
                video_info = self.get_video_info(file_path)
                if not video_info:
                    self.logger.error(
                        f"Could not get video info for {file_path.name}")
                    return False

                fps = video_info.get('fps', 15)
                dimensions = video_info.get('dimensions', (640, 480))

                # Record the best result information
                best_result = {
                    'size': float('inf'),
                    'path': None,
                    'success': False
                }

                # Try up to 3 attempts with increasingly aggressive settings
                MAX_ATTEMPTS = 3
                for attempt in range(1, MAX_ATTEMPTS + 1):
                    # Create temp file for this attempt
                    temp_output = Path(TEMP_FILE_DIR) / \
                        f"temp_gif_{uuid.uuid4()}.gif"

                    # Adjust FPS and dimensions based on attempt number
                    # Reduce FPS with each attempt
                    adjusted_fps = max(10, fps - (attempt - 1) * 5)
                    # Reduce scale with each attempt
                    adjusted_scale = max(0.3, 1.0 - (attempt - 1) * 0.2)

                    self.logger.info(
                        f"Attempt {attempt}/{MAX_ATTEMPTS}: Creating GIF with FPS {adjusted_fps}, scale {adjusted_scale:.2f}")

                    # Create the GIF with adjusted settings
                    success = self.create_gif(
                        file_path,
                        temp_output,
                        fps=adjusted_fps,
                        dimensions=(
                            int(dimensions[0] * adjusted_scale), int(dimensions[1] * adjusted_scale))
                    )

                    if not success or not temp_output.exists():
                        self.logger.warning(
                            f"Attempt {attempt} failed to create GIF")
                        continue

                    # Check the size of the created GIF
                    result_size = self.get_file_size(temp_output)
                    self.logger.info(
                        f"Attempt {attempt} result size: {result_size:.2f}MB (target: {target_size_mb:.2f}MB)")

                    # If this result is better than what we have, keep track of it
                    if result_size < best_result['size']:
                        best_result['size'] = result_size
                        best_result['path'] = temp_output
                        best_result['success'] = True

                    # If we've met the target size, we can stop
                    if result_size <= target_size_mb:
                        self.logger.success(
                            f"Target size met on attempt {attempt}: {result_size:.2f}MB")
                        # Copy the result to the output path
                        shutil.copy2(temp_output, gif_path)
                        return True

                # If we get here, we've exhausted all attempts
                # Use the best result we found, if any
                if best_result['success'] and best_result['path']:
                    self.logger.warning(
                        f"Failed to meet target size after {MAX_ATTEMPTS} attempts. "
                        f"Using best result: {best_result['size']:.2f}MB (target: {target_size_mb:.2f}MB)")
                    shutil.copy2(best_result['path'], gif_path)
                    return True
                else:
                    self.logger.error(
                        f"Failed to create GIF after {MAX_ATTEMPTS} attempts")
                    return False

            # If input is already a GIF, optimize it
            if file_path.suffix.lower() == '.gif':
                # Get the output GIF path
                gif_path = output_path.with_suffix('.gif')
                self.logger.info(f"Optimizing GIF: {gif_path.name}")

                # Try multiple optimization attempts until we meet the target size
                MAX_ATTEMPTS = 3
                best_result = {
                    'size': float('inf'),
                    'path': None,
                    'success': False
                }

                for attempt in range(1, MAX_ATTEMPTS + 1):
                    # Create temp file for this attempt
                    temp_output = Path(TEMP_FILE_DIR) / \
                        f"temp_opt_{uuid.uuid4()}.gif"

                    self.logger.info(
                        f"Attempt {attempt}/{MAX_ATTEMPTS}: Optimizing GIF")

                    # Try to optimize the GIF
                    final_size, success = self.optimize_gif(
                        file_path,
                        temp_output,
                        target_size_mb,
                        attempt_number=attempt  # Pass the attempt number
                    )

                    if not success or not temp_output.exists():
                        self.logger.warning(
                            f"Attempt {attempt} failed to optimize GIF")
                        continue

                    self.logger.info(
                        f"Attempt {attempt} result size: {final_size:.2f}MB (target: {target_size_mb:.2f}MB)")

                    # If this result is better than what we have, keep track of it
                    if final_size < best_result['size']:
                        best_result['size'] = final_size
                        best_result['path'] = temp_output
                        best_result['success'] = True

                    # If we've met the target size, we can stop
                    if final_size <= target_size_mb:
                        self.logger.success(
                            f"Target size met on attempt {attempt}: {final_size:.2f}MB")
                        # Copy the result to the output path
                        shutil.copy2(temp_output, gif_path)
                        return True

                # If we get here, we've exhausted all attempts
                # Use the best result we found, if any
                if best_result['success'] and best_result['path']:
                    self.logger.warning(
                        f"Failed to meet target size after {MAX_ATTEMPTS} attempts. "
                        f"Using best result: {best_result['size']:.2f}MB (target: {target_size_mb:.2f}MB)")
                    shutil.copy2(best_result['path'], gif_path)
                    return True
                else:
                    self.logger.error(
                        f"Failed to optimize GIF after {MAX_ATTEMPTS} attempts")
                    return False

            # Handle other file types - just copy them
            else:
                self.logger.info(
                    f"Copying file: {file_path.name} to {output_path.name}")
                shutil.copy2(file_path, output_path)
                return True

        except Exception as e:
            self.logger.error(
                f"Error processing file {file_path.name}: {str(e)}")
            traceback.print_exc()
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
        """Process all files in the queue with sequential processing - one at a time.
        Will continue processing each file until target size is reached or max attempts are exhausted.

        Workflow:
        1. Check MP4 files from input dir against output dir and use optimized versions if available
        2. Check GIF files from input dir against output dir and skip if already optimized
        3. Process GIFs in input dir that need optimization (one by one)
        4. Convert MP4s to GIFs and optimize them (one by one)

        Returns:
            List[Path]: List of processed file paths
        """
        start_time = time.time()
        processed_files = []

        # Set a flag to indicate processing is in progress
        self._processing_in_progress = True

        try:
            # First, find all files to process if queue is empty
            if self.task_queue.qsize() == 0:
                self.logger.info("Scanning for files to process...")
                self._find_gif_files()

            total_files = self.task_queue.qsize()

            if total_files == 0:
                self.logger.info("No files in queue to process")
                return []

            self.logger.info(
                f"Starting sequential processing of {total_files} files")
            self.logger.info(
                "Processing one file at a time - process in priority order")

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

            # Sort tasks by priority:
            # 1. Non-MP4 videos to convert to MP4
            # 2. GIFs to optimize
            # 3. MP4s to convert to GIFs
            def task_priority(task):
                if task.get('convert_to_mp4', False):
                    return 1  # Highest priority
                elif not task.get('is_video', False):
                    return 2  # GIFs to optimize
                else:
                    return 3  # MP4s to convert to GIFs

            tasks.sort(key=task_priority)

            target_size_mb = self.compression_settings['min_size_mb']
            max_attempts = 3  # Maximum number of optimization attempts

            # Create progress bar for tracking
            with tqdm(total=len(tasks), desc="Processing files", unit="file") as progress_bar:
                for task in tasks:
                    file_path = task['input_path']
                    output_path = task['output_path']
                    is_video = task.get('is_video', False)
                    convert_to_mp4 = task.get('convert_to_mp4', False)
                    final_path = task.get('final_path', output_path)

                    # Skip if already processed
                    if str(file_path) in self.processed_files:
                        self.logger.info(
                            f"Skipping already processed file: {file_path}")
                        progress_bar.update(1)
                        continue

                    self.logger.info(f"Processing file: {file_path}")

                    try:
                        # SCENARIO 1: Convert non-MP4 video to MP4
                        if convert_to_mp4:
                            self.logger.info(
                                f"Converting video to MP4: {file_path} → {output_path}")
                            success = self.convert_to_mp4(
                                file_path, output_path)

                            if not success:
                                self.logger.error(
                                    f"Failed to convert video to MP4: {file_path}")
                                self._stats['failed'] += 1
                                self.failed_files.append(str(file_path))
                            else:
                                self.logger.success(
                                    f"Successfully converted video to MP4: {output_path}")
                                processed_files.append(output_path)
                                self._stats['processed'] += 1

                            progress_bar.update(1)
                            # Mark as processed so we don't try again
                            self.processed_files.add(str(file_path))
                            continue

                        # SCENARIO 2: Optimize GIF file from input directory
                        elif not is_video:
                            attempt = 0
                            success = False
                            final_size = float('inf')

                            # Continue trying until target size is met or max attempts reached
                            while attempt < max_attempts and final_size > target_size_mb:
                                self.logger.info(
                                    f"Optimizing GIF (attempt {attempt+1}/{max_attempts}): {file_path}")

                                # Create temporary file for this attempt
                                temp_output = Path(
                                    TEMP_FILE_DIR) / f"opt_attempt_{attempt+1}_{uuid.uuid4()}.gif"

                                # Optimize GIF
                                final_size, success = self.optimize_gif(
                                    file_path, temp_output, target_size_mb, attempt_number=attempt)

                                if not success:
                                    self.logger.error(
                                        f"Failed to optimize GIF (attempt {attempt+1})")
                                    attempt += 1
                                    continue

                                self.logger.info(
                                    f"Optimization attempt {attempt+1} result: {final_size:.2f}MB (target: {target_size_mb:.2f}MB)")

                                # If we've reached target size or this is the final attempt, copy to final output
                                if final_size <= target_size_mb or attempt == max_attempts - 1:
                                    shutil.copy2(temp_output, output_path)
                                    if final_size <= target_size_mb:
                                        self.logger.success(
                                            f"Successfully optimized GIF: {final_size:.2f}MB")
                                    else:
                                        self.logger.warning(
                                            f"Best result after {attempt+1} attempts: {final_size:.2f}MB (target: {target_size_mb:.2f}MB)")
                                    break

                                attempt += 1

                            # After all attempts, check if we succeeded
                            if success:
                                processed_files.append(output_path)
                                self._stats['processed'] += 1
                            else:
                                self.logger.error(
                                    f"Failed to optimize GIF after {max_attempts} attempts")
                                self._stats['failed'] += 1
                                self.failed_files.append(str(file_path))

                            progress_bar.update(1)
                            # Mark as processed so we don't try again
                            self.processed_files.add(str(file_path))
                            continue

                        # SCENARIO 3: For MP4 videos, convert to GIF and optimize
                        elif is_video:
                            attempt = 0
                            success = False
                            final_success = False
                            final_size = float('inf')

                            # Continue trying until target size is met or max attempts reached
                            while attempt < max_attempts and final_size > target_size_mb:
                                # First convert MP4 to temporary GIF
                                self.logger.info(
                                    f"Converting video to GIF (attempt {attempt+1}/{max_attempts}): {file_path} → {output_path}")

                                # Adjust settings based on attempt number
                                # Get video file properties
                                video_info = self.get_video_info(file_path)
                                if not video_info:
                                    self.logger.error(
                                        f"Could not get video info for {file_path}")
                                    break

                                fps = video_info.get('fps', 15)
                                dimensions = video_info.get(
                                    'dimensions', (640, 480))

                                # Adjust settings based on attempt number
                                adjusted_fps = max(10, fps - (attempt * 5))
                                adjusted_scale = max(
                                    0.3, 1.0 - (attempt * 0.2))

                                # Create the GIF with adjusted settings
                                success = self.create_gif(
                                    file_path,
                                    output_path,
                                    fps=adjusted_fps,
                                    dimensions=(
                                        int(dimensions[0] * adjusted_scale),
                                        int(dimensions[1] * adjusted_scale))
                                )

                                if not success:
                                    self.logger.error(
                                        f"Failed to convert video to GIF: {file_path}")
                                    attempt += 1
                                    continue

                                # Verify the temporary GIF exists and has content
                                if not output_path.exists() or output_path.stat().st_size == 0:
                                    self.logger.error(
                                        f"Converted GIF is missing or empty: {output_path}")
                                    attempt += 1
                                    continue

                                # Check if the temporary GIF is too large to optimize
                                temp_gif_size = self.get_file_size(output_path)
                                size_ratio = temp_gif_size / target_size_mb

                                # Explicitly log the size comparison
                                self.logger.info(
                                    f"Temp GIF size: {temp_gif_size:.2f}MB, Target size: {target_size_mb:.2f}MB, Ratio: {size_ratio:.2f}x")

                                # Skip gifsicle if size ratio exceeds threshold (7.0)
                                if size_ratio > 7.0:
                                    self.logger.warning(
                                        f"Temp GIF too large for gifsicle optimization (ratio: {size_ratio:.2f}x), "
                                        f"using FFmpeg for aggressive compression")

                                    # Use FFmpeg for more aggressive compression
                                    final_size, final_success = self._compress_large_with_ffmpeg(
                                        output_path, final_path, target_size_mb)
                                else:
                                    # Optimize the temporary GIF to reach target size
                                    final_size, final_success = self.optimize_gif(
                                        output_path, final_path, target_size_mb, attempt_number=attempt)

                                self.logger.info(
                                    f"Conversion+optimization attempt {attempt+1} result: {final_size:.2f}MB (target: {target_size_mb:.2f}MB)")

                                # If size target met, we can stop
                                if final_size <= target_size_mb:
                                    self.logger.success(
                                        f"Successfully created and optimized GIF: {final_size:.2f}MB")
                                    break

                                attempt += 1

                            # After all attempts, check if we succeeded
                            if success and final_success:
                                processed_files.append(final_path)
                                self._stats['processed'] += 1
                            elif success:
                                self.logger.warning(
                                    f"Best result after {max_attempts} attempts: {final_size:.2f}MB (target: {target_size_mb:.2f}MB)")
                                # Use the best result we have even if it's not below target size
                                processed_files.append(final_path)
                                self._stats['processed'] += 1
                            else:
                                self.logger.error(
                                    f"Failed to convert/optimize video to GIF after {max_attempts} attempts")
                                self._stats['failed'] += 1
                                self.failed_files.append(str(file_path))

                            # Clean up the temporary file
                            try:
                                if output_path.exists() and output_path != final_path:
                                    output_path.unlink(missing_ok=True)
                            except Exception as cleanup_error:
                                self.logger.warning(
                                    f"Failed to remove temp file: {cleanup_error}")

                            progress_bar.update(1)
                            # Mark as processed so we don't try again
                            self.processed_files.add(str(file_path))
                            continue

                    except Exception as e:
                        self.logger.error(
                            f"Unexpected error processing {file_path}: {e}")
                        self._stats['failed'] += 1
                        self.failed_files.append(str(file_path))
                        progress_bar.update(1)
                        self.processed_files.add(str(file_path))

            # Check if we need to process more files
            # Keep looking for more files until there aren't any left
            remaining_files = self._find_gif_files()
            if remaining_files == -1 or self.task_queue.qsize() > 0:
                # Signal that there are more files to process
                self.logger.info("More files remain, continuing processing...")
                # Recursively process remaining files
                more_processed = self.process_all()
                processed_files.extend(more_processed)

            # Log processing stats
            elapsed_time = time.time() - start_time
            if elapsed_time > 0 and processed_files:
                fps = len(processed_files) / elapsed_time
                self.logger.info(
                    f"Processed {len(processed_files)} files in {elapsed_time:.1f}s ({fps:.2f} files/s)")
            else:
                self.logger.info(f"Processed {len(processed_files)} files")

            self.logger.info(f"Processing stats: {self._stats}")

            # Log performance statistics
            self.log_performance_stats()

            return processed_files

        finally:
            # Reset flag when processing is complete
            self._processing_in_progress = False

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

    def _find_gif_files(self, max_files_to_process=1):
        """Find GIF and MP4 files to process based on configuration.

        Args:
            max_files_to_process: Maximum number of files to queue at once (default is 1 for one-by-one processing)

        Returns:
            int: -1 if files were queued, 0 if no more files to process
        """
        # Get directories from imported constants
        input_dir = Path(INPUT_DIR)
        output_dir = Path(OUTPUT_DIR)
        temp_dir = Path(TEMP_FILE_DIR)
        target_size_mb = self.compression_settings['min_size_mb']

        # Ensure directories exist
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)
        temp_dir.mkdir(exist_ok=True)

        # Clean temp directory
        for temp_file in temp_dir.glob('*'):
            try:
                if temp_file.is_file():
                    temp_file.unlink()
            except Exception as e:
                self.logger.warning(
                    f"Failed to remove temp file {temp_file}: {e}")

        # Initialize tracking lists
        mp4_files_to_convert = []
        non_mp4_videos_to_convert = []
        gif_files_to_optimize = []

        # 1. Check all MP4 files from input dir against output dir
        mp4_files_in_input = list(input_dir.glob('*.mp4'))
        self.logger.info(
            f"Found {len(mp4_files_in_input)} MP4 files in input directory")

        for mp4_file in mp4_files_in_input:
            # Check if optimized GIF exists in output dir and is below max size
            expected_gif_output = output_dir / f"{mp4_file.stem}.gif"

            # Use optimized MP4 from output dir for future operations if it exists
            expected_mp4_output = output_dir / f"{mp4_file.stem}.mp4"

            if expected_gif_output.exists():
                gif_size = self.get_file_size(expected_gif_output)
                if gif_size <= target_size_mb:
                    self.logger.info(
                        f"Skipping {mp4_file.name} - optimized GIF already exists at {expected_gif_output.name} with size {gif_size:.2f}MB")
                    # Add to processed files to avoid re-processing
                    self.processed_files.add(str(mp4_file))
                    continue
                else:
                    self.logger.warning(
                        f"Found existing GIF for {mp4_file.name} but size {gif_size:.2f}MB exceeds target {target_size_mb}MB - will reprocess")

            # Add to conversion list, using output MP4 if available
            if expected_mp4_output.exists():
                self.logger.info(
                    f"Will use optimized MP4 from output dir: {expected_mp4_output.name}")
                mp4_files_to_convert.append(expected_mp4_output)
            else:
                mp4_files_to_convert.append(mp4_file)

        # Find non-MP4 video files in input directory that need to be converted to MP4 first
        for ext in SUPPORTED_VIDEO_FORMATS:
            if ext == '.mp4':
                continue  # Already handled MP4 files

            video_files = list(input_dir.glob(f'*{ext}'))
            self.logger.info(
                f"Found {len(video_files)} {ext} files in input directory")

            for video_file in video_files:
                # Check if MP4 version already exists in output dir
                expected_mp4_output = output_dir / f"{video_file.stem}.mp4"
                expected_gif_output = output_dir / f"{video_file.stem}.gif"

                # If optimized GIF already exists and is below max size, skip this file
                if expected_gif_output.exists():
                    gif_size = self.get_file_size(expected_gif_output)
                    if gif_size <= target_size_mb:
                        self.logger.info(
                            f"Skipping {video_file.name} - optimized GIF already exists at {expected_gif_output.name} with size {gif_size:.2f}MB")
                        # Add to processed files to avoid re-processing
                        self.processed_files.add(str(video_file))
                        continue

                # If MP4 version doesn't exist in output, add to non-MP4 conversion list
                if not expected_mp4_output.exists():
                    non_mp4_videos_to_convert.append(video_file)
                else:
                    # Use the existing MP4 in output dir for GIF conversion
                    self.logger.info(
                        f"Will use existing MP4 from output dir: {expected_mp4_output.name}")
                    mp4_files_to_convert.append(expected_mp4_output)

        # 2. Check all GIF files from input dir against output dir
        gif_files_in_input = list(input_dir.glob('*.gif'))
        self.logger.info(
            f"Found {len(gif_files_in_input)} GIF files in input directory")

        for gif_file in gif_files_in_input:
            # Check if optimized version already exists in output dir with the same filename
            expected_output = output_dir / f"{gif_file.stem}.gif"
            if expected_output.exists():
                output_size = self.get_file_size(expected_output)
                if output_size <= target_size_mb:
                    self.logger.info(
                        f"Skipping {gif_file.name} - optimized version exists at {expected_output.name} with size {output_size:.2f}MB")
                    # Add to processed files to avoid re-processing
                    self.processed_files.add(str(gif_file))
                    continue
                else:
                    self.logger.warning(
                        f"Found existing optimized GIF for {gif_file.name} but size {output_size:.2f}MB exceeds target {target_size_mb}MB - will reprocess")

            # Add to optimization list
            gif_files_to_optimize.append(gif_file)

        # Log findings
        self.logger.info(
            f"Found {len(non_mp4_videos_to_convert)} non-MP4 videos to convert to MP4")
        self.logger.info(
            f"Found {len(mp4_files_to_convert)} MP4 files to convert to GIFs")
        self.logger.info(
            f"Found {len(gif_files_to_optimize)} GIFs to optimize")

        # Reset queues and processed files if needed
        with self._file_queue_lock:
            # Clear the queue if it's not being processed already
            if not hasattr(self, '_processing_in_progress') or not self._processing_in_progress:
                while not self.task_queue.empty():
                    try:
                        self.task_queue.get(block=False)
                        self.task_queue.task_done()
                    except Exception:
                        pass

            # Clear the processed files set if it's a new processing run
            if not hasattr(self, '_partial_processing') or not self._partial_processing:
                self.processed_files = set() if not hasattr(
                    self, 'processed_files') else self.processed_files
                self.failed_files = [] if not hasattr(
                    self, 'failed_files') else self.failed_files
                self._stats = {'retried': 0, 'processed': 0,
                               'failed': 0, 'skipped': 0}
                self._partial_processing = True

            # Process priority:
            # 1. Non-MP4 videos to convert to MP4
            # 2. GIFs to optimize
            # 3. MP4s to convert to GIFs
            files_to_process = []

            # First prioritize non-MP4 videos to convert to MP4
            if non_mp4_videos_to_convert:
                files_to_process = non_mp4_videos_to_convert[:max_files_to_process]

                for file_path in files_to_process:
                    # Skip if already processed
                    if str(file_path) in self.processed_files:
                        continue

                    # Generate output path for the MP4
                    mp4_output_path = output_dir / f"{file_path.stem}.mp4"

                    # Add to task queue
                    self.task_queue.put({
                        'input_path': file_path,
                        'output_path': mp4_output_path,
                        'is_video': True,
                        'convert_to_mp4': True
                    })
            # Then prioritize GIFs to optimize if no non-MP4 videos
            elif gif_files_to_optimize:
                files_to_process = gif_files_to_optimize[:max_files_to_process]

                for file_path in files_to_process:
                    # Skip if already processed
                    if str(file_path) in self.processed_files:
                        continue

                    # Ensure the output has the same filename as the input
                    output_path = output_dir / f"{file_path.stem}.gif"

                    # Add to task queue
                    self.task_queue.put({
                        'input_path': file_path,
                        'output_path': output_path,
                        'is_video': False
                    })
            # Finally process MP4s if no GIFs to optimize
            elif mp4_files_to_convert:
                files_to_process = mp4_files_to_convert[:max_files_to_process]

                for file_path in files_to_process:
                    # Skip if already processed
                    if str(file_path) in self.processed_files:
                        continue

                    # Generate a temporary path for the converted GIF
                    temp_gif_path = temp_dir / f"{file_path.stem}.gif"

                    # Ensure the final output has the same filename as the input
                    final_output_path = output_dir / f"{file_path.stem}.gif"

                    # Add to task queue
                    self.task_queue.put({
                        'input_path': file_path,
                        'output_path': temp_gif_path,  # First convert to temp
                        'final_path': final_output_path,  # Then optimize to final
                        'is_video': True
                    })

        # Return -1 if we've queued any files; 0 if no more files left to process
        return -1 if self.task_queue.qsize() > 0 else 0

    def log_performance_stats(self):
        """Log detailed performance statistics."""
        try:
            # Get current resource usage
            cpu_usage = self.resource_monitor.get_cpu_usage()
            memory_usage = self.resource_monitor.get_memory_usage()
            gpu_usage = self.resource_monitor.get_gpu_usage() if self.gpu_enabled else None

            # Log current resource usage
            self.logger.info(f"Current resource usage:")
            self.logger.info(f"  ├ CPU: {cpu_usage:.1f}%")
            self.logger.info(
                f"  ├ Memory: {memory_usage['percent']:.1f}% ({memory_usage['used']:.1f}GB)")

            if gpu_usage and self.gpu_enabled:
                self.logger.info(
                    f"  ├ GPU: {gpu_usage['utilization']:.1f}% ({gpu_usage['memory_used']:.1f}GB)")

            # Log performance stats for monitored functions
            if hasattr(self, 'performance_stats') and self.performance_stats:
                self.logger.info("Performance statistics:")
                for func_name, stats in self.performance_stats.items():
                    avg_time = stats['total_time'] / max(1, stats['calls'])
                    self.logger.info(
                        f"  ├ {func_name}: {stats['calls']} calls, avg {avg_time:.3f}s, total {stats['total_time']:.3f}s")

            # Clear stats after logging
            self.performance_stats = {}

        except Exception as e:
            self.logger.error(f"Error logging performance stats: {e}")

    def update_performance_stats(self, func_name, elapsed_time):
        """Update performance statistics for a function."""
        if not hasattr(self, 'performance_stats'):
            self.performance_stats = {}

        with self._stats_lock:
            if func_name not in self.performance_stats:
                self.performance_stats[func_name] = {
                    'calls': 0, 'total_time': 0.0}

            self.performance_stats[func_name]['calls'] += 1
            self.performance_stats[func_name]['total_time'] += elapsed_time

    @log_function_call
    @performance_monitor
    def optimize_gif(self, input_path: Path, output_path: Path, target_size_mb: float, attempt_number: int = 0) -> Tuple[float, bool]:
        """Optimize a GIF file using appropriate methods based on file size.

        Args:
            input_path: Path to the input GIF
            output_path: Path to save the optimized GIF
            target_size_mb: Target size in MB
            attempt_number: Current attempt number (0-based) for progressive optimization

        Returns:
            Tuple[float, bool]: (final_size_mb, success)
        """
        try:
            # Get input file size
            input_size_mb = self.get_file_size(input_path)

            # If already small enough, just copy
            if input_size_mb <= target_size_mb:
                self.logger.info(
                    f"Input already meets size target: {input_size_mb:.2f}MB ≤ {target_size_mb:.2f}MB")
                shutil.copy2(input_path, output_path)
                return input_size_mb, True

            # Calculate size ratio
            size_ratio = input_size_mb / target_size_mb
            self.logger.info(
                f"Input size: {input_size_mb:.2f}MB, target: {target_size_mb:.2f}MB, ratio: {size_ratio:.1f}x")

            # Determine if we should skip gifsicle based on size ratio
            # If current file size is above 700% of max size, skip gifsicle optimization
            skip_gifsicle = size_ratio > 7.0

            if skip_gifsicle:
                self.logger.warning(
                    f"File too large for gifsicle ({size_ratio:.1f}x target), using FFmpeg compression directly")
                return self._compress_large_with_ffmpeg(input_path, output_path, target_size_mb)

            # Get dimensions for scaling decisions
            dimensions = self._get_gif_dimensions(input_path)
            if dimensions is None:
                self.logger.error(
                    f"Could not determine dimensions for {input_path}")
                return input_size_mb, False

            width, height = dimensions

            # Make optimization more aggressive with each attempt
            colors = 256
            lossy = 30
            scale = 1.0

            # Adjust settings based on attempt number
            if attempt_number >= 2:  # Third attempt or later
                colors = 64
                lossy = 100
                scale = 0.5
            elif attempt_number >= 1:  # Second attempt
                colors = 128
                lossy = 80
                scale = 0.7

            # Further adjust based on size ratio
            if size_ratio > 10.0:
                colors = min(colors, 32)
                lossy = max(lossy, 150)
                scale = min(scale, 0.4)
            elif size_ratio > 5.0:
                colors = min(colors, 64)
                lossy = max(lossy, 100)
                scale = min(scale, 0.6)

            # Log optimization settings
            self.logger.info(
                f"Optimization settings (attempt {attempt_number+1}): colors={colors}, lossy={lossy}, scale={scale:.2f}")

            # Try gifsicle
            gifsicle_success = self._compress_with_gifsicle(
                input_path, output_path, target_size_mb, colors, lossy, scale)

            # Check if gifsicle was successful and output exists
            if gifsicle_success and output_path.exists():
                result_size = self.get_file_size(output_path)

                # If result is good enough, return it
                if result_size <= target_size_mb:
                    self.logger.success(
                        f"Gifsicle optimization successful: {result_size:.2f}MB")
                    return result_size, True

                # If result is still too large but better than input, keep it
                if result_size < input_size_mb:
                    self.logger.warning(
                        f"Gifsicle result still above target: {result_size:.2f}MB > {target_size_mb:.2f}MB, "
                        f"but better than input ({input_size_mb:.2f}MB)")
                    return result_size, True

                # If result is worse than input, try FFmpeg
                self.logger.warning(
                    f"Gifsicle result ({result_size:.2f}MB) is worse than input ({input_size_mb:.2f}MB), "
                    f"trying FFmpeg compression")
            else:
                self.logger.warning(
                    f"Gifsicle optimization failed, trying FFmpeg compression")

            # Fall back to FFmpeg for more aggressive compression
            return self._compress_large_with_ffmpeg(input_path, output_path, target_size_mb)

        except Exception as e:
            self.logger.error(f"Error optimizing GIF: {str(e)}")
            traceback.print_exc()
            # Try to copy the input as a fallback
            try:
                shutil.copy2(input_path, output_path)
                return self.get_file_size(output_path), False
            except Exception as copy_e:
                self.logger.error(
                    f"Failed to copy input as fallback: {copy_e}")
                return float('inf'), False

    @log_function_call
    @performance_monitor
    def _compress_large_with_ffmpeg(self, input_path: Path, output_path: Path, target_size_mb: float) -> Tuple[float, bool]:
        """Compress large GIF files using FFmpeg with aggressive settings.

        This method is used as a fallback when gifsicle is not suitable for very large files.

        Args:
            input_path: Path to the input GIF
            output_path: Path to save the compressed GIF
            target_size_mb: Target size in MB

        Returns:
            Tuple[float, bool]: (final_size_mb, success)
        """
        try:
            # Get input file size and calculate ratio
            input_size_mb = self.get_file_size(input_path)
            size_ratio = input_size_mb / target_size_mb

            self.logger.info(
                f"FFmpeg compression for large GIF: {input_path.name}, "
                f"size ratio: {size_ratio:.1f}x, "
                f"size: {input_size_mb:.2f}MB, target: {target_size_mb:.2f}MB")

            # Get dimensions
            dimensions = self._get_gif_dimensions(input_path)
            if dimensions is None:
                self.logger.error(
                    f"Could not determine dimensions for {input_path}")
                return input_size_mb, False

            width, height = dimensions
            self.logger.info(f"Original dimensions: {width}x{height}")

            # Try multiple compression attempts with increasing aggressiveness
            max_attempts = 3
            best_result = {
                'size': float('inf'),
                'path': None,
                'success': False
            }

            for attempt in range(max_attempts):
                # Create temp file for this attempt
                temp_output = Path(TEMP_FILE_DIR) / \
                    f"ffmpeg_attempt_{attempt+1}_{uuid.uuid4()}.gif"

                # Adjust scale factor based on size ratio and attempt number
                scale_factor = 0.7
                fps = 15

                # First attempt - moderate compression
                if attempt == 0:
                    if size_ratio > 20.0:
                        scale_factor = 0.5
                        fps = 12
                    elif size_ratio > 10.0:
                        scale_factor = 0.6
                        fps = 14
                    elif size_ratio > 5.0:
                        scale_factor = 0.7
                        fps = 15
                # Second attempt - aggressive compression
                elif attempt == 1:
                    if size_ratio > 20.0:
                        scale_factor = 0.4
                        fps = 10
                    elif size_ratio > 10.0:
                        scale_factor = 0.5
                        fps = 12
                    elif size_ratio > 5.0:
                        scale_factor = 0.6
                        fps = 12
                # Third attempt - ultra-aggressive compression
                else:
                    if size_ratio > 20.0:
                        scale_factor = 0.3
                        fps = 8
                    elif size_ratio > 10.0:
                        scale_factor = 0.4
                        fps = 10
                    elif size_ratio > 5.0:
                        scale_factor = 0.5
                        fps = 10
                    else:
                        scale_factor = 0.5
                        fps = 12

                # Calculate new dimensions
                new_width = max(int(width * scale_factor), 320)
                new_height = max(int(height * scale_factor), 180)

                # Ensure dimensions are even
                new_width = new_width // 2 * 2
                new_height = new_height // 2 * 2

                self.logger.info(
                    f"Attempt {attempt+1}: dimensions {new_width}x{new_height} (scale: {scale_factor:.2f}), fps: {fps}")

                # Generate palette for better quality with smaller size
                palette_path = Path(TEMP_FILE_DIR) / \
                    f"palette_{uuid.uuid4()}.png"

                # Create a palette with limited colors
                palette_colors = 256
                if size_ratio > 20.0:
                    palette_colors = 64
                elif size_ratio > 10.0:
                    palette_colors = 128
                elif size_ratio > 5.0:
                    palette_colors = 192

                # Create palette
                palette_cmd = [
                    'ffmpeg',
                    '-i', str(input_path),
                    '-vf', f'fps={fps},scale={new_width}:{new_height}:flags=lanczos,palettegen=max_colors={palette_colors}:stats_mode=diff',
                    '-y', str(palette_path)
                ]

                palette_success = run_ffmpeg_command(palette_cmd)

                if not palette_success:
                    self.logger.warning(
                        f"Palette generation failed on attempt {attempt+1}, trying direct conversion")
                    # Fallback to direct conversion
                    cmd = [
                        'ffmpeg',
                        '-i', str(input_path),
                        '-vf', f'scale={new_width}:{new_height}:flags=lanczos,fps={fps}',
                        '-c:v', 'gif',
                        '-f', 'gif',
                        '-y', str(temp_output)
                    ]
                else:
                    # Use paletteuse with dithering options
                    dither_method = "bayer"
                    bayer_scale = 5

                    if size_ratio > 15.0:
                        bayer_scale = 3  # More efficient compression

                    cmd = [
                        'ffmpeg',
                        '-i', str(input_path),
                        '-i', str(palette_path),
                        '-lavfi', f'fps={fps},scale={new_width}:{new_height}:flags=lanczos[x];[x][1:v]paletteuse=dither={dither_method}:bayer_scale={bayer_scale}:diff_mode=rectangle',
                        '-y', str(temp_output)
                    ]

                # Run FFmpeg command
                success = run_ffmpeg_command(cmd)

                # Clean up palette
                if palette_path.exists():
                    try:
                        palette_path.unlink()
                    except Exception:
                        pass

                if success and temp_output.exists():
                    result_size = self.get_file_size(temp_output)
                    self.logger.info(
                        f"Attempt {attempt+1} result: {result_size:.2f}MB (target: {target_size_mb:.2f}MB)")

                    # If this is better than our previous best, update it
                    if result_size < best_result['size']:
                        # Remove previous best if it exists
                        if best_result['path'] and best_result['path'].exists():
                            try:
                                best_result['path'].unlink()
                            except Exception:
                                pass

                        best_result['size'] = result_size
                        best_result['path'] = temp_output
                        best_result['success'] = True

                    # If we've met the target size, we can stop
                    if result_size <= target_size_mb:
                        self.logger.success(
                            f"Target size met on attempt {attempt+1}: {result_size:.2f}MB")
                        shutil.copy2(temp_output, output_path)
                        return result_size, True
                else:
                    self.logger.warning(
                        f"FFmpeg compression failed on attempt {attempt+1}")

            # After all attempts, use the best result
            if best_result['success'] and best_result['path']:
                self.logger.warning(
                    f"Failed to meet target size after {max_attempts} attempts. "
                    f"Using best result: {best_result['size']:.2f}MB (target: {target_size_mb:.2f}MB)")
                shutil.copy2(best_result['path'], output_path)
                return best_result['size'], True

            # If all attempts failed, fall back to the original file
            self.logger.error(
                f"All FFmpeg compression attempts failed for {input_path.name}")
            shutil.copy2(input_path, output_path)
            return input_size_mb, False

        except Exception as e:
            self.logger.error(f"Error in FFmpeg compression: {str(e)}")
            traceback.print_exc()
            try:
                # Try to copy the input as a fallback
                shutil.copy2(input_path, output_path)
                return self.get_file_size(output_path), False
            except Exception:
                return float('inf'), False

    @log_function_call
    @performance_monitor
    def _compress_with_gifsicle(self, input_path: Path, output_path: Path, target_size_mb: float, colors: int = 256, lossy: int = 30, scale: float = 1.0) -> bool:
        """Compress a GIF using gifsicle with specific parameters.

        Args:
            input_path: Path to the input GIF
            output_path: Path to save the optimized GIF
            target_size_mb: Target size in MB
            colors: Number of colors in the output (default: 256)
            lossy: Lossy compression level (default: 30)
            scale: Scale factor for dimensions (default: 1.0)

        Returns:
            bool: True if compression was successful, False otherwise
        """
        try:
            # If scale is less than 1.0, resize the GIF first
            if scale < 1.0:
                dimensions = self._get_gif_dimensions(input_path)
                if dimensions is None:
                    self.logger.error(
                        f"Could not get dimensions for {input_path}")
                    return False

                width, height = dimensions
                new_width = max(int(width * scale), 320)
                new_height = max(int(height * scale), 180)

                # Ensure dimensions are even
                new_width = new_width // 2 * 2
                new_height = new_height // 2 * 2

                # Create a temporary file for the resized GIF
                temp_resized = Path(TEMP_FILE_DIR) / \
                    f"resized_{uuid.uuid4()}.gif"

                # Use FFmpeg to resize
                resize_cmd = [
                    'ffmpeg',
                    '-i', str(input_path),
                    '-vf', f'scale={new_width}:{new_height}:flags=lanczos',
                    '-y', str(temp_resized)
                ]

                if not run_ffmpeg_command(resize_cmd):
                    self.logger.error(
                        f"Failed to resize GIF for gifsicle optimization")
                    return False

                # Use the resized file as input for gifsicle
                gifsicle_input = temp_resized
            else:
                gifsicle_input = input_path

            # Run gifsicle for optimization
            gifsicle_cmd = [
                'gifsicle',
                '--optimize=3',
                '--colors', str(colors),
                '--lossy=' + str(lossy),
                '--no-conserve-memory',
                str(gifsicle_input),
                '-o', str(output_path)
            ]

            # Run with output capture to check for warnings
            gifsicle_process = subprocess.run(
                gifsicle_cmd, capture_output=True, text=True)
            gifsicle_stderr = gifsicle_process.stderr if hasattr(
                gifsicle_process, 'stderr') else ""

            # Cleanup temp file if created
            if scale < 1.0 and temp_resized.exists():
                try:
                    temp_resized.unlink(missing_ok=True)
                except Exception as e:
                    self.logger.warning(
                        f"Failed to remove temp file {temp_resized}: {e}")

            # Check for common warnings or errors
            if "out of memory" in gifsicle_stderr:
                self.logger.error(
                    f"Gifsicle out of memory error: {gifsicle_stderr}")
                return False

            if "huge GIF" in gifsicle_stderr:
                self.logger.warning(
                    f"Gifsicle huge GIF warning: {gifsicle_stderr}")
                # Continue as it may still work

            if gifsicle_process.returncode != 0:
                self.logger.error(
                    f"Gifsicle failed with code {gifsicle_process.returncode}: {gifsicle_stderr}")
                return False

            # Check if output file exists and has content
            if not output_path.exists() or output_path.stat().st_size == 0:
                self.logger.error("Gifsicle produced no output")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Error in gifsicle compression: {str(e)}")
            traceback.print_exc()
            return False

    def _get_gif_dimensions(self, file_path: Path) -> Optional[Tuple[int, int]]:
        """Get dimensions of a GIF file.

        Args:
            file_path: Path to the GIF file

        Returns:
            Optional tuple of (width, height) or None if dimensions couldn't be determined
        """
        try:
            # Check if we have cached dimensions
            cache_key = str(file_path)
            if cache_key in self.dimension_cache:
                return self.dimension_cache[cache_key]

            # Use FFprobe to get dimensions
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'csv=s=x:p=0',
                str(file_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.warning(
                    f"FFprobe failed for {file_path}: {result.stderr}")
                return None

            # Parse the output (format: width x height)
            dimensions_str = result.stdout.strip()
            if not dimensions_str:
                self.logger.warning(
                    f"No dimension output from FFprobe for {file_path}")
                return None

            try:
                width_str, height_str = dimensions_str.split('x')
                width = int(width_str)
                height = int(height_str)

                # Cache the result
                self.dimension_cache[cache_key] = (width, height)

                return (width, height)
            except ValueError:
                self.logger.warning(
                    f"Invalid dimension format from FFprobe: {dimensions_str}")
                return None

        except Exception as e:
            self.logger.error(f"Error getting GIF dimensions: {str(e)}")
            return None

    def _get_dimensions_with_retry(self, file_path: Path, max_retries: int = 3) -> Optional[Tuple[int, int]]:
        """Get dimensions of a file with retry mechanism.

        Args:
            file_path: Path to the file
            max_retries: Maximum number of retries

        Returns:
            Optional tuple of (width, height) or None if dimensions couldn't be determined
        """
        for attempt in range(max_retries):
            try:
                dimensions = self._get_gif_dimensions(file_path)
                if dimensions is not None:
                    return dimensions

                self.logger.warning(
                    f"Attempt {attempt+1}/{max_retries}: Failed to get dimensions for {file_path.name}")
                time.sleep(0.5)  # Wait before retrying
            except Exception as e:
                self.logger.warning(
                    f"Attempt {attempt+1}/{max_retries}: Error getting dimensions: {str(e)}")
                time.sleep(0.5)  # Wait before retrying

        # All attempts failed
        self.logger.error(
            f"Failed to get dimensions for {file_path.name} after {max_retries} attempts")
        return None

    def _validate_dimensions(self, dimensions: Optional[Tuple[int, int]]) -> bool:
        """Validate dimensions to ensure they are reasonable.

        We import this function from utils.video_dimensions but also implement it here
        as a fallback in case the import fails.

        Args:
            dimensions: Tuple of (width, height) or None

        Returns:
            bool: True if dimensions are valid, False otherwise
        """
        if dimensions is None:
            return False

        width, height = dimensions

        # Check minimum dimensions
        if width < 32 or height < 32:
            self.logger.warning(f"Dimensions too small: {width}x{height}")
            return False

        # Check maximum dimensions (to prevent processing extremely large files)
        if width > 7680 or height > 4320:  # 8K resolution limit
            self.logger.warning(f"Dimensions too large: {width}x{height}")
            return False

        # Check aspect ratio (prevent extremely skewed videos)
        aspect_ratio = max(width, height) / max(1, min(width, height))
        if aspect_ratio > 10:  # arbitrary limit
            self.logger.warning(
                f"Extreme aspect ratio: {aspect_ratio:.1f} ({width}x{height})")
            return False

        return True

    def get_file_size(self, file_path: Path) -> float:
        """Get file size in megabytes.

        Args:
            file_path: Path to the file

        Returns:
            float: File size in megabytes
        """
        try:
            return file_path.stat().st_size / (1024 * 1024)  # Convert bytes to MB
        except Exception as e:
            self.logger.error(f"Error getting file size: {str(e)}")
            return 0.0

    def get_video_info(self, file_path: Path) -> Dict[str, Any]:
        """Get video information including dimensions, duration, and FPS.

        Args:
            file_path: Path to the video file

        Returns:
            Dictionary containing video information
        """
        try:
            # Default values in case of failure
            info = {
                'dimensions': (640, 480),
                'duration': 0.0,
                'fps': 15,
                'bitrate': 0,
                'codec': 'unknown'
            }

            # Use FFprobe to get video information
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,r_frame_rate,duration,bit_rate,codec_name',
                '-of', 'json',
                str(file_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.warning(
                    f"FFprobe failed for {file_path}: {result.stderr}")
                return info

            try:
                data = json.loads(result.stdout)
                if 'streams' not in data or not data['streams']:
                    self.logger.warning(
                        f"No video streams found in {file_path}")
                    return info

                stream = data['streams'][0]

                # Get dimensions
                if 'width' in stream and 'height' in stream:
                    info['dimensions'] = (
                        int(stream['width']), int(stream['height']))

                # Get FPS
                if 'r_frame_rate' in stream:
                    try:
                        # Parse fraction like "30000/1001"
                        num, den = stream['r_frame_rate'].split('/')
                        info['fps'] = round(float(num) / float(den))
                    except (ValueError, ZeroDivisionError):
                        self.logger.warning(
                            f"Invalid FPS value: {stream['r_frame_rate']}")

                # Get duration
                if 'duration' in stream:
                    info['duration'] = float(stream['duration'])

                # Get bitrate
                if 'bit_rate' in stream:
                    info['bitrate'] = int(stream['bit_rate'])

                # Get codec
                if 'codec_name' in stream:
                    info['codec'] = stream['codec_name']

                return info

            except json.JSONDecodeError:
                self.logger.warning(
                    f"Invalid JSON from FFprobe for {file_path}")
                return info

        except Exception as e:
            self.logger.error(f"Error getting video info: {str(e)}")
            return {
                'dimensions': (640, 480),
                'duration': 0.0,
                'fps': 15,
                'bitrate': 0,
                'codec': 'unknown'
            }

    @log_function_call
    @performance_monitor
    def convert_to_mp4(self, input_path: Path, output_path: Path) -> bool:
        """Convert a video file to MP4 format.

        Args:
            input_path: Path to the input video file
            output_path: Path to save the output MP4 file

        Returns:
            bool: True if conversion was successful, False otherwise
        """
        try:
            # Get video info to determine optimal conversion settings
            video_info = self.get_video_info(input_path)

            # Use dimensions from input, or default if not available
            width, height = video_info.get('dimensions', (640, 480))

            # Limit maximum dimensions
            max_dimension = 1920
            if width > max_dimension or height > max_dimension:
                # Calculate scale factor to reduce dimensions
                scale = max_dimension / max(width, height)
                width = int(width * scale)
                height = int(height * scale)
                # Ensure even dimensions for video encoding
                width = width - (width % 2)
                height = height - (height % 2)

            # Build FFmpeg command with appropriate settings
            cmd = [
                'ffmpeg',
                '-i', str(input_path),
                '-c:v', 'libx264',  # Use H.264 codec for video
                '-preset', 'medium',  # Balance between speed and quality
                # Constant Rate Factor (0-51, lower means better quality)
                '-crf', '23',
                '-vf', f'scale={width}:{height}',  # Scale video
                '-pix_fmt', 'yuv420p',  # Standard pixel format for compatibility
                '-movflags', '+faststart',  # Optimize for web streaming
                '-y', str(output_path)
            ]

            # Add audio settings if the input has audio
            if 'audio' in input_path.name.lower() or self._check_has_audio(input_path):
                cmd.insert(5, '-c:a')
                cmd.insert(6, 'aac')  # Use AAC codec for audio
                cmd.insert(7, '-b:a')
                cmd.insert(8, '128k')  # Reasonable audio bitrate
            else:
                cmd.insert(5, '-an')  # No audio

            # Run the command
            success = run_ffmpeg_command(cmd)

            if not success:
                self.logger.error(f"FFmpeg conversion failed for {input_path}")
                return False

            # Verify the output file exists and has content
            if not output_path.exists() or output_path.stat().st_size == 0:
                self.logger.error(
                    f"Output file is missing or empty: {output_path}")
                return False

            self.logger.success(f"Successfully converted {input_path} to MP4")
            return True

        except Exception as e:
            self.logger.error(f"Error converting to MP4: {str(e)}")
            traceback.print_exc()
            return False

    def _check_has_audio(self, file_path: Path) -> bool:
        """Check if a video file has an audio stream.

        Args:
            file_path: Path to the video file

        Returns:
            bool: True if the file has audio, False otherwise
        """
        try:
            # Use FFprobe to check for audio streams
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'a:0',  # Select first audio stream
                '-show_entries', 'stream=codec_type',
                '-of', 'csv=p=0',
                str(file_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            # If there's output and it contains "audio", the file has audio
            return result.returncode == 0 and 'audio' in result.stdout.strip()

        except Exception as e:
            self.logger.warning(f"Error checking for audio: {str(e)}")
            return False

    @log_function_call
    @performance_monitor
    def _create_gif_aggressive(self, file_path: Path, output_path: Path, fps: int, dimensions: Tuple[int, int]) -> bool:
        """Create GIF with very aggressive optimization for large files.

        This method is used when the initial GIF conversion is too large.

        Args:
            file_path: Path to the input video file
            output_path: Path to save the optimized GIF
            fps: Target frames per second
            dimensions: Original dimensions as (width, height)

        Returns:
            bool: True if creation was successful, False otherwise
        """
        try:
            # Calculate aggressive scale factor based on original dimensions
            max_dimension = max(dimensions[0], dimensions[1])

            # Start with very aggressive scaling
            if max_dimension > 1920:
                scale_factor = 0.25  # 75% reduction for very large videos
            elif max_dimension > 1280:
                scale_factor = 0.35  # 65% reduction for large videos
            elif max_dimension > 720:
                scale_factor = 0.5   # 50% reduction for medium videos
            else:
                scale_factor = 0.6   # 40% reduction for small videos

            # Calculate new dimensions
            new_width = int(dimensions[0] * scale_factor)
            new_height = int(dimensions[1] * scale_factor)

            # Ensure even dimensions for video encoding
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)

            # Limit FPS for large files - more aggressive than standard
            target_fps = min(fps, 10)  # Cap at 10 fps for large files

            # Reduce colors and dithering for aggressive compression
            palette_colors = 64  # Use fewer colors for aggressive compression
            dither_mode = "bayer"  # Use bayer dithering for better compression

            # Set a very restrictive quality with high compression ratio
            ffmpeg_quality = 25  # Higher value means more compression

            # Temporary file for palette
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as palette_file:
                palette_path = palette_file.name

            try:
                # Step 1: Generate a heavily optimized color palette with reduced colors
                palette_cmd = [
                    'ffmpeg', '-v', 'error',
                    '-i', str(file_path),
                    '-vf', f'fps={target_fps},scale={new_width}:{new_height}:flags=lanczos,palettegen=max_colors={palette_colors}:stats_mode=diff',
                    '-y', palette_path
                ]

                self.logger.debug(
                    f"Running palette command: {' '.join(palette_cmd)}")

                # Generate palette file
                result = subprocess.run(
                    palette_cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    self.logger.error(
                        f"Failed to generate palette: {result.stderr}")
                    return False

                # Step 2: Create heavily optimized GIF with aggressive compression
                output_cmd = [
                    'ffmpeg', '-v', 'error',
                    '-i', str(file_path),
                    '-i', palette_path,
                    '-lavfi', f'fps={target_fps},scale={new_width}:{new_height}:flags=lanczos[x];[x][1:v]paletteuse=dither={dither_mode}:diff_mode=rectangle:new=1',
                    '-q:v', str(ffmpeg_quality),
                    '-y', str(output_path)
                ]

                self.logger.debug(
                    f"Running output command: {' '.join(output_cmd)}")

                # Create GIF file
                result = subprocess.run(
                    output_cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    self.logger.error(f"Failed to create GIF: {result.stderr}")
                    return False

                # Check if file exists and verify size
                if output_path.exists() and output_path.stat().st_size > 0:
                    output_size_mb = round(
                        output_path.stat().st_size / (1024 * 1024), 2)
                    self.logger.info(
                        f"Created GIF with aggressive settings: {output_size_mb} MB")
                    return True
                else:
                    self.logger.error("Output GIF is missing or empty")
                    return False

            finally:
                # Clean up temporary palette file
                if os.path.exists(palette_path):
                    os.unlink(palette_path)

        except Exception as e:
            self.logger.error(f"Error in _create_gif_aggressive: {str(e)}")
            traceback.print_exc()
            return False
