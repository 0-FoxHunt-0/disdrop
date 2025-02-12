import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from enum import Enum
from functools import wraps  # Add this import
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Generator
import psutil
import traceback
import json
from cachetools import TTLCache
import cv2  # Add this import at the top with other imports

# Fix the imports to use absolute paths since this is imported from main.py
from src.default_config import (GIF_COMPRESSION, GIF_PASS_OVERS, GIF_SIZE_TO_SKIP,
                                INPUT_DIR, OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS,
                                TEMP_FILE_DIR)
from src.logging_system import log_function_call, run_ffmpeg_command
from src.temp_file_manager import TempFileManager
from src.video_optimization import VideoProcessor
from src.utils.error_handler import VideoProcessingError
from src.utils.video_dimensions import get_video_dimensions


class ProcessingStatus(Enum):
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


class FileProcessor:
    """Base class for file processing operations."""

    def __init__(self):
        self.file_size_cache = TTLCache(maxsize=1000, ttl=3600)

    @staticmethod
    def wait_for_file_completion(file_path: Union[str, Path], timeout: int = 30) -> bool:
        """Wait for file to be completely written and accessible."""
        file_path = Path(file_path)
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Try to open the file in read mode
                with open(file_path, 'rb') as f:
                    # Try to seek to end to ensure complete file access
                    f.seek(0, 2)
                # Force sync to filesystem
                if hasattr(os, 'sync'):
                    os.sync()
                # Get initial size
                initial_size = file_path.stat().st_size
                # Wait a small interval
                time.sleep(0.1)
                # Check if size is stable
                if initial_size == file_path.stat().st_size:
                    return True
            except (IOError, OSError):
                time.sleep(0.1)
                continue
        return False

    def get_file_size(self, file_path: Union[str, Path], force_refresh: bool = True) -> float:
        """Get file size in MB with improved reliability."""
        try:
            file_path = Path(file_path)

            if force_refresh:
                # Clear any cached size
                self.file_size_cache.pop(str(file_path), None)

                # Wait for file to be completely written
                if not self.wait_for_file_completion(file_path):
                    logging.warning(
                        f"File may not be completely written: {file_path}")

            # Get fresh size
            size = file_path.stat().st_size / (1024 * 1024)
            # Update cache
            self.file_size_cache[str(file_path)] = size
            return size
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return float('inf')
        except Exception as e:
            logging.error(f"Error getting file size: {e}")
            return float('inf')

    @staticmethod
    def ensure_directory(directory: Path) -> None:
        """Ensure directory exists."""
        directory.mkdir(parents=True, exist_ok=True)


class FFmpegHandler:
    """Handles FFmpeg-related operations."""

    def __init__(self):
        self._current_process = None
        self._process_lock = threading.Lock()
        self._current_processes = set()  # Track all active processes

    def _kill_current_process(self):
        """Safely kill the current FFmpeg process."""
        with self._process_lock:
            for process in self._current_processes:
                try:
                    if sys.platform == 'win32':
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    else:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except Exception as e:
                    logging.error(f"Error killing process {process.pid}: {e}")
            self._current_processes.clear()

    @staticmethod
    def create_optimized_gif(file_path: Path, output_path: Path,
                             fps: int, dimensions: Tuple[int, int], settings: Dict) -> bool:
        """Create optimized GIF in a single pass."""
        try:
            cmd = [
                'ffmpeg',
                '-i', str(file_path),
                '-vf',
                f'fps={fps},'
                f'scale={dimensions[0]}:{dimensions[1]}:flags=lanczos,'
                'split[s0][s1];'
                '[s0]palettegen=max_colors={colors}:stats_mode=diff[p];'
                '[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle'.format(
                    colors=settings.get('colors', 256)
                ),
                '-y',
                str(output_path)
            ]

            # 2 minute timeout for FFmpeg
            return run_ffmpeg_command(cmd, timeout=120)

        except Exception as e:
            logging.error(f"GIF creation failed: {str(e)}")
            return False


class DynamicGIFOptimizer:
    """Handles dynamic GIF optimization with adaptive settings."""

    def __init__(self):
        self.dev_logger = logging.getLogger('developer')
        self.base_settings = {
            'colors': 256,
            'lossy_value': 15,
            'scale_factor': 1.0
        }
        self.optimization_history = {}

    def optimize_gif(self, input_path: Path, output_path: Path, target_size_mb: float = 15.0) -> Tuple[float, bool]:
        """
        Dynamically optimize GIF using adaptive settings based on results.

        Args:
            input_path: Path to input GIF
            output_path: Path to output optimized GIF
            target_size_mb: Target size in MB (default 15MB)

        Returns:
            Tuple[float, bool]: (final_size, success)
        """
        input_size = self.get_file_size(input_path)
        self.dev_logger.info(
            f"Starting optimization: {input_size:.2f}MB → Target: {target_size_mb}MB")

        current_settings = self.base_settings.copy()
        best_result = {'size': float('inf'), 'settings': None}
        attempt = 0
        max_attempts = 10

        size_ratio = target_size_mb / input_size
        current_settings['scale_factor'] = self._calculate_initial_scale_factor(
            input_size, size_ratio)

        self.dev_logger.info(
            f"Dynamic initial scale factor: {current_settings['scale_factor']:.3f} (based on {input_size:.1f}MB input)")

        while attempt < max_attempts:
            attempt += 1
            self.dev_logger.info(f"\nAttempt {attempt} with settings:")
            self.dev_logger.info(
                f"Scale: {current_settings['scale_factor']:.3f}")
            self.dev_logger.info(f"Colors: {current_settings['colors']}")
            self.dev_logger.info(f"Lossy: {current_settings['lossy_value']}")

            result_size = self._apply_optimization(
                input_path, output_path, current_settings)

            if result_size <= target_size_mb:
                self.dev_logger.info(f"Target achieved: {result_size:.2f}MB")
                return result_size, True

            if result_size < best_result['size']:
                best_result = {'size': result_size,
                               'settings': current_settings.copy()}

            new_settings = self._adjust_settings(
                current_settings, input_size, result_size, target_size_mb, attempt)

            if self._settings_similar(current_settings, new_settings):
                new_settings = self._get_aggressive_settings(
                    current_settings, attempt)

            current_settings = new_settings

            reduction = ((input_size - result_size) / input_size) * 100
            self.dev_logger.info(
                f"Result: {result_size:.2f}MB ({reduction:.1f}% reduction)")

        if best_result['settings']:
            self.dev_logger.info("Using best settings found...")
            final_size = self._apply_optimization(
                input_path, output_path, best_result['settings'])
            return final_size, final_size <= target_size_mb

        return input_size, False

    def _calculate_initial_scale_factor(self, input_size: float, size_ratio: float) -> float:
        """Calculate initial scale factor based on input size and size ratio."""
        if input_size > 200:
            return min(1.0, max(0.1, (size_ratio ** 0.75)))
        elif input_size > 100:
            return min(1.0, max(0.2, (size_ratio ** 0.65)))
        elif input_size > 50:
            return min(1.0, max(0.3, (size_ratio ** 0.5)))
        else:
            return min(1.0, max(0.4, (size_ratio ** 0.4)))

    def _apply_optimization(self, input_path: Path, output_path: Path, settings: Dict) -> float:
        """Apply optimization with given settings."""
        try:
            cmd = [
                'gifsicle',
                '--optimize=2',
                '--colors', str(settings['colors']),
                '--lossy=' + str(settings['lossy_value']),
                '--scale', str(settings['scale_factor']),
                '--no-conserve-memory',
                '--careful',
                '--threads=4'
            ]

            if settings['scale_factor'] < 1.0:
                from video_optimization import VideoProcessor
                width, height = VideoProcessor()._get_dimensions(input_path)
                if width and height:
                    new_width = int(width * settings['scale_factor'])
                    new_height = int(height * settings['scale_factor'])
                    cmd.extend(['--resize', f'{new_width}x{new_height}'])

            cmd.extend(['--batch', str(input_path), '-o', str(output_path)])

            if run_ffmpeg_command(cmd):
                return self.get_file_size(output_path)
            return float('inf')

        except Exception as e:
            self.dev_logger.error(f"Optimization error: {str(e)}")
            return float('inf')

    def _adjust_settings(self, current: Dict, input_size: float, result_size: float, target_size: float, attempt: int) -> Dict:
        """Dynamically adjust settings based on results with more gradual changes."""
        new_settings = current.copy()
        size_ratio = result_size / target_size

        if size_ratio > 2:
            new_settings['scale_factor'] *= 0.9
        elif size_ratio > 1.5:
            new_settings['scale_factor'] *= 0.95
        elif size_ratio > 1.2:
            new_settings['scale_factor'] *= 0.98

        if size_ratio > 2 and new_settings['colors'] > 128:
            new_settings['colors'] = max(128, new_settings['colors'] - 32)
        elif size_ratio > 1.5 and new_settings['colors'] > 192:
            new_settings['colors'] = max(192, new_settings['colors'] - 16)
        elif size_ratio > 1.2 and new_settings['colors'] > 224:
            new_settings['colors'] = max(224, new_settings['colors'] - 8)

        if size_ratio > 1.2:
            increase = min(10, (size_ratio - 1) * 15)
            new_settings['lossy_value'] = min(
                100, new_settings['lossy_value'] + increase)

        new_settings['scale_factor'] = max(0.5, new_settings['scale_factor'])
        new_settings['colors'] = max(128, new_settings['colors'])
        new_settings['lossy_value'] = min(100, new_settings['lossy_value'])

        return new_settings

    def _get_aggressive_settings(self, current: Dict, attempt: int) -> Dict:
        """Get more aggressive settings when normal adjustments aren't enough."""
        aggressive = current.copy()

        if attempt <= 3:
            aggressive['scale_factor'] *= 0.9
            aggressive['colors'] = max(192, aggressive['colors'] - 32)
            aggressive['lossy_value'] = min(60, aggressive['lossy_value'] + 15)
        elif attempt <= 5:
            aggressive['scale_factor'] *= 0.85
            aggressive['colors'] = max(156, aggressive['colors'] - 24)
            aggressive['lossy_value'] = min(80, aggressive['lossy_value'] + 10)
        else:
            aggressive['scale_factor'] *= 0.8
            aggressive['colors'] = max(128, aggressive['colors'] - 16)
            aggressive['lossy_value'] = min(100, aggressive['lossy_value'] + 5)

        return aggressive

    @staticmethod
    def _settings_similar(settings1: Dict, settings2: Dict, threshold: float = 0.1) -> bool:
        """Check if two settings are very similar."""
        return (abs(settings1['scale_factor'] - settings2['scale_factor']) < threshold and
                abs(settings1['colors'] - settings2['colors']) < 32 and
                abs(settings1['lossy_value'] - settings2['lossy_value']) < 20)

    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> float:
        """Get file size in MB."""
        return Path(file_path).stat().st_size / (1024 * 1024)


class GIFOptimizer(FileProcessor):
    """Handles GIF optimization operations."""

    def __init__(self, compression_settings: Dict = None):
        super().__init__()
        self.compression_settings = compression_settings or GIF_COMPRESSION
        self.failed_files = []
        self.dev_logger = logging.getLogger('developer')
        self._init_directories()
        self.dynamic_optimizer = DynamicGIFOptimizer()  # Add the new optimizer

    def _init_directories(self):
        """Initialize required directories."""
        for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_FILE_DIR]:
            self.ensure_directory(Path(directory))

    def optimize_gif(self, input_path: Path, output_path: Path, target_size_mb: float = 15.0) -> Tuple[float, bool]:
        """
        Dynamically optimize GIF using adaptive settings based on results.
        """
        input_size = self.get_file_size(input_path)
        self.dev_logger.info(
            f"Starting optimization: {input_size:.2f}MB → Target: {target_size_mb}MB")

        # Initialize optimization parameters with more conservative initial settings
        current_settings = self.base_settings.copy()
        best_result = {'size': float('inf'), 'settings': None}
        attempt = 0
        max_attempts = 10

        # More gradual initial scaling based on input size
        size_ratio = target_size_mb / input_size
        current_settings['scale_factor'] = min(
            1.0, max(0.7, size_ratio ** 0.25))  # More conservative scaling

        while attempt < max_attempts:
            attempt += 1
            self.dev_logger.info(f"\nAttempt {attempt} with settings:")
            self.dev_logger.info(
                f"Scale: {current_settings['scale_factor']:.3f}")
            self.dev_logger.info(f"Colors: {current_settings['colors']}")
            self.dev_logger.info(f"Lossy: {current_settings['lossy_value']}")

            # Try current settings
            result_size = self._apply_optimization(
                input_path, output_path, current_settings)

            if result_size <= target_size_mb:
                self.dev_logger.info(f"Target achieved: {result_size:.2f}MB")
                return result_size, True

            if result_size < best_result['size']:
                best_result = {'size': result_size,
                               'settings': current_settings.copy()}

            # Update settings based on results
            new_settings = self._adjust_settings(
                current_settings,
                input_size,
                result_size,
                target_size_mb,
                attempt
            )

            if self._settings_similar(current_settings, new_settings):
                new_settings = self._get_aggressive_settings(
                    current_settings, attempt)

            current_settings = new_settings

            # Log optimization progress
            reduction = ((input_size - result_size) / input_size) * 100
            self.dev_logger.info(
                f"Result: {result_size:.2f}MB ({reduction:.1f}% reduction)")

        # If we couldn't hit target, use best settings found
        if best_result['settings']:
            self.dev_logger.info("Using best settings found...")
            final_size = self._apply_optimization(
                input_path, output_path, best_result['settings'])
            return final_size, final_size <= target_size_mb

        return input_size, False


class GIFProcessor(GIFOptimizer):
    """Main GIF processing class combining optimization and conversion."""

    def __init__(self, compression_settings: Dict = None):
        super().__init__(compression_settings)
        # Initialize additional attributes
        self.ffmpeg = FFmpegHandler()
        self.dev_logger = logging.getLogger('developer')
        self._shutdown_event = threading.Event()
        self._processing_lock = threading.Lock()
        self._active_threads = set()
        self._threads_lock = threading.Lock()
        self._file_locks = {}
        self._file_locks_lock = threading.Lock()
        self._processing_cancelled = threading.Event()
        self.logging_lock = threading.Lock()
        self.processed_files = set()
        self.size_check_lock = threading.Lock()
        self.dynamic_optimizer = DynamicGIFOptimizer()
        self._processing_progress = {}
        self._progress_lock = threading.Lock()
        self.palette_cache = {}
        self.palette_lock = threading.Lock()
        self._processing_timeout = 600
        self.resource_manager = ResourceManager()
        self.retry_count = 3
        self.retry_delay = 1.0
        self._stats = {'processed': 0, 'failed': 0, 'retried': 0}
        self._stats_lock = threading.Lock()
        self.memory_manager = MemoryManager(threshold_mb=1500)
        self.batch_processor = BatchProcessor()
        self.stats_manager = ProcessingStats()
        self._file_cache = TTLCache(maxsize=100, ttl=300)
        self._cleanup_handlers = []  # Add this line
        signal.signal(signal.SIGINT, self._signal_handler)

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
        self._shutdown_event.set()

        # Cleanup thread pools first
        if hasattr(self.resource_manager, '_executor'):
            self.resource_manager._executor.shutdown(wait=False)

        # Give threads a chance to cleanup
        cleanup_timeout = 30
        start_time = time.time()

        while time.time() - start_time < cleanup_timeout:
            with self._threads_lock:
                if not self._active_threads:
                    break
            time.sleep(0.1)

        # Force cleanup remaining resources
        with self._file_locks_lock:
            self._file_locks.clear()

        # Run registered cleanup handlers
        for handler in self._cleanup_handlers:
            try:
                handler()
            except Exception as e:
                self.dev_logger.error(f"Cleanup handler failed: {str(e)}")

        try:
            temp_dir = Path(TEMP_FILE_DIR)
            if temp_dir.exists():
                for temp_file in temp_dir.glob("*"):
                    try:
                        if temp_file.is_file():
                            temp_file.unlink(missing_ok=True)
                    except Exception as e:
                        self.dev_logger.error(
                            f"Failed to cleanup {temp_file}: {str(e)}")
        except Exception as e:
            self.dev_logger.error(f"Error during cleanup: {str(e)}")

        # Clear caches
        self._file_cache.clear()
        self.dimension_cache.clear()
        self.palette_cache.clear()

    def register_cleanup_handler(self, handler: Callable) -> None:
        """Register a cleanup handler to be called during resource cleanup."""
        self._cleanup_handlers.append(handler)

    def _log_with_lock(self, level: str, message: str, file_id: str = "") -> None:
        """Thread-safe logging with deduplication."""
        with self.logging_lock:
            log_key = f"{file_id}:{message}"
            if log_key not in self.processed_files:
                if level == "info":
                    self.dev_logger.info(message)
                elif level == "error":
                    self.dev_logger.error(message)
                elif level == "warning":
                    self.dev_logger.warning(message)
                elif level == "success":
                    self.user_logger.success(message)
                self.processed_files.add(log_key)

    def create_gif(self, file_path: Path, palette_path: Path, output_gif: Path,
                   fps: int, dimensions: Tuple[int, int]) -> bool:
        """Create optimized GIF with better performance."""
        try:
            # Optimize GIF creation command
            cmd = [
                'ffmpeg', '-i', str(file_path),
                '-i', str(palette_path),
                '-lavfi',
                f'fps={fps},'
                f'scale={dimensions[0]}:{dimensions[1]}:flags=lanczos[x];'
                f'[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
                '-y',
                str(output_gif)
            ]

            # Add performance optimization flags
            cmd.extend([
                '-threads', '4',
                '-preset', 'faster',
                '-movflags', '+faststart'
            ])

            if run_ffmpeg_command(cmd):
                # Verify output
                if output_gif.exists() and output_gif.stat().st_size > 0:
                    gif_size = self.get_file_size(
                        output_gif, force_refresh=True)
                    logging.info(
                        f"[{fps}fps] Generated GIF ({gif_size:.2f}MB) -> Optimizing...")
                    return True
                return False
            return False

        except Exception as e:
            logging.error(f"GIF creation failed: {e}")
            return False

    def process_single_fps(self, args: Tuple) -> ProcessingResult:
        """Process a single optimization pass with optional settings override.

        Args:
            file_path: Path to the GIF or video file to process.
            output_path: Path to the output GIF file.
            is_video: Whether the input file is a video.
            pass_index: Index of the pass to use from GIF_PASS_OVERS.
            override_settings: Optional dictionary of settings to override.

        Returns:
            ProcessingResult: Result of the optimization pass.
        """

        file_path, output_path, is_video, fps, current_settings = args
        temp_dir = Path(TEMP_FILE_DIR)
        file_id = f"{Path(file_path).stem}_{fps}"

        # Ensure temp directory exists
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Single temporary file for the optimized GIF
        final_gif = temp_dir / f"{Path(output_path).stem}_{fps}_optimized.gif"

        try:
            # Register file for cleanup
            TempFileManager.register(final_gif)

            if self._processing_cancelled.is_set():
                return ProcessingResult(fps, float('inf'), None,
                                        ProcessingStatus.SIZE_THRESHOLD_EXCEEDED,
                                        "Processing cancelled")

            if is_video:
                # Get dimensions
                from video_optimization import VideoProcessor
                width, height = VideoProcessor()._get_dimensions(file_path)
                if not width or not height:
                    return ProcessingResult(fps, float('inf'), None,
                                            ProcessingStatus.DIMENSION_ERROR,
                                            f"Could not determine dimensions for {Path(file_path).name}")

                # Apply scale factor
                scale_factor = current_settings.get('scale_factor', 1.0)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                # Create optimized GIF in one step
                if not self.ffmpeg.create_optimized_gif(
                    file_path, final_gif, fps, (new_width,
                                                new_height), current_settings
                ):
                    return ProcessingResult(fps, float('inf'), None,
                                            ProcessingStatus.CONVERSION_ERROR,
                                            f"GIF creation failed for {Path(file_path).name}")

            # Get size after creation
            gif_size = self.get_file_size(final_gif, force_refresh=True)

            # Log progress
            self._log_with_lock("info",
                                f"Created GIF: {gif_size:.2f}MB at {fps}fps", file_id)

            return ProcessingResult(fps, gif_size, str(final_gif),
                                    ProcessingStatus.SUCCESS,
                                    f"Processed successfully - {gif_size:.2f}MB")

        except Exception as e:
            self._log_with_lock("error",
                                f"Error processing {Path(file_path).name}: {str(e)}",
                                file_id)
            return ProcessingResult(fps, float('inf'), None,
                                    ProcessingStatus.OPTIMIZATION_ERROR,
                                    str(e))
        finally:
            # Cleanup temp files
            if final_gif.exists():
                try:
                    final_gif.unlink()
                    TempFileManager.unregister(final_gif)
                except Exception as e:
                    self._log_with_lock("error",
                                        f"Failed to cleanup {final_gif}: {str(e)}",
                                        file_id)

    def performance_monitor(func: Callable) -> Callable:
        """Decorator to monitor function performance."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024

            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                end_memory = process.memory_info().rss / 1024 / 1024

                logging.debug(
                    f"Function: {func.__name__} | "
                    f"Time: {end_time - start_time:.2f}s | "
                    f"Memory: {end_memory - start_memory:.2f}MB"
                )
                return result
            except Exception as e:
                logging.error(
                    f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}")
                raise
        return wrapper

    def process_file(self, file_path: Path, output_path: Path, is_video: bool) -> None:
        """Process a single file with improved dimension detection."""
        file_lock = self.get_file_lock(str(file_path))

        try:
            with file_lock:
                if self._shutdown_event.is_set():
                    return

                # Get dimensions first
                dimensions = self._get_dimensions_with_retry(file_path)
                if not dimensions or None in dimensions:
                    self.dev_logger.error(
                        f"Could not determine dimensions for {file_path.name}")
                    self.failed_files.append(file_path)
                    return

                if is_video:
                    self._process_video(file_path, output_path, dimensions)
                else:
                    self._process_gif(file_path, output_path, dimensions)

        except Exception as e:
            self.dev_logger.error(
                f"Error processing {file_path.name}: {str(e)}")
            self.failed_files.append(file_path)
        finally:
            self._cleanup_file_resources(file_path)

    def _process_video(self, file_path: Path, output_path: Path, dimensions: Tuple[int, int]) -> None:
        """Process video with enhanced parallel optimization strategy."""
        temp_files = []
        try:
            # Get video dimensions with retry
            width, height = dimensions
            if not width or height:
                self.dev_logger.error(
                    f"Could not determine dimensions for {file_path.name}")
                self.failed_files.append(file_path)
                return

            min_fps, max_fps = self.compression_settings['fps_range']
            target_size = self.compression_settings.get('min_size_mb', 15.0)
            initial_size = self.get_file_size(file_path)

            # Improved optimization strategy with parallel processing
            optimization_configs = []
            # Progressive scale reduction
            scale_factors = [1.0, 0.75, 0.5, 0.25]
            color_configs = [256, 192, 128, 64]      # Color palette options

            # Generate configs for parallel processing
            for scale in scale_factors:
                for colors in color_configs:
                    optimization_configs.append({
                        'scale_factor': scale,
                        'colors': colors,
                        # Dynamic lossy value
                        'lossy_value': min(100, int(30 * (1/scale)))
                    })

            successful_results = []
            temp_files = []

            self.dev_logger.info(
                f"Starting parallel optimization for {file_path.name} ({initial_size:.2f}MB)")

            # Process configs in parallel
            with ThreadPoolExecutor(max_workers=min(4, os.cpu_count() or 1)) as executor:
                futures = []

                for config in optimization_configs:
                    if self._shutdown_event.is_set():
                        break

                    for fps in range(min_fps, max_fps + 1):
                        temp_gif = Path(
                            TEMP_FILE_DIR) / f"{file_path.stem}_{fps}_{config['scale_factor']:.2f}.gif"
                        temp_files.append(temp_gif)

                        future = executor.submit(
                            self._process_single_config,
                            file_path,
                            temp_gif,
                            fps,
                            width,
                            height,
                            config,
                            target_size
                        )
                        futures.append((future, fps, config))

                # Collect results as they complete
                for future, fps, config in zip(as_completed(futures), range(min_fps, max_fps + 1), optimization_configs):
                    try:
                        # 2-minute timeout per configuration
                        result = future.result(timeout=120)
                        if result and result.get('success'):
                            successful_results.append(result)

                            # Early exit if we find a good result
                            if result['size'] <= target_size and fps >= min_fps + 2:
                                break
                    except TimeoutError:
                        self.dev_logger.warning(
                            f"Timeout processing FPS {fps} with scale {config['scale_factor']}")
                    except Exception as e:
                        self.dev_logger.error(
                            f"Error processing FPS {fps}: {str(e)}")

            # Process final results
            if successful_results:
                # Sort by quality metrics (FPS, size, scale)
                best_result = max(
                    successful_results,
                    key=lambda x: (
                        x['fps'],
                        -x['size'],
                        x['settings']['scale_factor']
                    )
                )

                shutil.copy2(best_result['path'], output_path)
                self.dev_logger.info(
                    f"Final result: {best_result['fps']}FPS, "
                    f"Scale: {best_result['settings']['scale_factor']:.3f}, "
                    f"Size: {best_result['size']:.2f}MB"
                )
            else:
                self.dev_logger.error(
                    f"No successful results for {file_path.name}")
                self.failed_files.append(file_path)

        except Exception as e:
            self.dev_logger.error(
                f"Error processing {file_path.name}: {str(e)}")
            self.failed_files.append(file_path)
        finally:
            self._cleanup_temp_files(temp_files)

    def _process_single_config(self, file_path: Path, temp_gif: Path, fps: int,
                               width: int, height: int, config: dict, target_size: float) -> Optional[dict]:
        """Process a single optimization configuration."""
        try:
            # Create initial GIF
            if self.ffmpeg.create_optimized_gif(
                file_path,
                temp_gif,
                fps,
                (int(width * config['scale_factor']),
                 int(height * config['scale_factor'])),
                config
            ):
                size = self.get_file_size(temp_gif)

                # Only proceed with gifsicle if initial size is promising
                if size < min(90, target_size * 2):
                    optimized_gif = Path(TEMP_FILE_DIR) / \
                        f"{temp_gif.stem}_opt.gif"

                    cmd = [
                        'gifsicle',
                        '--optimize=3',
                        '--colors', str(config['colors']),
                        '--lossy=' + str(config['lossy_value']),
                        '--no-conserve-memory',
                        '--threads=2',
                        str(temp_gif),
                        '-o', str(optimized_gif)
                    ]

                    if run_ffmpeg_command(cmd) and optimized_gif.exists():
                        final_size = self.get_file_size(optimized_gif)
                        return {
                            'success': True,
                            'fps': fps,
                            'size': final_size,
                            'path': str(optimized_gif),
                            'settings': config
                        }

        except Exception as e:
            self.dev_logger.error(f"Configuration processing error: {str(e)}")

        return None

    def _cleanup_temp_files(self, temp_files: list[Path]) -> None:
        """Clean up temporary files with improved error handling."""
        for temp_file in temp_files:
            try:
                if temp_file and temp_file.exists():
                    temp_file.unlink()
                TempFileManager.unregister(temp_file)
            except Exception as e:
                self.dev_logger.error(
                    f"Failed to cleanup {temp_file}: {str(e)}")

    def _cleanup_file_resources(self, file_path: str) -> None:
        """Clean up resources associated with a file."""
        with self._file_locks_lock:
            if file_path in self._file_locks:
                del self._file_locks[file_path]

    def _wrapped_process_single_fps(self, args: Tuple) -> ProcessingResult:
        """Wrapper for process_single_fps with thread registration."""
        self._register_thread()
        try:
            return self.process_single_fps(args)
        finally:
            self._unregister_thread()

    @performance_monitor
    def process_all(self) -> List[Path]:
        """Process all GIF files in input directory."""
        self.failed_files = []
        try:
            # Process videos
            for video_format in SUPPORTED_VIDEO_FORMATS:
                if self._shutdown_event.is_set():
                    break

                for video_file in Path(INPUT_DIR).glob(f'*{video_format}'):
                    if self._shutdown_event.is_set():
                        break

                    output_gif = Path(OUTPUT_DIR) / f"{video_file.stem}.gif"
                    if not output_gif.exists():
                        self.process_file(
                            video_file, output_gif, is_video=True)

            # Process GIFs
            for gif_file in Path(INPUT_DIR).glob('*.gif'):
                if self._shutdown_event.is_set():
                    break

                output_gif = Path(OUTPUT_DIR) / f"{gif_file.stem}.gif"
                if not output_gif.exists():
                    self.process_file(gif_file, output_gif, is_video=False)

        except KeyboardInterrupt:
            self.dev_logger.warning("Processing interrupted by user")
            self._shutdown_event.set()
        except Exception as e:
            self.dev_logger.error(f"Error in process_all: {str(e)}")
        finally:
            self.cleanup_resources()

        return self.failed_files

    def _process_gif(self, gif_path: Path) -> Optional[Path]:
        """Process single GIF file with error handling."""
        try:
            if self._shutdown_event.is_set():
                return None

            file_size = self.get_file_size(gif_path)
            if file_size > GIF_SIZE_TO_SKIP:
                self.dev_logger.warning(
                    f"Skipping {gif_path.name} - {file_size:.1f}MB exceeds size limit")
                return None

            return gif_path
        except Exception as e:
            self.dev_logger.error(f"Error processing {gif_path}: {str(e)}")
            return None

    def _update_progress(self, file_id: str, status: str) -> None:
        """Update processing progress for a file."""
        with self._progress_lock:
            self._processing_progress[file_id] = {
                'status': status,
                'timestamp': time.time()
            }

    def _check_timeout(self, file_id: str) -> bool:
        """Check if processing has timed out."""
        with self._progress_lock:
            if file_id in self._processing_progress:
                elapsed = time.time() - \
                    self._processing_progress[file_id]['timestamp']
                return elapsed > self._processing_timeout
        return False

    def _signal_handler(self, signum, frame):
        """Enhanced signal handler with proper cleanup."""
        self.dev_logger.warning("\nReceived interrupt signal. Cleaning up...")
        try:
            self._shutdown_event.set()
            self.ffmpeg._kill_current_process()
            self.cleanup_resources()
        finally:
            sys.exit(0)

    def _process_single_config_with_retry(self, *args, **kwargs) -> Optional[dict]:
        """Process single configuration with retry logic."""
        for attempt in range(self.retry_count):
            try:
                return self._process_single_config(*args, **kwargs)
            except Exception as e:
                if attempt < self.retry_count - 1:
                    with self._stats_lock:
                        self._stats['retried'] += 1
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory."""
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        # Assume 500MB per process
        return max(1, min(10, int(available_memory / 500)))

    def _generate_optimization_configs(self, input_size: float, target_size: float) -> List[Dict]:
        """Generate optimized configuration list."""
        ratio = target_size / input_size
        configs = []

        # Dynamic scale factors based on size ratio
        scale_factors = [
            min(1.0, max(0.25, ratio ** 0.5)),
            min(0.75, max(0.2, ratio ** 0.6)),
            min(0.5, max(0.15, ratio ** 0.7)),
            min(0.25, max(0.1, ratio ** 0.8))
        ]

        for scale in scale_factors:
            for colors in [256, 192, 128, 64]:
                configs.append({
                    'scale_factor': scale,
                    'colors': colors,
                    'lossy_value': min(100, int(30 * (1/scale))),
                    # Priority for processing order
                    'priority': scale * (colors / 256)
                })

        # Sort by priority
        return sorted(configs, key=lambda x: x['priority'], reverse=True)

    def _check_early_exit(self, results: List[Dict]) -> bool:
        """Check if we can exit early with good enough results."""
        if not results:
            return False

        best_result = max(results, key=lambda x: (x['fps'], -x['size']))
        target_size = self.compression_settings.get('min_size_mb', 15.0)

        return (best_result['size'] <= target_size and
                best_result['fps'] >= self.compression_settings['fps_range'][0] + 2)

    def _get_dimensions_with_retry(self, file_path: Path, max_retries: int = 3) -> Tuple[Optional[int], Optional[int]]:
        """Get video dimensions with improved error handling and validation."""
        methods = [
            self._get_dimensions_ffprobe,
            self._get_dimensions_opencv,
            self._get_dimensions_ffmpeg
        ]

        for method in methods:
            for attempt in range(max_retries):
                try:
                    dimensions = method(file_path)
                    if self._validate_dimensions(dimensions):
                        self.dev_logger.info(
                            f"Got dimensions using {method.__name__}: {dimensions}")
                        return dimensions
                    self.dev_logger.warning(
                        f"Invalid dimensions from {method.__name__}: {dimensions}")
                except Exception as e:
                    self.dev_logger.warning(
                        f"{method.__name__} attempt {attempt + 1} failed: {str(e)}")
                    if attempt < max_retries - 1:
                        time.sleep(0.5 * (attempt + 1))
                        continue
                    break

        self.dev_logger.error(
            f"All dimension detection methods failed for {file_path}")
        return None, None

    def _validate_dimensions(self, dimensions: Tuple[Optional[int], Optional[int]]) -> bool:
        """Validate that dimensions are reasonable."""
        if not dimensions or len(dimensions) != 2:
            return False
        width, height = dimensions
        if not isinstance(width, int) or not isinstance(height, int):
            return False
        if width <= 0 or height <= 0:
            return False
        if width > 7680 or height > 4320:  # 8K resolution limit
            return False
        return True

    def _get_dimensions_ffprobe(self, file_path: Path) -> Tuple[int, int]:
        """Get dimensions using ffprobe with improved error handling."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height',
                '-of', 'json',
                str(file_path)
            ]
            output = subprocess.check_output(
                cmd, stderr=subprocess.PIPE, text=True)
            data = json.loads(output)

            if 'streams' in data and data['streams']:
                stream = data['streams'][0]
                if 'width' in stream and 'height' in stream:
                    return (int(stream['width']), int(stream['height']))
            raise ValueError("No valid stream data found")
        except Exception as e:
            raise ValueError(f"FFprobe failed: {str(e)}")

    def _get_dimensions_opencv(self, file_path: Path) -> Tuple[int, int]:
        """Get dimensions using OpenCV with proper resource management."""
        cap = None
        try:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                raise ValueError("Failed to open video file")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if width <= 0 or height <= 0:
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                else:
                    raise ValueError("Failed to read frame dimensions")

            return (width, height)
        except Exception as e:
            raise ValueError(f"OpenCV failed: {str(e)}")
        finally:
            if cap is not None:
                cap.release()

    def _get_dimensions_ffmpeg(self, file_path: Path) -> Tuple[int, int]:
        """Get dimensions using FFmpeg as last resort."""
        try:
            cmd = ['ffmpeg', '-i', str(file_path)]
            output = subprocess.run(cmd, capture_output=True, text=True).stderr
            matches = re.findall(r'Stream.*Video.* (\d+)x(\d+)', output)
            if matches:
                width, height = map(int, matches[0])
                if width > 0 and height > 0:
                    return (width, height)
            raise ValueError("No valid dimensions found in FFmpeg output")
        except Exception as e:
            raise ValueError(f"FFmpeg failed: {str(e)}")


def validate_compression_settings(settings: Dict[str, Any]) -> None:
    required_keys = ['fps_range', 'colors', 'lossy_value']
    if not all(key in settings for key in required_keys):
        raise ValueError(f"Missing required keys: {required_keys}")


def process_gifs(compression_settings: Optional[Dict[str, Any]] = None) -> List[Path]:
    if compression_settings is not None:
        validate_compression_settings(compression_settings)
    processor = GIFProcessor(compression_settings)
    return processor.process_all()


class ResourceManager:
    """Enhanced resource manager with proper cleanup."""

    def __init__(self):
        self.cpu_count = os.cpu_count() or 1
        self.memory_threshold = 85  # Percentage
        self._executor = None
        self._active_tasks = 0
        self._tasks_lock = threading.Lock()
        self._shutdown = False
        self._executors = []

    @property
    def max_workers(self) -> int:
        """Dynamically calculate max workers based on system load."""
        cpu_percent = psutil.cpu_percent()
        memory_percent = psutil.virtual_memory().percent

        if memory_percent > self.memory_threshold or cpu_percent > 90:
            return max(1, self.cpu_count // 4)
        elif cpu_percent > 75:
            return max(1, self.cpu_count // 2)
        return max(1, self.cpu_count - 1)

    def get_executor(self) -> ThreadPoolExecutor:
        """Get or create thread pool executor with tracking."""
        if self._shutdown:
            raise RuntimeError("ResourceManager is shut down")

        if not self._executor:
            self._executor = ThreadPoolExecutor(max_workers=self.max_workers)
            self._executors.append(self._executor)
        return self._executor

    def shutdown(self):
        """Properly shutdown all executors."""
        self._shutdown = True
        for executor in self._executors:
            try:
                executor.shutdown(wait=False)
            except Exception as e:
                logging.error(f"Error shutting down executor: {e}")
        self._executors.clear()


class MemoryManager:
    """Manages memory usage and cleanup."""

    def __init__(self, threshold_mb: int = 1000):
        self.threshold_mb = threshold_mb
        self._last_cleanup = time.time()
        self._cleanup_interval = 30  # seconds
        self._memory_usage = []
        self._lock = threading.Lock()

    def check_memory(self) -> bool:
        """Check if memory cleanup is needed."""
        current_memory = psutil.Process().memory_info().rss / (1024 * 1024)
        with self._lock:
            self._memory_usage.append(current_memory)
            if len(self._memory_usage) > 10:
                self._memory_usage.pop(0)

            if (current_memory > self.threshold_mb or
                    time.time() - self._last_cleanup > self._cleanup_interval):
                self.cleanup()
                return True
        return False

    def cleanup(self) -> None:
        """Perform memory cleanup."""
        gc.collect()
        with self._lock:
            self._last_cleanup = time.time()
            self._memory_usage.clear()


class BatchProcessor:
    """Handles batch processing of files with memory management."""

    def __init__(self, max_batch_size: int = 5):
        self.max_batch_size = max_batch_size
        self.memory_manager = MemoryManager()
        self.queue = queue.Queue()
        self._results = []
        self._lock = threading.Lock()

    def process_batch(self, items: List[Any], processor_func: Callable) -> Generator:
        """Process items in batches with memory management."""
        for i in range(0, len(items), self.max_batch_size):
            batch = items[i:i + self.max_batch_size]
            futures = []

            with ThreadPoolExecutor(max_workers=min(len(batch), os.cpu_count() or 2)) as executor:
                for item in batch:
                    futures.append(executor.submit(processor_func, item))

                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=120)
                        yield result
                    except Exception as e:
                        logging.error(f"Batch processing error: {str(e)}")
                        continue

            self.memory_manager.check_memory()


class ProcessingStats:
    """Tracks processing statistics and performance metrics."""

    def __init__(self):
        self.stats = {
            'processed': 0,
            'failed': 0,
            'total_time': 0,
            'memory_usage': [],
            'errors': []
        }
        self._lock = threading.Lock()

    def record_success(self, fps: int, size: float, settings: Dict) -> None:
        """Record successful processing."""
        with self._lock:
            self.stats['processed'] += 1
            self.stats.setdefault('fps_distribution', {}).setdefault(fps, 0)
            self.stats['fps_distribution'][fps] += 1

    def record_error(self, error: str) -> None:
        """Record processing error."""
        with self._lock:
            self.stats['failed'] += 1
            self.stats['errors'].append(error)

    def save_stats(self) -> None:
        """Save processing statistics."""
        try:
            with open('processing_stats.json', 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception:
            pass


class ProcessManager:
    """Enhanced process management with proper cleanup."""

    def cleanup(self):
        """Enhanced cleanup with process group handling."""
        if not self.process:
            return

        try:
            if self.is_windows:
                if self.process.poll() is None:
                    try:
                        self.process.terminate()
                        try:
                            self.process.wait(timeout=5)
                        except subprocess.TimeoutExpired:
                            self.process.kill()
                    except Exception:
                        if self.process.poll() is None:
                            subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.process.pid)],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                try:
                    pgid = os.getpgid(self.process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(pgid, signal.SIGKILL)
                except ProcessLookupError:
                    pass  # Process already terminated
        except Exception as e:
            logging.error(f"Failed to cleanup process: {e}")
            # Force kill as last resort
            if self.process.poll() is None:
                self.process.kill()
