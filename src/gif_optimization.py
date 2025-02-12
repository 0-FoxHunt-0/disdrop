import logging
import os
import queue
import re  # Add this import at the top
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
import ctypes
import gc
from typing import NamedTuple
from functools import lru_cache
import asyncio
from contextlib import asynccontextmanager
import numpy as np  # Add at top with other imports

# Fix the imports to use absolute paths since this is imported from main.py
from src.default_config import (GIF_COMPRESSION, GIF_PASS_OVERS, GIF_SIZE_TO_SKIP,
                                INPUT_DIR, OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS,
                                TEMP_FILE_DIR)
from src.logging_system import log_function_call, run_ffmpeg_command
from src.temp_file_manager import TempFileManager
from src.video_optimization import VideoProcessor
from src.utils.error_handler import VideoProcessingError
from src.utils.video_dimensions import get_video_dimensions
from .utils.resource_manager import ResourceMonitor, ResourceGuard


class OptimizationConfig(NamedTuple):
    scale_factor: float
    colors: int
    lossy_value: int
    dither_mode: str = 'bayer'
    bayer_scale: int = 3


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
    settings: Optional[OptimizationConfig] = None


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

            return run_ffmpeg_command(cmd)

        except Exception as e:
            logging.error(f"GIF creation failed: {str(e)}")
            return False


class DynamicGIFOptimizer:
    """Handles dynamic GIF optimization with adaptive settings."""

    def __init__(self):
        self.dev_logger = logging.getLogger('developer')
        self.dev_logger.setLevel(logging.DEBUG)
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
        """Dynamically adjust settings based on results, preserving color quality longer."""
        new_settings = current.copy()
        size_ratio = result_size / target_size

        # First try scaling down
        if size_ratio > 2:
            new_settings['scale_factor'] *= 0.85
        elif size_ratio > 1.5:
            new_settings['scale_factor'] *= 0.9
        elif size_ratio > 1.2:
            new_settings['scale_factor'] *= 0.95

        # Then increase lossy compression
        if size_ratio > 1.2:
            increase = min(15, (size_ratio - 1) * 20)
            new_settings['lossy_value'] = min(
                100, new_settings['lossy_value'] + increase)

        # Only reduce colors as a last resort (after attempt 3)
        if attempt > 3 and size_ratio > 1.5:
            if new_settings['colors'] > 192:
                new_settings['colors'] = 192
            elif new_settings['colors'] > 128 and size_ratio > 2:
                new_settings['colors'] = 128

        # Ensure minimum values
        new_settings['scale_factor'] = max(0.3, new_settings['scale_factor'])
        new_settings['colors'] = max(128, new_settings['colors'])
        new_settings['lossy_value'] = min(100, new_settings['lossy_value'])

        return new_settings

    def _get_aggressive_settings(self, current: Dict, attempt: int) -> Dict:
        """Get more aggressive settings when normal adjustments aren't enough."""
        aggressive = current.copy()

        # Start with scale and lossy adjustments
        if attempt <= 3:
            aggressive['scale_factor'] *= 0.85
            aggressive['lossy_value'] = min(80, aggressive['lossy_value'] + 20)
        elif attempt <= 5:
            aggressive['scale_factor'] *= 0.75
            aggressive['lossy_value'] = min(90, aggressive['lossy_value'] + 15)
        else:
            # Only reduce colors in later attempts
            aggressive['scale_factor'] *= 0.7
            aggressive['colors'] = max(128, aggressive['colors'] - 32)
            aggressive['lossy_value'] = min(
                100, aggressive['lossy_value'] + 10)

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

    def _analyze_source_colors(self, file_path: Path) -> int:
        """Analyze source file to determine actual color count."""
        try:
            # Using ffprobe to analyze color information
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'frame=pix_fmt',
                '-of', 'json',
                str(file_path)
            ]

            output = subprocess.check_output(cmd, text=True)
            data = json.loads(output)

            # Estimate colors based on pixel format
            pix_fmt = data.get('frames', [{}])[0].get('pix_fmt', '')

            if 'rgb24' in pix_fmt or 'bgr24' in pix_fmt:
                return 256  # Full color
            elif 'rgb8' in pix_fmt or 'bgr8' in pix_fmt:
                return 256
            elif 'gray' in pix_fmt:
                return 256  # Grayscale

            # Use first frame analysis as fallback
            temp_frame = Path(TEMP_FILE_DIR) / \
                f"temp_frame_{file_path.stem}.png"
            try:
                subprocess.run([
                    'ffmpeg', '-i', str(file_path),
                    '-vframes', '1',
                    '-y', str(temp_frame)
                ], capture_output=True)

                if temp_frame.exists():
                    img = cv2.imread(str(temp_frame))
                    if img is not None:
                        colors = len(np.unique(img.reshape(-1, 3), axis=0))
                        return min(256, colors)
            finally:
                if temp_frame.exists():
                    temp_frame.unlink()

            return 256  # Default fallback
        except Exception as e:
            self.dev_logger.warning(f"Color analysis failed: {e}")
            return 256  # Safe fallback

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

        # Add process lock before it's used
        self._process_lock = threading.Lock()

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
        self._shutdown_initiated = False
        signal.signal(signal.SIGINT, self._signal_handler)
        # Add resource monitor
        self.resource_monitor = ResourceMonitor()

        # Thread management
        self.worker_threads = []
        self.max_threads = 2  # Limit concurrent threads
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self._stop_workers = threading.Event()

        # Add immediate termination flag
        self._immediate_termination = threading.Event()
        # Register signal handlers for immediate termination
        signal.signal(signal.SIGINT, self._immediate_shutdown_handler)
        signal.signal(signal.SIGTERM, self._immediate_shutdown_handler)

        # Add async support
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        # Add async support
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Add dimension cache
        self.dimension_cache = TTLCache(maxsize=100, ttl=300)

        # Add progressive optimization
        self.progressive_optimization = True

        # Add automatic quality adjustment
        self.quality_manager = QualityManager()

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
            # Shutdown resource manager first
            if hasattr(self, 'resource_manager'):
                self.resource_manager.shutdown()

            # Kill FFmpeg processes
            if hasattr(self, 'ffmpeg'):
                self.ffmpeg._kill_current_process()

            # Wait for active threads
            cleanup_timeout = 30
            start_time = time.time()
            while time.time() - start_time < cleanup_timeout:
                with self._threads_lock:
                    if not self._active_threads:
                        break
                time.sleep(0.1)

            # Clear locks and resources
            with self._file_locks_lock:
                self._file_locks.clear()

            # Run cleanup handlers
            for handler in self._cleanup_handlers:
                try:
                    handler()
                except Exception as e:
                    self.dev_logger.error(f"Cleanup handler failed: {str(e)}")

            # Clean temp files
            self._cleanup_temp_directory()

            # Clear caches
            if hasattr(self, '_file_cache'):
                self._file_cache.clear()
            if hasattr(self, 'dimension_cache'):
                self.dimension_cache.clear()
            if hasattr(self, 'palette_cache'):
                self.palette_cache.clear()

        except Exception as e:
            self.dev_logger.error(f"Error during cleanup: {str(e)}")

        self._stop_workers.set()

        # Wait for threads to finish
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=5)

        self.worker_threads.clear()

        # Clear queues
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

    def _cleanup_temp_directory(self):
        """Clean up temporary directory."""
        try:
            temp_dir = Path(TEMP_FILE_DIR)
            if (temp_dir.exists()):
                for temp_file in temp_dir.glob("*"):
                    try:
                        if (temp_file.is_file()):
                            temp_file.unlink(missing_ok=True)
                    except Exception as e:
                        self.dev_logger.error(
                            f"Failed to cleanup {temp_file}: {str(e)}")
        except Exception as e:
            self.dev_logger.error(f"Error cleaning temp directory: {str(e)}")

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

    def create_gif(self, file_path: Path, output_path: Path,
                   fps: int, dimensions: Tuple[int, int]) -> bool:
        """Create optimized GIF with better performance."""
        try:
            # First pass - Create palette optimized GIF
            temp_palette = Path(TEMP_FILE_DIR) / \
                f"palette_{file_path.stem}.png"
            temp_output = Path(TEMP_FILE_DIR) / f"temp_{output_path.stem}.gif"

            try:
                # Generate optimized palette first
                palette_cmd = [
                    'ffmpeg', '-i', str(file_path),
                    '-vf', f'fps={fps},scale={dimensions[0]}:{dimensions[1]}:flags=lanczos,palettegen=max_colors=128:stats_mode=diff',
                    '-y', str(temp_palette)
                ]

                if not run_ffmpeg_command(palette_cmd):
                    return False

                # Create initial GIF with palette
                gif_cmd = [
                    'ffmpeg', '-i', str(file_path),
                    '-i', str(temp_palette),
                    '-lavfi',
                    f'fps={fps},'
                    f'scale={dimensions[0]}:{dimensions[1]}:flags=lanczos[x];'
                    f'[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
                    '-y',
                    str(temp_output)
                ]

                # Add performance optimization flags
                gif_cmd.extend([
                    '-threads', '4',
                    '-preset', 'faster',
                    '-movflags', '+faststart'
                ])

                if run_ffmpeg_command(gif_cmd):
                    # Verify output
                    if temp_output.exists() and temp_output.stat().st_size > 0:
                        gif_size = self.get_file_size(
                            temp_output, force_refresh=True)
                        self.dev_logger.success(
                            f"[{fps}fps] Generated GIF ({gif_size:.2f}MB) -> Optimizing..."
                        )
                        # Copy to final output
                        shutil.copy2(temp_output, output_path)
                        return True
                return False

            finally:
                # Cleanup temp files
                for temp_file in [temp_palette, temp_output]:
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except Exception as e:
                        self.dev_logger.error(
                            f"Failed to cleanup {temp_file}: {e}")

        except Exception as e:
            self.dev_logger.error(f"GIF creation failed: {str(e)}")
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
            # Add timeout to lock acquisition
            if not file_lock.acquire(timeout=5):
                self.dev_logger.error(
                    f"Failed to acquire lock for {file_path.name}")
                self.failed_files.append(file_path)
                return

            if self._shutdown_event.is_set():
                return

            # Check if file already meets size requirements
            file_size = self.get_file_size(file_path)
            target_size = self.compression_settings.get(
                'min_size_mb', 15.0)

            if file_size <= target_size:
                self.dev_logger.info(
                    f"File {file_path.name} already meets size requirements "
                    f"({file_size:.2f}MB <= {target_size}MB)")
                if not output_path.exists():
                    shutil.copy2(file_path, output_path)
                    self.dev_logger.info(
                        f"Copied original file to: {output_path}")
                return

            # Get dimensions first
            dimensions = self._get_dimensions_with_retry(file_path)

            # Add debug logging for dimension validation
            self.dev_logger.debug(
                f"Validating dimensions for {file_path.name}: {dimensions}")

            # Fix dimension validation
            if (not dimensions or
                len(dimensions) != 2 or
                    not all(isinstance(d, int) and d > 0 for d in dimensions)):
                self.dev_logger.error(
                    f"Could not determine valid dimensions for {file_path.name}")
                self.failed_files.append(file_path)
                return

            width, height = dimensions
            if width <= 0 or height <= 0:
                self.dev_logger.error(
                    f"Invalid dimensions for {file_path.name}: {width}x{height}")
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
            try:
                file_lock.release()
            except Exception:
                pass
            self._cleanup_file_resources(str(file_path))

    def _process_video(self, file_path: Path, output_path: Path, dimensions: Tuple[int, int]) -> None:
        """Process video with gradual scaling and optimization."""
        temp_files = []  # Keep track of all temp files
        success = False

        try:
            target_size = self.compression_settings.get('min_size_mb', 10.0)
            initial_size = self.get_file_size(file_path)

            # Analyze source colors first
            source_colors = self._analyze_source_colors(file_path)
            self.dev_logger.info(
                f"Source color analysis: {source_colors} colors")
            current_colors = min(256, source_colors)

            self.dev_logger.info(
                f"Starting processing for: {file_path.name}\n"
                f"Initial size: {initial_size:.2f}MB\n"
                f"Target size: {target_size:.2f}MB\n"
                f"Dimensions: {dimensions[0]}x{dimensions[1]}\n"
                f"Source colors: {source_colors}"
            )

            size_ratio = target_size / initial_size
            scale_factor = min(1.0, (size_ratio ** 0.5))

            attempt = 0
            max_attempts = 15
            current_size = initial_size
            fps = self.compression_settings["fps_range"][0]

            while attempt < max_attempts and current_size > target_size:
                self.dev_logger.info(
                    f"\nOptimization attempt {attempt + 1}/{max_attempts}")

                # Create new temp file with descriptive name
                step_name = f"attempt{attempt+1}_scale{scale_factor:.2f}_colors{current_colors}"
                current_temp_file = Path(
                    TEMP_FILE_DIR) / f"temp_{file_path.stem}_{step_name}.gif"
                temp_files.append(current_temp_file)  # Add to tracking list

                new_width = int(dimensions[0] * scale_factor // 2 * 2)
                new_height = int(dimensions[1] * scale_factor // 2 * 2)

                self.dev_logger.info(
                    f"Converting with settings:\n"
                    f"- Scale: {scale_factor:.3f}\n"
                    f"- Size: {new_width}x{new_height}\n"
                    f"- FPS: {fps}\n"
                    f"- Colors: {current_colors}"
                )

                # Initial conversion with palette generation
                palette_file = Path(TEMP_FILE_DIR) / f"palette_{step_name}.png"
                temp_files.append(palette_file)

                # Generate optimized palette first
                palette_cmd = [
                    'ffmpeg', '-i', str(file_path),
                    '-vf', f'fps={fps},scale={new_width}:{new_height}:flags=lanczos,palettegen=max_colors={current_colors}:stats_mode=diff',
                    '-y', str(palette_file)
                ]

                if not run_ffmpeg_command(palette_cmd):
                    self.dev_logger.error("Palette generation failed")
                    break

                # Convert using the generated palette
                convert_cmd = [
                    'ffmpeg', '-i', str(file_path),
                    '-i', str(palette_file),
                    '-lavfi', f'fps={fps},scale={new_width}:{new_height}:flags=lanczos [x];[x][1:v] paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle',
                    '-y', str(current_temp_file)
                ]

                if not run_ffmpeg_command(convert_cmd):
                    self.dev_logger.error(
                        f"FFmpeg conversion failed at scale {scale_factor:.3f}")
                    break

                current_size = self.get_file_size(current_temp_file)
                self.dev_logger.info(
                    f"Initial conversion result:\n"
                    f"- Size: {current_size:.2f}MB\n"
                    f"- Reduction: {((initial_size - current_size) / initial_size) * 100:.1f}%"
                )

                # Only try gifsicle on smaller files
                if current_size <= 60:
                    size_ratio = current_size / target_size
                    lossy_value = min(100, int(60 * size_ratio))

                    self.dev_logger.info(
                        f"Applying gifsicle optimization:\n"
                        f"- Colors: {current_colors}\n"
                        f"- Lossy: {lossy_value}"
                    )

                    optimized_path = Path(TEMP_FILE_DIR) / \
                        f"opt_{file_path.stem}_{step_name}.gif"
                    temp_files.append(optimized_path)

                    gifsicle_cmd = [
                        'gifsicle',
                        '--optimize=3',
                        '--colors', str(current_colors),
                        '--lossy=' + str(lossy_value),
                        '--no-conserve-memory',
                        '--threads=4',
                        str(current_temp_file),
                        '-o', str(optimized_path)
                    ]

                    if run_ffmpeg_command(gifsicle_cmd):
                        opt_size = self.get_file_size(optimized_path)
                        if opt_size < current_size:
                            current_size = opt_size
                            shutil.copy2(optimized_path, current_temp_file)

                # Check if target achieved
                if current_size <= target_size:
                    self.dev_logger.success(
                        f"\nTarget size achieved!\n"
                        f"Final size: {current_size:.2f}MB"
                    )
                    shutil.copy2(current_temp_file, output_path)
                    success = True
                    break

                # Adjust parameters for next attempt
                if attempt < max_attempts - 1:
                    size_ratio = current_size / target_size
                    old_scale = scale_factor

                    # First reduce scale
                    if size_ratio > 3:
                        scale_factor *= 0.7
                    elif size_ratio > 2:
                        scale_factor *= 0.8
                    elif size_ratio > 1.5:
                        scale_factor *= 0.9

                    # Only reduce colors after several attempts
                    if attempt > 5 and current_colors > source_colors * 0.75:
                        current_colors = int(source_colors * 0.75)
                    elif attempt > 8 and current_colors > source_colors * 0.5:
                        current_colors = int(source_colors * 0.5)

                    scale_factor = max(0.2, scale_factor)

                    self.dev_logger.info(
                        f"Adjusting parameters:\n"
                        f"- Scale: {old_scale:.3f} -> {scale_factor:.3f}\n"
                        f"- Colors: {current_colors}"
                    )

                attempt += 1

            # Handle final result
            if success:
                self.dev_logger.success(
                    f"\n{'='*50}\n"
                    f"Optimization successful!\n"
                    f"{'='*50}"
                )
            else:
                self.dev_logger.warning(
                    f"\n{'='*50}\n"
                    f"Could not reach target size after {max_attempts} attempts\n"
                    f"Final size: {current_size:.2f}MB ({current_size/target_size:.1f}x target)\n"
                    f"{'='*50}"
                )
                # Copy best result anyway
                if current_temp_file and current_temp_file.exists():
                    shutil.copy2(current_temp_file, output_path)

            # Log final stats
            if output_path.exists():
                final_size = self.get_file_size(output_path)
                total_reduction = (
                    (initial_size - final_size) / initial_size) * 100
                self.dev_logger.info(
                    f"\nProcessing complete for {file_path.name}:\n"
                    f"- Initial size: {initial_size:.2f}MB\n"
                    f"- Final size: {final_size:.2f}MB\n"
                    f"- Reduction: {total_reduction:.1f}%\n"
                    f"- Target size {'achieved' if final_size <= target_size else 'not achieved'}"
                )

        except Exception as e:
            self.dev_logger.error(
                f"Error processing {file_path.name}:\n"
                f"- Error type: {type(e).__name__}\n"
                f"- Error message: {str(e)}\n"
                f"- Traceback: {traceback.format_exc()}"
            )
            self.failed_files.append(file_path)
        finally:
            # Only cleanup if successful or exiting
            if success or self._should_exit():
                for temp_file in temp_files:
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except Exception as e:
                        self.dev_logger.error(
                            f"Failed to cleanup {temp_file}: {e}")

    def _compress_large_gif(self, gif_path: Path) -> None:
        """Additional compression for large GIFs."""
        try:
            temp_path = gif_path.with_name(f"temp_{gif_path.name}")
            # More aggressive optimization settings
            cmd = [
                'gifsicle',
                '--optimize=3',
                '--colors', '128',
                '--lossy=100',
                '--scale', '0.75',
                '--no-conserve-memory',
                str(gif_path),
                '-o', str(temp_path)
            ]

            if run_ffmpeg_command(cmd) and temp_path.exists():
                original_size = self.get_file_size(gif_path)
                compressed_size = self.get_file_size(temp_path)

                if compressed_size < original_size:
                    try:
                        gif_path.unlink()
                        temp_path.replace(gif_path)
                        self.dev_logger.success(
                            f"Additional compression succeeded: {original_size:.2f}MB -> {compressed_size:.2f}MB"
                        )
                    except Exception as e:
                        self.dev_logger.error(f"Failed to replace file: {e}")
                else:
                    self.dev_logger.skip(
                        f"Additional compression skipped - no size reduction achieved"
                    )
            elif temp_path.exists():
                temp_path.unlink()

        except Exception as e:
            self.dev_logger.error(f"Additional compression failed: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()

    def _ensure_worker_threads(self):
        """Ensure worker threads are running."""
        if not self.worker_threads:
            for _ in range(self.max_threads):
                thread = threading.Thread(
                    target=self._worker_thread, daemon=True)
                thread.start()
                self.worker_threads.append(thread)

    def _worker_thread(self):
        """Worker thread to process tasks from queue."""
        while not self._stop_workers.is_set():
            try:
                task = self.task_queue.get(timeout=1)
                if task:
                    file_path, temp_gif, fps, dimensions, config, target_size = task
                    result = self._process_single_config(
                        file_path, temp_gif, fps,
                        dimensions[0], dimensions[1],
                        config, target_size
                    )
                    self.result_queue.put(result)
                self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.dev_logger.error(f"Worker thread error: {e}")

    def _process_single_config(self, file_path: Path, temp_gif: Path, fps: int,
                               width: int, height: int, settings: Dict, target_size: float) -> Optional[Dict]:
        """Process a single optimization configuration."""
        try:
            # Create initial GIF
            if self.ffmpeg.create_optimized_gif(
                file_path,
                temp_gif,
                fps,
                (int(width * settings['scale_factor']),
                 int(height * settings['scale_factor'])),
                settings
            ):
                size = self.get_file_size(temp_gif)

                # Only proceed with gifsicle if initial size is promising
                if size < min(90, target_size * 2):
                    optimized_gif = Path(TEMP_FILE_DIR) / \
                        f"{temp_gif.stem}_opt.gif"

                    cmd = [
                        'gifsicle',
                        '--optimize=3',
                        '--colors', str(settings['colors']),
                        '--lossy=' + str(settings['lossy_value']),
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
                            'settings': settings
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

        # Clear process queue
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break

        # Force garbage collection
        gc.collect()

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
                if self._should_exit():
                    self.dev_logger.info(
                        "Gracefully stopping video processing...")
                    break

                for video_file in Path(INPUT_DIR).glob(f'*{video_format}'):
                    if self._should_exit():
                        break

                    output_gif = Path(OUTPUT_DIR) / f"{video_file.stem}.gif"
                    if not output_gif.exists():
                        self.process_file(
                            video_file, output_gif, is_video=True)

            # Process GIFs if not stopped
            if not self._should_exit():
                for gif_file in Path(INPUT_DIR).glob('*.gif'):
                    if self._should_exit():
                        self.dev_logger.info(
                            "Gracefully stopping GIF processing...")
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
            if self._should_exit():
                self.dev_logger.info("Cleaning up after graceful exit...")
            self.cleanup_resources()

        return self.failed_files

    def _process_gif(self, file_path: Path, output_path: Path, dimensions: Tuple[int, int]) -> None:
        """Process GIF with graceful exit support."""
        try:
            if self._should_exit():
                return

            file_size = self.get_file_size(file_path)
            if file_size > GIF_SIZE_TO_SKIP:
                self.dev_logger.warning(
                    f"Skipping {file_path.name} - {file_size:.1f}MB exceeds size limit")
                return None

            max_attempts = 3  # Add max_attempts definition
            for attempt in range(max_attempts):
                if self._should_exit():
                    self.dev_logger.info(
                        "Gracefully stopping GIF optimization...")
                    return

                # Try progressive optimization first
                if self.progressive_optimization:
                    try:
                        success, final_size = self.dynamic_optimizer.optimize_gif(
                            file_path, output_path, self.compression_settings['min_size_mb'])
                        if success:
                            self.dev_logger.info(
                                f"Progressive optimization succeeded: {final_size:.2f}MB")
                            return file_path
                    except Exception as e:
                        self.dev_logger.warning(
                            f"Progressive optimization failed: {e}")

                # Fall back to standard optimization
                try:
                    current_settings = self._get_optimized_configs(
                        file_size,
                        self.compression_settings['min_size_mb'],
                        30,  # Default FPS for GIFs
                        dimensions
                    )[0]  # Use best quality settings first

                    result = self._process_single_config(
                        file_path,
                        output_path,
                        self.compression_settings['fps_range'][0],
                        dimensions[0],
                        dimensions[1],
                        current_settings,
                        self.compression_settings['min_size_mb']
                    )

                    if result and result.get('success'):
                        return file_path

                except Exception as e:
                    self.dev_logger.error(
                        f"Standard optimization attempt {attempt + 1} failed: {e}")
                    if attempt == max_attempts - 1:
                        raise

                time.sleep(1)  # Brief pause between attempts

            return None

        except Exception as e:
            self.dev_logger.error(f"Error processing {file_path}: {str(e)}")
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
        if self._shutdown_initiated:
            return

        self.dev_logger.warning("\nReceived interrupt signal. Cleaning up...")
        try:
            self.cleanup_resources()
        finally:
            sys.exit(0)

    def _immediate_shutdown_handler(self, signum, frame):
        """Handle immediate shutdown when signal is received."""
        if self._shutdown_initiated:
            return

        self.dev_logger.warning(
            "\nReceived termination signal. Stopping all processes...")
        self._immediate_termination.set()
        self._shutdown_event.set()
        self._processing_cancelled.set()

        # Kill any running FFmpeg processes immediately
        self.ffmpeg._kill_current_process()

        # Clear queues immediately
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

        # Stop all worker threads
        self._stop_workers.set()

        # Cleanup resources
        self.cleanup_resources()
        sys.exit(0)

    def _should_exit(self) -> bool:
        """Enhanced exit check that includes immediate termination."""
        return (self._shutdown_event.is_set() or
                self._processing_cancelled.set() or
                self._shutdown_initiated or
                self._immediate_termination.is_set())

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
        last_dimensions = None
        last_error = None
        file_path = Path(file_path)

        if not file_path.exists():
            self.dev_logger.error(f"File not found: {file_path}")
            return None, None

        # Try to process file even if it's being written
        if not self.wait_for_file_completion(file_path, timeout=10):
            self.dev_logger.warning(f"File may still be writing: {file_path}")

        methods = [
            (self._get_dimensions_ffprobe, "FFprobe"),
            (self._get_dimensions_opencv, "OpenCV"),
            (self._get_dimensions_ffmpeg, "FFmpeg")
        ]

        for method, method_name in methods:
            for attempt in range(max_retries):
                try:
                    dimensions = method(file_path)

                    # Store last valid dimensions as backup
                    if dimensions and len(dimensions) == 2 and all(isinstance(d, int) for d in dimensions):
                        last_dimensions = dimensions

                    if self._validate_dimensions(dimensions):
                        self.dev_logger.debug(
                            f"Got valid dimensions using {method_name}: {dimensions}")
                        return dimensions

                except Exception as e:
                    last_error = str(e)
                    if attempt == max_retries - 1:
                        self.dev_logger.debug(
                            f"{method_name} failed after {max_retries} attempts: {str(e)}")

                if attempt < max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))

        # If we have last_dimensions, try to use them as fallback
        if last_dimensions:
            if self._validate_dimensions(last_dimensions, strict=False):
                self.dev_logger.warning(
                    f"Using fallback dimensions: {last_dimensions}")
                return last_dimensions

        self.dev_logger.error(
            f"All dimension detection methods failed for {file_path}. Last error: {last_error}")
        return None, None

    def _validate_dimensions(self, dimensions: Tuple[Optional[int], Optional[int]], strict: bool = True) -> bool:
        """Validate dimensions with configurable strictness."""
        try:
            if not dimensions or len(dimensions) != 2:
                return False

            width, height = dimensions
            if not isinstance(width, int) or not isinstance(height, int):
                return False

            if width <= 0 or height <= 0:
                return False

            # Strict validation includes aspect ratio and size limits
            if strict:
                # Max 8K resolution
                if width > 7680 or height > 4320:
                    return False

                # Minimum dimensions (adjust as needed)
                if width < 16 or height < 16:
                    return False

                # Reasonable aspect ratio (between 1:10 and 10:1)
                aspect_ratio = width / height
                if aspect_ratio < 0.1 or aspect_ratio > 10:
                    return False

            return True

        except Exception as e:
            self.dev_logger.debug(f"Dimension validation error: {str(e)}")
            return False

    def _get_dimensions_ffprobe(self, file_path: Path) -> Tuple[int, int]:
        """Get dimensions using ffprobe with improved reliability."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,rotation,sample_aspect_ratio,display_aspect_ratio',
                '-of', 'json',
                str(file_path)
            ]

            output = subprocess.check_output(
                cmd, stderr=subprocess.PIPE, text=True)
            data = json.loads(output)

            if not data.get('streams'):
                raise ValueError("No video streams found")

            stream = data['streams'][0]

            # Get base dimensions
            width = int(stream.get('width', 0))
            height = int(stream.get('height', 0))

            if width <= 0 or height <= 0:
                raise ValueError("Invalid dimensions in stream")

            # Handle rotation
            rotation = int(stream.get('rotation', '0') or '0')
            if rotation in [90, 270]:
                width, height = height, width

            # Handle pixel aspect ratio correction
            sar = stream.get('sample_aspect_ratio', '1:1')
            if sar and sar != '1:1':
                try:
                    num, den = map(int, sar.split(':'))
                    if num > 0 and den > 0:
                        width = int(width * (num / den))
                except:
                    pass

            return width, height

        except json.JSONDecodeError as e:
            raise ValueError(f"FFprobe JSON error: {e}")
        except subprocess.CalledProcessError as e:
            raise ValueError(f"FFprobe process error: {e}")
        except Exception as e:
            raise ValueError(f"FFprobe error: {str(e)}")

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
                # Try reading first frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                else:
                    raise ValueError("Failed to read frame dimensions")

            # Verify dimensions are valid
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid dimensions: {width}x{height}")

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
            output = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            ).stderr

            # Try multiple regex patterns
            patterns = [
                r'Stream.*Video.* (\d+)x(\d+)',
                r'Video: .* (\d+)x(\d+)',
                r', (\d+)x(\d+)[,\s]'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, output)
                if matches:
                    width, height = map(int, matches[0])
                    if width > 0 and height > 0:
                        return (width, height)

            raise ValueError("No valid dimensions found in FFmpeg output")
        except Exception as e:
            raise ValueError(f"FFmpeg dimension detection failed: {str(e)}")

    def _should_exit(self) -> bool:
        """Check if processing should stop."""
        return (self._shutdown_event.is_set() or
                self._processing_cancelled.is_set() or
                self._shutdown_initiated)

    def create_gif(self, file_path: Path, output_path: Path,
                   fps: int, dimensions: Tuple[int, int]) -> bool:
        """Create optimized GIF with enhanced compression."""
        try:
            # Initial size check
            original_size = self.get_file_size(file_path)
            if original_size > 1000:  # If source is larger than 1GB
                self.dev_logger.error(
                    f"Source file too large: {original_size:.2f}MB")
                return False

            # Calculate optimal dimensions while maintaining aspect ratio
            max_dimension = max(dimensions[0], dimensions[1])
            scale_factor = 1.0
            if max_dimension > 1280:  # Limit max dimension to 1280px
                scale_factor = 1280 / max_dimension

            new_width = int(dimensions[0] * scale_factor // 2 * 2)
            new_height = int(dimensions[1] * scale_factor // 2 * 2)

            # First pass - Create palette optimized GIF
            temp_palette = Path(TEMP_FILE_DIR) / \
                f"palette_{file_path.stem}.png"
            temp_output = Path(TEMP_FILE_DIR) / f"temp_{output_path.stem}.gif"

            try:
                # Generate optimized palette
                palette_cmd = [
                    'ffmpeg', '-i', str(file_path),
                    '-vf', f'fps={fps},scale={new_width}:{new_height}:flags=lanczos,palettegen=max_colors=128:stats_mode=diff',
                    '-y', str(temp_palette)
                ]

                if not run_ffmpeg_command(palette_cmd):
                    return False

                # Create initial GIF with palette
                gif_cmd = [
                    'ffmpeg', '-i', str(file_path),
                    '-i', str(temp_palette),
                    '-lavfi', f'fps={fps},scale={new_width}:{new_height}:flags=lanczos [x];[x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
                    '-y', str(temp_output)
                ]

                if not run_ffmpeg_command(gif_cmd):
                    return False

                # Second pass - Optimize with gifsicle
                if temp_output.exists():
                    gifsicle_cmd = [
                        'gifsicle',
                        '--optimize=3',
                        '--colors', '128',
                        '--lossy=80',
                        '--no-conserve-memory',
                        str(temp_output),
                        '-o', str(output_path)
                    ]

                    if not run_ffmpeg_command(gifsicle_cmd):
                        return False

                    # Verify final size
                    final_size = self.get_file_size(output_path)
                    if final_size > 100:  # If still too large, try extreme compression
                        self._compress_large_gif(output_path)
                        final_size = self.get_file_size(output_path)

                    self.dev_logger.info(
                        f"Generated GIF: Original={original_size:.2f}MB, Final={final_size:.2f}MB"
                    )
                    return True

                return False

            finally:
                # Cleanup temp files
                for temp_file in [temp_palette, temp_output]:
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except Exception as e:
                        self.dev_logger.error(
                            f"Failed to cleanup {temp_file}: {e}")

        except Exception as e:
            self.dev_logger.error(f"GIF creation failed: {str(e)}")
            return False

    def _compress_large_gif(self, gif_path: Path) -> None:
        """Additional compression for large GIFs."""
        try:
            temp_path = gif_path.with_name(f"temp_{gif_path.name}")

            # Extreme optimization settings
            cmd = [
                'gifsicle',
                '--optimize=3',
                '--colors', '64',        # Reduce colors
                '--lossy=100',           # Maximum compression
                '--scale', '0.7',        # Reduce dimensions
                '--no-conserve-memory',
                str(gif_path),
                '-o', str(temp_path)
            ]

            if run_ffmpeg_command(cmd) and temp_path.exists():
                # Replace original with compressed version
                temp_path.replace(gif_path)
            elif temp_path.exists():
                temp_path.unlink()

        except Exception as e:
            self.dev_logger.error(f"Additional compression failed: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()

    def cleanup_resources(self) -> None:
        """Enhanced cleanup with forceful file removal."""
        if self._shutdown_initiated:
            return

        self._shutdown_initiated = True
        self._shutdown_event.set()

        try:
            # Stop all processing first
            self._stop_workers.set()
            for thread in self.worker_threads:
                if thread.is_alive():
                    thread.join(timeout=5)

            # Kill any FFmpeg processes
            self._kill_ffmpeg_processes()

            # Forceful cleanup of temp files
            self._force_cleanup_temp_files()

        except Exception as e:
            self.dev_logger.error(f"Cleanup error: {e}")
            # Emergency cleanup as last resort
            self._emergency_cleanup()

    def _kill_ffmpeg_processes(self) -> None:
        """Kill all FFmpeg processes."""
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
            self.dev_logger.error(f"Failed to kill processes: {e}")

    def _force_cleanup_temp_files(self) -> None:
        """Force cleanup of temporary files and directory."""
        temp_dir = Path(TEMP_FILE_DIR)
        if not temp_dir.exists():
            return

        # Multiple attempts with process killing
        for attempt in range(3):
            try:
                for temp_file in temp_dir.glob("*"):
                    try:
                        if temp_file.is_file():
                            temp_file.unlink(missing_ok=True)
                    except PermissionError:
                        self._kill_ffmpeg_processes()
                        time.sleep(1)
                        temp_file.unlink(missing_ok=True)

                # Try to remove temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                break
            except Exception as e:
                self.dev_logger.error(
                    f"Cleanup attempt {attempt + 1} failed: {e}")
                if attempt == 2:  # Last attempt
                    self._emergency_cleanup()

    def _emergency_cleanup(self) -> None:
        """Emergency cleanup as last resort."""
        self.dev_logger.warning("Performing emergency cleanup...")
        try:
            # Kill all related processes
            self._kill_ffmpeg_processes()
            time.sleep(2)  # Wait for processes to die

            # Force delete temp directory
            temp_dir = Path(TEMP_FILE_DIR)
            if temp_dir.exists():
                if sys.platform == 'win32':
                    os.system(f'rmdir /S /Q "{temp_dir}"')
                else:
                    os.system(f'rm -rf "{temp_dir}"')
        except Exception as e:
            self.dev_logger.error(f"Emergency cleanup failed: {e}")

    def _get_video_frame_info(self, file_path: Path) -> Optional[Tuple[int, float]]:
        """Get total frames and duration of video.

        Args:
            file_path: Path to the video file

        Returns:
            Tuple[int, float]: (total_frames, duration_in_seconds) or None if failed
        """
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_frames,duration',
                '-of', 'json',
                str(file_path)
            ]

            output = subprocess.check_output(cmd, text=True)
            data = json.loads(output)

            if 'streams' in data and data['streams']:
                stream = data['streams'][0]
                frames = int(stream.get('nb_frames', 0))
                duration = float(stream.get('duration', 0))

                if frames == 0 or duration == 0:  # If ffprobe couldn't get frame count, estimate it
                    cmd = [
                        'ffmpeg', '-i', str(file_path), '-map', '0:v:0', '-c', 'copy', '-f', 'null', '-']
                    output = subprocess.run(
                        cmd, capture_output=True, text=True).stderr
                    matches = re.search(r'frame=\s*(\d+)', output)
                    if matches:
                        frames = int(matches.group(1))
                        # Verify frame count is reasonable
                        if frames > 0 and frames < 1000000:  # Sanity check
                            return frames, duration
                        else:
                            logging.warning(
                                f"Suspicious frame count: {frames}, falling back to duration-based estimate")
                            # Estimate frames from duration and common framerates
                            for fps in [30, 25, 24, 60]:
                                estimated = int(duration * fps)
                                if abs(estimated - frames) < frames * 0.1:  # Within 10%
                                    return estimated, duration

                if frames > 0 and duration > 0:
                    return frames, duration

            self.dev_logger.warning(
                f"Could not get accurate frame info for {file_path}")
            return None

        except Exception as e:
            self.dev_logger.error(f"Error getting frame info: {str(e)}")
            return None

    def _get_optimized_configs(self, input_size: float, target_size: float,
                               avg_fps: float, dimensions: Tuple[int, int]) -> List[Dict]:
        """Generate optimized configurations based on input characteristics.

        Args:
            input_size: Original file size in MB
            target_size: Target file size in MB
            avg_fps: Average FPS of the source
            dimensions: (width, height) of the source

        Returns:
            List[Dict]: List of optimization configurations sorted by priority
        """
        configs = []
        size_ratio = target_size / input_size
        max_dimension = max(dimensions)

        # Base scale factor on both size ratio and resolution
        if max_dimension > 1920:
            scale_factors = [0.75, 0.5, 0.35]
        elif max_dimension > 1280:
            scale_factors = [0.85, 0.75, 0.5]
        else:
            scale_factors = [1.0, 0.85, 0.75]

        # Keep high color count initially
        color_configs = [256]
        if size_ratio < 0.3:  # Only add lower color options for very large files
            color_configs.extend([192, 128])

        # Create configs prioritizing quality
        for scale in scale_factors:
            for colors in color_configs:
                # Increased base lossy value
                lossy_value = min(100, int(40 * (1/scale)))
                configs.append({
                    'scale_factor': scale,
                    'colors': colors,
                    'lossy_value': lossy_value,
                    'priority': scale * (colors / 256)
                })

        return sorted(configs, key=lambda x: x['priority'], reverse=True)

    def _check_early_exit(self, results: List[Dict]) -> bool:
        """Check if we can exit early with good enough results."""
        if not results:
            return False

        best_result = max(results, key=lambda x: (x['fps'], -x['size']))
        target_size = self.compression_settings.get('min_size_mb', 15.0)

        return (best_result['size'] <= target_size and
                best_result['fps'] >= self.compression_settings['fps_range'][0] + 2)

    async def process_file_async(self, file_path: Path, output_path: Path, is_video: bool) -> None:
        """Async version of process_file with better resource management."""
        async with managed_process_pool() as pool:
            try:
                # Get file lock with timeout
                file_lock = self.get_file_lock(str(file_path))
                if not await asyncio.get_event_loop().run_in_executor(None, file_lock.acquire, True, 5):
                    self.dev_logger.error(
                        f"Failed to acquire lock for {file_path.name}")
                    self.failed_files.append(file_path)
                    return

                # Check file size and target
                file_size = await asyncio.get_event_loop().run_in_executor(None, self.get_file_size, file_path)
                target_size = self.compression_settings.get(
                    'min_size_mb', 15.0)

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
                self.dev_logger.error(f"Async processing error: {str(e)}")
                self.failed_files.append(file_path)
            finally:
                if file_lock:
                    file_lock.release()

    def optimize_gif(self, input_path: Path, output_path: Path, target_size_mb: float = 15.0) -> Tuple[float, bool]:
        """Improved GIF optimization with progressive quality reduction."""
        # First try progressive optimization if enabled
        if self.progressive_optimization:
            try:
                # Get source FPS to limit maximum target FPS
                source_fps = self._get_source_fps(input_path)
                self.dev_logger.info(f"Source FPS: {source_fps}")

                # Adjust FPS range if needed
                fps_range = self.compression_settings['fps_range']
                if fps_range[1] > source_fps:
                    adjusted_range = (min(fps_range[0], source_fps), min(
                        fps_range[1], source_fps))
                    self.dev_logger.info(
                        f"Adjusting FPS range to match source: {adjusted_range}")
                    self.compression_settings['fps_range'] = adjusted_range

                result = self._progressive_optimize(
                    input_path, output_path, target_size_mb)
                if result.success:
                    return result.size, True
            except Exception as e:
                self.dev_logger.warning(
                    f"Progressive optimization failed: {e}")

        # Fall back to standard optimization
        try:
            file_size = self.get_file_size(input_path)
            dimensions = self._get_dimensions_with_retry(input_path)

            if not dimensions or not self._validate_dimensions(dimensions):
                return file_size, False

            # Get source colors
            source_colors = self._analyze_source_colors(input_path)

            # Start with high quality settings
            current_settings = self._get_optimized_configs(
                file_size,
                target_size_mb,
                self._get_source_fps(input_path),
                dimensions,
                source_colors
            )[0]

            result = self._process_single_config(
                input_path,
                output_path,
                min(self.compression_settings['fps_range']
                    [0], self._get_source_fps(input_path)),
                dimensions[0],
                dimensions[1],
                current_settings,
                target_size_mb
            )

            if result and result.get('success'):
                return result['size'], result['size'] <= target_size_mb

            return file_size, False

        except Exception as e:
            self.dev_logger.error(f"Standard optimization failed: {str(e)}")
            return file_size, False

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
            self.dev_logger.warning(f"Failed to get source FPS: {e}")
            return 30.0  # Safe default


class ProcessingResult(NamedTuple):
    """Improved result tracking."""
    success: bool
    size: float
    settings: OptimizationConfig
    error: Optional[str] = None


# Add cleanup handler improvements
def cleanup_with_retry(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.5 * (2 ** attempt))
    return wrapper


# Add memory monitoring improvements
class MemoryManager:
    """Manages memory usage and cleanup."""

    def __init__(self, threshold_mb: int = 1000):
        self.threshold_mb = threshold_mb
        self._last_cleanup = time.time()
        self._cleanup_interval = 30  # seconds
        self._memory_usage = []
        self._lock = threading.Lock()
        self._gc_threshold = 0.8  # 80% of threshold triggers cleanup

    def check_memory(self) -> bool:
        """Check if memory cleanup is needed with improved thresholds."""
        try:
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024 * 1024)

            with self._lock:
                self._memory_usage.append(current_memory)
                if len(self._memory_usage) > 10:
                    self._memory_usage.pop(0)

                avg_memory = sum(self._memory_usage) / len(self._memory_usage)

                should_cleanup = (
                    current_memory > self.threshold_mb or
                    avg_memory > self.threshold_mb * self._gc_threshold or
                    time.time() - self._last_cleanup > self._cleanup_interval
                )

                if should_cleanup:
                    self.cleanup()
                    return True
            return False
        except Exception as e:
            logging.error(f"Memory check error: {e}")
            return True  # Trigger cleanup on error

    def cleanup(self) -> None:
        """Enhanced memory cleanup with better resource management."""
        try:
            import gc
            gc.collect(generation=2)  # Full collection

            with self._lock:
                self._last_cleanup = time.time()
                self._memory_usage.clear()

            # Try to release memory back to OS on supported platforms
            if hasattr(gc, 'malloc_trim'):  # Python 3.7+ on Linux
                gc.malloc_trim()

            if sys.platform == 'win32':
                import ctypes
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
        except Exception as e:
            logging.error(f"Memory cleanup error: {e}")


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


class ResourceManager:
    """Enhanced resource manager with proper cleanup."""

    def __init__(self):
        self.cpu_count = os.cpu_count() or 1
        self.memory_threshold = 85
        self._executors = []
        self._executor = None
        self._shutdown = False
        self._tasks_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def get_executor(self) -> ThreadPoolExecutor:
        """Get or create thread pool executor with tracking."""
        if self._shutdown:
            return None

        with self._init_lock:
            if not self._executor:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers)
                self._executors.append(self._executor)
            return self._executor

    def shutdown(self):
        """Properly shutdown all executors."""
        self._shutdown = True
        with self._init_lock:
            if self._executor:
                try:
                    self._executor.shutdown(wait=False)
                except Exception as e:
                    logging.error(f"Error shutting down main executor: {e}")
                self._executor = None

            for executor in self._executors:
                try:
                    if not executor._shutdown:
                        executor.shutdown(wait=False)
                except Exception as e:
                    logging.error(f"Error shutting down executor: {e}")
            self._executors.clear()


@asynccontextmanager
async def managed_process_pool(max_workers: int = None):
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        yield pool
    finally:
        pool.shutdown(wait=True)


def _standalone_get_dimensions_ffprobe(file_path: Path) -> Tuple[int, int]:
    """Standalone ffprobe dimension detection."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,rotation,sample_aspect_ratio',
            '-of', 'json',
            str(file_path)
        ]

        output = subprocess.check_output(
            cmd, stderr=subprocess.PIPE, text=True)
        data = json.loads(output)

        if not data.get('streams'):
            raise ValueError("No video streams found")

        stream = data['streams'][0]
        width = int(stream.get('width', 0))
        height = int(stream.get('height', 0))

        if width <= 0 or height <= 0:
            raise ValueError("Invalid dimensions in stream")

        # Handle rotation
        rotation = int(stream.get('rotation', '0') or '0')
        if rotation in [90, 270]:
            width, height = height, width

        return width, height

    except Exception as e:
        raise ValueError(f"FFprobe dimension detection failed: {e}")


def _standalone_get_dimensions_opencv(file_path: Path) -> Tuple[int, int]:
    """Standalone OpenCV dimension detection."""
    cap = None
    try:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}")

        return (width, height)

    except Exception as e:
        raise ValueError(f"OpenCV dimension detection failed: {e}")
    finally:
        if cap is not None:
            cap.release()


def _standalone_get_dimensions_ffmpeg(file_path: Path) -> Tuple[int, int]:
    """Standalone FFmpeg dimension detection."""
    try:
        cmd = ['ffmpeg', '-i', str(file_path)]
        output = subprocess.run(cmd, capture_output=True, text=True).stderr

        patterns = [
            r'Stream.*Video.* (\d+)x(\d+)',
            r'Video: .* (\d+)x(\d+)',
            r', (\d+)x(\d+)[,\s]'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, output)
            if matches:
                width, height = map(int, matches[0])
                if width > 0 and height > 0:
                    return (width, height)

        raise ValueError("No valid dimensions found in FFmpeg output")
    except Exception as e:
        raise ValueError(f"FFmpeg dimension detection failed: {e}")


@lru_cache(maxsize=100)
def cached_dimension_detection(file_path: str) -> Tuple[int, int]:
    """Cached version of dimension detection using standalone functions."""
    try:
        methods = [
            ('ffprobe', _standalone_get_dimensions_ffprobe),
            ('opencv', _standalone_get_dimensions_opencv),
            ('ffmpeg', _standalone_get_dimensions_ffmpeg)
        ]

        for method_name, method in methods:
            try:
                dimensions = method(Path(file_path))
                if dimensions and all(isinstance(d, int) and d > 0 for d in dimensions):
                    return dimensions
            except Exception as e:
                logging.debug(f"{method_name} dimension detection failed: {e}")
                continue

        raise ValueError("All dimension detection methods failed")

    except Exception as e:
        logging.error(f"Dimension detection failed: {e}")
        return (0, 0)


def validate_compression_settings(settings: Dict[str, Any]) -> None:
    required_keys = ['fps_range', 'colors', 'lossy_value']
    if not all(key in settings for key in required_keys):
        raise ValueError(f"Missing required keys: {required_keys}")


def process_gifs(compression_settings: Optional[Dict[str, Any]] = None) -> List[Path]:
    if compression_settings is not None:
        validate_compression_settings(compression_settings)
    processor = GIFProcessor(compression_settings)
    return processor.process_all()


class QualityManager:
    """Manages quality settings adaptation based on results."""

    def __init__(self):
        self.quality_history = {}
        self._lock = threading.Lock()

    def get_settings(self, file_size: float, target_size: float) -> Dict:
        """Get optimized quality settings based on file size."""
        with self._lock:
            ratio = target_size / file_size
            return {
                'scale_factor': min(1.0, max(0.3, ratio ** 0.5)),
                'colors': 256 if ratio > 0.5 else 192,
                'lossy_value': min(100, int(50 * (1/ratio)))
            }

    def update_settings(self, settings: Dict, result_size: float, target_size: float) -> Dict:
        """Update settings based on optimization results."""
        with self._lock:
            new_settings = settings.copy()
            ratio = result_size / target_size

            if ratio > 1.5:
                new_settings['scale_factor'] *= 0.8
                new_settings['colors'] = min(new_settings['colors'], 192)
                new_settings['lossy_value'] = min(
                    100, new_settings['lossy_value'] + 20)
            elif ratio > 1.2:
                new_settings['scale_factor'] *= 0.9
                new_settings['lossy_value'] = min(
                    100, new_settings['lossy_value'] + 10)

            return new_settings
