# gif_optimization.py
import logging
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from cachetools import TTLCache

from default_config import (GIF_COMPRESSION, GIF_PASS_OVERS, GIF_SIZE_TO_SKIP,
                            INPUT_DIR, OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS,
                            TEMP_FILE_DIR)
from logging_system import log_function_call, run_ffmpeg_command
from temp_file_manager import TempFileManager


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
        # Base settings to start with - made more conservative
        self.base_settings = {
            'colors': 256,
            'lossy_value': 15,  # Reduced from 20 for better initial quality
            'scale_factor': 1.0
        }
        # Track optimization history
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
        self.dev_logger.info(f"Starting optimization: {input_size:.2f}MB → Target: {target_size_mb}MB")

        # Initialize optimization parameters
        current_settings = self.base_settings.copy()
        best_result = {'size': float('inf'), 'settings': None}
        attempt = 0
        max_attempts = 10

        # Calculate initial scaling based on input size - more dynamic approach
        size_ratio = target_size_mb / input_size
        if input_size > 200:
            # For very large files, be more aggressive
            current_settings['scale_factor'] = min(1.0, max(0.1, (size_ratio ** 0.75)))
        elif input_size > 100:
            # For large files
            current_settings['scale_factor'] = min(1.0, max(0.2, (size_ratio ** 0.65)))
        elif input_size > 50:
            # For medium files
            current_settings['scale_factor'] = min(1.0, max(0.3, (size_ratio ** 0.5)))
        else:
            # For smaller files, be more conservative
            current_settings['scale_factor'] = min(1.0, max(0.4, (size_ratio ** 0.4)))

        self.dev_logger.info(f"Dynamic initial scale factor: {current_settings['scale_factor']:.3f} (based on {input_size:.1f}MB input)")

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

            # If settings haven't changed significantly, try more aggressive approach
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

            # Handle dimensions if needed
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

    def _adjust_settings(self, current: Dict, input_size: float, result_size: float,
                         target_size: float, attempt: int) -> Dict:
        """Dynamically adjust settings based on results with more gradual changes."""
        new_settings = current.copy()

        # Calculate how far we are from target
        size_ratio = result_size / target_size

        # More gradual scale factor adjustments
        if size_ratio > 2:
            new_settings['scale_factor'] *= 0.9  # Reduced from 0.7
        elif size_ratio > 1.5:
            new_settings['scale_factor'] *= 0.95  # Reduced from 0.85
        elif size_ratio > 1.2:
            new_settings['scale_factor'] *= 0.98  # Reduced from 0.95

        # More gradual color reduction
        if size_ratio > 2 and new_settings['colors'] > 128:
            new_settings['colors'] = max(
                128, new_settings['colors'] - 32)  # More gradual reduction
        elif size_ratio > 1.5 and new_settings['colors'] > 192:
            new_settings['colors'] = max(192, new_settings['colors'] - 16)
        elif size_ratio > 1.2 and new_settings['colors'] > 224:
            new_settings['colors'] = max(224, new_settings['colors'] - 8)

        # More gradual lossy compression increase
        if size_ratio > 1.2:
            # Increase lossy value more gradually
            increase = min(10, (size_ratio - 1) * 15)  # More gradual increase
            new_settings['lossy_value'] = min(
                100, new_settings['lossy_value'] + increase)

        # Enforce minimum values to maintain quality
        new_settings['scale_factor'] = max(
            0.5, new_settings['scale_factor'])  # Increased minimum
        new_settings['colors'] = max(
            128, new_settings['colors'])  # Increased minimum
        new_settings['lossy_value'] = min(
            100, new_settings['lossy_value'])  # Reduced maximum

        return new_settings

    def _get_aggressive_settings(self, current: Dict, attempt: int) -> Dict:
        """Get more aggressive settings when normal adjustments aren't enough, but maintain quality."""
        aggressive = current.copy()

        # More gradual scaling reduction
        if attempt <= 3:
            aggressive['scale_factor'] *= 0.9
        elif attempt <= 5:
            aggressive['scale_factor'] *= 0.85
        else:
            aggressive['scale_factor'] *= 0.8

        # More gradual color reduction
        if attempt <= 3:
            aggressive['colors'] = max(192, aggressive['colors'] - 32)
        elif attempt <= 5:
            aggressive['colors'] = max(156, aggressive['colors'] - 24)
        else:
            aggressive['colors'] = max(128, aggressive['colors'] - 16)

        # More gradual lossy compression increase
        if attempt <= 3:
            aggressive['lossy_value'] = min(60, aggressive['lossy_value'] + 15)
        elif attempt <= 5:
            aggressive['lossy_value'] = min(80, aggressive['lossy_value'] + 10)
        else:
            aggressive['lossy_value'] = min(100, aggressive['lossy_value'] + 5)

        # Enforce quality bounds
        aggressive['scale_factor'] = max(0.5, aggressive['scale_factor'])
        aggressive['colors'] = max(128, aggressive['colors'])
        aggressive['lossy_value'] = min(100, aggressive['lossy_value'])

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
        self.dev_logger.info(f"Starting optimization: {input_size:.2f}MB → Target: {target_size_mb}MB")

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
        # Call parent class constructor first
        super().__init__(compression_settings)

        # Initialize additional GIFProcessor-specific attributes
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
        """Enhanced cleanup with timeout."""
        self._shutdown_event.set()

        # Give threads a chance to cleanup
        cleanup_timeout = 30  # 30 seconds
        start_time = time.time()

        while time.time() - start_time < cleanup_timeout:
            with self._threads_lock:
                if not self._active_threads:
                    break
            time.sleep(0.1)

        # Force cleanup remaining resources
        with self._file_locks_lock:
            self._file_locks.clear()

        try:
            temp_dir = Path(TEMP_FILE_DIR)
            if temp_dir.exists():
                for temp_file in temp_dir.glob("*"):
                    try:
                        if temp_file.is_file():
                            temp_file.unlink()
                    except Exception as e:
                        self.dev_logger.error(
                            f"Failed to cleanup {temp_file}: {str(e)}"
                        )
        except Exception as e:
            self.dev_logger.error(f"Error during cleanup: {str(e)}")

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

    def process_file(self, file_path: Path, output_path: Path, is_video: bool) -> None:
        """Process a single file with dynamic optimization for GIFs."""
        file_lock = self.get_file_lock(str(file_path))

        try:
            with file_lock:
                if self._shutdown_event.is_set():
                    return

                # For videos, maintain existing conversion logic
                if is_video:
                    file_size = self.get_file_size(file_path)
                    if file_size > GIF_SIZE_TO_SKIP:
                        self.dev_logger.warning(
                            f"Skipping video {file_path.name} - {file_size:.1f}MB exceeds size limit"
                        )
                        return
                    # Process video using existing logic
                    self._process_video(file_path, output_path)
                else:
                    # For GIFs, use new dynamic optimization
                    target_size = self.compression_settings.get(
                        'min_size_mb', 15.0)
                    optimized_size, success = self.dynamic_optimizer.optimize_gif(
                        file_path,
                        output_path,
                        target_size
                    )

                    if success:
                        self.dev_logger.info(f"Successfully optimized GIF: {optimized_size:.2f}MB")
                    else:
                        self.dev_logger.error(f"Failed to optimize GIF to target size: {optimized_size:.2f}MB")
                        self.failed_files.append(file_path)

        except Exception as e:
            self.dev_logger.error(f"Fatal error processing {file_path.name}: {e}")
            self.failed_files.append(file_path)
        finally:
            self._cleanup_file_resources(file_path)

    def _process_video(self, file_path: Path, output_path: Path) -> None:
        """Process video with progressive optimization strategy."""
        try:
            from video_optimization import VideoProcessor
            width, height = VideoProcessor()._get_dimensions(file_path)
            if not width or not height:
                self.dev_logger.error(
                    f"Could not determine dimensions for {file_path.name}")
                self.failed_files.append(file_path)
                return

            min_fps, max_fps = self.compression_settings['fps_range']
            target_size = self.compression_settings.get('min_size_mb', 15.0)
            initial_size = self.get_file_size(file_path)

            # Progressive optimization parameters
            current_settings = {
                'scale_factor': min(1.0, max(0.1, (target_size / initial_size) ** 0.5)),
                'colors': 256,
                'lossy_value': 30
            }

            successful_results = []
            temp_files = []

            self.dev_logger.info(
                f"Starting optimization for {file_path.name} ({initial_size:.2f}MB)")
            self.dev_logger.info(
                f"Initial scale factor: {current_settings['scale_factor']:.3f}")

            while current_settings['colors'] >= 64:  # Main optimization loop
                if self._shutdown_event.is_set():
                    break

                # Process all FPS values with current settings
                fps_results = []
                for fps in range(min_fps, max_fps + 1):
                    temp_gif = Path(
                        TEMP_FILE_DIR) / f"{file_path.stem}_{fps}_{current_settings['scale_factor']:.2f}.gif"
                    temp_files.append(temp_gif)

                    try:
                        # Create initial GIF - Fixed arguments here
                        if self.ffmpeg.create_optimized_gif(
                            file_path, 
                            temp_gif, 
                            fps,
                            (int(width * current_settings['scale_factor']),
                             int(height * current_settings['scale_factor'])),  # Combine dimensions into tuple
                            current_settings
                        ):
                            size = self.get_file_size(temp_gif)

                            # If size is under 90MB, try gifsicle optimization
                            if size < 90:
                                optimized_gif = Path(
                                    TEMP_FILE_DIR) / f"{temp_gif.stem}_opt.gif"
                                temp_files.append(optimized_gif)

                                cmd = [
                                    'gifsicle',
                                    '--optimize=3',
                                    '--colors', str(
                                        current_settings['colors']),
                                    '--lossy=' +
                                    str(current_settings['lossy_value']),
                                    '--no-conserve-memory',
                                    str(temp_gif),
                                    '-o', str(optimized_gif)
                                ]

                                if run_ffmpeg_command(cmd) and optimized_gif.exists():
                                    size = self.get_file_size(optimized_gif)

                                    fps_results.append({
                                        'fps': fps,
                                        'size': size,
                                        'path': str(optimized_gif),
                                        'settings': current_settings.copy()
                                    })

                                    self.dev_logger.info(
                                        f"FPS: {fps}, Scale: {current_settings['scale_factor']:.3f}, "
                                        f"Colors: {current_settings['colors']}, "
                                        f"Lossy: {current_settings['lossy_value']} -> {size:.2f}MB"
                                    )

                    except Exception as e:
                        self.dev_logger.error(
                            f"Error processing {fps}FPS: {str(e)}")
                        continue

                # Check if any results meet target size
                valid_results = [
                    r for r in fps_results if r['size'] <= target_size]
                if valid_results:
                    successful_results.extend(valid_results)
                    break

                # Adjust settings for next iteration
                if current_settings['scale_factor'] > 0.1:
                    # Reduce scale first
                    current_settings['scale_factor'] *= 0.75
                else:
                    # At minimum scale, reduce colors and increase lossy
                    current_settings['colors'] = max(
                        64, current_settings['colors'] - 32)
                    current_settings['lossy_value'] = min(
                        100, current_settings['lossy_value'] + 20)
                    current_settings['scale_factor'] = min(
                        1.0, max(0.1, (target_size / initial_size) ** 0.5))

                self.dev_logger.info(
                    f"\nAdjusting settings:"
                    f"\nScale: {current_settings['scale_factor']:.3f}"
                    f"\nColors: {current_settings['colors']}"
                    f"\nLossy: {current_settings['lossy_value']}"
                )

            # Process final results
            if successful_results:
                best_result = max(
                    successful_results,
                    # Prefer higher FPS, then smaller size
                    key=lambda x: (x['fps'], -x['size'])
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

    def _cleanup_temp_files(self, temp_files: list[Path]) -> None:
        """Clean up temporary files with improved error handling."""
        for temp_file in temp_files:
            try:
                if temp_file and temp_file.exists():
                    temp_file.unlink()
                TempFileManager.unregister(temp_file)
            except Exception as e:
                self.dev_logger.error(f"Failed to cleanup {temp_file}: {str(e)}")

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

    def process_all(self) -> List[Path]:
        """Enhanced process_all with better shutdown handling."""
        try:
            # Process videos
            for video_format in SUPPORTED_VIDEO_FORMATS:
                if self._shutdown_event.is_set():
                    break

                for video_file in Path(OUTPUT_DIR).glob(f'*{video_format}'):
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
            self._shutdown_event.set()
            self.dev_logger.warning("Gracefully shutting down...")
            self.cleanup_resources()
            raise
        finally:
            self.cleanup_resources()

        return self.failed_files

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
        """Handle interrupt signals."""
        self.dev_logger.warning("\nReceived interrupt signal. Cleaning up...")
        self._shutdown_event.set()
        self.ffmpeg._kill_current_process()
        self.cleanup_resources()
        sys.exit(0)


def validate_compression_settings(settings: Dict[str, Any]) -> None:
    required_keys = ['fps_range', 'colors', 'lossy_value']
    if not all(key in settings for key in required_keys):
        raise ValueError(f"Missing required keys: {required_keys}")


def process_gifs(compression_settings: Optional[Dict[str, Any]] = None) -> List[Path]:
    if compression_settings is not None:
        validate_compression_settings(compression_settings)
    processor = GIFProcessor(compression_settings)
    return processor.process_all()
