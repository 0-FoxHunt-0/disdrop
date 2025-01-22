# gif_optimization.py
import logging
import os
import shutil
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

    @staticmethod
    def has_audio_stream(video_path: Union[str, Path]) -> bool:
        """Check if video has audio stream."""
        try:
            from logging_system import run_ffmpeg_command
            result = run_ffmpeg_command([
                'ffprobe', '-v', 'error', '-select_streams', 'a',
                '-show_entries', 'stream=codec_type', '-of',
                'default=noprint_wrappers=1:nokey=1', str(video_path)
            ])
            return 'audio' in str(result)
        except Exception as e:
            logging.error(f"Error checking audio stream: {e}")
            return False

    @staticmethod
    def generate_palette(file_path: Path, palette_path: Path,
                         fps: int, dimensions: Tuple[int, int], settings: Dict) -> bool:
        try:
            from logging_system import run_ffmpeg_command
            cmd = [
                'ffmpeg', '-i', str(file_path),
                '-vf', f'fps={fps},scale={dimensions[0]
                                          }:{dimensions[1]}:flags=lanczos,palettegen',
                '-y', str(palette_path)
            ]
            success = run_ffmpeg_command(cmd)
            if success:
                if fps == settings['fps_range'][0]:
                    logging.info(f"Generated palettes | FPS: {settings['fps_range']} | "
                                 f"Resolution: {dimensions[0]}x{dimensions[1]}")
                return True
            return False
        except Exception as e:
            logging.error(f"Palette generation failed: {str(e)}")
            return False


class DynamicGIFOptimizer:
    """Handles dynamic GIF optimization with adaptive settings."""

    def __init__(self):
        self.dev_logger = logging.getLogger('developer')
        # Base settings to start with
        self.base_settings = {
            'colors': 256,
            'lossy_value': 20,
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
        self.dev_logger.info(f"Starting optimization: {
                             input_size:.2f}MB → Target: {target_size_mb}MB")

        # Initialize optimization parameters
        current_settings = self.base_settings.copy()
        best_result = {'size': float('inf'), 'settings': None}
        attempt = 0
        max_attempts = 10

        # Calculate initial scaling based on input size
        size_ratio = target_size_mb / input_size
        current_settings['scale_factor'] = min(
            1.0, max(0.1, size_ratio ** 0.5))

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
                '--optimize=3',
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
        """Dynamically adjust settings based on results."""
        new_settings = current.copy()

        # Calculate how far we are from target
        size_ratio = result_size / target_size

        # Adjust scale factor more aggressively if we're far from target
        if size_ratio > 2:
            new_settings['scale_factor'] *= 0.7
        elif size_ratio > 1.5:
            new_settings['scale_factor'] *= 0.85
        elif size_ratio > 1.2:
            new_settings['scale_factor'] *= 0.95

        # Adjust colors based on result size
        if size_ratio > 1.5 and new_settings['colors'] > 64:
            new_settings['colors'] = max(32, new_settings['colors'] // 2)
        elif size_ratio > 1.2 and new_settings['colors'] > 128:
            new_settings['colors'] = max(64, new_settings['colors'] - 64)

        # Increase lossy compression if needed
        if size_ratio > 1.2:
            new_settings['lossy_value'] = min(
                200, new_settings['lossy_value'] + 20)

        # Ensure minimum values
        new_settings['scale_factor'] = max(0.1, new_settings['scale_factor'])
        new_settings['colors'] = max(32, new_settings['colors'])

        return new_settings

    def _get_aggressive_settings(self, current: Dict, attempt: int) -> Dict:
        """Get more aggressive settings when normal adjustments aren't enough."""
        aggressive = current.copy()

        # More aggressive scaling
        aggressive['scale_factor'] *= 0.6

        # Reduce colors more dramatically
        aggressive['colors'] = max(32, aggressive['colors'] // 4)

        # Increase lossy compression significantly
        aggressive['lossy_value'] = min(200, aggressive['lossy_value'] * 2)

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

    def optimize_gif(self, input_gif: Path, output_gif: Path, settings: Dict) -> Tuple[float, bool]:
        """Enhanced GIF optimization using dynamic optimization."""
        try:
            if not input_gif.exists():
                self.dev_logger.error(f"Input GIF not found: {input_gif}")
                return float('inf'), False

            original_size = self.get_file_size(input_gif, force_refresh=True)
            self.dev_logger.info(f"Starting optimization: {
                                 original_size:.2f}MB")

            # Use the new dynamic optimizer
            target_size = self.compression_settings.get('min_size_mb', 15.0)
            optimized_size, success = self.dynamic_optimizer.optimize_gif(
                input_gif,
                output_gif,
                target_size
            )

            if success:
                reduction = ((original_size - optimized_size) /
                             original_size) * 100
                self.dev_logger.info(
                    f"Optimization complete: {
                        optimized_size:.2f}MB ({reduction:.1f}% reduction)"
                )
                return optimized_size, True
            else:
                self.dev_logger.error(f"Failed to optimize to target size: {
                                      optimized_size:.2f}MB")
                return optimized_size, False

        except Exception as e:
            self.dev_logger.error(f"Optimization error: {str(e)}")
            return float('inf'), False


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
        """Clean up all resources before shutdown."""
        self._shutdown_event.set()

        # Clean up file locks
        with self._file_locks_lock:
            self._file_locks.clear()

        # Clean up temporary files
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
        """Create GIF with enhanced quality settings and size verification."""
        from logging_system import run_ffmpeg_command
        try:
            cmd = [
                'ffmpeg', '-i', str(file_path), '-i', str(palette_path),
                '-lavfi', f'fps={fps},scale={dimensions[0]}:{
                    dimensions[1]}:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:diff_mode=rectangle',
                '-y', str(output_gif)
            ]

            if run_ffmpeg_command(cmd):
                # Wait for file to be completely written
                if not self.wait_for_file_completion(output_gif):
                    logging.error("Failed to verify GIF creation completion")
                    return False

                # Get accurate file size
                gif_size = self.get_file_size(output_gif, force_refresh=True)

                if gif_size > 0:
                    logging.info(
                        f"[{fps}fps] Generated GIF ({gif_size:.2f}MB) → Optimizing...")
                    logging.debug(f"GIF details: ({dimensions[0]}x{
                                  dimensions[1]}) -> {file_path.name}")
                    return True
                else:
                    logging.error(f"Invalid generated file size: {gif_size}MB")
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

        # Create unique temporary filenames
        temp_gif = temp_dir / f"{Path(output_path).stem}_{fps}.gif"
        final_gif = temp_dir / f"{Path(output_path).stem}_{fps}_optimized.gif"
        palette_path = temp_dir / \
            f"palette_{fps}_{current_settings.get('scale_factor', 1.0)}_{
                os.getpid()}.png"

        try:
            # Register files for cleanup
            for temp_file in [temp_gif, final_gif, palette_path]:
                TempFileManager.register(temp_file)

            if self._processing_cancelled.is_set():
                return ProcessingResult(fps, float('inf'), None,
                                        ProcessingStatus.SIZE_THRESHOLD_EXCEEDED,
                                        "Processing cancelled - size limit reached")

            # Handle large input GIFs differently than videos
            if not is_video:
                initial_size = self.get_file_size(
                    file_path, force_refresh=True)
                self._log_with_lock("info", f"Processing GIF: {
                                    initial_size:.2f}MB", file_id)

                # For large GIFs, apply aggressive optimization
                if initial_size > GIF_SIZE_TO_SKIP:
                    self._log_with_lock(
                        "info", "Large GIF detected - applying aggressive optimization", file_id)

                    # Copy to temp location
                    shutil.copy2(file_path, temp_gif)

                    # Calculate scale factor based on size
                    # The larger the file, the more we scale down
                    base_scale = 0.5  # Start at 50% for files just over the limit
                    size_ratio = GIF_SIZE_TO_SKIP / initial_size
                    scale_factor = min(base_scale, size_ratio)

                    # Use aggressive optimization settings for large files
                    aggressive_settings = current_settings.copy()
                    aggressive_settings.update({
                        'colors': 64,  # Reduce colors
                        'lossy_value': 200,  # Increase lossy compression
                        'scale_factor': scale_factor  # Apply calculated scale factor
                    })

                    # Log optimization strategy
                    self._log_with_lock("info",
                                        f"Applying aggressive optimization: scale={
                                            scale_factor:.2f}, "
                                        f"colors={
                                            aggressive_settings['colors']}, "
                                        f"lossy={
                                            aggressive_settings['lossy_value']}",
                                        file_id)

                    with self.size_check_lock:
                        optimized_size, should_continue = self.optimize_gif(
                            temp_gif, final_gif, aggressive_settings)

                    if optimized_size < initial_size:
                        reduction = (
                            (initial_size - optimized_size) / initial_size) * 100
                        self._log_with_lock("success",
                                            f"Optimization succeeded: {
                                                optimized_size:.2f}MB "
                                            f"({reduction:.1f}% reduction)",
                                            file_id)
                        return ProcessingResult(fps, optimized_size, str(final_gif),
                                                ProcessingStatus.SUCCESS,
                                                f"Large GIF optimized: {optimized_size:.2f}MB")

                    self._log_with_lock("warning",
                                        f"Failed to reduce GIF size significantly: {
                                            optimized_size:.2f}MB",
                                        file_id)
                    return ProcessingResult(fps, optimized_size, None,
                                            ProcessingStatus.OPTIMIZATION_ERROR,
                                            "Failed to achieve significant size reduction")

            # For videos, maintain the original size threshold logic
            elif is_video:
                file_size = self.get_file_size(file_path)
                if file_size > GIF_SIZE_TO_SKIP:
                    self._log_with_lock("warning",
                                        f"Skipping video conversion - {
                                            file_size:.1f}MB exceeds limit",
                                        file_id)
                    return ProcessingResult(fps, file_size, None,
                                            ProcessingStatus.SIZE_THRESHOLD_EXCEEDED,
                                            "Video too large to convert")

            # Rest of the original processing logic for normal-sized files
            if is_video:
                from video_optimization import VideoProcessor
                video_processor = VideoProcessor()
                width, height = video_processor._get_dimensions(file_path)

                if not width or not height:
                    return ProcessingResult(fps, float('inf'), None,
                                            ProcessingStatus.DIMENSION_ERROR,
                                            f"Could not determine dimensions for {Path(file_path).name}")

                # Apply current scale factor to dimensions
                scale_factor = current_settings.get('scale_factor', 1.0)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                self._log_with_lock("info",
                                    f"Scaling: {width}x{height} → {
                                        new_width}x{new_height}",
                                    file_id)

                # Generate palette and create GIF
                if not self.ffmpeg.generate_palette(Path(file_path), palette_path,
                                                    fps, (new_width, new_height), current_settings):
                    return ProcessingResult(fps, float('inf'), None,
                                            ProcessingStatus.PALETTE_ERROR,
                                            f"Palette generation failed for {Path(file_path).name}")

                if not self.create_gif(Path(file_path), palette_path, temp_gif,
                                       fps, (new_width, new_height)):
                    return ProcessingResult(fps, float('inf'), None,
                                            ProcessingStatus.CONVERSION_ERROR,
                                            f"GIF creation failed for {Path(file_path).name}")
            else:
                # For normal-sized GIFs, copy to temp location
                shutil.copy2(file_path, temp_gif)

            # Standard optimization for normal-sized files
            with self.size_check_lock:
                optimized_size, should_continue = self.optimize_gif(
                    temp_gif, final_gif, current_settings)

            if not should_continue:
                self._processing_cancelled.set()
                return ProcessingResult(fps, optimized_size, None,
                                        ProcessingStatus.SIZE_THRESHOLD_EXCEEDED,
                                        f"Stopping - {optimized_size:.1f}MB exceeds target")

            return ProcessingResult(fps, optimized_size, str(final_gif),
                                    ProcessingStatus.SUCCESS,
                                    f"Processed successfully - {optimized_size:.1f}MB")

        except Exception as e:
            self._log_with_lock("error",
                                f"Error processing {Path(file_path).name}: {
                                    str(e)}",
                                file_id)
            return ProcessingResult(fps, float('inf'), None,
                                    ProcessingStatus.OPTIMIZATION_ERROR,
                                    str(e))
        finally:
            # Cleanup temp files
            for temp_file in [temp_gif, palette_path]:
                if temp_file and temp_file.exists():
                    try:
                        temp_file.unlink()
                        TempFileManager.unregister(temp_file)
                    except Exception as e:
                        self._log_with_lock("error",
                                            f"Failed to cleanup {
                                                temp_file}: {str(e)}",
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
                            f"Skipping video {
                                file_path.name} - {file_size:.1f}MB exceeds size limit"
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
                        self.dev_logger.info(f"Successfully optimized GIF: {
                                             optimized_size:.2f}MB")
                    else:
                        self.dev_logger.error(f"Failed to optimize GIF to target size: {
                                              optimized_size:.2f}MB")
                        self.failed_files.append(file_path)

        except Exception as e:
            self.dev_logger.error(f"Fatal error processing {
                                  file_path.name}: {e}")
            self.failed_files.append(file_path)
        finally:
            self._cleanup_file_resources(file_path)

    def _process_video(self, file_path: Path, output_path: Path) -> None:
        """Process video files to GIF with parallel FPS processing using existing methods."""
        try:
            from video_optimization import VideoProcessor
            from concurrent.futures import ThreadPoolExecutor, as_completed

            video_processor = VideoProcessor()
            width, height = video_processor._get_dimensions(file_path)

            if not width or not height:
                self.dev_logger.error(
                    f"Could not determine dimensions for {file_path.name}")
                self.failed_files.append(file_path)
                return

            # Generate FPS values to try (inclusive range)
            min_fps, max_fps = self.compression_settings['fps_range']
            fps_values = range(min_fps, max_fps + 1)

            with self.logging_lock:
                self.dev_logger.info(f"Starting parallel processing for {
                                     len(fps_values)} FPS values")

            target_size = self.compression_settings.get('min_size_mb', 15.0)
            successful_results = []

            # Prepare processing arguments
            processing_args = [
                (file_path, output_path, True, fps, self.compression_settings)
                for fps in fps_values
            ]

            # Use ThreadPoolExecutor for parallel processing
            with ThreadPoolExecutor(max_workers=min(4, len(fps_values))) as executor:
                future_to_fps = {
                    executor.submit(self._wrapped_process_single_fps, args): args[3]
                    for args in processing_args
                }

                for future in as_completed(future_to_fps):
                    fps = future_to_fps[future]
                    try:
                        result = future.result()

                        if result.status == ProcessingStatus.SUCCESS and result.path:
                            successful_results.append({
                                'size': result.size,
                                'fps': fps,
                                'path': result.path
                            })
                            with self.logging_lock:
                                self.dev_logger.info(f"Result at {fps} FPS: {
                                                     result.size:.2f}MB")

                    except Exception as e:
                        with self.logging_lock:
                            self.dev_logger.error(f"Error processing {
                                                  fps} FPS: {str(e)}")

            # Handle the results
            if successful_results:
                try:
                    # Filter results under size limit and sort by FPS (highest first)
                    valid_results = [
                        r for r in successful_results if r['size'] <= target_size]
                    if valid_results:
                        best_result = max(
                            valid_results, key=lambda x: x['fps'])

                        # Ensure output directory exists
                        output_path.parent.mkdir(parents=True, exist_ok=True)

                        # Copy the best result to the output path
                        if best_result['path'] and Path(best_result['path']).exists():
                            with self.logging_lock:
                                self.dev_logger.info(
                                    f"Saving best result: {
                                        best_result['fps']} FPS, "
                                    f"{best_result['size']:.2f}MB → {
                                        output_path}"
                                )

                            # Ensure any existing file is removed
                            if output_path.exists():
                                output_path.unlink()

                            # Copy the best result to the final destination
                            shutil.copy2(best_result['path'], output_path)

                            # Verify the copy was successful
                            if output_path.exists() and output_path.stat().st_size > 0:
                                self.dev_logger.info(
                                    f"Successfully saved optimized GIF to {output_path}")
                            else:
                                raise Exception("Failed to verify output file")
                    else:
                        self.dev_logger.error(
                            f"No results under target size of {target_size}MB")
                        self.failed_files.append(file_path)

                    # Clean up all temporary results
                    for result in successful_results:
                        try:
                            if result['path'] and Path(result['path']).exists():
                                Path(result['path']).unlink()
                                TempFileManager.unregister(
                                    Path(result['path']))
                        except Exception as e:
                            self.dev_logger.warning(f"Failed to cleanup temporary file {
                                                    result['path']}: {e}")

                except Exception as e:
                    with self.logging_lock:
                        self.dev_logger.error(f"Error saving best result: {e}")
                    self.failed_files.append(file_path)

            else:
                with self.logging_lock:
                    self.dev_logger.error(
                        f"No successful results found for {file_path.name}")
                self.failed_files.append(file_path)

        except Exception as e:
            with self.logging_lock:
                self.dev_logger.error(f"Error processing video {
                                      file_path.name}: {e}")
            self.failed_files.append(file_path)

            # Ensure cleanup in case of error
            try:
                for result in successful_results:
                    if result['path'] and Path(result['path']).exists():
                        Path(result['path']).unlink()
                        TempFileManager.unregister(Path(result['path']))
            except Exception as cleanup_error:
                self.dev_logger.error(f"Error during cleanup: {cleanup_error}")

    def _cleanup_temp_files(self, temp_files: list[Path]) -> None:
        """Clean up temporary files with improved error handling."""
        for temp_file in temp_files:
            try:
                if temp_file and temp_file.exists():
                    temp_file.unlink()
                TempFileManager.unregister(temp_file)
            except Exception as e:
                self.dev_logger.error(f"Failed to cleanup {
                                      temp_file}: {str(e)}")

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
        """Process all files with improved shutdown handling."""
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


def validate_compression_settings(settings: Dict[str, Any]) -> None:
    required_keys = ['fps_range', 'colors', 'lossy_value']
    if not all(key in settings for key in required_keys):
        raise ValueError(f"Missing required keys: {required_keys}")


def process_gifs(compression_settings: Optional[Dict[str, Any]] = None) -> List[Path]:
    if compression_settings is not None:
        validate_compression_settings(compression_settings)
    processor = GIFProcessor(compression_settings)
    return processor.process_all()
