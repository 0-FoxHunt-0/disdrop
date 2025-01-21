# gif_optimization.py
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

from cachetools import TTLCache

from default_config import (GIF_COMPRESSION, GIF_PASS_OVERS, GIF_SIZE_TO_SKIP,
                            INPUT_DIR, OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS,
                            TEMP_FILE_DIR)
from logging_system import log_function_call
from temp_file_manager import TempFileManager


class ProcessingStatus(Enum):
    SUCCESS = "success"
    DIMENSION_ERROR = "dimension_error"
    PALETTE_ERROR = "palette_error"
    CONVERSION_ERROR = "conversion_error"
    OPTIMIZATION_ERROR = "optimization_error"
    FILE_ERROR = "file_error"


@dataclass
class ProcessingResult:
    fps: int
    size: float
    path: Optional[str]
    status: ProcessingStatus
    message: str = ""


class GIFOptimizationError(Exception):
    """Base exception for GIF optimization errors."""
    pass


class FileProcessor:
    """Base class for file processing operations."""

    def __init__(self):
        self.file_size_cache = TTLCache(maxsize=1000, ttl=3600)

    @staticmethod
    @lru_cache(maxsize=128)
    def get_file_size(file_path: Union[str, Path]) -> float:
        """Get file size in MB with error handling."""
        try:
            return Path(file_path).stat().st_size / (1024 * 1024)
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
                         fps: int, dimensions: Tuple[int, int]) -> bool:
        """Generate color palette for GIF conversion."""
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
                logging.info(f"Generated palette: {
                             fps}fps - ({dimensions[0]}x{dimensions[1]}) -> {file_path.name}")
                return True
            return False
        except Exception as e:
            logging.error(f"Palette generation failed: {e}")
            return False


class GIFOptimizer(FileProcessor):
    """Handles GIF optimization operations."""

    def __init__(self, compression_settings: Dict = None):
        super().__init__()
        self.compression_settings = compression_settings or GIF_COMPRESSION
        self.failed_files = []
        self.dev_logger = logging.getLogger('developer')
        self._init_directories()

    def _init_directories(self):
        """Initialize required directories."""
        for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_FILE_DIR]:
            self.ensure_directory(Path(directory))

    def optimize_gif(self, input_gif: Path, output_gif: Path, settings: Dict) -> float:
        """
        Optimize GIF using gifsicle with enhanced settings and capabilities.

        Args:
            input_gif: Path to input GIF file
            output_gif: Path to output GIF file
            settings: Dictionary containing optimization settings:
                     - colors: Number of colors (int)
                     - lossy_value: Lossy compression value (int)
                     - scale_factor: Optional scaling factor (float)

        Returns:
            float: Size of optimized GIF in MB, or float('inf') if optimization fails
        """
        from logging_system import run_ffmpeg_command
        try:
            # Get original size for comparison
            original_size = self.get_file_size(input_gif)
            self.dev_logger.info(f"Original GIF size: {original_size:.2f}MB")

            # Build gifsicle command with base optimization settings
            cmd = [
                'gifsicle',
                '--optimize=3',
                '--colors', str(settings['colors']),
                '--lossy=' + str(settings['lossy_value']),
                '--no-conserve-memory',
                '--careful',
                '--threads=4'
            ]

            # Apply scaling if specified
            scale_factor = settings.get('scale_factor')
            if scale_factor and scale_factor < 1.0:
                # Calculate new dimensions
                from video_optimization import VideoProcessor
                width, height = VideoProcessor()._get_dimensions(input_gif)
                if width and height:
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    cmd.extend(['--resize', f'{new_width}x{new_height}'])
                    self.dev_logger.info(
                        f"Applying scale factor {scale_factor:.2f} "
                        f"({width}x{height} -> {new_width}x{new_height})"
                    )

            # Add input and output paths
            cmd.extend([str(input_gif), '-o', str(output_gif)])

            # Run optimization
            success = run_ffmpeg_command(cmd)
            if not success:
                self.dev_logger.error("Gifsicle command failed")
                return float('inf')

            # Clear the file size cache to ensure we get the actual new size
            self.file_size_cache.clear()

            # Force a file system sync to ensure the file is completely written
            if hasattr(os, 'sync'):
                os.sync()

            # Wait a brief moment for file system to stabilize
            time.sleep(0.1)

            # Get the actual final size
            optimized_size = self.get_file_size(output_gif)
            compression_ratio = (
                original_size - optimized_size) / original_size * 100
            self.dev_logger.info(
                f"Optimized GIF size: {optimized_size:.2f}MB "
                f"(reduced by {compression_ratio:.1f}%)"
            )
            return optimized_size

        except Exception as e:
            self.dev_logger.error(f"GIF optimization failed: {e}")
            return float('inf')


class GIFProcessor(GIFOptimizer):
    """Main GIF processing class combining optimization and conversion."""

    def __init__(self, compression_settings: Dict = None):
        super().__init__(compression_settings)
        self.ffmpeg = FFmpegHandler()
        self.dev_logger = logging.getLogger('developer')
        self.user_logger = logging.getLogger('user')
        self._processing_cancelled = threading.Event()

    def create_gif(self, file_path: Path, palette_path: Path, output_gif: Path,
                   fps: int, dimensions: Tuple[int, int]) -> bool:
        """Create GIF with enhanced quality settings."""
        from logging_system import run_ffmpeg_command
        try:
            cmd = [
                'ffmpeg', '-i', str(file_path), '-i', str(palette_path),
                '-lavfi', f'fps={fps},scale={dimensions[0]}:{
                    dimensions[1]}:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:diff_mode=rectangle',
                '-y', str(output_gif)
            ]
            if run_ffmpeg_command(cmd):
                gif_size = self.get_file_size(output_gif)
                self.user_logger.info(
                    f"Generated GIF: {fps}fps - {gif_size:.2f}MB")
                self.dev_logger.debug(f"GIF details: ({dimensions[0]}x{
                                      dimensions[1]}) -> {file_path.name}")
                return True
            return False
        except Exception as e:
            self.dev_logger.error(f"GIF creation failed: {e}")
            return False

    @log_function_call
    def process_single_fps(self, args: Tuple) -> ProcessingResult:
        """Process GIF for a single FPS value with comprehensive error handling."""
        file_path, output_path, is_video, fps, current_settings = args
        temp_dir = Path(TEMP_FILE_DIR)
        temp_gif = temp_dir / f"{Path(output_path).stem}_{fps}.gif"
        final_gif = temp_dir / f"{Path(output_path).stem}_{fps}_optimized.gif"
        TempFileManager.register(temp_gif)
        TempFileManager.register(final_gif)
        palette_path = None

        try:
            if self._processing_cancelled.is_set():
                return ProcessingResult(fps, float('inf'), None,
                                        ProcessingStatus.OPTIMIZATION_ERROR,
                                        "Processing cancelled due to size threshold")

            # Initial GIF creation
            if is_video:
                # Create palette and initial GIF
                from video_optimization import VideoProcessor
                video_processor = VideoProcessor()
                width, height = video_processor._get_dimensions(file_path)

                if not width or not height:
                    return ProcessingResult(fps, float('inf'), None,
                                            ProcessingStatus.DIMENSION_ERROR,
                                            "Failed to get video dimensions")

                # Apply current scale factor to dimensions
                scale_factor = current_settings.get('scale_factor', 1.0)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                # Generate palette
                palette_path = temp_dir / \
                    f"palette_{fps}_{scale_factor}_{os.getpid()}.png"
                TempFileManager.register(palette_path)

                if not self.ffmpeg.generate_palette(Path(file_path), palette_path,
                                                    fps, (new_width, new_height)):
                    return ProcessingResult(fps, float('inf'), None,
                                            ProcessingStatus.PALETTE_ERROR,
                                            "Failed to generate palette")

                # Create initial GIF
                if not self.create_gif(Path(file_path), palette_path, temp_gif,
                                       fps, (new_width, new_height)):
                    return ProcessingResult(fps, float('inf'), None,
                                            ProcessingStatus.CONVERSION_ERROR,
                                            "Failed to create GIF")
            else:
                # For existing GIFs, copy to temp location
                shutil.copy(file_path, temp_gif)

            # Get initial size
            initial_size = self.get_file_size(temp_gif)
            self.dev_logger.info(f"Initial GIF size: {initial_size:.2f}MB")

            if initial_size > GIF_SIZE_TO_SKIP:
                self.dev_logger.warning(f"Skipping optimization for {
                                        temp_gif.name} due to size ({initial_size:.2f}MB)")
                return ProcessingResult(fps, initial_size, str(temp_gif),
                                        ProcessingStatus.SUCCESS,
                                        "Skipped optimization due to size")

            # Apply gifsicle optimization
            optimized_size = GIFOptimizer(self.compression_settings).optimize_gif(
                temp_gif, final_gif, current_settings)

            # Check if we need to cancel processing of higher FPS
            if optimized_size > self.compression_settings['min_size_mb']:
                self._processing_cancelled.set()
                return ProcessingResult(fps, optimized_size, None,
                                        ProcessingStatus.OPTIMIZATION_ERROR,
                                        "Size threshold exceeded")

            return ProcessingResult(fps, optimized_size, str(final_gif),
                                    ProcessingStatus.SUCCESS,
                                    "Successfully processed")

        except Exception as e:
            self.dev_logger.error(f"Processing failed: {e}")
            return ProcessingResult(fps, float('inf'), None,
                                    ProcessingStatus.OPTIMIZATION_ERROR,
                                    f"Processing failed: {str(e)}")

        finally:
            for temp_file in [temp_gif, palette_path]:
                if temp_file and Path(temp_file).exists():
                    TempFileManager.unregister(temp_file)
                    Path(temp_file).unlink()

    @log_function_call
    def process_file(self, file_path: Path, output_path: Path, is_video: bool) -> None:
        """Process a single file with multiple optimization attempts and dynamic scaling."""
        optimization_pass = 0
        MIN_SCALE = 0.1
        SCALE_REDUCTION = 0.1  # Reduce scale by 0.1 in each iteration

        while optimization_pass < len(GIF_PASS_OVERS):
            current_settings = GIF_PASS_OVERS[optimization_pass].copy()
            initial_scale = current_settings.get('scale_factor', 0.8)
            current_scale = initial_scale

            while current_scale >= MIN_SCALE:
                current_settings['scale_factor'] = current_scale
                fps_range = range(max(current_settings['fps_range']),
                                  min(current_settings['fps_range']) - 1, -1)

                self.dev_logger.info(
                    f"Starting optimization pass {optimization_pass + 1} "
                    f"with scale factor {current_scale:.2f}"
                )

                best_result = None
                self._processing_cancelled.clear()

                with ThreadPoolExecutor() as executor:
                    futures = []
                    for fps in fps_range:
                        future = executor.submit(
                            self.process_single_fps,
                            (str(file_path), str(output_path),
                             is_video, fps, current_settings)
                        )
                        futures.append(future)

                    for future in as_completed(futures):
                        try:
                            result = future.result()

                            if self._processing_cancelled.is_set():
                                for f in futures:
                                    if not f.done():
                                        f.cancel()
                                break

                            if result.status == ProcessingStatus.SUCCESS:
                                if best_result is None or result.size < best_result.size:
                                    best_result = result

                        except Exception as e:
                            self.dev_logger.error(
                                f"Future execution failed: {e}")

                # Check if we found a suitable result
                if best_result and best_result.size <= self.compression_settings['min_size_mb']:
                    shutil.move(best_result.path, output_path)
                    self.user_logger.info(
                        f"Successfully processed {file_path.name} with size {
                            best_result.size:.2f}MB "
                        f"(scale={current_scale:.2f})"
                    )
                    return

                # Reduce scale for next iteration
                current_scale = round(current_scale - SCALE_REDUCTION, 2)

                # If we have a best result, log it
                if best_result:
                    self.dev_logger.info(
                        f"Best result at scale {current_scale:.2f}: {
                            best_result.size:.2f}MB "
                        f"(target: {
                            self.compression_settings['min_size_mb']}MB)"
                    )

            # Move to next optimization pass if all scales failed
            optimization_pass += 1

        # If we exhausted all passes, use the best result we found (if any)
        if best_result:
            shutil.move(best_result.path, output_path)
            self.user_logger.warning(
                f"Best achievable size for {file_path.name} was {
                    best_result.size:.2f}MB "
                f"(scale={current_scale:.2f})"
            )
        else:
            if file_path not in self.failed_files:
                self.failed_files.append(file_path)
                self.user_logger.warning(f"Failed to process {file_path.name}")

    @log_function_call
    def process_all(self) -> List[Path]:
        """Process all files in input directory."""
        try:
            # Process videos
            for video_format in SUPPORTED_VIDEO_FORMATS:
                for video_file in Path(OUTPUT_DIR).glob(f'*{video_format}'):
                    output_gif = Path(OUTPUT_DIR) / f"{video_file.stem}.gif"
                    if not output_gif.exists():
                        self.process_file(
                            video_file, output_gif, is_video=True)

            # Process GIFs
            for gif_file in Path(INPUT_DIR).glob('*.gif'):
                output_gif = Path(OUTPUT_DIR) / f"{gif_file.stem}.gif"
                if not output_gif.exists():
                    self.process_file(gif_file, output_gif, is_video=False)

            return self.failed_files

        except Exception as e:
            self.dev_logger.error(f"Batch processing failed: {e}")
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
