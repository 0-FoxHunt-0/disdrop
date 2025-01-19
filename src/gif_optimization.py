# gif_optimization.py
import logging
import os
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union, Dict

from default_config import (GIF_COMPRESSION, GIF_PASS_OVERS, GIF_SIZE_TO_SKIP,
                            INPUT_DIR, OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS, TEMP_FILE_DIR)
from logging_system import log_function_call
from temp_file_manager import TempFileManager
from video_optimization import VideoProcessor


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
        self._init_directories()

    def _init_directories(self):
        """Initialize required directories."""
        for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_FILE_DIR]:
            self.ensure_directory(Path(directory))

    def optimize_gif(self, input_gif: Path, output_gif: Path,
                     colors: int, lossy: int) -> float:
        """Optimize GIF using gifsicle."""
        from logging_system import run_ffmpeg_command
        logging.info(f"Optimizing GIF: {input_gif.name}")
        try:
            cmd = [
                'gifsicle', '--optimize=3', '--colors', str(colors),
                '--lossy=' + str(lossy), '--no-conserve-memory',
                '--careful', '--threads=4',
                str(input_gif), '-o', str(output_gif)
            ]
            run_ffmpeg_command(cmd)
            size = self.get_file_size(output_gif)
            logging.info(f"Optimized GIF to: {size:.2f}MB")
            return size
        except Exception as e:
            logging.error(f"GIF optimization failed: {e}")
            return float('inf')


class GIFProcessor(GIFOptimizer):
    """Main GIF processing class combining optimization and conversion."""

    def __init__(self, compression_settings: Dict = None):
        super().__init__(compression_settings)
        self.ffmpeg = FFmpegHandler()
        self.dev_logger = logging.getLogger('developer')
        self.user_logger = logging.getLogger('user')

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
        file_path, output_path, is_video, fps, scale_factor = args
        temp_dir = Path(TEMP_FILE_DIR)
        temp_gif = temp_dir / \
            f"{Path(output_path).stem}_{fps}_{scale_factor:.2f}.gif"
        TempFileManager.register(temp_gif)

        try:
            width, height = VideoProcessor._get_dimensions(file_path)
            if not width or not height:
                return ProcessingResult(fps, float('inf'), None,
                                        ProcessingStatus.DIMENSION_ERROR,
                                        "Failed to get video dimensions")

            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            if new_height < self.compression_settings['min_height'] or \
               new_width < self.compression_settings['min_width']:
                return ProcessingResult(fps, float('inf'), None,
                                        ProcessingStatus.DIMENSION_ERROR,
                                        "Dimensions too small")

            if not is_video:
                initial_size = self.optimize_gif(file_path, temp_gif,
                                                 self.compression_settings['colors'],
                                                 self.compression_settings['lossy_value'])
            else:
                palette_path = temp_dir / \
                    f"palette_{fps}_{scale_factor}_{os.getpid()}.png"
                TempFileManager.register(palette_path)

                if not self.ffmpeg.generate_palette(Path(file_path), palette_path,
                                                    fps, (new_width, new_height)):
                    return ProcessingResult(fps, float('inf'), None,
                                            ProcessingStatus.PALETTE_ERROR,
                                            "Failed to generate palette")

                if not self.create_gif(Path(file_path), palette_path, temp_gif,
                                       fps, (new_width, new_height)):
                    return ProcessingResult(fps, float('inf'), None,
                                            ProcessingStatus.CONVERSION_ERROR,
                                            "Failed to create GIF")

                initial_size = self.get_file_size(temp_gif)
                if palette_path.exists():
                    palette_path.unlink()

            if initial_size > GIF_SIZE_TO_SKIP:
                self.dev_logger.warning(f"Skipping optimization for {
                                        temp_gif.name} due to size ({initial_size:.2f}MB)")
                return ProcessingResult(fps, initial_size, str(temp_gif),
                                        ProcessingStatus.SUCCESS,
                                        "Skipped optimization due to size")

            optimized_size = self.optimize_gif(temp_gif, temp_gif,
                                               self.compression_settings['colors'],
                                               self.compression_settings['lossy_value'])

            return ProcessingResult(fps, optimized_size, str(temp_gif),
                                    ProcessingStatus.SUCCESS,
                                    "Successfully processed")

        except Exception as e:
            self.dev_logger.error(f"Processing failed: {e}")
            return ProcessingResult(fps, float('inf'), None,
                                    ProcessingStatus.OPTIMIZATION_ERROR,
                                    f"Processing failed: {str(e)}")

        finally:
            TempFileManager.unregister(temp_gif)
            if palette_path.exists():
                TempFileManager.unregister(palette_path)
                palette_path.unlink()

    @log_function_call
    def process_file(self, file_path: Path, output_path: Path, is_video: bool) -> None:
        """Process a single file with multiple optimization attempts."""
        scale_factor = 1.0
        optimization_pass = 0

        while True:
            settings = (GIF_PASS_OVERS[optimization_pass]
                        if optimization_pass < len(GIF_PASS_OVERS)
                        else GIF_PASS_OVERS[-1])

            fps_range = range(max(settings['fps_range']),
                              min(settings['fps_range']) - 1, -1)

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.process_single_fps,
                                    (str(file_path), str(output_path),
                                     is_video, fps, scale_factor))
                    for fps in fps_range
                ]

                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result.status == ProcessingStatus.SUCCESS:
                            results.append(result)
                    except Exception as e:
                        self.dev_logger.error(f"Future execution failed: {e}")

            if results:
                best_result = min(results, key=lambda x: x.size)
                if best_result.size <= self.compression_settings['min_size_mb']:
                    shutil.move(best_result.path, output_path)
                    self.user_logger.info(
                        f"Successfully processed {file_path.name}")
                    if file_path in self.failed_files:
                        self.failed_files.remove(file_path)
                    return

            optimization_pass += 1
            if optimization_pass >= len(GIF_PASS_OVERS):
                if file_path not in self.failed_files:
                    self.failed_files.append(file_path)
                    self.user_logger.warning(
                        f"Failed to process {file_path.name}")
                break

            scale_factor = settings.get('scale_factor', scale_factor)

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


def process_gifs(compression_settings: Dict = None) -> List[Path]:
    """Main entry point for GIF processing with optional custom settings."""
    processor = GIFProcessor(compression_settings)
    return processor.process_all()

# Backwards compatibility


def legacy_process_gifs() -> List[Path]:
    """Maintains backwards compatibility with old implementation."""
    return process_gifs()
