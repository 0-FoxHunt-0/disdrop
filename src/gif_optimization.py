# gif_optimization.py
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

from default_config import (GIF_COMPRESSION, GIF_SIZE_TO_SKIP, INPUT_DIR,
                            OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS, TEMP_FILE_DIR)
from logging_system import log_function_call, run_ffmpeg_command
from temp_file_manager import TempFileManager
from video_optimization import get_video_dimensions


class ProcessingStatus(Enum):
    SUCCESS = "success"
    DIMENSION_ERROR = "dimension_error"
    PALETTE_ERROR = "palette_error"
    CONVERSION_ERROR = "conversion_error"
    OPTIMIZATION_ERROR = "optimization_error"


@dataclass
class ProcessingResult:
    fps: int
    size: float
    path: Optional[str]
    status: ProcessingStatus


class GIFProcessor:
    def __init__(self):
        self.failed_files = []
        self._init_directories()

    @staticmethod
    def _init_directories():
        """Ensure required directories exist."""
        for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_FILE_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    @lru_cache(maxsize=128)
    def has_audio_stream(video_path: Union[str, Path]) -> bool:
        """Check if video has audio stream with caching for performance."""
        try:
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

    def generate_palette(self, file_path: Path, palette_path: Path,
                         fps: int, dimensions: Tuple[int, int]) -> bool:
        """Generate color palette for GIF conversion."""
        try:
            cmd = [
                'ffmpeg', '-i', str(file_path),
                '-vf', f'fps={fps},scale={dimensions[0]
                                          }:{dimensions[1]}:flags=lanczos,palettegen',
                '-y', str(palette_path)
            ]
            success = run_ffmpeg_command(cmd)
            if success:
                logging.info(f"Generated palette: {
                             fps}fps - ({dimensions[0]}x{dimensions[1]}) - {file_path.name}")
                return True
            return False
        except Exception as e:
            logging.error(f"Palette generation failed: {e}")
            return False

    def create_gif(self, file_path: Path, palette_path: Path, output_gif: Path,
                   fps: int, dimensions: Tuple[int, int]) -> bool:
        """Create GIF with enhanced quality settings."""
        try:
            cmd = [
                'ffmpeg', '-i', str(file_path), '-i', str(palette_path),
                '-lavfi', f'fps={fps},scale={dimensions[0]}:{
                    dimensions[1]}:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:diff_mode=rectangle',
                '-y', str(output_gif)
            ]
            if run_ffmpeg_command(cmd):
                gif_size = self.get_file_size(output_gif)
                logging.info(f"Generated GIF: {
                             fps}fps - {gif_size:.2f}MB - ({dimensions[0]}x{dimensions[1]}) - {file_path.name}")
                return True
            return False
        except Exception as e:
            logging.error(f"GIF creation failed: {e}")
            return False

    def optimize_gif(self, input_gif: Path, output_gif: Path,
                     colors: int, lossy: int) -> float:
        """Optimize GIF with enhanced quality preservation."""
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

    def process_single_fps(self, args: Tuple) -> ProcessingResult:
        """Process GIF for a single FPS value with comprehensive error handling."""
        file_path, output_path, is_video, fps, scale_factor = args
        temp_dir = Path(TEMP_FILE_DIR)
        temp_gif = temp_dir / \
            f"{Path(output_path).stem}_{fps}_{scale_factor:.2f}.gif"
        TempFileManager.register(temp_gif)

        width, height = get_video_dimensions(file_path)
        if not width or not height:
            return ProcessingResult(fps, float('inf'), None, ProcessingStatus.DIMENSION_ERROR)

        new_width, new_height = int(
            width * scale_factor), int(height * scale_factor)
        if new_height < GIF_COMPRESSION['min_height'] or new_width < GIF_COMPRESSION['min_width']:
            return ProcessingResult(fps, float('inf'), None, ProcessingStatus.DIMENSION_ERROR)

        try:
            if not is_video:
                initial_size = self.optimize_gif(file_path, temp_gif,
                                                 GIF_COMPRESSION['colors'],
                                                 GIF_COMPRESSION['lossy_value'])
            else:
                palette_path = temp_dir / \
                    f"palette_{fps}_{scale_factor}_{os.getpid()}.png"
                TempFileManager.register(palette_path)

                if not self.generate_palette(Path(file_path), palette_path, fps, (new_width, new_height)):
                    return ProcessingResult(fps, float('inf'), None, ProcessingStatus.PALETTE_ERROR)

                if not self.create_gif(Path(file_path), palette_path, temp_gif, fps, (new_width, new_height)):
                    return ProcessingResult(fps, float('inf'), None, ProcessingStatus.CONVERSION_ERROR)

                initial_size = self.get_file_size(temp_gif)
                if palette_path.exists():
                    palette_path.unlink()

            if initial_size > GIF_SIZE_TO_SKIP:
                return ProcessingResult(fps, initial_size, str(temp_gif), ProcessingStatus.SUCCESS)

            optimized_size = self.optimize_gif(temp_gif, temp_gif,
                                               GIF_COMPRESSION['colors'],
                                               GIF_COMPRESSION['lossy_value'])

            return ProcessingResult(fps, optimized_size, str(temp_gif), ProcessingStatus.SUCCESS)

        except Exception as e:
            logging.error(f"Processing failed: {e}")
            return ProcessingResult(fps, float('inf'), None, ProcessingStatus.OPTIMIZATION_ERROR)

    def process_file(self, file_path: Path, output_path: Path, is_video: bool) -> None:
        """Process a single file with multiple FPS attempts using ThreadPoolExecutor."""
        scale_factor = 1.0
        while True:
            fps_range = range(max(GIF_COMPRESSION['fps_range']),
                              min(GIF_COMPRESSION['fps_range']) - 1, -1)

            with ThreadPoolExecutor() as executor:
                futures = [
                    executor.submit(self.process_single_fps,
                                    (str(file_path), str(output_path), is_video, fps, scale_factor))
                    for fps in fps_range
                ]

                results = []
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        if result.status == ProcessingStatus.SUCCESS:
                            results.append(result)
                    except Exception as e:
                        logging.error(f"Future execution failed: {e}")

            if results:
                best_result = min(results, key=lambda x: x.size)
                if best_result.size <= GIF_COMPRESSION['min_size_mb']:
                    shutil.move(best_result.path, output_path)
                    if file_path in self.failed_files:
                        self.failed_files.remove(file_path)
                    break

            scale_factor = self._adjust_scale_factor(scale_factor,
                                                     results[0].size if results else float('inf'))

            if scale_factor < 0.1:
                if file_path not in self.failed_files:
                    self.failed_files.append(file_path)
                break

    @staticmethod
    def _adjust_scale_factor(current_scale: float, result_size: float) -> float:
        """Dynamically adjust scale factor based on result size."""
        if result_size > 50:
            return current_scale * 0.3
        elif result_size > 40:
            return current_scale * 0.4
        elif result_size > 30:
            return current_scale * 0.5
        elif result_size > 20:
            return current_scale * 0.8
        elif result_size > 10:
            return current_scale * 0.9
        return current_scale * 0.95

    def process_all(self) -> List[Path]:
        """Process all files in input directory with enhanced error handling."""
        try:
            # Process videos from output directory
            for video_format in SUPPORTED_VIDEO_FORMATS:
                for video_file in Path(OUTPUT_DIR).glob(f'*{video_format}'):
                    output_gif = Path(OUTPUT_DIR) / f"{video_file.stem}.gif"
                    if not output_gif.exists():
                        self.process_file(
                            video_file, output_gif, is_video=True)

            # Process GIFs from input directory
            for gif_file in Path(INPUT_DIR).glob('*.gif'):
                output_gif = Path(OUTPUT_DIR) / f"{gif_file.stem}.gif"
                if not output_gif.exists():
                    self.process_file(gif_file, output_gif, is_video=False)

            return self.failed_files
        except Exception as e:
            logging.error(f"Batch processing failed: {e}")
            return self.failed_files


def process_gifs() -> List[Path]:
    """Main entry point for GIF processing."""
    processor = GIFProcessor()
    return processor.process_all()
