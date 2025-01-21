# gif_optimization.py
import logging
import os
import shutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
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

    def optimize_gif(self, input_gif: Path, output_gif: Path, settings: Dict) -> Tuple[float, bool]:
        """Enhanced GIF optimization with better handling of large files."""
        from logging_system import run_ffmpeg_command
        try:
            if not input_gif.exists():
                self.dev_logger.error(f"Input GIF not found: {input_gif}")
                return float('inf'), False

            original_size = self.get_file_size(input_gif, force_refresh=True)
            self.dev_logger.info(f"Starting optimization: {
                                 original_size:.2f}MB")

            # Build enhanced gifsicle command with better memory management
            cmd = [
                'gifsicle',
                '--optimize=3',
                '--colors', str(settings['colors']),
                '--lossy=' + str(settings['lossy_value']),
                '--no-conserve-memory',  # Use more memory for better optimization
                '--careful',
                '--threads=4'
            ]

            # Apply scaling if specified
            scale_factor = settings.get('scale_factor')
            if scale_factor and scale_factor < 1.0:
                from video_optimization import VideoProcessor
                width, height = VideoProcessor()._get_dimensions(input_gif)
                if width and height:
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    cmd.extend(['--resize', f'{new_width}x{new_height}'])
                    self.dev_logger.info(
                        f"Scaling: {width}x{height} → {new_width}x{new_height}"
                    )

            cmd.extend(['--batch', str(input_gif), '-o', str(output_gif)])

            # Run optimization with progress monitoring
            success = run_ffmpeg_command(cmd)
            if not success:
                self.dev_logger.error("Optimization command failed")
                return float('inf'), False

            # Verify result
            optimized_size = self.get_file_size(output_gif, force_refresh=True)
            if optimized_size > 0:
                reduction = ((original_size - optimized_size) /
                             original_size) * 100
                self.dev_logger.info(
                    f"Optimization complete: {
                        optimized_size:.2f}MB ({reduction:.1f}% reduction)"
                )

                should_continue = optimized_size <= self.compression_settings['min_size_mb']
                if not should_continue:
                    self.dev_logger.info(
                        f"Size ({optimized_size:.2f}MB) exceeds target "
                        f"({self.compression_settings['min_size_mb']}MB)"
                    )

                return optimized_size, should_continue
            else:
                self.dev_logger.error(
                    f"Invalid output size: {optimized_size}MB")
                return float('inf'), False

        except Exception as e:
            self.dev_logger.error(f"Optimization error: {str(e)}")
            return float('inf'), False


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

    @log_function_call
    def process_single_fps(self, args: Tuple) -> ProcessingResult:
        """Process GIF for a single FPS value with improved handling of large files."""
        file_path, output_path, is_video, fps, current_settings = args
        temp_dir = Path(TEMP_FILE_DIR)

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

            # Get initial file size for existing GIFs
            if not is_video:
                initial_size = self.get_file_size(
                    file_path, force_refresh=True)
                self.dev_logger.info(f"Processing GIF: {initial_size:.2f}MB")

                # For large GIFs, try direct optimization
                if initial_size > GIF_SIZE_TO_SKIP:
                    self.dev_logger.info(
                        f"Large GIF detected - attempting direct optimization")

                    # Copy to temp location
                    shutil.copy2(file_path, temp_gif)

                    # Use more aggressive optimization settings for large files
                    aggressive_settings = current_settings.copy()
                    aggressive_settings.update({
                        'colors': 64,
                        'lossy_value': 200,
                        'scale_factor': min(current_settings.get('scale_factor', 1.0), 0.5)
                    })

                    optimized_size, should_continue = self.optimize_gif(
                        temp_gif, final_gif, aggressive_settings)

                    if optimized_size < initial_size:
                        self.dev_logger.info(f"Direct optimization succeeded: {
                                             optimized_size:.2f}MB")
                        return ProcessingResult(fps, optimized_size, str(final_gif),
                                                ProcessingStatus.SUCCESS,
                                                f"Large GIF optimized: {optimized_size:.2f}MB")
                    else:
                        self.dev_logger.warning(f"GIF too large to optimize: {
                                                initial_size:.2f}MB")
                        return ProcessingResult(fps, initial_size, None,
                                                ProcessingStatus.SIZE_THRESHOLD_EXCEEDED,
                                                "File too large to optimize effectively")

            # For videos or smaller GIFs, proceed with normal processing
            if is_video:
                from video_optimization import VideoProcessor
                video_processor = VideoProcessor()
                width, height = video_processor._get_dimensions(file_path)

                if not width or not height:
                    return ProcessingResult(fps, float('inf'), None,
                                            ProcessingStatus.DIMENSION_ERROR,
                                            f"Could not determine dimensions for {file_path.name}")

                # Apply current scale factor to dimensions
                scale_factor = current_settings.get('scale_factor', 1.0)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                # Generate palette
                if not self.ffmpeg.generate_palette(Path(file_path), palette_path,
                                                    fps, (new_width, new_height), current_settings):
                    return ProcessingResult(fps, float('inf'), None,
                                            ProcessingStatus.PALETTE_ERROR,
                                            f"Palette generation failed for {file_path.name}")

                # Create initial GIF
                if not self.create_gif(Path(file_path), palette_path, temp_gif,
                                       fps, (new_width, new_height)):
                    return ProcessingResult(fps, float('inf'), None,
                                            ProcessingStatus.CONVERSION_ERROR,
                                            f"GIF creation failed for {file_path.name}")
            else:
                # For existing GIFs, copy to temp location
                shutil.copy2(file_path, temp_gif)

            # Ensure the file exists before optimization
            if not temp_gif.exists():
                return ProcessingResult(fps, float('inf'), None,
                                        ProcessingStatus.FILE_ERROR,
                                        f"Temporary GIF not found: {temp_gif}")

            # Get initial size
            initial_size = self.get_file_size(temp_gif, force_refresh=True)

            # Skip optimization if file is too large
            if initial_size > GIF_SIZE_TO_SKIP:
                self.dev_logger.info(
                    f"Skipping {temp_gif.name} - {initial_size:.1f}MB exceeds limit")
                return ProcessingResult(fps, initial_size, str(temp_gif),
                                        ProcessingStatus.SUCCESS,
                                        "Optimization skipped - file too large")

            # Apply optimization
            optimized_size, should_continue = self.optimize_gif(
                temp_gif, final_gif, current_settings)

            # If optimization produces a file larger than threshold, signal to stop processing
            if not should_continue:
                self._processing_cancelled.set()
                return ProcessingResult(fps, optimized_size, None,
                                        ProcessingStatus.SIZE_THRESHOLD_EXCEEDED,
                                        f"Stopping - {optimized_size:.1f}MB exceeds target")

            return ProcessingResult(fps, optimized_size, str(final_gif),
                                    ProcessingStatus.SUCCESS,
                                    f"Processed successfully - {optimized_size:.1f}MB")

        except Exception as e:
            self.dev_logger.error(f"Error processing {
                                  file_path.name}: {str(e)}")
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
                        self.dev_logger.error(f"Failed to cleanup {
                                              temp_file}: {str(e)}")

    @log_function_call
    def process_file(self, file_path: Path, output_path: Path, is_video: bool) -> None:
        """Process a single file with multiple optimization attempts and dynamic scaling."""
        try:
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
                        f"Pass {optimization_pass +
                                1}: Scale {current_scale:.2f} | "
                        f"Processing {file_path.name}"
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

                        try:
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
                        except KeyboardInterrupt:
                            self.dev_logger.info(
                                "Shutting down gracefully...")
                            # Cancel pending futures
                            for f in futures:
                                if not f.done():
                                    f.cancel()
                            # Allow cleanup to proceed
                            executor.shutdown(wait=True)
                            raise

                    # Check if we found a suitable result
                    if best_result and best_result.size <= self.compression_settings['min_size_mb']:
                        shutil.move(best_result.path, output_path)
                        self.user_logger.success(
                            f"Processed {file_path.name} | "
                            f"Size: {best_result.size:.1f}MB | "
                            f"Scale: {current_scale:.2f}"
                        )
                        return

                    current_scale = round(current_scale - SCALE_REDUCTION, 2)

                    if best_result:
                        self.dev_logger.info(
                            f"Best result: {best_result.size:.1f}MB | "
                            f"Target: {
                                self.compression_settings['min_size_mb']}MB | "
                            f"Scale: {current_scale:.2f}"
                        )

                optimization_pass += 1

            if best_result:
                shutil.move(best_result.path, output_path)
                self.user_logger.warning(
                    f"{file_path.name} | Best size: {
                        best_result.size:.1f}MB | "
                    f"Scale: {current_scale:.2f}"
                )
            else:
                if file_path not in self.failed_files:
                    self.failed_files.append(file_path)
                    self.user_logger.error(
                        f"Failed to process {file_path.name}")
        except KeyboardInterrupt:
            raise

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
