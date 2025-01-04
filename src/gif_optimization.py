# gif_optimization.py

import collections
import queue
import shutil
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import imagehash
import numpy as np
from PIL import Image

from default_config import (GIF_COMPRESSION, INPUT_DIR, OUTPUT_DIR,
                            SUPPORTED_VIDEO_FORMATS, TEMP_FILE_DIR)
from logging_system import run_ffmpeg_command
from temp_file_manager import TempFileManager


class ProcessingStatus(Enum):
    SUCCESS = auto()
    DIMENSION_ERROR = auto()
    PALETTE_ERROR = auto()
    CONVERSION_ERROR = auto()
    OPTIMIZATION_ERROR = auto()


@dataclass
class ProcessingResult:
    fps: int
    size: float
    path: Optional[str]
    status: ProcessingStatus


class LRUCache(OrderedDict):
    def __init__(self, maxsize=128):
        super().__init__()
        self.maxsize = maxsize

    def get(self, key, default=None):
        if key in self:
            self.move_to_end(key)
            return self[key]
        return default

    def __getitem__(self, key):
        value = super().__getitem__(key)
        self.move_to_end(key)
        return value

    def __setitem__(self, key, value):
        if key in self:
            self.move_to_end(key)
        super().__setitem__(key, value)
        if len(self) > self.maxsize:
            oldest = next(iter(self))
            del self[oldest]


class GIFProcessor:
    def __init__(self):
        self.failed_files = []
        self._init_directories()
        self.frame_hashes = {}
        self.processing_stats = collections.defaultdict(list)
        self._cache = LRUCache(maxsize=1000)

    def compute_frame_hash(self, frame):
        frame_bytes = frame.tobytes()
        cache_key = hash(frame_bytes)

        if cache_key in self._cache:
            return self._cache[cache_key]

        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        frame_hash = str(imagehash.average_hash(img))
        self._cache[cache_key] = frame_hash
        return frame_hash

    @staticmethod
    def _init_directories():
        for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_FILE_DIR]:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @staticmethod
    @lru_cache(maxsize=128)
    def has_audio_stream(video_path: Union[str, Path]) -> bool:
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

    def is_duplicate_frame(self, frame, threshold=0.95):
        frame_hash = self.compute_frame_hash(frame)

        # Check last 30 frames
        for existing_hash in list(self.frame_hashes.keys())[-30:]:
            if self.calculate_hash_similarity(frame_hash, existing_hash) > threshold:
                return True

        self.frame_hashes[frame_hash] = len(self.frame_hashes)
        return False

    def _cleanup_temp_files(self, temp_files: List[Path]) -> None:
        """Clean up temporary files after processing."""
        for temp_file in temp_files:
            try:
                if temp_file.exists():
                    temp_file.unlink()
                TempFileManager.unregister(temp_file)
            except Exception as e:
                logging.error(f"Failed to clean up temp file {temp_file}: {e}")

    @staticmethod
    def calculate_hash_similarity(hash1: str, hash2: str) -> float:
        return 1 - (bin(int(hash1, 16) ^ int(hash2, 16)).count('1') / 64)

    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> float:
        try:
            return Path(file_path).stat().st_size / (1024 * 1024)
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return float('inf')
        except Exception as e:
            logging.error(f"Error getting file size: {e}")
            return float('inf')

    def extract_unique_frames(self, video_path: Path, fps: int) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        unique_frames = []
        self.frame_hashes.clear()  # Reset frame hashes

        try:
            frame_interval = int(cap.get(cv2.CAP_PROP_FPS) / fps)
            frame_count = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    frame_hash = self.compute_frame_hash(frame)
                    if not any(self.calculate_hash_similarity(frame_hash, h) > 0.95
                               for h in list(self.frame_hashes.keys())[-30:]):
                        unique_frames.append(frame)
                        self.frame_hashes[frame_hash] = len(self.frame_hashes)

                frame_count += 1

            return unique_frames
        finally:
            cap.release()

    def generate_palette(self, file_path: Path, palette_path: Path,
                         fps: int, dimensions: Tuple[int, int]) -> bool:
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
                             fps}fps - ({dimensions[0]}x{dimensions[1]}) -> {file_path.name}")
                return True
            return False
        except Exception as e:
            logging.error(f"Palette generation failed: {e}")
            return False

    def create_gif(self, file_path: Path, palette_path: Path, output_gif: Path,
                   fps: int, dimensions: Tuple[int, int]) -> bool:
        try:
            # Extract frames and remove duplicates if it's a video
            if file_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS:
                unique_frames = self.extract_unique_frames(file_path, fps)
                if not unique_frames:
                    return False

                # Save unique frames to temporary video
                temp_video = TEMP_FILE_DIR / f"unique_{file_path.name}"
                self.save_frames_to_video(unique_frames, temp_video, fps)
                file_path = temp_video

            cmd = [
                'ffmpeg', '-i', str(file_path), '-i', str(palette_path),
                '-lavfi', f'fps={fps},scale={dimensions[0]}:{
                    dimensions[1]}:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:diff_mode=rectangle',
                '-y', str(output_gif)
            ]
            if run_ffmpeg_command(cmd):
                gif_size = self.get_file_size(output_gif)
                logging.info(f"Generated GIF: {
                             fps}fps - {gif_size:.2f}MB - ({dimensions[0]}x{dimensions[1]}) -> {file_path.stem}.gif")
                return True
            return False
        except Exception as e:
            logging.error(f"GIF creation failed: {e}")
            return False

    def extract_unique_frames(self, video_path: Path, fps: int) -> List[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        unique_frames = []
        self.frame_hashes = {}  # Reset frame hashes

        try:
            frame_count = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # Process every nth frame based on fps
                if frame_count % int(cap.get(cv2.CAP_PROP_FPS) / fps) == 0:
                    if not self.is_duplicate_frame(frame):
                        unique_frames.append(frame)

                frame_count += 1

            return unique_frames
        finally:
            cap.release()

    def save_frames_to_video(self, frames: List[np.ndarray], output_path: Path, fps: int):
        if not frames:
            return False

        height, width = frames[0].shape[:2]
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )

        try:
            for frame in frames:
                writer.write(frame)
            return True
        finally:
            writer.release()

    def optimize_gif(self, input_gif: Path, output_gif: Path,
                     colors: int, lossy: int) -> float:
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

    def process_single_fps(self, args: Tuple) -> ProcessingResult:
        file_path, output_path, is_video, fps, scale_factor = args

        try:
            with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as temp_file:
                temp_gif = Path(temp_file.name)

            width, height = get_video_dimensions(file_path)
            if not width or not height:
                return ProcessingResult(fps, float('inf'), None, ProcessingStatus.DIMENSION_ERROR)

            new_width = max(
                GIF_COMPRESSION['min_width'], int(width * scale_factor))
            new_height = max(
                GIF_COMPRESSION['min_height'], int(height * scale_factor))

            if is_video:
                result = self._process_video(
                    file_path, temp_gif, fps, new_width, new_height)
            else:
                result = self._process_gif(file_path, temp_gif)

            if result.status == ProcessingStatus.SUCCESS:
                self.processing_stats[file_path].append((fps, result.size))

            return result

        except Exception as e:
            logging.error(f"Processing failed: {e}", exc_info=True)
            return ProcessingResult(fps, float('inf'), None, ProcessingStatus.OPTIMIZATION_ERROR)
        finally:
            if 'temp_gif' in locals() and temp_gif.exists():
                temp_gif.unlink()

    def process_file(self, file_path: Path, output_path: Path, is_video: bool) -> None:
        scale_factor = 1.0
        best_result = None

        while scale_factor >= 0.1:
            fps_range = range(GIF_COMPRESSION['fps_range'][0],
                              GIF_COMPRESSION['fps_range'][1] - 1, -1)

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = [
                    executor.submit(self.process_single_fps,
                                    (str(file_path), str(output_path),
                                     is_video, fps, scale_factor))
                    for fps in fps_range
                ]

                results = [future.result() for future in as_completed(futures)
                           if future.result().status == ProcessingStatus.SUCCESS]

            if results:
                current_best = min(results, key=lambda x: x.size)
                if not best_result or current_best.size < best_result.size:
                    best_result = current_best

                if best_result.size <= GIF_COMPRESSION['min_size_mb']:
                    shutil.move(best_result.path, output_path)
                    if file_path in self.failed_files:
                        self.failed_files.remove(file_path)
                    return

            scale_factor = self._adjust_scale_factor(scale_factor,
                                                     results[0].size if results else float('inf'))

            if scale_factor < 0.1:
                if file_path not in self.failed_files:
                    self.failed_files.append(file_path)
                    logging.warning(
                        f"File {file_path.name} has reached minimum dimensions, will be skipped until next pass with lower settings.")
                break

    def _process_video(self, file_path: Path, temp_gif: Path, fps: int,
                       width: int, height: int) -> ProcessingResult:
        palette_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_palette:
                palette_path = Path(temp_palette.name)

            if not self.generate_palette(file_path, palette_path, fps, (width, height)):
                return ProcessingResult(fps, float('inf'), None, ProcessingStatus.PALETTE_ERROR)

            if not self.create_gif(file_path, palette_path, temp_gif, fps, (width, height)):
                return ProcessingResult(fps, float('inf'), None, ProcessingStatus.CONVERSION_ERROR)

            size = self.optimize_gif(temp_gif, temp_gif,
                                     GIF_COMPRESSION['colors'],
                                     GIF_COMPRESSION['lossy_value'])

            return ProcessingResult(fps, size, str(temp_gif), ProcessingStatus.SUCCESS)
        finally:
            if palette_path and palette_path.exists():
                palette_path.unlink()

    @staticmethod
    def _adjust_scale_factor(current_scale: float, result_size: float) -> float:
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
        try:
            # Log directories being checked
            logging.info(f"Checking input directory: {INPUT_DIR}")
            logging.info(f"Checking output directory: {OUTPUT_DIR}")

            # First process input GIFs
            logging.info(f"Found {len(gif_files)} GIF files")
            gif_files = list(Path(INPUT_DIR).glob('*.gif'))
            for gif_file in gif_files:
                output_gif = Path(OUTPUT_DIR) / f"{gif_file.stem}.gif"
                if not output_gif.exists():
                    logging.info(f"Processing GIF: {gif_file.name}")
                    if not self.process_file(gif_file, output_gif, is_video=False):
                        self.failed_files.append(gif_file)

            # Process compressed videos from output directory
            for video_format in SUPPORTED_VIDEO_FORMATS:
                video_files = list(Path(OUTPUT_DIR).glob(f'*{video_format}'))
                for video_file in video_files:
                    output_gif = Path(OUTPUT_DIR) / f"{video_file.stem}.gif"
                    if not output_gif.exists():
                        logging.info(f"Converting video to GIF: {
                                     video_file.name}")
                        if not self.process_file(video_file, output_gif, is_video=True):
                            self.failed_files.append(video_file)

            if not self.failed_files:
                logging.info("All files processed successfully")
            else:
                logging.warning(f"Failed to process {
                                len(self.failed_files)} files")

            return self.failed_files

        except Exception as e:
            logging.error(f"Batch processing failed: {e}", exc_info=True)
            return self.failed_files


def process_gifs() -> List[Path]:
    """Main entry point for GIF processing."""
    processor = GIFProcessor()
    return processor.process_all()
