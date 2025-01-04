# video_optimization.py

import os
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import stats

from default_config import (INPUT_DIR, OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS,
                            TEMP_FILE_DIR, VIDEO_COMPRESSION)
from logging_system import log_function_call, logging, run_ffmpeg_command
from temp_file_manager import TempFileManager


class VideoSizePredictor:
    def __init__(self):
        self.compression_history = []
        self.size_reduction_rate = None
        self.initial_file_size = None
        self.reset_threshold = 0.5  # Reset if new file size differs by >50%

    def _update_reduction_rate(self):
        if len(self.compression_history) >= 2:
            x = np.array([crf for crf, _ in self.compression_history])
            y = np.array([size for _, size in self.compression_history])
            slope, _, r_value, _, _ = stats.linregress(x, y)

            # Only use the reduction rate if we have a good fit
            if abs(r_value) > 0.7:
                self.size_reduction_rate = abs(slope)
            else:
                self.size_reduction_rate = None
        else:
            self.size_reduction_rate = None

    def update(self, crf: int, size: float):
        # Check if we need to reset history based on size difference
        if self.initial_file_size is None:
            self.initial_file_size = size
        elif abs(size - self.initial_file_size) / max(size, self.initial_file_size) > self.reset_threshold:
            logging.info(
                "File size differs significantly from previous file. Resetting compression history.")
            self.compression_history = []
            self.size_reduction_rate = None
            self.initial_file_size = size

        self.compression_history.append((crf, size))
        self.compression_history = sorted(
            # Keep last 5 points
            self.compression_history, key=lambda x: x[0])[-5:]
        self._update_reduction_rate()

    def predict_target_crf(self, current_size: float, target_size: float) -> Optional[int]:
        if not self.size_reduction_rate:
            return None

        size_difference = current_size - target_size

        # Calculate conservative CRF increase based on file size ratio
        size_ratio = current_size / target_size
        if size_ratio > 2:
            # For large size differences, use more conservative steps
            needed_crf_increase = min(
                int(size_difference / (self.size_reduction_rate * 2)), 8)
        else:
            # For smaller differences, use normal calculation but cap the increase
            needed_crf_increase = min(
                int(size_difference / self.size_reduction_rate), 4)

        current_crf = self.compression_history[-1][0]
        predicted_crf = current_crf + max(2, needed_crf_increase)
        predicted_crf = min(51, max(18, predicted_crf))

        # Add safety check for large jumps
        if predicted_crf - current_crf > 8:
            predicted_crf = current_crf + 8

        logging.info(f"""
            CRF prediction:
            - Current size: {current_size:.2f}MB
            - Target size: {target_size:.2f}MB
            - Size ratio: {size_ratio:.2f}
            - Size difference: {size_difference:.2f}MB
            - Conservative CRF increase: {needed_crf_increase}
            - Current CRF: {current_crf}
            - Predicted CRF: {predicted_crf}
        """)

        return predicted_crf


class VideoProcessor:
    def __init__(self, gpu_supported: bool = False):
        self.gpu_supported = gpu_supported
        self.size_predictor = VideoSizePredictor()
        self.failed_files = []
        logging.info(f"Initialized VideoProcessor with GPU support: {
                     gpu_supported}")

    def _get_encoder_settings(self) -> Dict:
        return {
            'gpu': {
                'codec': 'h264_nvenc',
                'preset': 'p7',
                'extra_params': ['-rc', 'vbr']
            },
            'cpu': {
                'codec': 'libx264',
                'preset': 'veryslow',
                'extra_params': []
            }
        }[('gpu' if self.gpu_supported else 'cpu')]

    def _handle_compression_error(self, video_path: Path, error: Exception) -> None:
        """Handle compression errors and add to failed files."""
        logging.error(f"Failed to compress {video_path}: {error}")
        if video_path not in self.failed_files:
            self.failed_files.append(video_path)

    @staticmethod
    def get_video_info(video_path: Path) -> Tuple[Optional[int], Optional[int], Optional[float]]:
        try:
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                   '-show_entries', 'stream=width,height,r_frame_rate',
                   '-of', 'json', str(video_path)]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                stream = data['streams'][0]
                num, den = map(int, stream['r_frame_rate'].split('/'))
                return stream['width'], stream['height'], num/den
            return None, None, None
        except Exception as e:
            logging.error(f"Failed to get video info: {e}")
            return None, None, None

    def compress_video(self, input_path: Path, output_path: Path,
                       scale_factor: float, crf: int) -> Tuple[bool, float]:
        width, height, fps = self.get_video_info(input_path)
        if not all((width, height, fps)):
            logging.error(f"Failed to get video info for {input_path}")
            return False, float('inf')

        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        new_width += new_width % 2
        new_height += new_height % 2

        target_bitrate = min(
            4000, int(4000 * (new_width * new_height) / (1920 * 1080)))
        target_bitrate = max(500, target_bitrate)

        logging.info(f"""
        Starting video compression:
        - Input: {input_path}
        - Original dimensions: {width}x{height}
        - New dimensions: {new_width}x{new_height}
        - Scale factor: {scale_factor}
        - CRF: {crf}
        - Target bitrate: {target_bitrate}kbps
        - GPU enabled: {self.gpu_supported}
        """)

        encoder = self._get_encoder_settings()
        filters = [f'scale={new_width}:{new_height}']
        if fps > 30:
            filters.append('fps=fps=30')
            logging.info(f"Limiting FPS to 30 (original: {fps})")

        command = [
            'ffmpeg', '-i', str(input_path),
            '-vf', ','.join(filters),
            '-c:v', encoder['codec'],
            '-preset', encoder['preset'],
            '-b:v', f'{target_bitrate}k',
            '-maxrate', f'{target_bitrate * 1.5}k',
            '-bufsize', f'{target_bitrate * 2}k',
            *encoder['extra_params'],
            '-crf' if encoder['codec'] == 'libx264' else '-cq',
            str(crf),
            '-c:a', 'copy',
            '-movflags', '+faststart',
            '-y', str(output_path)
        ]

        if self.gpu_supported:
            command.insert(1, '-hwaccel')
            command.insert(2, 'cuda')

        # logging.info(f"Executing FFmpeg command:\n{' '.join(command)}")
        success = run_ffmpeg_command(command)

        if success:
            size = os.path.getsize(output_path) / (1024 * 1024)
            logging.info(f"Compression successful: {
                size:.2f}MB output size for {input_path.name}")
        else:
            logging.error("Compression failed")

        return success, size

    def convert_to_mp4(self, input_path: Path, output_path: Path) -> bool:
        encoder = self._get_encoder_settings()
        command = [
            'ffmpeg', '-i', str(input_path),
            '-c:v', encoder['codec'],
            '-preset', encoder['preset'],
            '-crf' if encoder['codec'] == 'libx264' else '-cq', '18',
            '-c:a', 'aac',
            '-b:a', '192k',
            '-movflags', '+faststart',
            '-y', str(output_path)
        ]

        if self.gpu_supported:
            command.insert(1, '-hwaccel')
            command.insert(2, 'cuda')

        return run_ffmpeg_command(command)

    def process_video(self, video_path: Path, output_path: Path,
                      target_size: float) -> bool:
        temp_file = Path(TEMP_FILE_DIR) / f"temp_{video_path.name}"
        TempFileManager.register(temp_file)

        crf = VIDEO_COMPRESSION['crf']
        scale_factor = VIDEO_COMPRESSION['scale_factor']

        logging.info(f"""
        Starting video processing:
        - Input: {video_path}
        - Target size: {target_size}MB
        - Initial CRF: {crf}
        - Initial scale: {scale_factor}
        """)

        try:
            # Track the last compressed size to detect stagnation
            last_size = float('inf')
            while scale_factor >= 0.1:
                logging.info(f"Attempting compression with CRF={
                             crf}, scale={scale_factor}")
                success, size = self.compress_video(
                    video_path, temp_file, scale_factor, crf)

                if not success:
                    logging.error("Compression attempt failed")
                    return False

                logging.info(f"Compression result: {size:.2f}MB")
                self.size_predictor.update(crf, size)

                # Check if size is not improving
                if size >= last_size:
                    logging.warning(
                        f"No improvement in size. Stopping further attempts for CRF={crf}.")
                    break

                last_size = size

                if size <= target_size:
                    logging.info(f"Successfully achieved size of {target_size}MB  for {video_path.name}. "
                                 f"Final file size: {size:.2f}MB")
                    shutil.move(str(temp_file), str(output_path))
                    return True

                next_crf = self.size_predictor.predict_target_crf(
                    size, target_size)

                if next_crf and next_crf != crf:
                    logging.info(f"Adjusting CRF: {crf} -> {next_crf}")
                    crf = next_crf
                else:
                    new_crf = min(51, crf + 2)
                    logging.info(f"Incrementing CRF: {crf} -> {new_crf}")
                    crf = new_crf

                if crf >= 51 and scale_factor >= 0.1:
                    logging.info(f"Maximum CRF reached and target size not achieved. "
                                 f"Reducing scale factor: {scale_factor} -> {scale_factor * 0.75}")
                    scale_factor *= 0.5
                    # Increment CRF after scale reduction
                    crf = min(51, crf + 2)
                elif crf >= 51:
                    logging.warning(
                        f"Maximum CRF reached and no further scaling possible for {video_path}")
                    break

            logging.warning(f"Failed to achieve target size for {video_path}")
            return False

        except Exception as e:
            logging.error(f"Error processing video: {e}", exc_info=True)
            return False
        finally:
            logging.info("Cleaning up temporary files")
            TempFileManager.unregister(temp_file)
            if temp_file.exists():
                temp_file.unlink()

    def process_file(self, file_path: Path, output_path: Path, is_video: bool = True) -> bool:
        """Process a single video file.

        Args:
            file_path: Path to input video
            output_path: Path to save processed video
            is_video: Unused but kept for interface consistency with GIFProcessor

        Returns:
            bool: True if processing succeeded, False otherwise
        """
        return self.process_video(file_path, output_path, VIDEO_COMPRESSION['min_size_mb'])

    def process_all(self) -> List[Path]:
        # Convert non-MP4 videos to MP4
        for fmt in SUPPORTED_VIDEO_FORMATS:
            if fmt.lower() != '.mp4':
                for video in Path(INPUT_DIR).glob(f'*{fmt}'):
                    temp_mp4 = Path(TEMP_FILE_DIR) / f"{video.stem}.mp4"
                    final_mp4 = Path(INPUT_DIR) / f"{video.stem}.mp4"

                    if not final_mp4.exists():
                        logging.info(f"Converting {video.name} to MP4")
                        TempFileManager.register(temp_mp4)
                        if self.convert_to_mp4(video, temp_mp4):
                            shutil.move(str(temp_mp4), str(final_mp4))
                        else:
                            self.failed_files.append(video)
                        TempFileManager.unregister(temp_mp4)

        # Process MP4 files
        for video in Path(INPUT_DIR).glob('*.mp4'):
            output_path = Path(OUTPUT_DIR) / video.name
            if not output_path.exists():
                if not self.process_video(video, output_path,
                                          VIDEO_COMPRESSION['min_size_mb']):
                    self.failed_files.append(video)

        logging.info(f"All videos processed successfully. Failed files: {
            len(self.failed_files)}")

        return self.failed_files


def process_videos(gpu_supported: bool = False) -> List[Path]:
    processor = VideoProcessor(gpu_supported)
    return processor.process_all()
