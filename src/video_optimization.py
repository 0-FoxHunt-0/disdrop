import logging
import os
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from queue import Queue
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union

from default_config import (INPUT_DIR, OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS,
                            TEMP_FILE_DIR, VIDEO_COMPRESSION, VIDEO_SETTINGS)
from logging_system import log_function_call, run_ffmpeg_command
from temp_file_manager import TempFileManager
import temp_file_manager

logger = logging.getLogger(__name__)


class CompressionQuality(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class VideoMetadata:
    width: int
    height: int
    frame_rate: float
    has_audio: bool
    size_mb: float
    duration: float
    bitrate: int


class VideoEncoder:
    def __init__(self, settings: dict = VIDEO_SETTINGS):
        self.settings = settings
        self.is_gpu_available = self._check_gpu_availability()
        self._setup_encoder()

    def _setup_encoder(self):
        if self.is_gpu_available and self.settings['gpu']['enabled']:
            self.mode = 'gpu'
            self.encoder = self.settings['gpu']['encoders'][0]
            self.presets = self.settings['gpu']['presets']
        else:
            self.mode = 'cpu'
            self.encoder = self.settings['cpu']['encoders'][0]
            self.presets = self.settings['cpu']['presets']

    def encode_video(self, input_path: Path, output_path: Path,
                     quality_preset: str = 'balanced') -> bool:
        preset = self.presets[quality_preset]
        command = self._build_encoding_command(input_path, output_path, preset)

        for attempt in range(self.settings['general']['max_retries']):
            try:
                with ProcessManager(command, timeout=self.settings['general']['timeout']) as pm:
                    if pm.run():
                        return True
                    if attempt == self.settings['general']['max_retries'] - 1:
                        return False
            except Exception as e:
                logger.error(f"Encoding error: {e}")
                if attempt == self.settings['general']['max_retries'] - 1:
                    return False


class ProcessManager:
    def __init__(self, command: List[str], timeout: int):
        self.command = command
        self.timeout = timeout
        self.process = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def run(self) -> bool:
        try:
            self.process = subprocess.Popen(
                self.command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid
            )
            stdout, stderr = self.process.communicate(timeout=self.timeout)
            return self.process.returncode == 0
        except subprocess.TimeoutExpired:
            self.cleanup()
            raise

    def cleanup(self):
        if self.process:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)


class VideoWorker(Thread):
    def __init__(self, queue: Queue, results: Queue, encoder: VideoEncoder):
        super().__init__()
        self.queue = queue
        self.results = results
        self.encoder = encoder

    def run(self):
        while True:
            video = self.queue.get()
            if video is None:
                break

            output_path = self._get_output_path(video)
            success = self.encoder.encode_video(video, output_path)
            self.results.put(ProcessResult(video, success))


class VideoProcessor:
    def __init__(self, use_gpu: bool = True, temp_manager: Optional[TempFileManager] = None):
        self.use_gpu = use_gpu
        self.quality_profiles = {
            CompressionQuality.LOW: {"crf": 28, "scale": 0.7, "preset": "medium"},
            CompressionQuality.MEDIUM: {"crf": 23, "scale": 0.85, "preset": "slow"},
            CompressionQuality.HIGH: {"crf": 18,
                                      "scale": 1.0, "preset": "veryslow"}
        }
        self.temp_manager = temp_manager or TempFileManager()
        self.size_tolerance = 0.1  # 10% tolerance for target size
        self.min_scale = 0.1  # Minimum allowed scale factor

    def _create_temp_file(self, prefix: str, suffix: str) -> Path:
        """Create a temporary file with the given prefix and suffix."""
        temp_path = Path(TEMP_FILE_DIR) / \
            f"{prefix}_temp_{os.urandom(4).hex()}{suffix}"
        return temp_path

    def get_video_metadata(self, video_path: Path) -> Optional[VideoMetadata]:
        """Get comprehensive metadata for a video file."""
        try:
            width, height = self._get_dimensions(video_path)
            if width is None or height is None:
                raise ValueError("Failed to get video dimensions")

            frame_rate = self._get_frame_rate(video_path)
            has_audio = self._has_audio(video_path)
            size_mb = self._get_file_size(video_path)
            duration = self._get_duration(video_path)
            bitrate = self._calculate_optimal_bitrate(width, height, duration)

            return VideoMetadata(
                width=width,
                height=height,
                frame_rate=frame_rate,
                has_audio=has_audio,
                size_mb=size_mb,
                duration=duration,
                bitrate=bitrate
            )
        except Exception as e:
            logger.error(f"Failed to get metadata for {video_path}: {e}")
            return None

    def _get_duration(self, video_path: Path) -> float:
        """Get the duration of the video in seconds."""
        command = [
            'ffprobe', '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            str(video_path)
        ]
        try:
            output = subprocess.check_output(command, text=True).strip()
            return float(output)
        except (subprocess.CalledProcessError, ValueError) as e:
            logger.error(f"Failed to get duration for {video_path}: {e}")
            raise

    def _calculate_optimal_bitrate(self, width: int, height: int, duration: float) -> int:
        """Calculate optimal bitrate based on video dimensions and duration."""
        pixels = width * height
        # Basic bitrate calculation based on resolution
        if pixels <= 352 * 240:  # 240p
            base_bitrate = 400
        elif pixels <= 640 * 360:  # 360p
            base_bitrate = 800
        elif pixels <= 854 * 480:  # 480p
            base_bitrate = 1200
        elif pixels <= 1280 * 720:  # 720p
            base_bitrate = 2500
        elif pixels <= 1920 * 1080:  # 1080p
            base_bitrate = 4000
        else:  # 4K and above
            base_bitrate = 8000

        # Adjust bitrate based on duration
        if duration > 3600:  # Longer than 1 hour
            base_bitrate = int(base_bitrate * 0.9)
        elif duration < 60:  # Shorter than 1 minute
            base_bitrate = int(base_bitrate * 1.2)

        return base_bitrate

    def process_video(self, input_path: Path, output_path: Path, target_size_mb: float) -> bool:
        metadata = self.get_video_metadata(input_path)
        if not metadata:
            return False

        # If original size is already smaller than target, just copy the file
        if metadata.size_mb <= target_size_mb:
            shutil.copy2(str(input_path), str(output_path))
            logger.success(f"File already smaller than target: {
                           input_path.name}")
            return True

        temp_output = self._create_temp_file(
            input_path.stem, input_path.suffix)
        best_result = None
        best_size = float('inf')

        try:
            # Initial compression parameters
            params = self._calculate_compression_params(
                metadata, target_size_mb)

            while params["scale_factor"] >= self.min_scale:
                if self._compress_with_params(input_path, temp_output, params, metadata):
                    current_size = self._get_file_size(temp_output)
                    logger.info(f"Attempt with scale {
                                params['scale_factor']:.3f} resulted in size: {current_size:.2f}MB")

                    # Check if this result is better than previous attempts
                    if abs(current_size - target_size_mb) < abs(best_size - target_size_mb):
                        if best_result and best_result != temp_output:
                            best_result.unlink(missing_ok=True)
                        best_result = temp_output
                        best_size = current_size

                    # If we're within tolerance of target size, we're done
                    if current_size <= target_size_mb * (1 + self.size_tolerance):
                        shutil.move(str(temp_output), str(output_path))
                        return True

                    # Adjust parameters based on result
                    if current_size > target_size_mb:
                        # Need stronger compression
                        params = self._adjust_params_for_smaller_size(
                            params, current_size, target_size_mb)
                    else:
                        # Can reduce compression
                        params = self._adjust_params_for_larger_size(
                            params, current_size, target_size_mb)

                    # Create new temp file for next attempt
                    temp_output = self._create_temp_file(
                        input_path.stem, input_path.suffix)
                else:
                    logger.error(f"Compression failed for {input_path.name} at scale {
                                 params['scale_factor']:.3f}")
                    params["scale_factor"] *= 0.9  # Reduce scale on failure

            # If we didn't hit target size but have a best result, use it
            if best_result and best_result.exists():
                shutil.move(str(best_result), str(output_path))
                logger.warning(f"Could not reach target size of {target_size_mb}MB for {
                               input_path.name}. Best achieved: {best_size:.2f}MB")
                return True

            return False

        finally:
            # Cleanup any temporary files
            if temp_output.exists():
                temp_output.unlink()
            if best_result and best_result != temp_output and best_result.exists():
                best_result.unlink()

    def _adjust_params_for_smaller_size(self, params: Dict, current_size: float, target_size_mb: float) -> Dict:
        """Adjust parameters to achieve smaller file size."""
        new_params = params.copy()

        # Calculate how far we are from target
        size_ratio = current_size / target_size_mb

        # Adjust scale factor more aggressively for larger differences
        if size_ratio > 2:
            new_params["scale_factor"] = max(
                self.min_scale, params["scale_factor"] * 0.8)
        elif size_ratio > 1.5:
            new_params["scale_factor"] = max(
                self.min_scale, params["scale_factor"] * 0.9)
        else:
            new_params["scale_factor"] = max(
                self.min_scale, params["scale_factor"] * 0.95)

        # Adjust bitrate based on size ratio
        new_params["target_bitrate"] = int(
            params["target_bitrate"] / size_ratio)

        # Adjust CRF
        new_params["crf"] = min(51, params["crf"] + 2)

        return new_params

    def _adjust_params_for_larger_size(self, params: Dict, current_size: float, target_size_mb: float) -> Dict:
        """Adjust parameters to achieve larger file size."""
        new_params = params.copy()

        # Calculate how far we are from target
        size_ratio = target_size_mb / current_size

        # Adjust bitrate
        new_params["target_bitrate"] = int(
            params["target_bitrate"] * size_ratio * 1.1)

        # Adjust CRF
        new_params["crf"] = max(0, params["crf"] - 2)

        return new_params

    def _calculate_compression_params(self, metadata: VideoMetadata, target_size_mb: float) -> Dict:
        """Calculate initial compression parameters based on metadata and target size."""
        target_bitrate = int((target_size_mb * 8192) /
                             metadata.duration)  # Convert MB to kbps

        # Base scale factor on ratio of target to current size
        scale_ratio = (target_size_mb / metadata.size_mb) ** 0.5
        scale_factor = min(1.0, max(self.min_scale, scale_ratio))

        # Select initial CRF based on compression ratio needed
        compression_ratio = metadata.size_mb / target_size_mb
        if compression_ratio > 2:
            crf = 28  # Heavy compression needed
        elif compression_ratio > 1.5:
            crf = 23  # Medium compression needed
        else:
            crf = 18  # Light compression needed

        return {
            "target_bitrate": target_bitrate,
            "scale_factor": scale_factor,
            "crf": crf,
            "preset": "medium"  # Start with medium preset for speed
        }

    def _compress_with_params(self, input_path: Path, output_path: Path,
                              params: Dict, metadata: VideoMetadata) -> bool:
        command = self._build_compression_command(
            input_path, output_path, params, metadata)
        try:
            success = run_ffmpeg_command(command)
            if success:
                result_size = self._get_file_size(output_path)
                logger.success(f"Compressed {input_path.name} to {
                               result_size:.2f}MB")
                return True
            return False
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            return False

    def _build_compression_command(self, input_path: Path, output_path: Path,
                                   params: Dict, metadata: VideoMetadata) -> List[str]:
        new_width = int(metadata.width * params["scale_factor"])
        new_height = int(metadata.height * params["scale_factor"])

        # Ensure dimensions are even
        new_width += new_width % 2
        new_height += new_height % 2

        filter_string = (
            f"scale={new_width}:{
                new_height}:force_original_aspect_ratio=decrease,"
            f"pad={new_width}:{new_height}:(ow-iw)/2:(oh-ih)/2"
        )

        if metadata.frame_rate > 30:
            filter_string += ",fps=fps=30"

        if self.use_gpu:
            return [
                'ffmpeg', '-hwaccel', 'cuda', '-i', str(input_path),
                '-vf', filter_string,
                '-c:v', 'h264_nvenc',
                '-preset', 'p7',
                '-b:v', f"{params['target_bitrate']}k",
                '-maxrate', f"{int(params['target_bitrate'] * 1.5)}k",
                '-bufsize', f"{params['target_bitrate'] * 2}k",
                '-profile:v', 'main',
                '-rc', 'vbr',
                '-cq', str(params['crf']),
                '-c:a', 'copy' if metadata.has_audio else 'none',
                '-movflags', '+faststart',
                '-y', str(output_path)
            ]
        else:
            return [
                'ffmpeg', '-i', str(input_path),
                '-vf', filter_string,
                '-c:v', 'libx264',
                '-preset', params['preset'],
                '-b:v', f"{params['target_bitrate']}k",
                '-maxrate', f"{int(params['target_bitrate'] * 1.5)}k",
                '-bufsize', f"{params['target_bitrate'] * 2}k",
                '-profile:v', 'main',
                '-crf', str(params['crf']),
                '-c:a', 'copy' if metadata.has_audio else 'none',
                '-movflags', '+faststart',
                '-y', str(output_path)
            ]

    # Helper methods with better error handling
    def _get_dimensions(self, video_path: Path) -> Tuple[Optional[int], Optional[int]]:
        try:
            command = [
                'ffprobe', '-v', 'error', '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0',
                str(video_path)
            ]
            output = subprocess.check_output(command, text=True).strip()
            width, height = map(int, output.split('x'))
            return width, height
        except Exception as e:
            logger.error(f"Failed to get dimensions: {e}")
            return None, None

    def _get_frame_rate(self, video_path: Path) -> float:
        command = [
            'ffprobe', '-v', 'error', '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate', '-of',
            'default=noprint_wrappers=1:nokey=1', str(video_path)
        ]
        try:
            if run_ffmpeg_command(command):
                output = subprocess.check_output(command, text=True).strip()
                num, den = map(int, output.split('/'))
                return num / den if den != 0 else num
            raise ValueError("Failed to run ffprobe command")
        except Exception as e:
            logger.error(f"Failed to get frame rate: {e}")
            raise

    def _has_audio(self, video_path: Path) -> bool:
        command = ['ffprobe', '-i', str(video_path), '-show_streams',
                   '-select_streams', 'a', '-loglevel', 'error']
        try:
            if run_ffmpeg_command(command):
                output = subprocess.check_output(
                    command, stderr=subprocess.STDOUT)
                return len(output) > 0
            return False
        except Exception:
            return False

    def _get_file_size(self, file_path: Path) -> float:
        try:
            return os.path.getsize(file_path) / (1024 * 1024)
        except OSError as e:
            logger.error(f"Failed to get file size: {e}")
            raise


class BatchVideoProcessor:
    def __init__(self, use_gpu: bool = False, max_workers: Optional[int] = None,
                 input_dir: Optional[Path] = None, output_dir: Optional[Path] = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) * 2)
        self.processor = VideoProcessor(use_gpu=use_gpu)
        self.input_dir = Path(input_dir or INPUT_DIR)
        self.output_dir = Path(output_dir or OUTPUT_DIR)
        self.temp_dir = Path(TEMP_FILE_DIR)

        # Ensure directories exist
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def process_all_videos(self) -> List[Path]:
        if not list(self.input_dir.glob('*.*')):
            logger.warning(f"No video files found in {self.input_dir}")
            return []

        failed_files = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Process non-MP4 videos first
            for format in SUPPORTED_VIDEO_FORMATS:
                if format.lower() != '.mp4':
                    futures = []
                    for video_file in self.input_dir.glob(f'*{format}'):
                        if self._needs_processing(video_file):
                            futures.append(
                                executor.submit(
                                    self._convert_to_mp4, video_file)
                            )
                    for future in futures:
                        if not future.result():
                            failed_files.append(video_file)

            # Process MP4 videos
            futures = []
            for video_file in self.input_dir.glob('*.[mM][pP]4'):
                if self._needs_processing(video_file):
                    output_path = self.output_dir / f"{video_file.stem}.mp4"
                    futures.append(
                        executor.submit(self._process_single_mp4,
                                        video_file, output_path)
                    )

            for future in futures:
                if not future.result():
                    failed_files.append(video_file)

        # self._log_results(failed_files)
        return failed_files

    def _needs_processing(self, video_file: Path) -> bool:
        """Check if a video needs processing based on output existence and size."""
        output_path = self.output_dir / f"{video_file.stem}.mp4"

        # If output doesn't exist, needs processing
        if not output_path.exists():
            return True

        # If output exists, check if it meets size requirements
        try:
            output_size = output_path.stat().st_size / (1024 * 1024)  # Convert to MB
            if output_size > VIDEO_COMPRESSION['min_size_mb']:
                logger.warning(f"Existing output {output_path.name} ({output_size:.2f}MB) exceeds target size "
                               f"({VIDEO_COMPRESSION['min_size_mb']}MB). Will reprocess.")
                return True
        except OSError as e:
            logger.error(f"Error checking size of {output_path}: {e}")
            return True

        return False

    def _convert_to_mp4(self, video_file: Path) -> bool:
        temp_file = self.temp_dir / f"temp_{video_file.stem}.mp4"
        final_output = self.input_dir / f"{video_file.stem}.mp4"

        try:
            metadata = self.processor.get_video_metadata(video_file)
            if not metadata:
                return False

            params = {
                "crf": 18,
                "preset": "medium",
                "target_bitrate": metadata.bitrate
            }

            success = self.processor._compress_with_params(
                video_file, temp_file, params, metadata)

            if success:
                shutil.move(str(temp_file), str(final_output))
                logger.success(f"Converted to MP4: {video_file.name}")
                return True

            return False
        finally:
            if temp_file.exists():
                temp_file.unlink()

    def _process_single_mp4(self, video_file: Path, output_path: Path) -> bool:
        try:
            result = self.processor.process_video(
                video_file, output_path, VIDEO_COMPRESSION['min_size_mb'])

            # Verify the result meets size requirements
            if result:
                output_size = output_path.stat().st_size / (1024 * 1024)
                if output_size > VIDEO_COMPRESSION['min_size_mb']:
                    logger.error(f"Failed to reach target size for {video_file.name}: "
                                 f"got {output_size:.2f}MB, target was {VIDEO_COMPRESSION['min_size_mb']}MB")
                    return False
            return result
        except Exception as e:
            logger.error(f"Failed to process {video_file.name}: {e}")
            return False

    def _log_results(self, failed_files: List[Path]):
        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} files:")
            for file in failed_files:
                logger.warning(f"- {file.name}")
        else:
            logger.success("All videos processed successfully")
