import logging
import os
import shutil
import subprocess
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from default_config import (INPUT_DIR, OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS,
                            TEMP_FILE_DIR, VIDEO_COMPRESSION)
from logging_system import log_function_call, run_ffmpeg_command
from temp_file_manager import TempFileManager

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

    def get_video_metadata(self, video_path: Path) -> Optional[VideoMetadata]:
        try:
            width, height = self._get_dimensions(video_path)
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

    def process_video(self, input_path: Path, output_path: Path, target_size_mb: float) -> bool:
        temp_output = self._create_temp_file(
            input_path.stem, input_path.suffix)
        self.temp_manager.register(temp_output)

        try:
            metadata = self.get_video_metadata(input_path)
            if not metadata:
                return False

            params = self._calculate_compression_params(
                metadata, target_size_mb)
            if self._compress_with_params(input_path, temp_output, params, metadata):
                shutil.move(str(temp_output), str(output_path))
                return True
            return False
        finally:
            self.temp_manager.unregister(temp_output)
            if temp_output.exists():
                temp_output.unlink()

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
    def __init__(self, use_gpu: bool = False, max_workers: Optional[int] = None):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) * 2)
        self.processor = VideoProcessor(use_gpu=use_gpu)
        self.input_dir = Path(INPUT_DIR)
        self.output_dir = Path(OUTPUT_DIR)
        self.temp_dir = Path(TEMP_FILE_DIR)

    def process_all_videos(self) -> List[Path]:
        failed_files = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Process non-MP4 videos first
            for format in SUPPORTED_VIDEO_FORMATS:
                if format.lower() != '.mp4':
                    futures = []
                    for video_file in self.input_dir.glob(f'*{format}'):
                        futures.append(
                            executor.submit(self._convert_to_mp4, video_file)
                        )
                    for future in futures:
                        if not future.result():
                            failed_files.append(future.path)

            # Process MP4 videos
            futures = []
            for video_file in self.input_dir.glob('*.[mM][pP]4'):
                output_path = self.output_dir / f"{video_file.stem}.mp4"
                if not output_path.exists():
                    futures.append(
                        executor.submit(self._process_single_mp4,
                                        video_file, output_path)
                    )

            for future in futures:
                if not future.result():
                    failed_files.append(future.path)

        self._log_results(failed_files)
        return failed_files

    def _convert_to_mp4(self, video_file: Path) -> bool:
        temp_file = self.temp_dir / f"temp_{video_file.stem}.mp4"
        final_output = self.input_dir / f"{video_file.stem}.mp4"

        if final_output.exists():
            logger.skip(f"MP4 version exists: {video_file.name}")
            return True

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
            return self.processor.process_video(
                video_file, output_path, VIDEO_COMPRESSION['min_size_mb'])
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

# Function to maintain backward compatibility


def process_videos(gpu_supported: bool = False) -> List[Path]:
    processor = BatchVideoProcessor(use_gpu=gpu_supported)
    return processor.process_all_videos()
