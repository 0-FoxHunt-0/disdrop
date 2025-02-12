import logging
import os
import shutil
import subprocess
import threading
import time
import signal
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from threading import Event, Thread
from typing import Dict, List, Optional, Tuple, Union

# Updated imports with correct paths
from .default_config import (INPUT_DIR, OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS,
                             TEMP_FILE_DIR, VIDEO_COMPRESSION, VIDEO_SETTINGS)
from .logging_system import log_function_call, run_ffmpeg_command
from .temp_file_manager import TempFileManager
from .utils.video_dimensions import get_video_dimensions
from .utils.error_handler import validate_video_file, retry_video_processing, VideoProcessingError
from .utils.resource_manager import ResourceMonitor, ResourceGuard

logger = logging.getLogger(__name__)


class CompressionQuality(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    ULTRA = "ultra"


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
        self.is_windows = os.name == 'nt'

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

    def run(self) -> bool:
        try:
            # Different process creation for Windows and Unix
            if self.is_windows:
                self.process = subprocess.Popen(
                    self.command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP
                )
            else:
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
        """Enhanced cleanup with better error handling."""
        if not self.process:
            return

        try:
            if self.is_windows:
                if self.process.poll() is None:
                    # First try graceful termination
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        # Then force kill
                        try:
                            self.process.kill()
                            self.process.wait(timeout=2)
                        except:
                            # Last resort: taskkill
                            subprocess.run(['taskkill', '/F', '/T', '/PID', str(self.process.pid)],
                                           stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                # Unix systems: try SIGTERM first, then SIGKILL
                try:
                    pgid = os.getpgid(self.process.pid)
                    os.killpg(pgid, signal.SIGTERM)
                    try:
                        self.process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        os.killpg(pgid, signal.SIGKILL)
                        self.process.wait(timeout=2)
                except (ProcessLookupError, OSError):
                    pass  # Process already terminated
        except Exception as e:
            logger.error(f"Failed to cleanup process: {e}")
            # Ensure process is killed
            if self.process.poll() is None:
                try:
                    self.process.kill()
                except:
                    pass


class VideoProcessor:
    def __init__(self, use_gpu: bool = True, settings: dict = VIDEO_SETTINGS):
        self.use_gpu = use_gpu
        self.settings = settings
        self.gpu_available = self._check_gpu_availability() if use_gpu else False
        self.min_scale = 0.1
        # Adjust tolerance based on file size
        self.base_size_tolerance = 0.1  # 10% for small files
        self.max_compression_attempts = 12  # Increased from 8 to allow more attempts
        # Try extreme compression if within 20% of target
        self.extreme_compression_threshold = 1.2
        self.target_size_mb = VIDEO_COMPRESSION['min_size_mb']  # Add this line
        self._setup_encoder()
        self.dimension_cache = {}
        # Add resource monitor
        self.resource_monitor = ResourceMonitor()

    def _setup_encoder(self):
        if self.gpu_available and self.settings['gpu']['enabled']:
            self.mode = 'gpu'
            self.encoder = self.settings['gpu']['encoders'][0]
            self.presets = self.settings['gpu']['presets']
        else:
            self.mode = 'cpu'
            self.encoder = self.settings['cpu']['encoders'][0]
            self.presets = self.settings['cpu']['presets']

    def _get_temp_path(self, original_path: Path, suffix: str = "") -> Path:
        """Generate a temporary file path in the temp directory."""
        return Path(TEMP_FILE_DIR) / f"temp_{original_path.stem}{suffix}{original_path.suffix}"

    def process_videos(self, input_dir: Path, output_dir: Path, target_size_mb: float) -> List[Path]:
        """Process videos one at a time with optimized settings."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        failed_files = []

        video_files = list(input_dir.glob('*.[mM][pP]4'))
        video_files.sort(key=lambda x: x.stat().st_size, reverse=True)

        total_files = len(video_files)
        logger.info(f"Found {total_files} videos to process")

        for idx, video_file in enumerate(video_files, 1):
            try:
                original_size = video_file.stat().st_size / (1024 * 1024)
                output_path = output_dir / f"{video_file.stem}.mp4"

                # Check if a compressed version already exists and meets size requirements
                if output_path.exists():
                    existing_size = output_path.stat().st_size / (1024 * 1024)
                    if existing_size <= target_size_mb:
                        logger.info(f"Skipping {video_file.name}: Already compressed version exists({
                                    existing_size: .2f}MB)")
                        continue
                    else:
                        logger.info(f"Existing compressed version({
                                    existing_size: .2f}MB) exceeds target size({target_size_mb}MB), recompressing")

                logger.info(
                    f"Processing {idx}/{total_files}: {video_file.name} ({original_size:.2f}MB)")

                # Check if file already meets requirements
                if original_size <= target_size_mb:
                    logger.info(
                        f"File already meets size requirements ({original_size:.2f}MB <= {target_size_mb}MB)")
                    if not output_path.exists():
                        shutil.copy2(video_file, output_path)
                        logger.info(
                            f"Copied original file to output directory: {output_path}")
                    continue

                if not self._process_single_video(video_file, output_path, target_size_mb):
                    failed_files.append(video_file)
                else:
                    final_size = output_path.stat().st_size / (1024 * 1024)
                    reduction = ((original_size - final_size) /
                                 original_size) * 100
                    logger.success(f"Compressed {video_file.name}: {original_size: .2f}MB to {
                                   final_size: .2f}MB({reduction: .1f} % reduction)")
            except Exception as e:
                logger.error(f"Failed to process {video_file}: {e}")
                failed_files.append(video_file)

        return failed_files

    @retry_video_processing
    def _process_single_video(self, input_path: Path, output_path: Path, target_size_mb: float) -> bool:
        """Process a single video with dynamic quality adjustment."""
        try:
            with ResourceGuard(self.resource_monitor):
                if not validate_video_file(str(input_path)):
                    raise VideoProcessingError(
                        "Video validation failed", str(input_path))

                logger.info(f"Analyzing video metadata for {input_path.name}")
                metadata = self.get_video_metadata(input_path)
                if not metadata:
                    return False

                # Adjust tolerance based on original file size
                size_tolerance = self._get_size_tolerance(metadata.size_mb)
                logger.debug(f"Video info: {metadata.width}x{metadata.height}, {
                             metadata.frame_rate}fps, {metadata.duration: .1f}s")
                logger.debug(f"Target size: {target_size_mb}MB(tolerance: {
                             size_tolerance*100: .1f} % )")

                # For very large files, try two-pass encoding first
                if metadata.size_mb > 100 and metadata.size_mb / target_size_mb > 10:
                    logger.info(f"Large file detected({
                                metadata.size_mb: .2f}MB), attempting two-pass compression")
                    if self._two_pass_compression(input_path, output_path, target_size_mb, metadata):
                        return True
                    logger.info(
                        "Two-pass compression unsuccessful, falling back to standard compression")

                # Create working copy in temp directory
                temp_input = self._get_temp_path(input_path)
                shutil.copy2(input_path, temp_input)

                attempt = 0
                current_size = metadata.size_mb
                best_result = None
                best_size = float('inf')

                # Initial scale based on resolution and target size
                scale = self._calculate_initial_scale(metadata, target_size_mb)
                crf = self._calculate_initial_crf(metadata, target_size_mb)

                while attempt < self.max_compression_attempts and current_size > target_size_mb:
                    temp_output = self._get_temp_path(
                        input_path, f"_attempt_{attempt}")

                    # Calculate compression parameters
                    target_bitrate = self._calculate_target_bitrate(
                        target_size_mb, metadata.duration)

                    # Try compression with current parameters
                    success = self._compress_attempt(
                        temp_input, temp_output, metadata, target_bitrate, scale, crf
                    )

                    if success and temp_output.exists():
                        current_size = temp_output.stat().st_size / (1024 * 1024)

                        # Update best result if this attempt is better
                        if current_size < best_size:
                            if best_result and best_result.exists():
                                best_result.unlink()
                            best_result = temp_output
                            best_size = current_size

                        # Adjust parameters for next attempt if needed
                        if current_size > target_size_mb:
                            scale, crf = self._adjust_parameters(
                                scale, crf, current_size, target_size_mb
                            )
                            # Use the current best result as input for next attempt
                            if best_result and best_result.exists():
                                shutil.copy2(best_result, temp_input)
                        else:
                            # Target reached
                            shutil.move(best_result, output_path)
                            return True

                    attempt += 1

                # If we have a best result but didn't hit target, use it anyway
                if best_result and best_result.exists():
                    shutil.move(best_result, output_path)
                    logger.warning(
                        f"Could not reach target size of {target_size_mb}MB. Best achieved: {best_size:.2f}MB")
                    return True

                return False

        except VideoProcessingError as e:
            logger.error(f"Video processing error: {e}")
            return False
        except Exception as e:
            logger.exception(f"Processing failed for {input_path}: {e}")
            return False
        finally:
            # Cleanup temp files
            for temp_file in Path(TEMP_FILE_DIR).glob(f"temp_{input_path.stem}*"):
                try:
                    temp_file.unlink(missing_ok=True)
                except Exception as e:
                    logger.error(
                        f"Failed to cleanup temp file {temp_file}: {e}")

    def _calculate_initial_scale(self, metadata: VideoMetadata, target_size_mb: float) -> float:
        """Calculate initial scale factor based on video properties."""
        # Base scale on both resolution and compression needed
        compression_ratio = metadata.size_mb / target_size_mb
        resolution_factor = max(metadata.width, metadata.height) / 1920

        if compression_ratio > 4:
            scale = min(0.5, 1 / resolution_factor)
        elif compression_ratio > 2:
            scale = min(0.7, 1 / resolution_factor)
        else:
            scale = min(0.9, 1 / resolution_factor)

        return max(self.min_scale, scale)

    def _calculate_initial_crf(self, metadata: VideoMetadata, target_size_mb: float) -> int:
        """Calculate initial CRF value based on compression needs."""
        compression_ratio = metadata.size_mb / target_size_mb

        if compression_ratio > 4:
            return 28
        elif compression_ratio > 2:
            return 24
        else:
            return 20

    def _adjust_parameters(self, current_scale: float, current_crf: int,
                           current_size: float, target_size_mb: float) -> Tuple[float, int]:
        """Adjust compression parameters based on results."""
        size_ratio = current_size / target_size_mb

        # More aggressive adjustments when close to target but not quite there
        if 1.0 < size_ratio <= self.extreme_compression_threshold:
            new_scale = max(self.min_scale, current_scale *
                            0.85)  # More aggressive scaling
            # Allow higher CRF for final compression
            new_crf = min(current_crf + 3, 38)
            logger.debug(f"Close to target, using extreme compression: scale={
                         new_scale: .2f}, crf={new_crf}")
        elif size_ratio > 2:
            new_scale = max(self.min_scale, current_scale * 0.8)
            new_crf = min(current_crf + 4, 35)
            logger.debug(f"Size ratio > 2, aggressive adjustment: scale={
                         new_scale: .2f}, crf={new_crf}")
        elif size_ratio > 1.5:
            new_scale = max(self.min_scale, current_scale * 0.9)
            new_crf = min(current_crf + 2, 32)
            logger.debug(f"Size ratio > 1.5, moderate adjustment: scale={
                         new_scale: .2f}, crf={new_crf}")
        else:
            new_scale = max(self.min_scale, current_scale * 0.95)
            new_crf = min(current_crf + 1, 30)
            logger.debug(f"Size ratio <= 1.5, fine adjustment: scale={
                         new_scale: .2f}, crf={new_crf}")

        return new_scale, new_crf

    def _compress_attempt(self, input_path: Path, output_path: Path,
                          metadata: VideoMetadata, target_bitrate: int,
                          scale: float, crf: int) -> bool:
        """Attempt compression with given parameters."""
        try:
            new_width = int(metadata.width * scale // 2 * 2)
            new_height = int(metadata.height * scale // 2 * 2)

            logger.debug(f"Compression attempt with: scale={scale:.2f} ({new_width}x{new_height}), "
                         f"crf={crf}, bitrate={target_bitrate}k")

            # Ensure minimum dimensions
            new_width = max(16, new_width)
            new_height = max(16, new_height)

            # Base command with shared parameters
            command = [
                'ffmpeg', '-hide_banner', '-y',
                '-i', str(input_path),
                '-movflags', '+faststart',
                '-metadata', 'encoder=disdrop'
            ]

            # Video filters
            filters = [f'scale={new_width}:{new_height}:flags=lanczos']
            if metadata.frame_rate > 30:
                filters.append('fps=fps=30')

            # Add additional compression for files close to target size
            current_size = metadata.size_mb
            if current_size / self.target_size_mb <= self.extreme_compression_threshold:  # Changed this line
                # Slight sharpening to compensate for compression
                filters.append('unsharp=3:3:0.5:3:3:0.5')

            # GPU encoding command
            if self.use_gpu and self.gpu_available:
                command.extend([
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p2',  # Fixed preset instead of using settings
                    '-rc', 'vbr',
                    '-cq', str(crf),
                    '-qmin', str(max(0, crf - 5)),
                    '-qmax', str(min(51, crf + 5)),
                    '-profile:v', 'main',
                ])
            else:
                command.extend([
                    '-c:v', 'libx264',
                    '-preset', 'medium',  # Fixed preset instead of using settings
                    '-crf', str(crf),
                    '-profile:v', 'main',
                ])

            # Add filters and rate control
            command.extend([
                '-vf', ','.join(filters),
                '-b:v', f'{target_bitrate}k',
                '-maxrate', f'{int(target_bitrate * 1.5)}k',
                '-bufsize', f'{target_bitrate * 2}k',
            ])

            # Audio settings
            command.extend([
                '-c:a', 'aac',
                '-b:a', '128k' if metadata.has_audio else '96k',
                str(output_path)
            ])

            with ProcessManager(command, timeout=self.settings['general']['timeout']) as pm:
                return pm.run()

        except Exception as e:
            logger.exception(f"Compression attempt failed: {e}")
            return False

    def _build_optimal_command(self, input_path: Path, output_path: Path,
                               metadata: VideoMetadata, target_bitrate: int) -> List[str]:
        """Build FFmpeg command with optimal settings for quality and compression."""
        # Calculate optimal dimensions
        max_dimension = max(metadata.width, metadata.height)
        scale_factor = 1.0

        if max_dimension > 1920:
            scale_factor = 1920 / max_dimension

        new_width = int(metadata.width * scale_factor // 2 * 2)
        new_height = int(metadata.height * scale_factor // 2 * 2)

        # Base command with shared parameters
        base_command = [
            'ffmpeg', '-hide_banner', '-y',
            '-i', str(input_path),
            '-movflags', '+faststart',
            '-metadata', 'encoder=disdrop'
        ]

        # Video filters
        filters = []
        if scale_factor < 1.0:
            filters.append(f'scale={new_width}:{new_height}')
        if metadata.frame_rate > 30:
            filters.append('fps=30')

        # GPU encoding command
        if self.use_gpu and self.gpu_available:
            command = base_command + [
                '-c:v', 'h264_nvenc',
                '-preset', 'p2',  # Quality preset
                '-rc', 'vbr',     # Variable bitrate
                '-cq', '23',      # Quality target
                '-qmin', '19',    # Minimum quality
                '-qmax', '28',    # Maximum quality
                '-b:v', f'{target_bitrate}k',
                '-maxrate', f'{int(target_bitrate * 1.5)}k',
                '-bufsize', f'{target_bitrate * 2}k'
            ]
        # CPU encoding command
        else:
            command = base_command + [
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '23',
                '-b:v', f'{target_bitrate}k',
                '-maxrate', f'{int(target_bitrate * 1.5)}k',
                '-bufsize', f'{target_bitrate * 2}k'
            ]

        # Add filters if any
        if filters:
            command.extend(['-vf', ','.join(filters)])

        # Audio settings
        command.extend([
            '-c:a', 'aac',
            '-b:a', '128k',
            str(output_path)
        ])

        return command

    def _recompress_with_target(self, input_path: Path, output_path: Path,
                                target_size_mb: float, metadata: VideoMetadata) -> bool:
        """Recompress video with more aggressive settings to meet target size."""
        temp_path = output_path.with_name(f"temp_{output_path.name}")
        try:
            # Calculate more aggressive bitrate
            # Slightly lower than 8192
            target_bitrate = int((target_size_mb * 7800) / metadata.duration)

            # More aggressive compression settings
            if self.use_gpu and self.gpu_available:
                command = [
                    'ffmpeg', '-hide_banner', '-y',
                    '-i', str(input_path),
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p2',
                    '-rc', 'vbr',
                    '-cq', '27',      # Higher CQ value for more compression
                    '-qmin', '23',
                    '-qmax', '32',
                    '-b:v', f'{target_bitrate}k',
                    '-maxrate', f'{int(target_bitrate * 1.2)}k',
                    '-bufsize', f'{target_bitrate * 2}k',
                    '-vf', f'scale=iw*0.8:ih*0.8,fps=30',
                    '-c:a', 'aac',
                    '-b:a', '96k',
                    '-movflags', '+faststart',
                    str(temp_path)
                ]
            else:
                command = [
                    'ffmpeg', '-hide_banner', '-y',
                    '-i', str(input_path),
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-crf', '28',
                    '-b:v', f'{target_bitrate}k',
                    '-maxrate', f'{int(target_bitrate * 1.2)}k',
                    '-bufsize', f'{target_bitrate * 2}k',
                    '-vf', f'scale=iw*0.8:ih*0.8,fps=30',
                    '-c:a', 'aac',
                    '-b:a', '96k',
                    '-movflags', '+faststart',
                    str(temp_path)
                ]

            if self._run_ffmpeg(command):
                if temp_path.exists():
                    temp_path.replace(output_path)
                    return True
            return False

        finally:
            if temp_path.exists():
                temp_path.unlink(missing_ok=True)

    def _run_ffmpeg(self, command: List[str]) -> bool:
        """Run FFmpeg command with proper error handling."""
        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate()

            if process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr}")
                return False
            return True
        except Exception as e:
            logger.exception(f"FFmpeg execution failed: {e}")
            return False

    def _check_gpu_availability(self) -> bool:
        """Check if NVIDIA GPU encoding is available."""
        try:
            cmd = ['ffmpeg', '-hide_banner', '-encoders']
            result = subprocess.run(cmd, capture_output=True, text=True)
            return 'h264_nvenc' in result.stdout
        except Exception:
            return False

    def _create_temp_file(self, prefix: str, suffix: str) -> Path:
        """Create a temporary file with better error handling."""
        try:
            temp_path = Path(TEMP_FILE_DIR) / \
                f"{prefix}_temp_{os.urandom(4).hex()}{suffix}"
            if temp_path.exists():
                temp_path.unlink()
            return temp_path
        except Exception as e:
            logger.error(f"Failed to create temp file: {e}")
            raise

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

    # Helper methods with better error handling
    def _get_dimensions(self, video_path: Path) -> Tuple[Optional[int], Optional[int]]:
        """Get video dimensions using new utility with caching."""
        cache_key = str(video_path)
        if cache_key in self.dimension_cache:
            return self.dimension_cache[cache_key]

        try:
            dimensions = get_video_dimensions(str(video_path))
            if dimensions:
                self.dimension_cache[cache_key] = dimensions
                return dimensions
        except Exception as e:
            logger.error(f"Failed to get dimensions for {video_path}: {e}")

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

    def _calculate_target_bitrate(self, target_size_mb: float, duration: float) -> int:
        """Calculate target video bitrate based on target size and duration."""
        # Convert target size from MB to bits (MB * 8192 for kbps)
        # Reserve space for audio and container overhead based on duration
        audio_reserve = 0.10 if duration > 60 else 0.05  # More reserve for longer videos
        video_size_bits = target_size_mb * 8192 * (1 - audio_reserve)

        # Calculate target bitrate (bits/second)
        target_bitrate = int(video_size_bits / duration)

        # Dynamic bitrate bounds based on resolution and duration
        min_bitrate = 500 if duration > 60 else 800   # Higher minimum for short videos
        max_bitrate = 8000 if duration < 300 else 6000  # Lower maximum for long videos

        return max(min_bitrate, min(target_bitrate, max_bitrate))

    def _get_size_tolerance(self, original_size_mb: float) -> float:
        """Dynamic tolerance based on file size."""
        if original_size_mb > 100:
            return 0.05  # 5% tolerance for large files
        elif original_size_mb > 50:
            return 0.08  # 8% tolerance for medium files
        return self.base_size_tolerance

    def _two_pass_compression(self, input_path: Path, output_path: Path,
                              target_size_mb: float, metadata: VideoMetadata) -> bool:
        """Two-pass compression for better size control."""
        try:
            logger.info("Starting two-pass compression...")
            temp_output = self._get_temp_path(input_path, "_twopass")
            null_file = "NUL" if os.name == 'nt' else "/dev/null"

            # Calculate aggressive bitrate for large files
            target_bitrate = self._calculate_target_bitrate(
                target_size_mb, metadata.duration)
            scale = self._calculate_initial_scale(metadata, target_size_mb)

            # Common parameters
            filters = []
            new_width = int(metadata.width * scale // 2 * 2)
            new_height = int(metadata.height * scale // 2 * 2)
            filters.append(f'scale={new_width}:{new_height}:flags=lanczos')
            if metadata.frame_rate > 30:
                filters.append('fps=fps=30')

            # Base command
            base_command = [
                'ffmpeg', '-y', '-i', str(input_path),
                '-vf', ','.join(filters)
            ]

            if self.use_gpu and self.gpu_available:
                # GPU two-pass encoding
                pass1_command = base_command + [
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p2',
                    '-rc', 'vbr',
                    '-b:v', f'{target_bitrate}k',
                    '-maxrate', f'{int(target_bitrate * 1.5)}k',
                    '-bufsize', f'{target_bitrate * 2}k',
                    '-pass', '1',
                    '-f', 'null', null_file
                ]

                pass2_command = base_command + [
                    '-c:v', 'h264_nvenc',
                    '-preset', 'p2',
                    '-rc', 'vbr',
                    '-b:v', f'{target_bitrate}k',
                    '-maxrate', f'{int(target_bitrate * 1.5)}k',
                    '-bufsize', f'{target_bitrate * 2}k',
                    '-pass', '2',
                    '-c:a', 'aac',
                    '-b:a', '96k',
                    str(temp_output)
                ]
            else:
                # CPU two-pass encoding
                pass1_command = base_command + [
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-b:v', f'{target_bitrate}k',
                    '-pass', '1',
                    '-f', 'null', null_file
                ]

                pass2_command = base_command + [
                    '-c:v', 'libx264',
                    '-preset', 'medium',
                    '-b:v', f'{target_bitrate}k',
                    '-pass', '2',
                    '-c:a', 'aac',
                    '-b:a', '96k',
                    str(temp_output)
                ]

            # Run two-pass encoding
            logger.info("Running first pass...")
            with ProcessManager(pass1_command, timeout=self.settings['general']['timeout']) as pm1:
                if not pm1.run():
                    logger.error("First pass failed")
                    return False

            logger.info("Running second pass...")
            with ProcessManager(pass2_command, timeout=self.settings['general']['timeout']) as pm2:
                if not pm2.run():
                    logger.error("Second pass failed")
                    return False

            if temp_output.exists():
                logger.debug("Two-pass compression completed successfully")
                shutil.move(str(temp_output), str(output_path))
                return True

            return False

        except Exception as e:
            logger.exception(f"Two-pass compression failed: {e}")
            return False
        finally:
            if temp_output.exists():
                temp_output.unlink(missing_ok=True)

    def _calculate_target_bitrate(self, target_size_mb: float, duration: float) -> int:
        """Calculate target video bitrate with improved scaling for large files."""
        # Convert target size from MB to bits
        video_size_bits = target_size_mb * 8192

        # Adjust audio reserve based on duration
        audio_reserve = 0.15 if duration > 3600 else \
            0.12 if duration > 1800 else \
            0.10 if duration > 600 else \
            0.05

        video_size_bits *= (1 - audio_reserve)

        # Calculate base target bitrate
        target_bitrate = int(video_size_bits / duration)

        # Dynamic bitrate bounds
        min_bitrate = 400 if duration > 3600 else \
            500 if duration > 1800 else \
            600 if duration > 600 else \
            800

        max_bitrate = 4000 if duration > 3600 else \
            5000 if duration > 1800 else \
            6000 if duration > 600 else \
            8000

        return max(min_bitrate, min(target_bitrate, max_bitrate))

    def _get_encoder_threads(self) -> int:
        """Get safe number of encoder threads."""
        return self.resource_monitor.get_safe_thread_count()

    def _build_ffmpeg_command(self, input_path: Path, output_path: Path, settings: Dict) -> List[str]:
        """Build FFmpeg command with resource-aware thread count."""
        thread_count = self._get_encoder_threads()

        cmd = [
            'ffmpeg',
            '-i', str(input_path),
            '-threads', str(thread_count),
            # ...rest of existing command building code...
        ]
        return cmd
