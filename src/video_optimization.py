from src.base.processor import BaseProcessor
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
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

# Updated imports with correct paths
from .default_config import (INPUT_DIR, OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS,
                             TEMP_FILE_DIR, VIDEO_COMPRESSION, VIDEO_SETTINGS)
from .logging_system import log_function_call, run_ffmpeg_command, get_logger, performance_monitor
from .temp_file_manager import TempFileManager
from .utils.video_dimensions import get_video_dimensions
from .utils.error_handler import validate_video_file, retry_video_processing, VideoProcessingError
from src.gif_operations.resource_manager import ResourceMonitor, ResourceGuard

# Remove root logger, use module specific logger
logger = get_logger('video')


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

    def _check_gpu_availability(self) -> bool:
        """Check if GPU acceleration is available for encoding."""
        try:
            from .gpu_acceleration import check_gpu_support
            return check_gpu_support()
        except Exception as e:
            logger.warning(f"Error checking GPU availability: {e}")
            return False

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


# class VideoLogger:
#     """Enhanced logger for video processing"""

#     def __init__(self):
#         self._last_progress = 0
#         self._progress_shown = False
#         self._unified_logger = setup_application_logging()

#     def error(self, message: str, **kwargs):
#         self._unified_logger.logger.error(message, **kwargs)

#     def warning(self, message: str, **kwargs):
#         self._unified_logger.logger.warning(message, **kwargs)

#     def info(self, message: str, **kwargs):
#         self._unified_logger.logger.info(message, **kwargs)

#     def debug(self, message: str, **kwargs):
#         self._unified_logger.logger.debug(message, **kwargs)

#     def success(self, message: str, **kwargs):
#         self._unified_logger.logger.success(message, **kwargs)


class VideoProcessor(BaseProcessor):
    def __init__(self, logger=None, use_gpu: bool = True, gpu_settings: dict = None, settings: dict = VIDEO_SETTINGS):
        # Use provided logger or get a properly configured one
        self.logger = logger or get_logger('video')
        # Don't call super() with logger param since we've already set it
        super().__init__()

        self.settings = settings
        self.use_gpu = use_gpu
        self.gpu_settings = gpu_settings or {}

        # Log initialization
        self.logger.debug(f"Initializing VideoProcessor with GPU: {use_gpu}")
        if gpu_settings:
            self.logger.debug(f"GPU settings: {gpu_settings}")

        # Set up encoder based on settings
        self._setup_encoder()

        # Set up thread management
        self.max_workers = min(os.cpu_count() or 4, 4)
        self.executor = ThreadPoolExecutor(max_workers=self.max_workers)
        self.progress_callback = None
        self.resource_monitor = ResourceMonitor()

        # Initialize stats tracking
        self.processed_files = 0
        self.failed_files = []

        # Static configuration
        self.max_compression_attempts = 5
        self.min_scale = 0.4
        self.target_size_mb = 15.0
        self.base_size_tolerance = 0.10  # 10% tolerance
        self.extreme_compression_threshold = 1.2  # 20% over target

        # Dimension cache
        self.dimension_cache = {}

    @performance_monitor
    def _setup_encoder(self):
        """Set up video encoder with appropriate GPU settings."""
        try:
            self.encoder = VideoEncoder(self.settings)
            # Override GPU availability based on what was passed in
            if not self.use_gpu:
                self.logger.info("GPU acceleration disabled by configuration")
                self.encoder.is_gpu_available = False
                self.encoder._setup_encoder()  # Re-setup with GPU disabled
            elif self.gpu_settings:
                # Apply any specific GPU settings passed from outside
                self.encoder.is_gpu_available = True
                # Update encoder preferences if specific GPU type is known
                if 'preferred_encoder' in self.gpu_settings:
                    preferred = self.gpu_settings['preferred_encoder']
                    if preferred is not None:
                        self.logger.info(
                            f"Set {preferred.upper()} as preferred encoder")
                        # Update encoder settings if the preferred encoder is available
                        for encoder_type in self.settings:
                            if encoder_type in ['gpu', 'cpu']:
                                encoders = self.settings[encoder_type].get(
                                    'encoders', [])
                                for encoder in encoders:
                                    if preferred in encoder.lower():
                                        self.settings[encoder_type]['encoders'] = [encoder] + [
                                            e for e in encoders if e != encoder
                                        ]
                    else:
                        self.logger.warning(
                            "Preferred encoder is None, using default")
                self.encoder._setup_encoder()  # Re-setup with updated settings

            # Log encoder configuration
            self.logger.debug(
                f"Encoder configured: mode={self.encoder.mode}, encoder={self.encoder.encoder}")
        except Exception as e:
            self.logger.error(f"Error setting up encoder: {e}", exc_info=True)
            # Create a basic encoder in CPU mode as fallback
            self.encoder = VideoEncoder(self.settings)
            self.encoder.is_gpu_available = False
            self.encoder._setup_encoder()

    def _get_temp_path(self, original_path: Path, suffix: str = "") -> Path:
        """Generate a temporary file path in the temp directory."""
        try:
            timestamp = int(time.time() * 1000)
            return Path(TEMP_FILE_DIR) / f"temp_{timestamp}_{original_path.stem}{suffix}{original_path.suffix}"
        except Exception as e:
            self.logger.error(
                f"Failed to create temp path: {e}", exc_info=True)
            # Fallback
            return Path(TEMP_FILE_DIR) / f"temp_{int(time.time())}_{suffix}.mp4"

    def set_progress_callback(self, callback: Callable[[int, int, str], None]) -> None:
        """Set a callback function to report processing progress."""
        self.progress_callback = callback
        self.logger.debug("Progress callback set")

    def _report_progress(self, current: int, total: int, file_name: str) -> None:
        """Report processing progress through callback."""
        if self.progress_callback:
            try:
                self.progress_callback(current, total, file_name)
            except Exception as e:
                self.logger.error(
                    f"Progress callback error: {e}", exc_info=True)

    @performance_monitor
    def process_videos(self, input_dir: Path, output_dir: Path, target_size_mb: float) -> List[Path]:
        """Process videos with improved error handling and feedback."""
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        failed_files = []

        # Store target size for other methods to use
        self.target_size_mb = target_size_mb

        video_files = []
        for fmt in SUPPORTED_VIDEO_FORMATS:
            video_files.extend(input_dir.glob(f'*{fmt}'))

        if not video_files:
            self.logger.info("No video files found to process")
            return []

        # Sort files by size for better processing order
        video_files.sort(key=lambda x: x.stat().st_size, reverse=True)
        total_files = len(video_files)
        self.logger.info(f"Found {total_files} video files to process")

        for idx, video_file in enumerate(video_files, 1):
            try:
                original_size = video_file.stat().st_size / (1024 * 1024)
                output_path = output_dir / f"{video_file.stem}.mp4"

                self.logger.info(
                    f"\nProcessing {idx}/{total_files}: {video_file.name} ({original_size:.2f}MB)"
                )

                # Validate input file
                if not self._validate_input_file(video_file):
                    self.logger.error(
                        f"Invalid or corrupted file: {video_file.name}")
                    failed_files.append(video_file)
                    continue

                # Skip if output exists and meets size requirement
                if output_path.exists():
                    existing_size = output_path.stat().st_size / (1024 * 1024)
                    if existing_size <= target_size_mb:
                        self.logger.info(
                            f"Skipping {video_file.name} - Already compressed ({existing_size:.2f}MB)"
                        )
                        continue

                # Process the video
                if not self._process_single_video(video_file, output_path, target_size_mb):
                    self.logger.error(
                        f"Failed to create output for {video_file.name}"
                    )
                    failed_files.append(video_file)
                    continue

                # Verify the output
                if not self._verify_output(output_path, target_size_mb):
                    self.logger.error(
                        f"Output verification failed for {video_file.name}"
                    )
                    failed_files.append(video_file)
                    if output_path.exists():
                        output_path.unlink()

                self._report_progress(idx, total_files, video_file.name)

                # Log successful processing
                if output_path.exists():
                    self.processed_files += 1
                    self.logger.success(
                        f"Processed {video_file.name} - Original: {original_size:.2f}MB, Final: {output_path.stat().st_size / (1024 * 1024):.2f}MB"
                    )

            except Exception as e:
                self.logger.error(
                    f"Error processing {video_file.name}: {str(e)}", exc_info=True)
                failed_files.append(video_file)

        # Log summary
        self.logger.info(
            f"Video processing complete. Processed: {self.processed_files}, Failed: {len(failed_files)}")
        return failed_files

    def _validate_input_file(self, file_path: Path) -> bool:
        """Validate input video file."""
        try:
            # Check if file exists and is not empty
            if not file_path.exists() or file_path.stat().st_size == 0:
                self.logger.warning(
                    f"File doesn't exist or is empty: {file_path}")
                return False

            # Check if file is a valid video using ffprobe
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_type',
                '-of', 'json',
                str(file_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.warning(
                    f"FFprobe returned error code {result.returncode} for {file_path}")
                return False

            data = json.loads(result.stdout)
            is_valid = bool(data.get('streams'))
            if not is_valid:
                self.logger.warning(
                    f"No valid video streams found in {file_path}")
            return is_valid

        except json.JSONDecodeError:
            self.logger.error(
                f"Invalid JSON response from ffprobe for {file_path}", exc_info=True)
            return False
        except Exception as e:
            self.logger.error(
                f"Validation error for {file_path.name}: {e}", exc_info=True)
            return False

    def _verify_output(self, output_path: Path, target_size_mb: float) -> bool:
        """Verify the output video is valid and meets requirements."""
        try:
            if not output_path.exists():
                self.logger.warning(
                    f"Output file does not exist: {output_path}")
                return False

            # Check file size
            output_size = output_path.stat().st_size / (1024 * 1024)
            if output_size > target_size_mb * 1.1:  # Allow 10% tolerance
                self.logger.warning(
                    f"Output file too large: {output_size:.2f}MB > {target_size_mb * 1.1:.2f}MB")
                return False

            # Verify video integrity
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-show_entries', 'stream=codec_type',
                '-of', 'json',
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.warning(
                    f"FFprobe verification failed with code {result.returncode}")
                return False

            return True

        except Exception as e:
            self.logger.error(f"Output verification error: {e}", exc_info=True)
            return False

    @retry_video_processing
    def _process_single_video(self, input_path: Path, output_path: Path, target_size_mb: float) -> bool:
        """Process a single video with dynamic quality adjustment and timeout."""
        try:
            with ResourceGuard(self.resource_monitor):
                # Add timeout for metadata fetching
                metadata = self.get_video_metadata(input_path, timeout=30)
                if not metadata:
                    self.logger.error(
                        f"Failed to get metadata for {input_path}")
                    return False

                # Log more detailed progress information
                self.logger.info(
                    f"Processing {input_path.name}\n"
                    f"Size: {metadata['size_mb']:.2f}MB\n"
                    f"Duration: {metadata.get('duration', 0):.1f}s\n"
                    f"Resolution: {metadata.get('width', 0)}x{metadata.get('height', 0)}\n"
                    f"Target size: {target_size_mb}MB"
                )

                # Add a processing timeout based on file size
                # 5 minutes minimum, or 2 seconds per MB
                timeout = max(300, int(metadata['size_mb'] * 2))

                # Safely access metadata values with defaults
                size_mb = metadata.get('size_mb', 0)
                width = metadata.get('width', 0)
                height = metadata.get('height', 0)
                duration = metadata.get('duration', 0)
                frame_rate = metadata.get('frame_rate', 30)
                bitrate = metadata.get('bitrate', 0)

                # Validate essential metadata
                if not all([width, height, duration]):
                    self.logger.error(
                        f"Invalid metadata for {input_path}: missing essential values")
                    return False

                metadata = VideoMetadata(
                    width=width,
                    height=height,
                    frame_rate=frame_rate,
                    has_audio=metadata.get('has_audio', False),
                    size_mb=size_mb,
                    duration=duration,
                    bitrate=bitrate
                )

                # Adjust tolerance based on original file size
                size_tolerance = self._get_size_tolerance(metadata.size_mb)
                self.logger.debug(f"Video info: {metadata.width}x{metadata.height}, {
                    metadata.frame_rate}fps, {metadata.duration: .1f}s")
                self.logger.debug(f"Target size: {target_size_mb}MB(tolerance: {
                    size_tolerance*100: .1f} % )")

                # For very large files, try two-pass encoding first
                if metadata.size_mb > 100 and metadata.size_mb / target_size_mb > 10:
                    self.logger.info(f"Large file detected({
                        metadata.size_mb: .2f}MB), attempting two-pass compression")
                    if self._two_pass_compression(input_path, output_path, target_size_mb, metadata):
                        return True
                    self.logger.info(
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
                        self.logger.debug(
                            f"Attempt {attempt+1}: size={current_size:.2f}MB, scale={scale:.2f}, crf={crf}")

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
                    self.logger.warning(
                        f"Could not reach target size of {target_size_mb}MB. Best achieved: {best_size:.2f}MB")
                    return True

                return False

        except TimeoutError:
            self.logger.error(f"Processing timed out for {input_path}")
            return False
        except VideoProcessingError as e:
            self.logger.error(f"Video processing error: {e}")
            return False
        except Exception as e:
            self.logger.exception(f"Processing failed for {input_path}: {e}")
            return False
        finally:
            # Cleanup temp files
            for temp_file in Path(TEMP_FILE_DIR).glob(f"temp_{input_path.stem}*"):
                try:
                    temp_file.unlink(missing_ok=True)
                except Exception as e:
                    self.logger.error(
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
        try:
            compression_ratio = metadata.size_mb / \
                target_size_mb if metadata.size_mb > 0 else 4

            if compression_ratio > 4:
                return 28
            elif compression_ratio > 2:
                return 24
            else:
                return 20
        except (AttributeError, TypeError, ZeroDivisionError):
            # Return a safe default if there are any errors
            return 23

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
            self.logger.debug(f"Close to target, using extreme compression: scale={
                new_scale: .2f}, crf={new_crf}")
        elif size_ratio > 2:
            new_scale = max(self.min_scale, current_scale * 0.8)
            new_crf = min(current_crf + 4, 35)
            self.logger.debug(f"Size ratio > 2, aggressive adjustment: scale={
                new_scale: .2f}, crf={new_crf}")
        elif size_ratio > 1.5:
            new_scale = max(self.min_scale, current_scale * 0.9)
            new_crf = min(current_crf + 2, 32)
            self.logger.debug(f"Size ratio > 1.5, moderate adjustment: scale={
                new_scale: .2f}, crf={new_crf}")
        else:
            new_scale = max(self.min_scale, current_scale * 0.95)
            new_crf = min(current_crf + 1, 30)
            self.logger.debug(f"Size ratio <= 1.5, fine adjustment: scale={
                new_scale: .2f}, crf={new_crf}")

        return new_scale, new_crf

    def _compress_attempt(self, input_path: Path, output_path: Path,
                          metadata: VideoMetadata, target_bitrate: int,
                          scale: float, crf: int) -> bool:
        """Attempt compression with given parameters."""
        try:
            new_width = int(metadata.width * scale // 2 * 2)
            new_height = int(metadata.height * scale // 2 * 2)

            self.logger.debug(f"Compression attempt with: scale={scale:.2f} ({new_width}x{new_height}), "
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
            if self.use_gpu and self.encoder.is_gpu_available:
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
            self.logger.exception(f"Compression attempt failed: {e}")
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
        if self.use_gpu and self.encoder.is_gpu_available:
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
            if self.use_gpu and self.encoder.is_gpu_available:
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
        """Run FFmpeg command with proper error handling and dedicated logging."""
        from .logging_system import setup_ffmpeg_logging, run_ffmpeg_command

        try:
            # Make sure hide_banner is included
            if '-hide_banner' not in command:
                command.insert(1, '-hide_banner')

            # Use the dedicated FFmpeg logging system
            return run_ffmpeg_command(command)

        except Exception as e:
            self.logger.exception(f"FFmpeg execution failed: {e}")
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
            self.logger.error(f"Failed to create temp file: {e}")
            raise

    @performance_monitor
    def get_video_metadata(self, file_path: Path, timeout: int = 30) -> Optional[Dict[str, Any]]:
        """Get video metadata with timeout."""
        try:
            # Check cache first
            cache_key = str(file_path)
            if hasattr(self, 'metadata_cache') and cache_key in self.metadata_cache:
                self.logger.debug(f"Using cached metadata for {file_path}")
                return self.metadata_cache[cache_key]

            # Use ffprobe to get video metadata
            self.logger.debug(f"Retrieving metadata for {file_path}")
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                str(file_path)
            ]

            # Add timeout to subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout  # Add timeout parameter
            )

            if result.returncode != 0:
                self.logger.error(
                    f"Failed to get metadata: {result.stderr}")
                return None

            try:
                metadata = json.loads(result.stdout)
            except json.JSONDecodeError as e:
                self.logger.error(
                    f"Invalid JSON from ffprobe: {e}", exc_info=True)
                return None

            # Validate required metadata fields
            if not metadata or 'streams' not in metadata or not metadata['streams']:
                self.logger.error(f"Invalid metadata format for {file_path}")
                return None

            video_stream = next(
                (s for s in metadata['streams'] if s.get('codec_type') == 'video'), None)
            if not video_stream:
                self.logger.error(f"No video stream found in {file_path}")
                return None

            file_size = os.path.getsize(
                file_path) / (1024 * 1024)  # Convert to MB

            # Create metadata dict
            metadata_dict = {
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'duration': float(video_stream.get('duration', 0)),
                'bitrate': int(video_stream.get('bit_rate', 0)),
                'codec': video_stream.get('codec_name', 'unknown'),
                'frame_rate': float(eval(video_stream.get('r_frame_rate', '30/1'))),
                'has_audio': any(s.get('codec_type') == 'audio' for s in metadata['streams']),
                'size_mb': file_size
            }

            # Cache the result
            if not hasattr(self, 'metadata_cache'):
                self.metadata_cache = {}
            self.metadata_cache[cache_key] = metadata_dict

            return metadata_dict

        except subprocess.TimeoutExpired:
            self.logger.error(
                f"Metadata extraction timed out for {file_path}")
            return None
        except (subprocess.SubprocessError, KeyError, ValueError) as e:
            self.logger.error(
                f"Error extracting metadata from {file_path}: {str(e)}", exc_info=True)
            return None

    def process_video(self, input_path: Path, output_path: Path) -> bool:
        try:
            metadata = self.get_video_metadata(input_path)
            if not metadata:
                self.logger.error(
                    f"Could not process {input_path}: Failed to get metadata")
                return False

            # Calculate target bitrate
            target_size_bytes = 15 * 1024 * 1024  # 15MB
            duration = metadata.get('duration', 0)
            if duration <= 0:
                self.logger.error(
                    f"Invalid duration in metadata for {input_path}")
                return False

            target_bitrate = int((target_size_bytes * 8) / duration)

            # Construct ffmpeg command with validated parameters
            cmd = [
                'ffmpeg', '-y',
                '-i', str(input_path),
                '-c:v', 'h264_nvenc' if self.use_gpu else 'libx264',
                '-b:v', f'{target_bitrate}',
                '-preset', 'medium',
                '-movflags', '+faststart',
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error(
                    f"FFmpeg error processing {input_path}: {result.stderr}")
                return False

            return output_path.exists() and output_path.stat().st_size > 0

        except Exception as e:
            self.logger.error(f"Error processing {input_path}: {str(e)}")
            return False

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
            self.logger.error(f"Failed to get duration for {video_path}: {e}")
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
            self.logger.error(
                f"Failed to get dimensions for {video_path}: {e}")

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
            self.logger.error(f"Failed to get frame rate: {e}")
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
            self.logger.error(f"Failed to get file size: {e}")
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
            self.logger.info("Starting two-pass compression...")
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

            if self.use_gpu and self.encoder.is_gpu_available:
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
            self.logger.info("Running first pass...")
            with ProcessManager(pass1_command, timeout=self.settings['general']['timeout']) as pm1:
                if not pm1.run():
                    self.logger.error("First pass failed")
                    return False

            self.logger.info("Running second pass...")
            with ProcessManager(pass2_command, timeout=self.settings['general']['timeout']) as pm2:
                if not pm2.run():
                    self.logger.error("Second pass failed")
                    return False

            if temp_output.exists():
                self.logger.debug(
                    "Two-pass compression completed successfully")
                shutil.move(str(temp_output), str(output_path))
                return True

            return False

        except Exception as e:
            self.logger.exception(f"Two-pass compression failed: {e}")
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

    def _create_gif_aggressive(self, file_path: Path, output_path: Path, fps: int, dimensions: Tuple[int, int]) -> bool:
        """Create GIF with dynamically adjusted aggressive compression settings."""
        original_size = self.get_file_size(file_path)
        best_result_size = float('inf')
        best_result_path = None
        target_size = self.compression_settings['min_size_mb']

        try:
            # Create a unique temp directory for this process
            temp_dir = Path(TEMP_FILE_DIR) / \
                f"aggressive_{file_path.stem}_{int(time.time())}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            for config in self._get_compression_configs(original_size):
                if self._should_exit():
                    break

                try:
                    temp_palette = temp_dir / f"palette_{config['colors']}.png"
                    temp_output = temp_dir / f"output_{config['colors']}.gif"

                    # Generate palette
                    palette_cmd = [
                        'ffmpeg', '-i', str(file_path),
                        '-vf', f'fps={config["fps"]},scale={dimensions[0]*config["scale"]}:{dimensions[1]*config["scale"]}:flags=lanczos,palettegen=max_colors={config["colors"]}',
                        '-y', str(temp_palette)
                    ]

                    if not self._run_ffmpeg(palette_cmd):
                        continue

                    # Create GIF using palette
                    gif_cmd = [
                        'ffmpeg', '-i', str(file_path),
                        '-i', str(temp_palette),
                        '-filter_complex',
                        f'fps={config["fps"]},scale={dimensions[0]*config["scale"]}:{dimensions[1]*config["scale"]}:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5',
                        '-y', str(temp_output)
                    ]

                    if not self._run_ffmpeg(gif_cmd):
                        continue

                    # Check if file was created and optimize with gifsicle
                    if temp_output.exists():
                        # Optimize with gifsicle
                        optimize_cmd = [
                            'gifsicle',
                            '--optimize=3',
                            '--lossy=' + str(config['lossy']),
                            '--colors', str(config['colors']),
                            str(temp_output),
                            '-o', str(temp_output)
                        ]

                        if not self._run_ffmpeg(optimize_cmd):
                            continue

                        current_size = self.get_file_size(temp_output)
                        self.logger.info(
                            f"Aggressive attempt result: {current_size:.2f}MB")

                        # Update best result if this is better
                        if current_size < best_result_size:
                            if best_result_path and best_result_path.exists():
                                try:
                                    best_result_path.unlink()
                                except Exception:
                                    pass
                            best_result_size = current_size
                            best_result_path = temp_output
                            # Create a copy of the best result
                            best_copy = temp_dir / \
                                f"best_{config['colors']}.gif"
                            shutil.copy2(temp_output, best_copy)
                            best_result_path = best_copy

                        # If we've reached target size, we can stop
                        if current_size <= target_size:
                            shutil.copy2(best_result_path, output_path)
                            return True

                except Exception as e:
                    self.logger.error(
                        f"Error during aggressive compression attempt: {e}")
                    continue

            # If we have a best result but didn't reach target, use it anyway
            if best_result_path and best_result_path.exists():
                if best_result_size < original_size:
                    shutil.copy2(best_result_path, output_path)
                    self.logger.warning(
                        f"Using best achieved result: {best_result_size:.2f}MB")
                    return True

            return False

        except Exception as e:
            self.logger.error(f"Aggressive GIF creation failed: {str(e)}")
            return False
        finally:
            # Clean up temp directory and all its contents
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                self.logger.error(f"Failed to cleanup temp directory: {e}")

    def _get_compression_configs(self, original_size: float) -> List[Dict]:
        """Get compression configurations based on file size."""
        configs = []

        if original_size > 50:  # Very large files
            configs.extend([
                {'scale': 0.3, 'colors': 256, 'fps': 12, 'lossy': 100},
                {'scale': 0.4, 'colors': 192, 'fps': 15, 'lossy': 90}
            ])
        elif original_size > 25:  # Large files
            configs.extend([
                {'scale': 0.4, 'colors': 256, 'fps': 15, 'lossy': 90},
                {'scale': 0.5, 'colors': 192, 'fps': 20, 'lossy': 80}
            ])
        else:  # Moderate size files
            configs.extend([
                {'scale': 0.6, 'colors': 256, 'fps': 20, 'lossy': 80},
                {'scale': 0.7, 'colors': 256, 'fps': 25, 'lossy': 70}
            ])

        return configs

    # ...rest of existing code...
