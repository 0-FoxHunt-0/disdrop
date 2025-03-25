#!/usr/bin/env python3
import os
import logging
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import threading
import time
import concurrent.futures
import signal
from typing import Dict, Optional, List, Set, Tuple

# Import custom modules
from src.gpu_detector import GPUDetector, AccelerationType
from src.logging_system import LoggingSystem


class VideoProcessor:
    """
    Advanced video processor class that handles video detection, conversion, and optimization
    with GPU acceleration when available.

    Features:
    - Automatic detection of all video files in input directory
    - Intelligent conversion of non-MP4 files to MP4 format
    - Smart caching to avoid reprocessing files with identical settings
    - Hardware-accelerated processing using available GPU resources
    - Quality-focused optimization using efficient parameters
    - Supports batch processing with configurable thread count
    """

    # Define video file extensions to process
    VIDEO_EXTENSIONS = {
        '.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.webm',
        '.mpeg', '.mpg', '.m4v', '.3gp', '.ts', '.mts', '.m2ts'
    }

    # Output format
    OUTPUT_FORMAT = '.mp4'

    def __init__(self, config: Dict = None):
        """
        Initialize the video processor with configuration.

        Args:
            config: Configuration dictionary for the processor
        """
        self.config = config or {}

        # Get the logging system
        self.logging_system = LoggingSystem(self.config)
        self.logger = self.logging_system.get_logger('video_processor')

        # Start a section for initialization
        self.logging_system.start_new_log_section(
            "Video Processor Initialization")

        # Set up directories
        self.input_dir = Path(self.config.get(
            'directories', {}).get('input', './input')).resolve()
        self.output_dir = Path(self.config.get(
            'directories', {}).get('output', './output')).resolve()
        self.temp_dir = Path(self.config.get(
            'directories', {}).get('temp', './temp')).resolve()

        # Get batch processing settings from config
        self.batch_size = self.config.get(
            'processing', {}).get('batch_size', 1)
        self.num_threads = self.config.get('processing', {}).get('threads', 1)

        # Get video-specific processing settings if available
        video_processing = self.config.get('video', {}).get('processing', {})
        if video_processing:
            self.batch_size = video_processing.get(
                'batch_size', self.batch_size)
            self.num_threads = video_processing.get(
                'threads', self.num_threads)

        # Ensure directories exist
        self._ensure_directories()

        # Initialize GPU detector
        self.gpu_detector = GPUDetector()
        self.gpu_types, self.acceleration_types = self.gpu_detector.detect()
        self.preferred_acceleration = self.gpu_detector.get_preferred_acceleration()

        # Check for any dxdiag files that might have been generated during GPU detection
        self.logging_system.find_and_move_dxdiag_file()

        self.logger.info(
            f"Video processor initialized with {self.preferred_acceleration.name} acceleration")
        self.logger.info(
            f"Using {self.num_threads} threads and batch size of {self.batch_size}")

        # Cache for processed files to avoid duplicate processing
        self.processed_cache_file = self.output_dir / '.processed_cache.json'
        self.processed_cache = self._load_processed_cache()

        # Lock for cache file access
        self.cache_lock = threading.Lock()

        # Flag for shutdown detection - will be set by ResourceManager
        self.shutdown_requested = False

        # Current active processes
        self.active_processes = []
        self.active_processes_lock = threading.Lock()

        # Set up signal handlers for graceful shutdown
        self._setup_signal_handlers()

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        # Store original handlers to restore later
        self.original_sigint_handler = signal.getsignal(signal.SIGINT)
        self.original_sigterm_handler = signal.getsignal(signal.SIGTERM)

        # Set custom handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle interruption signals for graceful shutdown."""
        self.logger.warning(
            f"Received signal {signum}, initiating graceful shutdown...")

        # Set shutdown flag
        self.shutdown_requested = True

        # Terminate any active processes
        self._terminate_active_processes()

        # Save processing cache to preserve work done so far
        self._save_processed_cache()

        # Log termination
        self.logger.info(
            "Graceful shutdown initiated. Waiting for tasks to complete...")

    def _terminate_active_processes(self):
        """Terminate any active FFmpeg processes."""
        with self.active_processes_lock:
            for process in self.active_processes:
                if process and process.poll() is None:  # If process is still running
                    try:
                        self.logger.info(f"Terminating process {process.pid}")
                        process.terminate()
                    except Exception as e:
                        self.logger.error(f"Error terminating process: {e}")

            # Clear the list
            self.active_processes.clear()

    def _register_process(self, process):
        """Register an active subprocess."""
        with self.active_processes_lock:
            self.active_processes.append(process)

    def _unregister_process(self, process):
        """Remove a subprocess from the active list."""
        with self.active_processes_lock:
            if process in self.active_processes:
                self.active_processes.remove(process)

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for directory in [self.input_dir, self.output_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")

    def _load_processed_cache(self) -> Dict:
        """
        Load the cache of processed files.

        Returns:
            Dict: Cache of processed files with their settings hash
        """
        if self.processed_cache_file.exists():
            try:
                with open(self.processed_cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Could not load processed cache: {e}")
                return {}
        return {}

    def _save_processed_cache(self) -> None:
        """Save the cache of processed files."""
        try:
            with self.cache_lock:
                with open(self.processed_cache_file, 'w') as f:
                    json.dump(self.processed_cache, f)
        except IOError as e:
            self.logger.warning(f"Could not save processed cache: {e}")

    def _get_settings_hash(self, file_path: Path) -> str:
        """
        Create a hash of the file and current processing settings.

        Args:
            file_path: Path to the file

        Returns:
            str: Hash representing the file and current settings
        """
        # Get file modification time and size
        stat = file_path.stat()
        file_info = f"{file_path.name}:{stat.st_mtime}:{stat.st_size}"

        # Get relevant settings from config that would affect output
        settings = {
            'resolution': self.config.get('video', {}).get('resolution', '1080p'),
            'bitrate': self.config.get('video', {}).get('bitrate', '2M'),
            'codec': self.config.get('codec', 'h264'),
            'crf': self.config.get('video', {}).get('crf', 23),
            'preset': self.config.get('video', {}).get('preset', 'medium')
        }

        # Create a hash of file info and settings
        settings_str = f"{file_info}:{json.dumps(settings, sort_keys=True)}"
        return hashlib.md5(settings_str.encode()).hexdigest()

    def find_video_files(self) -> Tuple[List[Path], List[Path]]:
        """
        Find all video files in the input directory.

        Returns:
            Tuple[List[Path], List[Path]]: Lists of MP4 and non-MP4 video files
        """
        mp4_files = []
        other_video_files = []

        for file_path in self.input_dir.glob('**/*'):
            if file_path.is_file() and file_path.suffix.lower() in self.VIDEO_EXTENSIONS:
                if file_path.suffix.lower() == self.OUTPUT_FORMAT:
                    mp4_files.append(file_path)
                else:
                    other_video_files.append(file_path)

        self.logger.info(
            f"Found {len(mp4_files)} MP4 files and {len(other_video_files)} other video files")
        return mp4_files, other_video_files

    def get_existing_output_files(self) -> Dict[str, Path]:
        """
        Get a mapping of base filename to output file path for existing output files.

        Returns:
            Dict[str, Path]: Mapping of base filename to output file path
        """
        existing_files = {}

        for file_path in self.output_dir.glob(f'*{self.OUTPUT_FORMAT}'):
            if file_path.is_file():
                # Use stem as the key to match against non-MP4 files
                existing_files[file_path.stem] = file_path

        return existing_files

    def get_ffmpeg_acceleration_args(self) -> List[str]:
        """
        Get the appropriate FFmpeg acceleration arguments with improved RTX support.

        Returns:
            List[str]: FFmpeg command line arguments for hardware acceleration
        """
        accel_type = self.preferred_acceleration

        if accel_type == AccelerationType.CUDA:
            # For CUDA, use specific output format for better compatibility
            return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
        elif accel_type == AccelerationType.METAL:
            return ["-hwaccel", "videotoolbox"]
        elif accel_type == AccelerationType.DIRECTML:
            return ["-hwaccel", "d3d11va"]
        elif accel_type == AccelerationType.OPENCL:
            return ["-hwaccel", "opencl"]
        elif accel_type == AccelerationType.ROCM:
            return ["-hwaccel", "vaapi"]  # AMD's VA-API implementation
        elif accel_type == AccelerationType.ONEAPI:
            return ["-hwaccel", "qsv"]  # Intel QuickSync
        else:  # CPU fallback
            return []  # No hardware acceleration

    def get_codec_args(self) -> List[str]:
        """
        Get the appropriate codec arguments with improved settings for RTX cards.

        Returns:
            List[str]: FFmpeg codec arguments
        """
        accel_type = self.preferred_acceleration

        # Get desired codec and parameters from config
        target_codec = self.config.get('codec', 'h264')
        preset = self.config.get('video', {}).get('preset', 'medium')
        # Lower is better quality, higher is smaller file
        crf = self.config.get('video', {}).get('crf', 23)

        if target_codec == 'h264':
            if accel_type == AccelerationType.CUDA:
                # NVIDIA NVENC settings with improved RTX presets
                # Check if it's an RTX card by looking at the model string
                rtx_card = any('RTX' in gpu for gpu in self.gpu_types if hasattr(
                    gpu, 'model') and gpu.model)

                if rtx_card:
                    # RTX cards support better quality presets
                    return [
                        "-c:v", "h264_nvenc",
                        # Options: p1 (fastest) to p7 (best quality)
                        "-preset", "p4",
                        "-tune", "hq",    # High quality tuning
                        "-profile:v", "high",
                        "-rc:v", "vbr_hq",  # High-quality VBR mode
                        "-cq:v", str(crf)  # Quality level
                    ]
                else:
                    # Fallback for non-RTX NVIDIA cards
                    return [
                        "-c:v", "h264_nvenc",
                        "-preset", "p2",  # More compatible preset
                        "-profile:v", "high",
                        "-rc:v", "vbr",
                        "-cq:v", str(crf)
                    ]
            elif accel_type == AccelerationType.METAL:
                # Apple VideoToolbox settings
                return [
                    "-c:v", "h264_videotoolbox",
                    "-profile:v", "high",
                    "-q:v", str(crf)
                ]
            elif accel_type == AccelerationType.ROCM:
                # AMD VA-API settings
                return [
                    "-c:v", "h264_vaapi",
                    "-profile", "high",
                    "-quality", "high",
                    "-qp", str(crf)
                ]
            elif accel_type == AccelerationType.ONEAPI:
                # Intel QuickSync settings
                return [
                    "-c:v", "h264_qsv",
                    "-preset", preset,
                    "-profile:v", "high",
                    "-global_quality", str(crf)
                ]
            else:
                # CPU x264 settings (most flexible and highest quality)
                return [
                    "-c:v", "libx264",
                    "-preset", preset,
                    "-profile:v", "high",
                    "-crf", str(crf)
                ]

        elif target_codec == 'h265' or target_codec == 'hevc':
            if accel_type == AccelerationType.CUDA:
                # Check if it's an RTX card
                rtx_card = any('RTX' in gpu for gpu in self.gpu_types if hasattr(
                    gpu, 'model') and gpu.model)

                if rtx_card:
                    # RTX cards have better HEVC encoding
                    return [
                        "-c:v", "hevc_nvenc",
                        "-preset", "p4",
                        "-tune", "hq",
                        "-profile:v", "main",
                        "-rc:v", "vbr_hq",
                        "-cq:v", str(crf)
                    ]
                else:
                    return [
                        "-c:v", "hevc_nvenc",
                        "-preset", "p2",
                        "-profile:v", "main",
                        "-rc:v", "vbr",
                        "-cq:v", str(crf)
                    ]
            elif accel_type == AccelerationType.METAL:
                return [
                    "-c:v", "hevc_videotoolbox",
                    "-profile:v", "main",
                    "-q:v", str(crf)
                ]
            elif accel_type == AccelerationType.ROCM:
                return [
                    "-c:v", "hevc_vaapi",
                    "-profile", "main",
                    "-quality", "high",
                    "-qp", str(crf)
                ]
            elif accel_type == AccelerationType.ONEAPI:
                return [
                    "-c:v", "hevc_qsv",
                    "-preset", preset,
                    "-profile:v", "main",
                    "-global_quality", str(crf)
                ]
            else:
                return [
                    "-c:v", "libx265",
                    "-preset", preset,
                    "-crf", str(crf)
                ]

        else:
            # For other codecs, use CPU encoding
            self.logger.warning(
                f"No hardware acceleration for codec {target_codec}, using CPU")
            return ["-c:v", f"lib{target_codec}", "-crf", str(crf), "-preset", preset]

    def convert_to_mp4(self, input_file: Path) -> Optional[Path]:
        """
        Convert a non-MP4 file to MP4 format with improved GPU fallback.

        Args:
            input_file: Path to the input video file

        Returns:
            Optional[Path]: Path to the converted MP4 file, or None if conversion failed
        """
        # Create output path in temp directory
        temp_output = self.temp_dir / f"{input_file.stem}{self.OUTPUT_FORMAT}"

        # Log file for FFmpeg output
        log_file_path = self.temp_dir / f"{input_file.stem}_ffmpeg.log"

        # Check if FFMPEG exists and is executable
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            self.logger.error(
                "FFmpeg executable not found in PATH. Please install FFmpeg.")
            return None

        self.logger.debug(f"Using FFmpeg at: {ffmpeg_path}")

        # Get video duration first to show progress
        duration_seconds = self._get_video_duration(input_file)
        if duration_seconds is None:
            self.logger.warning(
                f"Could not determine duration of {input_file}. Progress bar will be indeterminate.")

        # Initialize acceleration strategy
        # First try with full hardware acceleration (decode+encode)
        # If that fails, try with just encoding acceleration
        # If that fails too, fall back to CPU
        acceleration_attempts = ["full", "encode_only", "cpu"]

        for acceleration_type in acceleration_attempts:
            try:
                # Build FFmpeg command
                cmd = [ffmpeg_path, "-y"]

                # Add input options based on acceleration strategy
                if acceleration_type == "full" and self.preferred_acceleration != AccelerationType.CPU:
                    # Full hardware acceleration (decode + encode)
                    cmd.extend(self.get_ffmpeg_acceleration_args())
                elif acceleration_type == "encode_only" and self.preferred_acceleration != AccelerationType.CPU:
                    # Skip hardware decoding, just use hw encoding
                    self.logger.info(
                        "Trying with hardware encoding only (no accelerated decoding)")
                    # No hwaccel args for input

                # Add input file
                cmd.extend(["-i", str(input_file)])

                # Setup encoding parameters based on acceleration strategy
                if acceleration_type != "cpu" and self.preferred_acceleration != AccelerationType.CPU:
                    # Use hardware encoder
                    codec_args = self.get_codec_args()
                    # For conversion, use a faster preset
                    for i, arg in enumerate(codec_args):
                        if arg == "-preset" and i + 1 < len(codec_args):
                            if "nvenc" in codec_args[i-1]:  # For NVENC
                                codec_args[i + 1] = "p2"  # Faster preset
                            else:
                                # Generic fast preset
                                codec_args[i + 1] = "fast"
                    cmd.extend(codec_args)
                else:
                    # CPU encoding fallback
                    self.logger.info("Using CPU encoding fallback")
                    cmd.extend(
                        ["-c:v", "libx264", "-preset", "fast", "-crf", "23"])

                # Audio settings
                cmd.extend(["-c:a", "aac", "-b:a", "128k"])

                # Add progress output
                cmd.extend(["-progress", "pipe:1"])

                # Add output file
                cmd.append(str(temp_output))

                # Log the command
                accel_type_str = {
                    "full": "full hardware acceleration",
                    "encode_only": "hardware encoding only",
                    "cpu": "CPU encoding"
                }[acceleration_type]

                self.logger.info(
                    f"Converting video to MP4 using {accel_type_str}: {input_file} → {temp_output}")
                self.logger.debug(f"FFmpeg command: {' '.join(cmd)}")

                # Execute the command
                with open(log_file_path, 'w') as log_file:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=log_file,
                        universal_newlines=True,
                        bufsize=1,
                        text=True
                    )

                    # Register and monitor the process
                    self._register_process(process)
                    try:
                        self._process_ffmpeg_output(
                            process, input_file.name, duration_seconds)
                    except KeyboardInterrupt:
                        self.logger.warning("Conversion interrupted by user")
                        process.terminate()
                        if temp_output.exists():
                            temp_output.unlink()
                        raise
                    finally:
                        self._unregister_process(process)

                    # Check exit code
                    if process.wait() != 0:
                        raise subprocess.CalledProcessError(
                            process.returncode, cmd)

                # Verify output file
                if not temp_output.exists() or temp_output.stat().st_size == 0:
                    raise RuntimeError(
                        f"Output file missing or empty: {temp_output}")

                # Success! Break out of the loop
                self.logger.success(
                    f"Video conversion successful using {accel_type_str}: {temp_output}")

                # Clean up log file on success
                if log_file_path.exists():
                    try:
                        log_file_path.unlink()
                    except Exception as e:
                        self.logger.warning(
                            f"Could not remove log file {log_file_path}: {e}")

                return temp_output

            except (subprocess.CalledProcessError, RuntimeError) as e:
                # Read error from log file
                error_msg = ""
                if log_file_path.exists():
                    try:
                        with open(log_file_path, 'r') as log_file:
                            error_lines = log_file.readlines()
                            error_msg = "".join(
                                error_lines[-10:]) if error_lines else "No error details"
                    except Exception:
                        error_msg = str(e)
                else:
                    error_msg = str(e)

                # If we're not on the last attempt, try next fallback
                if acceleration_type != "cpu":
                    self.logger.warning(
                        f"Conversion with {acceleration_type} failed. Error: {error_msg}")
                    self.logger.info(
                        f"Falling back to next acceleration method...")

                    # Clean up failed output
                    if temp_output.exists():
                        temp_output.unlink()
                else:
                    # All attempts failed
                    self.logger.error(
                        f"All conversion attempts failed: {error_msg}")
                    if temp_output.exists():
                        temp_output.unlink()
                    return None

            except Exception as e:
                self.logger.error(
                    f"Error during video conversion: {e}", exc_info=True)
                if temp_output.exists():
                    temp_output.unlink()
                return None

        # We should never reach here, but just in case
        return None

    def _get_video_duration(self, input_file: Path) -> Optional[float]:
        """
        Get the duration of a video file in seconds.

        Args:
            input_file: Path to the video file

        Returns:
            float: Duration in seconds, or None if it couldn't be determined
        """
        ffprobe_path = shutil.which("ffprobe")
        if ffprobe_path is None:
            self.logger.warning(
                "ffprobe executable not found, cannot determine video duration")
            return None

        try:
            cmd = [
                ffprobe_path,
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(input_file)
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            duration = float(result.stdout.strip())
            return duration
        except (subprocess.SubprocessError, ValueError) as e:
            self.logger.warning(f"Could not determine video duration: {e}")
            return None

    def _process_ffmpeg_output(self, process, filename: str, duration_seconds: Optional[float]):
        """
        Process FFmpeg progress output and update progress bar.

        Args:
            process: Subprocess object
            filename: Name of the file being processed
            duration_seconds: Duration of the video in seconds
        """
        import sys
        from datetime import timedelta

        progress_data = {}
        bar_length = 50  # Length of progress bar

        # Print initial progress bar
        if duration_seconds:
            sys.stdout.write(
                f"\rProcessing {filename}: [{'.' * bar_length}] 0.0% (0/{int(duration_seconds)}s)")
        else:
            sys.stdout.write(
                f"\rProcessing {filename}: [{'.' * bar_length}] ?%")
        sys.stdout.flush()

        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break

            if line:
                # Parse progress information
                line = line.strip()
                if '=' in line:
                    key, value = line.split('=', 1)
                    progress_data[key] = value

                    # Update progress bar if we have time information
                    if 'out_time_ms' in progress_data and duration_seconds:
                        try:
                            current_time = float(
                                progress_data['out_time_ms']) / 1_000_000
                            progress = min(
                                current_time / duration_seconds, 1.0)

                            # Format time as H:MM:SS
                            current_formatted = str(
                                timedelta(seconds=int(current_time)))
                            if current_formatted.startswith('0:'):
                                # Remove leading 0:
                                current_formatted = current_formatted[2:]

                            duration_formatted = str(
                                timedelta(seconds=int(duration_seconds)))
                            if duration_formatted.startswith('0:'):
                                # Remove leading 0:
                                duration_formatted = duration_formatted[2:]

                            # Create the progress bar
                            filled_length = int(bar_length * progress)
                            bar = '=' * filled_length + '.' * \
                                (bar_length - filled_length)

                            # Update the progress display
                            sys.stdout.write(
                                f"\rProcessing {filename}: [{bar}] {progress * 100:.1f}% ({current_formatted}/{duration_formatted})")
                            sys.stdout.flush()
                        except (ValueError, ZeroDivisionError):
                            pass
                    elif 'progress' in progress_data:
                        # Fallback for when duration isn't known
                        if progress_data['progress'] == 'end':
                            sys.stdout.write(
                                f"\rProcessing {filename}: [{'=' * bar_length}] 100.0% (Complete!)")
                            sys.stdout.flush()

        # Ensure we end with a newline
        sys.stdout.write("\n")
        sys.stdout.flush()

    def optimize_mp4(self, input_file: Path) -> Optional[Path]:
        """
        Optimize an MP4 file using appropriate hardware acceleration with improved fallback.

        Args:
            input_file: Path to the input MP4 file

        Returns:
            Optional[Path]: Path to the optimized file, or None if optimization failed
        """
        # Determine output filename and path
        if input_file.parent == self.temp_dir:
            # Use original name for temp files
            output_file = self.output_dir / input_file.name
        else:
            # Use "optimized_" prefix for files already in input dir
            output_file = self.output_dir / f"optimized_{input_file.name}"

        # Log file for FFmpeg output
        log_file_path = self.temp_dir / \
            f"{input_file.stem}_optimize_ffmpeg.log"

        # Check if this file already has been processed with current settings
        settings_hash = self._get_settings_hash(input_file)
        if input_file.name in self.processed_cache and self.processed_cache[input_file.name] == settings_hash:
            self.logger.info(
                f"Skipping already processed file with same settings: {input_file}")
            if output_file.exists():
                return output_file
            # If cache says processed but file doesn't exist, reprocess
            self.logger.warning(
                f"Cache indicates file was processed but output doesn't exist: {output_file}")

        # Check if FFMPEG exists and is executable
        ffmpeg_path = shutil.which("ffmpeg")
        if ffmpeg_path is None:
            self.logger.error(
                "FFmpeg executable not found in PATH. Please install FFmpeg.")
            return None

        self.logger.debug(f"Using FFmpeg at: {ffmpeg_path}")

        # Get video duration first to show progress
        duration_seconds = self._get_video_duration(input_file)
        if duration_seconds is None:
            self.logger.warning(
                f"Could not determine duration of {input_file}. Progress bar will be indeterminate.")

        # Get video parameters from config
        resolution = self.config.get('video', {}).get('resolution', '1080p')
        bitrate = self.config.get('video', {}).get('bitrate', '2M')

        # Map resolution to actual dimensions
        resolution_map = {
            '480p': '854:480',
            '720p': '1280:720',
            '1080p': '1920:1080',
            '2160p': '3840:2160',
            '4k': '3840:2160'
        }

        # Handle custom resolution format
        if resolution.lower() in resolution_map:
            scale = resolution_map.get(resolution.lower())
        elif 'x' in resolution or ':' in resolution:
            scale = resolution.replace(
                'x', ':') if 'x' in resolution else resolution
            self.logger.info(f"Using custom resolution scale: {scale}")
        else:
            scale = resolution_map.get('720p')
            self.logger.warning(
                f"Unrecognized resolution format '{resolution}', defaulting to 720p")

        # Try different acceleration strategies in sequence
        # 1. Full hardware acceleration (decode+encode)
        # 2. Hardware encoding only (no accelerated decoding)
        # 3. CPU fallback
        acceleration_attempts = ["full", "encode_only", "cpu"]

        for acceleration_type in acceleration_attempts:
            try:
                # Build FFmpeg command
                cmd = [ffmpeg_path, "-y"]

                # Add input options based on acceleration strategy
                if acceleration_type == "full" and self.preferred_acceleration != AccelerationType.CPU:
                    # Full hardware acceleration (decode + encode)
                    cmd.extend(self.get_ffmpeg_acceleration_args())
                elif acceleration_type == "encode_only" and self.preferred_acceleration != AccelerationType.CPU:
                    # Skip hardware decoding, just use hw encoding
                    self.logger.info(
                        "Trying with hardware encoding only (no accelerated decoding)")
                    # No hwaccel args for input

                # Add input file
                cmd.extend(["-i", str(input_file)])

                # Add scaling filter based on acceleration type
                if acceleration_type == "full" and self.preferred_acceleration == AccelerationType.CUDA:
                    # For CUDA full acceleration, use hardware scaling
                    cmd.extend(
                        ["-vf", f"scale_cuda={scale.replace(':', 'x')}"])
                else:
                    # For other modes, use standard scaling
                    cmd.extend(["-vf", f"scale={scale}"])

                # Add bitrate if specified
                if bitrate and not self.config.get('video', {}).get('use_crf_only', False):
                    cmd.extend(["-b:v", bitrate])

                # Add encoding parameters based on acceleration strategy
                if acceleration_type != "cpu" and self.preferred_acceleration != AccelerationType.CPU:
                    # Use hardware encoder
                    cmd.extend(self.get_codec_args())
                else:
                    # CPU encoding fallback
                    self.logger.info("Using CPU encoding fallback")
                    cmd.extend([
                        "-c:v", "libx264",
                        "-preset", self.config.get('video',
                                                   {}).get('preset', 'medium'),
                        "-crf", str(self.config.get('video',
                                    {}).get('crf', 23))
                    ])

                # Audio settings
                audio_bitrate = self.config.get(
                    'audio', {}).get('bitrate', '128k')
                cmd.extend(["-c:a", "aac", "-b:a", audio_bitrate])

                # Add faststart option for web streaming
                cmd.extend(["-movflags", "+faststart"])

                # Add progress output
                cmd.extend(["-progress", "pipe:1"])

                # Add output file
                cmd.append(str(output_file))

                # Log the command
                accel_type_str = {
                    "full": "full hardware acceleration",
                    "encode_only": "hardware encoding only",
                    "cpu": "CPU encoding"
                }[acceleration_type]

                self.logger.info(
                    f"Optimizing video using {accel_type_str}: {input_file} → {output_file}")
                self.logger.debug(f"FFmpeg command: {' '.join(cmd)}")

                # Execute the command
                with open(log_file_path, 'w') as log_file:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.PIPE,
                        stderr=log_file,
                        universal_newlines=True,
                        bufsize=1,
                        text=True
                    )

                    # Register and monitor the process
                    self._register_process(process)
                    try:
                        self._process_ffmpeg_output(
                            process, input_file.name, duration_seconds)
                    except KeyboardInterrupt:
                        self.logger.warning("Optimization interrupted by user")
                        process.terminate()
                        if output_file.exists():
                            output_file.unlink()
                        raise
                    finally:
                        self._unregister_process(process)

                    # Check exit code
                    if process.wait() != 0:
                        raise subprocess.CalledProcessError(
                            process.returncode, cmd)

                # Verify output file
                if not output_file.exists() or output_file.stat().st_size == 0:
                    raise RuntimeError(
                        f"Output file missing or empty: {output_file}")

                # Success! Break out of the loop
                self.logger.success(
                    f"Video optimization successful using {accel_type_str}: {output_file}")

                # Clean up log file on success and update cache
                if log_file_path.exists():
                    try:
                        log_file_path.unlink()
                    except Exception as e:
                        self.logger.warning(
                            f"Could not remove log file {log_file_path}: {e}")

                # Update the processed cache
                with self.cache_lock:
                    self.processed_cache[input_file.name] = settings_hash
                    self._save_processed_cache()

                return output_file

            except (subprocess.CalledProcessError, RuntimeError) as e:
                # Read error from log file
                error_msg = ""
                if log_file_path.exists():
                    try:
                        with open(log_file_path, 'r') as log_file:
                            error_lines = log_file.readlines()
                            error_msg = "".join(
                                error_lines[-10:]) if error_lines else "No error details"
                    except Exception:
                        error_msg = str(e)
                else:
                    error_msg = str(e)

                # If we're not on the last attempt, try next fallback
                if acceleration_type != "cpu":
                    self.logger.warning(
                        f"Optimization with {acceleration_type} failed. Error: {error_msg}")
                    self.logger.info(
                        f"Falling back to next acceleration method...")

                    # Clean up failed output
                    if output_file.exists():
                        output_file.unlink()
                else:
                    # All attempts failed
                    self.logger.error(
                        f"All optimization attempts failed: {error_msg}")
                    if output_file.exists():
                        output_file.unlink()
                    return None

            except Exception as e:
                self.logger.error(
                    f"Error during video optimization: {e}", exc_info=True)
                if output_file.exists():
                    output_file.unlink()
                return None

        # We should never reach here, but just in case
        return None

    def _process_file_batch(self, batch_files: List[Path]) -> Dict[Path, Optional[Path]]:
        """
        Process a batch of files, handling both conversion and optimization.

        Args:
            batch_files: List of video files to process

        Returns:
            Dict[Path, Optional[Path]]: Mapping of input paths to output paths
        """
        results = {}

        # Get existing output files
        existing_outputs = self.get_existing_output_files()

        # Converted MP4 files to process from this batch
        converted_mp4s = []

        # First, convert non-MP4 files
        for video_file in batch_files:
            if video_file.suffix.lower() != self.OUTPUT_FORMAT:
                # Check if already processed with same settings
                settings_hash = self._get_settings_hash(video_file)
                if (video_file.stem in existing_outputs and
                    video_file.stem in self.processed_cache and
                        self.processed_cache[video_file.stem] == settings_hash):
                    # Skip with notification
                    self.logger.info(
                        f"Skipping already processed file: {video_file}")
                    results[video_file] = existing_outputs[video_file.stem]
                else:
                    # Convert the file to MP4
                    converted = self.convert_to_mp4(video_file)
                    if converted:
                        converted_mp4s.append(converted)
                        results[video_file] = converted  # Temporary result
                    else:
                        results[video_file] = None
            else:
                # For MP4 files, check if already processed
                settings_hash = self._get_settings_hash(video_file)
                output_file = self.output_dir / f"optimized_{video_file.name}"
                if (video_file.name in self.processed_cache and
                    self.processed_cache[video_file.name] == settings_hash and
                        output_file.exists()):
                    # Skip with notification
                    self.logger.info(
                        f"Skipping already processed MP4: {video_file}")
                    results[video_file] = output_file
                else:
                    # Optimize the MP4 file
                    optimized = self.optimize_mp4(video_file)
                    results[video_file] = optimized

        # Process converted files
        for mp4_file in converted_mp4s:
            # Optimize the converted MP4
            optimized = self.optimize_mp4(mp4_file)

            # Find the original file that was converted to this
            for orig_file, converted in results.items():
                if converted == mp4_file:
                    results[orig_file] = optimized
                    break

            # Clean up temp file after optimization
            try:
                if mp4_file.exists():
                    mp4_file.unlink()
                    self.logger.debug(f"Removed temporary file: {mp4_file}")
            except OSError as e:
                self.logger.warning(
                    f"Could not remove temporary file {mp4_file}: {e}")

        return results

    def process_videos(self) -> Dict[Path, Optional[Path]]:
        """
        Process all video files found in the input directory.

        Returns:
            Dict[Path, Optional[Path]]: Dictionary mapping input paths to output paths (or None if processing failed)
        """
        # Start a new log section for this processing run
        self.logging_system.start_new_log_section("Video Processing Started")

        # Find all video files
        mp4_files, other_video_files = self.find_video_files()
        all_files = mp4_files + other_video_files

        # Return early if no files found
        if not all_files:
            self.logger.info("No video files found to process")
            return {}

        # Dictionary to store results
        results = {}

        # Reset shutdown flag
        self.shutdown_requested = False

        try:
            if self.batch_size > 1 and self.num_threads > 1:
                # Split files into batches
                batches = [all_files[i:i + self.batch_size]
                           for i in range(0, len(all_files), self.batch_size)]

                self.logger.info(
                    f"Processing {len(all_files)} files in {len(batches)} batches, using {self.num_threads} threads")

                # Process batches with thread pool
                futures = []
                with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                    # Submit batch processing tasks
                    for batch in batches:
                        futures.append(executor.submit(
                            self._process_file_batch, batch))

                    # Collect results as they complete
                    for future in concurrent.futures.as_completed(futures):
                        if self.shutdown_requested:
                            # Cancel all remaining futures that haven't started
                            for f in futures:
                                if not f.done() and not f.running():
                                    f.cancel()
                            self.logger.info(
                                "Shutdown requested, cancelling remaining tasks")
                            break

                        try:
                            batch_results = future.result()
                            results.update(batch_results)
                        except Exception as e:
                            self.logger.error(
                                f"Error processing batch: {e}", exc_info=True)
            else:
                # Process files sequentially
                self.logger.info(
                    f"Processing {len(all_files)} files sequentially")

                # Display overall progress
                import sys
                total_files = len(all_files)
                processed_files = 0

                # Process each file
                for file in all_files:
                    if self.shutdown_requested:
                        self.logger.info(
                            "Shutdown requested, stopping processing")
                        break

                    # Update overall progress
                    processed_files += 1
                    progress_percent = (processed_files / total_files) * 100
                    sys.stdout.write(
                        f"\rOverall progress: {processed_files}/{total_files} files ({progress_percent:.1f}%)")
                    sys.stdout.flush()

                    try:
                        batch_result = self._process_file_batch([file])
                        results.update(batch_result)
                    except KeyboardInterrupt:
                        self.logger.warning("Processing interrupted by user")
                        self.shutdown_requested = True
                        break
                    except Exception as e:
                        self.logger.error(f"Error processing file {file}: {e}")

                # End with newline
                sys.stdout.write("\n")
                sys.stdout.flush()

        except KeyboardInterrupt:
            self.logger.warning("Processing interrupted by user")
            self.shutdown_requested = True

        finally:
            # Always clean up temporary converted files
            self._cleanup_temp_files()

            # Always save the processing cache
            self._save_processed_cache()

        # Log summary
        successful = sum(1 for output in results.values()
                         if output is not None)
        failed = sum(1 for output in results.values() if output is None)
        skipped = len(all_files) - successful - failed

        if successful > 0:
            self.logger.success(
                f"Successfully processed {successful} video files")
        if failed > 0:
            self.logger.warning(f"Failed to process {failed} video files")
        if skipped > 0:
            self.logger.info(
                f"Skipped {skipped} video files due to interruption or errors")

        # Make sure all results have been properly moved to the output directory
        for input_file, output_file in results.items():
            if output_file and not output_file.exists():
                self.logger.error(
                    f"Output file {output_file} is missing after processing")

        return results

    def _cleanup_temp_files(self):
        """Clean up temporary files in the temp directory."""
        # Clean up temporary converted files
        for temp_file in self.temp_dir.glob(f'*{self.OUTPUT_FORMAT}'):
            if temp_file.is_file():
                try:
                    temp_file.unlink()
                    self.logger.debug(
                        f"Cleaned up temporary file: {temp_file}")
                except OSError as e:
                    self.logger.warning(
                        f"Could not remove temporary file {temp_file}: {e}")

        # Clean up log files
        for log_file in self.temp_dir.glob('*_ffmpeg.log'):
            if log_file.is_file():
                try:
                    log_file.unlink()
                    self.logger.debug(
                        f"Cleaned up log file: {log_file}")
                except OSError as e:
                    self.logger.warning(
                        f"Could not remove log file {log_file}: {e}")

    def __del__(self):
        """Clean up when the processor is deleted."""
        # Terminate any active processes
        self._terminate_active_processes()

        # Restore original signal handlers
        if hasattr(self, 'original_sigint_handler'):
            signal.signal(signal.SIGINT, self.original_sigint_handler)
        if hasattr(self, 'original_sigterm_handler'):
            signal.signal(signal.SIGTERM, self.original_sigterm_handler)

    def batch_process(self) -> Dict[Path, Optional[Path]]:
        """
        Process all video files in batch mode.

        Returns:
            Dict[Path, Optional[Path]]: Dictionary mapping input paths to output paths
        """
        try:
            self.logging_system.start_new_log_section("Batch Processing")
            self.logger.info(
                f"Starting batch processing with {self.num_threads} threads and batch size of {self.batch_size}")

            # Print information about signal handling
            print(
                "\n[INFO] Processing started. Press Ctrl+C to gracefully stop processing.")
            print(
                "[INFO] All completed files will be saved, and processing will be cleanly terminated.\n")

            # Process videos
            return self.process_videos()

        except KeyboardInterrupt:
            # Handle keyboard interrupt at the top level
            self.logger.warning("Batch processing interrupted by user")
            print("\n[INFO] Processing halted by user. Cleaning up...")

            # Clean up
            self._cleanup_temp_files()
            self._save_processed_cache()

            # Return any results we have so far
            return {}
        except Exception as e:
            self.logger.error(
                f"Error during batch processing: {e}", exc_info=True)
            print(f"\n[ERROR] An error occurred during processing: {e}")

            # Clean up
            self._cleanup_temp_files()
            self._save_processed_cache()

            return {}


# Example usage
if __name__ == "__main__":
    # Simple config for testing
    config = {
        'directories': {
            'input': './input',
            'output': './output',
            'temp': './temp'
        },
        'video': {
            'resolution': '720p',
            'bitrate': '1.5M',
            'crf': 23,
            'preset': 'medium',
            'processing': {
                'threads': 4,
                'batch_size': 4
            }
        },
        'audio': {
            'bitrate': '128k'
        },
        'codec': 'h264',
        'logging': {
            'level': 'INFO',
            'console_level': 'INFO',
            'file_level': 'DEBUG',
            'directory': './logs',
            'clear_logs': True
        }
    }

    # Initialize the logging system
    logging_system = LoggingSystem(config)
    logger = logging_system.get_logger('main')

    # Start with a section header for processing
    logging_system.start_new_log_section("Video Processing")

    logger.info("Initializing video processor")
    processor = VideoProcessor(config)

    # Get info about acceleration
    logger.info(f"Using acceleration: {processor.preferred_acceleration.name}")

    # Process all videos
    logger.info("Starting batch processing of videos")
    results = processor.batch_process()

    # Print results
    logger.info("\nProcessing results:")
    for input_file, output_file in results.items():
        status = "SUCCESS" if output_file else "FAILED"
        if output_file:
            logger.success(f"{input_file.name} → {output_file.name}")
        else:
            logger.error(f"{input_file.name} → N/A (processing failed)")
