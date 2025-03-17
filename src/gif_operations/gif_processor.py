# Standard library imports
import os
import time
import asyncio
import traceback
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any, NamedTuple, Set, Callable
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager
import threading
import subprocess
import logging
import json
from functools import wraps
import signal

# Project imports
from ..logging_system import ModernLogStyle, UnifiedLogger, performance_monitor, log_function_call, run_ffmpeg_command
from ..base.processor import BaseProcessor
from ..default_config import (
    GIF_COMPRESSION,
    GIF_PASS_OVERS,
    GIF_SIZE_TO_SKIP,
    VIDEO_COMPRESSION,
    VIDEO_SETTINGS
)
from ..utils.error_handler import VideoProcessingError


@dataclass
class OptimizationConfig:
    """Configuration for GIF optimization."""
    scale_factor: float
    colors: int
    lossy_value: int
    dither_mode: str = 'bayer'
    bayer_scale: int = 3


class ProcessingStatus(Enum):
    """Processing status enumeration."""
    SUCCESS = "success"
    DIMENSION_ERROR = "dimension_error"
    PALETTE_ERROR = "palette_error"
    CONVERSION_ERROR = "conversion_error"
    OPTIMIZATION_ERROR = "optimization_error"
    FILE_ERROR = "file_error"
    SIZE_THRESHOLD_EXCEEDED = "size_threshold_exceeded"
    ALREADY_PROCESSED = "already_processed"
    SKIPPED = "skipped"


@dataclass
class ProcessingResult:
    """Result of processing a file."""
    success: bool
    size: float  # Size in MB
    status: ProcessingStatus
    path: Optional[str] = None
    message: str = ""
    settings: Optional[OptimizationConfig] = None


class GIFProcessingStatus(Enum):
    """GIF processing status with visual indicators."""
    STARTING = ('⚡', ModernLogStyle.AZURE)
    PROCESSING = ('↻', ModernLogStyle.CYAN)
    OPTIMIZING = ('⚙', ModernLogStyle.SLATE)
    SUCCESS = ('✓', ModernLogStyle.EMERALD)
    WARNING = ('⚠', ModernLogStyle.AMBER)
    ERROR = ('✖', ModernLogStyle.ROSE)
    SKIPPED = ('→', ModernLogStyle.SLATE)


class GIFProcessor(BaseProcessor):
    """
    Handles the processing and optimization of GIF and video files.
    Implements a workflow to check, convert, and optimize files from input to output directory.
    """

    def __init__(self,
                 input_dir: Path,
                 output_dir: Path,
                 max_size_mb: float = 8.0,
                 logger=None,
                 gpu_enabled: bool = False,
                 gpu_settings: Dict = None):
        """
        Initialize the GIF processor.

        Args:
            input_dir: Directory containing source files
            output_dir: Directory where processed files will be saved
            max_size_mb: Maximum allowed file size in MB
            logger: Logger instance
            gpu_enabled: Whether to use GPU acceleration
            gpu_settings: GPU settings dictionary
        """
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.max_size_mb = max_size_mb
        self.logger = logger or logging.getLogger(__name__)
        self.gpu_enabled = gpu_enabled
        self.gpu_settings = gpu_settings or {}

        # Track files that need processing
        self.gifs_to_process: List[Path] = []
        self.videos_to_process: List[Path] = []
        self.failed_items: List[Path] = []

        # Processing stats
        self.stats = {
            'processed_count': 0,
            'skipped_count': 0,
            'failed_count': 0,
            'total_time': 0,
            'start_time': 0,
        }

        # Thread locks for resource access
        self.file_locks: Dict[str, threading.Lock] = {}
        self.lock = threading.Lock()

        # Flag for controlled shutdown
        self.shutdown_requested = False

        # Initialize components
        self._init_components()
        self._setup_signal_handlers()

    def _init_components(self):
        """Initialize processor components."""
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up any additional components needed

    def _setup_signal_handlers(self):
        """Set up signal handlers for graceful shutdown."""
        try:
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except (ImportError, AttributeError):
            self.logger.warning(
                "Signal handling not available on this platform")

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info("Shutdown signal received, finishing current task...")
        self.shutdown_requested = True

        # Call the immediate shutdown handler for proper cleanup
        self._immediate_shutdown_handler(signum, frame)

    def get_file_lock(self, file_path: str) -> threading.Lock:
        """
        Get a lock for a specific file path to prevent concurrent access.

        Args:
            file_path: Path to the file

        Returns:
            A threading Lock object
        """
        with self.lock:
            if file_path not in self.file_locks:
                self.file_locks[file_path] = threading.Lock()
            return self.file_locks[file_path]

    @contextmanager
    def file_lock_context(self, file_path: str):
        """
        Context manager for file locking.

        Args:
            file_path: Path to the file
        """
        lock = self.get_file_lock(file_path)
        try:
            lock.acquire()
            yield
        finally:
            lock.release()

    def cleanup_resources(self) -> None:
        """Clean up resources before shutdown."""
        self.logger.info("Cleaning up resources...")

        # Kill any running ffmpeg processes
        try:
            self._kill_ffmpeg_processes()
        except Exception as e:
            self.logger.error(f"Error killing ffmpeg processes: {e}")

        # Clean up any temporary files in the output directory
        try:
            temp_pattern = "*.temp.*"
            temp_files = list(Path(self.output_dir).glob(temp_pattern))
            if temp_files:
                self.logger.info(
                    f"Cleaning up {len(temp_files)} temporary files...")
                for temp_file in temp_files:
                    try:
                        temp_file.unlink(missing_ok=True)
                    except Exception as e:
                        self.logger.warning(
                            f"Failed to remove temp file {temp_file}: {e}")
        except Exception as e:
            self.logger.error(f"Error cleaning up temporary files: {e}")

        # Release any file locks
        try:
            with self.lock:
                self.file_locks.clear()
        except Exception as e:
            self.logger.error(f"Error clearing file locks: {e}")

        # Force garbage collection
        try:
            import gc
            gc.collect()
        except Exception as e:
            self.logger.error(f"Error during garbage collection: {e}")

        self.logger.info("Resource cleanup completed")

    def _kill_ffmpeg_processes(self):
        """Kill any running ffmpeg processes."""
        try:
            if os.name == 'nt':  # Windows
                subprocess.run(["taskkill", "/F", "/IM", "ffmpeg.exe"],
                               capture_output=True,
                               check=False)
            else:  # Unix/Linux
                subprocess.run(["pkill", "-f", "ffmpeg"],
                               capture_output=True,
                               check=False)
        except Exception as e:
            self.logger.error(f"Error killing ffmpeg processes: {str(e)}")

    def scan_directories(self) -> Tuple[int, int]:
        """
        Scan input and output directories to identify files that need processing.

        Returns:
            Tuple of (gif_count, video_count) to be processed
        """
        self.logger.info(f"Scanning input directory: {self.input_dir}")

        # Reset processing lists
        self.gifs_to_process = []
        self.videos_to_process = []

        # Get output files for quick lookup
        output_files = {f.name: f for f in self.output_dir.glob('*')}

        # Scan input directory
        for file_path in self.input_dir.glob('*'):
            if self.shutdown_requested:
                break

            if file_path.is_dir():
                continue

            # Check if it's a GIF file
            if file_path.suffix.lower() == '.gif':
                self._check_gif_file(file_path, output_files)

            # Check if it's a video file
            elif file_path.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv', '.webm'):
                self._check_video_file(file_path, output_files)

        self.logger.info(f"Scan complete. Found {len(self.gifs_to_process)} GIFs and "
                         f"{len(self.videos_to_process)} videos to process.")

        return len(self.gifs_to_process), len(self.videos_to_process)

    def _check_gif_file(self, file_path: Path, output_files: Dict[str, Path]):
        """
        Check if a GIF file needs processing.

        Args:
            file_path: Path to the GIF file
            output_files: Dictionary of files in the output directory
        """
        filename = file_path.name

        # Check if optimized version exists in output dir
        if filename in output_files:
            output_path = output_files[filename]
            output_size = self.get_file_size(output_path)

            # Skip if already optimized and under max size
            if output_size <= self.max_size_mb:
                self.logger.info(
                    f"Skipping {filename}: Already optimized ({output_size:.2f} MB)")
                self.stats['skipped_count'] += 1
                return

        # Add to processing list
        self.gifs_to_process.append(file_path)

    def _check_video_file(self, file_path: Path, output_files: Dict[str, Path]):
        """
        Check if a video file needs processing.

        Args:
            file_path: Path to the video file
            output_files: Dictionary of files in the output directory
        """
        # For non-MP4 videos, the target will be an MP4 file
        target_filename = file_path.with_suffix('.mp4').name

        # Check if MP4 version exists for non-MP4 videos
        if file_path.suffix.lower() != '.mp4':
            mp4_path = self.output_dir / target_filename

            if mp4_path.exists():
                self.logger.info(f"Skipping conversion of {file_path.name}: "
                                 f"MP4 version already exists")
            else:
                # Needs conversion to MP4
                self.videos_to_process.append(file_path)
                return
        else:
            target_filename = file_path.name

        # Check if GIF version exists and is under max size
        gif_filename = Path(target_filename).with_suffix('.gif').name
        if gif_filename in output_files:
            gif_path = output_files[gif_filename]
            gif_size = self.get_file_size(gif_path)

            # Skip if GIF exists and is under max size
            if gif_size <= self.max_size_mb:
                self.logger.info(f"Skipping {file_path.name}: "
                                 f"GIF version already exists ({gif_size:.2f} MB)")
                self.stats['skipped_count'] += 1
                return

        # Add to processing list
        self.videos_to_process.append(file_path)

    def get_file_size(self, file_path: Path) -> float:
        """
        Get the size of a file in megabytes.

        Args:
            file_path: Path to the file

        Returns:
            Size in MB
        """
        if not file_path.exists():
            return 0

        return file_path.stat().st_size / (1024 * 1024)

    @log_function_call
    @performance_monitor
    def process_all(self) -> List[Path]:
        """
        Process all files that need processing one by one.
        Follows a specific order: first GIFs, then videos to GIFs.

        Returns:
            List of failed items
        """
        self.stats['start_time'] = time.time()

        # Create a single logger instance to avoid duplicate logging
        if not isinstance(self.logger, UnifiedLogger):
            self.logger = UnifiedLogger('gif_processor')
            # Clear any existing handlers to prevent duplicate logging
            for handler in logging.getLogger('gif_processor').handlers[:]:
                if isinstance(handler, logging.StreamHandler) and handler.level == logging.INFO:
                    logging.getLogger('gif_processor').removeHandler(handler)

        try:
            # First scan directories to find files that need processing
            self.logger.start_phase("Scanning Directories")
            self.scan_directories()
            self.logger.end_phase()

            # Process GIF files one by one
            if self.gifs_to_process:
                self.logger.start_phase("Processing GIF Files")
                self._process_gif_files()
                self.logger.end_phase()

            # Process video files one by one
            if self.videos_to_process:
                self.logger.start_phase("Processing Video Files")
                self._process_video_files()
                self.logger.end_phase()

            # Calculate total processing time
            self.stats['total_time'] = time.time() - self.stats['start_time']

            # Log summary
            self.log_processing_summary()

            return self.failed_items

        except Exception as e:
            self.logger.error(f"Fatal error in processing: {str(e)}")
            traceback.print_exc()
            return self.failed_items
        finally:
            self.cleanup_resources()

    def _process_gif_files(self):
        """Process GIF files one by one with proper logging and error handling."""
        total_gifs = len(self.gifs_to_process)
        self.logger.info(f"Processing {total_gifs} GIF files")

        for index, gif_path in enumerate(self.gifs_to_process, 1):
            if self.shutdown_requested:
                break

            self.logger.start_timer(f"gif_{gif_path.name}")

            try:
                self.logger.info(
                    f"Processing GIF {index}/{total_gifs}: {gif_path.name}")
                output_path = self.output_dir / gif_path.name

                success = self.optimize_gif(gif_path, output_path)
                if success:
                    self.stats['processed_count'] += 1
                    self.logger.success(
                        f"Successfully optimized: {gif_path.name}")
                else:
                    self.failed_items.append(gif_path)
                    self.stats['failed_count'] += 1
                    self.logger.warning(f"Failed to optimize: {gif_path.name}")

            except Exception as e:
                self.failed_items.append(gif_path)
                self.stats['failed_count'] += 1
                self.logger.error(
                    f"Error processing {gif_path.name}: {str(e)}")
                traceback.print_exc()

            finally:
                self.logger.end_timer(
                    f"gif_{gif_path.name}", include_system_stats=True)
                # Force garbage collection after each file
                import gc
                gc.collect()

    def _process_video_files(self):
        """Process video files one by one with proper logging and error handling."""
        total_videos = len(self.videos_to_process)
        self.logger.info(f"Processing {total_videos} video files")

        for index, video_path in enumerate(self.videos_to_process, 1):
            if self.shutdown_requested:
                break

            self.logger.start_timer(f"video_{video_path.name}")

            try:
                self.logger.info(
                    f"Processing video {index}/{total_videos}: {video_path.name}")

                # Convert non-MP4 to MP4 first if needed
                mp4_path = video_path
                if video_path.suffix.lower() != '.mp4':
                    mp4_path = self.output_dir / \
                        video_path.with_suffix('.mp4').name

                    # Check if MP4 already exists in output dir
                    if mp4_path.exists():
                        self.logger.info(
                            f"MP4 version already exists: {mp4_path.name}")
                    else:
                        # Convert to MP4
                        success = self.convert_to_mp4(video_path, mp4_path)
                        if not success:
                            self.failed_items.append(video_path)
                            self.stats['failed_count'] += 1
                            self.logger.warning(
                                f"Failed to convert to MP4: {video_path.name}")
                            continue
                else:
                    # For MP4 files, ensure they exist in output directory
                    output_mp4 = self.output_dir / video_path.name
                    if not output_mp4.exists():
                        import shutil
                        self.logger.info(
                            f"Copying MP4 to output directory: {video_path.name}")
                        shutil.copy2(video_path, output_mp4)
                        mp4_path = output_mp4

                # Check if GIF already exists and is under max size
                gif_path = self.output_dir / \
                    Path(mp4_path.name).with_suffix('.gif')
                if gif_path.exists():
                    gif_size = self.get_file_size(gif_path)
                    if gif_size <= self.max_size_mb:
                        self.logger.info(
                            f"GIF already exists and is under max size: {gif_path.name} ({gif_size:.2f} MB)")
                        self.stats['skipped_count'] += 1
                        continue
                    else:
                        self.logger.info(
                            f"GIF exists but exceeds max size: {gif_path.name} ({gif_size:.2f} MB), recreating")

                # Now convert MP4 to GIF
                success = self.create_optimized_gif(mp4_path, gif_path)

                if success:
                    self.stats['processed_count'] += 1
                    output_size = self.get_file_size(gif_path)
                    self.logger.success(
                        f"Successfully created GIF: {gif_path.name} ({output_size:.2f} MB)")
                else:
                    self.failed_items.append(video_path)
                    self.stats['failed_count'] += 1
                    self.logger.warning(
                        f"Failed to create GIF: {gif_path.name}")

            except Exception as e:
                self.failed_items.append(video_path)
                self.stats['failed_count'] += 1
                self.logger.error(
                    f"Error processing {video_path.name}: {str(e)}")
                traceback.print_exc()

            finally:
                self.logger.end_timer(
                    f"video_{video_path.name}", include_system_stats=True)
                # Force garbage collection after each file
                import gc
                gc.collect()

    def log_processing_summary(self):
        """Log a summary of the processing results."""
        self.logger.info("=== Processing Summary ===")
        self.logger.info(
            f"Total files processed successfully: {self.stats['processed_count']}")
        self.logger.info(f"Total files skipped: {self.stats['skipped_count']}")
        self.logger.info(f"Total files failed: {self.stats['failed_count']}")
        self.logger.info(
            f"Total processing time: {self.stats['total_time']:.2f} seconds")

    @log_function_call
    @performance_monitor
    def optimize_gif(self, input_path: Path, output_path: Path) -> bool:
        """
        Optimize a GIF file to be under the maximum size while preserving quality.
        Uses a dynamic approach to find the best balance of settings.

        Args:
            input_path: Path to the input GIF
            output_path: Path to save the optimized GIF

        Returns:
            True if optimization was successful, False otherwise
        """
        try:
            # Get input file size
            input_size_mb = self.get_file_size(input_path)
            self.logger.info(
                f"Original size: {input_size_mb:.2f} MB, Target: {self.max_size_mb:.2f} MB")

            # If file is already under max size, just copy it
            if input_size_mb <= self.max_size_mb:
                import shutil
                shutil.copy(input_path, output_path)
                self.logger.info(
                    f"GIF already under max size, copied to output directory")
                return True

            # IMPORTANT: Skip gifsicle if file is too large (over 700% of max size)
            if input_size_mb > (self.max_size_mb * 7):
                self.logger.info(
                    f"File too large for gifsicle (>700% of max size), using aggressive FFmpeg optimization")
                return self._create_gif_aggressive(input_path, output_path)

            # Get original dimensions and analyze GIF properties
            dimensions = self._get_gif_dimensions(input_path)
            if not dimensions:
                self.logger.error("Failed to get GIF dimensions")
                return False

            width, height = dimensions
            max_dimension = max(width, height)

            # Get frame count and duration if possible
            frame_info = self._get_gif_frame_info(input_path)
            frame_count = frame_info.get('frame_count', 0)
            duration = frame_info.get('duration', 0)

            self.logger.info(
                f"GIF properties: {width}x{height}, {frame_count} frames, {duration:.2f}s duration")

            # Dynamic optimization strategy based on GIF properties
            optimization_strategies = self._generate_gif_optimization_strategies(
                input_size_mb, max_dimension, width, height, frame_count, duration)

            # Try each strategy until we find one that works
            for strategy_index, strategy in enumerate(optimization_strategies):
                if self.shutdown_requested:
                    return False

                self.logger.info(
                    f"Trying optimization strategy {strategy_index+1}/{len(optimization_strategies)}: {strategy}")

                # Create a temporary file for this attempt
                import uuid
                temp_id = uuid.uuid4().hex[:8]
                temp_output = output_path.with_suffix(f'.{temp_id}.gif')

                try:
                    success = self._compress_with_gifsicle(
                        input_path, temp_output, self.max_size_mb,
                        colors=strategy['colors'],
                        lossy=strategy['lossy'],
                        scale=strategy['scale'],
                        optimize_level=strategy.get('optimize_level', 3)
                    )

                    if success:
                        # Verify quality is acceptable
                        if self._verify_gif_quality(input_path, temp_output, strategy):
                            # Copy to final destination
                            import shutil
                            shutil.copy2(temp_output, output_path)
                            output_size = self.get_file_size(output_path)
                            self.logger.success(
                                f"Optimization successful: {output_size:.2f} MB with strategy: {strategy}")
                            return True
                        else:
                            self.logger.info(
                                "Quality check failed, trying next strategy")
                finally:
                    # Clean up temp file
                    if temp_output.exists():
                        try:
                            temp_output.unlink()
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to remove temp file: {e}")

            # If all strategies failed, try aggressive approach with FFmpeg
            self.logger.info(
                "Standard optimization failed, trying aggressive FFmpeg approach")
            return self._create_gif_aggressive(input_path, output_path)

        except Exception as e:
            self.logger.error(f"Error optimizing GIF: {str(e)}")
            traceback.print_exc()
            return False

    def _get_gif_frame_info(self, gif_path: Path) -> Dict[str, Any]:
        """
        Get information about GIF frames and duration.

        Args:
            gif_path: Path to the GIF file

        Returns:
            Dictionary with frame count and duration
        """
        try:
            # Use gifsicle to get info - need to capture output for parsing
            result = subprocess.run(
                ['gifsicle', '--info', str(gif_path)],
                capture_output=True,
                text=True
            )

            info = {'frame_count': 0, 'duration': 0.0}

            if result.returncode == 0:
                output = result.stdout

                # Parse frame count
                import re
                frame_match = re.search(r'(\d+) images', output)
                if frame_match:
                    info['frame_count'] = int(frame_match.group(1))

                # Try to estimate duration from delay info
                delay_matches = re.findall(r'delay (\d+\.?\d*)s', output)
                if delay_matches:
                    total_delay = sum(float(d) for d in delay_matches)
                    # If we have frame count, use average delay * frames
                    if info['frame_count'] > 0:
                        avg_delay = total_delay / len(delay_matches)
                        info['duration'] = avg_delay * info['frame_count']
                    else:
                        info['duration'] = total_delay

            # Fallback to ffprobe if gifsicle didn't work well
            if info['frame_count'] == 0:
                cmd = [
                    "ffprobe",
                    "-v", "error",
                    "-count_frames",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=nb_read_frames,duration",
                    "-of", "json",
                    str(gif_path)
                ]

                # For ffprobe, we need to capture the output to parse the JSON data
                result = subprocess.run(cmd, capture_output=True, text=True)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    if 'streams' in data and data['streams']:
                        stream = data['streams'][0]
                        if 'nb_read_frames' in stream:
                            info['frame_count'] = int(stream['nb_read_frames'])
                        if 'duration' in stream:
                            info['duration'] = float(stream['duration'])

            return info

        except Exception as e:
            self.logger.error(f"Error getting GIF frame info: {str(e)}")
            return {'frame_count': 0, 'duration': 0.0}

    def _generate_gif_optimization_strategies(self, input_size_mb: float, max_dimension: int,
                                              width: int, height: int, frame_count: int = 0,
                                              duration: float = 0) -> List[Dict]:
        """
        Generate a list of optimization strategies based on GIF characteristics.
        Strategies are ordered from least to most aggressive.

        Args:
            input_size_mb: Original file size in MB
            max_dimension: Maximum dimension (width or height)
            width: Original width
            height: Original height
            frame_count: Number of frames in the GIF
            duration: Duration of the GIF in seconds

        Returns:
            List of strategy dictionaries
        """
        strategies = []

        # Base scale factor on dimensions
        if max_dimension > 1920:
            scale_factors = [0.5, 0.4, 0.3, 0.25]
        elif max_dimension > 1280:
            scale_factors = [0.7, 0.6, 0.5, 0.4, 0.3]
        elif max_dimension > 720:
            scale_factors = [0.8, 0.7, 0.6, 0.5, 0.4]
        else:
            scale_factors = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]

        # Base colors and lossy values on input size relative to target
        size_ratio = input_size_mb / self.max_size_mb

        # Adjust based on frame count and duration if available
        if frame_count > 100:
            # Many frames - need more aggressive compression
            color_options = [128, 96, 64, 32]
            lossy_options = [30, 50, 70, 90]
        elif frame_count > 50:
            color_options = [192, 128, 96, 64]
            lossy_options = [20, 40, 60, 80]
        else:
            # Default options based on size ratio
            if size_ratio <= 2:
                color_options = [256, 192, 128, 96]
                lossy_options = [20, 30, 40, 50]
            elif size_ratio <= 4:
                color_options = [192, 128, 96, 64]
                lossy_options = [30, 45, 60, 75]
            else:
                color_options = [128, 96, 64, 32]
                lossy_options = [40, 60, 80, 100]

        # Optimization levels
        optimize_levels = [3]  # Start with highest

        # Generate combinations - start with higher quality
        for scale in scale_factors:
            for colors in color_options:
                for lossy in lossy_options:
                    for optimize_level in optimize_levels:
                        strategies.append({
                            'scale': scale,
                            'colors': colors,
                            'lossy': lossy,
                            'optimize_level': optimize_level,
                            'expected_width': int(width * scale),
                            'expected_height': int(height * scale)
                        })

        # Sort strategies by estimated quality impact (least aggressive first)
        def strategy_quality_impact(s):
            # Lower values = higher quality = less aggressive
            return (1/s['colors']) * (1/s['scale']) * (s['lossy']/100)

        return sorted(strategies, key=strategy_quality_impact)

    @log_function_call
    @performance_monitor
    def _compress_with_gifsicle(self, input_path: Path, output_path: Path,
                                target_size_mb: float, colors: int = 256,
                                lossy: int = 30, scale: float = 1.0) -> bool:
        """
        Compress a GIF using gifsicle.

        Args:
            input_path: Path to the input GIF
            output_path: Path to save the compressed GIF
            target_size_mb: Target file size in MB
            colors: Number of colors (256, 128, 64, etc.)
            lossy: Lossy compression value (0-200)
            scale: Scale factor (0.1-1.0)

        Returns:
            True if compression was successful and under target size, False otherwise
        """
        try:
            # Create temporary scaled file if scaling is needed
            if scale < 1.0:
                # Get dimensions
                dimensions = self._get_gif_dimensions(input_path)
                if not dimensions:
                    self.logger.error("Failed to get GIF dimensions")
                    return False

                width, height = dimensions
                new_width = int(width * scale)
                new_height = int(height * scale)

                # Scale with gifsicle
                temp_path = output_path.with_suffix('.temp.gif')
                scale_cmd = [
                    "gifsicle",
                    "--resize", f"{new_width}x{new_height}",
                    str(input_path),
                    "-o", str(temp_path)
                ]

                result = subprocess.run(scale_cmd, capture_output=True)
                if result.returncode != 0:
                    self.logger.error(
                        f"Gifsicle scaling failed: {result.stderr.decode()}")
                    if temp_path.exists():
                        temp_path.unlink()
                    return False

                # Use the scaled file for further compression
                compress_input = temp_path
            else:
                compress_input = input_path

            # Compress with gifsicle
            compress_cmd = [
                "gifsicle",
                "--optimize=3",
                f"--lossy={lossy}",
                f"--colors={colors}",
                str(compress_input),
                "-o", str(output_path)
            ]

            result = subprocess.run(compress_cmd, capture_output=True)

            # Clean up temp file if it exists
            if scale < 1.0 and compress_input.exists():
                compress_input.unlink()

            if result.returncode != 0:
                self.logger.error(
                    f"Gifsicle compression failed: {result.stderr.decode()}")
                return False

            # Check if the output exists and is under the target size
            if output_path.exists():
                output_size = self.get_file_size(output_path)
                if output_size <= target_size_mb:
                    return True
                else:
                    self.logger.info(
                        f"Output file still too large: {output_size:.2f} MB")
                    return False
            else:
                self.logger.error("Output file was not created")
                return False

        except Exception as e:
            self.logger.error(f"Error in _compress_with_gifsicle: {str(e)}")
            traceback.print_exc()
            return False

    @log_function_call
    @performance_monitor
    def _create_gif_aggressive(self, input_path: Path, output_path: Path) -> bool:
        """
        Create an optimized GIF using FFmpeg with aggressive settings.

        Args:
            input_path: Path to the input file (GIF or video)
            output_path: Path to save the optimized GIF

        Returns:
            True if creation was successful, False otherwise
        """
        try:
            # Get dimensions for scaling
            dimensions = self._get_gif_dimensions(input_path)
            if not dimensions:
                self.logger.error("Failed to get dimensions")
                return False

            width, height = dimensions

            # Try different scaling factors until we get under target size
            for scale in [0.8, 0.6, 0.5, 0.4, 0.3, 0.2]:
                if self.shutdown_requested:
                    return False

                new_width = int(width * scale)
                new_height = int(height * scale)

                # Ensure even dimensions
                new_width = new_width - (new_width % 2)
                new_height = new_height - (new_height % 2)

                self.logger.info(
                    f"Trying aggressive optimization with scale {scale} ({new_width}x{new_height})")

                # Create palette
                palette_path = str(output_path.with_suffix('.png'))
                palette_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(input_path),
                    "-vf", f"scale={new_width}:{new_height}:flags=lanczos,palettegen=max_colors=64",
                    palette_path
                ]

                # Use run_ffmpeg_command to properly log output to ffmpeg.log
                palette_success = run_ffmpeg_command(palette_cmd)
                if not palette_success:
                    self.logger.error("Palette generation failed")
                    continue

                # Create GIF with palette
                gif_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(input_path),
                    "-i", palette_path,
                    "-filter_complex",
                    f"scale={new_width}:{new_height}:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=1:diff_mode=rectangle",
                    "-r", "15",
                    str(output_path)
                ]

                # Use run_ffmpeg_command to properly log output to ffmpeg.log
                gif_success = run_ffmpeg_command(gif_cmd)

                # Clean up palette file
                if os.path.exists(palette_path):
                    os.unlink(palette_path)

                if not gif_success:
                    self.logger.error("GIF creation failed")
                    continue

                # Check if file is under target size
                if output_path.exists():
                    output_size = self.get_file_size(output_path)
                    if output_size <= self.max_size_mb:
                        self.logger.info(
                            f"Successfully created GIF: {output_size:.2f} MB")
                        return True
                    else:
                        self.logger.info(
                            f"Output still too large: {output_size:.2f} MB, trying smaller scale")

            self.logger.error(
                "Failed to create GIF under target size with all scaling options")
            return False

        except Exception as e:
            self.logger.error(f"Error in _create_gif_aggressive: {str(e)}")
            traceback.print_exc()
            return False

    @log_function_call
    @performance_monitor
    def create_optimized_gif(self, input_path: Path, output_path: Path) -> bool:
        """
        Create an optimized GIF from a video file.
        Processes the file with increasingly aggressive settings until target size is reached.

        Args:
            input_path: Path to the input video
            output_path: Path to save the optimized GIF

        Returns:
            True if creation was successful, False otherwise
        """
        try:
            # Get video dimensions
            video_info = self.get_video_info(input_path)
            if not video_info:
                self.logger.error(f"Failed to get video info for {input_path}")
                return False

            width = int(video_info.get('width', 640))
            height = int(video_info.get('height', 480))

            # Get original duration if available
            duration = float(video_info.get('duration', 0))
            original_fps = int(video_info.get('fps', 30))

            # Calculate target size based on duration if available
            target_size_mb = self.max_size_mb
            if duration > 0:
                # Adjust target size for very long videos (more aggressive compression)
                if duration > 60:  # More than 1 minute
                    target_size_mb = min(
                        target_size_mb, self.max_size_mb * 0.8)
                elif duration > 30:  # More than 30 seconds
                    target_size_mb = min(
                        target_size_mb, self.max_size_mb * 0.9)

            # Generate optimization strategies based on video properties
            strategies = self._generate_video_to_gif_strategies(
                width, height, original_fps, duration)

            # Try each strategy until we get under max size
            for strategy in strategies:
                if self.shutdown_requested:
                    return False

                fps = strategy['fps']
                scale = strategy['scale']
                colors = strategy['colors']
                dither = strategy['dither']

                new_width = int(width * scale)
                new_height = int(height * scale)

                # Ensure even dimensions
                new_width = new_width - (new_width % 2)
                new_height = new_height - (new_height % 2)

                self.logger.info(
                    f"Trying GIF creation with fps={fps}, scale={scale} ({new_width}x{new_height}), colors={colors}")

                # Create temporary files with unique names to avoid conflicts
                import uuid
                temp_id = uuid.uuid4().hex[:8]
                palette_path = str(output_path.with_suffix(f'.{temp_id}.png'))
                temp_gif_path = str(output_path.with_suffix(f'.{temp_id}.gif'))

                try:
                    # Create palette
                    palette_cmd = [
                        "ffmpeg", "-y",
                        "-i", str(input_path),
                        "-vf", f"fps={fps},scale={new_width}:{new_height}:flags=lanczos,palettegen=max_colors={colors}",
                        palette_path
                    ]

                    # Use run_ffmpeg_command to properly log output to ffmpeg.log
                    palette_success = run_ffmpeg_command(palette_cmd)
                    if not palette_success:
                        self.logger.error("Palette generation failed")
                        continue

                    # Create GIF with palette
                    gif_cmd = [
                        "ffmpeg", "-y",
                        "-i", str(input_path),
                        "-i", palette_path,
                        "-filter_complex",
                        f"fps={fps},scale={new_width}:{new_height}:flags=lanczos[x];[x][1:v]paletteuse=dither={dither}:bayer_scale={strategy['bayer_scale']}:diff_mode=rectangle",
                        temp_gif_path
                    ]

                    # Use run_ffmpeg_command to properly log output to ffmpeg.log
                    gif_success = run_ffmpeg_command(gif_cmd)

                    # Check if GIF creation was successful
                    if not gif_success:
                        self.logger.error("GIF creation failed")
                        continue

                    # Check if file exists and copy to output path
                    if os.path.exists(temp_gif_path):
                        # Check file size
                        temp_size_mb = os.path.getsize(
                            temp_gif_path) / (1024 * 1024)

                        if temp_size_mb <= target_size_mb:
                            # Success! Copy to final location
                            import shutil
                            shutil.copy2(temp_gif_path, output_path)
                            self.logger.success(
                                f"Successfully created GIF: {temp_size_mb:.2f} MB with settings: {strategy}")
                            return True
                        else:
                            # File still too large
                            self.logger.info(
                                f"Output still too large: {temp_size_mb:.2f} MB, trying more aggressive settings")

                            # If file is too large (>700% of target), don't try gifsicle optimization
                            if temp_size_mb > (target_size_mb * 7):
                                self.logger.info(
                                    f"File too large for gifsicle optimization (>700% of target)")
                                continue

                            # Try additional optimization with gifsicle
                            self.logger.info(
                                f"Attempting gifsicle optimization on the generated GIF")

                            # Try different gifsicle settings
                            for lossy in [30, 50, 70, 90]:
                                for colors_gifsicle in [256, 128, 64]:
                                    gifsicle_cmd = [
                                        "gifsicle",
                                        "--optimize=3",
                                        f"--lossy={lossy}",
                                        f"--colors={colors_gifsicle}",
                                        temp_gif_path,
                                        "-o", str(output_path)
                                    ]

                                    result = subprocess.run(
                                        gifsicle_cmd, capture_output=True)

                                    if result.returncode == 0 and output_path.exists():
                                        output_size_mb = self.get_file_size(
                                            output_path)
                                        if output_size_mb <= target_size_mb:
                                            self.logger.success(
                                                f"Successfully optimized GIF with gifsicle: {output_size_mb:.2f} MB")
                                            return True

                finally:
                    # Clean up temporary files
                    for temp_file in [palette_path, temp_gif_path]:
                        try:
                            if os.path.exists(temp_file):
                                os.unlink(temp_file)
                        except Exception as e:
                            self.logger.warning(
                                f"Failed to remove temp file {temp_file}: {e}")

            # If we get here, all strategies failed
            self.logger.error(
                "Failed to create GIF under target size with all options")
            return False

        except Exception as e:
            self.logger.error(f"Error in create_optimized_gif: {str(e)}")
            traceback.print_exc()
            return False

    def _generate_video_to_gif_strategies(self, width: int, height: int,
                                          original_fps: int, duration: float) -> List[Dict]:
        """
        Generate optimization strategies for video to GIF conversion.
        Strategies are ordered from least to most aggressive.

        Args:
            width: Original video width
            height: Original video height
            original_fps: Original video frame rate
            duration: Video duration in seconds

        Returns:
            List of strategy dictionaries
        """
        strategies = []

        # Base FPS on original video
        if original_fps > 30:
            fps_options = [20, 15, 12, 10, 8]
        elif original_fps > 24:
            fps_options = [15, 12, 10, 8]
        else:
            fps_options = [12, 10, 8, 6]

        # Adjust for duration
        if duration > 60:  # Long videos
            fps_options = [min(10, fps) for fps in fps_options]

        # Base scale on dimensions
        max_dimension = max(width, height)
        if max_dimension > 1920:
            scale_options = [0.5, 0.4, 0.3, 0.25]
        elif max_dimension > 1280:
            scale_options = [0.6, 0.5, 0.4, 0.3]
        elif max_dimension > 720:
            scale_options = [0.7, 0.6, 0.5, 0.4]
        else:
            scale_options = [0.8, 0.7, 0.6, 0.5]

        # Color and dither options
        color_options = [256, 192, 128, 96, 64]
        dither_options = ['bayer', 'floyd_steinberg', 'sierra2']
        bayer_scale_options = [3, 2, 1]

        # Generate combinations - start with higher quality
        for fps in fps_options:
            for scale in scale_options:
                for colors in color_options:
                    for dither in dither_options:
                        for bayer_scale in bayer_scale_options:
                            # Skip some combinations to reduce total attempts
                            if dither != 'bayer' and bayer_scale != 3:
                                continue

                            strategies.append({
                                'fps': fps,
                                'scale': scale,
                                'colors': colors,
                                'dither': dither,
                                'bayer_scale': bayer_scale
                            })

        # Sort strategies by estimated file size impact (least aggressive first)
        def strategy_impact(s):
            # Higher values = smaller file = more aggressive
            return (1/s['fps']) * (1/s['scale']) * (1/s['colors'])

        return sorted(strategies, key=strategy_impact)

    @log_function_call
    @performance_monitor
    def convert_to_mp4(self, input_path: Path, output_path: Path) -> bool:
        """
        Convert a video file to MP4 format.

        Args:
            input_path: Path to the input video
            output_path: Path to save the MP4 video

        Returns:
            True if conversion was successful, False otherwise
        """
        try:
            self.logger.info(f"Converting {input_path.name} to MP4")

            # Check if source has audio
            has_audio = self._check_has_audio(input_path)

            # Build FFmpeg command
            cmd = ["ffmpeg", "-y", "-i", str(input_path)]

            # Add audio options
            if has_audio:
                cmd.extend(["-c:a", "aac", "-b:a", "128k"])
            else:
                cmd.extend(["-an"])  # No audio

            # Add video options
            cmd.extend([
                "-c:v", "libx264",
                "-preset", "medium",
                "-crf", "23",
                "-pix_fmt", "yuv420p",
                str(output_path)
            ])

            # Use run_ffmpeg_command to properly log output to ffmpeg.log
            success = run_ffmpeg_command(cmd)
            if not success:
                self.logger.error("MP4 conversion failed")
                return False

            # Verify output file exists and has content
            if output_path.exists() and output_path.stat().st_size > 0:
                self.logger.info(
                    f"Successfully converted to MP4: {output_path.name}")
                return True
            else:
                self.logger.error("Output MP4 is missing or empty")
                return False

        except Exception as e:
            self.logger.error(f"Error in convert_to_mp4: {str(e)}")
            traceback.print_exc()
            return False

    def _check_has_audio(self, file_path: Path) -> bool:
        """
        Check if a video file has an audio stream.

        Args:
            file_path: Path to the video file

        Returns:
            True if the video has audio, False otherwise
        """
        try:
            cmd = [
                "ffprobe",
                "-i", str(file_path),
                "-show_streams",
                "-select_streams", "a",
                "-loglevel", "error"
            ]

            # Use run_ffmpeg_command to properly log output to ffmpeg.log
            # For ffprobe, we need to capture the output to check for audio streams
            result = subprocess.run(cmd, capture_output=True, text=True)

            # If the output contains "stream", it has an audio stream
            return "stream" in result.stdout.lower()

        except Exception as e:
            self.logger.error(f"Error checking for audio: {str(e)}")
            return False  # Assume no audio on error

    def _get_gif_dimensions(self, file_path: Path) -> Optional[Tuple[int, int]]:
        """
        Get the dimensions of a GIF or video file.

        Args:
            file_path: Path to the file

        Returns:
            Tuple of (width, height) or None if dimensions couldn't be determined
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height",
                "-of", "csv=s=x:p=0",
                str(file_path)
            ]

            # For ffprobe, we need to capture the output to parse dimensions
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self.logger.error(f"Failed to get dimensions: {result.stderr}")
                return None

            dimensions = result.stdout.strip().split('x')
            if len(dimensions) != 2:
                self.logger.error(
                    f"Invalid dimensions format: {result.stdout}")
                return None

            return int(dimensions[0]), int(dimensions[1])

        except Exception as e:
            self.logger.error(f"Error in _get_gif_dimensions: {str(e)}")
            return None

    def get_video_info(self, file_path: Path) -> Dict[str, Any]:
        """
        Get information about a video file.

        Args:
            file_path: Path to the video file

        Returns:
            Dictionary with video information
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-select_streams", "v:0",
                "-show_entries", "stream=width,height,duration,r_frame_rate",
                "-of", "json",
                str(file_path)
            ]

            # For ffprobe, we need to capture the output to parse the JSON data
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                self.logger.error(f"Failed to get video info: {result.stderr}")
                return {}

            import json
            data = json.loads(result.stdout)

            info = {}
            if 'streams' in data and len(data['streams']) > 0:
                stream = data['streams'][0]
                info['width'] = stream.get('width', 0)
                info['height'] = stream.get('height', 0)
                info['duration'] = stream.get('duration', 0)

                # Parse frame rate (comes as "24/1" or similar)
                if 'r_frame_rate' in stream:
                    try:
                        num, den = stream['r_frame_rate'].split('/')
                        info['fps'] = round(float(num) / float(den))
                    except (ValueError, ZeroDivisionError):
                        info['fps'] = 30  # Default

            return info

        except Exception as e:
            self.logger.error(f"Error in get_video_info: {str(e)}")
            return {}

    def _verify_gif_quality(self, input_path: Path, output_path: Path, strategy: Dict) -> bool:
        """
        Verify the quality of the optimized GIF meets minimum standards.
        """
        try:
            # Check if dimensions match expected
            output_dims = self._get_gif_dimensions(output_path)
            if not output_dims:
                return False

            out_width, out_height = output_dims
            if (abs(out_width - strategy['expected_width']) > 2 or
                    abs(out_height - strategy['expected_height']) > 2):
                return False

            # Check file integrity
            if not self._verify_gif_integrity(output_path):
                return False

            # Check if file size is reasonable (not too small)
            min_expected_size = self.get_file_size(
                input_path) * 0.1  # 10% of original
            actual_size = self.get_file_size(output_path)
            if actual_size < min_expected_size:
                return False

            # Check frame count hasn't been drastically reduced
            input_frames = self._get_gif_frame_info(
                input_path).get('frame_count', 0)
            output_frames = self._get_gif_frame_info(
                output_path).get('frame_count', 0)

            if input_frames > 0 and output_frames > 0:
                # Allow up to 10% frame reduction
                if output_frames < (input_frames * 0.9):
                    self.logger.warning(
                        f"Frame count reduced too much: {input_frames} -> {output_frames}")
                    return False

            return True

        except Exception as e:
            self.logger.error(f"Error in quality verification: {str(e)}")
            return False

    def _verify_gif_integrity(self, gif_path: Path) -> bool:
        """
        Verify GIF file integrity using gifsicle.
        """
        try:
            result = subprocess.run(
                ['gifsicle', '--info', str(gif_path)],
                capture_output=True,
                text=True
            )
            return result.returncode == 0
        except Exception:
            return False

    @log_function_call
    @performance_monitor
    def process_file(self, input_path: Path, output_path: Path, is_video: bool = False) -> bool:
        """
        Process a single file (GIF or video) to create an optimized GIF.
        This method is designed to be called directly from main.py.

        Args:
            input_path: Path to the input file
            output_path: Path to save the optimized GIF
            is_video: Whether the input file is a video

        Returns:
            True if processing was successful, False otherwise
        """
        try:
            self.logger.info(f"Processing file: {input_path.name}")

            # Check if output already exists and is under max size
            if output_path.exists():
                output_size = self.get_file_size(output_path)
                if output_size <= self.max_size_mb:
                    self.logger.info(
                        f"File already processed: {output_path.name} ({output_size:.2f} MB)")
                    return True

            if is_video:
                # For videos, first ensure we have an MP4 version
                if input_path.suffix.lower() != '.mp4':
                    mp4_path = output_path.with_suffix('.mp4')
                    if not mp4_path.exists():
                        success = self.convert_to_mp4(input_path, mp4_path)
                        if not success:
                            self.logger.error(
                                f"Failed to convert to MP4: {input_path.name}")
                            return False
                    input_path = mp4_path

                # Then create optimized GIF
                return self.create_optimized_gif(input_path, output_path)
            else:
                # For GIFs, optimize directly
                return self.optimize_gif(input_path, output_path)

        except Exception as e:
            self.logger.error(
                f"Error processing file {input_path.name}: {str(e)}")
            traceback.print_exc()
            return False

    def _immediate_shutdown_handler(self, signum, frame=None):
        """
        Handle immediate shutdown request.
        This is called when the user presses Ctrl+C or the process receives a SIGTERM.

        Args:
            signum: Signal number
            frame: Current stack frame (optional)
        """
        # Set shutdown flag first to stop any ongoing processing
        self.shutdown_requested = True

        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM" if signum == signal.SIGTERM else f"Signal {signum}"
        self.logger.warning(
            f"Immediate shutdown requested ({signal_name}), cleaning up resources...")

        try:
            # Use the centralized cleanup method
            self.cleanup_resources()

            self.logger.info("Shutdown complete")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
            traceback.print_exc()


def main():
    """
    This function is not meant to be called directly.
    The processor should be instantiated and used by main.py instead.
    """
    print("This script is not meant to be run directly.")
    print("Please use main.py to run the GIF processor.")
    import sys
    sys.exit(1)


if __name__ == "__main__":
    main()
