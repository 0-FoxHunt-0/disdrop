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
        Get the appropriate FFmpeg acceleration arguments based on detected hardware.

        Returns:
            List[str]: FFmpeg command line arguments for hardware acceleration
        """
        accel_type = self.preferred_acceleration

        if accel_type == AccelerationType.CUDA:
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
        Get the appropriate codec arguments based on hardware and settings.

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
                # NVIDIA NVENC settings
                return [
                    "-c:v", "h264_nvenc",
                    "-preset", "p4",  # Options: p1 (fastest) to p7 (slowest)
                    "-profile:v", "high",
                    "-rc:v", "vbr_hq",  # High-quality VBR mode
                    "-cq:v", str(crf)  # Quality level
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
                return [
                    "-c:v", "hevc_nvenc",
                    "-preset", "p4",
                    "-profile:v", "main",
                    "-rc:v", "vbr_hq",
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
        Convert a non-MP4 file to MP4 format.

        Args:
            input_file: Path to the input video file

        Returns:
            Optional[Path]: Path to the converted MP4 file, or None if conversion failed
        """
        # Create output path in temp directory
        temp_output = self.temp_dir / f"{input_file.stem}{self.OUTPUT_FORMAT}"

        try:
            # Check if ffmpeg is available
            if shutil.which("ffmpeg") is None:
                self.logger.error(
                    "FFmpeg not found. Please install FFmpeg to process videos.")
                return None

            # Build FFmpeg command - IMPORTANT: hwaccel args must come BEFORE input file
            cmd = ["ffmpeg", "-y"]

            # Add hardware acceleration arguments before input file
            cmd.extend(self.get_ffmpeg_acceleration_args())

            # Add input file
            cmd.extend(["-i", str(input_file)])

            # For conversion, use fast preset but maintain quality
            if self.preferred_acceleration == AccelerationType.CUDA:
                cmd.extend(
                    ["-c:v", "h264_nvenc", "-preset", "p2", "-cq:v", "23"])
            elif self.preferred_acceleration == AccelerationType.CPU:
                cmd.extend(
                    ["-c:v", "libx264", "-preset", "fast", "-crf", "23"])
            else:
                # For other hardware acceleration, use appropriate codec
                codec_args = self.get_codec_args()
                # Override preset to fast if present in codec args
                for i, arg in enumerate(codec_args):
                    if arg == "-preset" and i + 1 < len(codec_args):
                        codec_args[i + 1] = "fast"
                cmd.extend(codec_args)

            # Copy audio stream without re-encoding
            cmd.extend(["-c:a", "aac", "-b:a", "128k"])

            # Add output file
            cmd.append(str(temp_output))

            # Execute the command
            self.logger.info(
                f"Converting video to MP4: {input_file} to {temp_output}")
            self.logger.debug(f"FFmpeg command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            self.logger.success(f"Video conversion completed: {temp_output}")
            return temp_output

        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg error during conversion: {e.stderr}")
            if temp_output.exists():
                temp_output.unlink()
            return None
        except Exception as e:
            self.logger.error(
                f"Error during video conversion: {e}", exc_info=True)
            if temp_output.exists():
                temp_output.unlink()
            return None

    def optimize_mp4(self, input_file: Path) -> Optional[Path]:
        """
        Optimize an MP4 file using the appropriate hardware acceleration.

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

        # Handle custom resolution format (e.g., "1280x720" or "1280:720")
        if resolution.lower() in resolution_map:
            scale = resolution_map.get(resolution.lower())
        elif 'x' in resolution or ':' in resolution:
            # User provided custom resolution in format like "1280x720" or "1280:720"
            scale = resolution.replace(
                'x', ':') if 'x' in resolution else resolution
            self.logger.info(f"Using custom resolution scale: {scale}")
        else:
            # Default to 720p if resolution not recognized
            scale = resolution_map.get('720p')
            self.logger.warning(
                f"Unrecognized resolution format '{resolution}', defaulting to 720p")

        try:
            # Check if ffmpeg is available
            if shutil.which("ffmpeg") is None:
                self.logger.error(
                    "FFmpeg not found. Please install FFmpeg to process videos.")
                return None

            # Build FFmpeg command - IMPORTANT: hwaccel args must come BEFORE input file
            cmd = ["ffmpeg", "-y"]

            # Add hardware acceleration arguments before input
            cmd.extend(self.get_ffmpeg_acceleration_args())

            # Add input file
            cmd.extend(["-i", str(input_file)])

            # Add filters for scaling
            cmd.extend(["-vf", f"scale={scale}"])

            # Add bitrate if specified (otherwise use quality-based encoding)
            if bitrate and not self.config.get('video', {}).get('use_crf_only', False):
                cmd.extend(["-b:v", bitrate])

            # Add codec arguments (with quality settings)
            cmd.extend(self.get_codec_args())

            # Add audio settings - transcode audio to ensure compatibility
            audio_bitrate = self.config.get('audio', {}).get('bitrate', '128k')
            cmd.extend(["-c:a", "aac", "-b:a", audio_bitrate])

            # Add faststart option for web streaming
            cmd.extend(["-movflags", "+faststart"])

            # Add output file
            cmd.append(str(output_file))

            # Execute the command
            self.logger.info(
                f"Optimizing video: {input_file} to {output_file}")
            self.logger.debug(f"FFmpeg command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            # Save to processed cache
            with self.cache_lock:
                self.processed_cache[input_file.name] = settings_hash
                self._save_processed_cache()

            self.logger.success(f"Video optimization completed: {output_file}")
            return output_file

        except subprocess.CalledProcessError as e:
            self.logger.error(f"FFmpeg error during optimization: {e.stderr}")
            if output_file.exists():
                output_file.unlink()
            return None
        except Exception as e:
            self.logger.error(
                f"Error during video optimization: {e}", exc_info=True)
            if output_file.exists():
                output_file.unlink()
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
            self.logger.info(f"Processing {len(all_files)} files sequentially")

            # Process each file
            for file in all_files:
                if self.shutdown_requested:
                    self.logger.info("Shutdown requested, stopping processing")
                    break

                batch_result = self._process_file_batch([file])
                results.update(batch_result)

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

        # Log summary
        successful = sum(1 for output in results.values()
                         if output is not None)
        failed = sum(1 for output in results.values() if output is None)
        if successful > 0:
            self.logger.success(
                f"Successfully processed {successful} video files")
        if failed > 0:
            self.logger.warning(f"Failed to process {failed} video files")

        return results

    def batch_process(self) -> Dict[Path, Optional[Path]]:
        """
        Process all video files in batch mode.

        Returns:
            Dict[Path, Optional[Path]]: Dictionary mapping input paths to output paths
        """
        self.logging_system.start_new_log_section("Batch Processing")
        self.logger.info(
            f"Starting batch processing with {self.num_threads} threads and batch size of {self.batch_size}")
        return self.process_videos()


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
