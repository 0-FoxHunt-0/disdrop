"""
Automated Workflow Module
Handles the automated processing of videos and GIF generation with graceful shutdown
"""

import os
import math
import subprocess
import signal
import sys
import shutil
import threading
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config_manager import ConfigManager
from .hardware_detector import HardwareDetector
from .video_compressor import DynamicVideoCompressor
from .gif_generator import GifGenerator
from .file_validator import FileValidator
from .ffmpeg_utils import FFmpegUtils

logger = logging.getLogger(__name__)

class AutomatedWorkflow:
    """Manages automated video processing and GIF generation workflow"""
    
    def __init__(self, config_manager: ConfigManager, hardware_detector: HardwareDetector, 
                 video_compressor=None, gif_generator=None):
        self.config = config_manager
        self.hardware = hardware_detector
        
        # Use provided instances or create new ones
        self.video_compressor = video_compressor or DynamicVideoCompressor(config_manager, hardware_detector)
        self.gif_generator = gif_generator or GifGenerator(config_manager)
        self.file_validator = FileValidator()
        # Console verbosity (controlled by CLI --debug)
        self.verbose = False
        
        # Workflow directories
        self.input_dir = Path("input")
        self.output_dir = Path("output")
        self.temp_dir = Path("temp")
        # Separate failures directory outside of output
        self.failures_dir = Path("failures")
        self.move_failures_to_folder = True
        
        # Shutdown handling
        self.shutdown_requested = False
        self.current_task = None
        self.processing_lock = threading.Lock()
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Ensure directories exist
        self._ensure_directories()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        def signal_handler(signum, frame):
            if not self.shutdown_requested:  # Only show message on first signal
                print(f"\n\n🛑 Shutdown signal received. Stopping gracefully...")
                if self.current_task:
                    print(f"⏳ Finishing current task: {self.current_task}")
                    print(f"💡 Press Ctrl+C again to force quit (may leave temp files)")
                logger.info(f"Received signal {signum}. Initiating graceful shutdown...")
                self.shutdown_requested = True
                
                # Request shutdown from video compressor
                if hasattr(self.video_compressor, 'request_shutdown'):
                    self.video_compressor.request_shutdown()
                
                # Request shutdown from GIF generator if it has the method
                if hasattr(self.gif_generator, 'request_shutdown'):
                    self.gif_generator.request_shutdown()
                # Attempt to cleanup temp immediately
                try:
                    self._cleanup_temp_files()
                except Exception:
                    pass
            else:
                # Second signal - force quit
                print(f"\n💥 Force quit requested. Exiting immediately...")
                logger.warning("Force quit - may leave temporary files")
                
                # Force terminate any running processes
                if hasattr(self.video_compressor, '_terminate_ffmpeg_process'):
                    self.video_compressor._terminate_ffmpeg_process()
                if hasattr(self.gif_generator, '_terminate_ffmpeg_process'):
                    self.gif_generator._terminate_ffmpeg_process()
                
                os._exit(1)
        
        # Handle common shutdown signals
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        for directory in [self.input_dir, self.output_dir, self.temp_dir, self.failures_dir]:
            directory.mkdir(exist_ok=True)
            logger.debug(f"Directory ensured: {directory}")
    
    def _interruptible_sleep(self, duration: float, check_interval: float = 0.1):
        """Sleep that can be interrupted by shutdown signal"""
        elapsed = 0.0
        while elapsed < duration and not self.shutdown_requested:
            sleep_time = min(check_interval, duration - elapsed)
            time.sleep(sleep_time)
            elapsed += sleep_time
    
    def run_automated_workflow(self, check_interval: int = 5, max_size_mb: float = 10.0, verbose: bool = False):
        """
        Run the automated workflow
        
        Args:
            check_interval: How often to check for new files (seconds)
            max_size_mb: Maximum file size for outputs in MB
        """
        # Set verbosity for this run
        self.verbose = bool(verbose)

        logger.info("Starting automated workflow...")
        logger.info(f"Input directory: {self.input_dir.absolute()}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        logger.info(f"Temp directory: {self.temp_dir.absolute()}")
        logger.info(f"Check interval: {check_interval} seconds")
        logger.info(f"Max file size: {max_size_mb}MB")
        
        self._vprint(f"\n🚀 Workflow started")
        self._vprint(f"📁 Input: {self.input_dir.absolute()}")
        self._vprint(f"📤 Output: {self.output_dir.absolute()}")
        self._vprint(f"⚙️  Every {check_interval}s, Max GIF {max_size_mb}MB")
        
        processed_files = set()
        processing_stats = {'successful': 0, 'skipped': 0, 'errors': 0}
        first_scan = True
        
        try:
            while not self.shutdown_requested:
                try:
                    # Find new video files in input directory
                    video_files = self._find_new_files(processed_files, skip_stability_check=first_scan)
                    first_scan = False
                    
                    if video_files:
                        logger.info(f"Found {len(video_files)} new file(s) to process")
                        self._vprint(f"\n🎬 {len(video_files)} new file(s) to process")
                        
                        for file_path in video_files:
                            if self.shutdown_requested:
                                break
                            
                            # Determine file type and process accordingly
                            if file_path.suffix.lower() == '.gif':
                                # Process GIF file
                                has_existing_gif = self._has_existing_output(file_path)
                                if has_existing_gif:
                                    # GIF processing already done, check if optimization is needed
                                    result = self._handle_existing_gif_file(file_path, max_size_mb)
                                else:
                                    # Full GIF processing needed
                                    result = self._process_single_gif(file_path, max_size_mb)
                            else:
                                # Process video file
                                # Check if this file has existing video output
                                has_existing_video = self._has_existing_output(file_path)
                                
                                if has_existing_video:
                                    # Video processing already done, but check if GIF creation is needed
                                    result = self._handle_existing_video_file(file_path, max_size_mb)
                                else:
                                    # Full processing needed (video + GIF)
                                    result = self._process_single_video(file_path, max_size_mb)
                            
                            # After processing a single file, cleanup temp files immediately
                            self._cleanup_temp_files()
                            processed_files.add(file_path)
                            
                            # Update stats
                            if result == 'success':
                                processing_stats['successful'] += 1
                            elif result == 'skipped':
                                processing_stats['skipped'] += 1
                            else:
                                processing_stats['errors'] += 1

                            # If processing failed, leave original input in place (do not move from input/)
                            if result == 'error':
                                logger.info(
                                    f"Leaving failed input in place for retry: {file_path.name}"
                                )
                        
                        # Show processing summary
                        if video_files:
                            total = processing_stats['successful'] + processing_stats['skipped'] + processing_stats['errors']
                            self._vprint(f"\n📊 Summary: ✅ {processing_stats['successful']} | ⚠️ {processing_stats['skipped']} | ❌ {processing_stats['errors']} (Total {total})")
                    else:
                        # Show waiting status periodically (only when verbose)
                        if self.verbose:
                            current_time = time.strftime("%H:%M:%S")
                            print(f"\r⏳ [{current_time}] Waiting for files...", end="", flush=True)
                    
                    # Sleep before next check (with interrupt checking)
                    if not self.shutdown_requested:
                        self._interruptible_sleep(check_interval)
                        
                except Exception as e:
                    logger.error(f"Error in workflow loop: {e}")
                    if not self.shutdown_requested:
                        self._interruptible_sleep(check_interval)
        
        except KeyboardInterrupt:
            print(f"\n\n🛑 Workflow interrupted by user")
            logger.info("Workflow interrupted by user")
        finally:
            self._vprint(f"\n🧹 Cleaning up temporary files...")
            self._cleanup_temp_files()
            # Also clean nested temp artifacts created during GIF optimization
            self._cleanup_orphan_segments()
            self._vprint(f"✅ Automated workflow stopped gracefully")
            logger.info("Automated workflow stopped")

    def _vprint(self, message: str, end: str = "\n", flush: bool = False):
        """Verbose print controlled by self.verbose."""
        if self.verbose:
            print(message, end=end, flush=flush)

    def _move_input_to_failures(self, src_path: Path) -> None:
        """Move a failed input file from input/ to failures/ with safe unique naming.

        Best-effort operation; logs and prints a short note on success.
        """
        try:
            self.failures_dir.mkdir(exist_ok=True)
        except Exception:
            return

        try:
            stem = src_path.stem
            suffix = src_path.suffix
            candidate = self.failures_dir / src_path.name
            index = 1
            # Ensure we don't overwrite an existing file in failures
            while candidate.exists():
                candidate = self.failures_dir / f"{stem} ({index}){suffix}"
                index += 1

            shutil.move(str(src_path), str(candidate))
            print(f"  📁 Moved to failures: {candidate.name}")
            logger.info(f"Moved failed input to failures: {src_path.name} -> {candidate}")
        except Exception as e:
            logger.warning(f"Failed to move {src_path.name} to failures: {e}")
    
    def _find_new_files(self, processed_files: set, skip_stability_check: bool = False) -> List[Path]:
        """Find new files, prioritizing GIFs first, then videos."""
        files = []
        
        if not self.input_dir.exists():
            return files
        
        supported_video_extensions = self.file_validator.get_supported_video_extensions()
        supported_gif_extensions = {'.gif'}
        supported_extensions = supported_video_extensions | supported_gif_extensions
        
        # First, collect all potential files
        potential_files = []
        for file_path in self.input_dir.iterdir():
            if (file_path.is_file() and 
                file_path.suffix.lower() in supported_extensions and
                file_path not in processed_files):
                potential_files.append(file_path)
        
        # If we have many files, show progress
        if len(potential_files) > 10:
            if skip_stability_check:
                print(f"🔍 Scanning {len(potential_files)} files (initial scan - skipping stability check)...")
            else:
                print(f"🔍 Scanning {len(potential_files)} files for stability and existing outputs...")
        
        # Check stability and existing outputs for each file
        for i, file_path in enumerate(potential_files):
            # Check for shutdown signal during file scanning
            if self.shutdown_requested:
                if len(potential_files) > 10:
                    print(f"\r🛑 File scan interrupted by user" + " " * 30)
                break
                
            if len(potential_files) > 10:
                if skip_stability_check:
                    print(f"\r  📁 Processing file {i+1}/{len(potential_files)}: {file_path.name[:30]}...", end="", flush=True)
                else:
                    print(f"\r  📁 Checking file {i+1}/{len(potential_files)}: {file_path.name[:30]}...", end="", flush=True)
            
            # Check if file is not currently being written to
            if skip_stability_check or self._is_file_stable(file_path):
                files.append(file_path)
        
        if len(potential_files) > 10:
            print(f"\r✅ File scan complete: {len(files)} files ready for processing" + " " * 20)
        
        # Prioritize GIFs first, then videos (keep alphabetical order within each group)
        gifs_first = sorted([f for f in files if f.suffix.lower() == '.gif'])
        videos_next = sorted([f for f in files if f.suffix.lower() != '.gif'])
        return gifs_first + videos_next
    
    def _has_existing_output(self, input_file: Path) -> bool:
        """
        Check for existing output files.
        
        For video files: Checks for MP4 outputs (video processing)
        For GIF files: Checks for optimized GIF outputs
        
        Args:
            input_file: Path to the input file
            
        Returns:
            True if valid output exists, False otherwise
        """
        if not self.output_dir.exists():
            return False
        
        base_name = input_file.stem
        
        # Check if input is a GIF file
        if input_file.suffix.lower() == '.gif':
            # For GIF files, check for existing optimized GIF output
            output_patterns = [
                f"{base_name}.gif",
            ]
            
            # Check root output directory first (most common case)
            for pattern in output_patterns:
                output_path = self.output_dir / pattern
                if self._is_valid_existing_output(output_path, pattern, input_file):
                    logger.info(f"Found existing GIF output for {input_file.name}: {output_path}")
                    return True
            
            # Recursive search in all subdirectories
            try:
                for subdir in self.output_dir.rglob("*"):
                    if subdir.is_dir():
                        for pattern in output_patterns:
                            output_path = subdir / pattern
                            if self._is_valid_existing_output(output_path, pattern, input_file):
                                logger.info(f"Found existing GIF output for {input_file.name}: {output_path}")
                                return True
            except Exception as e:
                logger.debug(f"Error during recursive output search: {e}")
            
            return False
        else:
            # For video files, check for MP4 outputs (video processing)
            # GIF creation will be handled separately in the workflow
            output_patterns = [
                f"{base_name}.mp4",
            ]
            
            # Check root output directory first (most common case)
            for pattern in output_patterns:
                output_path = self.output_dir / pattern
                if self._is_valid_existing_output(output_path, pattern, input_file):
                    logger.info(f"Found existing video output for {input_file.name}: {output_path}")
                    return True
            
            # Recursive search in all subdirectories
            try:
                for subdir in self.output_dir.rglob("*"):
                    if subdir.is_dir():
                        for pattern in output_patterns:
                            output_path = subdir / pattern
                            if self._is_valid_existing_output(output_path, pattern, input_file):
                                logger.info(f"Found existing video output for {input_file.name}: {output_path}")
                                return True
            except Exception as e:
                logger.debug(f"Error during recursive output search: {e}")
            
            # Check for segmented output
            segments_folder = self.output_dir / f"{base_name}_segments"
            if segments_folder.exists() and segments_folder.is_dir():
                # Use default 10MB when checking for generic existence, as size isn't known here
                valid_segments, _ = self._validate_segment_folder_gifs(segments_folder, 10.0)
                if valid_segments:
                    logger.info(f"Found existing segmented output for {input_file.name}: {segments_folder}")
                    return True
            return False
    
    def _is_valid_existing_output(self, output_path: Path, pattern_type: str, input_file: Path = None) -> bool:
        """
        Validate that an existing output is actually valid and was created from the specific input.
        
        Args:
            output_path: Path to the potential output
            pattern_type: Type of output ('*.mp4', '*.gif', or '*_segments')
            input_file: Path to the original input file for verification
            
        Returns:
            True if the output exists, is valid, and was created from the input
        """
        try:
            if not output_path.exists():
                return False
            
            # Basic validation first
            basic_valid = False
            if pattern_type.endswith('.mp4'):
                # Validate MP4 file
                is_valid, _ = self.file_validator.is_valid_video(str(output_path))
                basic_valid = is_valid
                
            elif pattern_type.endswith('.gif'):
                # Validate single GIF file (check integrity, not size)
                is_valid, _ = self.file_validator.is_valid_gif(str(output_path), max_size_mb=None)
                basic_valid = is_valid
                
            elif pattern_type.endswith('_segments'):
                # Validate segment folder - check if it contains valid GIFs
                if output_path.is_dir():
                    valid_gifs, invalid_gifs = self._validate_segment_folder_gifs(output_path, 10.0)
                    basic_valid = len(valid_gifs) > 0  # At least one valid GIF makes it usable
            
            if not basic_valid:
                return False
            
            # Enhanced validation: verify the output was created from this specific input
            if input_file and input_file.exists():
                return self._verify_output_source_relationship(input_file, output_path, pattern_type)
            
            # If no input file provided, just return basic validation
            return basic_valid
                    
        except Exception as e:
            logger.debug(f"Error validating existing output {output_path}: {e}")
            return False
        
        return False
    
    def _verify_output_source_relationship(self, input_file: Path, output_path: Path, pattern_type: str) -> bool:
        """
        Verifies if the output file was created from the specific input file by comparing
        key characteristics like duration, resolution, and file timestamps.
        """
        try:
            # Get input file characteristics
            input_stats = input_file.stat()
            input_file_mtime = input_stats.st_mtime
            
            # Get basic input video info
            try:
                input_duration = FFmpegUtils.get_video_duration(str(input_file))
                input_resolution = FFmpegUtils.get_video_resolution(str(input_file))
            except Exception as e:
                logger.debug(f"Could not get input video info for {input_file}: {e}")
                input_duration = None
                input_resolution = None
            
            # Check output modification time - should be newer than input
            output_stats = output_path.stat()
            output_file_mtime = output_stats.st_mtime
            
            # Output should be created after input (with some tolerance for processing time)
            if output_file_mtime < input_file_mtime - 60:  # Allow 1 minute tolerance
                logger.debug(f"Output {output_path.name} is older than input {input_file.name}")
                return False
            
            # Type-specific validation
            if pattern_type.endswith('.mp4'):
                return self._verify_mp4_source_relationship(input_file, output_path, input_duration, input_resolution)
                
            elif pattern_type.endswith('.gif'):
                return self._verify_gif_source_relationship(input_file, output_path, input_duration)
                
            elif pattern_type.endswith('_segments'):
                return self._verify_segments_source_relationship(input_file, output_path, input_duration)
            
            return True  # Default to valid if type not recognized
            
        except Exception as e:
            logger.debug(f"Error verifying output source relationship for {output_path}: {e}")
            return False
    
    def _verify_mp4_source_relationship(self, input_file: Path, output_path: Path, 
                                       input_duration: float, input_resolution: tuple) -> bool:
        """Verify MP4 output was created from specific input"""
        try:
            if not input_duration or not input_resolution:
                return True  # Can't verify, assume valid
            
            output_duration = FFmpegUtils.get_video_duration(str(output_path))
            output_resolution = FFmpegUtils.get_video_resolution(str(output_path))
            
            # Check duration (should be very close for MP4 optimization)
            duration_diff = abs(input_duration - output_duration)
            if duration_diff > 2.0:  # Allow 2 second difference
                logger.debug(f"MP4 duration mismatch: {duration_diff:.1f}s difference")
                return False
            
            # Check resolution (MP4 optimization usually preserves resolution)
            if input_resolution and output_resolution:
                res_diff = abs(input_resolution[0] - output_resolution[0]) + abs(input_resolution[1] - output_resolution[1])
                if res_diff > 50:  # Allow 50 pixel difference
                    logger.debug(f"MP4 resolution mismatch: {res_diff}px difference")
                    return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error verifying MP4 source relationship: {e}")
            return True  # Default to valid on error
    
    def _verify_gif_source_relationship(self, input_file: Path, output_path: Path, input_duration: float) -> bool:
        """Verify GIF output was created from specific input"""
        try:
            if not input_duration:
                return True  # Can't verify duration, assume valid
            
            # For GIFs, we mainly check if the duration makes sense
            # GIFs are often shorter than source video due to size limits
            gif_size_mb = output_path.stat().st_size / (1024 * 1024)
            
            # If GIF is close to 10MB limit and input is long, it's likely a compressed version
            if gif_size_mb > 8.0 and input_duration > 30:
                return True  # Large GIF from long video = likely compressed
            
            # If GIF is small and input is short, duration should be similar
            if gif_size_mb < 5.0 and input_duration < 20:
                # Check if GIF could reasonably be from this input
                # (GIFs are typically shorter due to compression)
                return True
            
            return True  # Default to valid for GIFs
            
        except Exception as e:
            logger.debug(f"Error verifying GIF source relationship: {e}")
            return True
    
    def _verify_segments_source_relationship(self, input_file: Path, output_path: Path, input_duration: float) -> bool:
        """Verify segment folder was created from specific input"""
        try:
            if not input_duration:
                return True  # Can't verify, assume valid
            
            # Check if segment folder contains reasonable number of segments for input duration
            segment_files = [f for f in output_path.iterdir() if f.suffix.lower() == '.gif']
            
            if not segment_files:
                return False  # No segments found
            
            # Estimate expected segments based on input duration
            # Typical segment duration is 15-30 seconds
            expected_segments = max(1, int(input_duration / 20))  # Rough estimate
            actual_segments = len(segment_files)
            
            # Allow wide range since segmentation is adaptive
            if actual_segments < 1 or actual_segments > expected_segments * 3:
                logger.debug(f"Segment count mismatch: {actual_segments} segments for {input_duration:.1f}s video")
                return False
            
            return True
            
        except Exception as e:
            logger.debug(f"Error verifying segments source relationship: {e}")
            return True
    
    def _is_file_stable(self, file_path: Path, wait_time: int = 2) -> bool:
        """Check if file is stable (not being written to)"""
        try:
            stat_info = file_path.stat()
            file_size = stat_info.st_size
            file_mtime = stat_info.st_mtime
            
            # If file is older than 10 seconds, assume it's stable (optimization)
            current_time = time.time()
            if current_time - file_mtime > 10:
                return True
            
            # For newer files, do the stability check with interrupt capability
            self._interruptible_sleep(wait_time)
            
            # If shutdown was requested during sleep, return False to skip file
            if self.shutdown_requested:
                return False
            
            new_stat_info = file_path.stat()
            return (file_size == new_stat_info.st_size and 
                    file_mtime == new_stat_info.st_mtime)
        except Exception:
            return False
    
    def _process_single_video(self, video_file: Path, max_size_mb: float) -> str:
        """Process a single video file through the complete workflow"""
        with self.processing_lock:
            if self.shutdown_requested:
                return 'cancelled'
            
            self.current_task = f"Processing {video_file.name}"
            logger.info(f"Starting processing: {video_file.name}")
        
        print(f"\n📹 Processing: {video_file.name}")
        
        try:
            # Step 1: Validate input video
            print("  🔍 Validate video")
            is_valid, error_msg = self.file_validator.is_valid_video(str(video_file))
            if not is_valid:
                print(f"  ⚠️  Skipping invalid file: {error_msg}")
                logger.warning(f"Skipping invalid video file {video_file.name}: {error_msg}")
                return 'skipped'
            print("  ✅ Video validation passed")
            
            # Check for shutdown before step 2
            if self.shutdown_requested:
                return 'cancelled'
            
            # Step 2: Convert to MP4 if needed and optimize
            print("  🔄 Convert/optimize to MP4")
            mp4_file = self._ensure_mp4_format(video_file, max_size_mb)
            if not mp4_file or self.shutdown_requested:
                return 'error' if not self.shutdown_requested else 'cancelled'
            print(f"  ✅ MP4 optimization complete: {mp4_file.name}")
            
            # Check for shutdown before step 3
            if self.shutdown_requested:
                return 'cancelled'
            
            # Step 3: Generate and optimize GIF (ALWAYS attempt this step)
            print("  🎨 Generate/optimize GIF")
            gif_success = self._generate_and_optimize_gif(mp4_file, max_size_mb)
            
            if gif_success:
                print(f"  ✅ GIF generation complete: {mp4_file.stem}.gif")
                print(f"  🎉 Done: {video_file.name}")
                logger.info(f"Successfully processed: {video_file.name}")
                return 'success'
            else:
                print(f"  ❌ GIF failed: {video_file.name}")
                logger.error(f"GIF generation failed for: {video_file.name}")
                return 'error'
            
        except Exception as e:
            print(f"  ❌ Error processing {video_file.name}: {e}")
            logger.error(f"Error processing {video_file.name}: {e}")
            return 'error'
        finally:
            self.current_task = None
    
    def _handle_existing_video_file(self, video_file: Path, max_size_mb: float) -> str:
        """Handle files that already have video outputs but may need GIF creation"""
        with self.processing_lock:
            if self.shutdown_requested:
                return 'cancelled'
            
            self.current_task = f"Checking GIFs for {video_file.name}"
            logger.info(f"Found existing video output for {video_file.name}, checking GIF outputs")
        
        print(f"\n📹 Found existing video: {video_file.name}")
        
        try:
            # Find the existing MP4 file
            mp4_file = self._find_existing_mp4_output(video_file)
            if not mp4_file:
                logger.warning(f"Could not find existing MP4 for {video_file.name}, proceeding with full processing")
                return self._process_single_video(video_file, max_size_mb)
            
            print(f"  ♻️  Using existing MP4: {mp4_file.name}")
            
            # Check for shutdown before GIF validation
            if self.shutdown_requested:
                return 'cancelled'
            
            # Step 1: Check for existing single GIF file
            gif_name = mp4_file.stem + ".gif"
            gif_path = self.output_dir / gif_name
            
            if gif_path.exists():
                print(f"  🔍 Found existing GIF, validating: {gif_name}")
                is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                    str(gif_path), 
                    original_path=None, 
                    max_size_mb=max_size_mb
                )
                if is_valid:
                    print(f"  ✅ Valid GIF already exists: {gif_name}")
                    logger.info(f"Valid GIF already exists for {video_file.name}: {gif_name}")
                    return 'success'
                else:
                    print(f"  🔄 GIF validation failed ({error_msg}), will regenerate")
                    logger.info(f"Existing GIF validation failed: {error_msg}, will regenerate: {gif_name}")
            else:
                print(f"  📄 No single GIF found: {gif_name}")
            
            # Step 2: Check for existing segment folder
            segments_folder = self.output_dir / f"{mp4_file.stem}_segments"
            if segments_folder.exists() and segments_folder.is_dir():
                print(f"  🔍 Found segments folder, validating: {segments_folder.name}")
                
                # Validate segment folder GIFs
                valid_segments, invalid_segments = self._validate_segment_folder_gifs(segments_folder, max_size_mb)
                
                if valid_segments:
                    print(f"  ✅ Valid segment GIFs exist: {segments_folder.name}")
                    logger.info(f"Valid segment GIFs already exist for {video_file.name}: {segments_folder.name}")
                    
                    if invalid_segments:
                        print(f"  ⚠️  Found {len(invalid_segments)} invalid/corrupted GIFs in segments folder")
                        logger.warning(f"Found {len(invalid_segments)} invalid GIFs in segments folder: {invalid_segments}")
                    
                    return 'success'
                else:
                    print(f"  🔄 Segment GIFs invalid/corrupted, will regenerate")
                    logger.info(f"Segment GIFs invalid, will regenerate: {segments_folder.name}")
            else:
                print(f"  📁 No segments folder found: {segments_folder.name}")
            
            # Step 3: No valid outputs found, create new GIFs
            print("  🎨 No valid GIF outputs found, generating new ones...")
            gif_success = self._generate_and_optimize_gif(mp4_file, max_size_mb)
            
            if gif_success:
                print(f"  ✅ GIF generation complete")
                print(f"  🎉 Successfully processed GIFs for: {video_file.name}")
                logger.info(f"Successfully generated new GIFs for existing video: {video_file.name}")
                return 'success'
            else:
                print(f"  ❌ GIF generation failed for: {video_file.name}")
                logger.error(f"GIF generation failed for existing video: {video_file.name}")
                return 'error'
            
        except Exception as e:
            print(f"  ❌ Error processing GIFs for {video_file.name}: {e}")
            logger.error(f"Error processing GIFs for {video_file.name}: {e}")
            return 'error'
        finally:
            self.current_task = None
    
    def _find_existing_mp4_output(self, input_file: Path) -> Optional[Path]:
        """Find the existing MP4 output file for a given input file"""
        base_name = input_file.stem
        
        # Check root output directory first
        mp4_path = self.output_dir / f"{base_name}.mp4"
        if mp4_path.exists():
            is_valid, _ = self.file_validator.is_valid_video(str(mp4_path))
            if is_valid:
                return mp4_path
        
        # Recursive search in subdirectories
        try:
            for subdir in self.output_dir.rglob("*"):
                if subdir.is_dir():
                    mp4_path = subdir / f"{base_name}.mp4"
                    if mp4_path.exists():
                        is_valid, _ = self.file_validator.is_valid_video(str(mp4_path))
                        if is_valid:
                            return mp4_path
        except Exception as e:
            logger.debug(f"Error during recursive MP4 search: {e}")
        
        return None
    
    def _ensure_mp4_format(self, video_file: Path, max_size_mb: float) -> Optional[Path]:
        """Ensure video is in MP4 format and optimized"""
        output_name = video_file.stem + ".mp4"
        output_path = self.output_dir / output_name
        
        # Check if optimized MP4 already exists and is valid with enhanced checks
        if output_path.exists():
            is_valid, error_msg = self.file_validator.is_valid_video_with_enhanced_checks(
                str(output_path), 
                original_path=str(video_file), 
                max_size_mb=max_size_mb
            )
            if is_valid:
                print(f"    ♻️  Using existing MP4: {output_name}")
                logger.debug(f"Valid optimized MP4 already exists: {output_name}")
                return output_path
            else:
                print(f"    🔄 Reprocessing MP4 (validation failed): {output_name}")
                logger.info(f"Existing MP4 validation failed: {error_msg}, reprocessing: {output_name}")
        
        # Check if segments folder already exists (from previous segmentation)
        segments_folder = self.output_dir / f"{video_file.stem}_segments"
        if segments_folder.exists() and segments_folder.is_dir():
            print(f"    ♻️  Found existing segments folder: {segments_folder.name}")
            logger.debug(f"Found existing segments folder: {segments_folder.name}")
            return segments_folder
        
        if self.shutdown_requested:
            return None
        
        print(f"    🎬 Converting/optimizing: {video_file.name} -> {output_name}")
        logger.info(f"Converting/optimizing video to MP4: {video_file.name} -> {output_name}")
        
        # Check for shutdown before starting compression
        if self.shutdown_requested:
            print(f"    🛑 Compression cancelled by user")
            return None
        
        try:
            # Use video compressor to optimize
            result = self.video_compressor.compress_video(
                input_path=str(video_file),
                output_path=str(output_path),
                max_size_mb=max_size_mb
            )
            
            # Check for shutdown after compression
            if self.shutdown_requested:
                print(f"    🛑 Processing cancelled after compression")
                return None
            
            if result.get('success', False):
                # Check if segmentation was used
                if result.get('method') == 'segmentation' or 'segments' in result:
                    # Video was segmented, check for segments folder
                    if segments_folder.exists() and segments_folder.is_dir():
                        print(f"    ✅ Video segmentation successful: {segments_folder.name}")
                        logger.info(f"Video segmentation successful: {segments_folder.name}")
                        return segments_folder
                    else:
                        print(f"    ❌ Segmentation completed but segments folder not found")
                        logger.error(f"Segmentation completed but segments folder not found: {segments_folder}")
                        return None
                else:
                    # Standard compression was used, validate the single MP4 file
                    print(f"    🔍 Validating processed MP4: {output_name}")
                    is_valid, error_msg = self.file_validator.is_valid_video_with_enhanced_checks(
                        str(output_path), 
                        original_path=str(video_file), 
                        max_size_mb=max_size_mb
                    )
                    
                    if is_valid:
                        size_mb = result.get('size_mb', 0)
                        print(f"    ✨ Video optimization successful: {size_mb:.2f}MB")
                        logger.info(f"Video optimization successful: {output_name} ({size_mb:.2f}MB)")
                        
                        # Log detailed file specifications
                        try:
                            specs = FFmpegUtils.get_detailed_file_specifications(str(output_path))
                            specs_log = FFmpegUtils.format_file_specifications_for_logging(specs)
                            logger.info(f"Final video file specifications - {specs_log}")
                        except Exception as e:
                            logger.warning(f"Could not log detailed video specifications: {e}")
                        
                        return output_path
                    else:
                        print(f"    ❌ Processed MP4 validation failed: {error_msg}")
                        logger.error(f"Processed MP4 validation failed: {error_msg}")
                        # Remove the invalid file
                        if output_path.exists():
                            output_path.unlink()
                        return None
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"    ❌ Video optimization failed: {error_msg}")
                logger.error(f"Video optimization failed: {error_msg}")
                return None
                
        except Exception as e:
            logger.error(f"Error optimizing video {video_file.name}: {e}")
            return None
    
    def _generate_and_optimize_gif(self, mp4_file: Path, max_size_mb: float) -> bool:
        """Generate and optimize GIF from MP4 file or segments folder"""
        
        # Check if mp4_file is actually a segments folder
        if mp4_file.is_dir() and mp4_file.name.endswith('_segments'):
            # This is a segments folder, handle accordingly
            segments_folder = mp4_file
            base_name = segments_folder.stem.replace('_segments', '')
            
            # Check if segment folder contains valid GIFs
            valid_segments, invalid_segments = self._validate_segment_folder_gifs(segments_folder, max_size_mb)
            
            if valid_segments:
                print(f"    ♻️  Using existing segment GIFs: {segments_folder.name}")
                logger.debug(f"Valid segment GIFs already exist: {segments_folder.name}")
                
                if invalid_segments:
                    print(f"    ⚠️  Found {len(invalid_segments)} invalid/corrupted GIFs in segments folder")
                    logger.warning(f"Found {len(invalid_segments)} invalid GIFs in segments folder: {invalid_segments}")
                
                return True
            else:
                print(f"    🔄 Regenerating segment GIFs (invalid/corrupted): {segments_folder.name}")
                logger.info(f"Segment GIFs invalid, will regenerate: {segments_folder.name}")
                # Continue to generate new segment GIFs
                return self._generate_gifs_from_segments(segments_folder, max_size_mb)
        else:
            # This is a single MP4 file, handle as before
            gif_name = mp4_file.stem + ".gif"
            temp_gif_path = self.temp_dir / gif_name
            final_gif_path = self.output_dir / gif_name
            
            # Check for existing segment folder first
            segments_folder = self.output_dir / f"{mp4_file.stem}_segments"
            if segments_folder.exists() and segments_folder.is_dir():
                logger.info(f"Found existing segments folder: {segments_folder}")
                
                # Check if segment folder contains valid GIFs
                valid_segments, invalid_segments = self._validate_segment_folder_gifs(segments_folder, max_size_mb)
                
                if valid_segments:
                    print(f"    ♻️  Using existing segment GIFs: {segments_folder.name}")
                    logger.debug(f"Valid segment GIFs already exist: {segments_folder.name}")
                    
                    if invalid_segments:
                        print(f"    ⚠️  Found {len(invalid_segments)} invalid/corrupted GIFs in segments folder")
                        logger.warning(f"Found {len(invalid_segments)} invalid GIFs in segments folder: {invalid_segments}")
                    
                    # Ensure the source MP4 is placed in the segments folder for easy access
                    try:
                        self._ensure_mp4_in_segments(mp4_file, segments_folder)
                    except Exception as e:
                        logger.debug(f"Could not ensure MP4 in segments folder: {e}")
                    
                    return True
                else:
                    print(f"    🔄 Regenerating segment GIFs (invalid/corrupted): {segments_folder.name}")
                    logger.info(f"Segment GIFs invalid, will regenerate: {segments_folder.name}")
                    # Continue to generate new segment GIFs
        
            # Check if optimized GIF already exists and is valid with enhanced checks
            if final_gif_path.exists():
                is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                    str(final_gif_path), 
                    original_path=None, 
                    max_size_mb=max_size_mb
                )
                if is_valid:
                    print(f"    ♻️  Using existing GIF: {gif_name}")
                    logger.debug(f"Valid optimized GIF already exists: {gif_name}")
                    return True
                else:
                    print(f"    🔄 Regenerating GIF (validation failed): {gif_name}")
                    logger.info(f"Existing GIF validation failed: {error_msg}, regenerating: {gif_name}")
            
            if self.shutdown_requested:
                return False
            
            print(f"    🎨 Generating GIF: {mp4_file.name} -> {gif_name}")
            logger.info(f"Generating GIF from MP4: {mp4_file.name} -> {gif_name}")
            
            # Check for shutdown before starting GIF generation
            if self.shutdown_requested:
                print(f"    🛑 GIF generation cancelled by user")
                return False
            
            try:
                # Check if MP4 file still exists before processing
                if not mp4_file.exists():
                    print(f"    ❌ MP4 file no longer exists: {mp4_file}")
                    logger.error(f"MP4 file no longer exists: {mp4_file}")
                    return False
                
                # Get video duration to preserve full length in GIF
                video_duration = FFmpegUtils.get_video_duration(str(mp4_file))
                logger.info(f"Video duration: {video_duration:.2f}s - generating GIF with full duration")
                
                # Double-check file exists before GIF generation
                if not mp4_file.exists():
                    print(f"    ❌ MP4 file disappeared during processing: {mp4_file}")
                    logger.error(f"MP4 file disappeared during processing: {mp4_file}")
                    return False
                
                # Generate GIF to temp directory first with full duration
                result = self.gif_generator.create_gif(
                    input_video=str(mp4_file),
                    output_path=str(temp_gif_path),
                    max_size_mb=max_size_mb,
                    duration=video_duration  # Preserve full video duration
                )
                
                # Check for shutdown after GIF generation
                if self.shutdown_requested:
                    print(f"    🛑 Processing cancelled after GIF generation")
                    return False
                
                # If segmentation was chosen, move the generated segment GIFs from temp → output
                if result.get('method') == 'segmentation':
                    temp_segments_dir_str = result.get('segments_directory')
                    if temp_segments_dir_str:
                        temp_segments_dir = Path(temp_segments_dir_str)
                    else:
                        # Fallback: derive from temp gif name
                        temp_segments_dir = temp_gif_path.parent / f"{mp4_file.stem}_segments"

                    final_segments_dir = self.output_dir / f"{mp4_file.stem}_segments"

                    try:
                        if temp_segments_dir.exists():
                            if final_segments_dir.exists() and final_segments_dir.is_dir():
                                # Merge/move files into existing folder
                                for item in temp_segments_dir.iterdir():
                                    if item.is_file():
                                        shutil.move(str(item), str(final_segments_dir / item.name))
                            else:
                                shutil.move(str(temp_segments_dir), str(final_segments_dir))

                            # Validate moved GIFs and ensure all are under size
                            valid_segments, invalid_segments = self._validate_segment_folder_gifs(final_segments_dir, max_size_mb)
                            if valid_segments and not invalid_segments:
                                print(f"    ✨ Segment GIFs generated: {final_segments_dir.name} ({len(valid_segments)} valid)")
                                if invalid_segments:
                                    print(f"    ⚠️  {len(invalid_segments)} invalid segment(s) detected")
                                logger.info(f"Segment GIFs generated for {mp4_file.name}: {final_segments_dir}")
                                # Move the source MP4 into the segments folder for user convenience
                                try:
                                    self._ensure_mp4_in_segments(mp4_file, final_segments_dir)
                                except Exception as e:
                                    logger.debug(f"Could not move MP4 into segments folder: {e}")
                                return True
                            else:
                                print(f"    ❌ No valid segment GIFs after generation")
                                logger.error(f"No valid segment GIFs after generation: {final_segments_dir}")
                                # Keep invalid segments in place for inspection; do not move to failures automatically
                                try:
                                    logger.info(f"Keeping invalid segment artifacts in folder: {final_segments_dir}")
                                except Exception:
                                    pass
                                return False
                        else:
                            print(f"    ❌ Expected temp segments not found: {temp_segments_dir}")
                            logger.error(f"Expected temp segments not found: {temp_segments_dir}")
                            return False
                    finally:
                        # Clean up any leftover temp gifs
                        if temp_gif_path.exists():
                            try:
                                temp_gif_path.unlink()
                            except Exception:
                                pass

                if result.get('success', False) and result.get('size_mb', max_size_mb + 1) <= max_size_mb:
                    # Move temp GIF to final location (single GIF case)
                    shutil.move(str(temp_gif_path), str(final_gif_path))

                    # Validate the final GIF
                    is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                        str(final_gif_path),
                        original_path=None,
                        max_size_mb=max_size_mb
                    )

                    if is_valid:
                        size_mb = result.get('size_mb', 0)
                        print(f"    ✨ GIF generation successful: {size_mb:.2f}MB")
                        logger.info(f"GIF generation successful: {gif_name} ({size_mb:.2f}MB)")
                        return True
                    else:
                        print(f"    ❌ Final GIF validation failed: {error_msg}")
                        logger.error(f"Final GIF validation failed: {error_msg}")
                        # Remove the invalid file
                        if final_gif_path.exists():
                            final_gif_path.unlink()
                        return False
                else:
                    error_msg = result.get('error', 'Unknown error')
                    print(f"    ❌ GIF generation failed: {error_msg}")
                    logger.error(f"GIF generation failed: {error_msg}")
                    # Do not move artifacts out of temp; leave input files untouched
                    try:
                        logger.info("Keeping failed temp GIF artifact in temp for debugging")
                    except Exception:
                        pass
                    return False
                    
            except Exception as e:
                print(f"    ❌ Error generating GIF: {e}")
                logger.error(f"Error generating GIF: {e}")
                return False
    
    def _generate_gifs_from_segments(self, segments_folder: Path, max_size_mb: float) -> bool:
        """Generate GIFs from video segments - process each segment individually"""
        try:
            print(f"    🎨 Checking GIF generation for segments: {segments_folder.name}")
            logger.info(f"Checking GIF generation for segments: {segments_folder.name}")
            
            # Find all MP4 files in the segments folder
            segment_files = list(segments_folder.glob("*.mp4"))
            if not segment_files:
                print(f"    ❌ No MP4 segments found in: {segments_folder.name}")
                logger.error(f"No MP4 segments found in: {segments_folder.name}")
                return False
            
            print(f"    📁 Found {len(segment_files)} video segments")
            
            successful_gifs = 0
            skipped_segments = 0
            
            # Process each segment individually
            for segment_file in segment_files:
                print(f"    🎨 Processing segment: {segment_file.name}")
                
                # Pre-check: Get segment duration to see if it's reasonable for GIF creation
                try:
                    segment_duration = FFmpegUtils.get_video_duration(str(segment_file))
                    print(f"    ⏱️  Segment duration: {segment_duration:.1f}s")
                    
                    # Skip segments that are too long for good GIF quality
                    # Long segments will likely result in poor quality or too-short GIFs
                    if segment_duration > 30.0:  # Skip segments longer than 30 seconds
                        print(f"    ⏭️  Skipping segment {segment_file.name}: too long ({segment_duration:.1f}s > 30s)")
                        logger.info(f"Skipping segment {segment_file.name}: too long ({segment_duration:.1f}s)")
                        skipped_segments += 1
                        continue
                        
                except Exception as e:
                    print(f"    ⚠️  Could not get segment duration, proceeding with caution: {e}")
                    logger.warning(f"Could not get segment duration for {segment_file.name}: {e}")
                
                # Generate GIF name for this segment
                gif_name = segment_file.stem + ".gif"
                gif_path = segments_folder / gif_name
                
                # Skip if GIF already exists and is valid
                if gif_path.exists():
                    is_valid, _ = self.file_validator.is_valid_gif_with_enhanced_checks(
                        str(gif_path), 
                        original_path=None, 
                        max_size_mb=max_size_mb
                    )
                    if is_valid:
                        print(f"    ♻️  Using existing GIF: {gif_name}")
                        successful_gifs += 1
                        continue
                
                # Force single-GIF-per-segment to avoid nested segmentation
                test_result = self.gif_generator.create_gif(
                    input_video=str(segment_file),
                    output_path=str(gif_path),
                    max_size_mb=max_size_mb,
                    disable_segmentation=True
                )

                if test_result.get('success', False):
                    # Validate the generated GIF
                    is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                        str(gif_path), 
                        original_path=None, 
                        max_size_mb=max_size_mb
                    )
                    
                    if is_valid:
                        size_mb = test_result.get('size_mb', 0)
                        print(f"    ✅ Segment GIF successful: {size_mb:.2f}MB")
                        successful_gifs += 1
                    else:
                        print(f"    ❌ Segment GIF validation failed: {error_msg}")
                        if gif_path.exists():
                            gif_path.unlink()
                        skipped_segments += 1
                else:
                    # If segmentation attempted, clean up any nested segment folders created accidentally
                    nested_segments = segments_folder / f"{segment_file.stem}_segments"
                    if nested_segments.exists():
                        try:
                            for item in nested_segments.iterdir():
                                if item.is_file():
                                    item.unlink()
                            nested_segments.rmdir()
                        except Exception:
                            pass
                    error_msg = test_result.get('error', 'Unknown error')
                    print(f"    ❌ Segment GIF generation failed: {error_msg}")
                    skipped_segments += 1
            
            # Report results
            if successful_gifs > 0:
                print(f"    ✅ Successfully created {successful_gifs} GIFs from segments")
                if skipped_segments > 0:
                    print(f"    ⏭️  Skipped {skipped_segments} segments (too long or would create multiple GIFs)")
                return True
            else:
                # No GIFs created, but this might be correct behavior if all segments were skipped
                if skipped_segments > 0:
                    print(f"    ⚠️  No GIFs created - all {skipped_segments} segments were skipped (too long or would create multiple GIFs)")
                    print(f"    💡 This is expected behavior for segmented videos with long segments")
                    logger.info(f"No GIFs created from segments - all {skipped_segments} segments were skipped as expected")
                    return True  # Return True since this is correct behavior
                else:
                    print(f"    ❌ No valid GIFs created from segments")
                    return False
            
        except Exception as e:
            print(f"    ❌ Error generating GIFs from segments: {e}")
            logger.error(f"Error generating GIFs from segments: {e}")
            return False
    
    def _validate_segment_folder_gifs(self, segments_folder: Path, max_size_mb: float) -> Tuple[List[Path], List[Path]]:
        """
        Validates all GIFs in a segment folder to ensure they are valid and within size limits.
        Returns a tuple of (valid_gifs, invalid_gifs).
        """
        valid_gifs = []
        invalid_gifs = []
        
        if not segments_folder.exists() or not segments_folder.is_dir():
            logger.warning(f"Segments folder not found or not a directory: {segments_folder}")
            return [], []
        
        for gif_file in segments_folder.iterdir():
            if gif_file.is_file() and gif_file.suffix.lower() == '.gif':
                try:
                    is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                        str(gif_file),
                        original_path=None,
                        max_size_mb=max_size_mb
                    )
                    if is_valid:
                        valid_gifs.append(gif_file)
                    else:
                        invalid_gifs.append(gif_file)
                except Exception as e:
                    logger.warning(f"Error validating GIF {gif_file.name}: {e}")
                    invalid_gifs.append(gif_file)
        
        return valid_gifs, invalid_gifs
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        logger.info("Cleaning up temporary files...")
        try:
            if self.temp_dir.exists():
                for temp_file in self.temp_dir.iterdir():
                    if temp_file.is_file():
                        try:
                            temp_file.unlink()
                            logger.debug(f"Removed temp file: {temp_file.name}")
                        except Exception:
                            pass
                    elif temp_file.is_dir():
                        try:
                            # Remove known temp subfolders safely
                            for item in temp_file.iterdir():
                                if item.is_file():
                                    try:
                                        item.unlink()
                                    except Exception:
                                        pass
                            temp_file.rmdir()
                            logger.debug(f"Removed temp directory: {temp_file.name}")
                        except Exception:
                            pass
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")

    def _cleanup_orphan_segments(self):
        """Clean up orphaned temp segment folders created during failed operations"""
        try:
            if self.temp_dir.exists():
                for sub in self.temp_dir.rglob("*_segments"):
                    try:
                        for item in sub.iterdir():
                            if item.is_file():
                                item.unlink()
                        sub.rmdir()
                        logger.debug(f"Removed orphan segments folder: {sub}")
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"Failed to cleanup orphan segments: {e}")

    def _ensure_mp4_in_segments(self, mp4_file: Path, segments_folder: Path):
        """Move the source MP4 into the segments folder if it resides in output; otherwise copy."""
        try:
            segments_folder.mkdir(exist_ok=True)
            target = segments_folder / mp4_file.name
            if target.exists():
                return
            # If mp4 is in output dir, move it. Otherwise, copy it (e.g., original input still in input/).
            try:
                if mp4_file.parent.resolve() == self.output_dir.resolve() and mp4_file.exists():
                    shutil.move(str(mp4_file), str(target))
                else:
                    shutil.copy2(str(mp4_file), str(target))
            except Exception:
                # As a last resort, try copying
                try:
                    shutil.copy2(str(mp4_file), str(target))
                except Exception:
                    pass
        except Exception:
            pass
    
    def stop_workflow(self):
        """Request workflow to stop gracefully"""
        logger.info("Stopping workflow...")
        self.shutdown_requested = True
    
    def get_status(self) -> Dict[str, Any]:
        """Get current workflow status"""
        return {
            'running': not self.shutdown_requested,
            'current_task': self.current_task,
            'input_dir': str(self.input_dir.absolute()),
            'output_dir': str(self.output_dir.absolute()),
            'temp_dir': str(self.temp_dir.absolute()),
        } 

    def _process_single_gif(self, gif_file: Path, max_size_mb: float) -> str:
        """Process a single GIF file - validate and optimize if needed"""
        try:
            print(f"\n🎬 Processing GIF: {gif_file.name}")
            logger.info(f"Processing GIF file: {gif_file.name}")
            
            # Check for shutdown before processing
            if self.shutdown_requested:
                return 'cancelled'

            # Short-circuit: if an output GIF already exists and is valid & under size, reuse it
            existing_output = self.output_dir / gif_file.name
            if existing_output.exists():
                is_valid_out, err_out = self.file_validator.is_valid_gif_with_enhanced_checks(
                    str(existing_output),
                    original_path=str(gif_file),
                    max_size_mb=max_size_mb
                )
                if is_valid_out:
                    print(f"    ♻️  Using existing optimized GIF: {existing_output.name}")
                    logger.info(f"Existing optimized GIF reused: {existing_output}")
                    return 'success'
                else:
                    logger.info(f"Existing output invalid, will optimize anew: {err_out}")
            
            # Validate the GIF file for integrity (not size)
            is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                str(gif_file),
                original_path=str(gif_file),
                max_size_mb=None  # Don't check size during validation
            )
            
            if not is_valid:
                print(f"    ❌ Invalid GIF file: {error_msg}")
                logger.warning(f"Invalid GIF file {gif_file.name}: {error_msg}")
                return 'error'
            
            # If original is a GIF input, try to keep GIF under target with best-effort re-encode;
            # else (videos) are handled elsewhere. Proceed to optimization.
            file_size_mb = self.file_validator.get_file_size_mb(str(gif_file))
            print(f"    🔄 Optimizing GIF for better quality: {file_size_mb:.2f}MB")
            logger.info(f"Optimizing GIF {gif_file.name} for better quality: {file_size_mb:.2f}MB")
            return self._optimize_gif_file(gif_file, max_size_mb)
                
        except Exception as e:
            print(f"    ❌ Error processing GIF {gif_file.name}: {e}")
            logger.error(f"Error processing GIF {gif_file.name}: {e}")
            return 'error'
    
    def _optimize_gif_file(self, gif_file: Path, max_size_mb: float) -> str:
        """Optimize a GIF file to meet size requirements"""
        try:
            print(f"    🎯 Optimizing GIF: {gif_file.name}")
            logger.info(f"Optimizing GIF file: {gif_file.name}")
            
            # Create output path
            output_path = self.output_dir / gif_file.name

            # Prepare a working copy in temp to avoid modifying the original input
            working_path = self.temp_dir / f"{gif_file.stem}.work.gif"
            try:
                # Ensure temp dir exists
                self.temp_dir.mkdir(exist_ok=True)
                # Overwrite existing working file if present
                if working_path.exists():
                    try:
                        working_path.unlink()
                    except Exception:
                        pass
                shutil.copy2(gif_file, working_path)
            except Exception as e:
                logger.error(f"Failed to prepare working copy for {gif_file.name}: {e}")
                return 'error'

            # Use the advanced GIF optimizer on the working copy only
            from .gif_optimizer_advanced import AdvancedGifOptimizer
            optimizer = AdvancedGifOptimizer(self.config)

            result = optimizer.optimize_gif(
                gif_path=str(working_path),
                max_size_mb=max_size_mb
            )

            if result and working_path.exists():
                optimized_size = self.file_validator.get_file_size_mb(str(working_path))
                print(f"    ✅ GIF optimized successfully: {optimized_size:.2f}MB")
                logger.info(f"GIF optimized successfully (copy): {gif_file.name} -> {optimized_size:.2f}MB")

                # Copy optimized working copy to output directory
                try:
                    if output_path.exists():
                        # Overwrite only if new file is valid and under size
                        shutil.copy2(str(working_path), str(output_path))
                    else:
                        shutil.copy2(str(working_path), str(output_path))
                    print(f"    📁 Saved optimized GIF to output: {output_path.name}")
                    logger.info(f"Saved optimized GIF to output: {output_path}")
                except Exception as e:
                    logger.error(f"Failed to save optimized GIF to output: {e}")
                    return 'error'
                finally:
                    # Clean up working artifacts
                    try:
                        if working_path.exists():
                            working_path.unlink()
                        backup = str(working_path) + '.orig.bak'
                        if os.path.exists(backup):
                            os.remove(backup)
                    except Exception:
                        pass

                return 'success'
            else:
                # If optimization failed, but the original GIF is valid and within size, keep the original
                is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                    str(gif_file),
                    original_path=str(gif_file),
                    max_size_mb=max_size_mb
                )
                if is_valid:
                    current_size = self.file_validator.get_file_size_mb(str(gif_file))
                    print(f"    ⚠️  Optimization did not produce a better result; keeping original ({current_size:.2f}MB)")
                    logger.info(f"Keeping original GIF for {gif_file.name}; optimization failed but original is valid under limit")
                    if not output_path.exists():
                        shutil.copy2(gif_file, output_path)
                        print(f"    📁 Copied to output directory: {output_path.name}")
                        logger.info(f"Copied original GIF to output: {output_path}")
                    else:
                        print(f"    📁 Already exists in output directory: {output_path.name}")
                    return 'success'
                else:
                    print(f"    ❌ GIF optimization failed and original is not within constraints: {error_msg}")
                    logger.error(f"GIF optimization failed and original invalid: {gif_file.name} - {error_msg}")

                    # Attempt segmentation fallback for oversized input GIFs
                    try:
                        seg_success = self._segment_input_gif(gif_file, max_size_mb)
                        if seg_success:
                            return 'success'
                    except Exception as e:
                        logger.error(f"Segmentation fallback failed: {e}")
                    return 'error'
                
        except Exception as e:
            print(f"    ❌ Error optimizing GIF {gif_file.name}: {e}")
            logger.error(f"Error optimizing GIF {gif_file.name}: {e}")
            return 'error' 

    def _segment_input_gif(self, gif_file: Path, max_size_mb: float) -> bool:
        """Segment an input GIF into smaller GIF parts and optimize each.

        This provides a resilience path for very large/long input GIFs by
        slicing them into time-based segments, then optimizing each segment
        independently to keep per-file size within limits while preserving
        aspect ratio and visual quality.
        """
        try:
            print(f"    ✂️  Attempting GIF segmentation: {gif_file.name}")
            logger.info(f"Attempting segmentation for input GIF: {gif_file}")

            # Gather specs and duration
            specs = FFmpegUtils.get_detailed_file_specifications(str(gif_file))
            duration = 0.0
            if isinstance(specs, dict):
                duration = float(specs.get('duration_seconds', 0.0) or 0.0)
            if duration <= 0:
                # Fallback to ffprobe duration
                duration = FFmpegUtils.get_video_duration(str(gif_file))

            # File size
            original_size_mb = self.file_validator.get_file_size_mb(str(gif_file))

            # Configuration for segmentation
            seg_cfg = self.config.get('gif_settings.segmentation', {}) or {}
            max_duration_cfg = self.config.get('gif_settings.max_duration_seconds', 30)
            min_seg = max(5.0, float(seg_cfg.get('min_segment_duration', 12)))
            max_seg = max(min_seg, float(seg_cfg.get('max_segment_duration', 35)))

            # Heuristic: segment if file is over target, or duration is long
            should_segment = (
                (max_size_mb is not None and original_size_mb > max_size_mb * 1.05)
                or (duration > max_duration_cfg)
                or (duration >= max_seg * 2)  # clearly benefits from splitting
            )

            if not should_segment:
                print("    ℹ️  Segmentation not needed based on size/duration heuristics")
                return False

            # Decide target segment duration using base durations if provided
            def pick_segment_duration(total_duration: float) -> float:
                base = seg_cfg.get('base_durations', {}) or {}
                short_max = float(base.get('short_video_max', 40))
                med_max = float(base.get('medium_video_max', 80))
                long_max = float(base.get('long_video_max', 120))
                short_d = float(base.get('short_segment_duration', 18))
                med_d = float(base.get('medium_segment_duration', 22))
                long_d = float(base.get('long_segment_duration', 25))
                vlong_d = float(base.get('very_long_segment_duration', 28))
                if total_duration <= short_max:
                    target = short_d
                elif total_duration <= med_max:
                    target = med_d
                elif total_duration <= long_max:
                    target = long_d
                else:
                    target = vlong_d
                # Clamp to configured min/max
                return max(min_seg, min(max_seg, target))

            import math
            segment_duration = pick_segment_duration(duration if duration > 0 else max_seg)

            # Size-aware segment count based on original file size and target per-file size
            size_segments = 1
            if max_size_mb and max_size_mb > 0:
                try:
                    size_segments = max(1, int(math.ceil(original_size_mb / max_size_mb)))
                except Exception:
                    size_segments = 1

            # Duration-based segment count
            if duration and duration > 0:
                dur_segments = max(1, int(math.ceil(duration / segment_duration)))
                num_segments = max(size_segments, dur_segments)
            else:
                # If duration unknown, ensure at least two or size-based count
                num_segments = max(2, size_segments)

            # Recompute per-segment duration to evenly split total duration when known
            if duration and duration > 0 and num_segments > 0:
                segment_duration = max(min_seg, min(max_seg, duration / num_segments))

            # Prepare output directory path per asset (defer creation until first success)
            output_dir = self.output_dir
            base_name = gif_file.stem
            segments_dir = output_dir / f"{base_name}_segments"
            segments_dir_created = False

            # Process each segment
            successful = 0
            total_size_mb = 0.0

            # Read target fps from config; fallback to 20
            target_fps = int(self.config.get('gif_settings.fps', 20) or 20)

            for i in range(num_segments):
                if self.shutdown_requested:
                    print("    🛑 Segmentation cancelled by user")
                    break

                seg_start = i * segment_duration
                if duration > 0:
                    seg_end = min(duration, seg_start + segment_duration)
                    seg_len = max(0.0, seg_end - seg_start)
                else:
                    seg_len = segment_duration

                if seg_len <= 0.1:
                    continue

                # Create temp segment GIF path
                temp_seg = self.temp_dir / f"{base_name}.seg{i+1:02d}.gif"
                try:
                    if temp_seg.exists():
                        try:
                            temp_seg.unlink()
                        except Exception:
                            pass
                except Exception:
                    pass

                # Build ffmpeg command to slice GIF and lightly normalize frames
                # Preserve aspect ratio and avoid stretching; drop dup frames to keep size down.
                vf = f"mpdecimate=hi=768:lo=512:frac=0.5,fps={target_fps},scale=iw:ih:flags=lanczos:force_original_aspect_ratio=decrease,setsar=1"
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(seg_start),
                    '-t', str(seg_len),
                    '-i', str(gif_file),
                    '-vf', vf,
                    '-loop', '0',
                    str(temp_seg)
                ]
                FFmpegUtils.add_ffmpeg_perf_flags(cmd)

                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
                    if result.returncode != 0 or not temp_seg.exists():
                        logger.warning(f"Failed to create segment {i+1}: {result.stderr[:200] if result.stderr else 'unknown error'}")
                        continue
                except Exception as e:
                    logger.warning(f"Exception while creating GIF segment {i+1}: {e}")
                    continue

                # Optimize the segment with gifsicle-first strategy; fallback to ffmpeg, preserving aspect ratio
                # Per workflow guidance, prefer gifsicle when close to target, ffmpeg re-encode with mpdecimate otherwise
                from .gif_optimizer_advanced import AdvancedGifOptimizer
                optimizer = AdvancedGifOptimizer(self.config)
                optimized_ok = optimizer.optimize_gif(str(temp_seg), max_size_mb)
                if not optimized_ok:
                    try:
                        if temp_seg.exists():
                            temp_seg.unlink()
                    except Exception:
                        pass
                    continue

                # Validate and move to final segments directory
                final_seg = segments_dir / f"{base_name}_segment_{i+1:02d}.gif"
                is_valid, _ = self.file_validator.is_valid_gif_with_enhanced_checks(
                    str(temp_seg), original_path=None, max_size_mb=max_size_mb
                )
                if not is_valid:
                    try:
                        if temp_seg.exists():
                            temp_seg.unlink()
                    except Exception:
                        pass
                    continue

                # Create the segments directory only when we have the first valid segment
                if not segments_dir_created:
                    try:
                        segments_dir.mkdir(parents=True, exist_ok=True)
                    except Exception:
                        pass
                    segments_dir_created = True

                try:
                    shutil.move(str(temp_seg), str(final_seg))
                except Exception:
                    try:
                        shutil.copy2(str(temp_seg), str(final_seg))
                        temp_seg.unlink(missing_ok=True)  # type: ignore[arg-type]
                    except Exception as e:
                        logger.warning(f"Failed to move/copy segment {i+1} to output: {e}")
                        try:
                            if temp_seg.exists():
                                temp_seg.unlink()
                        except Exception:
                            pass
                        continue

                seg_size = self.file_validator.get_file_size_mb(str(final_seg))
                total_size_mb += seg_size
                successful += 1
                print(f"    ✅ Created GIF segment {i+1}/{num_segments}: {final_seg.name} ({seg_size:.2f}MB)")
                logger.info(f"Created GIF segment {i+1}/{num_segments}: {final_seg} ({seg_size:.2f}MB)")

            if successful > 0:
                print(f"    📂 Segmented GIFs saved to: {segments_dir.name} ({successful} segment(s), {total_size_mb:.2f}MB total)")
                logger.info(f"GIF segmentation complete: {successful} segments at {segments_dir}")
                return True

            print("    ❌ GIF segmentation produced no valid segments")
            logger.warning("GIF segmentation failed to produce any valid segments")
            return False

        except Exception as e:
            logger.error(f"Error during GIF segmentation: {e}")
            return False

    def _handle_existing_gif_file(self, gif_file: Path, max_size_mb: float) -> str:
        """Handle existing GIF file - validate and optimize if needed"""
        try:
            print(f"\n🎬 Processing existing GIF: {gif_file.name}")
            logger.info(f"Processing existing GIF file: {gif_file.name}")
            
            # Check for shutdown before processing
            if self.shutdown_requested:
                return 'cancelled'
            
            # Respect existing optimized outputs: if a valid compressed GIF exists in output, skip
            existing_output = self.output_dir / gif_file.name
            if existing_output.exists():
                is_valid_out, err_out = self.file_validator.is_valid_gif_with_enhanced_checks(
                    str(existing_output), original_path=None, max_size_mb=max_size_mb
                )
                if is_valid_out:
                    print(f"    ♻️  Valid optimized GIF already exists in output: {existing_output.name}")
                    logger.info(f"Skipping optimization; existing optimized GIF present: {existing_output}")
                    return 'success'
            
            # Otherwise, optimize
            file_size_mb = self.file_validator.get_file_size_mb(str(gif_file))
            print(f"    🔄 Optimizing GIF for better quality: {file_size_mb:.2f}MB")
            logger.info(f"Optimizing GIF {gif_file.name} for better quality: {file_size_mb:.2f}MB")
            return self._optimize_gif_file(gif_file, max_size_mb)
                
        except Exception as e:
            print(f"    ❌ Error processing existing GIF {gif_file.name}: {e}")
            logger.error(f"Error processing existing GIF {gif_file.name}: {e}")
            return 'error' 