"""
Automated Workflow Module
Handles the automated processing of videos and GIF generation with graceful shutdown
"""

import os
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
        
        # Workflow directories
        self.input_dir = Path("input")
        self.output_dir = Path("output")
        self.temp_dir = Path("temp")
        
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
            else:
                # Second signal - force quit
                print(f"\n💥 Force quit requested. Exiting immediately...")
                logger.warning("Force quit - may leave temporary files")
                
                # Force terminate any running processes
                if hasattr(self.video_compressor, '_terminate_ffmpeg_process'):
                    self.video_compressor._terminate_ffmpeg_process()
                
                os._exit(1)
        
        # Handle common shutdown signals
        signal.signal(signal.SIGINT, signal_handler)  # Ctrl+C
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)  # Termination signal
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        for directory in [self.input_dir, self.output_dir, self.temp_dir]:
            directory.mkdir(exist_ok=True)
            logger.debug(f"Directory ensured: {directory}")
    
    def _interruptible_sleep(self, duration: float, check_interval: float = 0.1):
        """Sleep that can be interrupted by shutdown signal"""
        elapsed = 0.0
        while elapsed < duration and not self.shutdown_requested:
            sleep_time = min(check_interval, duration - elapsed)
            time.sleep(sleep_time)
            elapsed += sleep_time
    
    def run_automated_workflow(self, check_interval: int = 5, max_size_mb: float = 10.0):
        """
        Run the automated workflow
        
        Args:
            check_interval: How often to check for new files (seconds)
            max_size_mb: Maximum file size for outputs in MB
        """
        logger.info("Starting automated workflow...")
        logger.info(f"Input directory: {self.input_dir.absolute()}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        logger.info(f"Temp directory: {self.temp_dir.absolute()}")
        logger.info(f"Check interval: {check_interval} seconds")
        logger.info(f"Max file size: {max_size_mb}MB")
        
        print(f"\n🚀 Automated workflow started successfully!")
        print(f"📁 Monitoring: {self.input_dir.absolute()}")
        print(f"📤 Output to: {self.output_dir.absolute()}")
        print(f"⚙️  Settings: Check every {check_interval}s, Max size {max_size_mb}MB")
        print(f"🎯 Supported formats: {', '.join(self.file_validator.get_supported_video_extensions())}")
        
        processed_files = set()
        processing_stats = {'successful': 0, 'skipped': 0, 'errors': 0}
        first_scan = True
        
        try:
            while not self.shutdown_requested:
                try:
                    # Find new video files in input directory
                    video_files = self._find_new_video_files(processed_files, skip_stability_check=first_scan)
                    first_scan = False
                    
                    if video_files:
                        logger.info(f"Found {len(video_files)} new video file(s) to process")
                        print(f"\n🎬 Found {len(video_files)} new video file(s) to process...")
                        
                        for video_file in video_files:
                            if self.shutdown_requested:
                                break
                            
                            # Check if this file has existing video output
                            has_existing_video = self._has_existing_output(video_file)
                            
                            if has_existing_video:
                                # Video processing already done, but check if GIF creation is needed
                                result = self._handle_existing_video_file(video_file, max_size_mb)
                            else:
                                # Full processing needed (video + GIF)
                                result = self._process_single_video(video_file, max_size_mb)
                            
                            processed_files.add(video_file)
                            
                            # Update stats
                            if result == 'success':
                                processing_stats['successful'] += 1
                            elif result == 'skipped':
                                processing_stats['skipped'] += 1
                            else:
                                processing_stats['errors'] += 1
                        
                        # Show processing summary
                        if video_files:
                            total = processing_stats['successful'] + processing_stats['skipped'] + processing_stats['errors']
                            print(f"\n📊 Processing Summary: ✅ {processing_stats['successful']} successful, "
                                  f"⚠️  {processing_stats['skipped']} skipped, ❌ {processing_stats['errors']} errors "
                                  f"(Total: {total})")
                    else:
                        # Show waiting status periodically
                        current_time = time.strftime("%H:%M:%S")
                        print(f"\r⏳ [{current_time}] Waiting for new video files in input/ folder...", end="", flush=True)
                    
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
            print(f"\n🧹 Cleaning up temporary files...")
            self._cleanup_temp_files()
            print(f"✅ Automated workflow stopped gracefully")
            logger.info("Automated workflow stopped")
    
    def _find_new_video_files(self, processed_files: set, skip_stability_check: bool = False) -> List[Path]:
        """Find new video files in input directory that haven't been processed yet"""
        video_files = []
        
        if not self.input_dir.exists():
            return video_files
        
        supported_extensions = self.file_validator.get_supported_video_extensions()
        
        # First, collect all potential video files
        potential_files = []
        for file_path in self.input_dir.iterdir():
            if (file_path.is_file() and 
                file_path.suffix.lower() in supported_extensions and
                file_path not in processed_files):
                potential_files.append(file_path)
        
        # If we have many files, show progress
        if len(potential_files) > 10:
            if skip_stability_check:
                print(f"🔍 Scanning {len(potential_files)} video files (initial scan - skipping stability check)...")
            else:
                print(f"🔍 Scanning {len(potential_files)} video files for stability and existing outputs...")
        
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
                video_files.append(file_path)
        
        if len(potential_files) > 10:
            print(f"\r✅ File scan complete: {len(video_files)} files ready for processing" + " " * 20)
        
        return sorted(video_files)
    
    def _has_existing_output(self, input_file: Path) -> bool:
        """
        Check for existing VIDEO output files only (not GIFs).
        
        This allows the workflow to skip video processing if an optimized MP4 already exists,
        but still proceed with GIF creation which is a separate step.
        
        Checks for:
        1. Optimized MP4 in root output directory
        2. Optimized MP4 in subdirectories (recursive search)
        
        Args:
            input_file: Path to the input video file
            
        Returns:
            True if valid MP4 output exists, False otherwise
        """
        if not self.output_dir.exists():
            return False
        
        base_name = input_file.stem
        
        # Only check for MP4 outputs (video processing)
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
                # Validate single GIF file
                is_valid, _ = self.file_validator.is_valid_gif(str(output_path), max_size_mb=10.0)
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
            print("  🔍 Step 1/3: Validating video file...")
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
            print("  🔄 Step 2/3: Converting/optimizing to MP4...")
            mp4_file = self._ensure_mp4_format(video_file, max_size_mb)
            if not mp4_file or self.shutdown_requested:
                return 'error' if not self.shutdown_requested else 'cancelled'
            print(f"  ✅ MP4 optimization complete: {mp4_file.name}")
            
            # Check for shutdown before step 3
            if self.shutdown_requested:
                return 'cancelled'
            
            # Step 3: Generate and optimize GIF (ALWAYS attempt this step)
            print("  🎨 Step 3/3: Generating and optimizing GIF...")
            gif_success = self._generate_and_optimize_gif(mp4_file, max_size_mb)
            
            if gif_success:
                print(f"  ✅ GIF generation complete: {mp4_file.stem}.gif")
                print(f"  🎉 Successfully processed: {video_file.name}")
                logger.info(f"Successfully processed: {video_file.name}")
                return 'success'
            else:
                print(f"  ❌ GIF generation failed for: {video_file.name}")
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
                    original_path=str(mp4_file), 
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
                
                # Validate segment folder GIFs (less strict validation for segments)
                valid_segments, invalid_segments = self._validate_segment_folder_gifs(segments_folder, max_size_mb)
                
                if valid_segments:
                    print(f"  ✅ Valid segments folder exists: {segments_folder.name} ({len(valid_segments)} valid GIFs)")
                    logger.info(f"Valid segments folder already exists for {video_file.name}: {segments_folder.name} with {len(valid_segments)} GIFs")
                    
                    if invalid_segments:
                        print(f"  ⚠️  Found {len(invalid_segments)} invalid/corrupted GIFs in segments folder")
                        logger.warning(f"Found {len(invalid_segments)} invalid GIFs in segments folder: {invalid_segments}")
                    
                    return 'success'
                else:
                    print(f"  🔄 All segments invalid/corrupted, will regenerate")
                    logger.info(f"All segments invalid, will regenerate: {segments_folder.name}")
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
                # Enhanced validation of the processed file
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
        """Generate and optimize GIF from MP4 file"""
        gif_name = mp4_file.stem + ".gif"
        temp_gif_path = self.temp_dir / gif_name
        final_gif_path = self.output_dir / gif_name
        
        # NEW: Check for existing segment folder first
        segments_folder = self.output_dir / f"{mp4_file.stem}_segments"
        if segments_folder.exists() and segments_folder.is_dir():
            logger.info(f"Found existing segments folder: {segments_folder}")
            
            # Check if segment folder contains valid GIFs
            valid_segments, invalid_segments = self._validate_segment_folder_gifs(segments_folder, max_size_mb)
            
            if valid_segments:
                print(f"    ♻️  Using existing segments folder: {segments_folder.name} ({len(valid_segments)} valid GIFs)")
                logger.debug(f"Valid segments folder already exists: {segments_folder.name} with {len(valid_segments)} GIFs")
                
                if invalid_segments:
                    print(f"    ⚠️  Found {len(invalid_segments)} invalid/corrupted GIFs in segments folder")
                    logger.warning(f"Found {len(invalid_segments)} invalid GIFs in segments folder: {invalid_segments}")
                
                return True
            else:
                print(f"    🔄 Regenerating segments (all GIFs invalid/corrupted): {segments_folder.name}")
                logger.info(f"All segments invalid, will regenerate: {segments_folder.name}")
                # Continue to generate new segments
        
        # Check if optimized GIF already exists and is valid with enhanced checks
        if final_gif_path.exists():
            is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                str(final_gif_path), 
                original_path=str(mp4_file), 
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
            # Get video duration to preserve full length in GIF
            video_duration = FFmpegUtils.get_video_duration(str(mp4_file))
            logger.info(f"Video duration: {video_duration:.2f}s - generating GIF with full duration")
            
            # Generate GIF to temp directory first with full duration
            result = self.gif_generator.create_gif(
                input_video=str(mp4_file),
                output_path=str(temp_gif_path),
                max_size_mb=max_size_mb,
                duration=video_duration  # Preserve full video duration
            )
            
            # Check for shutdown after GIF generation
            if self.shutdown_requested:
                print(f"    🛑 GIF processing cancelled after generation")
                if temp_gif_path.exists():
                    temp_gif_path.unlink()
                return False
            
            if not result.get('success', False):
                error_msg = result.get('error', 'Unknown error')
                print(f"    ❌ GIF generation failed: {error_msg}")
                logger.error(f"GIF generation failed: {error_msg}")
                
                # Clean up any temporary files from failed segmentation
                if result.get('method') == 'Video Segmentation':
                    temp_folder = result.get('temp_segments_folder')
                    temp_files = result.get('temp_files_to_cleanup', [])
                    
                    if temp_files:
                        logger.info(f"Cleaning up {len(temp_files)} temporary files from failed segmentation")
                        for temp_file in temp_files:
                            try:
                                if os.path.exists(temp_file):
                                    os.remove(temp_file)
                                    logger.debug(f"Cleaned up: {temp_file}")
                            except Exception as e:
                                logger.warning(f"Failed to clean up {temp_file}: {e}")
                    
                    if temp_folder and os.path.exists(temp_folder):
                        try:
                            shutil.rmtree(temp_folder)
                            logger.info(f"Cleaned up temp folder: {temp_folder}")
                        except Exception as e:
                            logger.warning(f"Failed to clean up temp folder {temp_folder}: {e}")
                
                return False
            
            # Handle segmented results differently
            if result.get('method') == 'Video Segmentation':
                # Move segments from temp folder to output folder
                temp_segments_folder = result.get('temp_segments_folder')
                base_name = result.get('base_name')
                segments = result.get('segments', [])
                
                if temp_segments_folder and base_name and segments:
                    # Create final segments folder in output directory
                    final_segments_folder = self.output_dir / f"{base_name}_segments"
                    final_segments_folder.mkdir(exist_ok=True)
                    
                    moved_segments = 0
                    total_moved_size = 0
                    
                    # Move segments from temp to output folder (move GIFs first for proper Windows thumbnails)
                    for segment in segments:
                        temp_path = segment.get('temp_path')
                        segment_name = segment.get('name')
                        
                        if temp_path and segment_name and os.path.exists(temp_path):
                            final_path = final_segments_folder / segment_name
                            
                            try:
                                shutil.move(temp_path, final_path)
                                moved_segments += 1
                                total_moved_size += segment.get('size_mb', 0)
                                logger.info(f"Moved segment: {segment_name} -> {final_path}")
                                
                                # Set Windows file attributes to ensure proper thumbnail generation
                                try:
                                    import subprocess
                                    # Force Windows to generate thumbnail for this GIF
                                    subprocess.run(['attrib', '+A', final_path], capture_output=True)
                                    logger.debug(f"Set Windows attributes for thumbnail: {segment_name}")
                                except Exception as thumb_e:
                                    logger.debug(f"Could not set Windows thumbnail attributes: {thumb_e}")
                                    
                            except Exception as e:
                                logger.error(f"Failed to move segment {segment_name}: {e}")
                                print(f"    ❌ Failed to move segment: {segment_name}")
                    
                    # Move the optimized MP4 video into the segments folder for easy access
                    mp4_source_path = str(mp4_file)
                    mp4_filename = mp4_file.name
                    mp4_dest_path = final_segments_folder / mp4_filename
                    
                    try:
                        if os.path.exists(mp4_source_path):
                            shutil.move(mp4_source_path, mp4_dest_path)
                            logger.info(f"Moved optimized MP4 to segments folder: {mp4_filename}")
                            print(f"    📹 Moved optimized video to segments folder")
                        else:
                            logger.warning(f"Optimized MP4 not found at expected location: {mp4_source_path}")
                    except Exception as e:
                        logger.error(f"Failed to move optimized MP4 to segments folder: {e}")
                        print(f"    ❌ Failed to move optimized video to segments folder")
                    
                    # Move summary file LAST (after all media files) so it doesn't interfere with Windows thumbnails
                    summary_file = Path(temp_segments_folder) / f"{base_name}_segments_info.txt"
                    if summary_file.exists():
                        # Rename summary to start with 'z' to ensure it comes last alphabetically
                        final_summary = final_segments_folder / f"zzz_{base_name}_segments_info.txt"
                        try:
                            shutil.move(summary_file, final_summary)
                            logger.info(f"Moved summary file: {final_summary}")
                        except Exception as e:
                            logger.error(f"Failed to move summary file: {e}")
                    
                    # Clean up temp folder - use more robust cleanup
                    try:
                        temp_segments_path = Path(temp_segments_folder)
                        if temp_segments_path.exists():
                            # First try to remove any remaining files
                            remaining_files = []
                            try:
                                remaining_files = list(temp_segments_path.iterdir())
                            except Exception:
                                pass
                            
                            if remaining_files:
                                logger.info(f"Cleaning up {len(remaining_files)} remaining files in temp folder")
                                for remaining_file in remaining_files:
                                    try:
                                        if remaining_file.is_file():
                                            remaining_file.unlink()
                                            logger.debug(f"Removed remaining file: {remaining_file.name}")
                                    except Exception as file_e:
                                        logger.warning(f"Could not remove remaining file {remaining_file.name}: {file_e}")
                            
                            # Now remove the folder
                            temp_segments_path.rmdir()
                            logger.info(f"Successfully cleaned up temp segments folder: {temp_segments_folder}")
                    except Exception as e:
                        logger.warning(f"Could not remove temp segments folder: {e}")
                        # Try using shutil.rmtree as fallback
                        try:
                            shutil.rmtree(temp_segments_folder)
                            logger.info(f"Successfully cleaned up temp folder using rmtree: {temp_segments_folder}")
                        except Exception as rmtree_e:
                            logger.error(f"Failed to clean up temp folder even with rmtree: {rmtree_e}")
                    
                    # Force Windows to refresh folder view for proper thumbnails
                    try:
                        import subprocess
                        import time
                        
                        # Multiple approaches to force Windows thumbnail generation
                        
                        # 1. Clear thumbnail cache for this folder
                        subprocess.run(['cmd', '/c', 'del', '/q', f'{os.environ.get("LOCALAPPDATA", "")}\\Microsoft\\Windows\\Explorer\\thumbcache_*.db'], 
                                     capture_output=True, check=False)
                        
                        # 2. Refresh the folder to help Windows recognize the media files
                        subprocess.run(['cmd', '/c', 'dir', str(final_segments_folder), '>nul'], 
                                     capture_output=True, check=False)
                        
                        # 3. Force Windows Explorer to refresh (if possible)
                        subprocess.run(['cmd', '/c', 'echo', 'refresh'], capture_output=True, check=False)
                        
                        # 4. Set folder to show large icons (Pictures view)
                        try:
                            # This registry approach might help set the folder view
                            import winreg
                            # Note: This is a simplified approach - full implementation would be more complex
                            logger.debug("Attempted to set folder view preferences")
                        except ImportError:
                            pass
                        
                        logger.debug("Applied multiple Windows thumbnail refresh methods")
                        
                        # Give Windows a moment to process
                        time.sleep(0.5)
                        
                    except Exception as refresh_e:
                        logger.debug(f"Could not refresh Windows folder view: {refresh_e}")
                    
                    print(f"    🎬 Video segmented: {moved_segments} parts ({total_moved_size:.2f}MB total)")
                    print(f"    📁 All files saved to: {final_segments_folder}")
                    print(f"    📹 Includes optimized MP4 + {moved_segments} GIF segments")
                    print(f"    🖼️  Folder should show first GIF as thumbnail (may take a moment)")
                    print(f"    💡 If no thumbnail appears, try refreshing the folder view (F5)")
                    logger.info(f"Video segmentation completed: {moved_segments} segments moved to {final_segments_folder}, "
                               f"Total: {total_moved_size:.2f}MB, MP4 included")
                else:
                    print(f"    ❌ Segmentation data incomplete - segments may remain in temp folder")
                    logger.error("Segmentation result missing required data for file movement")
                
                return True  # Segmentation is considered successful
            
            if self.shutdown_requested:
                return False
            
            # For single GIF results, validate the generated file
            # Enhanced validation of the generated GIF
            print(f"    🔍 Validating generated GIF: {gif_name}")
            is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                str(temp_gif_path), 
                original_path=str(mp4_file), 
                max_size_mb=max_size_mb
            )
            
            if not is_valid:
                print(f"    ❌ Generated GIF validation failed: {error_msg}")
                logger.error(f"Generated GIF validation failed: {error_msg}")
                
                # Clean up failed temp file
                if temp_gif_path.exists():
                    temp_gif_path.unlink()
                    logger.info(f"Deleted failed temp GIF: {temp_gif_path}")
                
                # Also remove any existing output file that might be invalid
                if final_gif_path.exists():
                    final_gif_path.unlink()
                    logger.info(f"Deleted invalid output GIF: {final_gif_path}")
                    print(f"    🗑️  Deleted invalid existing GIF")
                
                return False
            
            # Move validated GIF to output directory
            shutil.move(str(temp_gif_path), str(final_gif_path))
            size_mb = result.get('file_size_mb', 0)
            print(f"    ✨ GIF generation successful: {size_mb:.2f}MB")
            logger.info(f"GIF generation successful: {gif_name} ({size_mb:.2f}MB)")
            
            # Log detailed file specifications
            try:
                specs = FFmpegUtils.get_detailed_file_specifications(str(final_gif_path))
                specs_log = FFmpegUtils.format_file_specifications_for_logging(specs)
                logger.info(f"Final GIF file specifications - {specs_log}")
            except Exception as e:
                logger.warning(f"Could not log detailed GIF specifications: {e}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating GIF from {mp4_file.name}: {e}")
            # Clean up temp file if it exists
            if temp_gif_path.exists():
                try:
                    temp_gif_path.unlink()
                except Exception:
                    pass
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
                        original_path=str(gif_file), # No original path for segment GIFs
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
                        temp_file.unlink()
                        logger.debug(f"Removed temp file: {temp_file.name}")
        except Exception as e:
            logger.warning(f"Error cleaning up temp files: {e}")
    
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