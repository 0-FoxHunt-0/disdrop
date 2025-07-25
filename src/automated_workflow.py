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
from typing import Dict, Any, List, Optional
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed

from .config_manager import ConfigManager
from .hardware_detector import HardwareDetector
from .video_compressor import DynamicVideoCompressor
from .gif_generator import GifGenerator
from .file_validator import FileValidator

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
            else:
                # Second signal - force quit
                print(f"\n💥 Force quit requested. Exiting immediately...")
                logger.warning("Force quit - may leave temporary files")
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
        """Find new video files in input directory"""
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
                print(f"🔍 Scanning {len(potential_files)} video files for stability...")
        
        # Check stability for each file
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
            
            # Step 3: Generate and optimize GIF
            print("  🎨 Step 3/3: Generating and optimizing GIF...")
            self._generate_and_optimize_gif(mp4_file, max_size_mb)
            print(f"  ✅ GIF generation complete: {mp4_file.stem}.gif")
            
            print(f"  🎉 Successfully processed: {video_file.name}")
            logger.info(f"Successfully processed: {video_file.name}")
            return 'success'
            
        except Exception as e:
            print(f"  ❌ Error processing {video_file.name}: {e}")
            logger.error(f"Error processing {video_file.name}: {e}")
            return 'error'
        finally:
            self.current_task = None
    
    def _ensure_mp4_format(self, video_file: Path, max_size_mb: float) -> Optional[Path]:
        """Ensure video is in MP4 format and optimized"""
        output_name = video_file.stem + ".mp4"
        output_path = self.output_dir / output_name
        
        # Check if optimized MP4 already exists and is valid
        if output_path.exists():
            is_valid, _ = self.file_validator.is_valid_video(str(output_path))
            if (is_valid and 
                self.file_validator.is_video_under_size(str(output_path), max_size_mb)):
                print(f"    ♻️  Valid optimized MP4 already exists: {output_name}")
                logger.info(f"Valid optimized MP4 already exists: {output_name}")
                return output_path
            else:
                print(f"    🔄 Existing MP4 is invalid or too large, reprocessing...")
                logger.info(f"Existing MP4 is invalid or too large, reprocessing: {output_name}")
        
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
                size_mb = result.get('size_mb', 0)
                print(f"    ✨ Video optimization successful: {size_mb:.2f}MB")
                logger.info(f"Video optimization successful: {output_name} ({size_mb:.2f}MB)")
                return output_path
            else:
                error_msg = result.get('error', 'Unknown error')
                print(f"    ❌ Video optimization failed: {error_msg}")
                logger.error(f"Video optimization failed: {error_msg}")
                return None
                
        except Exception as e:
            logger.error(f"Error optimizing video {video_file.name}: {e}")
            return None
    
    def _generate_and_optimize_gif(self, mp4_file: Path, max_size_mb: float):
        """Generate and optimize GIF from MP4 file"""
        gif_name = mp4_file.stem + ".gif"
        temp_gif_path = self.temp_dir / gif_name
        final_gif_path = self.output_dir / gif_name
        
        # Check if optimized GIF already exists and is valid
        if final_gif_path.exists():
            is_valid, _ = self.file_validator.is_valid_gif(str(final_gif_path), max_size_mb)
            if is_valid:
                print(f"    ♻️  Valid optimized GIF already exists: {gif_name}")
                logger.info(f"Valid optimized GIF already exists: {gif_name}")
                return
            else:
                print(f"    🔄 Existing GIF is invalid or too large, regenerating...")
                logger.info(f"Existing GIF is invalid or too large, regenerating: {gif_name}")
        
        if self.shutdown_requested:
            return
        
        print(f"    🎨 Generating GIF: {mp4_file.name} -> {gif_name}")
        logger.info(f"Generating GIF from MP4: {mp4_file.name} -> {gif_name}")
        
        # Check for shutdown before starting GIF generation
        if self.shutdown_requested:
            print(f"    🛑 GIF generation cancelled by user")
            return
        
        try:
            # Generate GIF to temp directory first
            result = self.gif_generator.create_gif(
                input_video=str(mp4_file),
                output_path=str(temp_gif_path),
                max_size_mb=max_size_mb
            )
            
            # Check for shutdown after GIF generation
            if self.shutdown_requested:
                print(f"    🛑 GIF processing cancelled after generation")
                if temp_gif_path.exists():
                    temp_gif_path.unlink()
                return
            
            if not result.get('success', False):
                error_msg = result.get('error', 'Unknown error')
                print(f"    ❌ GIF generation failed: {error_msg}")
                logger.error(f"GIF generation failed: {error_msg}")
                return
            
            if self.shutdown_requested:
                return
            
            # Validate the generated GIF
            is_valid, error_msg = self.file_validator.is_valid_gif(str(temp_gif_path), max_size_mb)
            
            if not is_valid:
                logger.error(f"Generated GIF is invalid: {error_msg}")
                if temp_gif_path.exists():
                    temp_gif_path.unlink()
                return
            
            # Move validated GIF to output directory
            shutil.move(str(temp_gif_path), str(final_gif_path))
            size_mb = result.get('file_size_mb', 0)
            print(f"    ✨ GIF generation successful: {size_mb:.2f}MB")
            logger.info(f"GIF generation successful: {gif_name} ({size_mb:.2f}MB)")
            
        except Exception as e:
            logger.error(f"Error generating GIF from {mp4_file.name}: {e}")
            # Clean up temp file if it exists
            if temp_gif_path.exists():
                try:
                    temp_gif_path.unlink()
                except Exception:
                    pass
    
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