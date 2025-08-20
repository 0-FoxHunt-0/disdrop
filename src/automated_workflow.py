"""
Automated Workflow Module
Handles the automated processing of videos and GIF generation with graceful shutdown
"""

import os
import json
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
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

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
        # Provide a shared shutdown checker to subcomponents so threads respect global shutdown immediately
        self.shutdown_requested = False
        def _aw_shutdown_checker():
            return bool(getattr(self, 'shutdown_requested', False))
        self.gif_generator = gif_generator or GifGenerator(config_manager, shutdown_checker=_aw_shutdown_checker)
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
        
        # User preference: prefer single output file (1 segment) when possible
        self.prefer_single_segment = False
        # Generalized preference: prefer exactly N segments when possible
        self.preferred_segments: Optional[int] = None
        # Shutdown handling
        self.shutdown_requested = False
        self.current_task = None
        self.processing_lock = threading.Lock()

        # Caching of previously successful inputs
        self.use_cache: bool = True
        self._cache_index: Dict[str, Any] = {}
        self._cache_dir: Path = Path("temp") / 'cache'
        self._cache_file: Path = self._cache_dir / 'workflow_success_index.json'
        # Session timestamp to track current execution and clean up old cache entries
        self._session_start_time = time.time()
        # Cache persistence: track when cache was last used to allow persistence between executions
        self._cache_last_used_file = self._cache_dir / 'last_used.txt'
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
        # Ensure directories exist
        self._ensure_directories()
        # Load cache index best-effort and clean up old entries
        try:
            self._load_cache_index()
            self._cleanup_old_cache_entries()
            # Validate cache entries to remove any with missing outputs
            validation_result = self.validate_and_clean_cache()
            if validation_result['cleaned'] > 0:
                logger.info(f"Workflow startup: cleaned {validation_result['cleaned']} invalid cache entries")
        except Exception:
            self._cache_index = {}
    
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
                
                # Request shutdown from GIF optimizer if it exists
                if hasattr(self.gif_generator, 'optimizer') and hasattr(self.gif_generator.optimizer, 'request_shutdown'):
                    self.gif_generator.optimizer.request_shutdown()
                # Defer temp cleanup to normal exit to avoid races with in-flight tasks
            else:
                # Second signal - force quit
                print(f"\n💥 Force quit requested. Exiting immediately...")
                logger.warning("Force quit - may leave temporary files")
                
                # Force terminate any running processes
                if hasattr(self.video_compressor, '_terminate_ffmpeg_process'):
                    self.video_compressor._terminate_ffmpeg_process()
                if hasattr(self.gif_generator, '_terminate_ffmpeg_process'):
                    self.gif_generator._terminate_ffmpeg_process()
                if hasattr(self.gif_generator, 'optimizer') and hasattr(self.gif_generator.optimizer, '_terminate_ffmpeg_process'):
                    self.gif_generator.optimizer._terminate_ffmpeg_process()
                
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

        # Ensure nested cache directory exists for performance caching
        try:
            cache_dir = self.temp_dir / 'cache'
            cache_dir.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Cache directory ensured: {cache_dir}")
        except Exception as e:
            logger.warning(f"Could not ensure cache directory: {e}")
    
    def _interruptible_sleep(self, duration: float, check_interval: float = 0.1):
        """Sleep that can be interrupted by shutdown signal"""
        elapsed = 0.0
        while elapsed < duration and not self.shutdown_requested:
            sleep_time = min(check_interval, duration - elapsed)
            time.sleep(sleep_time)
            elapsed += sleep_time
    
    def run_automated_workflow(self, check_interval: int = 5, max_size_mb: float = 10.0, verbose: bool = False, max_files: Optional[int] = None, output_dir: Optional[str] = None, no_cache: bool = False):
        """
        Run the automated workflow
        
        Args:
            check_interval: How often to check for new files (seconds)
            max_size_mb: Maximum file size for outputs in MB
        """
        # Set verbosity for this run
        self.verbose = bool(verbose)

        # Allow overriding output directory via CLI
        if output_dir:
            try:
                self.output_dir = Path(output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not set output directory to '{output_dir}': {e}")
        # Configure cache usage
        self.use_cache = not bool(no_cache)

        logger.info("Starting automated workflow...")
        logger.info(f"Input directory: {self.input_dir.absolute()}")
        logger.info(f"Output directory: {self.output_dir.absolute()}")
        logger.info(f"Temp directory: {self.temp_dir.absolute()}")
        logger.info(f"Check interval: {check_interval} seconds")
        logger.info(f"Max file size: {max_size_mb}MB")
        
        # Ensure all existing segments folders have summaries
        try:
            self._ensure_all_segments_summaries_exist()
        except Exception as e:
            logger.debug(f"Could not ensure all segments summaries exist: {e}")
        
        self._vprint(f"\n🚀 Workflow started")
        self._vprint(f"📁 Input: {self.input_dir.absolute()}")
        self._vprint(f"📤 Output: {self.output_dir.absolute()}")
        self._vprint(f"⚙️  Every {check_interval}s, Max GIF {max_size_mb}MB")
        print(f"👀 File watching is active. Monitoring: {self.input_dir.absolute()} (Ctrl+C to stop)")
        
        processed_files = set()
        processing_stats = {'successful': 0, 'skipped': 0, 'errors': 0, 'processed': 0}
        first_scan = True
        idle_announced = False
        processed_count = 0
        start_time = time.time()
        
        try:
            while not self.shutdown_requested:
                try:
                    # Find new video files in input directory
                    video_files = self._find_new_files(processed_files, skip_stability_check=first_scan)
                    first_scan = False
                    
                    if video_files:
                        # Reset idle announcement when work resumes
                        idle_announced = False
                        logger.info(f"Found {len(video_files)} new file(s) to process")
                        self._vprint(f"\n🎬 {len(video_files)} new file(s) to process")
                        
                        for file_path in video_files:
                            if self.shutdown_requested:
                                break
                            # Respect max_files limiter if provided
                            if max_files is not None and processed_count >= int(max_files):
                                logger.info(f"Max files limit reached ({processed_count}/{max_files}). Stopping workflow.")
                                self.shutdown_requested = True
                                break

                            # Fast-skip using cache if enabled and previously verified success exists
                            if self._is_cached_success(file_path):
                                print(f"\n⚡ Skipping (cached success): {file_path.name}")
                                safe_name = self._safe_filename_for_logging(file_path.name)
                                logger.info(f"Skipping (cached success): {safe_name}")
                                result = 'skipped'
                                # After processing a single file, cleanup temp files immediately
                                self._cleanup_temp_files()
                                processed_files.add(file_path)
                                # Update stats and counters
                                processing_stats['skipped'] += 1
                                processed_count += 1
                                continue
                            
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

                            # Increment processed counter for any attempted file (success/skip/error)
                            processed_count += 1
                            processing_stats['processed'] = processed_count

                            # If processing failed, leave original input in place (do not move from input/)
                            if result == 'error':
                                logger.info(
                                    f"Leaving failed input in place for retry: {file_path.name}"
                                )
                        
                        # Show processing summary
                        if video_files:
                            total = processing_stats['successful'] + processing_stats['skipped'] + processing_stats['errors']
                            self._vprint(f"\n📊 Summary: ✅ {processing_stats['successful']} | ⚠️ {processing_stats['skipped']} | ❌ {processing_stats['errors']} (Total {total})")
                            # Immediately announce standby if we are now idle (no more files ready)
                            try:
                                more_files = self._find_new_files(processed_files, skip_stability_check=True)
                                if not more_files and not idle_announced:
                                    print(f"✅ All caught up. Watching for new files in: {self.input_dir.absolute()}", flush=True)
                                    logger.info("All caught up. Watching for files")
                                    idle_announced = True
                            except Exception:
                                pass
                    else:
                        # Show waiting status periodically (only when verbose)
                        if self.verbose:
                            current_time = time.strftime("%H:%M:%S")
                            print(f"\r⏳ [{current_time}] Waiting for files...", end="", flush=True)
                        else:
                            # In normal mode, show a one-time standby message when idle
                            if not idle_announced:
                                print(f"✅ All caught up. Watching for new files in: {self.input_dir.absolute()}", flush=True)
                                logger.info("All caught up. Watching for files")
                                idle_announced = True
                    
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
            
            # Print final workflow summary with cache statistics
            self._print_workflow_summary(processing_stats, start_time)
            
            self._vprint(f"✅ Automated workflow stopped gracefully")
            logger.info("Automated workflow stopped")

    def _vprint(self, message: str, end: str = "\n", flush: bool = False):
        """Verbose print controlled by self.verbose."""
        if self.verbose:
            print(message, end=end, flush=flush)

    def _record_success_for_input(self, input_path: Path) -> None:
        """Detect final outputs for the given input and record cache.

        Works for both video and GIF inputs by using the stem to find either a
        single GIF in `output/` or a `<stem>_segments` folder with GIFs.
        """
        try:
            stem = input_path.stem
            # Prefer segments when they exist and contain valid GIFs
            segments_dir = self.output_dir / f"{stem}_segments"
            if segments_dir.exists() and segments_dir.is_dir():
                valid_segs, _ = self._validate_segment_folder_gifs(segments_dir, 10.0)
                if valid_segs:
                    self._record_success_cache(input_path, 'segments', segments_dir)
                    return
            # Otherwise, check for single GIF
            single_gif = self.output_dir / f"{stem}.gif"
            if single_gif.exists():
                # Fast check is sufficient for cache record
                is_valid, _ = self.file_validator.is_valid_gif_fast(str(single_gif), max_size_mb=10.0)
                if is_valid:
                    self._record_success_cache(input_path, 'single_gif', single_gif)
        except Exception:
            pass

    def _load_cache_index(self) -> None:
        """Load cache index from disk."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        if self._cache_file.exists():
            try:
                with open(self._cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, dict):
                        self._cache_index = data
                    else:
                        self._cache_index = {}
            except Exception:
                self._cache_index = {}
        else:
            self._cache_index = {}

    def _cleanup_old_cache_entries(self) -> None:
        """Remove cache entries from previous executions to ensure only current session cache is used."""
        if not self._cache_index:
            return
        
        current_time = time.time()
        cleaned_count = 0
        keys_to_remove = []
        
        # Allow cache entries to persist between executions, but clean up very old ones (e.g., >24 hours)
        max_cache_age_hours = 24
        max_cache_age_seconds = max_cache_age_hours * 3600
        
        for key, record in self._cache_index.items():
            if not isinstance(record, dict):
                keys_to_remove.append(key)
                continue
                
            # Check if this cache entry is very old (older than max_cache_age_hours)
            entry_time = record.get('time', 0)
            if current_time - entry_time > max_cache_age_seconds:
                keys_to_remove.append(key)
                cleaned_count += 1
        
        # Remove old entries
        for key in keys_to_remove:
            del self._cache_index[key]
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} very old cache entries (older than {max_cache_age_hours} hours)")
            # Save the cleaned cache index
            self._save_cache_index()

    def _save_cache_index(self) -> None:
        """Persist cache index to disk (best-effort)."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
            with open(self._cache_file, 'w', encoding='utf-8') as f:
                json.dump(self._cache_index, f, indent=2)
        except Exception:
            pass

    def _make_cache_key(self, input_file: Path) -> str:
        try:
            return str(input_file.resolve())
        except Exception:
            return str(input_file)

    def _get_input_signature(self, input_file: Path) -> Dict[str, Any]:
        try:
            st = input_file.stat()
            return {'size': int(st.st_size), 'mtime': float(st.st_mtime)}
        except Exception:
            return {'size': None, 'mtime': None}

    def _is_cached_success(self, input_file: Path) -> bool:
        """Return True if input was previously verified successful and outputs still exist.

        This performs minimal checks for speed: input signature matches and basic output existence.
        Cache entries can persist between executions but are cleaned up if very old (>24 hours).
        """
        if not self.use_cache:
            return False
        key = self._make_cache_key(input_file)
        rec = self._cache_index.get(key)
        if not isinstance(rec, dict):
            safe_name = self._safe_filename_for_logging(input_file.name)
            logger.debug(f"Cache miss for {safe_name}: no valid record")
            return False
        
        # Verify signature matches
        sig = self._get_input_signature(input_file)
        if not sig or rec.get('input_signature') != sig:
            safe_name = self._safe_filename_for_logging(input_file.name)
            logger.debug(f"Cache miss for {safe_name}: signature mismatch")
            return False
        # Quick output existence check
        out_type = rec.get('type')
        out_path = rec.get('output')
        if not out_type or not out_path:
            safe_name = self._safe_filename_for_logging(input_file.name)
            logger.debug(f"Cache miss for {safe_name}: missing output info")
            return False
        try:
            if out_type == 'single_gif':
                exists = Path(out_path).exists()
                if exists:
                    safe_name = self._safe_filename_for_logging(input_file.name)
                    logger.debug(f"Cache hit for {safe_name}: single GIF exists")
                else:
                    safe_name = self._safe_filename_for_logging(input_file.name)
                    logger.debug(f"Cache miss for {safe_name}: single GIF missing - {out_path}")
                return exists
            elif out_type == 'segments':
                seg_dir = Path(out_path)
                if not seg_dir.exists() or not seg_dir.is_dir():
                    safe_name = self._safe_filename_for_logging(input_file.name)
                    logger.debug(f"Cache miss for {safe_name}: segments directory missing - {out_path}")
                    return False
                # Check if at least one GIF segment exists
                gif_segments = list(seg_dir.glob("*.gif"))
                if gif_segments:
                    safe_name = self._safe_filename_for_logging(input_file.name)
                    logger.debug(f"Cache hit for {safe_name}: {len(gif_segments)} GIF segments exist")
                    return True
                else:
                    safe_name = self._safe_filename_for_logging(input_file.name)
                    logger.debug(f"Cache miss for {safe_name}: no GIF segments found in {out_path}")
                    return False
            elif out_type == 'gif_input':
                exists = Path(out_path).exists()
                if exists:
                    safe_name = self._safe_filename_for_logging(input_file.name)
                    logger.debug(f"Cache hit for {safe_name}: GIF input exists")
                else:
                    safe_name = self._safe_filename_for_logging(input_file.name)
                    logger.debug(f"Cache miss for {safe_name}: GIF input missing - {out_path}")
                return exists
            elif out_type == 'single_mp4':
                # Do not treat MP4-only outputs as a final success for caching purposes.
                # An MP4 may exist from a previous run, but GIFs might still need generation.
                if Path(out_path).exists():
                    safe_name = self._safe_filename_for_logging(input_file.name)
                    logger.debug(
                        f"Cache record has MP4 for {safe_name}, not skipping; GIFs may still be needed"
                    )
                else:
                    safe_name = self._safe_filename_for_logging(input_file.name)
                    logger.debug(f"Cache miss for {safe_name}: single MP4 missing - {out_path}")
                return False
            else:
                safe_name = self._safe_filename_for_logging(input_file.name)
                logger.debug(f"Cache miss for {safe_name}: unknown output type '{out_type}'")
                return False
        except Exception as e:
            safe_name = self._safe_filename_for_logging(input_file.name)
            logger.debug(f"Cache miss for {safe_name}: error checking output: {e}")
            return False
        return False

    def _record_success_cache(self, input_file: Path, out_type: str, output_path: Path) -> None:
        """Record a successful processing outcome in the cache."""
        try:
            key = self._make_cache_key(input_file)
            self._cache_index[key] = {
                'type': out_type,
                'output': str(output_path),
                'input_signature': self._get_input_signature(input_file),
                'time': time.time(),
            }
            self._save_cache_index()
            logger.debug(f"Cached success for {input_file.name}: {out_type} -> {output_path}")
        except Exception as e:
            logger.warning(f"Failed to cache success for {input_file.name}: {e}")

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
    
    def _safe_filename_for_logging(self, filename: str) -> str:
        """Convert filename to safe string for logging, handling Unicode characters."""
        try:
            # Try to encode as ASCII, replace problematic characters
            return filename.encode('ascii', errors='replace').decode('ascii')
        except Exception:
            # Fallback: replace any non-ASCII characters with safe alternatives
            return ''.join(c if ord(c) < 128 else '_' for c in filename)

    def _find_new_files(self, processed_files: set, skip_stability_check: bool = False) -> List[Path]:
        """Find new files, sorted by creation time (newest first)."""
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
        
        # Sort by creation time descending (Windows Date Created equivalent), fallback to mtime
        def _creation_time_desc(path: Path) -> float:
            try:
                st = path.stat()
                # On Windows, st_ctime is creation time; elsewhere it may be metadata change time
                # This still provides a reasonable ordering, with mtime as a fallback
                return float(getattr(st, 'st_ctime', st.st_mtime))
            except Exception:
                try:
                    return float(path.stat().st_mtime)
                except Exception:
                    return 0.0

        # Newest first
        sorted_files = sorted(files, key=_creation_time_desc, reverse=True)
        return sorted_files
    
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
                # Fast validation for existing single GIF
                is_valid, _ = self.file_validator.is_valid_gif_fast(str(output_path), max_size_mb=None)
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
            safe_name = self._safe_filename_for_logging(video_file.name)
            logger.info(f"Starting processing: {safe_name}")
        
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
            if self.shutdown_requested:
                return 'cancelled'
            if not mp4_file:
                return 'error'
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
                    # Cache success for single gif
                    self._record_success_cache(video_file, 'single_gif', gif_path)
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
                    print(f"    ♻️  Using existing segment GIFs: {segments_folder.name}")
                    logger.debug(f"Valid segment GIFs already exist: {segments_folder.name}")
                    
                    if invalid_segments:
                        print(f"    ⚠️  Found {len(invalid_segments)} invalid/corrupted GIFs in segments folder")
                        logger.warning(f"Found {len(invalid_segments)} invalid GIFs in segments folder: {invalid_segments}")
                    
                    # Only move MP4 to segments folder AFTER we're sure we have valid segments
                    # and we're not going to regenerate them
                    try:
                        self._ensure_mp4_in_segments(mp4_file, segments_folder)
                    except Exception as e:
                        logger.debug(f"Could not ensure MP4 in segments folder: {e}")
                    
                    # Record cache for segments-based success
                    try:
                        self._record_success_cache(video_file, 'segments', segments_folder)
                    except Exception:
                        pass

                    return True
                else:
                    print(f"    🔄 Regenerating segment GIFs (invalid/corrupted): {segments_folder.name}")
                    logger.info(f"Segment GIFs invalid, will regenerate: {segments_folder.name}")
                    # Clean up invalid segments and continue to main GIF generation
                    # DO NOT move MP4 to segments folder yet - wait until after successful generation
                    print(f"    🧹 Cleaning up invalid segments...")
                    try:
                        for gif_file in segments_folder.glob("*.gif"):
                            try:
                                gif_file.unlink()
                                logger.debug(f"Removed invalid segment: {gif_file.name}")
                            except Exception as e:
                                logger.warning(f"Failed to remove invalid segment {gif_file.name}: {e}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up invalid segments: {e}")
                    
                    # Continue to main GIF generation instead of trying to regenerate from invalid segments
                    print(f"    🎨 Continuing to main GIF generation...")
                    # Continue to generate new segment GIFs
                    # DO NOT move MP4 to segments folder yet - wait until after successful generation
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
                # Cache successful existing MP4 output
                self._record_success_cache(video_file, 'single_mp4', output_path)
                return output_path
            else:
                print(f"    🔄 Reprocessing MP4 (validation failed): {output_name}")
                logger.info(f"Existing MP4 validation failed: {error_msg}, reprocessing: {output_name}")
        
        # Check if segments folder already exists (from previous segmentation)
        segments_folder = self.output_dir / f"{video_file.stem}_segments"
        if segments_folder.exists() and segments_folder.is_dir():
            print(f"    ♻️  Found existing segments folder: {segments_folder.name}")
            logger.debug(f"Found existing segments folder: {segments_folder.name}")
            
            # Validate existing segments
            valid_segments, invalid_segments = self._validate_segment_folder_gifs(segments_folder, max_size_mb)
            
            if valid_segments and not invalid_segments:
                print(f"    ♻️  Using existing segment GIFs: {segments_folder.name}")
                logger.debug(f"Valid segment GIFs already exist: {segments_folder.name}")
                
                # Ensure comprehensive summary exists for the segments folder
                self._ensure_segments_summary_exists(segments_folder)
                
                # Record success and return
                self._record_success_cache(video_file, 'segments', segments_folder)
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
            
            if result.get('cancelled'):
                return None
            if result.get('success', False):
                # Check if segmentation was used
                if result.get('method') == 'segmentation' or 'segments' in result:
                    # Video was segmented, check for segments folder
                    if segments_folder.exists() and segments_folder.is_dir():
                        print(f"    ✅ Video segmentation successful: {segments_folder.name}")
                        logger.info(f"Video segmentation successful: {segments_folder.name}")
                        # Best-effort: create a cover image in the segments folder too
                        try:
                            # Use the first segment as thumbnail source if exists, else the original input video
                            first_seg = None
                            try:
                                mp4s = sorted(list(segments_folder.glob('*.mp4')))
                                if mp4s:
                                    first_seg = mp4s[0]
                            except Exception:
                                first_seg = None
                            thumb_src = str(first_seg or output_path)
                            cover_jpg = segments_folder / 'folder.jpg'
                            if not cover_jpg.exists():
                                FFmpegUtils.extract_thumbnail_image(
                                    input_path=thumb_src,
                                    output_image_path=str(cover_jpg),
                                    time_position_seconds=1.0,
                                    width=640
                                )
                        except Exception as e:
                            logger.debug(f"Could not create segments folder cover image: {e}")
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
                        
                        # Cache successful MP4 output
                        self._record_success_cache(video_file, 'single_mp4', output_path)
                        
                        # Log detailed file specifications
                        try:
                            specs = FFmpegUtils.get_detailed_file_specifications(str(output_path))
                            specs_log = FFmpegUtils.format_file_specifications_for_logging(specs)
                            logger.info(f"Final video file specifications - {specs_log}")
                        except Exception as e:
                            logger.warning(f"Could not log detailed video specifications: {e}")

                        # Generate a folder.jpg thumbnail for Windows Explorer folder cover
                        try:
                            thumb_path = self.output_dir / 'folder.jpg'
                            # If already present, keep existing; otherwise, create from new MP4
                            if not thumb_path.exists():
                                ok = FFmpegUtils.extract_thumbnail_image(
                                    input_path=str(output_path),
                                    output_image_path=str(thumb_path),
                                    time_position_seconds=1.0,
                                    width=640
                                )
                                if ok:
                                    logger.info(f"Created folder cover image: {thumb_path}")
                        except Exception as e:
                            logger.debug(f"Could not create folder cover image: {e}")
                        
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
                
                # Record cache for segments-based success
                try:
                    src = Path(self.output_dir / f"{base_name}.mp4") if base_name else mp4_file
                except Exception:
                    src = mp4_file
                self._record_success_cache(src, 'segments', segments_folder)
                return True
            else:
                print(f"    🔄 Regenerating segment GIFs (invalid/corrupted): {segments_folder.name}")
                logger.info(f"Segment GIFs invalid, will regenerate: {segments_folder.name}")
                # Clean up invalid segments and continue to main GIF generation
                # DO NOT move MP4 to segments folder yet - wait until after successful generation
                print(f"    🧹 Cleaning up invalid segments...")
                try:
                    for gif_file in segments_folder.glob("*.gif"):
                        try:
                            gif_file.unlink()
                            logger.debug(f"Removed invalid segment: {gif_file.name}")
                        except Exception as e:
                            logger.warning(f"Failed to remove invalid segment {gif_file.name}: {e}")
                except Exception as e:
                    logger.warning(f"Error cleaning up invalid segments: {e}")
                
                # Continue to main GIF generation instead of trying to regenerate from invalid segments
                print(f"    🎨 Continuing to main GIF generation...")
                # Continue to generate new segment GIFs
                # DO NOT move MP4 to segments folder yet - wait until after successful generation
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
                    
                    # Only move MP4 to segments folder AFTER we're sure we have valid segments
                    # and we're not going to regenerate them
                    try:
                        self._ensure_mp4_in_segments(mp4_file, segments_folder)
                    except Exception as e:
                        logger.debug(f"Could not ensure MP4 in segments folder: {e}")

                    # Recreate comprehensive summary to include the MP4 just ensured
                    try:
                        self._create_comprehensive_segments_summary(segments_folder, mp4_file.stem)
                    except Exception:
                        pass

                    return True
                else:
                    print(f"    🔄 Regenerating segment GIFs (invalid/corrupted): {segments_folder.name}")
                    logger.info(f"Segment GIFs invalid, will regenerate: {segments_folder.name}")
                    # Clean up invalid segments and continue to main GIF generation
                    # DO NOT move MP4 to segments folder yet - wait until after successful generation
                    print(f"    🧹 Cleaning up invalid segments...")
                    try:
                        for gif_file in segments_folder.glob("*.gif"):
                            try:
                                gif_file.unlink()
                                logger.debug(f"Removed invalid segment: {gif_file.name}")
                            except Exception as e:
                                logger.warning(f"Failed to remove invalid segment {gif_file.name}: {e}")
                    except Exception as e:
                        logger.warning(f"Error cleaning up invalid segments: {e}")
                    
                    # Continue to main GIF generation instead of trying to regenerate from invalid segments
                    print(f"    🎨 Continuing to main GIF generation...")
                    # Continue to generate new segment GIFs
                    # DO NOT move MP4 to segments folder yet - wait until after successful generation
        
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
                    # Record cache for single GIF success
                    self._record_success_cache(mp4_file, 'single_gif', final_gif_path)
                    return True
                else:
                    print(f"    🔄 Regenerating GIF (validation failed): {gif_name}")
                    logger.info(f"Existing GIF validation failed: {error_msg}, regenerating: {gif_name}")
            
            if self.shutdown_requested:
                return False
            
            safe_name_in = self._safe_filename_for_logging(mp4_file.name)
            safe_name_out = self._safe_filename_for_logging(gif_name)
            print(f"    🎨 Generating GIF: {safe_name_in} -> {safe_name_out}")
            logger.info(f"Generating GIF from MP4: {safe_name_in} -> {safe_name_out}")
            
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

                            # Create comprehensive summary for both MP4 and GIF segments regardless of validity
                            try:
                                # Always use sanitized base name (without '_segments')
                                try:
                                    sanitized_base = final_segments_dir.name.replace('_segments', '')
                                except Exception:
                                    sanitized_base = final_segments_dir.name
                                self._create_comprehensive_segments_summary(final_segments_dir, sanitized_base)
                            except Exception as e:
                                logger.debug(f"Could not create comprehensive summary: {e}")

                            if valid_segments and not invalid_segments:
                                print(f"    ✨ Segment GIFs generated: {final_segments_dir.name} ({len(valid_segments)} valid)")
                                if invalid_segments:
                                    print(f"    ⚠️  {len(invalid_segments)} invalid segment(s) detected")
                                logger.info(f"Segment GIFs generated for {mp4_file.name}: {final_segments_dir}")

                                # Best-effort: create a cover image (folder.jpg) in the segments folder
                                try:
                                    cover_jpg = final_segments_dir / 'folder.jpg'
                                    if not cover_jpg.exists():
                                        # Prefer MP4 source for thumbnail; fallback to first valid GIF if MP4 unavailable
                                        thumb_src = str(mp4_file)
                                        if not mp4_file.exists():
                                            try:
                                                if valid_segments:
                                                    thumb_src = str(valid_segments[0])
                                            except Exception:
                                                thumb_src = None
                                        FFmpegUtils.extract_thumbnail_image(
                                            input_path=thumb_src,
                                            output_image_path=str(cover_jpg),
                                            time_position_seconds=1.0,
                                            width=640
                                        )
                                except Exception as e:
                                    logger.debug(f"Could not create segments folder cover image: {e}")

                                # Move the source MP4 into the segments folder for user convenience
                                try:
                                    self._ensure_mp4_in_segments(mp4_file, final_segments_dir)
                                except Exception as e:
                                    logger.debug(f"Could not move MP4 into segments folder: {e}")

                                # Recreate summary AFTER ensuring MP4 is present
                                try:
                                    try:
                                        sanitized_base = final_segments_dir.name.replace('_segments', '')
                                    except Exception:
                                        sanitized_base = final_segments_dir.name
                                    self._create_comprehensive_segments_summary(final_segments_dir, sanitized_base)
                                except Exception as e:
                                    logger.debug(f"Could not create comprehensive summary: {e}")
                                # Record cache for segments-based success
                                self._record_success_cache(mp4_file, 'segments', final_segments_dir)
                                
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
                        # Record cache for single GIF success
                        self._record_success_cache(mp4_file if mp4_file.exists() else final_gif_path, 'single_gif', final_gif_path)
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
            print(f"    🔄 Starting GIF generation for {len(segment_files)} segments...")
            
            successful_gifs = 0
            skipped_segments = 0

            # Limit parallelism based on configuration and system analysis
            max_workers = self._calculate_optimal_segmentation_workers()

            logger.info(f"Starting parallel GIF generation with {max_workers} workers for {len(segment_files)} segments")
            print(f"    🔄 Processing {len(segment_files)} segments with {max_workers} concurrent workers...")

            # Worker per segment; use a fresh GifGenerator instance to avoid shared ffmpeg process state
            def _process_segment(segment_file: Path) -> Tuple[bool, Optional[str]]:
                try:
                    if self.shutdown_requested:
                        return (False, 'shutdown')
                    # Duration pre-check
                    try:
                        segment_duration = FFmpegUtils.get_video_duration(str(segment_file))
                        print(f"    ⏱️  Segment duration: {segment_duration:.1f}s")
                        if segment_duration > 30.0:
                            print(f"    ⏭️  Skipping segment {segment_file.name}: too long ({segment_duration:.1f}s > 30s)")
                            logger.info(f"Skipping segment {segment_file.name}: too long ({segment_duration:.1f}s)")
                            return (False, 'too_long')
                    except Exception as e:
                        print(f"    ⚠️  Could not get segment duration, proceeding with caution: {e}")
                        logger.warning(f"Could not get segment duration for {segment_file.name}: {e}")

                    # Sanitize the filename to avoid Unicode encoding issues
                    safe_stem = self._safe_filename_for_filesystem(segment_file.stem)
                    gif_name = safe_stem + ".gif"
                    gif_path = segments_folder / gif_name
                    # Reuse if valid
                    if gif_path.exists():
                        is_valid, _ = self.file_validator.is_valid_gif_with_enhanced_checks(
                            str(gif_path), original_path=None, max_size_mb=max_size_mb
                        )
                        if is_valid:
                            print(f"    ♻️  Using existing GIF: {gif_name}")
                            return (True, None)

                    # Fresh generator instance for thread safety
                    local_generator = GifGenerator(self.config)
                    result = local_generator.create_gif(
                        input_video=str(segment_file),
                        output_path=str(gif_path),
                        max_size_mb=max_size_mb,
                        disable_segmentation=True
                    )
                    if result.get('success', False):
                        is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                            str(gif_path), original_path=None, max_size_mb=max_size_mb
                        )
                        if is_valid:
                            size_mb = result.get('size_mb', 0)
                            print(f"    ✅ Segment GIF successful: {size_mb:.2f}MB")
                            return (True, None)
                        else:
                            print(f"    ❌ Segment GIF validation failed: {error_msg}")
                            try:
                                if gif_path.exists():
                                    gif_path.unlink()
                            except Exception:
                                pass
                            return (False, 'invalid')
                    else:
                        # Cleanup accidental nested segments
                        nested_segments = segments_folder / f"{segment_file.stem}_segments"
                        if nested_segments.exists():
                            try:
                                for item in nested_segments.iterdir():
                                    if item.is_file():
                                        item.unlink()
                                nested_segments.rmdir()
                            except Exception:
                                pass
                        return (False, result.get('error', 'Unknown error'))
                except Exception as e:
                    logger.warning(f"Exception while processing segment {segment_file.name}: {e}")
                    return (False, str(e))

            from concurrent.futures import ThreadPoolExecutor, as_completed
            completed_count = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {executor.submit(_process_segment, seg): seg for seg in segment_files}
                for fut in as_completed(future_map):
                    ok, reason = False, None
                    try:
                        ok, reason = fut.result()
                    except Exception as e:
                        logger.warning(f"Segment future failed: {e}")
                        ok = False
                    if ok:
                        successful_gifs += 1
                        completed_count += 1
                        # Show progress with actual completion count
                        print(f"    ✅ GIF {completed_count}/{len(segment_files)} completed successfully")
                    else:
                        skipped_segments += 1
            
            # Report results
            if successful_gifs > 0:
                print(f"    🎉 GIF generation completed: {successful_gifs} GIFs created successfully!")
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
        
        # Collect GIF files first
        gif_files = sorted([f for f in segments_folder.iterdir() if f.is_file() and f.suffix.lower() == '.gif'])
        # Heuristic: infer expected number of segments from filenames like *_segment_01.gif
        expected = None
        try:
            import re
            indices = []
            for f in gif_files:
                m = re.search(r"_segment_(\d+)\\.gif$", f.name)
                if m:
                    try:
                        indices.append(int(m.group(1)))
                    except Exception:
                        pass
            if indices:
                expected = max(indices)
        except Exception:
            expected = None
        if not gif_files:
            return [], []
        
        # Determine reasonable parallelism using existing worker analysis
        try:
            max_workers = max(1, int(self._calculate_optimal_segmentation_workers()))
        except Exception:
            max_workers = max(1, min(4, (os.cpu_count() or 2)))
        
        def _validate(gf: Path) -> Tuple[Path, bool, str]:
            try:
                is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                    str(gf), original_path=None, max_size_mb=max_size_mb
                )
                return gf, bool(is_valid), error_msg or ""
            except Exception as e:
                return gf, False, str(e)
        
        from concurrent.futures import ThreadPoolExecutor, as_completed
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_validate, gf): gf for gf in gif_files}
            for fut in as_completed(future_map):
                gf, ok, _ = fut.result()
                if ok:
                    valid_gifs.append(gf)
                else:
                    invalid_gifs.append(gf)
        
        # If we can infer expected count, cross-check completeness; mark missing as invalid
        if expected is not None:
            present_count = len(gif_files)
            if present_count < expected:
                try:
                    missing = expected - present_count
                    print(f"    ⚠️  Segment completeness check: expected {expected}, found {present_count} (missing {missing})")
                    logger.warning(
                        f"Segment completeness check failed for {segments_folder.name}: expected {expected}, found {present_count}"
                    )
                except Exception:
                    pass
                # Consider folder invalid if incomplete; treat all as invalid to trigger regeneration
                return [], gif_files

        return valid_gifs, invalid_gifs
    
    def _calculate_optimal_segmentation_workers(self) -> int:
        """
        Calculate optimal number of workers based on configuration and system analysis
        """
        # Check if multiprocessing is enabled
        if not self.config.get('gif_settings.multiprocessing.enabled', True):
            return 1  # Disable multiprocessing if configured
        
        # Check if we should use dynamic analysis
        use_dynamic = self.config.get('gif_settings.multiprocessing.use_dynamic_analysis', True)
        
        if use_dynamic:
            try:
                # Import hardware detector to analyze optimal workers
                from .hardware_detector import HardwareDetector
                hardware = HardwareDetector()
                worker_analysis = hardware.analyze_optimal_segmentation_workers()
                
                # Get the analysis mode from config
                analysis_mode = self.config.get('gif_settings.multiprocessing.analysis_mode', 'recommended')
                
                if analysis_mode == 'conservative':
                    dynamic_workers = worker_analysis['conservative']
                elif analysis_mode == 'maximum_safe':
                    dynamic_workers = worker_analysis['maximum_safe']
                else:  # 'recommended' or default
                    dynamic_workers = worker_analysis['recommended']
                
                # Apply configurable limits
                config_max = self.config.get('gif_settings.multiprocessing.max_concurrent_segments', 4)
                final_workers = min(dynamic_workers, config_max)
                
                logger.info(f"Dynamic analysis: {analysis_mode} mode suggests {dynamic_workers} workers, "
                           f"limited to {final_workers} by config")
                
                return final_workers
                
            except Exception as e:
                logger.warning(f"Dynamic worker analysis failed: {e}, falling back to static config")
                # Fall back to static configuration
                pass
        
        # Static configuration fallback
        return self.config.get('gif_settings.multiprocessing.max_concurrent_segments', 2)
    
    def _cleanup_temp_files(self):
        """Clean up temporary files"""
        logger.info("Cleaning up temporary files...")
        try:
            if self.temp_dir.exists():
                for temp_file in self.temp_dir.iterdir():
                    # Skip the cache directory - it should be preserved
                    if temp_file.name == 'cache':
                        continue
                        
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
                logger.debug(f"MP4 already exists in segments folder: {target.name}")
                return
            
            # If mp4 is in output dir, move it. Otherwise, copy it (e.g., original input still in input/).
            try:
                if mp4_file.parent.resolve() == self.output_dir.resolve() and mp4_file.exists():
                    shutil.move(str(mp4_file), str(target))
                    logger.info(f"Moved MP4 to segments folder: {mp4_file.name} -> {target}")
                else:
                    shutil.copy2(str(mp4_file), str(target))
                    logger.info(f"Copied MP4 to segments folder: {mp4_file.name} -> {target}")
            except Exception as e:
                logger.warning(f"Failed to move/copy MP4 to segments folder: {e}")
                # As a last resort, try copying
                try:
                    shutil.copy2(str(mp4_file), str(target))
                    logger.info(f"Fallback copy MP4 to segments folder: {mp4_file.name} -> {target}")
                except Exception as copy_e:
                    logger.error(f"Failed to copy MP4 to segments folder: {copy_e}")
        except Exception as e:
            logger.error(f"Error ensuring MP4 in segments folder: {e}")
    
    def _regenerate_invalid_segments(self, segments_folder: Path, max_size_mb: float) -> bool:
        """Regenerate invalid segments from the MP4 file in the segments folder"""
        try:
            # Find the MP4 file in the segments folder
            mp4_files = list(segments_folder.glob("*.mp4"))
            if not mp4_files:
                logger.warning(f"No MP4 file found in segments folder for regeneration: {segments_folder}")
                return False
            
            mp4_file = mp4_files[0]  # Use the first MP4 found
            logger.info(f"Regenerating segments from MP4: {mp4_file.name}")
            
            # Use the existing GIF generation logic to recreate segments
            gif_success = self._generate_and_optimize_gif(mp4_file, max_size_mb)
            
            if gif_success:
                logger.info(f"Successfully regenerated segments for: {mp4_file.name}")
                return True
            else:
                logger.error(f"Failed to regenerate segments for: {mp4_file.name}")
                return False
                
        except Exception as e:
            logger.error(f"Error regenerating invalid segments: {e}")
            return False
    
    def _attempt_gif_repair(self, gif_file: Path, max_size_mb: float) -> bool:
        """Attempt to repair a corrupted GIF file using multiple strategies"""
        try:
            print(f"    🔧 Attempting to repair corrupted GIF: {gif_file.name}")
            logger.info(f"Attempting to repair corrupted GIF: {gif_file.name}")
            
            # Strategy 1: Try to read and rewrite using PIL
            try:
                from PIL import Image
                repaired_path = self.temp_dir / f"{gif_file.stem}.repaired.gif"
                
                with Image.open(gif_file) as img:
                    # Save as new GIF to potentially fix corruption
                    img.save(repaired_path, 'GIF', save_all=True, loop=0)
                
                # Validate the repaired file
                is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                    str(repaired_path),
                    original_path=str(gif_file),
                    max_size_mb=max_size_mb
                )
                
                if is_valid:
                    current_size = self.file_validator.get_file_size_mb(str(repaired_path))
                    print(f"    ✅ Successfully repaired corrupted GIF ({current_size:.2f}MB)")
                    logger.info(f"Successfully repaired corrupted GIF {gif_file.name}")
                    
                    # Move repaired file to output
                    output_path = self.output_dir / gif_file.name
                    if output_path.exists():
                        output_path.unlink()
                    shutil.move(str(repaired_path), str(output_path))
                    print(f"    📁 Saved repaired GIF to output: {output_path.name}")
                    logger.info(f"Saved repaired GIF to output: {output_path}")
                    
                    return True
                else:
                    logger.warning(f"PIL repair failed: {error_msg}")
                    try:
                        repaired_path.unlink()
                    except Exception:
                        pass
                        
            except Exception as repair_error:
                logger.warning(f"PIL repair failed for {gif_file.name}: {repair_error}")
            
            # Strategy 2: Try FFmpeg repair
            try:
                repaired_path = self.temp_dir / f"{gif_file.stem}.ffmpeg_repaired.gif"
                
                cmd = [
                    'ffmpeg', '-y',
                    '-i', str(gif_file),
                    '-vf', 'fps=15,scale=iw:ih:flags=lanczos',
                    '-loop', '0',
                    str(repaired_path)
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and repaired_path.exists():
                    # Validate the repaired file
                    is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                        str(repaired_path),
                        original_path=str(gif_file),
                        max_size_mb=max_size_mb
                    )
                    
                    if is_valid:
                        current_size = self.file_validator.get_file_size_mb(str(repaired_path))
                        print(f"    ✅ Successfully repaired corrupted GIF using FFmpeg ({current_size:.2f}MB)")
                        logger.info(f"Successfully repaired corrupted GIF {gif_file.name} using FFmpeg")
                        
                        # Move repaired file to output
                        output_path = self.output_dir / gif_file.name
                        if output_path.exists():
                            output_path.unlink()
                        shutil.move(str(repaired_path), str(output_path))
                        print(f"    📁 Saved repaired GIF to output: {output_path.name}")
                        logger.info(f"Saved repaired GIF to output: {output_path}")
                        
                        return True
                    else:
                        logger.warning(f"FFmpeg repair failed: {error_msg}")
                        try:
                            repaired_path.unlink()
                        except Exception:
                            pass
                else:
                    logger.warning(f"FFmpeg repair command failed: {result.stderr}")
                    try:
                        repaired_path.unlink()
                    except Exception:
                        pass
                        
            except Exception as repair_error:
                logger.warning(f"FFmpeg repair failed for {gif_file.name}: {repair_error}")
            
            # All repair strategies failed
            print(f"    ❌ All repair strategies failed for {gif_file.name}")
            logger.error(f"All repair strategies failed for {gif_file.name}")
            return False
            
        except Exception as e:
            logger.error(f"Error in GIF repair process: {e}")
            return False
    
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

            # Check if an output GIF already exists and validate it
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
                    # Cache success for gif input
                    self._record_success_cache(gif_file, 'gif_input', existing_output)
                    return 'success'
                else:
                    print(f"    ⚠️  Existing output invalid ({err_out}), will remove and regenerate")
                    logger.info(f"Existing output invalid, removing and regenerating: {err_out}")
                    # Remove the invalid existing output
                    try:
                        existing_output.unlink()
                        print(f"    🗑️  Removed invalid existing output: {existing_output.name}")
                        logger.info(f"Removed invalid existing output: {existing_output.name}")
                    except Exception as e:
                        logger.warning(f"Failed to remove invalid existing output: {e}")
            
            # Validate the input GIF file for integrity (not size)
            is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                str(gif_file),
                original_path=str(gif_file),
                max_size_mb=None  # Don't check size during validation
            )
            
            if not is_valid:
                print(f"    ❌ Input GIF file invalid: {error_msg}")
                logger.warning(f"Input GIF file {gif_file.name}: {error_msg}")
                
                # Try to regenerate from existing output if it was removed above
                if not existing_output.exists():
                    print(f"    🔄 Attempting to regenerate from input despite validation failure...")
                    logger.info(f"Attempting to regenerate from invalid input GIF: {gif_file.name}")
                    # Continue with optimization attempt even if validation fails
                    # This allows the optimizer to potentially fix corrupted files
                else:
                    # Try to repair the corrupted GIF
                    print(f"    🔧 Attempting to repair corrupted input GIF...")
                    repair_success = self._attempt_gif_repair(gif_file, max_size_mb)
                    
                    if repair_success:
                        print(f"    ✅ Successfully repaired corrupted GIF: {gif_file.name}")
                        logger.info(f"Successfully repaired corrupted GIF: {gif_file.name}")
                        return 'success'
                    else:
                        print(f"    ❌ Failed to repair corrupted GIF: {gif_file.name}")
                        logger.error(f"Failed to repair corrupted GIF: {gif_file.name}")
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
                
                # Try to copy the file, but handle corruption gracefully
                try:
                    shutil.copy2(gif_file, working_path)
                except Exception as copy_error:
                    logger.warning(f"Failed to copy corrupted GIF {gif_file.name}: {copy_error}")
                    # Try to read and rewrite the file to potentially fix corruption
                    try:
                        from PIL import Image
                        with Image.open(gif_file) as img:
                            # Save as new GIF to potentially fix corruption
                            img.save(working_path, 'GIF', save_all=True, loop=0)
                        logger.info(f"Successfully repaired corrupted GIF {gif_file.name} using PIL")
                    except Exception as repair_error:
                        logger.error(f"Failed to repair corrupted GIF {gif_file.name}: {repair_error}")
                        return 'error'
                        
            except Exception as e:
                logger.error(f"Failed to prepare working copy for {gif_file.name}: {e}")
                return 'error'

            # Use the advanced GIF optimizer with quality targeting on the working copy
            from .gif_optimizer_advanced import AdvancedGifOptimizer
            optimizer = AdvancedGifOptimizer(self.config)

            result = optimizer.optimize_gif(
                gif_path=str(working_path),
                max_size_mb=max_size_mb
            )

            if result and working_path.exists():
                optimized_size = self.file_validator.get_file_size_mb(str(working_path))
                
                # Validate size before copying to output
                if optimized_size > max_size_mb:
                    print(f"    ❌ Optimized GIF still exceeds size limit: {optimized_size:.2f}MB > {max_size_mb:.2f}MB")
                    logger.warning(f"Optimized GIF exceeds size limit: {gif_file.name} ({optimized_size:.2f}MB > {max_size_mb:.2f}MB)")
                    return 'error'
                
                print(f"    ✅ GIF optimized successfully: {optimized_size:.2f}MB")
                logger.info(f"GIF optimized successfully (copy): {gif_file.name} -> {optimized_size:.2f}MB")

                # Copy optimized working copy to output directory
                try:
                    if output_path.exists():
                        output_path.unlink()  # Remove existing file first
                    shutil.copy2(str(working_path), str(output_path))
                    
                    # Final validation after copy
                    final_size = self.file_validator.get_file_size_mb(str(output_path))
                    if final_size > max_size_mb:
                        print(f"    ❌ Final output exceeds size limit: {final_size:.2f}MB > {max_size_mb:.2f}MB")
                        logger.error(f"Final output exceeds size limit: {output_path.name} ({final_size:.2f}MB)")
                        output_path.unlink()  # Remove the invalid file
                        return 'error'
                    
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
                # If optimization failed, try to handle the corrupted input file
                print(f"    ⚠️  Optimization failed, attempting fallback strategies...")
                logger.info(f"Optimization failed for {gif_file.name}, attempting fallback strategies")
                
                # Strategy 1: Try to repair the corrupted input file
                try:
                    from PIL import Image
                    repaired_path = self.temp_dir / f"{gif_file.stem}.repaired.gif"
                    
                    with Image.open(gif_file) as img:
                        # Save as new GIF to potentially fix corruption
                        img.save(repaired_path, 'GIF', save_all=True, loop=0)
                    
                    # Validate the repaired file
                    is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                        str(repaired_path),
                        original_path=str(gif_file),
                        max_size_mb=max_size_mb
                    )
                    
                    if is_valid:
                        current_size = self.file_validator.get_file_size_mb(str(repaired_path))
                        print(f"    ✅ Successfully repaired corrupted GIF ({current_size:.2f}MB)")
                        logger.info(f"Successfully repaired corrupted GIF {gif_file.name}")
                        
                        # Copy repaired file to output
                        if output_path.exists():
                            output_path.unlink()
                        shutil.copy2(repaired_path, output_path)
                        print(f"    📁 Saved repaired GIF to output: {output_path.name}")
                        logger.info(f"Saved repaired GIF to output: {output_path}")
                        
                        # Clean up
                        try:
                            repaired_path.unlink()
                        except Exception:
                            pass
                        
                        return 'success'
                    else:
                        logger.warning(f"Repaired GIF still invalid: {error_msg}")
                        try:
                            repaired_path.unlink()
                        except Exception:
                            pass
                        
                except Exception as repair_error:
                    logger.warning(f"Failed to repair corrupted GIF {gif_file.name}: {repair_error}")
                
                # Strategy 2: Check if original is valid and within size
                is_valid, error_msg = self.file_validator.is_valid_gif_with_enhanced_checks(
                    str(gif_file),
                    original_path=str(gif_file),
                    max_size_mb=max_size_mb
                )
                if is_valid:
                    current_size = self.file_validator.get_file_size_mb(str(gif_file))
                    print(f"    ⚠️  Optimization failed but original is valid; keeping original ({current_size:.2f}MB)")
                    logger.info(f"Keeping original GIF for {gif_file.name}; optimization failed but original is valid under limit")
                    if not output_path.exists():
                        shutil.copy2(gif_file, output_path)
                        print(f"    📁 Copied to output directory: {output_path.name}")
                        logger.info(f"Copied original GIF to output: {output_path}")
                    else:
                        print(f"    📁 Already exists in output directory: {output_path.name}")
                    return 'success'
                
                # Strategy 3: All attempts failed
                print(f"    ❌ All optimization and repair strategies failed for {gif_file.name}")
                logger.error(f"All optimization and repair strategies failed for {gif_file.name}")
                return 'error'
                
        except Exception as e:
            print(f"    ❌ Error optimizing GIF {gif_file.name}: {e}")
            logger.error(f"Error optimizing GIF {gif_file.name}: {e}")
            return 'error' 

    def _segment_input_gif(self, gif_file: Path, max_size_mb: float, preferred_segments: Optional[int] = None) -> bool:
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
            size_threshold_multiplier = float(seg_cfg.get('size_threshold_multiplier', 2.5))

            # Heuristic: segment if over target or duration is long, with optional single-file preference
            # If user prefers 1 segment, attempt single-file path first by setting should_segment False here
            if preferred_segments is None and getattr(self, 'preferred_segments', None) is not None:
                preferred_segments = int(self.preferred_segments)
            if preferred_segments is not None:
                # Force segmentation attempt according to preference
                should_segment = True
            elif getattr(self, 'prefer_single_segment', False):
                # Legacy single-file preference path
                should_segment = False
            else:
                should_segment = (
                    (max_size_mb is not None and original_size_mb > max_size_mb)
                    or (duration > max_duration_cfg)
                    or (duration >= max_seg * 2)
                )

            if not should_segment:
                print("    ℹ️  Segmentation not needed based on size/duration heuristics")
                return False

            # Prefer single file before splitting: try optimizing whole GIF first
            try:
                prefer_single_first = bool(self.config.get('gif_settings.segmentation.prefer_single_file_first', True))
            except Exception:
                prefer_single_first = True
            if prefer_single_first and (preferred_segments is None or int(preferred_segments) == 1):
                print("    🧪 Trying single-file optimization before segmentation...")
                logger.info("Attempting single-file optimization before segmentation")
                single_attempt = self._optimize_gif_file(gif_file, max_size_mb)
                if single_attempt == 'success':
                    print("    ✅ Single-file optimization succeeded; skipping segmentation")
                    logger.info("Single-file optimization succeeded; segmentation skipped")
                    return True
                else:
                    print("    ↪️  Single-file attempt did not meet target; proceeding with segmentation")
                    logger.info("Single-file attempt failed to meet target; proceeding with segmentation")

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

            # Size-aware segment count based purely on size ratio
            # Ensure at least ceil(original_size / target_size) segments when oversized
            size_segments = 1
            if max_size_mb and max_size_mb > 0:
                try:
                    if original_size_mb > max_size_mb:
                        size_segments = max(1, int(math.ceil(original_size_mb / max_size_mb)))
                except Exception:
                    size_segments = 1

            # Duration-based segment count
            if preferred_segments is not None:
                num_segments = max(1, int(preferred_segments))
            else:
                if duration and duration > 0:
                    dur_segments = max(1, int(math.ceil(duration / segment_duration)))
                    num_segments = max(size_segments, dur_segments)
                else:
                    # If duration unknown, ensure at least two or size-based count
                    num_segments = max(2, size_segments)

            # Recompute per-segment duration to evenly split total duration when known
            if duration and duration > 0 and num_segments > 0:
                segment_duration = max(min_seg, min(max_seg, duration / num_segments))

            # Prepare output directory path per asset (defer creation of segments dir until first success)
            output_dir = self.output_dir
            base_name = gif_file.stem
            # Decide if we truly need multiple segments. If forced preference = 1 or prefer_single_segment is set,
            # write a single GIF directly to output. Otherwise, use a segments folder when num_segments > 1.
            if preferred_segments is not None:
                use_segments_folder = num_segments > 1 and preferred_segments != 1
            else:
                use_segments_folder = (num_segments > 1) and (not getattr(self, 'prefer_single_segment', False))
            segments_dir = output_dir / f"{base_name}_segments"
            single_output_path = output_dir / f"{base_name}.gif"
            segments_dir_created = False

            # Process each segment (parallelized)
            successful = 0
            total_size_mb = 0.0

            # Read target fps from config; fallback to 20
            target_fps = int(self.config.get('gif_settings.fps', 20) or 20)

            # Limit parallelism based on configuration and system analysis
            max_workers = self._calculate_optimal_segmentation_workers()

            logger.info(f"Starting parallel GIF segmentation with {max_workers} workers for {num_segments} segments")
            print(f"    🔄 Processing {num_segments} segments with {max_workers} concurrent workers...")

            # Pre-create segments directory when using multi-segment mode to avoid race conditions
            if use_segments_folder:
                try:
                    segments_dir.mkdir(parents=True, exist_ok=True)
                except Exception:
                    pass
                segments_dir_created = True

            from .gif_optimizer_advanced import AdvancedGifOptimizer
            optimizer = AdvancedGifOptimizer(self.config, shutdown_checker=lambda: self.shutdown_requested)

            def _process_segment(index: int) -> Optional[Tuple[int, Path, float]]:
                try:
                    if self.shutdown_requested:
                        return None
                    seg_start = index * segment_duration
                    if duration > 0:
                        seg_end = min(duration, seg_start + segment_duration)
                        seg_len = max(0.0, seg_end - seg_start)
                    else:
                        seg_len = segment_duration
                    if seg_len <= 0.1:
                        return None
                    temp_seg = self.temp_dir / f"{base_name}.seg{index+1:02d}.gif"
                    try:
                        if temp_seg.exists():
                            temp_seg.unlink()
                    except Exception:
                        pass
                    # Downscale segments to configured max width while preserving AR to keep sizes under target
                    try:
                        max_width = int(self.config.get('gif_settings.width', 360) or 360)
                        # Get height setting - if -1, we'll calculate it to preserve aspect ratio
                        height_setting = self.config.get('gif_settings.height', -1)
                    except Exception:
                        max_width = 360
                        height_setting = -1
                    
                    # Get video info to calculate proper scaling that preserves aspect ratio
                    try:
                        from .ffmpeg_utils import FFmpegUtils
                        video_info = FFmpegUtils.get_video_info(str(gif_file))
                        if video_info and 'width' in video_info and 'height' in video_info:
                            original_width = video_info['width']
                            original_height = video_info['height']
                            aspect_ratio = original_width / original_height
                            
                            # Calculate height that maintains aspect ratio
                            calculated_height = int(max_width / aspect_ratio)
                            
                            # Ensure height is even (required for some codecs)
                            if calculated_height % 2 != 0:
                                calculated_height = calculated_height + 1
                            
                            # Use calculated dimensions to preserve aspect ratio
                            scale_filter = f"scale={max_width}:{calculated_height}:flags=lanczos"
                            logger.info(f"GIF segment scaling: original {original_width}x{original_height} -> {max_width}x{calculated_height} (AR: {aspect_ratio:.3f})")
                        else:
                            # Fallback to aspect ratio preservation
                            scale_filter = f"scale={max_width}:-2:flags=lanczos"
                            logger.info(f"GIF segment scaling: using fallback AR preservation {max_width}:-2")
                    except Exception as e:
                        logger.warning(f"Could not calculate proper scaling, using fallback: {e}")
                        # Fallback to aspect ratio preservation
                        scale_filter = f"scale={max_width}:-2:flags=lanczos"
                    
                    vf = (
                        f"mpdecimate=hi=512:lo=256:frac=0.3,"
                        f"fps={target_fps},"
                        f"{scale_filter}"
                    )
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
                    # Run with shutdown-aware subprocess to allow fast termination
                    result = optimizer._run_subprocess_with_shutdown_check(cmd, timeout=180)
                    if result.returncode != 0 or not temp_seg.exists():
                        err_excerpt = getattr(result, 'stderr', '')
                        logger.warning(f"Failed to create segment {index+1}: {err_excerpt[:200] if err_excerpt else 'unknown error'}")
                        return None
                    # Optimize with quality target for stronger convergence
                    opt_ok = optimizer.optimize_gif(
                        gif_path=str(temp_seg),
                        max_size_mb=max_size_mb
                    )
                    if not opt_ok:
                        try:
                            temp_seg.unlink()
                        except Exception:
                            pass
                        return None
                    # Validate and move
                    final_seg = (segments_dir / f"{base_name}_segment_{index+1:02d}.gif") if use_segments_folder else single_output_path
                    is_valid, _ = self.file_validator.is_valid_gif_with_enhanced_checks(str(temp_seg), original_path=None, max_size_mb=max_size_mb)
                    if not is_valid:
                        try:
                            temp_seg.unlink()
                        except Exception:
                            pass
                        return None
                    try:
                        if not use_segments_folder and final_seg.exists():
                            final_seg.unlink()
                        shutil.move(str(temp_seg), str(final_seg))
                    except Exception:
                        try:
                            shutil.copy2(str(temp_seg), str(final_seg))
                            temp_seg.unlink(missing_ok=True)  # type: ignore[arg-type]
                        except Exception:
                            return None
                    sz = self.file_validator.get_file_size_mb(str(final_seg))
                    logger.info(f"Created GIF segment {index+1}/{num_segments}: {final_seg} ({sz:.2f}MB)")
                    return (index, final_seg, sz)
                except Exception as e:
                    logger.warning(f"Exception while processing GIF segment {index+1}: {e}")
                    return None

            indices = list(range(num_segments))
            results: List[Tuple[int, Path, float]] = []
            completed_count = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_process_segment, idx) for idx in indices]
                pending = set(futures)
                try:
                    while pending:
                        # Check for shutdown and cancel promptly
                        if self.shutdown_requested:
                            for f in list(pending):
                                f.cancel()
                            try:
                                executor.shutdown(wait=False, cancel_futures=True)
                            except TypeError:
                                executor.shutdown(wait=False)
                            break

                        done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                        for fut in done:
                            res = None
                            try:
                                res = fut.result()
                            except Exception as e:
                                logger.warning(f"Segment future failed: {e}")
                            if res:
                                results.append(res)
                                completed_count += 1
                                # Show progress with actual completion count
                                print(f"    ✅ GIF segment {completed_count}/{num_segments} completed: {res[1].name} ({res[2]:.2f}MB)")
                finally:
                    if self.shutdown_requested:
                        try:
                            executor.shutdown(wait=False, cancel_futures=True)
                        except TypeError:
                            executor.shutdown(wait=False)

            # Aggregate
            successful = len(results)
            total_size_mb = sum(r[2] for r in results)

            if successful > 0:
                if use_segments_folder:
                    # If exactly one segment succeeded, prefer to present it as a single GIF
                    if successful == 1:
                        try:
                            seg_files = list(segments_dir.glob(f"{base_name}_segment_*.gif"))
                            if seg_files:
                                src = seg_files[0]
                                
                                # Validate size before moving to output
                                src_size = self.file_validator.get_file_size_mb(str(src))
                                if src_size > max_size_mb:
                                    print(f"    ❌ Single segment exceeds size limit: {src_size:.2f}MB > {max_size_mb:.2f}MB")
                                    logger.error(f"Single segment exceeds size limit: {src.name} ({src_size:.2f}MB)")
                                    # Clean up the oversized segment
                                    try:
                                        src.unlink()
                                    except Exception:
                                        pass
                                    return False
                                
                                # Replace existing file if present
                                if single_output_path.exists():
                                    try:
                                        single_output_path.unlink()
                                    except Exception:
                                        pass
                                shutil.move(str(src), str(single_output_path))
                                
                                # Final validation after move
                                final_size = self.file_validator.get_file_size_mb(str(single_output_path))
                                if final_size > max_size_mb:
                                    print(f"    ❌ Final single output exceeds size limit: {final_size:.2f}MB > {max_size_mb:.2f}MB")
                                    logger.error(f"Final single output exceeds size limit: {single_output_path.name} ({final_size:.2f}MB)")
                                    single_output_path.unlink()  # Remove the invalid file
                                    return False
                                # Attempt to clean up empty segments folder
                                try:
                                    # First, remove any remaining files
                                    remaining_files = list(segments_dir.iterdir())
                                    for extra in remaining_files:
                                        try:
                                            if extra.is_file():
                                                extra.unlink()
                                                logger.debug(f"Removed remaining file: {extra}")
                                            elif extra.is_dir():
                                                # Remove subdirectories recursively
                                                import shutil
                                                shutil.rmtree(extra)
                                                logger.debug(f"Removed remaining directory: {extra}")
                                        except Exception as e:
                                            logger.warning(f"Failed to remove {extra}: {e}")
                                    
                                    # Now attempt to remove the segments folder
                                    if segments_dir.exists():
                                        segments_dir.rmdir()
                                        logger.info(f"Successfully cleaned up segments folder: {segments_dir}")
                                except Exception as e:
                                    logger.warning(f"Failed to clean up segments folder {segments_dir}: {e}")
                                    # List what's preventing cleanup
                                    try:
                                        if segments_dir.exists():
                                            remaining = list(segments_dir.iterdir())
                                            logger.warning(f"Segments folder not empty, remaining items: {[str(item) for item in remaining]}")
                                    except Exception:
                                        pass
                                print(f"    📁 Saved optimized GIF to output: {single_output_path.name} ({total_size_mb:.2f}MB)")
                                logger.info(f"Single-segment GIF moved to output: {single_output_path} ({total_size_mb:.2f}MB)")
                                # Record cache for single GIF success (source = base MP4 if present)
                                try:
                                    src_input = self.output_dir / f"{base_name}.mp4"
                                    cache_input = src_input if src_input.exists() else segments_folder
                                    self._record_success_cache(cache_input, 'single_gif', single_output_path)
                                except Exception:
                                    pass
                                return True
                        except Exception:
                            # If relocation fails, fall back to reporting segments folder
                            pass
                    print(f"    📂 Segmented GIFs saved to: {segments_dir.name} ({successful} segment(s), {total_size_mb:.2f}MB total)")
                    logger.info(f"GIF segmentation complete: {successful} segments at {segments_dir}")
                    # Record cache for segments-based success (source = base MP4 if present)
                    try:
                        src_input = self.output_dir / f"{base_name}.mp4"
                        cache_input = src_input if src_input.exists() else segments_folder
                        self._record_success_cache(cache_input, 'segments', segments_dir)
                    except Exception:
                        pass
                else:
                    # Single segment result saved directly as a single GIF
                    print(f"    📁 Saved optimized GIF to output: {single_output_path.name} ({total_size_mb:.2f}MB)")
                    logger.info(f"Single-segment GIF saved to output: {single_output_path} ({total_size_mb:.2f}MB)")
                    # Record cache for single GIF success (source = base MP4 if present)
                    try:
                        src_input = self.output_dir / f"{base_name}.mp4"
                        cache_input = src_input if src_input.exists() else segments_folder
                        self._record_success_cache(cache_input, 'single_gif', single_output_path)
                    except Exception:
                        pass
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
                # Use fast validator to avoid 7-15s cost per file; we only need a quick accept here
                is_valid_out, err_out = self.file_validator.is_valid_gif_fast(
                    str(existing_output), max_size_mb=max_size_mb
                )
                if is_valid_out:
                    print(f"    ♻️  Valid optimized GIF already exists in output: {existing_output.name}")
                    logger.info(f"Skipping optimization; existing optimized GIF present: {existing_output}")
                    # Cache success for gif input
                    self._record_success_cache(gif_file, 'gif_input', existing_output)
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

    def clear_cache(self) -> None:
        """Clear the entire cache index."""
        self._cache_index.clear()
        try:
            if self._cache_file.exists():
                self._cache_file.unlink()
            logger.info("Cache cleared successfully")
        except Exception as e:
            logger.warning(f"Failed to clear cache file: {e}")

    def validate_and_clean_cache(self) -> Dict[str, int]:
        """Validate all cache entries and remove invalid ones (missing outputs, etc.)."""
        if not self._cache_index:
            return {'total': 0, 'valid': 0, 'invalid': 0, 'cleaned': 0}
        
        total_entries = len(self._cache_index)
        valid_entries = 0
        invalid_entries = 0
        keys_to_remove = []
        
        for key, record in self._cache_index.items():
            if not isinstance(record, dict):
                keys_to_remove.append(key)
                invalid_entries += 1
                continue
            
            # Check if output still exists
            out_type = record.get('type')
            out_path = record.get('output')
            
            if not out_type or not out_path:
                keys_to_remove.append(key)
                invalid_entries += 1
                continue
            
            # Validate output existence based on type
            is_valid = False
            try:
                if out_type == 'single_gif':
                    is_valid = Path(out_path).exists()
                elif out_type == 'segments':
                    seg_dir = Path(out_path)
                    is_valid = seg_dir.exists() and seg_dir.is_dir() and list(seg_dir.glob("*.gif"))
                elif out_type == 'gif_input':
                    is_valid = Path(out_path).exists()
                elif out_type == 'single_mp4':
                    is_valid = Path(out_path).exists()
                else:
                    is_valid = False
            except Exception:
                is_valid = False
            
            if is_valid:
                valid_entries += 1
            else:
                keys_to_remove.append(key)
                invalid_entries += 1
        
        # Remove invalid entries
        for key in keys_to_remove:
            del self._cache_index[key]
        
        if keys_to_remove:
            logger.info(f"Cleaned up {len(keys_to_remove)} invalid cache entries")
            self._save_cache_index()
        
        return {
            'total': total_entries,
            'valid': valid_entries,
            'invalid': invalid_entries,
            'cleaned': len(keys_to_remove)
        }

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        if not self._cache_index:
            return {
                'total_entries': 0, 
                'current_session_entries': 0, 
                'old_entries': 0,
                'session_start_time': self._session_start_time,
                'cache_age_info': 'No cache entries'
            }
        
        current_time = time.time()
        current_session_entries = 0
        old_entries = 0
        oldest_entry_age = float('inf')
        newest_entry_age = 0
        
        for record in self._cache_index.values():
            if isinstance(record, dict):
                entry_time = record.get('time', 0)
                entry_age = current_time - entry_time
                
                if entry_age < 3600:  # Less than 1 hour
                    current_session_entries += 1
                else:
                    old_entries += 1
                
                oldest_entry_age = min(oldest_entry_age, entry_age)
                newest_entry_age = max(newest_entry_age, entry_age)
        
        # Format age information
        if oldest_entry_age != float('inf'):
            oldest_hours = oldest_entry_age / 3600
            newest_hours = newest_entry_age / 3600
            cache_age_info = f"Oldest: {oldest_hours:.1f}h, Newest: {newest_hours:.1f}h"
        else:
            cache_age_info = 'No valid entries'
        
        return {
            'total_entries': len(self._cache_index),
            'current_session_entries': current_session_entries,
            'old_entries': old_entries,
            'session_start_time': self._session_start_time,
            'cache_age_info': cache_age_info
        }

    def _print_cache_stats(self) -> None:
        """Print cache statistics for monitoring."""
        if not self.use_cache:
            return
        
        stats = self.get_cache_stats()
        if stats['total_entries'] > 0:
            print(f"\n📊 Cache Stats: {stats['current_session_entries']} current, {stats['old_entries']} old entries")
            logger.info(f"Cache stats: {stats['current_session_entries']} current, {stats['old_entries']} old entries")

    def _print_workflow_summary(self, processing_stats: Dict[str, int], start_time: float) -> None:
        """Print workflow summary with cache information."""
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"🎬 Workflow Complete!")
        print(f"⏱️  Total time: {elapsed:.1f} seconds")
        print(f"📁 Files processed: {processing_stats.get('processed', 0)}")
        print(f"✅ Successful: {processing_stats.get('successful', 0)}")
        print(f"❌ Failed: {processing_stats.get('failed', 0)}")
        print(f"⚡ Skipped (cached): {processing_stats.get('skipped', 0)}")
        print(f"🔄 Retries: {processing_stats.get('retries', 0)}")
        
        # Print cache statistics
        self._print_cache_stats()
        
        print(f"{'='*60}")
        logger.info(f"Workflow completed in {elapsed:.1f}s. "
                   f"Processed: {processing_stats.get('processed', 0)}, "
                   f"Success: {processing_stats.get('successful', 0)}, "
                   f"Failed: {processing_stats.get('failed', 0)}, "
                   f"Skipped: {processing_stats.get('skipped', 0)}")

    def _create_comprehensive_segments_summary(self, segments_folder: Path, base_name: str) -> None:
        """
        Create a comprehensive summary file for both MP4 and GIF segments in a segments folder.
        The summary will be prefixed with '~' to ensure it appears last in folder listings.
        Uses a temporary file and atomic replace to avoid Windows file locking issues.
        """
        try:
            import time
            from .ffmpeg_utils import FFmpegUtils
            
            # Find all MP4 and GIF files in the segments folder
            mp4_files = list(segments_folder.glob("*.mp4"))
            gif_files = list(segments_folder.glob("*.gif"))
            
            if not mp4_files and not gif_files:
                logger.info(f"No MP4 or GIF files found in segments folder: {segments_folder}")
                return
            
            # Create summary filename with '~' prefix to ensure it appears last
            summary_file = segments_folder / f"~{base_name}_comprehensive_summary.txt"
            temp_file = segments_folder / f"~{base_name}_comprehensive_summary.txt.tmp"

            # Best-effort clear attributes on existing file
            try:
                if summary_file.exists():
                    subprocess.run(['attrib', '-R', '-S', '-H', str(summary_file)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            except Exception:
                pass
            # Remove stale temp file if present
            try:
                if temp_file.exists():
                    temp_file.unlink()
            except Exception:
                pass

            with open(temp_file, 'w', encoding='utf-8') as f:
                f.write("Comprehensive Segments Summary\n")
                f.write("=============================\n\n")
                f.write(f"Base Name: {base_name}\n")
                f.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total MP4 Segments: {len(mp4_files)}\n")
                f.write(f"Total GIF Segments: {len(gif_files)}\n")
                f.write(f"Total Files: {len(mp4_files) + len(gif_files)}\n\n")

                # MP4 details
                if mp4_files:
                    f.write("MP4 Segments Details:\n")
                    f.write("--------------------\n")
                    for idx, mp4_file in enumerate(mp4_files, 1):
                        try:
                            info = FFmpegUtils.get_video_info(str(mp4_file))
                            size_mb = mp4_file.stat().st_size / (1024 * 1024)
                            f.write(f"MP4 Segment {idx:03d}: {mp4_file.name}\n")
                            f.write(f"  Duration: {info.get('duration', 0.0):.2f}s\n")
                            f.write(f"  FPS: {info.get('fps', 0.0):.2f}\n")
                            f.write(f"  Frame Count: {info.get('frame_count', 0)}\n")
                            f.write(f"  Resolution: {info.get('width', 0)}x{info.get('height', 0)}\n")
                            f.write(f"  Codec: {info.get('codec', 'unknown')}\n")
                            f.write(f"  Bitrate: {info.get('bitrate', 0)} kbps\n")
                            f.write(f"  Size: {size_mb:.2f}MB\n\n")
                        except Exception as e:
                            f.write(f"MP4 Segment {idx:03d}: {mp4_file.name}\n")
                            f.write(f"  Error reading info: {e}\n\n")

                    f.write("MP4 Summary:\n")
                    f.write(f"  Total MP4 Segments: {len(mp4_files)}\n")
                    
                    # Calculate totals for MP4 summary
                    total_mp4_size = 0.0
                    total_mp4_duration = 0.0
                    total_mp4_frames = 0
                    
                    for mp4_file in mp4_files:
                        try:
                            info = FFmpegUtils.get_video_info(str(mp4_file))
                            total_mp4_size += mp4_file.stat().st_size / (1024 * 1024)
                            total_mp4_duration += float(info.get('duration', 0.0) or 0.0)
                            total_mp4_frames += int(info.get('frame_count', 0) or 0)
                        except Exception:
                            pass
                    
                    f.write(f"  Total Size: {total_mp4_size:.2f}MB\n")
                    f.write(f"  Total Duration: {total_mp4_duration:.2f}s\n")
                    f.write(f"  Total Frames: {total_mp4_frames}\n")
                    if len(mp4_files) > 0:
                        f.write(f"  Average Size: {total_mp4_size/len(mp4_files):.2f}MB\n")
                        f.write(f"  Average Duration: {total_mp4_duration/len(mp4_files):.2f}s\n")
                        f.write(f"  Average FPS: {total_mp4_frames/total_mp4_duration if total_mp4_duration>0 else 0:.2f}\n")
                    f.write("\n")

                # GIF details
                if gif_files:
                    f.write("GIF Segments Details:\n")
                    f.write("--------------------\n")
                    for idx, gif_file in enumerate(gif_files, 1):
                        try:
                            # Get comprehensive GIF information using the existing method
                            gif_info = self._get_gif_info_for_summary(str(gif_file))
                            size_mb = gif_file.stat().st_size / (1024 * 1024)
                            duration = gif_info.get('duration', 0.0)
                            fps = gif_info.get('fps', 0.0)
                            width = gif_info.get('width', 0)
                            height = gif_info.get('height', 0)
                            estimated_frames = int(round(duration * fps)) if duration > 0 and fps > 0 else 0
                            
                            f.write(f"GIF Segment {idx:03d}: {gif_file.name}\n")
                            f.write(f"  Duration: {duration:.2f}s\n")
                            f.write(f"  FPS: {fps:.2f}\n")
                            f.write(f"  Estimated Frame Count: {estimated_frames}\n")
                            f.write(f"  Resolution: {width}x{height}\n")
                            f.write(f"  Size: {size_mb:.2f}MB\n\n")
                        except Exception as e:
                            f.write(f"GIF Segment {idx:03d}: {gif_file.name}\n")
                            f.write(f"  Error reading info: {e}\n\n")

                    f.write("GIF Summary:\n")
                    f.write(f"  Total GIF Segments: {len(gif_files)}\n")
                    
                    # Calculate totals for GIF summary
                    total_gif_size = 0.0
                    total_gif_duration = 0.0
                    total_gif_frames = 0
                    
                    for gif_file in gif_files:
                        try:
                            gif_info = self._get_gif_info_for_summary(str(gif_file))
                            total_gif_size += gif_file.stat().st_size / (1024 * 1024)
                            total_gif_duration += gif_info.get('duration', 0.0)
                            fps = gif_info.get('fps', 0.0)
                            duration = gif_info.get('duration', 0.0)
                            if duration > 0 and fps > 0:
                                total_gif_frames += int(round(duration * fps))
                        except Exception:
                            pass
                    
                    f.write(f"  Total Size: {total_gif_size:.2f}MB\n")
                    f.write(f"  Total Duration: {total_gif_duration:.2f}s\n")
                    f.write(f"  Estimated Total Frames: {total_gif_frames}\n")
                    if len(gif_files) > 0:
                        f.write(f"  Average Size: {total_gif_size/len(gif_files):.2f}MB\n")
                        f.write(f"  Average Duration: {total_gif_duration/len(gif_files):.2f}s\n")
                        f.write(f"  Average FPS: {total_gif_frames/total_gif_duration if total_gif_duration>0 else 0:.2f}\n")
                    f.write("\n")

                # Overall
                total_files = len(mp4_files) + len(gif_files)
                f.write("Overall Summary:\n")
                f.write("---------------\n")
                f.write(f"Total Files: {total_files}\n")
                f.write(f"File Types: MP4 ({len(mp4_files)}), GIF ({len(gif_files)})\n")
                
                # Calculate overall totals
                total_size = 0.0
                total_duration = 0.0
                total_frames = 0
                
                # Add MP4 totals
                for mp4_file in mp4_files:
                    try:
                        info = FFmpegUtils.get_video_info(str(mp4_file))
                        total_size += mp4_file.stat().st_size / (1024 * 1024)
                        total_duration += float(info.get('duration', 0.0) or 0.0)
                        total_frames += int(info.get('frame_count', 0) or 0)
                    except Exception:
                        pass
                
                # Add GIF totals
                for gif_file in gif_files:
                    try:
                        gif_info = self._get_gif_info_for_summary(str(gif_file))
                        total_size += gif_file.stat().st_size / (1024 * 1024)
                        total_duration += gif_info.get('duration', 0.0)
                        fps = gif_info.get('fps', 0.0)
                        duration = gif_info.get('duration', 0.0)
                        if duration > 0 and fps > 0:
                            total_frames += int(round(duration * fps))
                    except Exception:
                        pass
                
                f.write(f"Total Size: {total_size:.2f}MB\n")
                f.write(f"Total Duration: {total_duration:.2f}s\n")
                f.write(f"Total Frames: {total_frames}\n")
                if total_duration > 0:
                    f.write(f"Overall Average FPS: {total_frames/total_duration:.2f}\n")

            # Atomic replace
            try:
                os.replace(str(temp_file), str(summary_file))
            except Exception:
                try:
                    subprocess.run(['attrib', '-R', '-S', '-H', str(summary_file)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                    os.replace(str(temp_file), str(summary_file))
                except Exception:
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except Exception:
                        pass
                    raise

            # Do not hide the summary; folder.jpg controls thumbnail selection

        except Exception as e:
            logger.error(f"Failed to create comprehensive segments summary: {e}")
            print(f"    ❌ Failed to create comprehensive summary: {e}")

    def _get_gif_info_for_summary(self, gif_path: str) -> Dict[str, Any]:
        """
        Get GIF information for summary generation using FFmpeg.
        This is a simplified version focused on summary needs.
        """
        try:
            import subprocess
            import re
            
            # Get basic file info
            file_size = os.path.getsize(gif_path)
            file_size_mb = file_size / (1024 * 1024)
            
            # Try to get GIF-specific info using FFmpeg
            cmd = ['ffmpeg', '-i', gif_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            
            duration = 0
            fps = 12.0  # Default GIF FPS
            width = 320
            height = 240
            
            if result.stderr:
                # Parse FFmpeg output for GIF info
                lines = result.stderr.split('\n')
                for line in lines:
                    if 'Duration:' in line:
                        # Extract duration
                        duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})', line)
                        if duration_match:
                            h, m, s, cs = map(int, duration_match.groups())
                            duration = h * 3600 + m * 60 + s + cs / 100
                    
                    elif 'Video:' in line:
                        # Extract resolution
                        res_match = re.search(r'(\d+)x(\d+)', line)
                        if res_match:
                            width, height = map(int, res_match.groups())
                        
                        # Extract FPS if available
                        fps_match = re.search(r'(\d+(?:\.\d+)?) fps', line)
                        if fps_match:
                            fps = float(fps_match.group(1))
            
            return {
                'duration': duration,
                'fps': fps,
                'width': width,
                'height': height,
                'file_size_mb': file_size_mb
            }
            
        except Exception as e:
            logger.warning(f"Error getting GIF info for summary: {e}")
            return {
                'duration': 0,
                'fps': 12.0,
                'width': 320,
                'height': 240,
                'file_size_mb': 0
            }

    def _ensure_segments_summary_exists(self, segments_folder: Path) -> None:
        """Ensure a comprehensive summary exists for the segments folder"""
        try:
            # Prefer sanitized base name (without '_segments')
            base_name = segments_folder.stem.replace('_segments', '')
            sanitized = segments_folder / f"~{base_name}_comprehensive_summary.txt"
            legacy = segments_folder / f"~{segments_folder.name}_comprehensive_summary.txt"

            if sanitized.exists():
                logger.debug(f"Summary already exists (sanitized): {sanitized.name}")
                return

            # If only legacy exists, rename it to sanitized name
            if legacy.exists() and not sanitized.exists():
                try:
                    # Clear attributes best-effort, then rename
                    try:
                        subprocess.run(['attrib', '-R', '-S', '-H', str(legacy)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                    except Exception:
                        pass
                    legacy.rename(sanitized)
                    logger.info(f"Renamed legacy summary to sanitized name: {sanitized.name}")
                    return
                except Exception as e:
                    logger.debug(f"Could not rename legacy summary: {e}")

            # Create sanitized summary if none exists
            self._create_comprehensive_segments_summary(segments_folder, base_name)
            logger.info(f"Created comprehensive summary for existing segments folder: {segments_folder.name}")
         
        except Exception as e:
            logger.debug(f"Could not ensure segments summary exists: {e}")

    def _ensure_all_segments_summaries_exist(self) -> None:
        """Create comprehensive summaries for all existing segmented folders that don't have them"""
        try:
            # Find all segments folders in output directory (recursive)
            segments_folders: List[Path] = []
            for subdir in self.output_dir.rglob('*'):
                try:
                    if subdir.is_dir() and subdir.name.endswith('_segments'):
                        segments_folders.append(subdir)
                except Exception:
                    continue
            
            if not segments_folders:
                logger.debug("No existing segments folders found")
                return
            
            logger.info(f"Found {len(segments_folders)} existing segments folders, ensuring summaries exist")
            
            for segments_folder in segments_folders:
                try:
                    self._ensure_segments_summary_exists(segments_folder)
                except Exception as e:
                    logger.debug(f"Could not ensure summary for {segments_folder.name}: {e}")
                    
        except Exception as e:
            logger.debug(f"Could not ensure all segments summaries exist: {e}")

