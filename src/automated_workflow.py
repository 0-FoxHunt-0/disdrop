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
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED

from .config_manager import ConfigManager
from .logger_setup import get_app_base_dir
from .hardware_detector import HardwareDetector
from .video_processing.video_compressor import DynamicVideoCompressor
from .gif_processing.gif_generator import GifGenerator
from .file_validator import FileValidator
from .ffmpeg_utils import FFmpegUtils
from .error_handler import ErrorHandler, ErrorCategory
from .utils.segments_naming import sanitize_segments_base_name, segments_summary_path
from .utils.segments_summary import write_segments_summary

logger = logging.getLogger(__name__)

CORE_TRACKER_METRICS = {
    'cache_hits',
    'cache_misses',
    'retries',
    'segmentation_events',
    'guardrail_events',
    'timeout_events',
    'validation_failures',
    'summary_cleanups',
    'mp4_moves'
}


@dataclass
class AnalysisTracker:
    """
    Lightweight tracker for workflow diagnostics (cache hits, retries, guardrails, etc.).
    Access is confined to the workflow thread, so no explicit locking is required.
    """
    counts: Dict[str, int] = field(default_factory=dict)
    recent_events: List[str] = field(default_factory=list)
    max_recent_events: int = 20

    def _remember(self, message: Optional[str]) -> None:
        if not message:
            return
        self.recent_events.append(message)
        if len(self.recent_events) > self.max_recent_events:
            self.recent_events.pop(0)

    def bump(self, metric: str, amount: int = 1) -> int:
        if amount <= 0:
            return self.counts.get(metric, 0)
        self.counts[metric] = self.counts.get(metric, 0) + amount
        return self.counts[metric]

    def record_cache_hit(self, context: str) -> None:
        self.bump('cache_hits')
        if context:
            self.bump(f'cache_hit_{context}')
        self._remember(f"cache_hit:{context}")

    def record_cache_miss(self, context: str) -> None:
        self.bump('cache_misses')
        if context:
            self.bump(f'cache_miss_{context}')
        self._remember(f"cache_miss:{context}")

    def record_retry(self, context: str) -> None:
        self.bump('retries')
        if context:
            self.bump(f'retry_{context}')
        self._remember(f"retry:{context}")

    def record_segmentation(self, context: str) -> None:
        self.bump('segmentation_events')
        if context:
            self.bump(f'segmentation_{context}')
        self._remember(f"segmentation:{context}")

    def record_guardrail(self, context: str) -> None:
        self.bump('guardrail_events')
        if context:
            self.bump(f'guardrail_{context}')
        self._remember(f"guardrail:{context}")

    def record_timeout(self, context: str) -> None:
        self.bump('timeout_events')
        if context:
            self.bump(f'timeout_{context}')
        self._remember(f"timeout:{context}")

    def record_validation_failure(self, context: str, amount: int = 1) -> None:
        self.bump('validation_failures', amount)
        if context:
            self.bump(f'validation_fail_{context}', amount)
        qualifier = f"+{amount}" if amount > 1 else ""
        self._remember(f"validation_failure:{context}{qualifier}")

    def record_summary_cleanup(self, context: str) -> None:
        self.bump('summary_cleanups')
        if context:
            self.bump(f'summary_cleanup_{context}')
        self._remember(f"summary_cleanup:{context}")

    def record_mp4_move(self, context: str) -> None:
        self.bump('mp4_moves')
        if context:
            self.bump(f'mp4_move_{context}')
        self._remember(f"mp4_move:{context}")

    def snapshot(self) -> Dict[str, Any]:
        """Return a shallow copy of counters and recent events for reporting."""
        return {
            'counts': dict(self.counts),
            'recent_events': list(self.recent_events[-self.max_recent_events:])
        }

    def top_metrics(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Return the top metrics by count, excluding derived per-context entries."""
        base_metrics = [
            (metric, value)
            for metric, value in self.counts.items()
            if metric in CORE_TRACKER_METRICS or '_' not in metric
        ]
        return sorted(base_metrics, key=lambda item: item[1], reverse=True)[:limit]

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

        # Ensure video compressor also gets the shutdown checker
        if hasattr(self.video_compressor, 'shutdown_requested'):
            # Video compressor uses its own shutdown flag, but we'll sync it with ours
            pass
        self.file_validator = FileValidator()
        # Console verbosity (controlled by CLI --debug)
        self.verbose = False
        
        # Enhanced error handling for batch processing resilience
        self.error_handler = ErrorHandler()
        # Per-run diagnostics tracker
        self.analysis_tracker = AnalysisTracker()
        
        # Workflow directories (default under installed package base)
        base_dir = Path(get_app_base_dir())

        # Segmentation control flag (propagated from CLI)
        self.force_single_file = False
        self.input_dir = base_dir / "input"
        self.output_dir = base_dir / "output"
        self.temp_dir = base_dir / "temp"
        # Separate failures directory outside of output
        self.failures_dir = base_dir / "failures"
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
        self._cache_dir: Path = self.temp_dir / 'cache'
        self._cache_file: Path = self._cache_dir / 'workflow_success_index.json'
        # Session timestamp to track current execution and clean up old cache entries
        self._session_start_time = time.time()
        # Cache persistence: track when cache was last used to allow persistence between executions
        self._cache_last_used_file = self._cache_dir / 'last_used.txt'
        # Validation manifest (metadata cache for detection/validation fast-paths)
        self.file_scan_parallelism = max(1, int(self.config.get('gif_settings.performance.file_scan_parallelism', 4) or 4))
        ttl_hours = self.config.get('gif_settings.performance.validation_manifest_ttl_hours', 12) or 12
        try:
            ttl_seconds = max(3600.0, float(ttl_hours) * 3600.0)
        except (TypeError, ValueError):
            ttl_seconds = 43200.0
        self._validation_manifest_ttl = ttl_seconds
        self._validation_manifest: Dict[str, Any] = {}
        self._validation_manifest_lock = threading.Lock()
        self._validation_manifest_file: Path = self._cache_dir / 'validation_manifest.json'
        
        # Signal handlers are now handled by the CLI, not here
        
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
        
        try:
            self._load_validation_manifest()
            self._cleanup_validation_manifest()
        except Exception:
            self._validation_manifest = {}
    
    
    def _ensure_directories(self):
        """Ensure all required directories exist"""
        for directory in [self.input_dir, self.output_dir, self.temp_dir, self.failures_dir]:
            directory.mkdir(exist_ok=True)
            logger.debug(f"Directory ensured: {directory}")

    def _safe_filename_for_filesystem(self, filename: str) -> str:
        """
        Convert filename to a safe string for filesystem operations.

        Delegates to the shared sanitizer so all components derive identical folder/summary names.
        """
        return sanitize_segments_base_name(filename)
    
    def _interruptible_sleep(self, duration: float, check_interval: float = 0.1):
        """Sleep that can be interrupted by shutdown signal"""
        elapsed = 0.0
        while elapsed < duration and not self.shutdown_requested:
            sleep_time = min(check_interval, duration - elapsed)
            time.sleep(sleep_time)
            elapsed += sleep_time
    
    def run_automated_workflow(self, check_interval: int = 5, max_size_mb: float = 10.0, verbose: bool = False, max_files: Optional[int] = None, input_dir: Optional[str] = None, output_dir: Optional[str] = None, no_cache: bool = False, max_input_size_bytes: Optional[int] = None):
        """
        Run the automated workflow
        
        Args:
            check_interval: How often to check for new files (seconds)
            max_size_mb: Maximum file size for outputs in MB
        """
        # Set verbosity for this run
        self.verbose = bool(verbose)

        # Allow overriding input/output directories via CLI
        if input_dir:
            try:
                self.input_dir = Path(input_dir)
                self.input_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not set input directory to '{input_dir}': {e}")
        if output_dir:
            try:
                self.output_dir = Path(output_dir)
                self.output_dir.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.warning(f"Could not set output directory to '{output_dir}': {e}")
        # Configure cache usage
        self.use_cache = not bool(no_cache)

        # Remove duplicate segment summaries before scanning existing folders
        self.cleanup_segments_summary_files()

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
                            cached_success = self._is_cached_success(file_path)
                            if cached_success:
                                if self.use_cache:
                                    try:
                                        self.analysis_tracker.record_cache_hit('workflow_loop')
                                    except Exception:
                                        logger.debug("Analysis tracker cache hit recording failed", exc_info=True)
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
                            else:
                                if self.use_cache:
                                    try:
                                        self.analysis_tracker.record_cache_miss('workflow_loop')
                                    except Exception:
                                        logger.debug("Analysis tracker cache miss recording failed", exc_info=True)
                            
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
        single GIF in the mirrored output folder or a `<stem>_segments` folder with GIFs.
        """
        try:
            stem = input_path.stem
            # Get the mirrored output location
            relative_path = self._get_relative_path(input_path)
            output_category_dir = self.output_dir / relative_path.parent
            
            # Prefer segments when they exist and contain valid GIFs
            segments_dir = output_category_dir / f"{stem}_segments"
            if segments_dir.exists() and segments_dir.is_dir():
                valid_segs, _ = self._validate_segment_folder_gifs(segments_dir, 10.0)
                if valid_segs:
                    self._record_success_cache(input_path, 'segments', segments_dir)
                    return
            # Otherwise, check for single GIF
            single_gif = output_category_dir / f"{stem}.gif"
            if single_gif.exists():
                # Fast check is sufficient for cache record
                is_valid, _ = self.file_validator.is_valid_gif_fast(str(single_gif), max_size_mb=10.0)
                if is_valid:
                    self._record_success_cache(input_path, 'single_gif', single_gif)
        except Exception:
            pass

    def _handle_segmentation_fallback_result(self, result: Dict[str, Any], input_file: Path, max_size_mb: float) -> str:
        """
        Handle the result of segmentation fallback from video compression
        
        Args:
            result: Compression result dictionary with segmentation information
            input_file: Original input file path
            max_size_mb: Target size limit
            
        Returns:
            Processing result status ('success', 'error', 'cancelled')
        """
        try:
            if not result.get('is_segmented_output', False):
                # Not a segmented output, handle as normal
                return 'success' if result.get('success', False) else 'error'
            
            # Check multiple possible keys: segments_folder (fallback), output_folder (regular), output_file (fallback)
            segments_folder_path = result.get('segments_folder') or result.get('output_folder') or result.get('output_file')
            
            # Fallback: construct path from input file if all keys missing
            if not segments_folder_path:
                input_dir = os.path.dirname(str(input_file))
                base_name = os.path.splitext(os.path.basename(str(input_file)))[0]
                segments_folder_path = os.path.join(input_dir, f"{base_name}_segments")
                logger.info(f"segments_folder_path missing from result, constructed fallback from input: {segments_folder_path}")
            
            # Convert to absolute path before checking existence
            segments_folder_path = os.path.abspath(segments_folder_path)
            logger.debug(f"Checking segments folder at: {segments_folder_path}")
            
            if not segments_folder_path or not os.path.exists(segments_folder_path):
                logger.error(f"Segmentation fallback completed but segments folder not found")
                logger.error(f"Expected path: {segments_folder_path}")
                logger.error(f"Result keys: {list(result.keys())}")
                logger.error(f"Result output_folder: {result.get('output_folder')}")
                logger.error(f"Result segments_folder: {result.get('segments_folder')}")
                logger.error(f"Result output_file: {result.get('output_file')}")
                logger.error(f"Input file: {input_file}")
                return 'error'
            
            segments_folder = Path(segments_folder_path)
            segments = result.get('segments', [])
            
            print(f"    ✅ Segmentation fallback successful: {segments_folder.name}")
            print(f"    📁 Created {len(segments)} segments totaling {result.get('size_mb', 0):.2f}MB")
            
            logger.info(f"Segmentation fallback successful for {input_file.name}: "
                       f"{len(segments)} segments in {segments_folder}")
            
            # Validate segments and organize for batch processing
            organized_result = self._organize_segmented_outputs(
                segments_folder, segments, input_file, max_size_mb
            )
            
            if organized_result.get('success', False):
                # Record success for caching
                self._record_success_cache(input_file, 'segments', segments_folder)
                
                # Generate GIFs from segments if needed
                gif_success = self._generate_gifs_from_segments(segments_folder, max_size_mb)
                
                if gif_success:
                    print(f"    🎨 GIF generation from segments successful")
                    logger.info(f"GIF generation from segments successful: {segments_folder}")
                    return 'success'
                else:
                    print(f"    ⚠️  Segmentation successful but GIF generation failed")
                    logger.warning(f"Segmentation successful but GIF generation failed: {segments_folder}")
                    return 'success'  # Still consider it success since video segmentation worked
            else:
                logger.error(f"Failed to organize segmented outputs: {organized_result.get('error', 'Unknown error')}")
                return 'error'
                
        except Exception as e:
            logger.error(f"Error handling segmentation fallback result: {e}")
            return 'error'

    def _organize_segmented_outputs(self, segments_folder: Path, segments: List[Dict[str, Any]], 
                                  input_file: Path, max_size_mb: float) -> Dict[str, Any]:
        """
        Organize segmented outputs for batch processing
        
        Args:
            segments_folder: Path to segments folder
            segments: List of segment information
            input_file: Original input file
            max_size_mb: Target size limit
            
        Returns:
            Organization result dictionary
        """
        try:
            # Ensure segments folder exists
            segments_folder.mkdir(parents=True, exist_ok=True)
            
            base_name = input_file.stem
            summary_path = write_segments_summary(
                segments_folder,
                base_name,
                logger=logger,
                analysis_tracker=self.analysis_tracker,
            )
            
            logger.info(f"Organized {len(segments)} segmented outputs in {segments_folder}")
            
            return {
                'success': True,
                'segments_folder': segments_folder,
                'summary_path': summary_path,
                'num_segments': len(segments)
            }
            
        except Exception as e:
            logger.error(f"Error organizing segmented outputs: {e}")
            return {
                'success': False,
                'error': str(e)
            }

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

    def _load_validation_manifest(self) -> None:
        """Load validation manifest metadata for fast detection skips."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        manifest_data: Dict[str, Any] = {}
        if self._validation_manifest_file.exists():
            try:
                with open(self._validation_manifest_file, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        manifest_data = loaded
            except Exception as e:
                logger.debug(f"Failed to load validation manifest: {e}")
        with self._validation_manifest_lock:
            self._validation_manifest = manifest_data

    def _save_validation_manifest(self) -> None:
        """Persist validation manifest to disk (best-effort)."""
        try:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        try:
            with self._validation_manifest_lock:
                snapshot = dict(self._validation_manifest)
            with open(self._validation_manifest_file, 'w', encoding='utf-8') as f:
                json.dump(snapshot, f, indent=2)
        except Exception as e:
            logger.debug(f"Failed to save validation manifest: {e}")

    def _cleanup_validation_manifest(self) -> None:
        """Remove stale or invalid manifest entries."""
        with self._validation_manifest_lock:
            if not self._validation_manifest:
                return
            now = time.time()
            keys_to_remove = []
            for key, record in self._validation_manifest.items():
                if not isinstance(record, dict):
                    keys_to_remove.append(key)
                    continue
                validated_at = float(record.get('validated_at', 0) or 0)
                if validated_at and now - validated_at > self._validation_manifest_ttl:
                    keys_to_remove.append(key)
                    continue
                path_str = record.get('path')
                if not path_str or not Path(path_str).exists():
                    keys_to_remove.append(key)
            for key in keys_to_remove:
                self._validation_manifest.pop(key, None)
            dirty = bool(keys_to_remove)
        if dirty:
            self._save_validation_manifest()

    def _manifest_key(self, file_path: Path) -> str:
        try:
            return str(file_path.resolve())
        except Exception:
            return str(file_path)

    def _check_cached_validation(
        self, file_path: Path, file_type: str, max_size_mb: Optional[float]
    ) -> Tuple[bool, bool, Dict[str, Any], Optional[str]]:
        """Return cached validation status if available."""
        if not file_path.exists():
            return False, False, {}, "File missing"
        signature = self._get_path_signature(file_path)
        key = self._manifest_key(file_path)
        with self._validation_manifest_lock:
            entry = self._validation_manifest.get(key)
        if not isinstance(entry, dict):
            return False, False, {}, None
        if entry.get('type') != file_type:
            return False, False, {}, None
        validated_at = float(entry.get('validated_at', 0) or 0)
        if validated_at and time.time() - validated_at > self._validation_manifest_ttl:
            return False, False, {}, None
        if signature != entry.get('signature'):
            return False, False, {}, None
        metadata = entry.get('metadata') or {}
        error_msg = entry.get('error')
        is_valid = bool(entry.get('valid'))
        if is_valid and max_size_mb is not None:
            size_mb = metadata.get('size_mb')
            if size_mb is not None and size_mb > max_size_mb:
                return True, False, metadata, f"File too large: {size_mb:.2f}MB > {max_size_mb}MB (cached)"
        return True, is_valid, metadata, error_msg

    def _update_validation_manifest(
        self,
        file_path: Path,
        file_type: str,
        valid: bool,
        metadata: Optional[Dict[str, Any]],
        error: Optional[str],
        max_size_mb: Optional[float]
    ) -> None:
        entry = {
            'path': str(file_path),
            'type': file_type,
            'valid': bool(valid),
            'error': error,
            'validated_at': time.time(),
            'signature': self._get_path_signature(file_path),
            'metadata': metadata or {},
            'size_limit_mb': max_size_mb,
        }
        key = self._manifest_key(file_path)
        with self._validation_manifest_lock:
            self._validation_manifest[key] = entry
        self._save_validation_manifest()

    def _get_relative_path(self, file_path: Path) -> Path:
        """Get relative path of file from input directory"""
        try:
            return file_path.relative_to(self.input_dir)
        except ValueError:
            return Path(file_path.name)
    
    def _find_source_video_for_gif(self, gif_file: Path) -> Optional[str]:
        """
        Attempt to locate a source video (e.g., MP4) that corresponds to the provided GIF.
        A richer source video lets the GIF optimizer run higher quality stages (palette
        regeneration, fps adjustments) without re-encoding from the already-compressed GIF.
        """
        base_name = gif_file.stem
        default_exts = ['.mp4', '.mov', '.mkv', '.webm', '.avi']
        try:
            config_exts = self.config.get('gif_settings.source_video_search.extensions', default_exts) or default_exts
        except Exception:
            config_exts = default_exts
        normalized_exts = [ext.lower() if isinstance(ext, str) else '.mp4' for ext in config_exts]
        
        candidate_dirs: List[Path] = []
        seen: set = set()
        
        def enqueue(path: Optional[Path]) -> None:
            if not path:
                return
            try:
                resolved = Path(path)
            except Exception:
                return
            if resolved in seen:
                return
            seen.add(resolved)
            candidate_dirs.append(resolved)
        
        # Start with the GIF's directory and a few ancestors
        enqueue(gif_file.parent)
        try:
            for parent in list(gif_file.parents)[:4]:
                enqueue(parent)
        except Exception:
            pass
        
        # Include segments directory next to the GIF if it exists
        enqueue(gif_file.parent / f"{base_name}_segments")
        
        # Consider output/input mirrors when relative path is available
        relative_path = self._get_relative_path(gif_file)
        if relative_path != Path(gif_file.name):
            rel_parent = relative_path.parent
            enqueue(self.output_dir / rel_parent)
            enqueue(self.input_dir / rel_parent)
            enqueue((self.output_dir / rel_parent) / f"{base_name}_segments")
        
        # Always check global output/temp directories
        enqueue(self.output_dir)
        enqueue(self.temp_dir)
        
        # Allow user-defined extra directories from config
        try:
            extra_dirs = self.config.get('gif_settings.source_video_search.extra_dirs', []) or []
        except Exception:
            extra_dirs = []
        for extra in extra_dirs:
            try:
                enqueue(Path(extra))
            except Exception:
                continue
        
        for directory in candidate_dirs:
            try:
                if not directory or not directory.exists() or not directory.is_dir():
                    continue
            except Exception:
                continue
            
            # Exact filename match first
            found_path = None
            for ext in normalized_exts:
                potential = directory / f"{base_name}{ext}"
                try:
                    if potential.exists():
                        found_path = potential
                        break
                except Exception:
                    continue
            if not found_path:
                # Fallback: look for files that start with the base name (case sensitive) but ensure extension matches
                try:
                    for match in directory.glob(f"{base_name}.*"):
                        if match.suffix.lower() in normalized_exts and match.exists():
                            found_path = match
                            break
                except Exception:
                    pass
            
            if found_path:
                logger.debug(f"[Workflow] Matched source video for {gif_file.name}: {found_path}")
                return str(found_path)
        
        logger.debug(f"[Workflow] No source video located for {gif_file.name}")
        return None
    
    def _make_cache_key(self, input_file: Path) -> str:
        """Create cache key using relative path to handle files with same name in different folders"""
        try:
            relative_path = self._get_relative_path(input_file)
            return str(relative_path)
        except Exception:
            return str(input_file.name)

    def _get_path_signature(self, file_path: Path) -> Dict[str, Any]:
        try:
            st = file_path.stat()
            return {'size': int(st.st_size), 'mtime': float(st.st_mtime)}
        except Exception:
            return {'size': None, 'mtime': None}

    def _get_input_signature(self, input_file: Path) -> Dict[str, Any]:
        try:
            return self._get_path_signature(input_file)
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
        
        Preserves folder structure from input directory.
        Best-effort operation; logs and prints a short note on success.
        """
        try:
            self.failures_dir.mkdir(exist_ok=True)
        except Exception:
            return

        try:
            # Get relative path to preserve folder structure
            relative_path = self._get_relative_path(src_path)
            
            # Create target directory structure in failures/
            target_dir = self.failures_dir / relative_path.parent
            target_dir.mkdir(parents=True, exist_ok=True)
            
            stem = src_path.stem
            suffix = src_path.suffix
            candidate = target_dir / src_path.name
            index = 1
            # Ensure we don't overwrite an existing file in failures
            while candidate.exists():
                candidate = target_dir / f"{stem} ({index}){suffix}"
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
        files: List[Path] = []
        if not self.input_dir.exists():
            return files
        
        scan_start = time.perf_counter()
        supported_video_extensions = self.file_validator.get_supported_video_extensions()
        supported_gif_extensions = {'.gif'}
        supported_extensions = supported_video_extensions | supported_gif_extensions
        
        potential_files = [
            file_path
            for file_path in self.input_dir.rglob("*")
            if (
                file_path.is_file()
                and file_path.suffix.lower() in supported_extensions
                and file_path not in processed_files
            )
        ]

        try:
            max_allowed = getattr(self, 'max_input_size_bytes', None)
        except Exception:
            max_allowed = None

        use_parallel = (
            not skip_stability_check
            and self.file_scan_parallelism > 1
            and len(potential_files) > max(8, self.file_scan_parallelism)
        )

        if len(potential_files) > 10:
            if use_parallel:
                print(
                    f"🔍 Scanning {len(potential_files)} files with {self.file_scan_parallelism} workers..."
                )
            elif skip_stability_check:
                print(f"🔍 Scanning {len(potential_files)} files (initial scan - skipping stability check)...")
            else:
                print(f"🔍 Scanning {len(potential_files)} files for stability and existing outputs...")

        def _evaluate_candidate(candidate: Path, emit_console: bool = False) -> Optional[Path]:
            if self.shutdown_requested:
                return None
            try:
                if max_allowed is not None and candidate.exists():
                    size_bytes = candidate.stat().st_size
                    if size_bytes > int(max_allowed):
                        safe_name = self._safe_filename_for_logging(candidate.name)
                        message = f"{safe_name}: {size_bytes/1024/1024:.2f}MB exceeds max input size"
                        if emit_console:
                            print(f"    ⏭️  Skipping {message}")
                        logger.info(f"Skipping {candidate} due to max input size limit: {size_bytes} > {max_allowed} bytes")
                        return None
            except Exception as size_error:
                logger.debug(f"Could not enforce max input size on {candidate}: {size_error}")
            stable = skip_stability_check or self._is_file_stable(candidate)
            return candidate if stable else None

        if use_parallel and potential_files:
            max_workers = min(self.file_scan_parallelism, len(potential_files))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_map = {
                    executor.submit(_evaluate_candidate, file_path, False): file_path
                    for file_path in potential_files
                }
                completed = 0
                for future in as_completed(future_map):
                    completed += 1
                    if self.shutdown_requested:
                        break
                    try:
                        result = future.result()
                        if result:
                            files.append(result)
                    except Exception as e:
                        logger.debug(f"Parallel scan error for {future_map[future]}: {e}")
                    if len(potential_files) > 10 and completed % max_workers == 0:
                        print(
                            f"\r  📁 Checked {completed}/{len(potential_files)} files...",
                            end="",
                            flush=True,
                        )
        else:
            for i, file_path in enumerate(potential_files):
                if self.shutdown_requested:
                    if len(potential_files) > 10:
                        print(f"\r🛑 File scan interrupted by user" + " " * 30)
                    break
                if len(potential_files) > 10:
                    status = "Processing" if skip_stability_check else "Checking"
                    print(
                        f"\r  📁 {status} file {i+1}/{len(potential_files)}: {file_path.name[:30]}...",
                        end="",
                        flush=True,
                    )
                result = _evaluate_candidate(file_path, emit_console=True)
                if result:
                    files.append(result)

        if len(potential_files) > 10:
            print(f"\r✅ File scan complete: {len(files)} files ready for processing" + " " * 20)

        def _creation_time_desc(path: Path) -> float:
            try:
                st = path.stat()
                return float(getattr(st, 'st_ctime', st.st_mtime))
            except Exception:
                try:
                    return float(path.stat().st_mtime)
                except Exception:
                    return 0.0

        sorted_files = sorted(files, key=_creation_time_desc, reverse=True)
        scan_duration = time.perf_counter() - scan_start
        logger.debug(
            f"File scan completed in {scan_duration:.3f}s "
            f"({len(sorted_files)} ready / {len(potential_files)} candidates, parallel={'yes' if use_parallel else 'no'})"
        )
        return sorted_files
    
    def _has_existing_output(self, input_file: Path) -> bool:
        """
        Check for existing output files in the mirrored folder structure.
        
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
        
        # Get relative path to determine correct output location
        relative_path = self._get_relative_path(input_file)
        output_category_dir = self.output_dir / relative_path.parent
        
        # Check if input is a GIF file
        if input_file.suffix.lower() == '.gif':
            # For GIF files, check for existing optimized GIF output
            output_patterns = [
                f"{base_name}.gif",
            ]
            
            # Check in the mirrored category folder first
            for pattern in output_patterns:
                output_path = output_category_dir / pattern
                if self._is_valid_existing_output(output_path, pattern, input_file):
                    logger.info(f"Found existing GIF output for {input_file.name}: {output_path}")
                    return True
            
            return False
        else:
            # For video files, check for MP4 outputs (video processing)
            # GIF creation will be handled separately in the workflow
            output_patterns = [
                f"{base_name}.mp4",
            ]
            
            # Check in the mirrored category folder first
            for pattern in output_patterns:
                output_path = output_category_dir / pattern
                if self._is_valid_existing_output(output_path, pattern, input_file):
                    logger.info(f"Found existing video output for {input_file.name}: {output_path}")
                    return True
            
            # Check for segmented output in the mirrored category folder
            segments_folder = output_category_dir / f"{base_name}_segments"
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
                cached_available, cached_valid, _meta, _err = self._check_cached_validation(output_path, 'video', None)
                if cached_available:
                    basic_valid = cached_valid
                else:
                    probe_valid, metadata, probe_error = self.file_validator.probe_video_metadata(str(output_path))
                    if not probe_valid:
                        enhanced_valid, enhanced_error = self.file_validator.is_valid_video(str(output_path))
                        basic_valid = bool(enhanced_valid)
                        error_msg = enhanced_error or probe_error
                    else:
                        basic_valid = True
                        error_msg = None
                    self._update_validation_manifest(
                        output_path, 'video', basic_valid, metadata, error_msg, None
                    )
                
            elif pattern_type.endswith('.gif'):
                cached_available, cached_valid, _meta, _err = self._check_cached_validation(output_path, 'gif', None)
                if cached_available:
                    basic_valid = cached_valid
                else:
                    probe_valid, metadata, probe_error = self.file_validator.probe_gif_metadata(str(output_path), max_size_mb=None)
                    if not probe_valid:
                        enhanced_valid, enhanced_error = self.file_validator.is_valid_gif_with_enhanced_checks(
                            str(output_path), original_path=None, max_size_mb=None
                        )
                        basic_valid = bool(enhanced_valid)
                        error_msg = enhanced_error or probe_error
                    else:
                        basic_valid = True
                        error_msg = None
                    self._update_validation_manifest(
                        output_path, 'gif', basic_valid, metadata, error_msg, None
                    )
                
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
                try:
                    self.analysis_tracker.record_validation_failure('input_video')
                except Exception:
                    logger.debug("Analysis tracker validation failure recording failed", exc_info=True)
                return 'skipped'
            print("  ✅ Video validation passed")
            
            # Check for shutdown before step 2
            if self.shutdown_requested:
                return 'cancelled'
            
            # Step 2: Convert to MP4 if needed and optimize
            print("  🔄 Convert/optimize to MP4")
            try:
                mp4_file = self._ensure_mp4_format(video_file, max_size_mb)
                if self.shutdown_requested:
                    return 'cancelled'
                if not mp4_file:
                    logger.error(f"MP4 optimization failed for {video_file.name}")
                    print(f"  ❌ MP4 optimization failed - continuing with other files")
                    return 'error'
                print(f"  ✅ MP4 optimization complete: {mp4_file.name}")
            except Exception as mp4_error:
                logger.error(f"MP4 processing error for {video_file.name}: {mp4_error}")
                print(f"  ❌ MP4 processing error: {str(mp4_error)[:100]}... - continuing with other files")
                return 'error'
            
            # Check for shutdown before step 3
            if self.shutdown_requested:
                return 'cancelled'
            
            # Step 3: Generate and optimize GIF (ALWAYS attempt this step)
            print("  🎨 Generate/optimize GIF")
            try:
                gif_success = self._generate_and_optimize_gif(mp4_file, max_size_mb)
                
                if gif_success:
                    print(f"  ✅ GIF generation complete: {mp4_file.stem}.gif")
                    print(f"  🎉 Done: {video_file.name}")
                    logger.info(f"Successfully processed: {video_file.name}")
                    return 'success'
                else:
                    print(f"  ❌ GIF generation failed - continuing with other files")
                    logger.error(f"GIF generation failed for: {video_file.name}")
                    return 'error'
            except Exception as gif_error:
                logger.error(f"GIF processing error for {video_file.name}: {gif_error}")
                print(f"  ❌ GIF processing error: {str(gif_error)[:100]}... - continuing with other files")
                return 'error'
            
        except Exception as e:
            # Use centralized error handler for consistent categorization and logging
            error = self.error_handler.handle_error(
                exception=e,
                file_path=str(video_file),
                context="automated_workflow_video_processing",
                continue_processing=True
            )
            
            # User-friendly console output based on error category
            if error.category == ErrorCategory.BITRATE_VALIDATION:
                print(f"  ❌ Bitrate validation error - file may be too long for target size")
                print(f"     Suggestion: Try segmentation or increase target size")
            elif error.category == ErrorCategory.SEGMENTATION:
                print(f"  ❌ Segmentation error - unable to split video appropriately")
                print(f"     Suggestion: Check video duration and complexity")
            elif error.category == ErrorCategory.ENCODER:
                print(f"  ❌ Encoder error - video encoding failed")
                print(f"     Suggestion: Try software encoding or check FFmpeg")
            elif error.category == ErrorCategory.MEMORY:
                print(f"  ❌ Memory error - insufficient system resources")
                print(f"     Suggestion: Reduce resolution or enable segmentation")
            else:
                print(f"  ❌ Processing error: {str(e)[:100]}...")
                print(f"     Continuing with other files...")
            
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
            
            # Step 1: Check for existing single GIF file in the same category folder as MP4
            gif_name = mp4_file.stem + ".gif"
            gif_path = mp4_file.parent / gif_name
            
            if gif_path.exists():
                print(f"  🔍 Found existing GIF, validating: {gif_name}")
                cached_available, cached_valid, _cached_meta, cached_error = self._check_cached_validation(
                    gif_path, 'gif', max_size_mb
                )
                if cached_available:
                    logger.debug(
                        f"Existing GIF validation for {gif_name} served from manifest cache (valid={cached_valid})"
                    )
                    if cached_valid:
                        print(f"  ✅ Valid GIF already exists: {gif_name}")
                        logger.info(f"Valid GIF already exists for {video_file.name}: {gif_name}")
                        self._record_success_cache(video_file, 'single_gif', gif_path)
                        return 'success'
                    else:
                        cached_reason = cached_error or "cached validation failure"
                        print(f"  🔄 GIF validation failed ({cached_reason}), will regenerate")
                        try:
                            self.analysis_tracker.record_validation_failure('existing_gif_manifest')
                        except Exception:
                            logger.debug("Analysis tracker existing GIF manifest failure recording failed", exc_info=True)
                        logger.info(f"Existing GIF validation failed (cached): {cached_reason}, will regenerate: {gif_name}")
                else:
                    validation_start = time.perf_counter()
                    probe_valid, metadata, probe_error = self.file_validator.probe_gif_metadata(
                        str(gif_path), max_size_mb=max_size_mb
                    )
                    if not probe_valid:
                        enhanced_valid, enhanced_error = self.file_validator.is_valid_gif_with_enhanced_checks(
                            str(gif_path), original_path=None, max_size_mb=max_size_mb
                        )
                        final_valid = bool(enhanced_valid)
                        error_msg = enhanced_error or probe_error
                    else:
                        final_valid = True
                        error_msg = None
                    self._update_validation_manifest(
                        gif_path, 'gif', final_valid, metadata, error_msg, max_size_mb
                    )
                    validation_elapsed = time.perf_counter() - validation_start
                    logger.debug(
                        f"Existing GIF validation for {gif_name} took {validation_elapsed:.3f}s (cached=False, result={final_valid})"
                    )
                    if final_valid:
                        print(f"  ✅ Valid GIF already exists: {gif_name}")
                        logger.info(f"Valid GIF already exists for {video_file.name}: {gif_name}")
                        self._record_success_cache(video_file, 'single_gif', gif_path)
                        return 'success'
                    print(f"  🔄 GIF validation failed ({error_msg}), will regenerate")
                    try:
                        self.analysis_tracker.record_validation_failure('existing_gif_probe')
                    except Exception:
                        logger.debug("Analysis tracker existing GIF probe failure recording failed", exc_info=True)
                    logger.info(f"Existing GIF validation failed: {error_msg}, will regenerate: {gif_name}")
            else:
                print(f"  📄 No single GIF found: {gif_name}")
            
            # Step 2: Check for existing segment folder in the same category folder as MP4
            segments_folder = mp4_file.parent / f"{mp4_file.stem}_segments"
            if segments_folder.exists() and segments_folder.is_dir():
                print(f"  🔍 Found segments folder, validating: {segments_folder.name}")
                
                # Validate segment folder GIFs
                segments_validation_start = time.perf_counter()
                valid_segments, invalid_segments = self._validate_segment_folder_gifs(segments_folder, max_size_mb)
                segments_validation_elapsed = time.perf_counter() - segments_validation_start
                logger.debug(
                    f"Segment folder validation for {segments_folder.name} took {segments_validation_elapsed:.3f}s "
                    f"(valid={len(valid_segments)}, invalid={len(invalid_segments)})"
                )
                
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

                    return 'success'
                else:
                    print(f"    🔄 Regenerating segment GIFs (invalid/corrupted): {segments_folder.name}")
                    try:
                        self.analysis_tracker.record_validation_failure('segment_gif_folder', amount=max(1, len(invalid_segments)))
                    except Exception:
                        logger.debug("Analysis tracker segment GIF folder failure recording failed", exc_info=True)
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
        """Find the existing MP4 output file for a given input file in the mirrored folder structure"""
        base_name = input_file.stem
        
        # Check the mirrored output location first
        relative_path = self._get_relative_path(input_file)
        output_category_dir = self.output_dir / relative_path.parent
        mp4_path = output_category_dir / f"{base_name}.mp4"
        if mp4_path.exists():
            cached_available, cached_valid, _cached_meta, _cached_error = self._check_cached_validation(
                mp4_path, 'video', None
            )
            if cached_available and cached_valid:
                logger.debug(f"Existing MP4 validated via manifest cache: {mp4_path.name}")
                return mp4_path
            validation_start = time.perf_counter()
            probe_valid, metadata, probe_error = self.file_validator.probe_video_metadata(str(mp4_path))
            if not probe_valid:
                enhanced_valid, enhanced_error = self.file_validator.is_valid_video(str(mp4_path))
                final_valid = bool(enhanced_valid)
                error_msg = enhanced_error or probe_error
            else:
                final_valid = True
                error_msg = None
            self._update_validation_manifest(mp4_path, 'video', final_valid, metadata, error_msg, None)
            validation_elapsed = time.perf_counter() - validation_start
            logger.debug(
                f"Existing MP4 validation for {mp4_path.name} took {validation_elapsed:.3f}s (cached=False, result={final_valid})"
            )
            if final_valid:
                return mp4_path
        
        return None
    
    def _ensure_mp4_format(self, video_file: Path, max_size_mb: float) -> Optional[Path]:
        """Ensure video is in MP4 format and optimized, preserving folder structure"""
        # Get relative path to preserve folder structure
        relative_path = self._get_relative_path(video_file)
        output_category_dir = self.output_dir / relative_path.parent
        
        # Create output directory structure
        output_category_dir.mkdir(parents=True, exist_ok=True)
        
        output_name = video_file.stem + ".mp4"
        output_path = output_category_dir / output_name
        
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
        
        # Check if segments folder already exists (from previous segmentation) in the mirrored location
        segments_folder = output_category_dir / f"{video_file.stem}_segments"
        if segments_folder.exists() and segments_folder.is_dir():
            print(f"    ♻️  Found existing segments folder: {segments_folder.name}")
            logger.debug(f"Found existing segments folder: {segments_folder.name}")
            
            # Prefer reusing existing MP4 segments if present and valid under target
            try:
                mp4_files = sorted([f for f in segments_folder.iterdir() if f.is_file() and f.suffix.lower() == '.mp4'])
            except Exception:
                mp4_files = []
            if mp4_files:
                all_valid = True
                for mp4 in mp4_files:
                    # Important: Do not compare segment duration against original video.
                    # Validate integrity and size only to prevent false duration ratio failures.
                    ok, err = self.file_validator.is_valid_video_with_enhanced_checks(
                        str(mp4), original_path=None, max_size_mb=max_size_mb
                    )
                    if not ok:
                        all_valid = False
                        logger.info(f"Existing MP4 segment invalid or too large, will reprocess: {mp4.name} ({err})")
                        break
                if all_valid:
                    print(f"    ♻️  Using existing MP4 segments: {segments_folder.name}")
                    logger.info(f"Using existing MP4 segments: {segments_folder}")
                    # Ensure comprehensive summary exists for the segments folder
                    try:
                        self._ensure_segments_summary_exists(segments_folder)
                    except Exception:
                        pass
                    # Record success and return the folder as the output artifact
                    self._record_success_cache(video_file, 'segments', segments_folder)
                    return segments_folder

            # If no valid MP4 segments, fall back to checking GIF segments for reuse
            valid_segments, invalid_segments = self._validate_segment_folder_gifs(segments_folder, max_size_mb)
            if valid_segments and not invalid_segments:
                print(f"    ♻️  Using existing segment GIFs: {segments_folder.name}")
                logger.debug(f"Valid segment GIFs already exist: {segments_folder.name}")
                try:
                    self._ensure_segments_summary_exists(segments_folder)
                except Exception:
                    pass
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
                max_size_mb=max_size_mb,
                force_single_file=self.force_single_file
            )
            
            # Check for shutdown after compression
            if self.shutdown_requested:
                print(f"    🛑 Processing cancelled after compression")
                return None
            
            if result.get('cancelled'):
                return None

            guardrail_triggered = result.get('guardrail_triggered')
            guardrail_reason = result.get('guardrail_reason', 'video_guardrail')
            if guardrail_triggered:
                guardrail_details = result.get('guardrail_details')
                print(f"    ⚠️ Guardrail triggered ({guardrail_reason}); routing through segmentation-safe output")
                logger.info(
                    "MP4 guardrail activated (%s): %s",
                    guardrail_reason,
                    guardrail_details or 'no additional details'
                )
                try:
                    self.analysis_tracker.record_guardrail(guardrail_reason)
                except Exception:
                    logger.debug("Analysis tracker guardrail recording failed", exc_info=True)

            if result.get('success', False):
                # Check if segmentation was used (including segmentation fallback)
                if (result.get('method') == 'segmentation' or 
                    result.get('method') == 'segmentation_fallback' or 
                    result.get('is_segmented_output', False) or 
                    'segments' in result):
                    
                    # Handle segmentation fallback result
                    fallback_status = self._handle_segmentation_fallback_result(result, video_file, max_size_mb)
                    
                    if fallback_status == 'success':
                        # Get the segments folder path from result
                        # Check multiple possible keys: segments_folder (fallback), output_folder (regular), output_file (fallback)
                        segments_folder_path = result.get('segments_folder') or result.get('output_folder') or result.get('output_file')
                        
                        # Fallback: construct path from output_path if all keys missing
                        if not segments_folder_path:
                            output_dir = os.path.dirname(str(output_path))
                            base_name = os.path.splitext(os.path.basename(str(output_path)))[0]
                            segments_folder_path = os.path.join(output_dir, f"{base_name}_segments")
                            logger.info(f"segments_folder_path missing from result, constructed fallback: {segments_folder_path}")
                        
                        # Convert to absolute path and validate
                        segments_folder_path = os.path.abspath(segments_folder_path)
                        logger.debug(f"Checking segments folder at: {segments_folder_path}")
                        
                        if segments_folder_path and os.path.exists(segments_folder_path):
                            segments_folder = Path(segments_folder_path)

                            if guardrail_triggered:
                                try:
                                    self.analysis_tracker.record_segmentation(guardrail_reason)
                                except Exception:
                                    logger.debug("Analysis tracker segmentation guardrail recording failed", exc_info=True)
                            
                            # Verify folder contains segments before returning
                            try:
                                mp4s = list(segments_folder.glob('*.mp4'))
                                if not mp4s:
                                    logger.warning(f"Segments folder exists but contains no MP4 files: {segments_folder_path}")
                                    # Don't fail - folder exists, segments might be processing
                                else:
                                    logger.debug(f"Segments folder validated: {len(mp4s)} MP4 files found in {segments_folder_path}")
                            except Exception as e:
                                logger.warning(f"Could not verify segments in folder {segments_folder_path}: {e}")
                            
                            # Best-effort: create a cover image in the segments folder
                            try:
                                # Use the first segment as thumbnail source if exists, else the original input video
                                first_seg = None
                                try:
                                    mp4s = sorted(list(segments_folder.glob('*.mp4')))
                                    if mp4s:
                                        first_seg = mp4s[0]
                                except Exception:
                                    first_seg = None
                                thumb_src = str(first_seg or video_file)
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
                            logger.error(f"Segmentation completed but segments folder not found. Expected path: {segments_folder_path}")
                            logger.error(f"Result keys: {list(result.keys())}")
                            logger.error(f"Result output_folder: {result.get('output_folder')}")
                            logger.error(f"Result segments_folder: {result.get('segments_folder')}")
                            logger.error(f"Result output_file: {result.get('output_file')}")
                            print(f"    ❌ Segmentation completed but segments folder not found")
                            print(f"    Expected: {segments_folder_path}")
                            return None
                    else:
                        print(f"    ❌ Segmentation fallback handling failed")
                        logger.error(f"Segmentation fallback handling failed")
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

                        # Note: Do not create folder.jpg for single MP4 outputs; reserved for segmented outputs only
                        
                        return output_path
                    else:
                        print(f"    ❌ Processed MP4 validation failed: {error_msg}")
                        logger.error(f"Processed MP4 validation failed: {error_msg}")
                        try:
                            self.analysis_tracker.record_validation_failure('mp4_output')
                        except Exception:
                            logger.debug("Analysis tracker MP4 validation failure recording failed", exc_info=True)
                        
                        # Check if the error is about file size being too large or file is empty (encoding failure)
                        if ("too large" in error_msg.lower() or "file size" in error_msg.lower() or 
                            "file is empty" in error_msg.lower()):
                            print(f"    🔁 Retrying with segmentation allowed ({error_msg})...")
                            logger.info(f"Single-file validation failed ({error_msg}); retrying with segmentation permitted "
                                        f"(encoder may still produce a single file).")
                            try:
                                self.analysis_tracker.record_retry('mp4_validation')
                            except Exception:
                                logger.debug("Analysis tracker MP4 retry recording failed", exc_info=True)
                            
                            # Remove the oversized/invalid file before retry
                            if output_path.exists():
                                output_path.unlink()
                            
                            try:
                                result = self.video_compressor.compress_video(
                                    input_path=str(video_file),
                                    output_path=str(output_path),
                                    max_size_mb=max_size_mb,
                                    force_single_file=False
                                )
                                
                                if not result.get('success', False):
                                    print(f"    ❌ Retry with segmentation allowed failed: {result.get('error', 'Unknown error')}")
                                    logger.error(f"Segmentation-enabled retry failed: {result.get('error', 'Unknown error')}")
                                    return None
                                
                                segmentation_used = (
                                    result.get('is_segmented_output')
                                    or bool(result.get('segments'))
                                    or bool(result.get('segments_folder'))
                                    or result.get('method') in {'segmentation', 'segmentation_fallback'}
                                )
                                
                                if segmentation_used:
                                    try:
                                        self.analysis_tracker.record_guardrail('video_segmentation_retry')
                                        self.analysis_tracker.record_segmentation('video_mp4_retry')
                                    except Exception:
                                        logger.debug("Analysis tracker segmentation recording failed", exc_info=True)
                                    fallback_status = self._handle_segmentation_fallback_result(result, video_file, max_size_mb)
                                    
                                    if fallback_status == 'success':
                                        segments_folder_path = (
                                            result.get('segments_folder')
                                            or result.get('output_folder')
                                            or result.get('output_file')
                                        )
                                        if segments_folder_path and os.path.exists(segments_folder_path):
                                            return Path(segments_folder_path)
                                        print(f"    ❌ Segmentation fallback completed but segments folder not found")
                                        logger.error(f"Segmentation fallback completed but segments folder not found")
                                        return None
                                    
                                    print(f"    ❌ Segmentation fallback handling failed")
                                    logger.error(f"Segmentation fallback handling failed")
                                    return None
                                
                                # No segmentation was required; validate the new single-file output
                                print(f"    ✅ Retry succeeded without segmentation; validating MP4...")
                                logger.info("Segmentation-enabled retry produced a compliant single MP4; validating output.")
                                
                                is_valid, retry_error = self.file_validator.is_valid_video_with_enhanced_checks(
                                    str(output_path),
                                    original_path=str(video_file),
                                    max_size_mb=max_size_mb
                                )
                                
                                if is_valid:
                                    size_mb = result.get('size_mb', 0)
                                    print(f"    ✨ Video optimization successful after retry: {size_mb:.2f}MB")
                                    logger.info(f"Video optimization successful after retry: {output_name} ({size_mb:.2f}MB)")
                                    self._record_success_cache(video_file, 'single_mp4', output_path)
                                    try:
                                        specs = FFmpegUtils.get_detailed_file_specifications(str(output_path))
                                        specs_log = FFmpegUtils.format_file_specifications_for_logging(specs)
                                        logger.info(f"Final video file specifications - {specs_log}")
                                    except Exception as e:
                                        logger.warning(f"Could not log detailed video specifications: {e}")
                                    return output_path
                                
                                print(f"    ❌ Retry output validation failed: {retry_error}")
                                logger.error(f"Retry output validation failed: {retry_error}")
                                try:
                                    self.analysis_tracker.record_validation_failure('mp4_retry_output')
                                except Exception:
                                    logger.debug("Analysis tracker retry validation failure recording failed", exc_info=True)
                                if output_path.exists():
                                    output_path.unlink()
                                return None
                            
                            except Exception as e:
                                print(f"    ❌ Segmentation retry error: {e}")
                                logger.error(f"Segmentation retry error: {e}")
                                return None
                        else:
                            # Other validation error, remove the invalid file
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
                    # The MP4 should be in the same directory as the segments folder
                    src = segments_folder.parent / f"{base_name}.mp4" if base_name else mp4_file
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
            # Determine output location based on MP4 file location (preserve folder structure)
            mp4_category_dir = mp4_file.parent
            
            gif_name = mp4_file.stem + ".gif"
            temp_gif_path = self.temp_dir / gif_name
            final_gif_path = mp4_category_dir / gif_name
            
            # Check for existing segment folder first in the same category folder
            segments_folder = mp4_category_dir / f"{mp4_file.stem}_segments"
            source_mp4_path = self._resolve_mp4_source_path(mp4_file, segments_folder)
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
                        write_segments_summary(
                            segments_folder,
                            mp4_file.stem,
                            logger=logger,
                            analysis_tracker=self.analysis_tracker,
                        )
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
                if not source_mp4_path.exists():
                    print(f"    ❌ MP4 file no longer exists: {source_mp4_path}")
                    logger.error(f"MP4 file no longer exists: {source_mp4_path}")
                    return False
                
                # Get video duration to preserve full length in GIF
                video_duration = FFmpegUtils.get_video_duration(str(source_mp4_path))
                logger.info(f"Video duration: {video_duration:.2f}s - generating GIF with full duration")
                
                # Double-check file exists before GIF generation
                if not source_mp4_path.exists():
                    print(f"    ❌ MP4 file disappeared during processing: {source_mp4_path}")
                    logger.error(f"MP4 file disappeared during processing: {source_mp4_path}")
                    return False
                
                # Generate GIF to temp directory first with full duration
                result = self.gif_generator.create_gif(
                    input_video=str(source_mp4_path),
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

                    # Place segments in the same directory as the MP4 file (preserve folder structure)
                    final_segments_dir = mp4_file.parent / f"{mp4_file.stem}_segments"

                    try:
                        if temp_segments_dir.exists():
                            # Track files that should be moved
                            temp_files_before = set()
                            if temp_segments_dir.exists() and temp_segments_dir.is_dir():
                                temp_files_before = {item.name for item in temp_segments_dir.iterdir() if item.is_file()}
                            
                            if final_segments_dir.exists() and final_segments_dir.is_dir():
                                # Merge/move files into existing folder
                                for item in temp_segments_dir.iterdir():
                                    if item.is_file():
                                        final_path = final_segments_dir / item.name
                                        shutil.move(str(item), str(final_path))
                                        # Verify file was moved
                                        if not final_path.exists():
                                            logger.error(f"File move failed: {item.name} not found in final location")
                                        elif item.exists():
                                            logger.warning(f"File still exists in temp after move: {item.name}, attempting cleanup")
                                            try:
                                                item.unlink()
                                            except Exception:
                                                pass
                            else:
                                # Track files before move to ensure all are moved
                                temp_files_before_move = []
                                try:
                                    for item in temp_segments_dir.iterdir():
                                        if item.is_file():
                                            temp_files_before_move.append(item.name)
                                except Exception:
                                    pass
                                
                                shutil.move(str(temp_segments_dir), str(final_segments_dir))
                                
                                # Verify folder was moved (not copied) and all files are present
                                if temp_segments_dir.exists():
                                    logger.warning(f"Temp segments folder still exists after move: {temp_segments_dir}, attempting cleanup")
                                    try:
                                        # Try to remove if empty or contains only metadata
                                        remaining = list(temp_segments_dir.iterdir())
                                        if not remaining:
                                            temp_segments_dir.rmdir()
                                        else:
                                            logger.warning(f"Temp segments folder not empty after move: {[str(f) for f in remaining]}")
                                            # Check if any segment files are still in temp
                                            remaining_files = [f for f in remaining if f.is_file() and f.suffix.lower() in ['.gif', '.mp4'] and not f.name.startswith('~')]
                                            if remaining_files:
                                                logger.error(f"Segment files still in temp folder after move: {[f.name for f in remaining_files]}")
                                                # Try to move remaining files to final location
                                                for remaining_file in remaining_files:
                                                    try:
                                                        target = final_segments_dir / remaining_file.name
                                                        if not target.exists():
                                                            shutil.move(str(remaining_file), str(target))
                                                            logger.info(f"Moved remaining segment file to final location: {remaining_file.name}")
                                                    except Exception as e:
                                                        logger.warning(f"Could not move remaining file {remaining_file.name}: {e}")
                                    except Exception as e:
                                        logger.debug(f"Could not clean up temp segments folder: {e}")
                                
                                # Verify all files are in final location
                                if final_segments_dir.exists():
                                    final_files = {item.name for item in final_segments_dir.iterdir() if item.is_file()}
                                    missing_files = [f for f in temp_files_before_move if f not in final_files]
                                    if missing_files:
                                        logger.warning(f"Some files missing from final segments folder after move: {missing_files}")
                                    else:
                                        logger.debug(f"All {len(temp_files_before_move)} files successfully moved to final segments folder")

                            # Validate that final segments folder contains the expected files
                            if final_segments_dir.exists():
                                final_files = {item.name for item in final_segments_dir.iterdir() if item.is_file() and not item.name.startswith('~')}
                                # Check for segment files in parent directory that should have been moved
                                parent_dir = final_segments_dir.parent
                                parent_segment_files = []
                                try:
                                    for item in parent_dir.iterdir():
                                        if item.is_file() and item.name.startswith(mp4_file.stem) and item.suffix.lower() in ['.gif', '.mp4']:
                                            # Check if this looks like a segment file that should be in segments folder
                                            if '_segment_' in item.name.lower() or '_part_' in item.name.lower():
                                                parent_segment_files.append(item)
                                except Exception:
                                    pass
                                
                                if parent_segment_files:
                                    logger.warning(f"Found segment files in parent directory that should be in segments folder: {[f.name for f in parent_segment_files]}")
                                    # Move any orphaned segment files to segments folder
                                    for orphan_file in parent_segment_files:
                                        try:
                                            target = final_segments_dir / orphan_file.name
                                            if not target.exists():
                                                shutil.move(str(orphan_file), str(target))
                                                logger.info(f"Moved orphaned segment file to segments folder: {orphan_file.name}")
                                            else:
                                                logger.debug(f"Orphaned segment file already exists in segments folder, removing from parent: {orphan_file.name}")
                                                orphan_file.unlink()
                                        except Exception as e:
                                            logger.warning(f"Could not move orphaned segment file {orphan_file.name}: {e}")

                            # Validate moved GIFs and ensure all are under size
                            valid_segments, invalid_segments = self._validate_segment_folder_gifs(final_segments_dir, max_size_mb)

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
                                        thumb_src = str(source_mp4_path)
                                        if not source_mp4_path.exists():
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
                                    sanitized_base = final_segments_dir.name.replace('_segments', '')
                                except Exception:
                                    sanitized_base = final_segments_dir.name
                                try:
                                    write_segments_summary(
                                        final_segments_dir,
                                        sanitized_base,
                                        logger=logger,
                                        analysis_tracker=self.analysis_tracker,
                                    )
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
                                # Create summary for invalid segments case so incomplete segments still have a summary
                                try:
                                    sanitized_base = final_segments_dir.name.replace('_segments', '')
                                except Exception:
                                    sanitized_base = final_segments_dir.name
                                try:
                                    write_segments_summary(
                                        final_segments_dir,
                                        sanitized_base,
                                        logger=logger,
                                        analysis_tracker=self.analysis_tracker,
                                    )
                                except Exception as e:
                                    logger.debug(f"Could not create comprehensive summary for invalid segments: {e}")
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
                        cache_source = source_mp4_path if source_mp4_path.exists() else final_gif_path
                        self._record_success_cache(cache_source, 'single_gif', final_gif_path)
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

            # Reuse GifGenerator to avoid repeated hardware detection overhead
            # GifGenerator is thread-safe as it uses subprocess calls to ffmpeg
            shared_generator = self.gif_generator
            
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

                    # Reuse shared generator (thread-safe via subprocess calls)
                    result = shared_generator.create_gif(
                        input_video=str(segment_file),
                        output_path=str(gif_path),
                        max_size_mb=max_size_mb,
                        disable_segmentation=True
                    )
                    
                    # If first attempt failed, retry with reduced duration
                    if not result.get('success', False):
                        error_msg = result.get('error', 'Unknown error')
                        logger.warning(f"Segment {segment_file.name} initial attempt failed: {error_msg}")
                        
                        # Check if failure was due to size limit - if so, try reducing duration
                        if 'exceeds limit' in str(error_msg).lower() or 'size' in str(error_msg).lower():
                            # Get segment duration and try with reduced duration
                            try:
                                segment_duration = FFmpegUtils.get_video_duration(str(segment_file))
                                # Reduce duration by 20% or 2 seconds, whichever is larger
                                reduced_duration = max(5.0, segment_duration * 0.8)
                                if reduced_duration < segment_duration:
                                    logger.info(f"Retrying segment {segment_file.name} with reduced duration: {reduced_duration:.2f}s (was {segment_duration:.2f}s)")
                                    
                                    # Clean up failed attempt if it exists
                                    if gif_path.exists():
                                        try:
                                            gif_path.unlink()
                                        except Exception:
                                            pass
                                    
                                    # Retry with reduced duration by using start_time and duration
                                    result = shared_generator.create_gif(
                                        input_video=str(segment_file),
                                        output_path=str(gif_path),
                                        max_size_mb=max_size_mb,
                                        start_time=0,
                                        duration=reduced_duration,
                                        disable_segmentation=True
                                    )
                                    
                                    if result.get('success', False):
                                        is_valid, validation_error = self.file_validator.is_valid_gif_with_enhanced_checks(
                                            str(gif_path), original_path=None, max_size_mb=max_size_mb
                                        )
                                        if is_valid:
                                            size_mb = result.get('size_mb', 0)
                                            print(f"    ✅ Segment GIF successful (retry with reduced duration): {size_mb:.2f}MB")
                                            logger.info(f"Segment {segment_file.name} succeeded on retry with reduced duration")
                                            return (True, None)
                                        else:
                                            logger.warning(f"Segment {segment_file.name} retry validation failed: {validation_error}")
                                    else:
                                        logger.warning(f"Segment {segment_file.name} retry with reduced duration also failed: {result.get('error', 'Unknown error')}")
                            except Exception as retry_e:
                                logger.warning(f"Exception during segment {segment_file.name} retry: {retry_e}")
                    
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
                try:
                    for fut in as_completed(future_map):
                        if self.shutdown_requested:
                            logger.info("Shutdown requested during segment processing, cancelling remaining tasks...")
                            # Cancel any remaining futures
                            for remaining_fut in future_map:
                                if not remaining_fut.done():
                                    remaining_fut.cancel()
                            break

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
                finally:
                    # Clean up any remaining temp files
                    try:
                        self._cleanup_temp_files()
                    except Exception:
                        pass
                # End of ThreadPoolExecutor context
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
                m = re.search(r"_segment_(\d+)\.gif$", f.name)
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
                cached_available, cached_valid, _meta, cached_error = self._check_cached_validation(
                    gf, 'gif', max_size_mb
                )
                if cached_available:
                    if not cached_valid and cached_error:
                        logger.debug(f"Segment GIF manifest cached failure for {gf.name}: {cached_error}")
                    return gf, cached_valid, cached_error or ""
                validation_start = time.perf_counter()
                probe_valid, metadata, probe_error = self.file_validator.probe_gif_metadata(
                    str(gf), max_size_mb=max_size_mb
                )
                if not probe_valid:
                    enhanced_valid, enhanced_error = self.file_validator.is_valid_gif_with_enhanced_checks(
                        str(gf), original_path=None, max_size_mb=max_size_mb
                    )
                    final_valid = bool(enhanced_valid)
                    error_msg = enhanced_error or probe_error
                else:
                    final_valid = True
                    error_msg = None
                self._update_validation_manifest(gf, 'gif', final_valid, metadata, error_msg, max_size_mb)
                validation_elapsed = time.perf_counter() - validation_start
                logger.debug(
                    f"Segment GIF validation for {gf.name} took {validation_elapsed:.3f}s "
                    f"(cached=False, result={final_valid})"
                )
                return gf, final_valid, error_msg or ""
            except Exception as e:
                logger.debug(f"Segment GIF validation error for {gf}: {e}")
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
                    # Identify which segments are missing
                    present_indices = set()
                    for f in gif_files:
                        m = re.search(r"_segment_(\d+)\.gif$", f.name)
                        if m:
                            try:
                                present_indices.add(int(m.group(1)))
                            except Exception:
                                pass
                    missing_indices = [i for i in range(1, expected + 1) if i not in present_indices]
                    missing_list = ", ".join([f"segment_{i:02d}" for i in missing_indices])
                    
                    print(f"    ⚠️  Segment completeness check: expected {expected}, found {present_count} (missing {missing}: {missing_list})")
                    logger.warning(
                        f"Segment completeness check failed for {segments_folder.name}: expected {expected}, found {present_count}, "
                        f"missing segments: {missing_list}"
                    )
                except Exception as e:
                    logger.warning(f"Error identifying missing segments: {e}")
                    print(f"    ⚠️  Segment completeness check: expected {expected}, found {present_count} (missing {expected - present_count})")
                    logger.warning(
                        f"Segment completeness check failed for {segments_folder.name}: expected {expected}, found {present_count}"
                    )
                # Consider folder invalid if incomplete; treat all as invalid to trigger regeneration
                try:
                    self.analysis_tracker.record_validation_failure('segment_gif_completeness', amount=len(gif_files) or 1)
                except Exception:
                    logger.debug("Analysis tracker segment completeness failure recording failed", exc_info=True)
                return [], gif_files

        if invalid_gifs:
            try:
                self.analysis_tracker.record_validation_failure('segment_gif_validation', amount=len(invalid_gifs))
            except Exception:
                logger.debug("Analysis tracker segment GIF validation failure recording failed", exc_info=True)
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

    @staticmethod
    def _is_within_directory(candidate: Path, root: Path) -> bool:
        """
        Return True if candidate path is inside root (inclusive), using normalized absolute paths.
        """
        try:
            candidate_path = os.path.normcase(os.path.normpath(str(candidate.resolve())))
            root_path = os.path.normcase(os.path.normpath(str(root.resolve())))
        except Exception:
            return False
        
        if candidate_path == root_path:
            return True
        
        if not root_path.endswith(os.sep):
            root_path = root_path + os.sep
        return candidate_path.startswith(root_path)

    def _resolve_mp4_source_path(self, mp4_file: Path, segments_folder: Optional[Path] = None) -> Path:
        """
        Return an existing MP4 path, falling back to copies that were moved into segments folders.

        Args:
            mp4_file: The original expected MP4 path.
            segments_folder: Optional explicit segments folder to search first.

        Returns:
            Path to an existing MP4 file if found; otherwise returns the original path.
        """
        try:
            if mp4_file and mp4_file.exists():
                return mp4_file
        except Exception:
            pass

        candidates: List[Path] = []
        if segments_folder and segments_folder.is_dir():
            candidates.append(segments_folder / mp4_file.name)
        try:
            inferred_segments = mp4_file.parent / f"{mp4_file.stem}_segments"
            candidates.append(inferred_segments / mp4_file.name)
        except Exception:
            pass

        seen: set = set()
        for candidate in candidates:
            if not candidate:
                continue
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            if candidate.exists():
                logger.info(
                    f"Resolved MP4 source via segments fallback: {candidate} (missing original: {mp4_file})"
                )
                return candidate

        return mp4_file

    def _ensure_mp4_in_segments(self, mp4_file: Path, segments_folder: Path):
        """Move the source MP4 into the segments folder if it resides in output; otherwise copy."""
        try:
            segments_folder.mkdir(exist_ok=True)
            target = segments_folder / mp4_file.name
            output_root = self.output_dir
            mp4_parent = mp4_file.parent
            mp4_within_output = self._is_within_directory(mp4_parent, output_root)
            
            def _remove_source_duplicate(reason: str) -> None:
                if not mp4_within_output:
                    return
                if not mp4_file.exists():
                    return
                try:
                    mp4_file.unlink()
                    logger.info(f"Removed duplicate MP4 from output ({reason}): {mp4_file.name}")
                except Exception as removal_error:
                    logger.warning(f"Could not remove duplicate MP4 ({reason}): {removal_error}")
            
            if target.exists():
                logger.debug(f"MP4 already exists in segments folder: {target.name}")
                _remove_source_duplicate("target_exists")
                return
            
            # If mp4 lives anywhere under the output directory, move it. Otherwise copy it (e.g. source still in input)
            try:
                should_move = mp4_file.exists() and mp4_within_output
                if should_move:
                    shutil.move(str(mp4_file), str(target))
                    logger.info(f"Moved MP4 to segments folder: {mp4_file.name} -> {target}")
                    try:
                        self.analysis_tracker.record_mp4_move('segments_move')
                    except Exception:
                        logger.debug("Analysis tracker MP4 move recording failed", exc_info=True)
                    
                    # Validate that MP4 was moved correctly
                    if not target.exists():
                        logger.error(f"MP4 move failed: {target.name} does not exist in segments folder")
                    elif mp4_file.exists():
                        logger.warning(f"MP4 still exists in parent directory after move: {mp4_file.name}, attempting cleanup")
                        try:
                            mp4_file.unlink()
                            logger.info(f"Removed leftover MP4 from parent directory: {mp4_file.name}")
                        except Exception as e:
                            logger.warning(f"Could not remove leftover MP4 from parent directory: {e}")
                    else:
                        logger.debug(f"MP4 successfully moved from parent to segments folder: {mp4_file.name}")
                else:
                    shutil.copy2(str(mp4_file), str(target))
                    logger.info(f"Copied MP4 to segments folder: {mp4_file.name} -> {target}")
                    try:
                        self.analysis_tracker.record_mp4_move('segments_copy')
                    except Exception:
                        logger.debug("Analysis tracker MP4 copy recording failed", exc_info=True)
                    
                    # Validate copy operation
                    if not target.exists():
                        logger.error(f"MP4 copy failed: {target.name} does not exist in segments folder")
                    _remove_source_duplicate("post_copy_cleanup")
            except Exception as e:
                logger.warning(f"Failed to move/copy MP4 to segments folder: {e}")
                # As a last resort, try copying
                try:
                    shutil.copy2(str(mp4_file), str(target))
                    logger.info(f"Fallback copy MP4 to segments folder: {mp4_file.name} -> {target}")
                    try:
                        self.analysis_tracker.record_mp4_move('segments_copy')
                    except Exception:
                        logger.debug("Analysis tracker fallback MP4 copy recording failed", exc_info=True)
                    
                    # Validate fallback copy
                    if not target.exists():
                        logger.error(f"Fallback MP4 copy failed: {target.name} does not exist in segments folder")
                    _remove_source_duplicate("fallback_copy_cleanup")
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
                    
                    # Move repaired file to output (preserve folder structure)
                    relative_path = self._get_relative_path(gif_file)
                    output_category_dir = self.output_dir / relative_path.parent
                    output_category_dir.mkdir(parents=True, exist_ok=True)
                    output_path = output_category_dir / gif_file.name
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
                
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    encoding='utf-8',
                    errors='replace',
                    timeout=60
                )
                
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
                        
                        # Move repaired file to output (preserve folder structure)
                        relative_path = self._get_relative_path(gif_file)
                        output_category_dir = self.output_dir / relative_path.parent
                        output_category_dir.mkdir(parents=True, exist_ok=True)
                        output_path = output_category_dir / gif_file.name
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
                        
            except subprocess.TimeoutExpired as repair_timeout:
                logger.warning(f"FFmpeg repair timed out for {gif_file.name}: {repair_timeout}")
                try:
                    self.analysis_tracker.record_timeout('gif_ffmpeg_repair')
                except Exception:
                    logger.debug("Analysis tracker FFmpeg repair timeout recording failed", exc_info=True)
                try:
                    if 'repaired_path' in locals() and repaired_path.exists():
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

            # Check if an output GIF already exists and validate it (preserve folder structure)
            relative_path = self._get_relative_path(gif_file)
            output_category_dir = self.output_dir / relative_path.parent
            output_category_dir.mkdir(parents=True, exist_ok=True)
            existing_output = output_category_dir / gif_file.name
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
            
            # Log initial file size
            initial_size = self.file_validator.get_file_size_mb(str(gif_file))
            logger.debug(f"[Workflow] Initial file size: {initial_size:.2f}MB, target: {max_size_mb:.2f}MB")
            logger.info(f"Starting GIF optimization: {gif_file.name} ({initial_size:.2f}MB -> target: {max_size_mb:.2f}MB)")
            
            # Create output path (preserve folder structure)
            relative_path = self._get_relative_path(gif_file)
            output_category_dir = self.output_dir / relative_path.parent
            output_category_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_category_dir / gif_file.name

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
                    working_size = self.file_validator.get_file_size_mb(str(working_path))
                    logger.debug(f"[Workflow] Working copy created: {working_size:.2f}MB")
                except Exception as copy_error:
                    logger.warning(f"Failed to copy corrupted GIF {gif_file.name}: {copy_error}")
                    # Try to read and rewrite the file to potentially fix corruption
                    try:
                        from PIL import Image
                        with Image.open(gif_file) as img:
                            # Save as new GIF to potentially fix corruption
                            img.save(working_path, 'GIF', save_all=True, loop=0)
                        working_size = self.file_validator.get_file_size_mb(str(working_path))
                        logger.info(f"Successfully repaired corrupted GIF {gif_file.name} using PIL, size: {working_size:.2f}MB")
                    except Exception as repair_error:
                        logger.error(f"Failed to repair corrupted GIF {gif_file.name}: {repair_error}")
                        return 'error'
                        
            except Exception as e:
                logger.error(f"Failed to prepare working copy for {gif_file.name}: {e}")
                return 'error'

            # Try to find a higher-quality source video for re-encoding/optimization
            source_video = self._find_source_video_for_gif(gif_file)

            # Use the GIF optimizer on the working copy
            from .gif_processing.gif_optimizer import GifOptimizer
            optimizer = GifOptimizer(self.config)

            logger.debug(f"[Workflow] Calling optimizer.optimize_gif() with target: {max_size_mb:.2f}MB, source_video: {source_video}")
            result = optimizer.optimize_gif(
                gif_path=str(working_path),
                max_size_mb=max_size_mb,
                source_video=source_video
            )
            logger.debug(f"[Workflow] Optimizer returned: {result}, working file exists: {working_path.exists()}")

            if result and working_path.exists():
                optimized_size = self.file_validator.get_file_size_mb(str(working_path))
                reduction_pct = ((initial_size - optimized_size) / initial_size * 100) if initial_size > 0 else 0
                logger.debug(f"[Workflow] Optimization successful: {optimized_size:.2f}MB (reduction: {reduction_pct:.1f}% from {initial_size:.2f}MB)")
                
                # Validate size before copying to output
                if optimized_size > max_size_mb:
                    print(f"    ❌ Optimized GIF still exceeds size limit: {optimized_size:.2f}MB > {max_size_mb:.2f}MB")
                    logger.warning(f"Optimized GIF exceeds size limit: {gif_file.name} ({optimized_size:.2f}MB > {max_size_mb:.2f}MB)")
                    logger.debug(f"[Workflow] Optimization failed reason: size exceeded (optimized: {optimized_size:.2f}MB > target: {max_size_mb:.2f}MB)")
                    return 'error'
                
                print(f"    ✅ GIF optimized successfully: {optimized_size:.2f}MB")
                logger.info(f"GIF optimized successfully (copy): {gif_file.name} -> {optimized_size:.2f}MB (reduced from {initial_size:.2f}MB by {reduction_pct:.1f}%)")

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
                
                # Log why optimization failed
                if not result:
                    logger.debug(f"[Workflow] Optimization failed: optimizer returned False")
                elif not working_path.exists():
                    logger.debug(f"[Workflow] Optimization failed: working file does not exist after optimization")
                else:
                    working_size = self.file_validator.get_file_size_mb(str(working_path))
                    logger.debug(f"[Workflow] Optimization failed: working file exists but size is {working_size:.2f}MB (target: {max_size_mb:.2f}MB)")
                
                # Check if working file exists and is valid (excluding size) before attempting repair
                # This helps distinguish between size failures and actual corruption
                needs_repair = False
                if working_path.exists():
                    working_size = self.file_validator.get_file_size_mb(str(working_path))
                    logger.debug(f"[Workflow] Working file exists: {working_size:.2f}MB, validating for corruption...")
                    # Validate working file without size check to see if it's actually corrupted
                    is_valid_no_size, error_msg_no_size = self.file_validator.is_valid_gif_with_enhanced_checks(
                        str(working_path),
                        original_path=str(gif_file),
                        max_size_mb=None  # Skip size validation
                    )
                    if not is_valid_no_size:
                        # File exists but is corrupted - needs repair
                        needs_repair = True
                        logger.info(f"Working file exists but is corrupted: {error_msg_no_size}")
                        logger.debug(f"[Workflow] Optimization failed reason: corruption detected ({error_msg_no_size})")
                    else:
                        # File is valid but just over size limit - don't repair
                        logger.info(f"Working file is valid but exceeds size limit ({working_size:.2f}MB > {max_size_mb:.2f}MB) - skipping repair")
                        logger.debug(f"[Workflow] Optimization failed reason: size limit exceeded (valid file but {working_size:.2f}MB > {max_size_mb:.2f}MB)")
                else:
                    # File doesn't exist - might be corruption issue
                    needs_repair = True
                    logger.info(f"Working file does not exist after optimization - attempting repair")
                    logger.debug(f"[Workflow] Optimization failed reason: working file missing after optimization")
                
                # Strategy 1: Try to repair the corrupted input file (only if corruption detected)
                if needs_repair:
                    try:
                        from PIL import Image
                        repaired_path = self.temp_dir / f"{gif_file.stem}.repaired.gif"
                        
                        # Get original file size for comparison
                        original_size = self.file_validator.get_file_size_mb(str(gif_file))
                        logger.info(f"Original file size: {original_size:.2f}MB")
                        
                        with Image.open(gif_file) as img:
                            # Save as new GIF to potentially fix corruption
                            img.save(repaired_path, 'GIF', save_all=True, loop=0)
                        
                        # Check if repaired file is larger than original - if so, skip it
                        repaired_size = self.file_validator.get_file_size_mb(str(repaired_path))
                        if repaired_size > original_size:
                            logger.warning(f"Repair increased file size ({repaired_size:.2f}MB > {original_size:.2f}MB) - skipping repair result")
                            try:
                                repaired_path.unlink()
                            except Exception:
                                pass
                        else:
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
                else:
                    logger.info(f"Skipping repair - optimization failed due to size limit, not corruption")
                
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
                
                # Strategy 3: Try segmentation as fallback if file is valid but over size limit
                # Only attempt segmentation if the file is valid (not corrupted) but exceeds size
                file_is_valid, file_error = self.file_validator.is_valid_gif_with_enhanced_checks(
                    str(gif_file),
                    original_path=str(gif_file),
                    max_size_mb=None  # Don't check size, just validate integrity
                )
                if file_is_valid:
                    file_size = self.file_validator.get_file_size_mb(str(gif_file))
                    if file_size > max_size_mb:
                        print(f"    ✂️  Attempting segmentation as fallback strategy...")
                        logger.info(f"Attempting segmentation fallback for {gif_file.name} (size: {file_size:.2f}MB > target: {max_size_mb:.2f}MB)")
                        try:
                            # Skip single-file optimization since we already tried it and it failed
                            seg_result = self._segment_input_gif(gif_file, max_size_mb, skip_single_optimization=True)
                            if seg_result:
                                print(f"    ✅ Segmentation fallback succeeded for {gif_file.name}")
                                logger.info(f"Segmentation fallback succeeded for {gif_file.name}")
                                return 'success'
                            else:
                                print(f"    ⚠️  Segmentation fallback failed for {gif_file.name}")
                                logger.warning(f"Segmentation fallback failed for {gif_file.name}")
                        except Exception as seg_error:
                            logger.warning(f"Segmentation fallback error for {gif_file.name}: {seg_error}")
                            print(f"    ⚠️  Segmentation fallback error: {seg_error}")
                
                # Strategy 4: All attempts failed
                print(f"    ❌ All optimization and repair strategies failed for {gif_file.name}")
                logger.error(f"All optimization and repair strategies failed for {gif_file.name}")
                
                # Log final summary of why optimization failed
                final_summary = f"[Workflow] Optimization failure summary: initial={initial_size:.2f}MB, target={max_size_mb:.2f}MB"
                if working_path.exists():
                    working_size = self.file_validator.get_file_size_mb(str(working_path))
                    final_summary += f", working_file={working_size:.2f}MB"
                else:
                    final_summary += ", working_file=missing"
                logger.debug(final_summary)
                logger.debug(f"[Workflow] All strategies attempted: optimizer returned {result}, needs_repair={needs_repair if 'needs_repair' in locals() else 'unknown'}")
                return 'error'
                
        except Exception as e:
            print(f"    ❌ Error optimizing GIF {gif_file.name}: {e}")
            logger.error(f"Error optimizing GIF {gif_file.name}: {e}")
            logger.exception(f"[Workflow] Exception during optimization: {e}")
            return 'error' 

    def _segment_input_gif(self, gif_file: Path, max_size_mb: float, preferred_segments: Optional[int] = None, skip_single_optimization: bool = False) -> bool:
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
            try:
                self.analysis_tracker.record_segmentation('gif_input')
                self.analysis_tracker.record_guardrail('gif_segmentation')
            except Exception:
                logger.debug("Analysis tracker GIF segmentation recording failed", exc_info=True)

            # Prefer single file before splitting: try optimizing whole GIF first
            # Skip if we're already in a segmentation fallback (optimization already tried and failed)
            if not skip_single_optimization:
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
            else:
                logger.debug("Skipping single-file optimization (already attempted in fallback path)")

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
            # Preserve folder structure from input
            relative_path = self._get_relative_path(gif_file)
            output_dir = self.output_dir / relative_path.parent
            output_dir.mkdir(parents=True, exist_ok=True)
            
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

            # Pre-calculate required scale factor for segments based on target size
            # Estimate expected segment size (original_size / num_segments, with some overhead)
            estimated_segment_size_mb = (original_size_mb / num_segments) * 1.2  # 20% overhead for safety
            scale_factor = 1.0
            if estimated_segment_size_mb > max_size_mb and max_size_mb > 0:
                # Calculate scale factor: area scales as width^2, so scale = sqrt(size_ratio)
                size_ratio = max_size_mb / estimated_segment_size_mb
                scale_factor = max(0.3, min(1.0, size_ratio ** 0.5))  # Clamp between 0.3 and 1.0
                logger.debug(f"Pre-calculated segment scale factor: {scale_factor:.3f} (estimated {estimated_segment_size_mb:.2f}MB -> target {max_size_mb:.2f}MB)")
            
            # Calculate target width based on scale factor
            try:
                video_info = FFmpegUtils.get_video_info(str(gif_file))
                if video_info and 'width' in video_info:
                    original_width = video_info['width']
                    base_width = int(self.config.get('gif_settings.width', 360) or 360)
                    # Apply scale factor to base width, but don't go below quality floors
                    quality_floors = self.config.get('gif_settings.quality_floors', {}) or {}
                    min_width = int(quality_floors.get('min_width_aggressive', 360) or 360)
                    target_segment_width = max(min_width, int(base_width * scale_factor))
                    # Ensure even
                    target_segment_width = (target_segment_width // 2) * 2
                    logger.info(f"Segment target width: {target_segment_width}px (from {base_width}px, scale={scale_factor:.3f})")
                else:
                    target_segment_width = None
            except Exception as e:
                logger.debug(f"Could not pre-calculate segment width: {e}")
                target_segment_width = None
            
            base_source_video = self._find_source_video_for_gif(gif_file)

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

            from .gif_processing.gif_optimizer import GifOptimizer
            optimizer = GifOptimizer(self.config, shutdown_checker=lambda: self.shutdown_requested)

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
                    
                    # Try to find source video for segment creation (preferred over GIF)
                    segment_source_video = base_source_video
                    if segment_source_video:
                        logger.debug(f"[Workflow] Using shared source video for segment creation: {segment_source_video}")
                    
                    # Use GifGenerator to create segments with proper compression if source video available
                    # Otherwise, create from GIF but with aggressive settings
                    if segment_source_video and os.path.exists(segment_source_video):
                        # Use GifGenerator to create segment from source video with proper compression
                        logger.debug(f"Creating segment {index+1}/{num_segments} from source video: start={seg_start:.2f}s, duration={seg_len:.2f}s")
                        try:
                            # Get settings for segment creation
                            from .gif_processing.gif_generator import GifGenerator
                            segment_generator = GifGenerator(self.config, shutdown_checker=lambda: self.shutdown_requested)
                            
                            # Create settings with aggressive compression for segments
                            # Use pre-calculated target width if available, otherwise use config default
                            segment_width = target_segment_width if target_segment_width is not None else int(self.config.get('gif_settings.width', 360) or 360)
                            segment_settings = {
                                'max_size_mb': max_size_mb,
                                'width': segment_width,
                                'height': int(self.config.get('gif_settings.height', -1) or -1),
                                'fps': target_fps,
                                'colors': int(self.config.get('gif_settings.colors', 256) or 256),
                                'dither': str(self.config.get('gif_settings.dither', 'bayer') or 'bayer'),
                                'lossy': int(self.config.get('gif_settings.lossy', 100) or 100),  # More aggressive for segments
                                'max_duration': seg_len
                            }
                            
                            # Create segment using GifGenerator
                            result = segment_generator._create_single_gif(
                                input_video=segment_source_video,
                                output_path=str(temp_seg),
                                settings=segment_settings,
                                start_time=seg_start,
                                duration=seg_len
                            )
                            
                            if not result.get('success', False) or not temp_seg.exists():
                                logger.error(f"Segment {index+1}/{num_segments} creation from source video failed")
                                return None
                            
                            # Check if segment is already under target
                            seg_size_mb = os.path.getsize(temp_seg) / (1024 * 1024)
                            if seg_size_mb > max_size_mb:
                                # Try to optimize it
                                logger.debug(f"Segment {index+1}/{num_segments} from source is {seg_size_mb:.2f}MB, optimizing...")
                                opt_ok = optimizer.optimize_gif(
                                    gif_path=str(temp_seg),
                                    max_size_mb=max_size_mb,
                                    source_video=segment_source_video  # Pass source for re-encoding
                                )
                                if not opt_ok:
                                    seg_size_mb = os.path.getsize(temp_seg) / (1024 * 1024) if temp_seg.exists() else max_size_mb + 1
                                    logger.error(f"Segment {index+1}/{num_segments} optimization failed: "
                                               f"file size {seg_size_mb:.2f}MB exceeds limit {max_size_mb:.2f}MB")
                                    try:
                                        temp_seg.unlink()
                                    except Exception:
                                        pass
                                    return None
                        except Exception as e:
                            logger.warning(f"Failed to create segment from source video, falling back to GIF: {e}")
                            segment_source_video = None  # Fall back to GIF method
                    
                    # Fallback: create segment from GIF file (less efficient but works)
                    if not segment_source_video or not temp_seg.exists():
                        logger.debug(f"Creating segment {index+1}/{num_segments} from GIF: start={seg_start:.2f}s, duration={seg_len:.2f}s")
                        # Downscale segments to pre-calculated target width or configured max width
                        try:
                            # Use pre-calculated target width if available, otherwise use config default
                            max_width = target_segment_width if target_segment_width is not None else int(self.config.get('gif_settings.width', 360) or 360)
                            # Get height setting - if -1, we'll calculate it to preserve aspect ratio
                            height_setting = self.config.get('gif_settings.height', -1)
                        except Exception:
                            max_width = target_segment_width if target_segment_width is not None else 360
                            height_setting = -1
                        
                        # Get video info to calculate proper scaling that preserves aspect ratio
                        try:
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
                            # Detect timeout by checking stderr for 'TimeoutExpired' string
                            # (the _run_subprocess_with_shutdown_check returns TimeoutResult with stderr='TimeoutExpired')
                            error_type = "timeout" if (err_excerpt and 'TimeoutExpired' in str(err_excerpt)) else "ffmpeg_error"
                            if error_type == "timeout":
                                try:
                                    self.analysis_tracker.record_timeout('gif_segment_ffmpeg')
                                except Exception:
                                    logger.debug("Analysis tracker timeout recording failed", exc_info=True)
                            logger.error(f"Segment {index+1}/{num_segments} creation failed ({error_type}): FFmpeg returncode={result.returncode}, "
                                       f"file_exists={temp_seg.exists()}")
                            if err_excerpt:
                                logger.error(f"Segment {index+1} FFmpeg stderr: {err_excerpt[:500]}")
                            if hasattr(result, 'stdout') and result.stdout:
                                logger.debug(f"Segment {index+1} FFmpeg stdout: {result.stdout[:200]}")
                            return None
                        
                        # Try to find source video for segment re-encoding
                        try:
                            # For segments, the source is the original GIF file (which may have come from a video)
                            # But ideally we'd have the original video - check if it exists
                            potential_video = gif_file.parent / f"{base_name}.mp4"
                            if not potential_video.exists():
                                potential_video = output_dir / f"{base_name}.mp4"
                            if potential_video.exists():
                                segment_source_video = str(potential_video)
                                logger.debug(f"[Workflow] Found source video for segment re-encoding: {segment_source_video}")
                        except Exception:
                            pass
                        
                        # Optimize with quality target for stronger convergence
                        logger.debug(f"Optimizing segment {index+1}/{num_segments}")
                        opt_ok = optimizer.optimize_gif(
                            gif_path=str(temp_seg),
                            max_size_mb=max_size_mb,
                            source_video=segment_source_video
                        )
                        if not opt_ok:
                            # Check if file exists and get its size to understand why optimization failed
                            if temp_seg.exists():
                                try:
                                    seg_size_mb = temp_seg.stat().st_size / (1024 * 1024)
                                    logger.error(f"Segment {index+1}/{num_segments} optimization failed: "
                                               f"file size {seg_size_mb:.2f}MB exceeds limit {max_size_mb:.2f}MB")
                                except Exception:
                                    logger.error(f"Segment {index+1}/{num_segments} optimization failed: unable to get file size")
                            else:
                                logger.error(f"Segment {index+1}/{num_segments} optimization failed: file does not exist")
                            try:
                                temp_seg.unlink()
                            except Exception:
                                pass
                            return None
                    # Validate and move
                    final_seg = (segments_dir / f"{base_name}_segment_{index+1:02d}.gif") if use_segments_folder else single_output_path
                    logger.debug(f"Validating segment {index+1}/{num_segments} before final move")
                    is_valid, validation_error = self.file_validator.is_valid_gif_with_enhanced_checks(str(temp_seg), original_path=None, max_size_mb=max_size_mb)
                    if not is_valid:
                        logger.error(f"Segment {index+1}/{num_segments} validation failed: {validation_error or 'unknown validation error'}")
                        # Log file details for debugging
                        if temp_seg.exists():
                            try:
                                seg_size_mb = temp_seg.stat().st_size / (1024 * 1024)
                                logger.debug(f"Segment {index+1} file size: {seg_size_mb:.2f}MB, limit: {max_size_mb:.2f}MB")
                            except Exception:
                                pass
                        try:
                            temp_seg.unlink()
                        except Exception:
                            pass
                        return None
                    try:
                        if not use_segments_folder and final_seg.exists():
                            final_seg.unlink()
                        shutil.move(str(temp_seg), str(final_seg))
                        
                        # Validate that file was moved correctly
                        if not final_seg.exists():
                            logger.error(f"File move failed: {final_seg} does not exist after move from {temp_seg}")
                            return None
                        
                        # Verify temp file is gone (unless it's a copy operation)
                        if temp_seg.exists():
                            logger.warning(f"Temp file still exists after move: {temp_seg}. This may indicate a copy operation.")
                            # Try to remove the temp file if it still exists
                            try:
                                temp_seg.unlink()
                                logger.debug(f"Removed leftover temp file: {temp_seg}")
                            except Exception:
                                pass
                        
                    except Exception as e:
                        logger.warning(f"Error moving segment file from {temp_seg} to {final_seg}: {e}")
                        try:
                            shutil.copy2(str(temp_seg), str(final_seg))
                            temp_seg.unlink(missing_ok=True)  # type: ignore[arg-type]
                            
                            # Validate copy operation
                            if not final_seg.exists():
                                logger.error(f"File copy failed: {final_seg} does not exist after copy from {temp_seg}")
                                return None
                        except Exception as copy_err:
                            logger.error(f"Both move and copy failed for segment {index+1}: {copy_err}")
                            return None
                    
                    # Final validation: ensure file exists in expected location
                    if not final_seg.exists():
                        logger.error(f"Segment file validation failed: {final_seg} does not exist")
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
                                    src_input = segments_dir.parent / f"{base_name}.mp4"
                                    cache_input = src_input if src_input.exists() else segments_dir
                                    self._record_success_cache(cache_input, 'single_gif', single_output_path)
                                except Exception:
                                    pass
                                return True
                        except Exception:
                            # If relocation fails, fall back to reporting segments folder
                            pass
                    print(f"    📂 Segmented GIFs saved to: {segments_dir.name} ({successful} segment(s), {total_size_mb:.2f}MB total)")
                    logger.info(f"GIF segmentation complete: {successful} segments at {segments_dir}")
                    
                    # Create comprehensive summary for the segments folder
                    try:
                        write_segments_summary(
                            segments_dir,
                            base_name,
                            logger=logger,
                            analysis_tracker=self.analysis_tracker,
                        )
                        logger.debug(f"Created comprehensive summary for segments folder: {segments_dir.name}")
                    except Exception as e:
                        logger.warning(f"Could not create comprehensive summary for segments folder: {e}")
                    
                    # Record cache for segments-based success (source = base MP4 if present)
                    try:
                        src_input = segments_dir.parent / f"{base_name}.mp4"
                        cache_input = src_input if src_input.exists() else segments_dir
                        self._record_success_cache(cache_input, 'segments', segments_dir)
                    except Exception:
                        pass
                else:
                    # Single segment result saved directly as a single GIF
                    print(f"    📁 Saved optimized GIF to output: {single_output_path.name} ({total_size_mb:.2f}MB)")
                    logger.info(f"Single-segment GIF saved to output: {single_output_path} ({total_size_mb:.2f}MB)")
                    # Record cache for single GIF success (source = base MP4 if present)
                    try:
                        src_input = single_output_path.parent / f"{base_name}.mp4"
                        cache_input = src_input if src_input.exists() else single_output_path.parent
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
            
            # Respect existing optimized outputs: if a valid compressed GIF exists in output, skip (preserve folder structure)
            relative_path = self._get_relative_path(gif_file)
            output_category_dir = self.output_dir / relative_path.parent
            output_category_dir.mkdir(parents=True, exist_ok=True)
            existing_output = output_category_dir / gif_file.name
            if existing_output.exists():
                cached_available, cached_valid, _cached_meta, cached_error = self._check_cached_validation(
                    existing_output, 'gif', max_size_mb
                )
                if cached_available and cached_valid:
                    print(f"    ♻️  Valid optimized GIF already exists in output: {existing_output.name}")
                    logger.info(f"Skipping optimization; existing optimized GIF present: {existing_output}")
                    # Cache success for gif input
                    self._record_success_cache(gif_file, 'gif_input', existing_output)
                    return 'success'
                if not cached_available:
                    validation_start = time.perf_counter()
                    probe_valid, metadata, probe_error = self.file_validator.probe_gif_metadata(
                        str(existing_output), max_size_mb=max_size_mb
                    )
                    if not probe_valid:
                        enhanced_valid, enhanced_error = self.file_validator.is_valid_gif_with_enhanced_checks(
                            str(existing_output), original_path=None, max_size_mb=max_size_mb
                        )
                        final_valid = bool(enhanced_valid)
                        validation_error = enhanced_error or probe_error
                    else:
                        final_valid = True
                        validation_error = None
                    self._update_validation_manifest(
                        existing_output, 'gif', final_valid, metadata, validation_error, max_size_mb
                    )
                    validation_elapsed = time.perf_counter() - validation_start
                    logger.debug(
                        f"Existing optimized GIF validation took {validation_elapsed:.3f}s "
                        f"(cached=False, result={final_valid})"
                    )
                    if final_valid:
                        print(f"    ♻️  Valid optimized GIF already exists in output: {existing_output.name}")
                        logger.info(f"Skipping optimization; existing optimized GIF present: {existing_output}")
                        self._record_success_cache(gif_file, 'gif_input', existing_output)
                        return 'success'
                else:
                    logger.debug(
                        f"Manifest indicates {existing_output.name} is invalid ({cached_error}); re-optimizing GIF input."
                    )
            
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

    def _print_cache_stats(self, tracker_counts: Optional[Dict[str, int]] = None) -> None:
        """Print cache statistics for monitoring."""
        if not self.use_cache:
            return
        
        stats = self.get_cache_stats()
        cache_section_printed = False
        if stats['total_entries'] > 0:
            print(f"\n📊 Cache Stats: {stats['current_session_entries']} current, {stats['old_entries']} old entries")
            print(f"   Age window: {stats.get('cache_age_info', 'unknown')}")
            logger.info(
                f"Cache stats: {stats['current_session_entries']} current, "
                f"{stats['old_entries']} old entries ({stats.get('cache_age_info', 'n/a')})"
            )
            cache_section_printed = True

        if tracker_counts:
            hits = tracker_counts.get('cache_hits', 0)
            misses = tracker_counts.get('cache_misses', 0)
            total_checks = hits + misses
            if total_checks > 0:
                if not cache_section_printed:
                    print("\n📊 Cache Stats:")
                    cache_section_printed = True
                hit_rate = (hits / total_checks) * 100 if total_checks else 0.0
                print(f"   Hit rate: {hit_rate:.1f}% ({hits} / {total_checks} checks)")
                logger.info(f"Cache hit rate: {hit_rate:.1f}% ({hits}/{total_checks})")

    def _print_workflow_summary(self, processing_stats: Dict[str, int], start_time: float) -> None:
        """Print enhanced workflow summary with error analysis and cache information."""
        elapsed = time.time() - start_time
        total_processed = int(processing_stats.get('processed') or 0)
        successful = int(processing_stats.get('successful') or 0)
        failed = int(processing_stats.get('errors') or 0)
        skipped = int(processing_stats.get('skipped') or 0)
        success_rate = (successful / total_processed * 100.0) if total_processed else 0.0

        tracker_snapshot: Dict[str, Any] = {}
        tracker_counts: Dict[str, int] = {}
        try:
            tracker_snapshot = self.analysis_tracker.snapshot()
            tracker_counts = tracker_snapshot.get('counts', {})
        except Exception:
            tracker_snapshot = {}
            tracker_counts = {}

        error_summary = self.error_handler.get_error_summary()

        print(f"\n{'='*60}")
        print(f"🎬 Workflow Complete!")
        print(f"⏱️  Total time: {elapsed:.1f} seconds")
        print(f"📁 Files processed: {total_processed}")
        print(f"✅ Successful: {successful}")
        print(f"❌ Failed: {failed}")
        print(f"⚡ Skipped (cached): {skipped}")
        
        # Enhanced error analysis if there were failures
        if failed > 0:
            print(f"\n📊 Error Analysis:")
            categories = error_summary.get('categories', {})
            if categories:
                print(f"   Error breakdown:")
                for category, count in categories.items():
                    percentage = (count / failed * 100.0) if failed else 0.0
                    print(f"   • {category}: {count} files ({percentage:.1f}%)")
            severity_distribution = error_summary.get('severity_distribution', {})
            if severity_distribution:
                print(f"   Severity levels:")
                for severity, count in severity_distribution.items():
                    percentage = (count / failed * 100.0) if failed else 0.0
                    print(f"   • {severity}: {count} ({percentage:.1f}%)")
            retryable_errors = error_summary.get('retryable_errors')
            non_retryable_errors = error_summary.get('non_retryable_errors')
            if retryable_errors is not None and non_retryable_errors is not None:
                print(f"   Retryable vs non-retryable: {retryable_errors} / {non_retryable_errors}")
            top_failures = self.error_handler.get_top_failures()
            if top_failures:
                print(f"   Top failure signals:")
                for entry in top_failures:
                    sample = entry.get('sample_message') or ''
                    if len(sample) > 80:
                        sample = sample[:77] + "..."
                    print(f"   • {entry.get('category')}: {entry.get('count')} (e.g., {sample})")
            most_common = error_summary.get('most_common_category')
            if most_common:
                print(f"   Top category: {most_common}")
            print(f"   🛡️  Batch resilience: {success_rate:.1f}% success rate maintained")
            # Log comprehensive error analysis
            self.error_handler.log_batch_summary(total_processed, successful)

        # Diagnostics from tracker
        diagnostic_lines = []
        metric_labels = [
            ('retries', "🔁 Retries"),
            ('segmentation_events', "✂️  Segment guardrails"),
            ('guardrail_events', "🛡️ Guardrail activations"),
            ('timeout_events', "⏱️ Timeouts"),
            ('validation_failures', "🧪 Validation failures"),
            ('summary_cleanups', "🧹 Summary cleanups"),
        ]
        for metric, label in metric_labels:
            value = tracker_counts.get(metric, 0)
            if value:
                diagnostic_lines.append(f"{label}: {value}")
        if diagnostic_lines:
            print(f"\n📈 Diagnostics:")
            for line in diagnostic_lines:
                print(f"   {line}")
        recent_events = tracker_snapshot.get('recent_events') or []
        if recent_events:
            highlighted = ", ".join(recent_events[-5:])
            print(f"   Recent events: {highlighted}")
        
        # Print cache statistics with hit/miss data
        self._print_cache_stats(tracker_counts)
        
        print(f"{'='*60}")
        
        # Enhanced logging with error context
        if failed > 0:
            logger.info(
                f"Workflow completed with errors in {elapsed:.1f}s. "
                f"Processed: {total_processed}, Success: {successful}, "
                f"Failed: {failed}, Skipped: {skipped}"
            )
            logger.info(f"Batch processing resilience maintained: {success_rate:.1f}% success rate")
        else:
            logger.info(
                f"Workflow completed successfully in {elapsed:.1f}s. "
                f"Processed: {total_processed}, Success: {successful}, "
                f"Skipped: {skipped}"
            )
        
        # Reset error handler for next session
        self.error_handler.reset()

    def cleanup_segments_summary_files(self) -> None:
        """Best-effort cleanup to enforce a single summary per segments folder."""
        try:
            self._deduplicate_all_segments_summaries()
        except Exception as e:
            logger.debug(f"Could not deduplicate segment summaries: {e}")

    def _canonical_segments_summary_path(self, segments_folder: Path, base_name: Optional[str] = None) -> Path:
        """
        Derive the canonical summary filename for a segments folder using the shared sanitizer.
        """
        return segments_summary_path(segments_folder, base_name)

    def _deduplicate_segments_summaries(self, segments_folder: Path, base_name: Optional[str] = None) -> Optional[Path]:
        """
        Ensure only a single canonical summary file exists inside the provided segments folder.
        """
        try:
            if not segments_folder.exists() or not segments_folder.is_dir():
                return None

            canonical_path = self._canonical_segments_summary_path(segments_folder, base_name)
            performed_cleanup = False
            summary_candidates = [
                path for path in segments_folder.glob("~*_comprehensive_summary.txt") if path.is_file()
            ]

            if not summary_candidates:
                return canonical_path

            def _clear_attributes(path: Path) -> None:
                try:
                    subprocess.run(
                        ['attrib', '-R', '-S', '-H', str(path)],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        check=False
                    )
                except Exception:
                    pass

            def _remove_path(path: Path) -> None:
                try:
                    _clear_attributes(path)
                    path.unlink()
                except Exception as remove_err:
                    logger.debug(f"Could not remove duplicate summary {path}: {remove_err}")

            if canonical_path.exists():
                for candidate in summary_candidates:
                    if candidate != canonical_path:
                        _remove_path(candidate)
                        performed_cleanup = True
                if performed_cleanup:
                    try:
                        self.analysis_tracker.record_summary_cleanup('segments_folder')
                    except Exception:
                        logger.debug("Analysis tracker summary cleanup recording failed", exc_info=True)
                return canonical_path

            # Canonical file missing - promote the newest summary file to canonical
            newest_candidate = max(summary_candidates, key=lambda p: p.stat().st_mtime if p.exists() else 0)
            if newest_candidate != canonical_path:
                try:
                    _clear_attributes(newest_candidate)
                    newest_candidate.rename(canonical_path)
                    performed_cleanup = True
                except Exception as rename_err:
                    logger.debug(f"Could not rename {newest_candidate} to canonical summary: {rename_err}")
                    try:
                        shutil.copy2(str(newest_candidate), str(canonical_path))
                        performed_cleanup = True
                    except Exception as copy_err:
                        logger.debug(f"Could not copy {newest_candidate} to canonical summary: {copy_err}")

            for candidate in summary_candidates:
                if candidate != canonical_path and candidate.exists():
                    _remove_path(candidate)
                    performed_cleanup = True

            if performed_cleanup:
                try:
                    self.analysis_tracker.record_summary_cleanup('segments_folder')
                except Exception:
                    logger.debug("Analysis tracker summary cleanup recording failed", exc_info=True)
            return canonical_path
        except Exception as e:
            logger.debug(f"Failed to deduplicate summaries in {segments_folder}: {e}")
            return None

    def _deduplicate_all_segments_summaries(self) -> None:
        """Scan all output segment folders and enforce single-summary rule."""
        try:
            segments_folders: List[Path] = []
            for subdir in self.output_dir.rglob('*'):
                try:
                    if subdir.is_dir() and subdir.name.endswith('_segments'):
                        segments_folders.append(subdir)
                except Exception:
                    continue

            for segments_folder in segments_folders:
                try:
                    # First deduplicate summaries
                    self._deduplicate_segments_summaries(segments_folder)
                    # Then check if folder is empty and remove it if so
                    self._remove_empty_segments_folder(segments_folder)
                except Exception as folder_err:
                    logger.debug(f"Could not process {segments_folder}: {folder_err}")
        except Exception as e:
            logger.debug(f"Could not enumerate segment folders for deduplication: {e}")

    def _remove_empty_segments_folder(self, segments_folder: Path) -> bool:
        """
        Remove a segments folder if it contains no media files.
        
        Args:
            segments_folder: Path to the segments folder to check and potentially remove
            
        Returns:
            True if folder was removed, False otherwise
        """
        try:
            # Check if folder exists and is a directory
            if not segments_folder.exists() or not segments_folder.is_dir():
                return False
            
            # Check for segment files (MP4 or GIF) - exclude summary files and other metadata
            segment_files = []
            try:
                for item in segments_folder.iterdir():
                    if item.is_file():
                        # Check if it's a segment file (MP4 or GIF), not a summary or metadata file
                        if item.suffix.lower() in ['.mp4', '.gif']:
                            # Exclude summary files and other metadata
                            if not item.name.startswith('~') and not item.name.startswith('folder.'):
                                segment_files.append(item)
            except Exception as e:
                logger.debug(f"Error checking for segment files in {segments_folder}: {e}")
                return False
            
            # If media files found, don't remove
            if segment_files:
                return False
            
            # No media files found - remove the folder
            # First, clear Windows file attributes on folder contents (best-effort)
            try:
                for item in segments_folder.iterdir():
                    try:
                        subprocess.run(
                            ['attrib', '-R', '-S', '-H', str(item)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            check=False
                        )
                    except Exception:
                        pass
            except Exception as e:
                logger.debug(f"Could not clear attributes in {segments_folder}: {e}")
            
            # Remove the folder
            try:
                shutil.rmtree(segments_folder)
                logger.info(f"Removed empty segments folder: {segments_folder.name}")
                try:
                    self.analysis_tracker.record_summary_cleanup('empty_segments_folder')
                except Exception:
                    logger.debug("Analysis tracker empty segments cleanup recording failed", exc_info=True)
                return True
            except Exception as remove_err:
                logger.debug(f"Could not remove empty segments folder {segments_folder}: {remove_err}")
                return False
                
        except Exception as e:
            logger.debug(f"Failed to check/remove empty segments folder {segments_folder}: {e}")
            return False

    def _ensure_segments_summary_exists(self, segments_folder: Path) -> None:
        """Ensure a comprehensive summary exists for the segments folder"""
        try:
            # Check if segments folder contains any actual segment files before creating summary
            if not segments_folder.exists() or not segments_folder.is_dir():
                logger.debug(f"Segments folder does not exist or is not a directory: {segments_folder}")
                return
            
            # Check for segment files (MP4 or GIF) - exclude summary files and other metadata
            segment_files = []
            try:
                for item in segments_folder.iterdir():
                    if item.is_file():
                        # Check if it's a segment file (MP4 or GIF), not a summary or metadata file
                        if item.suffix.lower() in ['.mp4', '.gif']:
                            # Exclude summary files and other metadata
                            if not item.name.startswith('~') and not item.name.startswith('folder.'):
                                segment_files.append(item)
            except Exception as e:
                logger.warning(f"Error checking for segment files in {segments_folder}: {e}")
                return
            
            # Only create summary if there are actual segment files
            if not segment_files:
                # Remove empty segments folder
                if not self._remove_empty_segments_folder(segments_folder):
                    logger.warning(f"Segments folder is empty (no segment files found): {segments_folder.name}, skipping summary creation")
                return
            
            # Prefer sanitized base name (without '_segments')
            base_name = segments_folder.stem.replace('_segments', '')

            canonical_summary = self._canonical_segments_summary_path(segments_folder, base_name)
            self._deduplicate_segments_summaries(segments_folder, base_name)

            if canonical_summary.exists():
                logger.debug(f"Summary already exists: {canonical_summary.name}")
                return

            # Create sanitized summary if none exists
            write_segments_summary(
                segments_folder,
                base_name,
                logger=logger,
                analysis_tracker=self.analysis_tracker,
            )
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

