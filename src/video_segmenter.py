"""
Video Segmentation Module
Handles segmentation of large video files into smaller, manageable segments
Similar to GIF segmentation but optimized for video files
"""

import os
import subprocess
import tempfile
import shutil
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
from tqdm import tqdm
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from .config_manager import ConfigManager
from .hardware_detector import HardwareDetector
from .ffmpeg_utils import FFmpegUtils

logger = logging.getLogger(__name__)

class VideoSegmenter:
    def __init__(self, config_manager: ConfigManager, hardware_detector: HardwareDetector):
        self.config = config_manager
        self.hardware = hardware_detector
        self.temp_dir = self.config.get_temp_dir()
        
        # Shutdown handling
        self.shutdown_requested = False
        self.current_ffmpeg_process = None
        self._proc_lock = threading.Lock()
        self._active_processes: set[subprocess.Popen] = set()
        self._print_lock = threading.Lock()
        # Track temporary compressors used during optimization so we can terminate them on shutdown
        self._active_temp_lock = threading.Lock()
        self._active_temp_compressors: set = set()
        
        # Multiprocessing configuration - dynamic analysis with fallback
        self.max_workers = self._calculate_optimal_workers()
        
        # Global FFmpeg process limiter to prevent GPU/CPU saturation across segment creation/optimization
        try:
            encoder, accel_type = self.hardware.get_best_encoder("h264")
        except Exception:
            encoder, accel_type = ("libx264", "software")
        # Allow config override; default to 1 for hardware encoders, 2 for software
        default_slots = 1 if accel_type != 'software' else 2
        slots = int(self.config.get('video_compression.segmentation.max_concurrent_ffmpeg_processes', default_slots) or default_slots)
        slots = max(1, slots)
        self._ffmpeg_semaphore = threading.Semaphore(slots)
        
        # Optimization stage concurrency (applies to post-creation optimizations)
        self._optimization_workers = int(self.config.get('video_compression.segmentation.max_concurrent_optimizations', 1) or 1)
        
        logger.info(f"Video segmenter initialized with {self.max_workers} concurrent workers")

    def _threadsafe_print(self, message: str) -> None:
        """Print from multiple worker threads without interleaving lines."""
        try:
            with self._print_lock:
                print(message, flush=True)
        except Exception:
            try:
                print(message)
            except Exception:
                pass
    
    def _calculate_optimal_workers(self) -> int:
        """
        Calculate optimal number of workers based on configuration and system analysis
        """
        # Prefer video segmentation concurrency settings when provided
        if self.config.get('video_compression.segmentation.max_concurrent_segments') is not None:
            try:
                value = int(self.config.get('video_compression.segmentation.max_concurrent_segments'))
                return max(1, value)
            except Exception:
                pass
        
        # Backward compatibility with GIF settings if no video-specific key is present
        if not self.config.get('gif_settings.multiprocessing.enabled', True):
            return 1  # Disable multiprocessing if configured
        
        # Check if we should use dynamic analysis
        use_dynamic = self.config.get('gif_settings.multiprocessing.use_dynamic_analysis', True)
        
        if use_dynamic:
            try:
                # Use hardware detector to analyze optimal workers
                worker_analysis = self.hardware.analyze_optimal_segmentation_workers()
                
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
        
    def should_segment_video(self, duration: float, video_info: Dict[str, Any], max_size_mb: float) -> bool:
        """
        Determine if a video should be split into segments based on estimated file size
        Video segmentation is now a LAST RESORT - only when single file compression cannot achieve decent quality

        Args:
            duration: Video duration in seconds
            video_info: Video metadata including complexity, resolution, etc.
            max_size_mb: Target size limit

        Returns:
            True if video should be split into segments (last resort only)
        """

        # Get segmentation settings from config
        segmentation_config = self.config.get('video_compression.segmentation', {})

        # Log original vs target size, but do not force segmentation solely on original size
        original_size_mb = video_info.get('size_bytes', 0) / (1024 * 1024)
        logger.info(f"Video segmentation check: original size {original_size_mb:.1f}MB, target {max_size_mb}MB, "
                   f"duration {duration:.1f}s, resolution {video_info.get('width', 0)}x{video_info.get('height', 0)}")

        # Create optimal video parameters for size estimation
        estimation_params = {
            'width': min(1280, video_info.get('width', 1280)),
            'height': min(720, video_info.get('height', 720)),
            'fps': min(30, video_info.get('fps', 30)),
            'bitrate': '1000k',  # Conservative bitrate for estimation
            'crf': 28  # Conservative quality for estimation
        }

        # Preserve aspect ratio for estimation
        if video_info.get('width') and video_info.get('height'):
            original_width = video_info['width']
            original_height = video_info['height']
            original_aspect_ratio = original_width / original_height

            max_width = estimation_params['width']
            max_height = estimation_params['height']

            if original_aspect_ratio > 1:  # Landscape
                new_width = min(original_width, max_width)
                new_height = int(new_width / original_aspect_ratio)
                if new_height > max_height:
                    new_height = max_height
                    new_width = int(new_height * original_aspect_ratio)
            else:  # Portrait or square
                new_height = min(original_height, max_height)
                new_width = int(new_height * original_aspect_ratio)
                if new_width > max_width:
                    new_width = max_width
                    new_height = int(new_width / original_aspect_ratio)

            # Ensure even dimensions
            estimation_params['width'] = new_width - (new_width % 2)
            estimation_params['height'] = new_height - (new_height % 2)

        logger.info(f"Estimation parameters: {estimation_params['width']}x{estimation_params['height']}, "
                   f"{estimation_params['fps']}fps, {estimation_params['bitrate']}, CRF {estimation_params['crf']}")

        # Estimate file size with optimal compression
        estimated_size_mb = self._estimate_video_size(estimation_params, duration, video_info)

        # Try with more aggressive compression first to see if single video is feasible
        aggressive_params = estimation_params.copy()
        aggressive_params.update({
            'fps': 24,
            'bitrate': '800k',
            'crf': 32
        })

        aggressive_size_mb = self._estimate_video_size(aggressive_params, duration, video_info)
        logger.info(f"Aggressive estimation: {aggressive_size_mb:.1f}MB (target: {max_size_mb * 1.5:.1f}MB)")

        # Try with very aggressive compression as last resort
        very_aggressive_params = estimation_params.copy()
        very_aggressive_params.update({
            'fps': 20,
            'bitrate': '600k',
            'crf': 35
        })

        very_aggressive_size_mb = self._estimate_video_size(very_aggressive_params, duration, video_info)
        logger.info(f"Very aggressive estimation: {very_aggressive_size_mb:.1f}MB (target: {max_size_mb * 1.8:.1f}MB)")

        # Only segment if even very aggressive compression cannot achieve target
        if very_aggressive_size_mb > max_size_mb * 1.8:
            logger.info(f"Video segmentation recommended: even very aggressive compression estimates {very_aggressive_size_mb:.1f}MB > "
                       f"{max_size_mb * 1.8:.1f}MB")
            return True

        # Primary decision: split if estimated size exceeds limit significantly (increased threshold)
        size_threshold_multiplier = segmentation_config.get('size_threshold_multiplier', 3.0)  # Increased from 2.5
        threshold_size = max_size_mb * size_threshold_multiplier
        logger.info(f"Estimated size: {estimated_size_mb:.1f}MB, threshold: {threshold_size:.1f}MB "
                   f"(multiplier: {size_threshold_multiplier})")

        if estimated_size_mb > threshold_size:
            logger.info(f"Video segmentation recommended: estimated size {estimated_size_mb:.1f}MB > "
                       f"{threshold_size:.1f}MB threshold")
            return True

        # Secondary decision: split if estimated size is close to limit but video has challenging characteristics
        if estimated_size_mb > max_size_mb * 2.2:  # Increased from 1.8
            complexity = video_info.get('complexity_score', 5.0)
            motion_level = video_info.get('motion_level', 'medium')

            # Split only if very high complexity or motion makes compression unpredictable
            if complexity >= 9.0 or motion_level == 'very_high':  # Increased complexity threshold
                logger.info(f"Video segmentation recommended: estimated size {estimated_size_mb:.1f}MB with "
                           f"challenging characteristics (complexity: {complexity:.1f}, motion: {motion_level})")
                return True

        # Fallback: still split extremely long videos regardless of estimated size (reduced limit for longer segments)
        fallback_duration_limit = segmentation_config.get('fallback_duration_limit', 300)  # Reduced from 600 to 300 for longer segments
        if duration > fallback_duration_limit:
            logger.info(f"Video segmentation recommended: duration {duration}s exceeds fallback limit {fallback_duration_limit}s")
            return True

        logger.info(f"Single video recommended: estimated size {estimated_size_mb:.1f}MB (aggressive: {aggressive_size_mb:.1f}MB, "
                   f"very aggressive: {very_aggressive_size_mb:.1f}MB) within acceptable range for {max_size_mb}MB target")
        return False
    
    def segment_video(self, input_video: str, output_base_path: str, platform: str = None,
                     max_size_mb: float = None, start_time: float = 0, 
                     duration: float = None) -> Dict[str, Any]:
        """
        Split a large video into multiple high-quality segments
        
        Args:
            input_video: Path to input video
            output_base_path: Base path for output (will create folder)
            platform: Target platform for compression settings
            max_size_mb: Maximum file size per segment
            start_time: Start time in video
            duration: Duration to process
            
        Returns:
            Dictionary with results including segment paths and metadata
        """
        
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Input video file not found: {input_video}")
        
        logger.info(f"Starting video segmentation: {input_video} -> {output_base_path}")
        
        # Get platform configuration
        platform_config = {}
        if platform:
            platform_config = self.config.get_platform_config(platform, 'video_compression')
            logger.info(f"Using platform configuration for: {platform}")
        
        # Get video information
        video_info = self._get_video_info(input_video)
        
        # Determine target file size
        target_size_mb = max_size_mb or platform_config.get('max_file_size_mb') or self.config.get('video_compression.max_file_size_mb', 10)
        
        # Adjust duration if needed
        if duration is not None:
            actual_duration = min(duration, video_info['duration'] - start_time)
        else:
            actual_duration = video_info['duration'] - start_time
        
        # Check if video should be segmented
        if self.should_segment_video(actual_duration, video_info, target_size_mb):
            logger.info("Video will be split into high-quality segments")
            print("🎬 Video will be split into high-quality segments")
            print(f"    📊 Processing {actual_duration:.1f}s video with target size {target_size_mb}MB per segment")
            results = self._split_video_into_segments(
                input_video, output_base_path, start_time, actual_duration, 
                video_info, platform_config, target_size_mb
            )
            
            # If segmentation failed, clean up any temp files
            if not results.get('success', False):
                self._cleanup_temp_files(results.get('temp_files', []))
                return results
            
            # Move segments to final location
            final_results = self._finalize_segments(results, output_base_path)
            return final_results
        else:
            logger.info("Video does not need segmentation, processing as single file")
            return self._process_single_video(input_video, output_base_path, start_time, 
                                           actual_duration, video_info, platform_config, target_size_mb)
    
    def _split_video_into_segments(self, input_video: str, output_base_path: str, 
                                  start_time: float, duration: float, 
                                  video_info: Dict[str, Any], platform_config: Dict[str, Any],
                                  target_size_mb: float) -> Dict[str, Any]:
        """
        Split a long/complex video into multiple high-quality video segments
        
        Args:
            input_video: Path to input video
            output_base_path: Base path for output (will create folder)
            start_time: Start time in video
            duration: Duration to process
            video_info: Video metadata
            platform_config: Platform-specific configuration
            target_size_mb: Target size per segment
            
        Returns:
            Dictionary with results including segment paths and metadata
        """
        
        # Create temp folder for segments first
        base_name = os.path.splitext(os.path.basename(output_base_path))[0]
        temp_segments_folder = os.path.join(self.temp_dir, f"{base_name}_segments_temp")
        
        os.makedirs(temp_segments_folder, exist_ok=True)
        logger.info(f"Created temp segments folder: {temp_segments_folder}")
        
        # Calculate optimal segment duration
        segment_duration = self._calculate_optimal_segment_duration(duration, video_info, target_size_mb)

        # Calculate number of segments
        num_segments = math.ceil(duration / segment_duration)
        # Equalize per-segment duration so all segments are similar length
        # This ensures the final segment isn't disproportionately short
        if num_segments > 0:
            equalized_duration = duration / num_segments
            # Keep within the originally calculated duration bound just in case
            segment_duration = min(segment_duration, equalized_duration)

        print(f"    🔄 Starting video segmentation: {duration:.1f}s video -> {num_segments} segments of ~{segment_duration:.1f}s each")
        logger.info(f"Splitting {duration:.1f}s video into {num_segments} segments of ~{segment_duration:.1f}s each (equalized)")
        
        segments = []
        temp_files = []
        
        # Load segmentation prefs from config
        prefer_fast_seek = bool(self.config.get('video_compression.segmentation.prefer_fast_seek', True))
        
        try:
            # Process segments in parallel with limited concurrency
            def _process_segment(i: int) -> Optional[Dict[str, Any]]:
                try:
                    if self.shutdown_requested:
                        return None
                        
                    # Use equalized boundaries; clamp the last segment to the exact end to avoid drift
                    segment_start = start_time + (i * segment_duration)
                    if i < num_segments - 1:
                        segment_end = start_time + ((i + 1) * segment_duration)
                    else:
                        segment_end = start_time + duration
                    segment_duration_actual = max(0.0, segment_end - segment_start)
                    
                    if segment_duration_actual <= 0:
                        return None
                    
                    # Create segment filename
                    segment_filename = f"{base_name}_segment_{i+1:03d}.mp4"
                    segment_path = os.path.join(temp_segments_folder, segment_filename)
                    
                    self._threadsafe_print(
                        f"    🔄 Processing segment {i+1}/{num_segments}: {segment_duration_actual:.1f}s "
                        f"({segment_start:.1f}s - {segment_end:.1f}s)"
                    )
                    logger.info(f"Processing segment {i+1}/{num_segments}: {segment_duration_actual:.1f}s "
                               f"({segment_start:.1f}s - {segment_end:.1f}s)")
                    
                    # Create high-quality segment with retry logic (creation-only; optimization deferred)
                    max_retries = 2
                    segment_result = None
                    for retry_attempt in range(max_retries + 1):
                        if retry_attempt > 0:
                            logger.warning(f"Retry {retry_attempt}/{max_retries} for segment {i+1}")
                            # Wait briefly before retry
                            time.sleep(1)
                        
                        segment_result = self._create_high_quality_segment(
                            input_video, segment_path, segment_start, segment_duration_actual,
                            video_info, platform_config, target_size_mb
                        )
                        
                        if segment_result.get('success', False):
                            # Success - break out of retry loop
                            break
                        else:
                            error_msg = segment_result.get('error', 'Unknown error')
                            if retry_attempt < max_retries:
                                logger.warning(f"Segment {i+1} creation failed (attempt {retry_attempt + 1}), will retry: {error_msg}")
                            else:
                                logger.error(f"Segment {i+1} creation failed after {max_retries + 1} attempts: {error_msg}")
                    
                    if segment_result and segment_result.get('success', False):
                        segment_size_mb = segment_result.get('size_mb', 0)
                        needs_optimization = segment_size_mb > target_size_mb
                        segment_result['optimization_applied'] = False
                        
                        segment_info = {
                            'index': i + 1,
                            'path': segment_path,
                            'start_time': segment_start,
                            'duration': segment_duration_actual,
                            'size_mb': segment_size_mb,
                            'method': segment_result.get('method', 'unknown'),
                            'quality_score': segment_result.get('quality_score', 0),
                            'optimization_applied': False,
                            'needs_optimization': needs_optimization
                        }
                        
                        logger.info(f"Segment {i+1} completed: {segment_size_mb:.2f}MB "
                                   f"using {segment_result.get('method', 'unknown')} method")
                        
                        return segment_info
                    else:
                        logger.error(f"Failed to create segment {i+1} after all retries")
                        return None
                        
                except Exception as e:
                    logger.error(f"Exception while processing segment {i+1}: {e}")
                    return None
            
            # Process segments in parallel with limited concurrency
            indices = list(range(num_segments))
            results: List[Dict[str, Any]] = []
            completed_count = 0
            
            logger.info(f"Starting parallel video segmentation with {self.max_workers} workers for {num_segments} segments")
            self._threadsafe_print(f"    🔄 Processing {num_segments} segments with {self.max_workers} concurrent workers...")
            
            with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                futures = [executor.submit(_process_segment, i) for i in indices]
                pending = set(futures)
                try:
                    while pending:
                        if self.shutdown_requested:
                            for f in list(pending):
                                f.cancel()
                            try:
                                executor.shutdown(wait=False, cancel_futures=True)
                            except TypeError:
                                executor.shutdown(wait=False)
                            break
                        # Poll for any completions
                        done = []
                        for f in list(pending):
                            if f.done():
                                done.append(f)
                        if not done:
                            time.sleep(0.1)
                            continue
                        for fut in done:
                            pending.discard(fut)
                            try:
                                segment_info = fut.result()
                            except Exception as e:
                                logger.warning(f"Segment future failed: {e}")
                                continue
                            if segment_info:
                                segments.append(segment_info)
                                temp_files.append(segment_info['path'])
                                completed_count += 1
                                self._threadsafe_print(
                                    f"    ✅ Segment {completed_count}/{num_segments} completed: {segment_info['size_mb']:.2f}MB "
                                    f"using {segment_info.get('method', 'unknown')} method")
                finally:
                    if self.shutdown_requested:
                        try:
                            executor.shutdown(wait=False, cancel_futures=True)
                        except TypeError:
                            executor.shutdown(wait=False)
            
            # Check if all segments were created successfully
            if len(segments) != num_segments:
                logger.error(f"Only {len(segments)}/{num_segments} segments were created successfully")
                return {
                    'success': False,
                    'error': f"Only {len(segments)}/{num_segments} segments were created successfully",
                    'temp_files': temp_files
                }
            
            # Post-creation optimization phase (batch oversize segments with limited concurrency)
            oversized = [s for s in segments if s.get('needs_optimization')]
            if oversized:
                logger.info(f"Starting optimization phase for {len(oversized)} oversized segments (target {target_size_mb}MB)")
                self._threadsafe_print(f"    🛠️ Optimizing {len(oversized)} oversized segments (concurrency: {self._optimization_workers})...")
                
                def _optimize_wrapper(seg: Dict[str, Any]) -> Tuple[int, Optional[Dict[str, Any]]]:
                    try:
                        if self.shutdown_requested:
                            return seg['index'], None
                        # Ensure we don't overlap with any other FFmpeg-heavy work
                        with self._ffmpeg_semaphore:
                            result = self._optimize_oversized_segment(
                                seg['path'], seg['duration'], video_info, platform_config, target_size_mb
                            )
                        return seg['index'], result
                    except Exception as e:
                        logger.warning(f"Optimization failed for segment {seg['index']}: {e}")
                        return seg['index'], None
                
                # Run optimizations with limited concurrency
                with ThreadPoolExecutor(max_workers=max(1, self._optimization_workers)) as opt_exec:
                    futures = {opt_exec.submit(_optimize_wrapper, s): s for s in oversized}
                    for fut in futures:
                        try:
                            idx, opt_res = fut.result()
                        except Exception as e:
                            logger.warning(f"Optimization future failed: {e}")
                            continue
                        # Find and update the segment entry
                        for seg in segments:
                            if seg['index'] == idx:
                                if opt_res and opt_res.get('success', False):
                                    seg['size_mb'] = opt_res.get('size_mb', seg['size_mb'])
                                    seg['method'] = f"{seg.get('method', 'unknown')}_optimized"
                                    seg['optimization_applied'] = True
                                    seg['needs_optimization'] = False
                                    self._threadsafe_print(
                                        f"    ✅ Optimized segment {idx}/{num_segments}: {seg['size_mb']:.2f}MB")
                                else:
                                    self._threadsafe_print(
                                        f"    ⚠️ Optimization skipped/failed for segment {idx}")
                                break

            print(f"    🎉 All {len(segments)} segments completed successfully!")
            logger.info(f"All {len(segments)} segments completed successfully")
            
            return {
                'success': True,
                'segments': segments,
                'num_segments': len(segments),
                'total_duration': duration,
                'segment_duration': segment_duration,
                'temp_folder': temp_segments_folder,
                'temp_files': temp_files,
                'base_name': base_name,
                'target_size_mb': target_size_mb
            }
            
        except Exception as e:
            logger.error(f"Error during video segmentation: {e}")
            return {
                'success': False,
                'error': str(e),
                'temp_files': temp_files
            }
    
    def _calculate_optimal_segment_duration(self, total_duration: float, video_info: Dict[str, Any], target_size_mb: float) -> float:
        """Calculate optimal duration for each segment based on content characteristics and target size"""

        complexity = video_info.get('complexity_score', 5.0)
        motion_level = video_info.get('motion_level', 'medium')

        # Favor longer segments for better quality and fewer total segments
        logger.info(f"Calculating optimal segment duration for {total_duration:.1f}s video - favoring longer segments")

        # Calculate base duration to target 12-15MB segments (increased from 8-9MB)
        # This allows for longer segments while staying under typical file size limits
        bitrate = video_info.get('bitrate', 0)
        if bitrate > 0:
            # Use actual bitrate to estimate duration for larger target size
            # Account for compression efficiency (segments often compress better)
            target_bits = (target_size_mb * 1.2) * 8 * 1024 * 1024  # 120% of target for video, 15% for audio
            estimated_duration = target_bits / bitrate
            # Use longer duration multiplier to create fewer, larger segments
            base_duration = min(estimated_duration * 3, total_duration / 2)  # At most 2 segments for most videos
        else:
            # Fallback: aim for significantly longer segments (increased from 90s max)
            # For longer videos, try to keep segments to 3-4 minutes maximum
            if total_duration <= 300:  # Up to 5 minutes
                base_duration = total_duration / max(1, min(3, int(total_duration / 120)))  # Max 2 minutes per segment
            elif total_duration <= 600:  # 5-10 minutes
                base_duration = total_duration / max(2, min(4, int(total_duration / 150)))  # Max 2.5 minutes per segment
            else:  # Over 10 minutes
                base_duration = total_duration / max(3, min(6, int(total_duration / 180)))  # Max 3 minutes per segment

        # Adjust for content complexity - be more lenient for longer segments
        if complexity >= 9 or motion_level == 'very_high':
            # Reduce duration only for extremely complex content
            base_duration *= 0.9
        elif complexity >= 7:
            # Slight reduction for high complexity
            base_duration *= 0.95
        elif complexity < 5:
            # Can afford even longer segments for simple content
            base_duration *= 1.15

        # Ensure reasonable bounds (configurable) - bias toward ~2 segments when total ~240s
        min_seg = int(self.config.get('video_compression.segmentation.min_segment_duration', 30) or 30)
        max_seg = int(self.config.get('video_compression.segmentation.max_segment_duration', 240) or 240)
        # If total duration fits into ~2 segments under max_seg, bias duration toward half the total
        if total_duration > 0 and total_duration <= max_seg * 2.2:
            desired = max(total_duration / 2.0, min_seg)
            base_duration = max(min(desired, max_seg), min_seg)
        final_duration = max(min_seg, min(max_seg, base_duration))

        # Calculate expected number of segments
        expected_segments = math.ceil(total_duration / final_duration)

        logger.info(f"Optimized segment duration to: {final_duration:.1f}s "
                    f"(complexity: {complexity:.1f}, motion: {motion_level}, "
                    f"target: {target_size_mb}MB, expected segments: {expected_segments})")

        return final_duration
    
    def _create_high_quality_segment(self, input_video: str, output_path: str, 
                                   start_time: float, duration: float, 
                                   video_info: Dict[str, Any], platform_config: Dict[str, Any],
                                   target_size_mb: float) -> Dict[str, Any]:
        """Create a high-quality video segment using optimized parameters"""
        
        # Validate duration parameter
        if duration <= 0 or duration > 600:  # Sanity check: 0-10 minutes
            logger.error(f"Invalid segment duration: {duration}s - this should be between 0 and 600 seconds")
            return {'success': False, 'error': f'Invalid duration: {duration}s'}
        
        try:
            # Calculate segment-specific compression parameters
            segment_params = self._calculate_segment_compression_params(
                video_info, platform_config, target_size_mb, duration
            )
            
            # Apply content-aware adjustments
            segment_params = self._apply_content_aware_adjustments(segment_params, video_info)
            
            logger.info(f"Creating segment with optimization: {segment_params['width']}x{segment_params['height']}, "
                       f"{duration:.1f}s @ {segment_params['fps']}fps")

            # Build FFmpeg command for segment creation (accurate seek: place -ss after -i)
            encoder = segment_params.get('encoder', 'libx264')
            accel_type = segment_params.get('acceleration_type', 'software')
            try:
                logger.info(f"Encoder selected: {encoder} ({accel_type})")
            except Exception:
                pass
            prefer_fast_seek = bool(self.config.get('video_compression.segmentation.prefer_fast_seek', True))
            if prefer_fast_seek:
                # Fast demuxer seek to avoid decoding from beginning on long videos
                cmd = [
                    'ffmpeg', '-y',
                    '-ss', str(start_time),
                    '-i', input_video,
                    '-t', str(duration),
                    '-c:v', encoder,
                ]
            else:
                cmd = [
                    'ffmpeg', '-y',
                    '-i', input_video,
                    '-ss', str(start_time),
                    '-t', str(duration),
                    '-c:v', encoder,
                ]
            # Quality/bitrate selection
            if accel_type == 'software':
                # Software: CRF and preset are valid
                if 'crf' in segment_params:
                    cmd.extend(['-crf', str(segment_params.get('crf'))])
                if 'preset' in segment_params:
                    cmd.extend(['-preset', segment_params.get('preset', 'medium')])
            else:
                # Hardware encoders (AMF/NVENC/QSV): avoid software-style preset strings
                if 'bitrate' in segment_params:
                    cmd.extend(['-b:v', str(segment_params['bitrate'])])
                # Do not append -preset here; vendor presets differ and can break (e.g., AMF)
            cmd.extend([
                '-vf', f"scale={segment_params['width']}:{segment_params['height']}:flags=lanczos",
                '-r', str(segment_params['fps']),
                '-c:a', 'aac',
                '-b:a', '128k',
                '-ac', '2',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                output_path
            ])
            
            # Execute FFmpeg command
            try:
                # Add performance flags (threads, analyzeduration) best-effort
                try:
                    FFmpegUtils.add_ffmpeg_perf_flags(cmd)
                except Exception:
                    pass
                # Use Popen + polling to support fast shutdown; limit concurrency to avoid GPU/CPU saturation
                # IMPORTANT: Do not capture stdout/stderr to avoid pipe buffer deadlocks on verbose ffmpeg output
                with self._ffmpeg_semaphore:
                    process = subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL
                    )
                with self._proc_lock:
                    self._active_processes.add(process)
                start_time_poll = time.time()
                # Dynamic timeout based on encoder type and segment duration (configurable; can be disabled)
                try:
                    enable_timeout = bool(self.config.get('video_compression.segmentation.enable_timeout', True))
                except Exception:
                    enable_timeout = True
                if enable_timeout:
                    if accel_type == 'software':
                        timeout_seconds = max(duration * 6.0, 600)
                    else:
                        timeout_seconds = max(duration * 4.0, 360)
                while process.poll() is None:
                    if self.shutdown_requested:
                        self._terminate_ffmpeg_process(process)
                        return {
                            'success': False,
                            'error': 'Shutdown requested during segment creation'
                        }
                    if enable_timeout and (time.time() - start_time_poll > timeout_seconds):
                        logger.warning("Segment creation timeout, terminating...")
                        self._terminate_ffmpeg_process(process)
                        return {
                            'success': False,
                            'error': 'Segment creation timed out'
                        }
                    time.sleep(0.1)
                # Process finished; get return code
                rc = process.returncode
                with self._proc_lock:
                    if process in self._active_processes:
                        self._active_processes.discard(process)
                
                if rc == 0:
                    # Validate segment is readable before marking success
                    if not os.path.exists(output_path):
                        logger.error(f"Segment file not created: {output_path}")
                        return {'success': False, 'error': 'Segment file not created'}
                    
                    # Quick validation with ffprobe
                    try:
                        probe_cmd = [
                            'ffprobe', '-v', 'quiet', '-print_format', 'json',
                            '-show_format', '-show_streams', output_path
                        ]
                        probe_result = subprocess.run(
                            probe_cmd, capture_output=True, text=True, timeout=10
                        )
                        if probe_result.returncode != 0:
                            logger.error(f"Segment validation failed: ffprobe returned {probe_result.returncode}")
                            logger.error(f"Segment may be corrupted: {output_path}")
                            # Remove corrupted file
                            try:
                                os.remove(output_path)
                            except Exception:
                                pass
                            return {
                                'success': False,
                                'error': f'Segment validation failed (ffprobe rc={probe_result.returncode})'
                            }
                        
                        # Parse basic info to ensure it's valid
                        import json
                        probe_data = json.loads(probe_result.stdout)
                        if 'format' not in probe_data or 'duration' not in probe_data.get('format', {}):
                            logger.error("Segment validation: missing format/duration info")
                            return {'success': False, 'error': 'Segment missing required metadata'}
                            
                    except subprocess.TimeoutExpired:
                        logger.error("Segment validation timeout")
                        return {'success': False, 'error': 'Segment validation timeout'}
                    except Exception as e:
                        logger.error(f"Segment validation exception: {e}")
                        return {'success': False, 'error': f'Segment validation failed: {e}'}
                    
                    # Get segment info
                    segment_size = os.path.getsize(output_path) / (1024 * 1024)
                    result = {
                        'success': True,
                        'segment_duration': duration,
                        'segment_start_time': start_time,
                        'optimization_type': 'segment_compression',
                        'size_mb': segment_size,
                        'method': 'direct_ffmpeg',
                        'output_file': output_path
                    }
                    
                    logger.info(f"Segment created and validated: {segment_size:.2f}MB")
                    return result
                else:
                    logger.error(f"FFmpeg failed with return code {rc}")
                    return {
                        'success': False,
                        'error': f'FFmpeg failed with return code {rc}'
                    }
                    
            except subprocess.TimeoutExpired:
                logger.error("Segment creation timed out")
                return {
                    'success': False,
                    'error': 'Segment creation timed out'
                }
            except Exception as e:
                logger.error(f"Error creating segment: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
            
        except Exception as e:
            logger.error(f"Error creating high-quality segment: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_segment_compression_params(self, video_info: Dict[str, Any], 
                                            platform_config: Dict[str, Any], 
                                            target_size_mb: float, duration: float) -> Dict[str, Any]:
        """Calculate compression parameters optimized for video segments"""
        
        # Get best encoder
        encoder, accel_type = self.hardware.get_best_encoder("h264")
        
        # Base parameters
        params = {
            'encoder': encoder,
            'acceleration_type': accel_type,
            'width': video_info['width'],
            'height': video_info['height'],
            'fps': min(video_info['fps'], 30),
        }
        # Adaptive resolution selection for segments
        try:
            if bool(self.config.get('video_compression.segmentation.adapt_resolution', True)):
                from .video_compressor import DynamicVideoCompressor
                temp_compressor = DynamicVideoCompressor(self.config, self.hardware)
                seg_info = dict(video_info)
                seg_info['duration'] = duration
                opt_w, opt_h = temp_compressor._calculate_optimal_resolution(seg_info, target_size_mb, platform_config or {})
                # Ensure even dimensions
                if opt_w % 2 != 0:
                    opt_w -= 1
                if opt_h % 2 != 0:
                    opt_h -= 1
                if opt_w > 0 and opt_h > 0:
                    params['width'] = opt_w
                    params['height'] = opt_h
                    logger.info(f"Segment adaptive resolution: {opt_w}x{opt_h}")
        except Exception as e:
            logger.warning(f"Adaptive resolution selection failed: {e}")
        
        # Content-aware quality adjustment
        complexity = video_info.get('complexity_score', 5.0)
        motion_level = video_info.get('motion_level', 'medium')
        
        # Adaptive CRF based on content complexity
        if accel_type == 'software':
            base_crf = 23  # Higher quality starting point
            if complexity > 7:
                params['crf'] = base_crf + 3  # Higher CRF for complex content
            elif complexity < 3:
                params['crf'] = base_crf - 2  # Lower CRF for simple content
            else:
                params['crf'] = base_crf
        
        # Motion-aware preset selection
        if motion_level == 'high':
            params['preset'] = 'faster'  # Faster preset for high motion
        elif motion_level == 'low':
            params['preset'] = 'slower'  # Slower preset for low motion
        else:
            params['preset'] = 'medium'
        
        # Calculate bitrate for segment
        params['bitrate'] = self._calculate_segment_bitrate(video_info, target_size_mb, duration)
        
        # Platform-specific adjustments
        if platform_config:
            params.update(self._apply_platform_constraints(params, platform_config))
        
        return params
    
    def _calculate_segment_bitrate(self, video_info: Dict[str, Any], target_size_mb: float, duration: float) -> int:
        """Calculate bitrate for video segment"""
        
        # Base calculation accounting for audio overhead
        target_bits = target_size_mb * 8 * 1024 * 1024 * 0.85  # 85% for video, 15% for audio
        base_bitrate = int(target_bits / duration / 1000)  # Convert to kbps
        
        # Adjust based on content complexity
        complexity = video_info.get('complexity_score', 5.0)
        motion_level = video_info.get('motion_level', 'medium')
        
        # Complexity adjustment factor
        if complexity > 7:
            bitrate_multiplier = 1.2  # Need more bitrate for complex content
        elif complexity < 3:
            bitrate_multiplier = 0.8  # Can use less bitrate for simple content
        else:
            bitrate_multiplier = 1.0
        
        # Motion adjustment
        motion_multipliers = {'low': 0.9, 'medium': 1.0, 'high': 1.1}
        bitrate_multiplier *= motion_multipliers.get(motion_level, 1.0)
        
        # Resolution efficiency factor
        pixel_count = video_info['width'] * video_info['height']
        if pixel_count > 1920 * 1080:  # 4K+ content
            bitrate_multiplier *= 1.3
        elif pixel_count < 1280 * 720:  # Lower resolution
            bitrate_multiplier *= 0.8
        
        final_bitrate = int(base_bitrate * bitrate_multiplier)
        
        # Ensure minimum viable bitrate
        min_bitrate = 200
        final_bitrate = max(final_bitrate, min_bitrate)
        
        # Return as string with 'k' suffix for FFmpeg
        return f"{final_bitrate}k"
    
    def _optimize_oversized_segment(self, segment_path: str, duration: float, 
                                  video_info: Dict[str, Any], platform_config: Dict[str, Any], 
                                  target_size_mb: float) -> Dict[str, Any]:
        """
        Optimize an oversized segment using direct compression with progressive quality fallback.
        
        Implements progressive quality floor fallback for segments:
        - Attempt 1: quality floor = 70/100 (high quality)
        - Attempt 2: quality floor = 60/100 (medium-high quality)
        - Attempt 3: quality floor = 50/100 (medium quality)
        - Attempt 4: quality floor = 45/100 (acceptable quality - final fallback)
        """
        
        # Import the video compressor to use its optimization capabilities
        from .video_compressor import DynamicVideoCompressor
        
        # Progressive quality floor attempts for segments
        quality_attempts = [0, 1, 2, 3]  # Maps to floors [70, 60, 50, 45]
        last_error = None
        
        for attempt in quality_attempts:
            temp_compressor = None
            temp_optimized_path = segment_path.replace('.mp4', f'_optimized_temp_q{attempt}.mp4')
            try:
                # Create a temporary compressor instance with our existing config and hardware
                temp_compressor = DynamicVideoCompressor(self.config, self.hardware)
                # Track temp compressor so shutdown can terminate it
                try:
                    with self._active_temp_lock:
                        self._active_temp_compressors.add(temp_compressor)
                except Exception:
                    pass
                
                quality_floor = temp_compressor.quality_scorer.get_segment_quality_floor(attempt)
                logger.info(f"🎯 Segment optimization attempt {attempt + 1}/4 with quality floor {quality_floor:.1f}/100")
                logger.info(f"Applying direct compression optimization to oversized segment: {segment_path}")
                logger.info(f"Target size for optimization: {target_size_mb}MB")
                
                # Use full target size to honor configured/default 10MB limit
                optimization_target = target_size_mb
                
                # Get video info for the segment (it may be different from the original video)
                segment_video_info = temp_compressor._analyze_video_content(segment_path)
                
                # Try different compression strategies to optimize the segment
                optimization_result = None
                
                # Strategy 1: Advanced optimization (best quality but may fail)
                try:
                    if self.shutdown_requested:
                        try:
                            temp_compressor.request_shutdown()
                        except Exception:
                            pass
                        return { 'success': False, 'error': 'shutdown' }
                    logger.info("Attempting advanced optimization strategy")
                    optimization_result = temp_compressor._compress_with_advanced_optimization(
                        input_path=segment_path,
                        output_path=temp_optimized_path,
                        target_size_mb=optimization_target,
                        platform_config=platform_config or {},
                        video_info=segment_video_info
                    )
                except Exception as e:
                    logger.warning(f"Advanced optimization failed, trying standard: {e}")
                    optimization_result = None
                
                # Strategy 2: Standard optimization (fallback)
                if not optimization_result or not optimization_result.get('success', False):
                    logger.info("Attempting standard optimization strategy")
                    try:
                        if self.shutdown_requested:
                            try:
                                temp_compressor.request_shutdown()
                            except Exception:
                                pass
                        
                        if self.shutdown_requested:
                            return { 'success': False, 'error': 'shutdown' }
                        optimization_result = temp_compressor._compress_with_standard_optimization(
                            input_path=segment_path,
                            output_path=temp_optimized_path,
                            target_size_mb=optimization_target,
                            platform_config=platform_config or {},
                            video_info=segment_video_info
                        )
                    except Exception as e:
                        logger.warning(f"Standard optimization failed, trying adaptive: {e}")
                        optimization_result = None
                
                # Strategy 3: Adaptive quality (last resort)
                if not optimization_result or not optimization_result.get('success', False):
                    logger.info("Attempting adaptive quality optimization strategy")
                    try:
                        if self.shutdown_requested:
                            try:
                                temp_compressor.request_shutdown()
                            except Exception:
                                pass
                            return { 'success': False, 'error': 'shutdown' }
                        optimization_result = temp_compressor._compress_with_adaptive_quality(
                            input_path=segment_path,
                            output_path=temp_optimized_path,
                            target_size_mb=optimization_target,
                            platform_config=platform_config or {},
                            video_info=segment_video_info
                        )
                    except Exception as e:
                        logger.error(f"All optimization strategies failed: {e}")
                        optimization_result = None
                
                # Check results of optimization attempts
                if optimization_result and optimization_result.get('success', False):
                    optimized_size_mb = optimization_result.get('size_mb', 0)

                    # Strict size enforcement: must be <= target
                    if optimized_size_mb <= target_size_mb:
                        if os.path.exists(temp_optimized_path):
                            shutil.move(temp_optimized_path, segment_path)
                            logger.info(f"✅ Segment optimization successful (attempt {attempt + 1}): {optimized_size_mb:.2f}MB (<= {target_size_mb}MB)")
                            return {
                                'success': True,
                                'size_mb': optimized_size_mb,
                                'method': optimization_result.get('method', 'compression_optimization'),
                                'original_method': 'segment_creation',
                                'compression_applied': True,
                                'quality_attempt': attempt
                            }
                        logger.warning("Optimized segment file not found after processing")
                        last_error = 'Optimized segment file not found'
                        # Continue to next quality attempt
                        continue

                    # Near-limit retry: one more attempt if within configured percent over target
                    try:
                        near_pct = float(self.config.get('video_compression.size_control.near_limit_retry_percent', 0.08))
                    except Exception:
                        near_pct = 0.08
                    if optimized_size_mb <= target_size_mb * (1.0 + near_pct):
                        logger.info(f"Near-limit retry: {optimized_size_mb:.2f}MB within {(near_pct*100):.1f}% of target; attempting adaptive retry")
                        try:
                            from .video_compressor import DynamicVideoCompressor
                            temp_compressor2 = DynamicVideoCompressor(self.config, self.hardware)
                            seg_info = temp_compressor2._analyze_video_content(segment_path)
                            retry_res = temp_compressor2._compress_with_adaptive_resolution(
                                input_path=segment_path,
                                video_info=seg_info,
                                platform_config=platform_config or {},
                                target_size_mb=target_size_mb
                            )
                        except Exception as e:
                            logger.warning(f"Near-limit adaptive retry failed: {e}")
                            retry_res = None
                        if retry_res and retry_res.get('temp_file') and os.path.exists(retry_res['temp_file']):
                            new_size = os.path.getsize(retry_res['temp_file']) / (1024 * 1024)
                            if new_size <= target_size_mb:
                                shutil.move(retry_res['temp_file'], segment_path)
                                logger.info(f"✅ Near-limit retry succeeded (attempt {attempt + 1}): {new_size:.2f}MB (<= {target_size_mb}MB)")
                                return {
                                    'success': True,
                                    'size_mb': new_size,
                                    'method': retry_res.get('strategy', 'adaptive_resolution'),
                                    'original_method': 'segment_creation',
                                    'compression_applied': True,
                                    'quality_attempt': attempt
                                }
                            else:
                                try:
                                    os.remove(retry_res['temp_file'])
                                except Exception:
                                    pass

                    # Optimization didn't help enough - try next quality floor
                    if os.path.exists(temp_optimized_path):
                        os.remove(temp_optimized_path)
                    last_error = f'Optimization insufficient: {optimized_size_mb:.2f}MB still too large'
                    logger.warning(f"⚠️  Attempt {attempt + 1} did not sufficiently reduce size: {optimized_size_mb:.2f}MB > {target_size_mb}MB")
                    # Continue to next quality attempt
                    continue
                else:
                    # All optimization strategies failed for this attempt
                    if os.path.exists(temp_optimized_path):
                        os.remove(temp_optimized_path)
                    error_msg = optimization_result.get('error', 'All optimization strategies failed') if optimization_result else 'All optimization strategies failed'
                    logger.warning(f"⚠️  Attempt {attempt + 1} optimization failed: {error_msg}")
                    last_error = error_msg
                    # Continue to next quality attempt
                    continue
            except Exception as e:
                logger.warning(f"⚠️  Attempt {attempt + 1} encountered exception: {e}")
                last_error = str(e)
                # Clean up any temp files for this attempt
                if temp_optimized_path and os.path.exists(temp_optimized_path):
                    try:
                        os.remove(temp_optimized_path)
                    except Exception:
                        pass
                # Continue to next quality attempt
                continue
            finally:
                # Ensure temp compressor is removed from active tracking for this attempt
                try:
                    with self._active_temp_lock:
                        if temp_compressor and temp_compressor in self._active_temp_compressors:
                            self._active_temp_compressors.discard(temp_compressor)
                except Exception:
                    pass
        
        # All quality attempts exhausted
        logger.error(f"❌ All 4 quality attempts failed for segment optimization. Last error: {last_error}")
        return {
            'success': False,
            'error': f'All quality attempts exhausted: {last_error}'
        }
    
    def _apply_content_aware_adjustments(self, params: Dict[str, Any], video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply content-aware adjustments to compression parameters"""
        
        complexity = video_info.get('complexity_score', 5.0)
        motion_level = video_info.get('motion_level', 'medium')
        
        # Adjust FPS based on motion level
        if motion_level == 'high':
            params['fps'] = min(params['fps'], 30)  # Keep high FPS for high motion
        elif motion_level == 'low':
            params['fps'] = min(params['fps'], 24)  # Lower FPS for low motion
        
        # Adjust quality based on complexity
        if 'crf' in params:
            if complexity > 7:
                params['crf'] = min(params['crf'] + 2, 35)  # Higher CRF for complex content
            elif complexity < 3:
                params['crf'] = max(params['crf'] - 2, 18)  # Lower CRF for simple content
        
        return params
    
    def _apply_platform_constraints(self, params: Dict[str, Any], platform_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply platform-specific constraints to parameters"""
        constraints = {}
        
        # Resolution constraints
        if 'max_width' in platform_config and 'max_height' in platform_config:
            max_width = platform_config['max_width']
            max_height = platform_config['max_height']
            
            scale_factor = min(max_width / params['width'], max_height / params['height'], 1.0)
            constraints['width'] = int(params['width'] * scale_factor)
            constraints['height'] = int(params['height'] * scale_factor)
            
            # Ensure even dimensions
            constraints['width'] = constraints['width'] if constraints['width'] % 2 == 0 else constraints['width'] - 1
            constraints['height'] = constraints['height'] if constraints['height'] % 2 == 0 else constraints['height'] - 1
        
        # FPS constraints
        if 'fps' in platform_config:
            constraints['fps'] = min(platform_config['fps'], params['fps'])
        
        return constraints
    
    def _estimate_video_size(self, params: Dict[str, Any], duration: float, video_info: Dict[str, Any]) -> float:
        """Estimate video file size based on parameters and duration"""
        
        # Calculate bitrate in kbps
        if 'bitrate' in params:
            if isinstance(params['bitrate'], str):
                # Parse bitrate string (e.g., "1000k")
                bitrate_str = params['bitrate'].lower()
                if bitrate_str.endswith('k'):
                    bitrate = int(bitrate_str[:-1])
                elif bitrate_str.endswith('m'):
                    bitrate = int(bitrate_str[:-1]) * 1000
                else:
                    bitrate = int(bitrate_str)
            else:
                bitrate = params['bitrate']
        else:
            # Estimate bitrate based on resolution and quality
            pixel_count = params['width'] * params['height']
            fps = params.get('fps', 30)
            crf = params.get('crf', 28)
            
            # Rough bitrate estimation
            if pixel_count > 1920 * 1080:  # 4K+
                bitrate = 4000
            elif pixel_count > 1280 * 720:  # 1080p
                bitrate = 2000
            elif pixel_count > 854 * 480:  # 480p
                bitrate = 1000
            else:
                bitrate = 500
            
            # Adjust for quality
            if crf < 20:
                bitrate *= 1.5
            elif crf > 35:
                bitrate *= 0.7
        
        # Calculate size in MB
        bits_per_second = bitrate * 1000
        total_bits = bits_per_second * duration
        size_mb = total_bits / (8 * 1024 * 1024)
        
        # Add audio overhead (typically 10-15%)
        size_mb *= 1.15
        
        return size_mb
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get comprehensive video information using FFprobe"""
        try:
            # Lightweight ffprobe: avoid frame-level enumeration which is slow on long videos
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams',
                video_path
            ]
            
            # Increase timeout for large files to prevent premature timeouts
            try:
                file_size_mb_for_timeout = os.path.getsize(video_path) / (1024 * 1024)
            except Exception:
                file_size_mb_for_timeout = 0
            probe_timeout = 60
            if file_size_mb_for_timeout > 1000:
                probe_timeout = 240
            elif file_size_mb_for_timeout > 300:
                probe_timeout = 120

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=probe_timeout
            )
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
            
            stdout_text = result.stdout or ""
            probe_data = json.loads(stdout_text) if stdout_text.strip() else {}
            
            # Find video stream
            video_stream = next(
                (stream for stream in probe_data['streams'] if stream['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise ValueError("No video stream found in file")
            
            # Basic video properties
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            duration = float(probe_data['format'].get('duration', 0))
            bitrate = int(probe_data['format'].get('bit_rate', 0))
            from .ffmpeg_utils import FFmpegUtils
            fps = FFmpegUtils.parse_fps(video_stream.get('r_frame_rate', '30/1'))
            
            # Content complexity analysis
            complexity_score = self._calculate_content_complexity(probe_data, width, height, duration)
            
            # Motion analysis
            motion_level = self._estimate_motion_level(video_stream, duration)
            
            # Scene change detection
            scene_changes = self._estimate_scene_changes(duration, fps)
            
            return {
                'width': width,
                'height': height,
                'duration': duration,
                'bitrate': bitrate,
                'fps': fps,
                'codec': video_stream.get('codec_name', 'unknown'),
                'size_bytes': int(probe_data['format'].get('size', 0)),
                'complexity_score': complexity_score,
                'motion_level': motion_level,
                'scene_changes': scene_changes,
                'aspect_ratio': width / height if height > 0 else 1.0,
                'pixel_count': width * height,
                'bitrate_per_pixel': bitrate / (width * height) if (width * height) > 0 else 0
            }
            
        except subprocess.TimeoutExpired as e:
            logger.warning(f"ffprobe analysis timed out after {getattr(e, 'timeout', 'unknown')}s; using basic video info")
            return self._get_basic_video_info(video_path)
        except Exception as e:
            logger.warning(f"Failed to analyze video content quickly; using basic info: {e}")
            # Fallback to basic analysis
            return self._get_basic_video_info(video_path)
    
    def _calculate_content_complexity(self, probe_data: Dict, width: int, height: int, duration: float) -> float:
        """Calculate content complexity score (0-10, higher = more complex)"""
        try:
            # Base complexity from resolution and duration
            pixel_complexity = math.log10(width * height) / 6.0 * 10  # Normalize to 0-10
            
            # Estimate complexity from bitrate if available
            format_info = probe_data.get('format', {})
            bitrate = int(format_info.get('bit_rate', 0))
            
            if bitrate > 0:
                # Expected bitrate for resolution
                expected_bitrate = (width * height * 30 * 0.1)  # Rough estimate
                bitrate_complexity = min(bitrate / expected_bitrate * 3, 10)
            else:
                bitrate_complexity = 5  # Default medium complexity
            
            # Combine factors
            complexity = (pixel_complexity * 0.4 + bitrate_complexity * 0.6)
            return max(0, min(10, complexity))
            
        except Exception:
            return 5.0  # Default medium complexity
    
    def _estimate_motion_level(self, video_stream: Dict, duration: float) -> str:
        """Estimate motion level in video"""
        from .ffmpeg_utils import FFmpegUtils
        fps = FFmpegUtils.parse_fps(video_stream.get('r_frame_rate', '30/1'))
        
        if fps >= 50:
            return "high"
        elif fps >= 30:
            return "medium"
        else:
            return "low"
    
    def _estimate_scene_changes(self, duration: float, fps: float) -> int:
        """Estimate number of scene changes"""
        # Rough estimate: average video has a scene change every 3-5 seconds
        return max(1, int(duration / 4))
    
    def _get_basic_video_info(self, video_path: str) -> Dict[str, Any]:
        """Fallback method for basic video info"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
                video_path
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30
            )
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
            
            stdout_text = result.stdout or ""
            probe_data = json.loads(stdout_text) if stdout_text.strip() else {}
            
            # Find video stream
            video_stream = next(
                (stream for stream in probe_data['streams'] if stream['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise ValueError("No video stream found in file")
            
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            duration = float(probe_data['format'].get('duration', 0))
            
            return {
                'width': width,
                'height': height,
                'duration': duration,
                'bitrate': int(probe_data['format'].get('bit_rate', 0)),
                'fps': FFmpegUtils.parse_fps(video_stream.get('r_frame_rate', '30/1')),
                'codec': video_stream.get('codec_name', 'unknown'),
                'size_bytes': int(probe_data['format'].get('size', 0)),
                'complexity_score': 5.0,  # Default medium complexity
                'motion_level': 'medium',
                'scene_changes': max(1, int(duration / 4)),
                'aspect_ratio': width / height if height > 0 else 1.0,
                'pixel_count': width * height,
                'bitrate_per_pixel': 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get basic video info: {e}")
            raise
    
    def _create_segments_summary(self, segments_folder: str, segments: List[Dict[str, Any]], 
                               base_name: str, total_duration: float):
        """Create a summary file with segment information"""
        try:
            summary_file = os.path.join(segments_folder, f"{base_name}_segments_summary.txt")
            
            with open(summary_file, 'w') as f:
                f.write(f"Video Segments Summary\n")
                f.write(f"=====================\n\n")
                f.write(f"Base Name: {base_name}\n")
                f.write(f"Total Duration: {total_duration:.2f} seconds\n")
                f.write(f"Number of Segments: {len(segments)}\n")
                f.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"Segment Details:\n")
                f.write(f"---------------\n")
                
                total_size = 0
                optimized_count = 0
                for segment in segments:
                    f.write(f"Segment {segment['index']:03d}:\n")
                    f.write(f"  Duration: {segment['duration']:.2f}s\n")
                    f.write(f"  Start Time: {segment['start_time']:.2f}s\n")
                    f.write(f"  Size: {segment['size_mb']:.2f}MB\n")
                    f.write(f"  Method: {segment['method']}\n")
                    f.write(f"  Quality Score: {segment['quality_score']:.1f}\n")
                    if segment.get('optimization_applied', False):
                        f.write(f"  Optimization Applied: Yes\n")
                        optimized_count += 1
                    else:
                        f.write(f"  Optimization Applied: No\n")
                    f.write(f"  File: {os.path.basename(segment['path'])}\n\n")
                    
                    total_size += segment['size_mb']
                
                f.write(f"Total Size: {total_size:.2f}MB\n")
                f.write(f"Average Segment Size: {total_size/len(segments):.2f}MB\n")
                f.write(f"Segments Optimized: {optimized_count}/{len(segments)}\n")
            
            logger.info(f"Created segments summary: {summary_file}")
            
        except Exception as e:
            logger.warning(f"Failed to create segments summary: {e}")
    
    def _finalize_segments(self, results: Dict[str, Any], output_base_path: str) -> Dict[str, Any]:
        """Move segments from temp folder to final location with strict size enforcement"""
        try:
            temp_folder = results['temp_folder']
            base_name = results['base_name']
            target_size_mb = results.get('target_size_mb') or self.config.get('video_compression.max_file_size_mb', 10)
            reject_oversize = bool(self.config.get('video_compression.segmentation.reject_oversize_on_finalize', True))
            
            # Prepare final segments; delay folder creation until we have a valid segment
            output_dir = os.path.dirname(output_base_path)
            final_segments_folder = os.path.join(output_dir, f"{base_name}_segments")
            folder_created = False
            
            # Move segments to final location
            final_segments = []
            for segment in results['segments']:
                temp_path = segment['path']
                try:
                    size_mb = os.path.getsize(temp_path) / (1024 * 1024)
                except Exception:
                    size_mb = segment.get('size_mb', float('inf'))

                if reject_oversize and size_mb > target_size_mb:
                    logger.warning(f"Skipping oversize segment {segment.get('index')} at finalize: {size_mb:.2f}MB > {target_size_mb}MB")
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except Exception:
                        pass
                    continue

                if not folder_created:
                    if not os.path.exists(final_segments_folder):
                        os.makedirs(final_segments_folder, exist_ok=True)
                    folder_created = True

                final_filename = f"{base_name}_segment_{segment['index']:03d}.mp4"
                final_path = os.path.join(final_segments_folder, final_filename)

                shutil.move(temp_path, final_path)

                # Update segment info
                segment['path'] = final_path
                segment['filename'] = final_filename
                segment['size_mb'] = size_mb
                final_segments.append(segment)
            
            # Clean up temp folder
            try:
                shutil.rmtree(temp_folder)
            except Exception as e:
                logger.warning(f"Failed to clean up temp folder {temp_folder}: {e}")
            
            # Create comprehensive summary for the segments folder
            try:
                self._create_comprehensive_segments_summary(final_segments_folder, base_name)
            except Exception as e:
                logger.warning(f"Could not create comprehensive summary: {e}")
            
            if not final_segments:
                logger.error("All segments exceeded size limit; no outputs written")
                return {
                    'success': False,
                    'error': 'All segments exceeded size limit; no outputs written',
                    'temp_files': results.get('temp_files', [])
                }

            return {
                'success': True,
                'segments': final_segments,
                'num_segments': len(final_segments),
                'total_duration': results['total_duration'],
                'segment_duration': results['segment_duration'],
                'output_folder': final_segments_folder,
                'base_name': base_name
            }
            
        except Exception as e:
            logger.error(f"Error finalizing segments: {e}")
            return {
                'success': False,
                'error': str(e),
                'temp_files': results.get('temp_files', [])
            }
    
    def _process_single_video(self, input_video: str, output_path: str, start_time: float,
                             duration: float, video_info: Dict[str, Any], 
                             platform_config: Dict[str, Any], target_size_mb: float) -> Dict[str, Any]:
        """Process video as single file without segmentation"""
        try:
            # Calculate compression parameters
            params = self._calculate_segment_compression_params(video_info, platform_config, target_size_mb, duration)
            
            # Build FFmpeg command
            cmd = [
                'ffmpeg', '-y',
                '-i', input_video,
                '-c:v', params['encoder'],
                '-vf', f"scale={params['width']}:{params['height']}:flags=lanczos",
                '-r', str(params['fps']),
                # Apply software-specific quality flags if using software encoding
                *([arg for arg in (['-crf', str(params['crf'])] if 'crf' in params else [])]),
                *([arg for arg in (['-preset', params.get('preset', 'medium')] if params.get('acceleration_type') == 'software' else [])]),
                # Bitrate control (always include)
                '-b:v', params['bitrate'],
                '-c:a', 'aac',
                '-b:a', '96k',
                '-ac', '2',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            # Do not inject generic -hwaccel here; encoding choice already set by encoder
            try:
                logger.info(f"Encoder selected: {params.get('encoder', 'unknown')} ({params.get('acceleration_type', 'software')})")
            except Exception:
                pass
            
            # Execute FFmpeg command
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=300
            )
            
            if result.returncode == 0:
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                # Note: Do not create folder.jpg for single MP4 outputs; reserved for segmented outputs only
                return {
                    'success': True,
                    'size_mb': file_size,
                    'method': 'direct_ffmpeg',
                    'output_file': output_path
                }
            else:
                logger.error(f"FFmpeg failed: {result.stderr}")
                return {
                    'success': False,
                    'error': f"FFmpeg failed: {result.stderr}"
                }
                
        except Exception as e:
            logger.error(f"Error processing single video: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _cleanup_temp_files(self, temp_files: List[str]):
        """Clean up temporary files"""
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
    
    def request_shutdown(self):
        """Request graceful shutdown of the segmenter"""
        logger.info("Shutdown requested for video segmenter")
        self.shutdown_requested = True
        # Terminate all active FFmpeg processes
        with self._proc_lock:
            active = list(self._active_processes)
        for proc in active:
            try:
                self._terminate_ffmpeg_process(proc)
            except Exception:
                pass
        # Terminate any active temporary compressors used during optimization
        try:
            with self._active_temp_lock:
                temp_compressors = list(self._active_temp_compressors)
            for comp in temp_compressors:
                try:
                    if hasattr(comp, 'request_shutdown'):
                        comp.request_shutdown()
                except Exception:
                    pass
        except Exception:
            pass
    
    def _terminate_ffmpeg_process(self, process: Optional[subprocess.Popen] = None):
        """Terminate an FFmpeg process gracefully"""
        target = process
        # Backward-compatibility: if no process is passed, use current_ffmpeg_process if present
        if target is None:
            target = self.current_ffmpeg_process
        if target and target.poll() is None:
            try:
                target.terminate()
                try:
                    target.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.warning("FFmpeg process did not terminate gracefully, forcing kill...")
                    target.kill()
                    target.wait()
                logger.info("FFmpeg process terminated successfully")
            except Exception as e:
                logger.error(f"Error terminating FFmpeg process: {e}")
            finally:
                with self._proc_lock:
                    if target in self._active_processes:
                        self._active_processes.discard(target)
                if target is self.current_ffmpeg_process:
                    self.current_ffmpeg_process = None 

    def _create_comprehensive_segments_summary(self, segments_folder: str, base_name: str) -> None:
        """
        Create a comprehensive summary file for both MP4 and GIF segments in a segments folder.
        The summary will be prefixed with '~' to ensure it appears last in folder listings.
        """
        try:
            import time
            from .ffmpeg_utils import FFmpegUtils
            
            # Find all MP4 and GIF files in the segments folder
            mp4_files = [f for f in os.listdir(segments_folder) if f.lower().endswith('.mp4')]
            gif_files = [f for f in os.listdir(segments_folder) if f.lower().endswith('.gif')]
            
            if not mp4_files and not gif_files:
                logger.info(f"No MP4 or GIF files found in segments folder: {segments_folder}")
                return
            
            # Create summary filename with '~' prefix to ensure it appears last
            summary_file = os.path.join(segments_folder, f"~{base_name}_comprehensive_summary.txt")
            temp_file = os.path.join(segments_folder, f"~{base_name}_comprehensive_summary.txt.tmp")
            # Best-effort clear attributes
            try:
                if os.path.exists(summary_file):
                    subprocess.run(['attrib', '-R', '-S', '-H', summary_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            except Exception:
                pass
            
            # Write to temp file first
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
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
                    for idx, name in enumerate(mp4_files, 1):
                        path = os.path.join(segments_folder, name)
                        try:
                            info = FFmpegUtils.get_video_info(path)
                            size_mb = os.path.getsize(path) / (1024 * 1024)
                            f.write(f"MP4 Segment {idx:03d}: {name}\n")
                            f.write(f"  Duration: {info.get('duration', 0.0):.2f}s\n")
                            f.write(f"  FPS: {info.get('fps', 0.0):.2f}\n")
                            f.write(f"  Frame Count: {info.get('frame_count', 0)}\n")
                            f.write(f"  Resolution: {info.get('width', 0)}x{info.get('height', 0)}\n")
                            f.write(f"  Codec: {info.get('codec', 'unknown')}\n")
                            f.write(f"  Bitrate: {info.get('bitrate', 0)} kbps\n")
                            f.write(f"  Size: {size_mb:.2f}MB\n\n")
                        except Exception as e:
                            f.write(f"MP4 Segment {idx:03d}: {name}\n")
                            f.write(f"  Error reading info: {e}\n\n")

                    f.write("MP4 Summary:\n")
                    f.write(f"  Total MP4 Segments: {len(mp4_files)}\n\n")

                # GIF details
                if gif_files:
                    f.write("GIF Segments Details:\n")
                    f.write("--------------------\n")
                    for idx, name in enumerate(gif_files, 1):
                        path = os.path.join(segments_folder, name)
                        try:
                            size_mb = os.path.getsize(path) / (1024 * 1024)
                            f.write(f"GIF Segment {idx:03d}: {name}\n")
                            f.write(f"  Size: {size_mb:.2f}MB\n\n")
                        except Exception as e:
                            f.write(f"GIF Segment {idx:03d}: {name}\n")
                            f.write(f"  Error reading info: {e}\n\n")

                    f.write("GIF Summary:\n")
                    f.write(f"  Total GIF Segments: {len(gif_files)}\n\n")

                # Overall
                total_files = len(mp4_files) + len(gif_files)
                f.write("Overall Summary:\n")
                f.write("---------------\n")
                f.write(f"Total Files: {total_files}\n")
                f.write(f"File Types: MP4 ({len(mp4_files)}), GIF ({len(gif_files)})\n")

            # Atomic replace
            try:
                os.replace(temp_file, summary_file)
            except Exception:
                try:
                    subprocess.run(['attrib', '-R', '-S', '-H', summary_file], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                    os.replace(temp_file, summary_file)
                except Exception:
                    try:
                        if os.path.exists(temp_file):
                            os.remove(temp_file)
                    except Exception:
                        pass
                    raise

            # Do not hide the summary; folder.jpg controls thumbnail selection

        except Exception as e:
            logger.warning(f"Failed to create segments summary: {e}")

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
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30
            )
            
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