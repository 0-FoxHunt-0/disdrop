"""
GIF Segmenter
Handles segmentation of long videos into multiple GIF files
"""

import os
import math
import re
from typing import Dict, Any, List, Tuple, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import logging

from .gif_utils import safe_file_operation, create_unique_temp_filename
from .gif_config import GifConfigHelper

logger = logging.getLogger(__name__)


class GifSegmenter:
    """Handles segmentation of videos into multiple GIF files"""
    
    def __init__(self, config_manager, gif_generator, shutdown_checker: Optional[Callable[[], bool]] = None):
        """
        Initialize GIF segmenter.
        
        Args:
            config_manager: ConfigManager instance
            gif_generator: GifGenerator instance for creating individual segments
            shutdown_checker: Optional shutdown checker callback
        """
        self.config = config_manager
        self.config_helper = GifConfigHelper(config_manager)
        self.gif_generator = gif_generator
        self._shutdown_checker = shutdown_checker or (lambda: False)
    
    def should_segment(self, input_video: str, duration: float, settings: Dict[str, Any]) -> bool:
        """
        Determine if video should be segmented.
        
        Args:
            input_video: Path to input video
            duration: Video duration in seconds
            settings: Settings dictionary with max_duration, max_size_mb, etc.
        
        Returns:
            True if segmentation should be used
        """
        seg_config = self.config_helper.get_segmentation_config()
        
        # Check if segmentation is preferred
        if not seg_config.get('prefer_single_file_first', True):
            # If not preferring single file first, segment if duration exceeds limit
            max_duration = settings.get('max_duration', 30.0)
            if duration > max_duration:
                return True
        
        # Check duration limit
        fallback_duration_limit = seg_config.get('fallback_duration_limit', 180)
        if duration > fallback_duration_limit:
            logger.debug(f"Segmentation recommended: duration {duration:.1f}s > {fallback_duration_limit}s")
            return True
        
        # Check size threshold (if we can estimate)
        try:
            max_size_mb = settings.get('max_size_mb', 10.0)
            size_threshold_multiplier = seg_config.get('size_threshold_multiplier', 2.5)
            
            # Estimate GIF size
            estimated_size = self._estimate_gif_size(input_video, duration, settings)
            if estimated_size > max_size_mb * size_threshold_multiplier:
                logger.debug(f"Segmentation recommended: estimated size {estimated_size:.2f}MB > {max_size_mb * size_threshold_multiplier:.2f}MB")
                return True
        except Exception as e:
            logger.debug(f"Could not estimate size for segmentation decision: {e}")
        
        return False
    
    def _estimate_gif_size(self, input_video: str, duration: float, settings: Dict[str, Any]) -> float:
        """
        Estimate GIF file size based on video properties and settings.
        
        Args:
            input_video: Path to input video
            duration: Video duration in seconds
            settings: Settings dictionary
        
        Returns:
            Estimated size in MB
        """
        try:
            seg_config = self.config_helper.get_segmentation_config()
            estimation = seg_config.get('estimation', {})
            
            # Get estimation parameters
            default_fps = estimation.get('default_fps', 20)
            default_colors = estimation.get('default_colors', 256)
            default_lossy = estimation.get('default_lossy', 60)
            
            # Get video resolution (simplified estimate)
            width = settings.get('width', 360)
            height = settings.get('height', 360)
            if height == -1:
                height = width  # Assume square for estimation
            
            # Estimate frames
            frames = int(duration * default_fps)
            
            # Estimate bytes per frame (rough approximation)
            pixels_per_frame = width * height
            bytes_per_pixel = (default_colors / 256.0) * 0.5  # Rough estimate
            lossy_factor = 1.0 - (default_lossy / 150.0) * 0.4
            
            estimated_bytes = pixels_per_frame * frames * bytes_per_pixel * lossy_factor
            estimated_mb = estimated_bytes / (1024 * 1024)
            
            return max(0.1, estimated_mb)
        except Exception:
            # Fallback: rough estimate based on duration
            return duration * 0.5  # 0.5MB per second rough estimate
    
    def create_segments(self, input_video: str, output_path: str, settings: Dict[str, Any],
                       start_time: float, duration: float) -> Dict[str, Any]:
        """
        Create multiple GIF segments from video.
        
        Args:
            input_video: Path to input video
            output_path: Base output path (segments will be in subdirectory)
            settings: Settings dictionary
            start_time: Start time in seconds
            duration: Total duration to segment
        
        Returns:
            Dictionary with success status, segments_created, total_size_mb, segments_directory
        """
        try:
            if self._shutdown_checker():
                return {'success': False, 'error': 'Shutdown requested before starting segmentation'}
            
            seg_config = self.config_helper.get_segmentation_config()
            
            # Calculate segment duration
            max_segment_duration = min(settings.get('max_duration', 30.0), seg_config.get('max_segment_duration', 35))
            min_segment_duration = seg_config.get('min_segment_duration', 12)
            
            # Determine optimal segment duration based on video length
            base_durations = seg_config.get('base_durations', {})
            if duration <= base_durations.get('short_video_max', 40):
                preferred_duration = base_durations.get('short_segment_duration', 18)
            elif duration <= base_durations.get('medium_video_max', 80):
                preferred_duration = base_durations.get('medium_segment_duration', 22)
            elif duration <= base_durations.get('long_video_max', 120):
                preferred_duration = base_durations.get('long_segment_duration', 25)
            else:
                preferred_duration = base_durations.get('very_long_segment_duration', 28)
            
            segment_duration = min(max_segment_duration, max(min_segment_duration, preferred_duration))
            
            # Calculate number of segments
            num_segments = max(1, math.ceil(duration / segment_duration))
            # Equalize segment durations
            if num_segments > 0:
                equalized_duration = duration / num_segments
                segment_duration = min(max_segment_duration, max(min_segment_duration, equalized_duration))
            
            # Create output directory for segments
            output_dir = os.path.dirname(output_path) if os.path.dirname(output_path) else '.'
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            safe_base_name = self._safe_filename_for_filesystem(base_name)
            segments_dir = os.path.join(output_dir, f"{safe_base_name}_segments")
            
            try:
                os.makedirs(segments_dir, exist_ok=True)
            except Exception as e:
                logger.warning(f"Failed to create segments directory: {e}")
                return {'success': False, 'error': f'Failed to create segments directory: {e}'}
            
            successful_segments = 0
            total_size = 0.0
            max_size_mb = settings.get('max_size_mb', 10.0)
            
            # Get optimal number of workers
            max_workers = self._calculate_optimal_workers()
            
            logger.info(f"Starting parallel GIF segmentation: {num_segments} segments with {max_workers} workers")
            
            # Process segments in parallel
            def _process_segment(i: int) -> Tuple[bool, float, str]:
                """Process a single segment"""
                try:
                    if self._shutdown_checker():
                        return (False, 0.0, f"{safe_base_name}_segment_{i+1:02d}.gif")
                    
                    segment_start = start_time + (i * segment_duration)
                    raw_end = start_time + duration
                    segment_end = min(raw_end, segment_start + segment_duration)
                    if i == num_segments - 1:
                        segment_end = raw_end
                    
                    segment_duration_actual = max(0.0, segment_end - segment_start)
                    if segment_duration_actual <= 0:
                        return (False, 0.0, f"zero-length segment {i+1}")
                    
                    segment_name = f"{safe_base_name}_segment_{i+1:02d}.gif"
                    segment_path = os.path.join(segments_dir, segment_name)
                    
                    # Apply quality scaling for longer segments if enabled
                    segment_settings = settings.copy()
                    quality_scaling = seg_config.get('quality_scaling', {})
                    if quality_scaling.get('enabled', True) and segment_duration_actual > max_segment_duration * 0.8:
                        # Apply quality reduction for longer segments
                        fps_reduction = quality_scaling.get('long_segment_fps_reduction', 0.9)
                        color_reduction = quality_scaling.get('long_segment_color_reduction', 0.9)
                        
                        if 'fps' in segment_settings:
                            segment_settings['fps'] = int(segment_settings['fps'] * fps_reduction)
                        if 'colors' in segment_settings:
                            segment_settings['colors'] = int(segment_settings['colors'] * color_reduction)
                    
                    # Create segment
                    result = self.gif_generator.create_gif(
                        input_video=input_video,
                        output_path=segment_path,
                        platform=None,
                        max_size_mb=max_size_mb,
                        start_time=segment_start,
                        duration=segment_duration_actual,
                        disable_segmentation=True  # Prevent nested segmentation
                    )
                    
                    # Retry with reduced duration if size limit exceeded
                    if not result.get('success', False):
                        err = result.get('error', 'Unknown error')
                        if 'exceeds limit' in str(err).lower() or 'size' in str(err).lower():
                            reduced_duration = max(5.0, segment_duration_actual * 0.8)
                            if reduced_duration < segment_duration_actual:
                                logger.info(f"Retrying segment {i+1} with reduced duration: {reduced_duration:.2f}s")
                                try:
                                    if os.path.exists(segment_path):
                                        os.remove(segment_path)
                                    
                                    result = self.gif_generator.create_gif(
                                        input_video=input_video,
                                        output_path=segment_path,
                                        platform=None,
                                        max_size_mb=max_size_mb,
                                        start_time=segment_start,
                                        duration=reduced_duration,
                                        disable_segmentation=True
                                    )
                                except Exception as retry_e:
                                    logger.warning(f"Exception during segment {i+1} retry: {retry_e}")
                    
                    if result.get('success', False):
                        sz = float(result.get('size_mb', 0.0) or 0.0)
                        logger.info(f"Created segment {i+1}/{num_segments}: {segment_name} ({sz:.2f}MB)")
                        return (True, sz, segment_name)
                    else:
                        err = result.get('error', 'Unknown error')
                        logger.warning(f"Failed to create segment {i+1}/{num_segments}: {err}")
                        return (False, 0.0, segment_name)
                
                except Exception as e:
                    logger.warning(f"Exception in segment {i+1}: {e}")
                    return (False, 0.0, f"{safe_base_name}_segment_{i+1:02d}.gif")
            
            # Execute in parallel
            indices = list(range(num_segments))
            results: List[Tuple[bool, float, str]] = []
            completed_count = 0
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(_process_segment, i) for i in indices]
                pending = set(futures)
                
                try:
                    while pending:
                        if self._shutdown_checker():
                            logger.info("Shutdown requested during segmentation, cancelling tasks...")
                            for f in list(pending):
                                f.cancel()
                            try:
                                executor.shutdown(wait=False, cancel_futures=True)
                            except TypeError:
                                executor.shutdown(wait=False)
                            break
                        
                        done, pending = wait(pending, timeout=0.2, return_when=FIRST_COMPLETED)
                        for fut in done:
                            try:
                                ok, size_mb, name = fut.result()
                                if ok:
                                    successful_segments += 1
                                    total_size += size_mb
                                    completed_count += 1
                                    logger.info(f"Segment {completed_count}/{num_segments} completed: {name} ({size_mb:.2f}MB)")
                            except Exception as e:
                                logger.warning(f"Segment future failed: {e}")
                finally:
                    if self._shutdown_checker():
                        try:
                            executor.shutdown(wait=False, cancel_futures=True)
                        except TypeError:
                            executor.shutdown(wait=False)
            
            if self._shutdown_checker():
                return {'success': False, 'error': 'Shutdown requested during segmentation', 'segments_created': successful_segments}
            
            if successful_segments > 0:
                logger.info(f"All {successful_segments} GIF segments completed successfully. Total size: {total_size:.2f}MB")
                
                # Create comprehensive summary for the segments folder
                try:
                    if hasattr(self.gif_generator, '_create_comprehensive_segments_summary'):
                        self.gif_generator._create_comprehensive_segments_summary(segments_dir, safe_base_name)
                except Exception as e:
                    logger.warning(f"Could not create comprehensive summary: {e}")
                
                return {
                    'success': True,
                    'segments_created': successful_segments,
                    'total_size_mb': total_size,
                    'segments_directory': segments_dir,
                    'method': 'segmentation'
                }
            else:
                return {'success': False, 'error': 'Failed to create any segments'}
        
        except Exception as e:
            logger.error(f"Error creating segmented GIFs: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}
    
    def _calculate_optimal_workers(self) -> int:
        """Calculate optimal number of workers for parallel segmentation"""
        try:
            mp_config = self.config_helper.get_multiprocessing_config()
            
            if not mp_config.get('enabled', True):
                return 1
            
            max_concurrent = mp_config.get('max_concurrent_segments', 4)
            use_dynamic = mp_config.get('use_dynamic_analysis', True)
            
            if not use_dynamic:
                return max_concurrent
            
            # Dynamic analysis based on system capabilities
            import os as os_module
            cpu_count = os_module.cpu_count() or 4
            
            analysis_mode = mp_config.get('analysis_mode', 'recommended')
            
            if analysis_mode == 'conservative':
                # Use fewer workers to avoid overwhelming system
                workers = min(max_concurrent, max(1, cpu_count // 2))
            elif analysis_mode == 'maximum_safe':
                # Use more workers but still safe
                workers = min(max_concurrent, max(1, cpu_count - 1))
            else:  # recommended
                # Balanced approach
                workers = min(max_concurrent, max(1, (cpu_count * 3) // 4))
            
            return max(1, workers)
        except Exception:
            return 2  # Safe fallback
    
    def _safe_filename_for_filesystem(self, filename: str) -> str:
        """
        Sanitize filename to avoid filesystem issues with problematic Unicode characters.
        
        Args:
            filename: Original filename
        
        Returns:
            Sanitized filename safe for filesystem use
        """
        # Remove or replace problematic characters
        # Keep alphanumeric, spaces, hyphens, underscores, dots
        safe = re.sub(r'[^\w\s\-_.]', '_', filename)
        # Remove leading/trailing spaces and dots
        safe = safe.strip(' .')
        # Limit length
        if len(safe) > 200:
            safe = safe[:200]
        return safe if safe else 'gif_segment'

