"""
GIF Generation Module
Converts videos to optimized GIFs for social media platforms
"""

import os
import subprocess
import shutil
from typing import Dict, Any, Optional, Tuple, List, Callable
import logging
from pathlib import Path
from PIL import Image, ImageSequence
import cv2
import time # Added for time.sleep
import json
import hashlib
import re
from concurrent.futures import ThreadPoolExecutor
import threading

from ..config_manager import ConfigManager
from ..ffmpeg_utils import FFmpegUtils
from .gif_optimizer import GifOptimizer
from .gif_segmenter import GifSegmenter
from .gif_config import GifConfigHelper
from .gif_utils import temp_file_context, temp_dir_context, safe_file_operation, get_gif_info

logger = logging.getLogger(__name__)

class GifGenerator:
    """Advanced GIF generator with optimization and platform-specific settings"""
    
    def __init__(self, config_manager: ConfigManager, shutdown_checker: Optional[Callable[[], bool]] = None):
        self.config = config_manager
        self.config_helper = GifConfigHelper(config_manager)
        self.ffmpeg_utils = FFmpegUtils()
        self._shutdown_checker: Callable[[], bool] = shutdown_checker or (lambda: False)
        self.optimizer = GifOptimizer(config_manager, shutdown_checker=self._is_shutdown_requested)
        self.segmenter = GifSegmenter(config_manager, self, shutdown_checker=self._is_shutdown_requested)
        self.shutdown_requested = False
        self.current_ffmpeg_process = None
        self._ffmpeg_processes = []  # Track all running FFmpeg processes
        self._shutdown_lock = threading.Lock()  # Protect shutdown state
        
        # Get temp directory
        try:
            self.temp_dir = self.config.get_temp_dir()
            if self.temp_dir:
                os.makedirs(self.temp_dir, exist_ok=True)
        except (OSError, PermissionError, AttributeError):
            self.temp_dir = os.path.abspath('temp')
            try:
                os.makedirs(self.temp_dir, exist_ok=True)
            except (OSError, PermissionError):
                pass
    
    def _is_shutdown_requested(self) -> bool:
        """Return True if shutdown was requested locally or by external checker."""
        try:
            return bool(self.shutdown_requested or (self._shutdown_checker() if self._shutdown_checker else False))
        except Exception:
            return bool(self.shutdown_requested)

    def _safe_text_for_logging(self, text: str) -> str:
        """Return a version of text safe for cp1252 consoles by replacing unsupported chars."""
        try:
            return text.encode('cp1252', errors='replace').decode('cp1252', errors='replace')
        except Exception:
            try:
                return text.encode('ascii', errors='replace').decode('ascii', errors='replace')
            except Exception:
                return ''.join(c if ord(c) < 128 else '?' for c in text)

    def create_gif(self, input_video: str, output_path: str, platform: str = None,
                  max_size_mb: int = None, start_time: float = 0, 
                  duration: float = None, disable_segmentation: bool = False) -> Dict[str, Any]:
        """
        Create optimized GIF from video with iterative quality optimization
        
        Args:
            input_video: Path to input video file
            output_path: Path for output GIF
            platform: Target platform (twitter, discord, slack, etc.)
            max_size_mb: Maximum file size in MB
            start_time: Start time in seconds (default: 0)
            duration: Duration in seconds (default: use platform/config limit)
            disable_segmentation: If True, prevents the GIF generator from segmenting the video
            
        Returns:
            Dict with success status and metadata
        """
        try:
            # Validate input
            if not os.path.exists(input_video):
                return {'success': False, 'error': f'Input video not found: {input_video}'}
            
            # Get platform settings
            settings = self._get_platform_settings(platform, max_size_mb)
            
            # Get video info
            video_info = self.ffmpeg_utils.get_video_info(input_video)
            if not video_info:
                return {'success': False, 'error': 'Could not get video information'}
            
            # Determine duration
            if duration is None:
                duration = min(settings['max_duration'], video_info.get('duration', settings['max_duration']))
            
            # Quick feasibility pre-check to avoid long attempts when single-file is impossible
            try:
                quick_enabled = bool(self.config.get('gif_settings.single_gif.quick_feasibility.enabled', True))
            except Exception:
                quick_enabled = True
            if (not disable_segmentation) and quick_enabled and (not self._is_segmented_video(input_video)):
                logger.info("Starting quick single-GIF feasibility check before full optimization")
                print("    🧪 Quick feasibility check: testing harsh single-file settings...")
                feasible, feas_size_mb, feas_path = self._quick_single_feasibility_check(input_video, settings, start_time, duration)
                if feasible:
                    try:
                        target_mb = float(settings.get('max_size_mb', 10.0) or 10.0)
                    except Exception:
                        target_mb = 10.0
                    logger.info(f"Feasibility check PASSED: size={feas_size_mb:.2f}MB <= target {target_mb:.2f}MB; attempting single-file")
                    # Adopt feasibility artifact if present to skip duplicate work
                    if feas_path and os.path.exists(feas_path):
                        try:
                            out_dir = os.path.dirname(output_path)
                            if out_dir and not os.path.exists(out_dir):
                                os.makedirs(out_dir)
                            shutil.move(feas_path, output_path)
                            print(f"    ✅ Feasibility artifact adopted: {feas_size_mb:.2f}MB <= target {target_mb:.2f}MB")
                            return {
                                'success': True,
                                'size_mb': feas_size_mb,
                                'duration': duration,
                                'fps': settings['fps'],
                                'scale': settings.get('scale') or settings.get('width'),
                                'method': 'single_feasibility_adopted'
                            }
                        except Exception as adopt_e:
                            logger.debug(f"Could not adopt feasibility artifact, proceeding to full encode: {adopt_e}")
                            # Clean up temp file if adoption failed
                            try:
                                import time as time_module
                                max_retries = 3
                                retry_delay = 0.1
                                for attempt in range(max_retries):
                                    try:
                                        os.remove(feas_path)
                                        logger.debug(f"Cleaned up feasibility temp file after adoption failure: {os.path.basename(feas_path)}")
                                        break
                                    except (OSError, PermissionError) as e:
                                        if attempt < max_retries - 1:
                                            time_module.sleep(retry_delay * (attempt + 1))
                                        else:
                                            logger.warning(f"Could not clean up feasibility temp file after adoption failure: {os.path.basename(feas_path)}: {e}")
                            except Exception as cleanup_e:
                                logger.debug(f"Error cleaning up feasibility temp file after adoption failure: {cleanup_e}")
                    print(f"    ✅ Feasibility check passed: {feas_size_mb:.2f}MB <= target {target_mb:.2f}MB; attempting single-file GIF first")
                else:
                    logger.info("Feasibility check FAILED: falling back to segmentation to save time")
                    try:
                        max_size_mb = float(settings.get('max_size_mb', 10.0) or 10.0)
                    except Exception:
                        max_size_mb = 10.0
                    if feas_size_mb is not None:
                        print(f"    ↩️  Feasibility check failed: {feas_size_mb:.2f}MB > target {max_size_mb:.2f}MB; falling back to segmentation")
                    else:
                        print("    ↩️  Feasibility check failed: falling back to segmentation")
                    
                    # Clean up feasibility temp file if it exists (not adopted since check failed)
                    if feas_path and os.path.exists(feas_path):
                        try:
                            import time as time_module
                            # On Windows, files may be locked briefly after subprocess closes
                            # Retry with small delays to handle file locking
                            max_retries = 3
                            retry_delay = 0.1  # 100ms
                            
                            for attempt in range(max_retries):
                                try:
                                    os.remove(feas_path)
                                    logger.debug(f"Cleaned up feasibility temp file: {os.path.basename(feas_path)}")
                                    break
                                except (OSError, PermissionError) as e:
                                    if attempt < max_retries - 1:
                                        time_module.sleep(retry_delay * (attempt + 1))
                                    else:
                                        # Last attempt failed, log but don't raise
                                        logger.warning(f"Could not clean up {os.path.basename(feas_path)}: {e}")
                        except Exception as e:
                            logger.debug(f"Error cleaning up feasibility temp file {feas_path}: {e}")
                    
                    return self.segmenter.create_segments(input_video, output_path, settings, start_time, duration)

            # Prefer single file before splitting: attempt single-GIF first
            if not disable_segmentation and self._should_split_video(input_video, duration, settings):
                logger.info(
                    f"Segmentation suggested (duration {duration:.1f}s or estimated size). Trying single-file creation first."
                )
                single_result = self._create_single_gif(input_video, output_path, settings, start_time, duration)
                if single_result and single_result.get('success', False):
                    logger.info("Single-file GIF succeeded; skipping segmentation")
                    return single_result
                logger.info("Single-file attempt failed or not under target; proceeding with segmentation as fallback")
                print("    ↩️  Single-file attempt failed/not under target; proceeding with segmentation")
                return self.segmenter.create_segments(input_video, output_path, settings, start_time, duration)
            
            # Create single GIF with robust fallback
            logger.info(f"Creating single GIF: {self._safe_text_for_logging(str(input_video))} -> {self._safe_text_for_logging(str(output_path))}")
            single_result = None
            try:
                single_result = self._create_single_gif(input_video, output_path, settings, start_time, duration)
            except Exception as e:
                logger.warning(f"Single-file GIF generation threw exception: {e}")
                single_result = {'success': False, 'error': str(e)}
            if single_result and single_result.get('success', False):
                return single_result
            # If single-file failed and segmentation is allowed, fallback to segmentation even if split wasn't suggested
            if not disable_segmentation and (not self._is_segmented_video(input_video)):
                logger.info("Single-file GIF failed; falling back to segmentation")
                print("    ↩️  Single-file GIF failed; falling back to segmentation")
                return self.segmenter.create_segments(input_video, output_path, settings, start_time, duration)
            return single_result
            
        except Exception as e:
            logger.error(f"Error creating GIF: {e}")
            return {'success': False, 'error': str(e)}
    
    def _calculate_optimal_scale(self, video_info: Dict[str, Any], target_width: int, max_size_mb: float) -> Tuple[int, int]:
        """
        Calculate optimal scaling dimensions that preserve aspect ratio and quality
        
        Args:
            video_info: Video information from FFmpeg
            target_width: Target width in pixels
            max_size_mb: Maximum file size in MB
            
        Returns:
            Tuple of (width, height) for optimal scaling
        """
        try:
            original_width = video_info.get('width', 1920)
            original_height = video_info.get('height', 1080)
            
            if not original_width or not original_height:
                logger.warning(f"Invalid video dimensions: {original_width}x{original_height}, using fallback")
                return target_width, -2  # Fallback to original behavior
            
            # Calculate aspect ratio
            aspect_ratio = original_width / original_height
            
            # Calculate height that maintains aspect ratio
            calculated_height = int(target_width / aspect_ratio)
            
            # Ensure height is even (required for some codecs)
            if calculated_height % 2 != 0:
                calculated_height = calculated_height + 1
            
            # Log the scaling calculation for debugging
            logger.info(f"GIF scaling calculation: original {original_width}x{original_height} (AR: {aspect_ratio:.3f}) -> target {target_width}x{calculated_height}")
            
            # If the calculated dimensions would result in a very small file,
            # we can afford to use higher quality
            estimated_pixels = target_width * calculated_height
            if estimated_pixels < 300000:  # Less than 300k pixels
                # Use higher quality scaling
                logger.info(f"Small estimated file size ({estimated_pixels:,} pixels), using calculated dimensions")
                return target_width, calculated_height
            
            # For larger videos, use the calculated dimensions
            logger.info(f"Standard estimated file size ({estimated_pixels:,} pixels), using calculated dimensions")
            return target_width, calculated_height
            
        except Exception as e:
            logger.warning(f"Error calculating optimal scale, using fallback: {e}")
            return target_width, -2  # Fallback to original behavior

    def _get_platform_settings(self, platform: str, max_size_mb: Optional[float]) -> Dict[str, Any]:
        """Get platform-specific settings using config helper"""
        # Get base optimization config
        opt_config = self.config_helper.get_optimization_config()
        
        # Start with base config
        settings = {
            'max_size_mb': opt_config['max_file_size_mb'],
            'max_duration': opt_config['max_duration_seconds'],
            'fps': opt_config['fps'],
            'width': opt_config['width'],
            'height': opt_config['height'],
            'colors': opt_config['colors'],
            'dither': opt_config['dither'],
            'lossy': opt_config['lossy'],
            'palette_max_colors': opt_config['colors']
        }
        
        # Override with platform-specific settings if platform specified
        if platform:
            platform_settings = self.config_helper.get_platform_settings(platform.lower())
            settings.update({
                'max_size_mb': platform_settings.get('max_file_size_mb', settings['max_size_mb']),
                'max_duration': platform_settings.get('max_duration', settings['max_duration']),
                'width': platform_settings.get('max_width', settings['width']),
                'height': platform_settings.get('max_height', settings['height']),
                'colors': platform_settings.get('colors', settings['colors']),
                'dither': platform_settings.get('dither', settings['dither']),
                'lossy': platform_settings.get('lossy', settings['lossy']),
                'palette_max_colors': platform_settings.get('colors', settings['colors'])
            })
        
        # Override with provided max_size_mb if specified
        if max_size_mb is not None:
            settings['max_size_mb'] = max_size_mb
        
        # Merge quality floors
        quality_floors = self.config_helper.get_quality_floors()
        if quality_floors.get('enforce', True):
            settings['min_width'] = quality_floors.get('min_width', 0)
            settings['min_fps'] = quality_floors.get('min_fps', 0)
            settings['floors_enforce'] = True
        
        return settings
    
    def _should_split_video(self, input_video: str, duration: float, settings: Dict[str, Any]) -> bool:
        """Determine if video should be split into segments"""
        # Check if this is already a segmented video (from video segmentation)
        # If so, disable GIF segmentation entirely
        if self._is_segmented_video(input_video):
            logger.info(f"Video appears to be from segmentation, disabling GIF segmentation: {input_video}")
            return False
        
        # Use segmenter's should_segment method
        return self.segmenter.should_segment(input_video, duration, settings)
    
    def _is_segmented_video(self, input_video: str) -> bool:
        """Check if the video is from a segmented video folder"""
        try:
            # Check if the video is in a segments folder
            video_path = Path(input_video)
            parent_dir = video_path.parent
            
            # Check if parent directory name contains '_segments'
            if '_segments' in parent_dir.name:
                return True
            
            # Check if the video filename contains segment indicators (case-insensitive)
            video_name = video_path.stem.lower()
            if any(indicator in video_name for indicator in ['segment', 'part', 'split']):
                return True
            
            # Check if the video is in a segments folder (case insensitive)
            if 'segments' in parent_dir.name.lower():
                return True
            
            # Check if any parent directory contains '_segments' (for nested paths)
            try:
                for parent in parent_dir.parents:
                    if '_segments' in parent.name or 'segments' in parent.name.lower():
                        return True
            except Exception:
                pass
            
            # Check filename pattern for segment naming (e.g., *_segment_01.*)
            filename_lower = video_path.name.lower()
            if re.search(r'_segment_\d+\.', filename_lower) or re.search(r'_part_\d+\.', filename_lower):
                return True
            
            return False
        except Exception as e:
            logger.warning(f"Error checking if video is segmented: {e}")
            return False
    
    def _estimate_gif_size(self, input_video: str, duration: float, settings: Dict[str, Any]) -> float:
        """Estimate GIF file size based on video characteristics"""
        try:
            # Get video resolution
            resolution = self.ffmpeg_utils.get_video_resolution(input_video)
            if not resolution:
                return settings['max_size_mb']  # Conservative estimate
            
            width, height = resolution
            
            # Scale down if needed
            target_width = settings.get('scale') or settings.get('width', 360)
            if width > target_width:
                scale_factor = target_width / width
                width = int(width * scale_factor)
                height = int(height * scale_factor)
            
            # Rough size estimation (very approximate)
            pixels = width * height
            fps = settings['fps']
            frames = duration * fps
            
            # Rough size calculation (this is very approximate)
            estimated_size_mb = (pixels * frames * 0.1) / (1024 * 1024)  # Rough compression factor
            
            return estimated_size_mb
            
        except Exception as e:
            logger.warning(f"Could not estimate GIF size: {e}")
            return settings['max_size_mb']  # Conservative estimate
    
    def _create_single_gif(self, input_video: str, output_path: str, settings: Dict[str, Any], 
                          start_time: float, duration: float, deadline: Optional[float] = None) -> Dict[str, Any]:
        """Create a single optimized GIF"""
        try:
            # Check if input video still exists
            if not os.path.exists(input_video):
                return {'success': False, 'error': f'Input video no longer exists: {input_video}'}
            
            # Create output directory (idempotent, race-safe)
            output_dir = os.path.dirname(output_path)
            if output_dir:
                try:
                    os.makedirs(output_dir, exist_ok=True)
                except FileExistsError:
                    pass
            
            # Optionally precompute palette in parallel while running split-palette encode
            enable_parallel_palette = bool(self.config.get('gif_settings.performance.single_gif_parallel.precompute_palette', True))
            palette_future = None
            palette_tmp_path: Optional[str] = None
            palette_cancel_event: Optional[threading.Event] = None
            palette_generator: Optional[GifGenerator] = None

            if enable_parallel_palette:
                try:
                    palette_cancel_event = threading.Event()
                    # Use a dedicated generator so we can cancel only the palette run without affecting main encode
                    palette_generator = GifGenerator(
                        self.config,
                        shutdown_checker=lambda: bool(self._is_shutdown_requested() or (palette_cancel_event.is_set() if palette_cancel_event else False))
                    )
                    def _gen_palette() -> Optional[str]:
                        return palette_generator._generate_palette(input_video, settings, start_time, duration)
                    with ThreadPoolExecutor(max_workers=1) as _exec:
                        palette_future = _exec.submit(_gen_palette)
                        try:
                            # Run split-palette while palette is being generated in background
                            split_success = self._create_gif_split_palette(input_video, output_path, settings, start_time, duration)
                            if split_success:
                                success = True
                                # Cancel palette work if still running
                                if palette_cancel_event:
                                    palette_cancel_event.set()
                                # Wait briefly for palette future to acknowledge cancellation (best-effort)
                                try:
                                    if palette_future and not palette_future.done():
                                        # Give it a moment to respond to shutdown signal
                                        import time as time_module
                                        time_module.sleep(0.1)
                                        # Try to cancel if not started
                                        if not palette_future.running():
                                            palette_future.cancel()
                                except Exception:
                                    pass
                            else:
                                # Need two-step path. Obtain palette result (or compute synchronously if failed)
                                try:
                                    palette_tmp_path = palette_future.result(timeout=60) if palette_future else None
                                except Exception:
                                    palette_tmp_path = None
                                if not palette_tmp_path:
                                    # Background generation did not succeed; generate palette now
                                    palette_tmp_path = self._generate_palette(input_video, settings, start_time, duration)
                                if not palette_tmp_path:
                                    return {'success': False, 'error': 'Failed to generate palette'}
                                # Double-check input video exists before creating GIF
                                if not os.path.exists(input_video):
                                    if os.path.exists(palette_tmp_path):
                                        os.unlink(palette_tmp_path)
                                    return {'success': False, 'error': f'Input video disappeared during processing: {input_video}'}
                                success = self._create_gif_with_palette(input_video, output_path, palette_tmp_path, settings, start_time, duration)
                        finally:
                            # Ensure executor and futures are properly cleaned up
                            try:
                                # Cancel palette future if still pending
                                if palette_future and not palette_future.done():
                                    if palette_cancel_event:
                                        palette_cancel_event.set()
                                    try:
                                        palette_future.cancel()
                                    except Exception:
                                        pass
                                # Wait briefly for any running tasks to finish (with timeout)
                                try:
                                    if palette_future and not palette_future.done():
                                        import time as time_module
                                        palette_future.result(timeout=0.5)  # Short timeout to avoid blocking
                                except Exception:
                                    pass
                            except Exception:
                                pass
                finally:
                    # Don't delete cached palettes - preserve them for future reuse
                    # Only clean up temporary palettes that are not in the cache directory
                    try:
                        if palette_tmp_path and os.path.exists(palette_tmp_path):
                            cache_dir = os.path.join(self.config.get_temp_dir(), 'palette_cache')
                            if not palette_tmp_path.startswith(cache_dir):
                                # Only delete if it's not a cached palette
                                os.unlink(palette_tmp_path)
                    except Exception:
                        pass
            else:
                # Non-parallel original flow
                split_success = self._create_gif_split_palette(input_video, output_path, settings, start_time, duration)
                if not split_success:
                    palette_path = self._generate_palette(input_video, settings, start_time, duration)
                    if not palette_path:
                        return {'success': False, 'error': 'Failed to generate palette'}
                    if not os.path.exists(input_video):
                        if os.path.exists(palette_path):
                            os.unlink(palette_path)
                        return {'success': False, 'error': f'Input video disappeared during processing: {input_video}'}
                    success = self._create_gif_with_palette(input_video, output_path, palette_path, settings, start_time, duration)
                    # Don't delete cached palettes - preserve them for future reuse
                    # Only clean up temporary palettes that are not in the cache directory
                    if os.path.exists(palette_path):
                        cache_dir = os.path.join(self.config.get_temp_dir(), 'palette_cache')
                        if not palette_path.startswith(cache_dir):
                            # Only delete if it's not a cached palette
                            os.unlink(palette_path)
                else:
                    success = True
            
            if success:
                # Get final file size
                if os.path.exists(output_path):
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    
                    # If still over limit, use optimizer to target size precisely
                    if size_mb > settings['max_size_mb']:
                        logger.info(f"GIF size ({size_mb:.2f}MB) exceeds limit ({settings['max_size_mb']:.2f}MB), optimizing...")
                        optimization_succeeded = False
                        try:
                            # Use optimizer to optimize the existing GIF file
                            # Pass source video if available for better re-encoding
                            optimization_succeeded = self.optimizer.optimize_gif(
                                output_path, settings['max_size_mb'], 
                                source_video=input_video if os.path.exists(input_video) else None
                            )
                            if optimization_succeeded:
                                # Verify final size and update size_mb to reflect optimized file
                                try:
                                    final_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                                    logger.info(f"Optimization completed: {final_size_mb:.2f}MB <= {settings['max_size_mb']:.2f}MB")
                                    # Update size_mb to reflect the optimized file size
                                    size_mb = final_size_mb
                                except Exception as size_e:
                                    logger.warning(f"Failed to verify final size: {size_e}")
                                    # Fallback: re-stat the file to get current size
                                    try:
                                        if os.path.exists(output_path):
                                            size_mb = os.path.getsize(output_path) / (1024 * 1024)
                                    except Exception:
                                        pass
                        except Exception as e:
                            logger.warning(f"Optimization failed: {e}")
                            optimization_succeeded = False
                        
                        if not optimization_succeeded:
                            # Get the actual final size from the file that exists
                            final_size_mb = None
                            try:
                                if os.path.exists(output_path):
                                    final_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                            except Exception:
                                pass
                            
                            # Fallback to original size_mb if we can't get file size
                            if final_size_mb is None:
                                final_size_mb = size_mb
                            
                            logger.error(f"Final GIF exceeds limit after optimization attempts: {final_size_mb:.2f}MB > {settings['max_size_mb']:.2f}MB")
                            
                            # Clean up output file if it exists and is over target
                            try:
                                if os.path.exists(output_path):
                                    # Check if file is actually over target before deleting
                                    current_size = os.path.getsize(output_path) / (1024 * 1024)
                                    if current_size > settings['max_size_mb']:
                                        os.remove(output_path)
                            except Exception as cleanup_e:
                                logger.warning(f"Failed to clean up oversized output file: {cleanup_e}")
                            
                            return {
                                'success': False,
                                'error': f"Final GIF size ({final_size_mb:.2f}MB) exceeds limit ({settings['max_size_mb']:.2f}MB)",
                                'size_mb': final_size_mb
                            }
                    
                    # Re-stat the file to ensure we have the most current size before returning
                    # This handles cases where optimization succeeded but size_mb wasn't updated
                    try:
                        if os.path.exists(output_path):
                            current_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                            size_mb = current_size_mb
                    except Exception:
                        # If we can't re-stat, use existing size_mb value
                        pass
                    
                    return {
                        'success': True,
                        'size_mb': size_mb,
                        'duration': duration,
                        'fps': settings['fps'],
                        'scale': settings.get('scale') or settings.get('width', 360)
                    }
                else:
                    return {'success': False, 'error': 'GIF file not created'}
            else:
                return {'success': False, 'error': 'Failed to create GIF'}
                
        except Exception as e:
            logger.error(f"Error creating single GIF: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_palette(self, input_video: str, settings: Dict[str, Any], 
                         start_time: float, duration: float, timeout_override: Optional[int] = None) -> Optional[str]:
        """Generate optimized palette for GIF creation"""
        try:
            if self.shutdown_requested:
                return None
            # Safely handle the input video path
            safe_input_path = FFmpegUtils._safe_file_path(input_video)
            
            # Check if input video exists before starting
            if not os.path.exists(safe_input_path):
                logger.error(f"Input video not found for palette generation: {safe_input_path}")
                return None
            
            # Build filter with mpdecimate + fps + scale (no crop)
            pre_chain = []
            # Support aggressive settings for feasibility checks
            if settings.get('mpdecimate_aggressive'):
                pre_chain.append('mpdecimate=hi=768:lo=64:frac=0.4')
            else:
                pre_chain.append('mpdecimate=hi=512:lo=256:frac=0.3')
            pre_chain.append(f"fps={settings['fps']}")
            
            # Get video info for optimal scaling
            video_info = self.ffmpeg_utils.get_video_info(safe_input_path)
            # Prefer explicit width/height when provided (feasibility path may set width directly)
            if 'width' in settings and int(settings.get('width') or 0) > 0:
                optimal_width = int(settings['width'])
                optimal_height = -2 if int(settings.get('height', -1) or -1) == -1 else int(settings['height'])
            else:
                optimal_width, optimal_height = self._calculate_optimal_scale(
                    video_info, settings.get('scale', settings.get('width', 480)), settings['max_size_mb']
                )
            
            # Enforce quality floors if configured
            try:
                if settings.get('floors_enforce'):
                    min_w = int(settings.get('min_width') or 0)
                    min_f = int(settings.get('min_fps') or 0)
                    if optimal_width < min_w:
                        optimal_width = min_w
                    if int(settings.get('fps') or 0) < min_f:
                        settings['fps'] = min_f
            except Exception:
                pass

            # Improved scaling: preserve aspect ratio better and use higher quality
            if optimal_height == -2:
                # Auto height already preserves aspect ratio
                pre_chain.append(f"scale={optimal_width}:-2:flags=lanczos")
            else:
                # Explicit dimensions: use force_original_aspect_ratio=decrease to preserve aspect ratio
                pre_chain.append(f"scale={optimal_width}:{optimal_height}:flags=lanczos:force_original_aspect_ratio=decrease")
            max_colors = int(settings.get('palette_max_colors', settings.get('colors', 256)))
            vf = ','.join(pre_chain + [f"palettegen=max_colors={max_colors}:stats_mode=diff:reserve_transparent=1"])  # Improved palette settings

            # Palette cache directory (writable user temp)
            cache_dir = os.path.join(self.config.get_temp_dir(), 'palette_cache')
            try:
                os.makedirs(cache_dir, exist_ok=True)
            except FileExistsError:
                pass
            
            # Generate cache key using calculated dimensions (not scale from settings) for accurate matching
            # Use exact fps for cache precision (avoid false cache hits from rounding)
            fps_normalized = float(settings.get('fps', 20))
            # Use exact colors for cache precision (avoid false cache hits from normalization)
            colors_normalized = max_colors
            
            key_payload = {
                'path': safe_input_path,
                'mtime': os.path.getmtime(safe_input_path),
                'start': float(start_time or 0),
                'duration': float(duration or 0),
                'fps': fps_normalized,
                'width': optimal_width,
                'height': optimal_height,
                'colors': colors_normalized,
                'mpdecimate_aggressive': settings.get('mpdecimate_aggressive', False),
                # Include mpdecimate settings for cache precision (avoid false cache hits)
                'mpdecimate_hi': 768 if settings.get('mpdecimate_aggressive') else 512,
                'mpdecimate_lo': 64 if settings.get('mpdecimate_aggressive') else 256,
                'mpdecimate_frac': 0.4 if settings.get('mpdecimate_aggressive') else 0.3,
            }
            key_str = json.dumps(key_payload, sort_keys=True, separators=(',', ':'))
            digest = hashlib.sha1(key_str.encode('utf-8')).hexdigest()
            palette_path = os.path.join(cache_dir, f"pal_{digest}.png")

            # Return cached palette if exists
            if os.path.exists(palette_path):
                logger.debug(f"Using cached palette: {palette_path}")
                return palette_path
            
            # Build FFmpeg command for palette generation over the selected duration
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', safe_input_path,
                '-vf', vf,
                '-f', 'image2',
                palette_path
            ]
            # Add performance flags
            FFmpegUtils.add_ffmpeg_perf_flags(cmd)
            
            # Log the command for debugging
            logger.debug(f"Palette generation command: {' '.join(cmd)}")
            
            logger.debug(f"Generating palette: {' '.join(cmd)}")
            
            # Calculate timeout based on video duration
            # For short videos, use duration * 2, for longer videos use duration * 0.5, but cap at reasonable limits
            if timeout_override is not None:
                calculated_timeout = timeout_override
            else:
                # Base timeout: 30 seconds minimum
                base_timeout = 30
                # Scale with duration: for short videos (<60s) use 2x, for longer use 0.5x
                if duration < 60:
                    calculated_timeout = max(base_timeout, int(duration * 2))
                else:
                    calculated_timeout = max(base_timeout, int(duration * 0.5))
                # Cap at 5 minutes (300 seconds) to prevent extremely long timeouts
                calculated_timeout = min(calculated_timeout, 300)
            
            logger.debug(f"Palette generation timeout: {calculated_timeout}s (based on duration: {duration:.2f}s)")
            
            # Run FFmpeg (attempt 1)
            result = self._run_ffmpeg(cmd, timeout=calculated_timeout)
            if result.returncode == 0 and os.path.exists(palette_path):
                return palette_path

            # If first attempt failed, try a quick cleanup and retry to handle stale/locked files
            try:
                if os.path.exists(palette_path):
                    # Remove zero-byte or partial files before retry
                    try:
                        if os.path.getsize(palette_path) == 0:
                            os.remove(palette_path)
                    except Exception:
                        # Best-effort: attempt attribute clear on Windows before remove
                        try:
                            subprocess.run(['attrib', '-R', '-S', '-H', palette_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                            os.remove(palette_path)
                        except Exception:
                            pass
            except Exception:
                pass

            retry_path = os.path.join(cache_dir, f"pal_{digest}_r1.png")
            retry_cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', safe_input_path,
                '-vf', vf,
                '-f', 'image2',
                retry_path
            ]
            FFmpegUtils.add_ffmpeg_perf_flags(retry_cmd)
            logger.debug(f"Palette generation retry: {' '.join(retry_cmd)}")
            # Use the same calculated timeout for retry
            retry_result = self._run_ffmpeg(retry_cmd, timeout=calculated_timeout)
            if retry_result.returncode == 0 and os.path.exists(retry_path):
                return retry_path

            logger.error(f"Palette generation failed: {result.stderr or ''} | retry: {retry_result.stderr or ''}")
            return None
                
        except FileNotFoundError as e:
            logger.error(f"File not found error during palette generation: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating palette: {e}")
            return None
    
    def _create_gif_with_palette(self, input_video: str, output_path: str, palette_path: str,
                                settings: Dict[str, Any], start_time: float, duration: float, timeout_override: Optional[int] = None) -> bool:
        """Create GIF using generated palette"""
        try:
            if self.shutdown_requested:
                return False
            # Safely handle the input video path
            safe_input_path = FFmpegUtils._safe_file_path(input_video)
            
            # Check if input video and palette still exist
            if not os.path.exists(safe_input_path):
                logger.error(f"Input video not found for GIF creation: {safe_input_path}")
                return False
            
            if not os.path.exists(palette_path):
                logger.error(f"Palette file not found for GIF creation: {palette_path}")
                return False
            
            # Build filter graph with mpdecimate, fps, scale and paletteuse (no crop)
            pre_chain = []
            if settings.get('mpdecimate_aggressive'):
                pre_chain.append('mpdecimate=hi=768:lo=64:frac=0.4')
            else:
                pre_chain.append('mpdecimate=hi=512:lo=256:frac=0.3')
            pre_chain.append(f"fps={settings['fps']}")
            
            # Get video info for optimal scaling
            video_info = self.ffmpeg_utils.get_video_info(safe_input_path)
            
            # Use configured width/height for proper aspect ratio preservation
            target_width = settings.get('width', 360)
            target_height = settings.get('height', -1)
            # Enforce quality floors if configured
            try:
                if settings.get('floors_enforce'):
                    min_w = int(settings.get('min_width') or 0)
                    min_f = int(settings.get('min_fps') or 0)
                    if target_width < min_w:
                        target_width = min_w
                    if int(settings.get('fps') or 0) < min_f:
                        settings['fps'] = min_f
            except Exception:
                pass
            
            if target_height == -1:
                # Preserve aspect ratio by only specifying width
                scale_filter = f"scale={target_width}:-2:flags=lanczos"
                logger.info(f"GIF scaling: preserving aspect ratio with width={target_width}, height=auto")
            else:
                # Use both dimensions if explicitly specified, with force_original_aspect_ratio=decrease to preserve aspect ratio
                scale_filter = f"scale={target_width}:{target_height}:flags=lanczos:force_original_aspect_ratio=decrease"
                logger.info(f"GIF scaling: using specified dimensions {target_width}x{target_height} with aspect ratio preservation")
            
            pre_chain.append(scale_filter)
            # Normalize sample aspect ratio to avoid stretching in some viewers
            pre_chain.append('setsar=1')
            dither = settings.get('dither', settings.get('gif_settings.dither', 'floyd_steinberg'))
            lavfi = ','.join(pre_chain) + f" [x]; [x][1:v] paletteuse=dither={dither}:diff_mode=rectangle:new=1"

            # Build FFmpeg command for GIF creation
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', safe_input_path,
                '-i', palette_path,
                '-lavfi', lavfi,
                '-f', 'gif',
                output_path
            ]
            # Add performance flags
            FFmpegUtils.add_ffmpeg_perf_flags(cmd)
            
            # Log the command for debugging
            logger.debug(f"GIF creation command: {' '.join(cmd)}")
            
            logger.debug(f"Creating GIF: {' '.join(cmd)}")
            
            # Run FFmpeg
            result = self._run_ffmpeg(cmd, timeout=(timeout_override if timeout_override is not None else 120))
            
            if result.returncode == 0 and os.path.exists(output_path):
                return True
            else:
                logger.error(f"GIF creation failed: {result.stderr}")
                return False
                
        except FileNotFoundError as e:
            logger.error(f"File not found error during GIF creation: {e}")
            return False
        except Exception as e:
            logger.error(f"Error creating GIF with palette: {e}")
            return False

    def _create_gif_split_palette(self, input_video: str, output_path: str,
                                  settings: Dict[str, Any], start_time: float, duration: float, timeout_override: Optional[int] = None) -> bool:
        """Create GIF using split palette single-pass pipeline with mpdecimate and optional crop"""
        try:
            if self.shutdown_requested:
                return False
            safe_input_path = FFmpegUtils._safe_file_path(input_video)
            if not os.path.exists(safe_input_path):
                logger.error(f"Input video not found for split-palette GIF: {safe_input_path}")
                return False

            # Build filter chain without crop
            pre = []
            pre.append('mpdecimate=hi=512:lo=256:frac=0.3')
            pre.append(f"fps={settings['fps']}")
            
            # Get video info for optimal scaling
            video_info = self.ffmpeg_utils.get_video_info(safe_input_path)
            
            # Use configured width/height for proper aspect ratio preservation
            target_width = settings.get('width', 360)
            target_height = settings.get('height', -1)
            # Enforce quality floors if configured (consistent with other paths)
            try:
                if settings.get('floors_enforce'):
                    min_w = int(settings.get('min_width') or 0)
                    min_f = int(settings.get('min_fps') or 0)
                    if target_width < min_w:
                        target_width = min_w
                    if int(settings.get('fps') or 0) < min_f:
                        settings['fps'] = min_f
            except Exception:
                pass
            
            if target_height == -1:
                # Preserve aspect ratio by only specifying width
                scale_filter = f"scale={target_width}:-2:flags=lanczos"
                logger.info(f"GIF scaling: preserving aspect ratio with width={target_width}, height=auto")
            else:
                # Use both dimensions if explicitly specified, with force_original_aspect_ratio=decrease to preserve aspect ratio
                scale_filter = f"scale={target_width}:{target_height}:flags=lanczos:force_original_aspect_ratio=decrease"
                logger.info(f"GIF scaling: using specified dimensions {target_width}x{target_height} with aspect ratio preservation")
            
            pre.append(scale_filter)
            # Normalize SAR for consistent display
            pre.append('setsar=1')
            chain = ','.join(pre)

            filter_complex = (
                f"{chain},split[a][b];" 
                f"[a]palettegen=stats_mode=diff:reserve_transparent=1:max_colors=256[p];"
                f"[b][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle:new=1"
            )

            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', safe_input_path,
                '-filter_complex', filter_complex,
                '-f', 'gif',
                output_path
            ]
            FFmpegUtils.add_ffmpeg_perf_flags(cmd)

            logger.debug(f"Split-palette GIF creation: {' '.join(cmd)}")
            result = self._run_ffmpeg(cmd, timeout=(timeout_override if timeout_override is not None else 180))
            return result.returncode == 0 and os.path.exists(output_path)
        except Exception as e:
            logger.debug(f"Split-palette creation failed: {e}")
            return False

    def _detect_crop_filter(self, input_video: str, start_time: float, duration: float) -> str:
        """Detect crop suggestion using ffmpeg cropdetect and return crop filter string or empty"""
        try:
            # Respect configuration: crop detection disabled by default to avoid unintended cropping
            crop_enabled = bool(self.config.get('gif_settings.crop_detection.enabled', False))
            if not crop_enabled:
                return ''

            cmd = [
                'ffmpeg', '-v', 'error',
                '-ss', str(start_time),
                '-t', str(min(5.0, duration) if duration else 5.0),
                '-i', input_video,
                '-vf', 'cropdetect=24:16:0',
                '-frames:v', '100',
                '-f', 'null', '-'
            ]
            FFmpegUtils.add_ffmpeg_perf_flags(cmd)
            result = self._run_ffmpeg(cmd, timeout=20)
            # Parse stderr for last crop=
            crop = None
            for line in (result.stderr or '').splitlines():
                if 'crop=' in line:
                    try:
                        crop = line.split('crop=')[-1].split()[0]
                    except Exception:
                        continue
            if crop and 'x' in crop and ':' in crop:
                # Validate that suggested crop meaningfully removes borders (avoid slight crops)
                try:
                    # Example format: crop=width:height:x:y
                    parts = crop.replace('crop=', '').split(':')
                    cw, ch, cx, cy = [int(float(p)) for p in parts[:4]]
                    info = self.ffmpeg_utils.get_video_info(input_video) or {}
                    iw = int(info.get('width') or 0)
                    ih = int(info.get('height') or 0)
                    if iw > 0 and ih > 0 and cw > 0 and ch > 0:
                        left = max(0, cx)
                        top = max(0, cy)
                        right = max(0, iw - (cx + cw))
                        bottom = max(0, ih - (cy + ch))
                        # Thresholds from config (defaults chosen to prevent minor crops)
                        min_border_px = int(self.config.get('gif_settings.crop_detection.min_border_px', 12))
                        min_border_ratio = float(self.config.get('gif_settings.crop_detection.min_border_ratio', 0.04))
                        hori_ratio = (left + right) / max(1, iw)
                        vert_ratio = (top + bottom) / max(1, ih)
                        significant_hori = (left >= min_border_px or right >= min_border_px or hori_ratio >= min_border_ratio)
                        significant_vert = (top >= min_border_px or bottom >= min_border_px or vert_ratio >= min_border_ratio)
                        if significant_hori or significant_vert:
                            return f"crop={cw}:{ch}:{cx}:{cy}"
                        else:
                            logger.debug("Crop detection ignored: borders below significance thresholds")
                except Exception:
                    # If we cannot validate, err on side of not cropping
                    pass
        except Exception:
            pass
        return ''

    # Shutdown/termination support for graceful exit
    def request_shutdown(self):
        """Request generator to stop and terminate any running FFmpeg process."""
        with self._shutdown_lock:
            if not self.shutdown_requested:  # Only log and terminate if not already shutting down
                logger.info("Shutdown requested for GIF generator")
                self.shutdown_requested = True
                try:
                    if hasattr(self, 'optimizer') and hasattr(self.optimizer, 'request_shutdown'):
                        self.optimizer.request_shutdown()
                except Exception:
                    pass
                logger.info("Terminating all FFmpeg processes in GIF generator...")
                self._terminate_ffmpeg_process()

    def _terminate_ffmpeg_process(self):
        """Terminate all tracked FFmpeg processes gracefully"""
        # Terminate current process
        if self.current_ffmpeg_process and self.current_ffmpeg_process.poll() is None:
            try:
                logger.info("Terminating current FFmpeg process in GIF generator...")
                # Try graceful termination first
                self.current_ffmpeg_process.terminate()

                # Wait a bit for graceful termination
                try:
                    self.current_ffmpeg_process.wait(timeout=5)
                    logger.info("FFmpeg process terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination fails
                    logger.warning("FFmpeg process did not terminate gracefully, forcing kill...")
                    self.current_ffmpeg_process.kill()
                    self.current_ffmpeg_process.wait()
                    logger.info("FFmpeg process killed")

            except (OSError, subprocess.SubprocessError) as e:
                logger.error(f"Error terminating current FFmpeg process in GIF generator: {e}", exc_info=True)
            finally:
                with self._shutdown_lock:
                    if self.current_ffmpeg_process in self._ffmpeg_processes:
                        self._ffmpeg_processes.remove(self.current_ffmpeg_process)
                self.current_ffmpeg_process = None

        # Terminate any other tracked processes
        with self._shutdown_lock:
            processes_to_terminate = list(self._ffmpeg_processes)
        
        for process in processes_to_terminate:
            if process and process.poll() is None:
                try:
                    logger.info("Terminating additional FFmpeg process in GIF generator...")
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        logger.warning("Additional FFmpeg process did not terminate gracefully, forcing kill...")
                        process.kill()
                        process.wait()
                    logger.info("Additional FFmpeg process terminated successfully")
                except (OSError, subprocess.SubprocessError) as e:
                    logger.error(f"Error terminating additional FFmpeg process in GIF generator: {e}", exc_info=True)
                finally:
                    with self._shutdown_lock:
                        if process in self._ffmpeg_processes:
                            self._ffmpeg_processes.remove(process)

    def _run_ffmpeg(self, cmd: list, timeout: int = 120):
        """Run FFmpeg with the ability to terminate early on shutdown."""
        class FFResult:
            pass
        
        # Check if shutdown was requested before starting
        if self._is_shutdown_requested():
            r = FFResult()
            r.stdout = ''
            r.stderr = 'Shutdown requested before FFmpeg execution'
            r.returncode = 1
            return r
        
        process = None
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            self.current_ffmpeg_process = process
            with self._shutdown_lock:
                self._ffmpeg_processes.append(process)
            
            # Poll periodically to check for shutdown requests
            start_time = time.time()
            while process and process.poll() is None:
                if self._is_shutdown_requested():
                    logger.info("Shutdown requested during FFmpeg execution, terminating...")
                    self._terminate_ffmpeg_process()
                    r = FFResult()
                    r.stdout = ''
                    r.stderr = 'Shutdown requested during execution'
                    r.returncode = 1
                    return r
                
                # Check timeout
                if time.time() - start_time > timeout:
                    logger.warning(f"FFmpeg timeout after {timeout}s, terminating...")
                    self._terminate_ffmpeg_process()
                    r = FFResult()
                    r.stdout = ''
                    r.stderr = 'TimeoutExpired'
                    r.returncode = 1
                    return r
                
                time.sleep(0.1)  # Short sleep to avoid busy waiting
            
            # Process completed normally
            if process is None:
                # Process was not created or was terminated, return error result
                r = FFResult()
                r.stdout = ''
                r.stderr = 'Process was terminated or not created'
                r.returncode = 1
                return r
            
            stdout, stderr = process.communicate()
            rc = process.returncode
            
            r = FFResult()
            r.stdout = stdout
            r.stderr = stderr
            r.returncode = rc
            return r
            
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            logger.error(f"Error running FFmpeg: {e}", exc_info=True)
            r = FFResult()
            r.stdout = ''
            r.stderr = str(e)
            r.returncode = 1
            return r
        finally:
            # Always cleanup process tracking
            if process:
                with self._shutdown_lock:
                    if process in self._ffmpeg_processes:
                        self._ffmpeg_processes.remove(process)
            if self.current_ffmpeg_process == process:
                self.current_ffmpeg_process = None
    
    # Old _create_segmented_gifs method removed - now using GifSegmenter

    def _create_comprehensive_segments_summary(self, segments_dir: str, base_name: str) -> None:
        """Create a comprehensive summary across all files in a segments folder.

        - Includes both MP4 and GIF files if present
        - Writes FPS, Duration, Frame Count (estimated for GIF), Resolution, Codec/Bitrate (for MP4), and Size
        - Prefixed with '~' so it's listed last and avoided as a thumbnail
        """
        try:
            summary_path = os.path.join(segments_dir, f"~{base_name}_comprehensive_summary.txt")
            temp_path = os.path.join(segments_dir, f"~{base_name}_comprehensive_summary.txt.tmp")

            # Collect files
            entries = [e for e in os.listdir(segments_dir) if os.path.isfile(os.path.join(segments_dir, e))]
            mp4_files = sorted([e for e in entries if e.lower().endswith('.mp4')])
            gif_files = sorted([e for e in entries if e.lower().endswith('.gif')])

            if not mp4_files and not gif_files:
                return

            import time as _time

            total_mp4_size = 0.0
            total_mp4_duration = 0.0
            total_mp4_frames = 0

            total_gif_size = 0.0
            total_gif_duration = 0.0
            total_gif_frames = 0

            # Best-effort clear attributes on existing file
            try:
                if os.path.exists(summary_path):
                    subprocess.run(['attrib', '-R', '-S', '-H', summary_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
            except Exception:
                pass

            # Write to temp file first
            try:
                if os.path.exists(temp_path):
                    os.remove(temp_path)
            except Exception:
                pass

            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write("Comprehensive Segments Summary\n")
                f.write("=============================\n\n")
                f.write(f"Base Name: {base_name}\n")
                f.write(f"Created: {_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total MP4 Segments: {len(mp4_files)}\n")
                f.write(f"Total GIF Segments: {len(gif_files)}\n")
                f.write(f"Total Files: {len(mp4_files) + len(gif_files)}\n\n")

                # MP4 details
                if mp4_files:
                    f.write("MP4 Segments Details:\n")
                    f.write("--------------------\n")
                    for idx, name in enumerate(mp4_files, 1):
                        path = os.path.join(segments_dir, name)
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
                            total_mp4_size += size_mb
                            total_mp4_duration += float(info.get('duration', 0.0) or 0.0)
                            total_mp4_frames += int(info.get('frame_count', 0) or 0)
                        except Exception as e:
                            f.write(f"MP4 Segment {idx:03d}: {name}\n")
                            f.write(f"  Error reading info: {e}\n\n")

                    f.write("MP4 Summary:\n")
                    f.write(f"  Total Size: {total_mp4_size:.2f}MB\n")
                    f.write(f"  Total Duration: {total_mp4_duration:.2f}s\n")
                    f.write(f"  Total Frames: {total_mp4_frames}\n")
                    if len(mp4_files) > 0:
                        f.write(f"  Average Size: {total_mp4_size/len(mp4_files):.2f}MB\n")
                        f.write(f"  Average Duration: {total_mp4_duration/len(mp4_files):.2f}s\n")
                        f.write(f"  Average FPS: {total_mp4_frames/total_mp4_duration if total_mp4_duration>0 else 0:.2f}\n\n")

                # GIF details
                if gif_files:
                    f.write("GIF Segments Details:\n")
                    f.write("--------------------\n")
                    for idx, name in enumerate(gif_files, 1):
                        path = os.path.join(segments_dir, name)
                        try:
                            g = self.get_gif_info(path)
                            size_mb = os.path.getsize(path) / (1024 * 1024)
                            duration = float(g.get('duration') or 0.0)
                            fps = float(g.get('fps') or 0.0)
                            est_frames = int(round(duration * fps)) if duration > 0 and fps > 0 else 0
                            width = height = 0
                            if isinstance(g.get('resolution'), (list, tuple)) and len(g['resolution']) == 2:
                                width, height = g['resolution']
                            f.write(f"GIF Segment {idx:03d}: {name}\n")
                            f.write(f"  Duration: {duration:.2f}s\n")
                            f.write(f"  FPS: {fps:.2f}\n")
                            f.write(f"  Estimated Frame Count: {est_frames}\n")
                            f.write(f"  Resolution: {width}x{height}\n")
                            f.write(f"  Size: {size_mb:.2f}MB\n\n")
                            total_gif_size += size_mb
                            total_gif_duration += duration
                            total_gif_frames += est_frames
                        except Exception as e:
                            f.write(f"GIF Segment {idx:03d}: {name}\n")
                            f.write(f"  Error reading info: {e}\n\n")

                    f.write("GIF Summary:\n")
                    f.write(f"  Total Size: {total_gif_size:.2f}MB\n")
                    f.write(f"  Total Duration: {total_gif_duration:.2f}s\n")
                    f.write(f"  Estimated Total Frames: {total_gif_frames}\n")
                    if len(gif_files) > 0:
                        f.write(f"  Average Size: {total_gif_size/len(gif_files):.2f}MB\n")
                        f.write(f"  Average Duration: {total_gif_duration/len(gif_files):.2f}s\n")
                        f.write(f"  Average FPS: {total_gif_frames/total_gif_duration if total_gif_duration>0 else 0:.2f}\n\n")

                # Overall
                total_files = len(mp4_files) + len(gif_files)
                f.write("Overall Summary:\n")
                f.write("---------------\n")
                f.write(f"Total Files: {total_files}\n")
                total_size = total_mp4_size + total_gif_size
                total_duration = total_mp4_duration + total_gif_duration
                total_frames = total_mp4_frames + total_gif_frames
                f.write(f"Total Size: {total_size:.2f}MB\n")
                f.write(f"Total Duration: {total_duration:.2f}s\n")
                f.write(f"Total Frames: {total_frames}\n")
                if total_duration > 0:
                    f.write(f"Overall Average FPS: {total_frames/total_duration:.2f}\n")
                f.write(f"File Types: MP4 ({len(mp4_files)}), GIF ({len(gif_files)})\n")

            # Atomic replace
            try:
                os.replace(temp_path, summary_path)
            except Exception:
                try:
                    subprocess.run(['attrib', '-R', '-S', '-H', summary_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                    os.replace(temp_path, summary_path)
                except Exception:
                    try:
                        if os.path.exists(temp_path):
                            os.remove(temp_path)
                    except Exception:
                        pass
                    raise

            # Do not hide the summary; folder.jpg controls thumbnail selection

        except Exception as e:
            logger.warning(f"Comprehensive summary creation failed for {segments_dir}: {e}")
    
    def _safe_filename_for_filesystem(self, filename: str) -> str:
        """Convert filename to safe string for filesystem operations, handling problematic Unicode characters."""
        try:
            # Replace problematic Unicode characters that cause Windows encoding issues
            # Specifically handle the problematic character U+29F8 (⧸) and similar
            safe_chars = []
            for char in filename:
                if ord(char) < 128:
                    # Keep ASCII characters
                    safe_chars.append(char)
                elif char == '⧸':  # U+29F8 - avoid creating path separators on Windows; substitute underscore
                    safe_chars.append('_')
                elif char in ['/', '\\', ':', '*', '?', '"', '<', '>', '|']:
                    # Replace Windows filesystem-invalid characters with underscore
                    safe_chars.append('_')
                else:
                    # Replace other Unicode characters with underscore
                    safe_chars.append('_')
            
            safe_name = ''.join(safe_chars)
            
            # Ensure the filename isn't empty after sanitization
            if not safe_name.strip():
                safe_name = 'sanitized_filename'
            
            return safe_name
        except Exception:
            # Fallback: replace any non-ASCII characters with safe alternatives
            return ''.join(c if ord(c) < 128 else '_' for c in filename)
    
    def _quick_single_feasibility_check(self, input_video: str, settings: Dict[str, Any], 
                                        start_time: float, duration: float) -> Tuple[bool, Optional[float], Optional[str]]:
        """Fast pre-check: try an aggressively compressed single GIF to test feasibility.

        Returns (True, size_mb) if a heavily-compressed full-length single GIF can be kept under the size target.
        Returns (False, size_mb) if even harsh settings cannot meet the size target (suggest segmentation). When
        an early error occurs before size is measurable, returns (False, None).
        """
        try:
            if self._is_shutdown_requested():
                return False, None
            # Derive harsh settings from config, with safe defaults
            try:
                target_width = int(self.config.get('gif_settings.single_gif.quick_feasibility.width', 240))
            except Exception:
                target_width = 240
            try:
                target_fps = int(self.config.get('gif_settings.single_gif.quick_feasibility.fps', 12))
            except Exception:
                target_fps = 12
            max_size_mb = float(settings.get('max_size_mb', 10.0) or 10.0)
            logger.info(f"Feasibility settings: width={target_width}, fps={target_fps}, target={max_size_mb:.2f}MB, duration={duration:.1f}s")

            # Construct settings overlay that mirrors the real encode parameters
            feas_settings = dict(settings)
            # Use exact fps/width/height/scale as real path for accurate prediction
            feas_settings['fps'] = int(settings.get('fps', target_fps) or target_fps)
            feas_settings['width'] = int(settings.get('width', target_width) or target_width)
            feas_settings['height'] = int(settings.get('height', -1) or -1)
            # Keep default palette and dither consistent with the main encode path
            feas_settings['palette_max_colors'] = int(settings.get('palette_max_colors', 256))
            feas_settings['dither'] = settings.get('dither', 'sierra2_4a')
            feas_settings.pop('mpdecimate_aggressive', None)
            

            # Generate palette and GIF to temp dir
            temp_dir = self.config.get_temp_dir()
            base = os.path.splitext(os.path.basename(input_video))[0]
            temp_gif = os.path.join(temp_dir, f"{base}.feasibility.gif.tmp")
            palette_path = None
            try:
                # Use longer timeout for palette generation in feasibility check to handle complex videos
                palette_path = self._generate_palette(input_video, feas_settings, start_time, duration, timeout_override=90)
                if not palette_path:
                    return False, None, None
                ok = self._create_gif_with_palette(input_video, temp_gif, palette_path, feas_settings, start_time, duration)
                if not ok or not os.path.exists(temp_gif):
                    return False, None, None
                size_mb = os.path.getsize(temp_gif) / (1024 * 1024)
                fits = size_mb <= max_size_mb
                if fits:
                    logger.info(f"Feasibility result: size={size_mb:.2f}MB <= target {max_size_mb:.2f}MB")
                    return True, size_mb, temp_gif
                else:
                    logger.info(f"Feasibility initial: size={size_mb:.2f}MB > target {max_size_mb:.2f}MB (exceeds by {size_mb - max_size_mb:.2f}MB)")
                    # Try a very fast optimization pass to approximate final outcome and avoid false negatives
                    try:
                        prev = {
                            'fast_mode': getattr(self.optimizer, 'fast_mode', False),
                            'gifsicle_optimize_level': getattr(self.optimizer, 'gifsicle_optimize_level', 2),
                            'skip_gifsicle_far_over_ratio': getattr(self.optimizer, 'skip_gifsicle_far_over_ratio', 0.35),
                            'near_target_max_runs': getattr(self.optimizer, 'near_target_max_runs', 12),
                        }
                        # Quick mode: tighter budgets
                        self.optimizer.fast_mode = True
                        self.optimizer.gifsicle_optimize_level = 2
                        self.optimizer.skip_gifsicle_far_over_ratio = 0.35
                        self.optimizer.near_target_max_runs = 8
                        # Pass source video for better re-encoding in feasibility check
                        ok = self.optimizer.optimize_gif(
                            temp_gif, max_size_mb,
                            source_video=input_video if os.path.exists(input_video) else None
                        )
                    except Exception as opt_e:
                        logger.debug(f"Feasibility quick-opt failed: {opt_e}")
                        ok = False
                    finally:
                        try:
                            self.optimizer.fast_mode = prev['fast_mode']
                            self.optimizer.gifsicle_optimize_level = prev['gifsicle_optimize_level']
                            self.optimizer.skip_gifsicle_far_over_ratio = prev['skip_gifsicle_far_over_ratio']
                            self.optimizer.near_target_max_runs = prev['near_target_max_runs']
                        except Exception:
                            pass
                    # Recalculate size after quick optimization
                    try:
                        if os.path.exists(temp_gif):
                            size_mb = os.path.getsize(temp_gif) / (1024 * 1024)
                            fits = size_mb <= max_size_mb
                            if fits:
                                logger.info(f"Feasibility quick-opt result: size={size_mb:.2f}MB <= target {max_size_mb:.2f}MB")
                                return True, size_mb, temp_gif
                            else:
                                logger.info(f"Feasibility quick-opt result: size={size_mb:.2f}MB > target {max_size_mb:.2f}MB")
                                return False, size_mb, temp_gif
                    except Exception as size_e:
                        logger.debug(f"Failed to get size after quick optimization: {size_e}")
                        # If we can't get size, return with None size but keep temp_gif for caller cleanup
                        return False, None, temp_gif
                    # If temp_gif doesn't exist, return with None
                    return False, None, None
            finally:
                # Clean up palette file (always safe to delete - not cached if from feasibility check)
                try:
                    if palette_path and os.path.exists(palette_path):
                        # Check if palette is in cache directory - if not, delete it
                        cache_dir = os.path.join(self.config.get_temp_dir(), 'palette_cache')
                        if not palette_path.startswith(cache_dir):
                            os.unlink(palette_path)
                except Exception:
                    pass
                # Note: temp_gif is not deleted here - caller is responsible for cleanup
                # If caller doesn't adopt it, caller must clean it up
        except Exception as e:
            logger.debug(f"Feasibility check failed with exception, assuming not feasible: {e}")
            # Clean up temp_gif if it exists and we're returning early due to exception
            try:
                if 'temp_gif' in locals() and temp_gif and os.path.exists(temp_gif):
                    import time as time_module
                    max_retries = 3
                    retry_delay = 0.1
                    for attempt in range(max_retries):
                        try:
                            os.remove(temp_gif)
                            logger.debug(f"Cleaned up feasibility temp file after exception: {os.path.basename(temp_gif)}")
                            break
                        except (OSError, PermissionError) as cleanup_e:
                            if attempt < max_retries - 1:
                                time_module.sleep(retry_delay * (attempt + 1))
                            else:
                                logger.warning(f"Could not clean up feasibility temp file after exception: {os.path.basename(temp_gif)}: {cleanup_e}")
            except Exception:
                pass
            return False, None, None
    
    def optimize_existing_gif(self, gif_path: str, max_size_mb: float) -> bool:
        """
        Optimize an existing GIF file to meet size target.
        
        Args:
            gif_path: Path to GIF file to optimize
            max_size_mb: Maximum target size in MB
        
        Returns:
            True if optimization succeeded, False otherwise
        """
        return self.optimizer.optimize_gif(gif_path, max_size_mb)
    
    def get_gif_info(self, gif_path: str) -> Dict[str, Any]:
        """Get information about a GIF file"""
        try:
            if not os.path.exists(gif_path):
                return {'error': 'File not found'}
            
            # Use utility function for basic info
            info = get_gif_info(gif_path)
            
            # Add file_size_bytes for compatibility
            if 'file_size_mb' in info:
                info['file_size_bytes'] = int(info['file_size_mb'] * 1024 * 1024)
            
            # Add resolution tuple for compatibility
            if 'width' in info and 'height' in info:
                info['resolution'] = (info['width'], info['height'])
            
            return info
            
        except Exception as e:
            logger.error(f"Error getting GIF info: {e}")
            return {'error': str(e)}
    
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
                from ..hardware_detector import HardwareDetector
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
    
    def _set_windows_thumbnail_attributes(self, gif_path: str) -> bool:
        """Set Windows thumbnail attributes for GIF file"""
        try:
            if os.name == 'nt':  # Windows
                # Remove read-only, system, and hidden attributes
                subprocess.run(['attrib', '-H', '-S', '-R', gif_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                
                # Set archive attribute
                subprocess.run(['attrib', '+A', gif_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                
                logger.debug(f"Set Windows thumbnail attributes for: {gif_path}")
                return True
        except Exception as e:
            logger.debug(f"Could not set Windows thumbnail attributes: {e}")
            # Not critical, continue without thumbnail optimization
            return False