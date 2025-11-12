"""
GIF Optimizer
Main optimizer class with stage-based pipeline for GIF optimization
"""

import os
import subprocess
import json
import hashlib
import shutil
import time
from typing import Dict, Any, Optional, Callable, Tuple
import logging

from .gif_utils import (
    temp_file_context, temp_dir_context, safe_file_operation,
    get_gif_info, validate_gif, create_unique_temp_filename
)
from .gif_config import GifConfigHelper
from .gif_analysis import (
    analyze_motion, analyze_frame_similarity, analyze_scene_complexity,
    analyze_motion_segments
)
from .optimization_strategies import get_strategy, OptimizationStrategy

logger = logging.getLogger(__name__)


# Custom exceptions
class GifProcessingError(Exception):
    """Base exception for GIF processing errors"""
    pass


class GifOptimizationError(GifProcessingError):
    """Exception raised when GIF optimization fails"""
    pass


class GifOptimizer:
    """Main GIF optimizer with stage-based optimization pipeline"""
    
    def __init__(self, config_manager, shutdown_checker: Optional[Callable[[], bool]] = None):
        """
        Initialize GIF optimizer.
        
        Args:
            config_manager: ConfigManager instance
            shutdown_checker: Optional shutdown checker callback
        """
        self.config = config_manager
        self.config_helper = GifConfigHelper(config_manager)
        
        # Get temp directory
        try:
            self.temp_dir = self.config.get_temp_dir()
            if self.temp_dir:
                os.makedirs(self.temp_dir, exist_ok=True)
        except (OSError, PermissionError, AttributeError) as e:
            logger.warning(f"Failed to get/create temp dir from config: {e}")
            self.temp_dir = os.path.abspath('temp')
            try:
                os.makedirs(self.temp_dir, exist_ok=True)
            except (OSError, PermissionError):
                pass
        
        # Shutdown checker
        self._shutdown_checker: Callable[[], bool] = shutdown_checker or (lambda: False)
        self.shutdown_requested = False
        self.current_ffmpeg_process = None
        
        # Performance config
        perf_cfg = self.config_helper.get_performance_config()
        self.fast_mode = perf_cfg['fast_mode']
        self.gifsicle_optimize_level = perf_cfg['gifsicle_optimize_level']
        self.skip_gifsicle_far_over_ratio = perf_cfg['skip_gifsicle_far_over_ratio']
        self.near_target_max_runs = perf_cfg['near_target_max_runs']
        
        # Palette cache directory
        self.palette_cache_dir = os.path.join(self.temp_dir, 'palette_cache')
        try:
            os.makedirs(self.palette_cache_dir, exist_ok=True)
        except (OSError, PermissionError):
            pass
    
    def request_shutdown(self):
        """Request graceful shutdown of the optimizer"""
        logger.info("Shutdown requested for GIF optimizer")
        self.shutdown_requested = True
        self._terminate_ffmpeg_process()
    
    def _terminate_ffmpeg_process(self):
        """Terminate the current FFmpeg process gracefully"""
        if self.current_ffmpeg_process and self.current_ffmpeg_process.poll() is None:
            try:
                logger.info("Terminating FFmpeg process in GIF optimizer...")
                self.current_ffmpeg_process.terminate()
                try:
                    self.current_ffmpeg_process.wait(timeout=5)
                    logger.info("FFmpeg process terminated gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("FFmpeg process did not terminate gracefully, forcing kill...")
                    self.current_ffmpeg_process.kill()
                    self.current_ffmpeg_process.wait()
                    logger.info("FFmpeg process killed")
            except (OSError, subprocess.SubprocessError) as e:
                logger.error(f"Error terminating FFmpeg process: {e}", exc_info=True)
            finally:
                self.current_ffmpeg_process = None
    
    def _predict_gif_size(self, width: int, height: int, fps: float, duration: float, 
                          colors: int, lossy: int = 0) -> float:
        """
        Predict GIF size in MB based on parameters.
        
        Args:
            width: GIF width in pixels
            height: GIF height in pixels
            fps: Frames per second
            duration: Duration in seconds
            colors: Number of colors in palette
            lossy: Lossy compression level (0-200)
        
        Returns:
            Predicted size in MB
        """
        # Formula: base_size * compression_factor
        pixels_per_frame = width * height
        frames = fps * duration
        # Base estimate: bytes per pixel varies with compression
        base_bytes = pixels_per_frame * frames * 0.5  # Base estimate
        
        # Color factor: fewer colors = smaller file (non-linear)
        color_factor = (colors / 256.0) ** 0.7
        
        # Lossy factor: higher lossy = smaller file
        lossy_factor = 1.0 - (lossy / 200.0) * 0.5
        
        predicted_bytes = base_bytes * color_factor * lossy_factor
        return predicted_bytes / (1024 * 1024)
    
    def _calculate_required_parameters(self, current_size_mb: float, target_size_mb: float,
                                      current_params: Dict[str, Any], gif_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate parameters needed to meet target size.
        
        Args:
            current_size_mb: Current file size in MB
            target_size_mb: Target file size in MB
            current_params: Current parameters (width, fps, colors, etc.)
            gif_info: GIF information (width, height, fps, duration)
        
        Returns:
            Dictionary with calculated parameters
        """
        if current_size_mb <= target_size_mb:
            return current_params.copy()
        
        reduction_ratio = target_size_mb / current_size_mb
        
        # Get current values
        current_width = current_params.get('width', gif_info.get('width', 360))
        current_height = current_params.get('height', gif_info.get('height', 360))
        current_fps = current_params.get('fps', gif_info.get('fps', 20))
        current_colors = current_params.get('colors', 256)
        current_lossy = current_params.get('lossy', 0)
        duration = gif_info.get('duration', 10.0)
        
        # Calculate required reductions
        # Width reduction: use cube root since area is width^2
        # For aggressive reduction, we need to reduce multiple factors
        if reduction_ratio < 0.3:
            # Very aggressive: reduce all factors significantly
            width_factor = reduction_ratio ** (1/3)  # Cube root for area
            fps_factor = reduction_ratio ** 0.4
            color_factor = reduction_ratio ** 0.3
            lossy_increase = 150
        elif reduction_ratio < 0.5:
            # Aggressive: reduce width and fps
            width_factor = reduction_ratio ** 0.4
            fps_factor = reduction_ratio ** 0.5
            color_factor = reduction_ratio ** 0.4
            lossy_increase = 100
        elif reduction_ratio < 0.7:
            # Moderate: balanced reduction
            width_factor = reduction_ratio ** 0.5
            fps_factor = reduction_ratio ** 0.6
            color_factor = reduction_ratio ** 0.5
            lossy_increase = 60
        else:
            # Small reduction: minimal changes
            width_factor = reduction_ratio ** 0.6
            fps_factor = reduction_ratio ** 0.7
            color_factor = reduction_ratio ** 0.6
            lossy_increase = 30
        
        # Calculate new values
        new_width = max(2, int((current_width * width_factor) // 2 * 2))  # Ensure even
        new_height = max(2, int((current_height * width_factor) // 2 * 2))  # Preserve aspect ratio
        new_fps = max(6, int(round(current_fps * fps_factor)))
        new_colors = max(32, int(round(current_colors * color_factor)))
        new_lossy = min(200, current_lossy + lossy_increase)
        
        # Get quality floors (may be bypassed in aggressive mode)
        quality_floors = self.config_helper.get_quality_floors()
        allow_aggressive = self.config_helper.get_optimization_config().get('allow_aggressive_compression', False)
        
        if not allow_aggressive and quality_floors.get('enforce', True):
            min_width = quality_floors.get('min_width', 320)
            min_fps = quality_floors.get('min_fps', 18)
            new_width = max(new_width, min_width)
            new_fps = max(new_fps, min_fps)
        else:
            # Aggressive mode: use lower floors
            min_width_aggressive = quality_floors.get('min_width_aggressive', 240)
            min_fps_aggressive = quality_floors.get('min_fps_aggressive', 12)
            new_width = max(new_width, min_width_aggressive)
            new_fps = max(new_fps, min_fps_aggressive)
        
        return {
            'width': new_width,
            'height': new_height,
            'fps': new_fps,
            'colors': new_colors,
            'lossy': new_lossy,
            'dither': current_params.get('dither', 'bayer' if new_colors < 128 else 'floyd_steinberg')
        }
    
    def optimize_gif(self, gif_path: str, max_size_mb: float, source_video: Optional[str] = None) -> bool:
        """
        Optimize GIF file to meet size target using stage-based pipeline.
        
        Args:
            gif_path: Path to GIF file to optimize
            max_size_mb: Maximum target size in MB
            source_video: Optional path to source video for re-encoding (preferred over GIF re-encoding)
        
        Returns:
            True if optimization succeeded and file is under target, False otherwise
        """
        try:
            if self._shutdown_checker():
                return False
            
            if not os.path.exists(gif_path):
                logger.error(f"GIF file not found: {gif_path}")
                return False
            
            target_bytes = int(max_size_mb * 1024 * 1024)
            original_bytes = os.path.getsize(gif_path)
            original_size_mb = original_bytes / 1024 / 1024
            target_size_mb = max_size_mb
            
            logger.info(f"Optimizing GIF: current={original_size_mb:.2f}MB, target={target_size_mb:.2f}MB")
            
            # Early exit if already under target
            if original_bytes <= target_bytes:
                logger.info(f"GIF already under target: {original_size_mb:.2f}MB <= {target_size_mb:.2f}MB")
                return True
            
            # Backup original (best-effort, in temp dir)
            backup_path = None
            try:
                backup_path = create_unique_temp_filename("gif_backup", ".gif", self.temp_dir)
                safe_file_operation(shutil.copy2, gif_path, backup_path)
            except Exception:
                backup_path = None
            
            # Run optimization stages
            success = self._run_optimization_stages(gif_path, target_bytes, target_size_mb, original_size_mb, source_video)
            
            # Cleanup backup
            if backup_path and os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                except Exception:
                    pass
            
            if success:
                final_size_mb = os.path.getsize(gif_path) / 1024 / 1024
                logger.info(f"GIF optimization succeeded: {final_size_mb:.2f}MB <= {target_size_mb:.2f}MB")
            else:
                final_size_mb = os.path.getsize(gif_path) / 1024 / 1024
                logger.warning(f"GIF optimization did not meet target: {final_size_mb:.2f}MB > {target_size_mb:.2f}MB")
            
            return success
            
        except Exception as e:
            logger.error(f"Error optimizing GIF: {e}", exc_info=True)
            return False
    
    def _run_optimization_stages(self, gif_path: str, target_bytes: int, 
                                 target_size_mb: float, original_size_mb: float,
                                 source_video: Optional[str] = None) -> bool:
        """
        Run optimization stages in sequence.
        
        Args:
            gif_path: Path to GIF file
            target_bytes: Target size in bytes
            target_size_mb: Target size in MB
            original_size_mb: Original size in MB
            source_video: Optional source video path for re-encoding
        
        Returns:
            True if target met, False otherwise
        """
        current_bytes = os.path.getsize(gif_path)
        current_size_mb = current_bytes / 1024 / 1024
        
        logger.info(f"Starting optimization stages: {current_size_mb:.2f}MB -> {target_size_mb:.2f}MB target")
        
        # Stage 1: Lossless optimization (try even if far over target - it might still help)
        if self._shutdown_checker():
            return False
        
        over_ratio = (current_bytes - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
        logger.debug(f"Stage 1 (lossless): over_ratio={over_ratio:.2%}, threshold={self.skip_gifsicle_far_over_ratio:.2%}")
        
        # Try lossless optimization even if far over target - it can still reduce size
        # Only skip if gifsicle is not available
        if self._is_tool_available("gifsicle"):
            logger.info("Stage 1: Attempting lossless gifsicle optimization...")
            if self._stage_lossless_optimization(gif_path, target_bytes):
                current_bytes = os.path.getsize(gif_path)
                current_size_mb = current_bytes / 1024 / 1024
                logger.info(f"Stage 1 completed: {current_size_mb:.2f}MB")
                if current_bytes <= target_bytes:
                    logger.info("Target met after Stage 1")
                    return True
            else:
                logger.debug("Stage 1: Lossless optimization skipped or failed")
        else:
            logger.warning("Stage 1: gifsicle not available, skipping lossless optimization")
        
        # Stage 2: Source re-encoding (if source available - preferred over GIF re-encoding)
        if self._shutdown_checker():
            return False
        
        if source_video and os.path.exists(source_video):
            logger.info(f"Stage 2: Attempting re-encoding from source video: {source_video}")
            if self._reencode_from_source(source_video, gif_path, target_size_mb):
                current_bytes = os.path.getsize(gif_path)
                current_size_mb = current_bytes / 1024 / 1024
                logger.info(f"Stage 2 completed: {current_size_mb:.2f}MB")
                if current_bytes <= target_bytes:
                    logger.info("Target met after Stage 2")
                    return True
            else:
                logger.debug("Stage 2: Source re-encoding failed or did not meet target")
        else:
            logger.debug("Stage 2: No source video available, skipping")
        
        # Stage 3: Gifsicle lossy compression (ALWAYS attempt this for files over target)
        if self._shutdown_checker():
            return False
        
        logger.info(f"Stage 3: Attempting gifsicle lossy compression ({current_size_mb:.2f}MB -> {target_size_mb:.2f}MB target)...")
        if not self._is_tool_available("gifsicle"):
            logger.error("Stage 3: gifsicle not available! Cannot compress GIF. Install gifsicle to enable compression.")
            return False
        
        if self._stage_gifsicle_lossy_compression(gif_path, target_bytes, target_size_mb):
            current_bytes = os.path.getsize(gif_path)
            current_size_mb = current_bytes / 1024 / 1024
            logger.info(f"Stage 3 completed: {current_size_mb:.2f}MB")
            if current_bytes <= target_bytes:
                logger.info("Target met after Stage 3")
                return True
        else:
            logger.warning("Stage 3: Gifsicle lossy compression failed or did not reduce size sufficiently")
        
        # Stage 4: Final polish (only if within 5% of target)
        if self._shutdown_checker():
            return False
        
        current_bytes = os.path.getsize(gif_path)
        if current_bytes > target_bytes:
            over_ratio = (current_bytes - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
            if over_ratio <= 0.05:
                logger.info("Stage 4: Attempting final polish (within 5% of target)...")
                self._stage_final_polish(gif_path, target_bytes)
                current_bytes = os.path.getsize(gif_path)
                current_size_mb = current_bytes / 1024 / 1024
                logger.info(f"Stage 4 completed: {current_size_mb:.2f}MB")
            else:
                logger.debug(f"Stage 4: Skipped (over_ratio={over_ratio:.2%} > 5%)")
        
        final_size_mb = os.path.getsize(gif_path) / 1024 / 1024
        success = current_bytes <= target_bytes
        logger.info(f"Optimization stages complete: {final_size_mb:.2f}MB, target met: {success}")
        return success
    
    def _stage_lossless_optimization(self, gif_path: str, target_bytes: int) -> bool:
        """
        Stage 1: Lossless gifsicle optimization.
        
        Returns:
            True if optimization was applied (even if not under target)
        """
        if not self._is_tool_available("gifsicle"):
            return False
        
        original_size = os.path.getsize(gif_path)
        original_over_ratio = (original_size - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
        
        # For very large files, still attempt lossless optimization - it can help reduce size
        # even if it doesn't get us to target. Only skip if it's extremely unlikely to help.
        # The skip threshold is now just a warning, not a hard skip.
        if original_over_ratio >= self.skip_gifsicle_far_over_ratio:
            logger.debug(f"File is far over target ({original_over_ratio:.2%}), but attempting lossless optimization anyway...")
        
        with temp_file_context("lossless_opt", ".gif", self.temp_dir) as temp_output:
            if self._gifsicle_lossless_optimize(gif_path, temp_output):
                if os.path.exists(temp_output):
                    temp_size = os.path.getsize(temp_output)
                    if temp_size < original_size:
                        try:
                            safe_file_operation(os.replace, temp_output, gif_path)
                            reduction_mb = (original_size - temp_size) / 1024 / 1024
                            logger.info(f"Lossless optimization applied: {temp_size / 1024 / 1024:.2f}MB (reduced by {reduction_mb:.2f}MB)")
                            return True
                        except Exception as e:
                            logger.debug(f"Failed to replace with lossless optimized file: {e}")
                    else:
                        logger.debug(f"Lossless optimization did not reduce size ({temp_size / 1024 / 1024:.2f}MB >= {original_size / 1024 / 1024:.2f}MB)")
            else:
                logger.debug("Lossless gifsicle optimization failed")
        
        return False
    
    def _stage_adaptive_search(self, gif_path: str, target_bytes: int) -> bool:
        """
        Stage 2: Bounded gifsicle search for near-target cases.
        
        Returns:
            True if target met
        """
        if not self._is_tool_available("gifsicle"):
            return False
        
        with temp_file_context("adaptive_search", ".gif", self.temp_dir) as temp_output:
            if self._bounded_gifsicle_near_target(gif_path, temp_output, target_bytes):
                if os.path.exists(temp_output):
                    try:
                        safe_file_operation(os.replace, temp_output, gif_path)
                        logger.debug(f"Adaptive search succeeded")
                        return True
                    except Exception as e:
                        logger.debug(f"Failed to replace with adaptive search result: {e}")
        
        return False
    
    def _reencode_from_source(self, source_video: str, output_gif: str, target_size_mb: float) -> bool:
        """
        Re-encode GIF from source video with calculated parameters to meet target size.
        This is preferred over re-encoding an already-compressed GIF.
        
        Args:
            source_video: Path to source video file
            output_gif: Path to output GIF file
            target_size_mb: Target size in MB
        
        Returns:
            True if re-encoding succeeded and file is under target
        """
        try:
            if self._shutdown_checker():
                return False
            
            if not os.path.exists(source_video):
                logger.debug(f"Source video not found for re-encoding: {source_video}")
                return False
            
            # Get video info
            from ..ffmpeg_utils import FFmpegUtils
            ffmpeg_utils = FFmpegUtils()
            video_info = ffmpeg_utils.get_video_info(source_video)
            if not video_info:
                return False
            
            # Get current GIF info to understand what we're optimizing
            gif_info = get_gif_info(output_gif) if os.path.exists(output_gif) else {}
            current_size_mb = os.path.getsize(output_gif) / 1024 / 1024 if os.path.exists(output_gif) else target_size_mb * 2
            
            # Calculate required parameters
            current_params = {
                'width': gif_info.get('width', video_info.get('width', 360)),
                'height': gif_info.get('height', video_info.get('height', 360)),
                'fps': gif_info.get('fps', 20),
                'colors': 256,
                'lossy': 0
            }
            
            required_params = self._calculate_required_parameters(
                current_size_mb, target_size_mb, current_params, gif_info
            )
            
            # Use GifGenerator to create GIF with calculated parameters
            # We need to import it here to avoid circular imports
            from .gif_generator import GifGenerator
            generator = GifGenerator(self.config, shutdown_checker=self._shutdown_checker)
            
            # Create settings dict from required params
            settings = {
                'max_size_mb': target_size_mb,
                'width': required_params['width'],
                'height': required_params.get('height', -1),
                'fps': required_params['fps'],
                'colors': required_params['colors'],
                'dither': required_params.get('dither', 'bayer'),
                'lossy': required_params.get('lossy', 0),
                'max_duration': video_info.get('duration', 30.0)
            }
            
            # Create temp output first
            temp_output = create_unique_temp_filename("reencode_source", ".gif", self.temp_dir)
            
            # Create GIF from source with calculated parameters
            duration = min(settings['max_duration'], video_info.get('duration', 30.0))
            result = generator._create_single_gif(
                source_video, temp_output, settings, 0.0, duration
            )
            
            if result.get('success', False) and os.path.exists(temp_output):
                temp_size_mb = os.path.getsize(temp_output) / 1024 / 1024
                if temp_size_mb <= target_size_mb:
                    # Replace output with temp
                    try:
                        safe_file_operation(os.replace, temp_output, output_gif)
                        logger.debug(f"Source re-encoding succeeded: {temp_size_mb:.2f}MB <= {target_size_mb:.2f}MB")
                        return True
                    except Exception as e:
                        logger.debug(f"Failed to replace with re-encoded file: {e}")
                else:
                    # Clean up temp file if it's too large
                    try:
                        os.remove(temp_output)
                    except Exception:
                        pass
            
            return False
            
        except Exception as e:
            logger.debug(f"Source re-encoding failed: {e}")
            return False
    
    def _stage_gifsicle_lossy_compression(self, gif_path: str, target_bytes: int, target_size_mb: float) -> bool:
        """
        Stage 3: Gifsicle lossy compression with calculated parameters.
        This replaces FFmpeg re-encoding which often makes files larger.
        
        Returns:
            True if target met
        """
        if not self._is_tool_available("gifsicle"):
            return False
        
        gif_info = get_gif_info(gif_path)
        current_bytes = os.path.getsize(gif_path)
        current_size_mb = current_bytes / 1024 / 1024
        
        # Calculate required parameters
        current_params = {
            'width': gif_info.get('width', 360),
            'height': gif_info.get('height', 360),
            'fps': gif_info.get('fps', 20),
            'colors': 256,
            'lossy': 0
        }
        
        required_params = self._calculate_required_parameters(
            current_size_mb, target_size_mb, current_params, gif_info
        )
        
        logger.info(f"Calculated compression parameters: colors={required_params['colors']}, lossy={required_params['lossy']}, width={required_params['width']}")
        
        # Try multiple parameter combinations
        param_combinations = []
        
        # Primary: use calculated parameters
        param_combinations.append({
            'colors': required_params['colors'],
            'lossy': required_params['lossy'],
            'scale': required_params['width'] / float(current_params['width']),
            'name': 'primary'
        })
        
        # Secondary: more aggressive if needed
        if current_size_mb > target_size_mb * 1.5:
            param_combinations.append({
                'colors': max(32, required_params['colors'] - 32),
                'lossy': min(200, required_params['lossy'] + 20),
                'scale': required_params['width'] / float(current_params['width']) * 0.9,
                'name': 'secondary (aggressive)'
            })
        
        # Tertiary: even more aggressive
        if current_size_mb > target_size_mb * 2.0:
            param_combinations.append({
                'colors': max(32, required_params['colors'] - 64),
                'lossy': min(200, required_params['lossy'] + 40),
                'scale': required_params['width'] / float(current_params['width']) * 0.8,
                'name': 'tertiary (very aggressive)'
            })
        
        logger.info(f"Trying {len(param_combinations)} compression parameter combination(s)...")
        
        best_result = None
        best_size = current_bytes
        temp_files_to_cleanup = []
        
        try:
            for i, params in enumerate(param_combinations):
                if self._shutdown_checker():
                    break
                
                param_name = params.get('name', f'combination {i+1}')
                logger.info(f"  Attempt {i+1}/{len(param_combinations)} ({param_name}): colors={params['colors']}, lossy={params['lossy']}, scale={params['scale']:.2f}")
                
                temp_output = create_unique_temp_filename(f"gifsicle_lossy_{i}", ".gif", self.temp_dir)
                temp_files_to_cleanup.append(temp_output)
                
                if self._run_gifsicle(
                    gif_path, temp_output,
                    colors=params['colors'],
                    lossy=params['lossy'],
                    scale=params['scale']
                ):
                    if os.path.exists(temp_output):
                        temp_size = os.path.getsize(temp_output)
                        temp_size_mb = temp_size / 1024 / 1024
                        
                        # Check if output is larger than input (shouldn't happen with gifsicle)
                        if temp_size > current_bytes * 1.1:
                            logger.warning(f"  Result {i+1}: {temp_size_mb:.2f}MB is >10% larger than input ({current_size_mb:.2f}MB), skipping")
                            continue
                        
                        reduction_pct = ((current_bytes - temp_size) / current_bytes * 100) if current_bytes > 0 else 0
                        logger.info(f"  Result {i+1}: {temp_size_mb:.2f}MB (reduced by {reduction_pct:.1f}%)")
                        
                        if temp_size <= target_bytes:
                            try:
                                safe_file_operation(os.replace, temp_output, gif_path)
                                logger.info(f"  ✓ Target met with {param_name} compression!")
                                # Cleanup remaining temp files
                                for tf in temp_files_to_cleanup:
                                    if tf != temp_output:
                                        try:
                                            if os.path.exists(tf):
                                                os.remove(tf)
                                        except Exception:
                                            pass
                                return True
                            except Exception as e:
                                logger.warning(f"  Failed to replace file: {e}")
                        
                        if temp_size < best_size:
                            # Remove previous best from cleanup list
                            if best_result and best_result in temp_files_to_cleanup:
                                temp_files_to_cleanup.remove(best_result)
                                try:
                                    if os.path.exists(best_result):
                                        os.remove(best_result)
                                except Exception:
                                    pass
                            best_size = temp_size
                            best_result = temp_output
                    else:
                        logger.warning(f"  Result {i+1}: gifsicle completed but output file not found")
                else:
                    logger.warning(f"  Result {i+1}: gifsicle compression failed")
            
            # Use best result if available and better than current
            if best_result and os.path.exists(best_result) and best_size < current_bytes:
                try:
                    safe_file_operation(os.replace, best_result, gif_path)
                    best_size_mb = best_size / 1024 / 1024
                    reduction_mb = (current_bytes - best_size) / 1024 / 1024
                    reduction_pct = ((current_bytes - best_size) / current_bytes * 100) if current_bytes > 0 else 0
                    logger.info(f"Applied best gifsicle lossy compression: {best_size_mb:.2f}MB (reduced by {reduction_mb:.2f}MB, {reduction_pct:.1f}%)")
                    # Cleanup remaining temp files
                    for tf in temp_files_to_cleanup:
                        if tf != best_result:
                            try:
                                if os.path.exists(tf):
                                    os.remove(tf)
                            except Exception:
                                pass
                    success = best_size <= target_bytes
                    if not success:
                        logger.warning(f"Best compression result ({best_size_mb:.2f}MB) still exceeds target ({target_size_mb:.2f}MB)")
                    return success
                except Exception as e:
                    logger.error(f"Failed to apply best result: {e}")
            else:
                if not best_result:
                    logger.warning("No successful compression attempts - all gifsicle runs failed or produced larger files")
                elif best_size >= current_bytes:
                    logger.warning(f"Best compression result ({best_size / 1024 / 1024:.2f}MB) was not better than original ({current_size_mb:.2f}MB)")
        finally:
            # Cleanup any remaining temp files
            for tf in temp_files_to_cleanup:
                if tf != best_result:
                    try:
                        if os.path.exists(tf):
                            os.remove(tf)
                    except Exception:
                        pass
        
        return False
    
    def _stage_final_polish(self, gif_path: str, target_bytes: int) -> bool:
        """
        Stage 4: Final polish with gifsicle squeeze for small overages.
        
        Returns:
            True if target met
        """
        if not self._is_tool_available("gifsicle"):
            return False
        
        current_bytes = os.path.getsize(gif_path)
        over_ratio = (current_bytes - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
        
        # Only try if within 15% over target
        if over_ratio > 0.15:
            return False
        
        with temp_file_context("final_polish", ".gif", self.temp_dir) as temp_output:
            if self._gifsicle_squeeze_small_overage(gif_path, temp_output):
                if os.path.exists(temp_output):
                    temp_size = os.path.getsize(temp_output)
                    if temp_size < current_bytes:
                        try:
                            safe_file_operation(os.replace, temp_output, gif_path)
                            logger.debug(f"Final polish applied: {temp_size / 1024 / 1024:.2f}MB")
                            return temp_size <= target_bytes
                        except Exception as e:
                            logger.debug(f"Failed to replace with polished file: {e}")
        
        return False
    
    def _should_fallback_to_segmentation(self, gif_path: str, max_size_mb: float) -> bool:
        """
        Determine if segmentation fallback should be used.
        
        Args:
            gif_path: Path to GIF file
            max_size_mb: Target size in MB
        
        Returns:
            True if segmentation should be attempted
        """
        try:
            gif_info = get_gif_info(gif_path)
            current_size_mb = os.path.getsize(gif_path) / 1024 / 1024
            duration = gif_info.get('duration', 0)
            
            seg_config = self.config_helper.get_segmentation_config()
            size_threshold = seg_config['size_threshold_multiplier']
            duration_limit = seg_config['fallback_duration_limit']
            
            # Check size threshold
            if current_size_mb > max_size_mb * size_threshold:
                logger.debug(f"Segmentation recommended: size {current_size_mb:.2f}MB > {max_size_mb * size_threshold:.2f}MB")
                return True
            
            # Check duration limit
            if duration > duration_limit:
                logger.debug(f"Segmentation recommended: duration {duration:.1f}s > {duration_limit}s")
                return True
            
            return False
        except Exception:
            return False
    
    # Helper methods for gifsicle operations
    def _is_tool_available(self, tool_name: str) -> bool:
        """Check if a tool is available"""
        try:
            result = subprocess.run(
                [tool_name, "--version"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def _gifsicle_lossless_optimize(self, input_path: str, output_path: str) -> bool:
        """Run lossless gifsicle optimization"""
        try:
            cmd = [
                "gifsicle",
                f"--optimize={max(1, min(3, self.gifsicle_optimize_level))}",
                "--careful",
                "--no-comments",
                "--no-extensions",
                "--no-names",
                "--same-loopcount",
                input_path,
                "--output",
                output_path,
            ]
            timeout_sec = 30 if self.fast_mode else 60
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout_sec
            )
            return result.returncode == 0 and os.path.exists(output_path)
        except Exception as e:
            logger.debug(f"Lossless gifsicle optimize failed: {e}")
            return False
    
    def _bounded_gifsicle_near_target(self, input_path: str, best_output_path: str, target_bytes: int) -> bool:
        """Bounded gifsicle search for near-target cases"""
        try:
            # Conservative ladders
            if self.fast_mode:
                color_steps = [256, 192]
                lossy_steps = [0, 20]
                scale_steps = [1.0, 0.96]
            else:
                color_steps = [256, 224, 208, 192, 176, 160]
                lossy_steps = [0, 10, 20, 30]
                scale_steps = [1.0, 0.96, 0.92]
            
            original_size = os.path.getsize(input_path)
            best: Tuple[int, str] = (original_size, '')
            
            # Try quick color-only first
            temp_files_to_cleanup = []
            try:
                for colors in color_steps:
                    if self._shutdown_checker():
                        break
                    
                    temp = create_unique_temp_filename(f"near_c{colors}", ".gif", self.temp_dir)
                    temp_files_to_cleanup.append(temp)
                    
                    if self._run_gifsicle(input_path, temp, colors=colors, lossy=0, scale=1.0):
                        size = os.path.getsize(temp)
                        if size <= target_bytes:
                            safe_file_operation(shutil.copy2, temp, best_output_path)
                            # Cleanup all temp files
                            for tf in temp_files_to_cleanup:
                                try:
                                    if os.path.exists(tf):
                                        os.remove(tf)
                                except Exception:
                                    pass
                            return True
                        if size < best[0]:
                            # Remove previous best from cleanup list if exists
                            if best[1] and best[1] in temp_files_to_cleanup:
                                temp_files_to_cleanup.remove(best[1])
                                try:
                                    if os.path.exists(best[1]):
                                        os.remove(best[1])
                                except Exception:
                                    pass
                            best = (size, temp)
                
                # Small grid over lossy and scale
                max_runs = self.near_target_max_runs
                runs = 0
                
                for scale in scale_steps:
                    for colors in color_steps:
                        for lossy in lossy_steps:
                            if runs >= max_runs or self._shutdown_checker():
                                break
                            
                            temp = create_unique_temp_filename(f"near_s{int(scale*100)}_c{colors}_l{lossy}", ".gif", self.temp_dir)
                            temp_files_to_cleanup.append(temp)
                            
                            if self._run_gifsicle(input_path, temp, colors=colors, lossy=lossy, scale=scale):
                                runs += 1
                                size = os.path.getsize(temp)
                                if size <= target_bytes:
                                    safe_file_operation(shutil.copy2, temp, best_output_path)
                                    # Cleanup all temp files
                                    for tf in temp_files_to_cleanup:
                                        try:
                                            if os.path.exists(tf):
                                                os.remove(tf)
                                        except Exception:
                                            pass
                                    return True
                                if size < best[0]:
                                    # Remove previous best from cleanup list
                                    if best[1] and best[1] in temp_files_to_cleanup:
                                        temp_files_to_cleanup.remove(best[1])
                                        try:
                                            if os.path.exists(best[1]):
                                                os.remove(best[1])
                                        except Exception:
                                            pass
                                    best = (size, temp)
                        
                        if runs >= max_runs:
                            break
                    if runs >= max_runs:
                        break
            
                # Use best if within 5% of target
                if best[1] and os.path.exists(best[1]):
                    if best[0] <= int(target_bytes * 1.05):
                        safe_file_operation(shutil.copy2, best[1], best_output_path)
                        # Cleanup all temp files
                        for tf in temp_files_to_cleanup:
                            try:
                                if os.path.exists(tf):
                                    os.remove(tf)
                            except Exception:
                                pass
                        return True
                
                # Cleanup all temp files
                for tf in temp_files_to_cleanup:
                    try:
                        if os.path.exists(tf):
                            os.remove(tf)
                    except Exception:
                        pass
                
                return False
            except Exception as e:
                # Cleanup on error
                for tf in temp_files_to_cleanup:
                    try:
                        if os.path.exists(tf):
                            os.remove(tf)
                    except Exception:
                        pass
                raise
        except Exception as e:
            logger.debug(f"Bounded gifsicle search failed: {e}")
            return False
    
    def _run_gifsicle(self, input_path: str, output_path: str, colors: int, lossy: int, scale: float) -> bool:
        """Execute gifsicle with given parameters"""
        try:
            cmd = [
                "gifsicle",
                f"--optimize={max(1, min(3, self.gifsicle_optimize_level))}",
                "--careful",
                "--no-comments",
                "--no-extensions",
                "--no-names",
            ]
            if scale and abs(scale - 1.0) > 1e-6:
                cmd.extend(["--scale", f"{scale:.4f}"])
            if colors and colors < 256:
                cmd.extend(["--colors", str(max(2, min(256, colors)))])
            if lossy and lossy > 0:
                cmd.extend(["--lossy", str(max(1, min(150, int(lossy))))])
            cmd.extend([input_path, "--output", output_path])
            
            timeout_sec = 45 if self.fast_mode else 120
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout_sec
            )
            return result.returncode == 0 and os.path.exists(output_path)
        except Exception as e:
            logger.debug(f"gifsicle run failed: {e}")
            return False
    
    def _gifsicle_squeeze_small_overage(self, gif_path: str, output_path: str) -> bool:
        """Tiny gifsicle squeeze for small overages"""
        try:
            cmd = [
                'gifsicle', '--optimize=3', '--careful', '--lossy=30',
                gif_path, '--output', output_path
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=120
            )
            if result.returncode == 0 and os.path.exists(output_path):
                return os.path.getsize(output_path) < os.path.getsize(gif_path)
            return False
        except Exception:
            return False
    
    # Helper methods for FFmpeg operations
    def _build_scale_filter(self, width: int, height: int) -> str:
        """Build scale filter with proper aspect ratio preservation"""
        if width == -1 and height == -1:
            return "scale=iw:ih:flags=lanczos"
        elif height == -1:
            return f"scale={width}:-2:flags=lanczos"
        else:
            return f"scale={width}:{height}:flags=lanczos"
    
    def _get_palette_cache_key(self, input_video: str, method: str, width: int, height: int,
                               fps: int, colors: int, mpdecimate_frac: float = 0.3,
                               stats_mode: str = 'diff', dither: str = 'sierra2_4a') -> str:
        """Generate cache key for palette file"""
        try:
            try:
                mtime = os.path.getmtime(input_video) if os.path.exists(input_video) else 0
            except (OSError, PermissionError):
                mtime = 0
            
            key_payload = {
                'path': input_video,
                'mtime': mtime,
                'method': method,
                'width': width,
                'height': height,
                'fps': float(fps),
                'colors': colors,
                'mpdecimate_frac': float(mpdecimate_frac),
                'stats_mode': stats_mode,
                'dither': dither
            }
            key_str = json.dumps(key_payload, sort_keys=True, separators=(',', ':'))
            digest = hashlib.sha1(key_str.encode('utf-8')).hexdigest()
            return os.path.join(self.palette_cache_dir, f"pal_{method}_{digest}.png")
        except Exception as e:
            logger.debug(f"Error generating palette cache key: {e}")
            return create_unique_temp_filename(f"{method}_palette", ".png", self.temp_dir)
    
    def _ffmpeg_palette_reencode(self, input_path: str, output_path: str, new_width: int,
                                 fps: int, max_colors: int = 256,
                                 mpdecimate_frac: Optional[float] = None,
                                 stats_mode: Optional[str] = None,
                                 dither: Optional[str] = None) -> bool:
        """
        Re-encode GIF via FFmpeg using mpdecimate + palettegen/paletteuse.
        
        Args:
            input_path: Input GIF path
            output_path: Output GIF path (must be in temp_dir)
            new_width: Target width
            fps: Target FPS
            max_colors: Maximum colors in palette
            mpdecimate_frac: Optional mpdecimate frac parameter
            stats_mode: Optional stats_mode
            dither: Optional dithering strategy
        """
        try:
            if self._shutdown_checker():
                return False
            
            # Ensure output is in temp_dir
            if not output_path.startswith(self.temp_dir):
                logger.warning(f"Output path not in temp_dir, using temp_dir: {output_path}")
                output_path = create_unique_temp_filename("ffmpeg_reencode", ".gif", self.temp_dir)
            
            input_size_mb = os.path.getsize(input_path) / 1024 / 1024 if os.path.exists(input_path) else 0
            
            # Determine mpdecimate frac
            if mpdecimate_frac is None:
                try:
                    similarity_analysis = analyze_frame_similarity(input_path, self.temp_dir)
                    mpdecimate_frac = similarity_analysis.get('optimal_frac', 0.3)
                except Exception:
                    mpdecimate_frac = 0.3
            
            mpdecimate_frac = max(0.1, min(0.5, mpdecimate_frac))
            
            # Determine stats_mode and dither
            if stats_mode is None or dither is None:
                try:
                    complexity_analysis = analyze_scene_complexity(input_path, self.temp_dir)
                    if stats_mode is None:
                        stats_mode = complexity_analysis.get('stats_mode', 'diff')
                    if dither is None:
                        dither = complexity_analysis.get('dither', 'sierra2_4a')
                except Exception:
                    if stats_mode is None:
                        stats_mode = 'diff'
                    if dither is None:
                        dither = 'sierra2_4a'
            
            # Validate
            if stats_mode not in ['full', 'diff']:
                stats_mode = 'diff'
            
            valid_dithers = ['bayer', 'floyd_steinberg', 'sierra2_4a', 'sierra2', 'sierra', 'heckbert']
            if dither not in valid_dithers:
                dither = 'sierra2_4a'
            
            # Get palette path
            palette_path = self._get_palette_cache_key(
                input_path, 'reencode', new_width, -1, fps, max_colors,
                mpdecimate_frac, stats_mode, dither
            )
            
            # Generate palette if not cached
            palette_cached = os.path.exists(palette_path)
            if not palette_cached:
                try:
                    os.makedirs(self.palette_cache_dir, exist_ok=True)
                except Exception:
                    pass
                
                pre = [
                    f'mpdecimate=hi=512:lo=256:frac={mpdecimate_frac:.2f}',
                    f'fps={fps}',
                    self._build_scale_filter(new_width, -1)
                ]
                vf_palette = ','.join(pre + [f'palettegen=max_colors={int(max_colors)}:stats_mode={stats_mode}'])
                
                cmd1 = [
                    'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', input_path, '-vf', vf_palette, '-frames:v', '1', palette_path
                ]
                
                if self._shutdown_checker():
                    return False
                
                result1 = self._run_subprocess_with_shutdown_check(cmd1, timeout=180)
                if result1.returncode != 0 or not os.path.exists(palette_path):
                    try:
                        if os.path.exists(palette_path):
                            os.remove(palette_path)
                    except Exception:
                        pass
                    return False
            
            # Apply palette
            pre = [
                f'mpdecimate=hi=512:lo=256:frac={mpdecimate_frac:.2f}',
                f'fps={fps}',
                self._build_scale_filter(new_width, -1)
            ]
            lavfi = ','.join(pre) + f' [x]; [x][1:v] paletteuse=dither={dither}:diff_mode=rectangle'
            
            cmd2 = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-i', input_path, '-i', palette_path,
                '-lavfi', lavfi,
                '-loop', '0', output_path
            ]
            
            if self._shutdown_checker():
                return False
            
            result2 = self._run_subprocess_with_shutdown_check(cmd2, timeout=300)
            
            success = result2.returncode == 0 and os.path.exists(output_path)
            if success:
                output_size_mb = os.path.getsize(output_path) / 1024 / 1024
                if output_size_mb > input_size_mb * 1.1:
                    logger.warning(f"Output {output_size_mb:.2f}MB is >10% larger than input {input_size_mb:.2f}MB")
                    return False
            
            return success
        except Exception as e:
            if self._shutdown_checker():
                logger.info("FFmpeg palette re-encode interrupted by shutdown")
            else:
                logger.debug(f"FFmpeg palette re-encode exception: {e}")
            return False
    
    def _run_subprocess_with_shutdown_check(self, cmd, timeout=120, **kwargs):
        """Run subprocess with shutdown checking and process tracking"""
        if self.shutdown_requested or self._shutdown_checker():
            class ShutdownResult:
                def __init__(self):
                    self.returncode = 1
                    self.stdout = ''
                    self.stderr = 'Shutdown requested before execution'
            return ShutdownResult()
        
        try:
            self.current_ffmpeg_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                **kwargs
            )
            
            start_time = time.time()
            process = self.current_ffmpeg_process
            
            while process and process.poll() is None:
                if self.shutdown_requested or self._shutdown_checker():
                    logger.info("Shutdown requested during subprocess execution, terminating...")
                    self._terminate_ffmpeg_process()
                    class ShutdownResult:
                        def __init__(self):
                            self.returncode = 1
                            self.stdout = ''
                            self.stderr = 'Shutdown requested during execution'
                    return ShutdownResult()
                
                if time.time() - start_time > timeout:
                    logger.warning(f"Subprocess timeout after {timeout}s, terminating...")
                    self._terminate_ffmpeg_process()
                    class TimeoutResult:
                        def __init__(self):
                            self.returncode = 1
                            self.stdout = ''
                            self.stderr = 'TimeoutExpired'
                    return TimeoutResult()
                
                time.sleep(0.1)
                process = self.current_ffmpeg_process
            
            if process is None:
                class TerminatedResult:
                    def __init__(self):
                        self.returncode = 1
                        self.stdout = ''
                        self.stderr = 'Process was terminated'
                return TerminatedResult()
            
            stdout, stderr = process.communicate()
            
            class NormalResult:
                def __init__(self, returncode, stdout, stderr):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr
            
            return NormalResult(process.returncode, stdout, stderr)
            
        except (subprocess.SubprocessError, OSError, ValueError) as e:
            logger.error(f"Error running subprocess: {e}", exc_info=True)
            class ErrorResult:
                def __init__(self):
                    self.returncode = 1
                    self.stdout = ''
                    self.stderr = str(e)
            return ErrorResult()
        finally:
            self.current_ffmpeg_process = None

