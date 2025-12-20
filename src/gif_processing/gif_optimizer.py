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
import math
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
from .pil_compression import PILCompressor

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
        self.long_clip_guardrails = self.config_helper.get_long_clip_guardrails()
        
        # Quality guardrails config
        quality_guardrails = self.config_helper.get_quality_guardrails()
        self.min_quality_threshold = quality_guardrails.get('min_quality_threshold', 3.0)
        self.enable_early_exit = quality_guardrails.get('enable_early_exit', True)
        
        # Advanced gifsicle config
        self.multi_pass_enabled = perf_cfg.get('multi_pass_enabled', True)
        self.adaptive_strategy = perf_cfg.get('adaptive_strategy', True)
        self.progressive_refinement = perf_cfg.get('progressive_refinement', True)
        self.content_aware = perf_cfg.get('content_aware', True)
        self.parameter_presets = perf_cfg.get('parameter_presets', {})
        self.advanced_options = perf_cfg.get('advanced_options', {})
        # Stage-level time budgets (seconds) to prevent runaway processing; honour config
        gifsicle_budget = int(perf_cfg.get('gifsicle_time_budget_seconds', 25 if self.fast_mode else 45))
        stage_budget_cfg = perf_cfg.get('stage_time_budgets', {}) or {}
        self.stage_time_budgets = {
            'lossless': max(1, int(stage_budget_cfg.get('lossless', 8))),
            'gifsicle': max(1, int(stage_budget_cfg.get('gifsicle', gifsicle_budget))),
            'palette': max(1, int(stage_budget_cfg.get('palette', 12))),
        }
        self.last_gifsicle_timeout = False
        
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
    
    def _is_converged(self, current_size_mb: float, previous_size_mb: Optional[float], 
                      target_size_mb: float, threshold_percent: float = 2.0) -> bool:
        """
        Check if optimizer has converged (result within threshold of previous attempt).
        
        Args:
            current_size_mb: Current optimization result size in MB
            previous_size_mb: Previous optimization result size in MB (None if first attempt)
            target_size_mb: Target size in MB
            threshold_percent: Percentage threshold for convergence detection (default 2.0%)
        
        Returns:
            True if optimizer has converged (result within threshold of previous attempt)
        """
        if previous_size_mb is None:
            return False
        size_diff = abs(current_size_mb - previous_size_mb)
        size_diff_percent = (size_diff / max(previous_size_mb, 1e-6)) * 100
        converged = size_diff_percent <= threshold_percent
        if converged:
            logger.info(
                f"Optimizer converged: current={current_size_mb:.2f}MB, previous={previous_size_mb:.2f}MB, "
                f"diff={size_diff_percent:.2f}% (threshold={threshold_percent}%)"
            )
        return converged
    
    def _calculate_required_parameters(self, current_size_mb: float, target_size_mb: float,
                                      current_params: Dict[str, Any], gif_info: Dict[str, Any],
                                      gif_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate parameters needed to meet target size with content-aware adjustments.
        
        Args:
            current_size_mb: Current file size in MB
            target_size_mb: Target file size in MB
            current_params: Current parameters (width, fps, colors, etc.)
            gif_info: GIF information (width, height, fps, duration)
            gif_path: Optional path to GIF for content analysis
        
        Returns:
            Dictionary with calculated parameters
        """
        if current_size_mb <= target_size_mb:
            return current_params.copy()
        
        reduction_ratio = target_size_mb / current_size_mb
        
        # Content-aware analysis for better parameter selection
        motion_level = 'medium'
        complexity_level = 'medium'
        frame_count = gif_info.get('frame_count', 0)
        has_transparency = gif_info.get('has_transparency', False)
        
        if self.content_aware and gif_path and os.path.exists(gif_path):
            try:
                # Analyze motion to adjust FPS intelligently
                motion_analysis = analyze_motion(gif_path, current_params.get('fps', 20), self.temp_dir)
                motion_level = motion_analysis.get('motion_level', 'medium')
                
                # Analyze complexity to adjust colors intelligently
                complexity_analysis = analyze_scene_complexity(gif_path, self.temp_dir)
                complexity_level = complexity_analysis.get('complexity_level', 'medium')
                
                # Get frame count for frame-aware optimization
                frame_count = gif_info.get('frame_count', 0)
                
                logger.debug(f"Content analysis: motion={motion_level}, complexity={complexity_level}, frames={frame_count}")
            except Exception as e:
                logger.debug(f"Content analysis failed, using defaults: {e}")
        
        # Get current values (needed for both adaptive and fallback paths)
        current_width = current_params.get('width', gif_info.get('width', 360))
        current_height = current_params.get('height', gif_info.get('height', 360))
        current_fps = current_params.get('fps', gif_info.get('fps', 20))
        current_colors = current_params.get('colors', 256)
        current_lossy = current_params.get('lossy', 0)
        duration = gif_info.get('duration', 10.0)
        
        # Use adaptive strategy if enabled
        if self.adaptive_strategy:
            strategy = self._adaptive_gifsicle_strategy(current_size_mb, target_size_mb, gif_info)
            # Use strategy-based parameter selection
            # Select initial parameters from strategy ranges based on reduction needed
            if reduction_ratio < 0.5:
                # Large reduction needed - use more aggressive strategy values
                colors_idx = min(2, len(strategy['colors_range']) - 1)
                lossy_idx = min(2, len(strategy['lossy_range']) - 1)
                scale_idx = min(2, len(strategy['scale_range']) - 1)
            elif reduction_ratio < 0.7:
                # Moderate reduction - use balanced strategy values
                colors_idx = min(1, len(strategy['colors_range']) - 1)
                lossy_idx = min(1, len(strategy['lossy_range']) - 1)
                scale_idx = min(1, len(strategy['scale_range']) - 1)
            else:
                # Small reduction - use conservative strategy values
                colors_idx = 0
                lossy_idx = 0
                scale_idx = 0
            
            # Adjust based on content analysis
            if motion_level == 'high' and lossy_idx > 0:
                lossy_idx = max(0, lossy_idx - 1)  # Use less lossy for high motion
            if complexity_level == 'high' and colors_idx > 0:
                colors_idx = max(0, colors_idx - 1)  # Preserve more colors for high complexity
            
            # Get values from strategy
            target_colors = strategy['colors_range'][colors_idx] if colors_idx < len(strategy['colors_range']) else 192
            target_lossy = strategy['lossy_range'][lossy_idx] if lossy_idx < len(strategy['lossy_range']) else 60
            target_scale = strategy['scale_range'][scale_idx] if scale_idx < len(strategy['scale_range']) else 0.92
            
            # Calculate target width from scale
            target_width = int(current_width * target_scale)
            
            # Enforce quality floors even in adaptive strategy path
            quality_floors = self.config_helper.get_quality_floors()
            allow_aggressive = self.config_helper.get_optimization_config().get('allow_aggressive_compression', False)
            
            if not allow_aggressive and quality_floors.get('enforce', True):
                min_width = quality_floors.get('min_width', 360)
                min_fps = quality_floors.get('min_fps', 15)
                target_width = max(target_width, min_width)
                # Note: FPS is handled separately in FFmpeg stage, but we ensure it's at least min_fps
            else:
                # Aggressive mode: still honor floors to prevent unreadable outputs
                min_width_aggressive = quality_floors.get('min_width_aggressive', 360)
                target_width = max(target_width, min_width_aggressive)
            
            # Note: GIF-only specific floors (min 160 colors, max 100 lossy, min 0.85 scale)
            # are applied in _stage_gifsicle_lossy_compression where we have source_video context.
            # Here we only apply general quality floors.
            
            return {
                'width': max(64, target_width),
                'height': int(current_height * target_scale),
                'fps': current_fps,  # FPS handled separately in FFmpeg stage
                'colors': target_colors,
                'lossy': target_lossy,
                'dither': 'floyd_steinberg' if target_colors >= 192 else 'sierra2_4a'  # Default dither for quality
            }
        
        # Calculate required reductions with content-aware adjustments
        # Width reduction: use cube root since area is width^2
        # Adjust factors based on content analysis
        
        # Motion-aware FPS adjustment
        fps_motion_multiplier = {
            'low': 0.85,     # Low motion: can reduce FPS more
            'medium': 1.0,   # Medium motion: standard reduction
            'high': 1.15     # High motion: preserve FPS better
        }.get(motion_level, 1.0)
        
        # Complexity-aware color adjustment
        color_complexity_multiplier = {
            'low': 0.85,     # Low complexity: can use fewer colors
            'medium': 1.0,   # Medium complexity: standard colors
            'high': 1.15     # High complexity: preserve more colors
        }.get(complexity_level, 1.0)
        
        if reduction_ratio < 0.3:
            # Very aggressive: reduce all factors significantly
            width_factor = reduction_ratio ** (1/3)  # Cube root for area
            fps_factor = (reduction_ratio ** 0.4) * fps_motion_multiplier
            color_factor = (reduction_ratio ** 0.3) * color_complexity_multiplier
            lossy_increase = 120  # Reduced from 150
        elif reduction_ratio < 0.5:
            # Aggressive: reduce width and fps
            width_factor = reduction_ratio ** 0.4
            fps_factor = (reduction_ratio ** 0.5) * fps_motion_multiplier
            color_factor = (reduction_ratio ** 0.4) * color_complexity_multiplier
            lossy_increase = 80  # Reduced from 100
        elif reduction_ratio < 0.7:
            # Moderate: balanced reduction
            width_factor = reduction_ratio ** 0.5
            fps_factor = (reduction_ratio ** 0.6) * fps_motion_multiplier
            color_factor = (reduction_ratio ** 0.5) * color_complexity_multiplier
            lossy_increase = 50  # Reduced from 60
        else:
            # Small reduction: minimal changes
            width_factor = reduction_ratio ** 0.6
            fps_factor = (reduction_ratio ** 0.7) * fps_motion_multiplier
            color_factor = (reduction_ratio ** 0.6) * color_complexity_multiplier
            lossy_increase = 25  # Reduced from 30
        
        # Calculate new values with improved quality floors to reduce artifacts
        # Minimum width increased to 360px to avoid unwatchable outputs
        # Minimum FPS increased to 15 for smoother motion
        # Minimum colors increased from 32 to 128 to reduce banding
        new_width = max(360, int((current_width * width_factor) // 2 * 2))  # Ensure even
        new_height = max(360, int((current_height * width_factor) // 2 * 2))  # Preserve aspect ratio
        new_fps = max(15, int(round(current_fps * fps_factor)))
        new_colors = max(128, int(round(current_colors * color_factor)))
        new_colors = min(256, new_colors)
        # Limit lossy to max 120 (reduced from 200) to prevent severe degradation
        new_lossy = min(120, current_lossy + lossy_increase)
        
        # Get quality floors (may be bypassed in aggressive mode)
        quality_floors = self.config_helper.get_quality_floors()
        allow_aggressive = self.config_helper.get_optimization_config().get('allow_aggressive_compression', False)
        
        if not allow_aggressive and quality_floors.get('enforce', True):
            min_width = quality_floors.get('min_width', 360)
            min_fps = quality_floors.get('min_fps', 15)
            new_width = max(new_width, min_width)
            new_fps = max(new_fps, min_fps)
        else:
            # Aggressive mode: honor the same floors to prevent unreadable segments
            min_width_aggressive = quality_floors.get('min_width_aggressive', 360)
            min_fps_aggressive = quality_floors.get('min_fps_aggressive', 15)
            new_width = max(new_width, min_width_aggressive)
            new_fps = max(new_fps, min_fps_aggressive)
        
        # Improved dither selection based on color count
        # Use better dithering algorithms to reduce artifacts
        if new_colors >= 192:
            dither = 'floyd_steinberg'  # Best for gradients with many colors
        elif new_colors >= 128:
            dither = 'sierra2_4a'  # Good balance
        else:
            dither = 'bayer'  # Simpler for few colors
        
        return {
            'width': new_width,
            'height': new_height,
            'fps': new_fps,
            'colors': new_colors,
            'lossy': new_lossy,
            'dither': dither
        }
    
    def optimize_gif(self, gif_path: str, max_size_mb: float, source_video: Optional[str] = None,
                     force_guardrails: bool = False, previous_result_size_mb: Optional[float] = None,
                     skip_stage2: bool = False) -> Tuple[bool, float, bool]:
        """
        Optimize GIF file to meet size target using stage-based pipeline.
        
        Args:
            gif_path: Path to GIF file to optimize
            max_size_mb: Maximum target size in MB
            source_video: Optional path to source video for re-encoding (preferred over GIF re-encoding)
            force_guardrails: Force guardrail heuristics even if clip duration/frame budget is low
            previous_result_size_mb: Optional previous optimization result size for convergence detection
        
        Returns:
            Tuple of (success: bool, final_size_mb: float, converged: bool)
            - success: True if optimization succeeded and file is under target
            - final_size_mb: Final file size in MB after optimization
            - converged: True if optimizer has converged (result within 2% of previous attempt)
        """
        try:
            if self._shutdown_checker():
                return False, 0.0, False
            
            if not os.path.exists(gif_path):
                logger.error(f"GIF file not found: {gif_path}")
                return False, 0.0, False
            
            target_bytes = int(max_size_mb * 1024 * 1024)
            original_bytes = os.path.getsize(gif_path)
            original_size_mb = original_bytes / 1024 / 1024
            target_size_mb = max_size_mb
            
            logger.info(f"Optimizing GIF: current={original_size_mb:.2f}MB, target={target_size_mb:.2f}MB")
            
            # Early exit if already under target
            if original_bytes <= target_bytes:
                logger.info(f"GIF already under target: {original_size_mb:.2f}MB <= {target_size_mb:.2f}MB")
                return True, original_size_mb, False
            
            # Backup original (best-effort, in temp dir)
            backup_path = None
            try:
                backup_path = create_unique_temp_filename("gif_backup", ".gif", self.temp_dir)
                safe_file_operation(shutil.copy2, gif_path, backup_path)
            except Exception:
                backup_path = None
            
            # Run optimization stages
            success, converged = self._run_optimization_stages(
                gif_path,
                target_bytes,
                target_size_mb,
                original_size_mb,
                source_video if not skip_stage2 else None,  # Skip Stage 2 if flag is set
                force_guardrails=force_guardrails,
                previous_result_size_mb=previous_result_size_mb
            )
            
            # Cleanup backup
            if backup_path and os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                except Exception:
                    pass
            
            final_size_mb = os.path.getsize(gif_path) / 1024 / 1024
            
            # Check for convergence if we have a previous result
            if not converged and previous_result_size_mb is not None:
                converged = self._is_converged(final_size_mb, previous_result_size_mb, target_size_mb)
            
            if success:
                logger.info(f"GIF optimization succeeded: {final_size_mb:.2f}MB <= {target_size_mb:.2f}MB")
            else:
                logger.warning(f"GIF optimization did not meet target: {final_size_mb:.2f}MB > {target_size_mb:.2f}MB")
                if converged:
                    logger.warning(
                        f"Optimizer has converged (result within 2% of previous attempt). "
                        f"Further optimization attempts unlikely to improve result."
                    )
                
                # Log detailed specs of best result for debugging and segmentation consideration
                try:
                    gif_info = get_gif_info(gif_path)
                    logger.warning(
                        f"Single-GIF optimization failed. "
                        f"Best: {final_size_mb:.2f}MB / {target_size_mb:.2f}MB. "
                        f"Specs: {gif_info.get('width', 'unknown')}x{gif_info.get('height', 'unknown')}, "
                        f"{gif_info.get('fps', 'unknown')} fps, {gif_info.get('frame_count', 'unknown')} frames, "
                        f"{gif_info.get('duration', 'unknown'):.1f}s. Attempting segmentation..."
                    )
                except Exception as e:
                    logger.debug(f"Could not get GIF info: {e}")
            
            return success, final_size_mb, converged
            
        except Exception as e:
            logger.error(f"Error optimizing GIF: {e}", exc_info=True)
            return False, 0.0, False
    
    def _run_optimization_stages(self, gif_path: str, target_bytes: int, 
                                 target_size_mb: float, original_size_mb: float,
                                 source_video: Optional[str] = None,
                                 force_guardrails: bool = False,
                                 previous_result_size_mb: Optional[float] = None) -> Tuple[bool, bool]:
        """
        Run optimization stages in sequence.
        
        Args:
            gif_path: Path to GIF file
            target_bytes: Target size in bytes
            target_size_mb: Target size in MB
            original_size_mb: Original size in MB
            source_video: Optional source video path for re-encoding
            force_guardrails: Force guardrail heuristics
            previous_result_size_mb: Optional previous optimization result size for convergence detection
        
        Returns:
            Tuple of (success: bool, converged: bool)
            - success: True if target met
            - converged: True if optimizer has converged (result within 2% of previous attempt)
        """
        current_bytes = os.path.getsize(gif_path)
        current_size_mb = current_bytes / 1024 / 1024
        
        logger.info(f"Starting optimization stages: {current_size_mb:.2f}MB -> {target_size_mb:.2f}MB target")
        stage_start_time = time.time()
        gif_info_cache: Optional[Dict[str, Any]] = None
        
        # Stage 1: Lossless optimization (try even if far over target - it might still help)
        if self._shutdown_checker():
            elapsed = time.time() - stage_start_time
            logger.info(f"Optimization stages cancelled (shutdown requested) after {elapsed:.2f}s")
            return False, False
        
        stage1_start = time.time()
        over_ratio = (current_bytes - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
        logger.debug(f"Stage 1 (lossless): over_ratio={over_ratio:.2%}, threshold={self.skip_gifsicle_far_over_ratio:.2%}")
        
        # Fast-fail: Skip lossless optimization when over_ratio is extreme (>250%)
        # Lossless optimization is unlikely to help when file is 2.5x+ over target
        EXTREME_OVER_RATIO_THRESHOLD = 2.5  # 250%
        if over_ratio > EXTREME_OVER_RATIO_THRESHOLD:
            elapsed = time.time() - stage1_start
            logger.info(f"Stage 1: Skipping lossless optimization (over_ratio={over_ratio:.2%} > {EXTREME_OVER_RATIO_THRESHOLD:.0%} threshold) [exit_reason=fast_fail, elapsed={elapsed:.2f}s]")
        # Try lossless optimization even if far over target - it can still reduce size
        # Only skip if gifsicle is not available or extreme over_ratio
        elif self._is_tool_available("gifsicle"):
            logger.info("Stage 1: Attempting lossless gifsicle optimization...")
            if self._stage_lossless_optimization(gif_path, target_bytes):
                current_bytes = os.path.getsize(gif_path)
                current_size_mb = current_bytes / 1024 / 1024
                elapsed = time.time() - stage1_start
                logger.info(f"Stage 1 completed: {current_size_mb:.2f}MB [exit_reason=success, elapsed={elapsed:.2f}s]")
                if current_bytes <= target_bytes:
                    total_elapsed = time.time() - stage_start_time
                    logger.info(f"Target met after Stage 1 [total_elapsed={total_elapsed:.2f}s]")
                    return True, False
            else:
                elapsed = time.time() - stage1_start
                logger.debug(f"Stage 1: Lossless optimization skipped or failed [exit_reason=failed, elapsed={elapsed:.2f}s]")
        else:
            elapsed = time.time() - stage1_start
            logger.warning(f"Stage 1: gifsicle not available, skipping lossless optimization [exit_reason=tool_unavailable, elapsed={elapsed:.2f}s]")
        
        # Stage 2: Source re-encoding (if source available - preferred over GIF re-encoding)
        # IMPROVED: Keep Stage 2 result as baseline for Stage 3 if it's better than original
        # FIX: Add retry limits, convergence checking, and early exit conditions
        if self._shutdown_checker():
            return False, False
        
        if source_video and os.path.exists(source_video):
            # Check convergence before Stage 2 - if converged, skip Stage 2
            skip_stage2_converged = False
            skip_stage2_projection = False
            if previous_result_size_mb is not None:
                if self._is_converged(current_size_mb, previous_result_size_mb, target_size_mb):
                    logger.info(
                        f"Stage 2: Skipping (converged: {current_size_mb:.2f}MB vs previous {previous_result_size_mb:.2f}MB)"
                    )
                    skip_stage2_converged = True
                else:
                    logger.debug(
                        f"Stage 2: Not converged (current: {current_size_mb:.2f}MB, previous: {previous_result_size_mb:.2f}MB)"
                    )

            # Projection guard: skip Stage 2 when predicted result is not materially better
            if not skip_stage2_converged:
                if gif_info_cache is None:
                    try:
                        gif_info_cache = get_gif_info(gif_path)
                    except Exception:
                        gif_info_cache = {}
                duration_val = float(gif_info_cache.get('duration', 0.0) or 0.0)
                fps_val = float(gif_info_cache.get('fps', 0.0) or 0.0)
                if (not fps_val or fps_val <= 0) and duration_val > 0:
                    try:
                        frame_count_val = float(gif_info_cache.get('frame_count', 0.0) or 0.0)
                        if frame_count_val > 0:
                            fps_val = frame_count_val / duration_val
                    except Exception:
                        fps_val = 0.0
                current_params = {
                    'width': gif_info_cache.get('width', 360),
                    'height': gif_info_cache.get('height', gif_info_cache.get('width', 360)),
                    'fps': max(fps_val, 1.0) if fps_val else 20.0,
                    'colors': gif_info_cache.get('colors', 256),
                    'lossy': gif_info_cache.get('lossy', 0),
                }
                try:
                    projected_params = self._calculate_required_parameters(
                        current_size_mb,
                        target_size_mb,
                        current_params,
                        gif_info_cache,
                        gif_path=gif_path
                    )
                    predicted_size_mb = self._predict_gif_size(
                        int(projected_params.get('width', current_params['width'] or 360)),
                        int(projected_params.get('height', current_params['height'] or current_params['width'] or 360)),
                        float(projected_params.get('fps', current_params['fps'] or 20.0)),
                        float(duration_val or gif_info_cache.get('duration', 0.0) or 0.0),
                        int(projected_params.get('colors', current_params['colors'] or 256)),
                        int(projected_params.get('lossy', current_params['lossy'] or 0)),
                    )
                    if predicted_size_mb >= current_size_mb * 0.9:
                        skip_stage2_projection = True
                        logger.info(
                            "Stage 2: Skipping re-encode (projected %.2fMB >= 90%% of current %.2fMB) [exit_reason=projection_skip]",
                            predicted_size_mb,
                            current_size_mb,
                        )
                except Exception as projection_e:
                    logger.debug(f"Stage 2 projection check failed, proceeding with re-encode: {projection_e}")
            
            if not skip_stage2_converged and not skip_stage2_projection:
                # Fast-fail: Skip Stage 2 re-encoding when over_ratio is extreme (>250%)
                # Re-encoding is unlikely to help when file is 2.5x+ over target
                stage2_start = time.time()
                current_over_ratio = (current_bytes - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
                EXTREME_OVER_RATIO_THRESHOLD = 2.5  # 250%
                if current_over_ratio > EXTREME_OVER_RATIO_THRESHOLD:
                    elapsed = time.time() - stage2_start
                    logger.info(f"Stage 2: Skipping re-encoding (over_ratio={current_over_ratio:.2%} > {EXTREME_OVER_RATIO_THRESHOLD:.0%} threshold, will try Stage 3 lossy compression) [exit_reason=fast_fail, elapsed={elapsed:.2f}s]")
                else:
                    logger.info(f"Stage 2: Attempting re-encoding from source video: {source_video}")
                    
                    # Create temp backup of current state before Stage 2
                    with temp_file_context("stage2_backup", ".gif", self.temp_dir) as stage2_backup:
                        try:
                            safe_file_operation(shutil.copy2, gif_path, stage2_backup)
                            backup_size = os.path.getsize(stage2_backup)
                        except Exception as e:
                            logger.debug(f"Could not create Stage 2 backup: {e}")
                            stage2_backup = None
                            backup_size = os.path.getsize(gif_path)
                        
                        # Track Stage 2 attempts and results for retry limits and progress detection
                        stage2_max_attempts = 3
                        stage2_attempt = 0
                        stage2_previous_size_mb = current_size_mb
                        stage2_seen_sizes = set()  # Track seen result sizes to detect no progress
                        size_tolerance_mb = 1.0  # Consider results within 1MB as identical
                        
                        while stage2_attempt < stage2_max_attempts:
                            stage2_attempt += 1
                            logger.info(f"Stage 2 attempt {stage2_attempt}/{stage2_max_attempts}")
                            
                            # Early exit: if result is identical to input (within tolerance), skip to Stage 3
                            if abs(current_size_mb - backup_size / 1024 / 1024) < size_tolerance_mb:
                                logger.info(
                                    f"Stage 2: Result identical to input ({current_size_mb:.2f}MB), skipping to Stage 3"
                                )
                                break
                            
                            # Early exit: if we've seen this size before, no progress is being made
                            current_size_rounded = round(current_size_mb / size_tolerance_mb) * size_tolerance_mb
                            if current_size_rounded in stage2_seen_sizes:
                                logger.warning(
                                    f"Stage 2: No progress detected (seen size {current_size_mb:.2f}MB before), "
                                    f"skipping to Stage 3 after {stage2_attempt - 1} attempts"
                                )
                                break
                            stage2_seen_sizes.add(current_size_rounded)
                            
                            if self._reencode_from_source(
                                source_video,
                                gif_path,
                                target_size_mb,
                                force_guardrails=force_guardrails,
                                previous_result_size_mb=stage2_previous_size_mb,
                                seen_sizes=stage2_seen_sizes
                            ):
                                current_bytes = os.path.getsize(gif_path)
                                current_size_mb = current_bytes / 1024 / 1024
                                logger.info(f"Stage 2 attempt {stage2_attempt} completed: {current_size_mb:.2f}MB")
                                
                                if current_bytes <= target_bytes:
                                    logger.info("Target met after Stage 2")
                                    return True, False
                                
                                # Check if result is identical to previous attempt (no progress)
                                if abs(current_size_mb - stage2_previous_size_mb) < size_tolerance_mb:
                                    logger.warning(
                                        f"Stage 2: No progress made (result {current_size_mb:.2f}MB same as previous "
                                        f"{stage2_previous_size_mb:.2f}MB), stopping Stage 2"
                                    )
                                    break
                                
                                # Check if result is worse than previous attempt
                                if current_size_mb > stage2_previous_size_mb + size_tolerance_mb:
                                    logger.warning(
                                        f"Stage 2: Result worse than previous ({current_size_mb:.2f}MB > "
                                        f"{stage2_previous_size_mb:.2f}MB), stopping Stage 2"
                                    )
                                    # Restore to previous best
                                    if stage2_backup and os.path.exists(stage2_backup):
                                        safe_file_operation(shutil.copy2, stage2_backup, gif_path)
                                        current_bytes = backup_size
                                        current_size_mb = backup_size / 1024 / 1024
                                    break
                                
                                # Update previous size for next iteration
                                stage2_previous_size_mb = current_size_mb
                            else:
                                logger.debug(f"Stage 2 attempt {stage2_attempt}: Source re-encoding failed or did not improve")
                                # If this is not the first attempt, break to avoid infinite retries
                                if stage2_attempt > 1:
                                    logger.info("Stage 2: Multiple failures, stopping retries")
                                    break
                        
                        # Final check: if Stage 2 result is better and within chaining range
                        stage2_elapsed = time.time() - stage2_start
                        if os.path.exists(gif_path):
                            final_bytes = os.path.getsize(gif_path)
                            final_size_mb = final_bytes / 1024 / 1024
                            
                            if final_bytes < backup_size:
                                over_ratio = (final_bytes - target_bytes) / target_bytes if target_bytes > 0 else 1.0
                                if over_ratio <= 0.30:  # Within 30% of target and improved
                                    logger.info(
                                        f"Stage 2 result ({final_size_mb:.2f}MB, {over_ratio*100:.1f}% over target) "
                                        f"is close to target and better than before. Using as baseline for Stage 3. [exit_reason=success, elapsed={stage2_elapsed:.2f}s]"
                                    )
                            else:
                                logger.info(
                                    f"Stage 2 result ({final_size_mb:.2f}MB) improved size but still far over target "
                                    f"({over_ratio*100:.1f}%). Keeping for subsequent stages."
                                )
                        elif final_bytes > backup_size:
                            # Stage 2 made it worse - restore backup
                            if stage2_backup and os.path.exists(stage2_backup):
                                logger.debug("Stage 2 increased size, restoring pre-Stage-2 state")
                                safe_file_operation(shutil.copy2, stage2_backup, gif_path)
                                current_bytes = backup_size
                                current_size_mb = backup_size / 1024 / 1024
                        else:
                            # Same size - update current_size_mb
                            current_bytes = final_bytes
                            current_size_mb = final_size_mb
            else:
                logger.debug("Stage 2: Skipped (converged or no source video)")
        else:
            logger.debug("Stage 2: No source video available, skipping")
        
        # Stage 3: Gifsicle lossy compression (ALWAYS attempt this for files over target)
        if self._shutdown_checker():
            return False, False
        
        logger.info(
            "Stage 3: Attempting gifsicle lossy compression (%.2fMB -> %.2fMB target)...",
            current_size_mb,
            target_size_mb,
        )
        gifsicle_available = self._is_tool_available("gifsicle")
        gifsicle_attempted = False
        gifsicle_improved = False
        
        if gifsicle_available:
            size_before_gifsicle = current_bytes
            # _stage_gifsicle_lossy_compression returns True only when it managed to get under
            # the target, but it may still reduce size even when it returns False. We therefore
            # treat the boolean as \"made a successful under-target attempt\" and always re-stat
            # the file afterwards to determine actual size and improvement.
            logger.debug(f"Stage 3: Starting gifsicle lossy compression on {os.path.basename(gif_path)} ({size_before_gifsicle / (1024*1024):.2f}MB -> {target_size_mb:.2f}MB target)")
            self._stage_gifsicle_lossy_compression(gif_path, target_bytes, target_size_mb, source_video=source_video)
            gifsicle_attempted = True
            
            if os.path.exists(gif_path):
                current_bytes = os.path.getsize(gif_path)
                current_size_mb = current_bytes / 1024 / 1024
                gifsicle_improved = current_bytes < size_before_gifsicle
                logger.debug(f"Stage 3: After gifsicle, file {os.path.basename(gif_path)} is {current_size_mb:.2f}MB (was {size_before_gifsicle / (1024*1024):.2f}MB, improved: {gifsicle_improved})")
                
                if gifsicle_improved:
                    logger.info(
                        "Stage 3 (gifsicle) completed: %.2fMB (was %.2fMB)",
                        current_size_mb,
                        size_before_gifsicle / 1024 / 1024,
                    )
                else:
                    logger.warning(
                        "Stage 3: Gifsicle lossy compression did not reduce size (still %.2fMB)",
                        current_size_mb,
                    )
                
                if current_bytes <= target_bytes:
                    logger.info("Target met after Stage 3 (gifsicle)")
                    return True, False
            else:
                logger.warning("Stage 3: GIF file missing after gifsicle compression")
                return False, False
        else:
            logger.warning("Stage 3: gifsicle not available, will try FFmpeg and other fallbacks")
        
        # Stage 3b: Near-target bounded search (when gifsicle got close but not under)
        # Verify file exists and get current size
        if not os.path.exists(gif_path):
            logger.warning(f"Stage 3b: GIF file does not exist: {gif_path}")
            current_bytes = 0
            current_size_mb = 0.0
        else:
            current_bytes = os.path.getsize(gif_path)
            current_size_mb = current_bytes / 1024 / 1024
            over_ratio = (current_bytes - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
            
            # Verify we're using the optimized file, not the original
            # If gifsicle improved, the file should be smaller than original_size_mb
            if gifsicle_improved and current_size_mb > original_size_mb * 0.9:
                logger.warning(
                    f"Stage 3b: File size ({current_size_mb:.2f}MB) suggests we may be using original file "
                    f"instead of optimized. Expected < {original_size_mb * 0.9:.2f}MB. "
                    f"Verifying file replacement succeeded..."
                )
                # Re-check file to ensure it's the optimized version
                if os.path.exists(gif_path):
                    recheck_size = os.path.getsize(gif_path)
                    recheck_size_mb = recheck_size / 1024 / 1024
                    if abs(recheck_size - current_bytes) > 1024:  # More than 1KB difference
                        logger.warning(f"Stage 3b: File size changed between checks: {current_size_mb:.2f}MB -> {recheck_size_mb:.2f}MB")
                        current_bytes = recheck_size
                        current_size_mb = recheck_size_mb
                        over_ratio = (current_bytes - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
            
            # Get near-target config (shared with later stages)
            optimization_cfg = self.config_helper.get_optimization_config()
            near_target_config = optimization_cfg.get('near_target', {})
            near_threshold = near_target_config.get('threshold_percent', 15) / 100.0
            mode = near_target_config.get('mode', 'both')
            abs_threshold_mb = float(near_target_config.get('absolute_mb_threshold', 0.0) or 0.0)
            
            gifsicle_near_target = gifsicle_attempted and gifsicle_improved and 0 < over_ratio <= near_threshold
            
            if gifsicle_near_target and self._is_tool_available("gifsicle") and mode in ['bounded_search', 'both']:
                logger.info(
                    "Stage 3b: Gifsicle got close (over by %.1f%%), attempting bounded search on %.2fMB file...",
                    over_ratio * 100,
                    current_size_mb
                )
                if self._stage_adaptive_search(gif_path, target_bytes):
                    current_bytes = os.path.getsize(gif_path)
                    current_size_mb = current_bytes / 1024 / 1024
                    logger.info("Stage 3b (bounded search) completed: %.2fMB", current_size_mb)
                    if current_bytes <= target_bytes:
                        logger.info("Target met after Stage 3b (bounded search)")
                        return True, False
        
        # Stage 3c: FFmpeg fallback when gifsicle fails, makes no progress, or file is still far over target
        if self._shutdown_checker():
            return False, False
        
        # Ensure we have current file size for Stage 3c
        if os.path.exists(gif_path):
            current_bytes = os.path.getsize(gif_path)
            current_size_mb = current_bytes / 1024 / 1024
        else:
            logger.warning("Stage 3c: GIF file does not exist")
            return False, False
        
        if current_bytes > target_bytes:
            over_ratio = (current_bytes - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
            size_delta_mb = current_size_mb - target_size_mb
            guardrail_context, gif_info_cache = self._guardrail_context_for_gif(
                gif_path,
                force_guardrails,
                gif_info_cache
            )
            absolute_trigger = bool(
                guardrail_context and abs_threshold_mb > 0 and size_delta_mb >= abs_threshold_mb
            )
            
            # Decide whether FFmpeg fallback should be attempted
            # Strategy: Skip FFmpeg fallback when gifsicle just optimized the GIF and result is close to target,
            # since re-encoding from an already-optimized GIF often increases file size rather than reducing it.
            use_ffmpeg = False
            if not gifsicle_available or not gifsicle_attempted:
                # Never ran gifsicle: FFmpeg is our main option
                use_ffmpeg = True
            elif not gifsicle_improved:
                # Gifsicle couldn't make any progress; try FFmpeg
                use_ffmpeg = True
            else:
                # Gifsicle improved the file but it is still over target. Only use FFmpeg
                # if we are still significantly over the target (beyond near-threshold band).
                # Note: _stage_ffmpeg_fallback_compression will do additional checks to skip
                # when gifsicle_just_applied=True and result is close to target (<20% over).
                if over_ratio > near_threshold:
                    use_ffmpeg = True
                elif absolute_trigger:
                    logger.info(
                        "Stage 3c: Guardrail clip %.2fMB over target (threshold %.2fMB); enabling FFmpeg fallback despite near-target ratio",
                        size_delta_mb,
                        abs_threshold_mb
                    )
                    use_ffmpeg = True
                else:
                    logger.info(
                        "Stage 3c: Skipping FFmpeg fallback for near-target gifsicle result "
                        "(over by %.1f%%, threshold %.0f%%)",
                        over_ratio * 100,
                        near_threshold * 100,
                    )
            
            # Respect near_target.mode == 'bounded_search', which explicitly prefers gifsicle-only paths
            if use_ffmpeg and mode == 'bounded_search' and gifsicle_near_target:
                logger.info(
                    "Stage 3c: Near-target mode is 'bounded_search'; not running FFmpeg fallback "
                    "for near-target result (over by %.1f%%)",
                    over_ratio * 100,
                )
                use_ffmpeg = False
            
            if use_ffmpeg:
                logger.info(
                    "Stage 3c: Attempting FFmpeg fallback compression (%.2fMB -> %.2fMB target)...",
                    current_size_mb,
                    target_size_mb,
                )
                # Pass information about whether gifsicle was just applied
                gifsicle_just_applied = gifsicle_attempted and gifsicle_improved
                if self._stage_ffmpeg_fallback_compression(gif_path, target_bytes, target_size_mb,
                                                           gifsicle_just_applied=gifsicle_just_applied):
                    current_bytes = os.path.getsize(gif_path)
                    current_size_mb = current_bytes / 1024 / 1024
                    logger.info("Stage 3c (FFmpeg fallback) completed: %.2fMB", current_size_mb)
                    if current_bytes <= target_bytes:
                        logger.info("Target met after Stage 3c (FFmpeg fallback)")
                        return True, False
                else:
                    logger.warning("Stage 3c: FFmpeg fallback compression failed")
                    # Try progressive resolution reduction as additional fallback, but only when
                    # still significantly over the target size.
                    current_bytes = os.path.getsize(gif_path)
                    current_size_mb = current_bytes / 1024 / 1024
                    if current_bytes > target_bytes:
                        progressive_threshold = 0.25  # Only for files ≥25% over target
                        over_ratio = (current_bytes - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
                        if over_ratio >= progressive_threshold:
                            logger.info(
                                "Stage 3c.1: Attempting progressive resolution reduction "
                                "(%.2fMB -> %.2fMB target, over by %.1f%%)...",
                                current_size_mb,
                                target_size_mb,
                                over_ratio * 100,
                            )
                            if self._progressive_resolution_reduction(gif_path, target_bytes, target_size_mb):
                                current_bytes = os.path.getsize(gif_path)
                                current_size_mb = current_bytes / 1024 / 1024
                                logger.info(
                                    "Stage 3c.1 (progressive resolution) completed: %.2fMB",
                                    current_size_mb,
                                )
                                if current_bytes <= target_bytes:
                                    logger.info("Target met after Stage 3c.1 (progressive resolution)")
                                    return True, False
                        else:
                            logger.debug(
                                "Stage 3c.1: Skipping progressive resolution reduction "
                                "(over_ratio=%.2f%% < %.0f%% threshold)",
                                over_ratio * 100,
                                progressive_threshold * 100,
                            )
        
        # Stage 3d: PIL-based compression (when still over target)
        if self._shutdown_checker():
            return False, False
        
        current_bytes = os.path.getsize(gif_path)
        if current_bytes > target_bytes:
            current_size_mb = current_bytes / 1024 / 1024
            over_ratio = (current_bytes - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
            
            # Try PIL compression for near-target files (lowered from 5% to 1% threshold)
            if over_ratio > 0.01:  # More than 1% over
                logger.info(f"Stage 3d: Attempting PIL-based compression ({current_size_mb:.2f}MB -> {target_size_mb:.2f}MB target)...")
                if self._stage_pil_compression(gif_path, target_bytes, target_size_mb):
                    current_bytes = os.path.getsize(gif_path)
                    current_size_mb = current_bytes / 1024 / 1024
                    logger.info(f"Stage 3d (PIL compression) completed: {current_size_mb:.2f}MB")
                    if current_bytes <= target_bytes:
                        logger.info("Target met after Stage 3d (PIL compression)")
                        return True, False
                else:
                    logger.debug("Stage 3d: PIL compression failed or did not reduce size sufficiently")
        
        # Stage 4 removed: was duplicate of Stage 3b (both called _bounded_gifsicle_near_target)
        # Final success check
        current_bytes = os.path.getsize(gif_path)
        final_size_mb = current_bytes / 1024 / 1024
        success = current_bytes <= target_bytes
        
        # Check for convergence if we have a previous result
        converged = False
        if previous_result_size_mb is not None:
            converged = self._is_converged(final_size_mb, previous_result_size_mb, target_size_mb)
            # Also check if we're within 2% of target but still over (optimizer reached its limit)
            if not success and not converged:
                over_ratio = (current_bytes - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
                if over_ratio <= 0.02:  # Within 2% of target
                    converged = True
                    logger.info(
                        f"Optimizer reached limit: result ({final_size_mb:.2f}MB) is within 2% of target "
                        f"({target_size_mb:.2f}MB) but still over. Further optimization unlikely to help."
                    )
        
        if not success:
            # Log detailed warning when optimization exhausts all methods
            over_pct = ((current_bytes - target_bytes) / target_bytes) * 100
            logger.warning(
                f"Single-GIF optimization exhausted all methods. "
                f"Best result: {final_size_mb:.2f}MB (target: {target_size_mb:.2f}MB, {over_pct:.1f}% over). "
                f"Consider segmentation for this file."
            )
        
        logger.info(f"Optimization stages complete: {final_size_mb:.2f}MB, target met: {success}, converged: {converged}")
        return success, converged
    
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
        
        # Verify input file exists and log its size for debugging
        if not os.path.exists(gif_path):
            logger.warning(f"Adaptive search: Input file does not exist: {gif_path}")
            return False
        
        input_size = os.path.getsize(gif_path)
        input_size_mb = input_size / (1024 * 1024)
        target_mb = target_bytes / (1024 * 1024)
        logger.debug(f"Adaptive search: Starting from {input_size_mb:.2f}MB file (target: {target_mb:.2f}MB), path: {os.path.basename(gif_path)}")
        
        with temp_file_context("adaptive_search", ".gif", self.temp_dir) as temp_output:
            if self._bounded_gifsicle_near_target(gif_path, temp_output, target_bytes):
                if os.path.exists(temp_output):
                    try:
                        safe_file_operation(os.replace, temp_output, gif_path)
                        final_size = os.path.getsize(gif_path)
                        final_size_mb = final_size / (1024 * 1024)
                        logger.debug(f"Adaptive search succeeded: {final_size_mb:.2f}MB")
                        return True
                    except Exception as e:
                        logger.debug(f"Failed to replace with adaptive search result: {e}")
        
        return False
    
    def _reencode_with_params(self, source_video: str, output_path: str, target_size_mb: float, 
                              params: Dict[str, Any], video_info: Dict[str, Any]) -> bool:
        """
        Helper method to re-encode with specific parameters.
        
        Args:
            source_video: Path to source video file
            output_path: Path to output GIF file
            target_size_mb: Target size in MB
            params: Encoding parameters (width, fps, colors, etc.)
            video_info: Video information dict
        
        Returns:
            True if re-encoding succeeded
        """
        try:
            if self._shutdown_checker():
                return False
            
            from .gif_generator import GifGenerator
            generator = GifGenerator(self.config, shutdown_checker=self._shutdown_checker)
            
            # Create settings dict from params
            colors = int(params.get('colors', 256))
            colors = max(2, min(256, colors))
            if colors >= 192:
                dither = 'floyd_steinberg'
            elif colors >= 128:
                dither = 'sierra2_4a'
            else:
                dither = 'bayer'
            
            settings = {
                'max_size_mb': target_size_mb,
                'width': params['width'],
                'height': params.get('height', -1),
                'fps': params['fps'],
                'colors': colors,
                'palette_max_colors': colors,
                'dither': dither,
                'lossy': params.get('lossy', 0),
                'max_duration': video_info.get('duration', 30.0)
            }
            
            # Create GIF from source with calculated parameters
            duration = min(settings['max_duration'], video_info.get('duration', 30.0))
            palette_timeout_override = None
            try:
                timeout_cfg = self.config.get('gif_settings.performance.timeouts.palette', {}) or {}
            except Exception:
                timeout_cfg = {}
            reencode_multiplier = float(timeout_cfg.get('reencode_duration_multiplier', 0.0) or 0.0)
            if reencode_multiplier > 0 and duration > 0:
                min_seconds = int(timeout_cfg.get('min_seconds', 30))
                max_seconds = int(timeout_cfg.get('max_seconds', 240))
                palette_timeout_override = int(
                    max(min_seconds, min(max_seconds, math.ceil(duration * reencode_multiplier)))
                )
            result = generator._create_single_gif(
                source_video, output_path, settings, 0.0, duration, skip_optimizer=True,
                disable_parallel_palette=True,
                palette_timeout_override=palette_timeout_override
            )
            
            return result.get('success', False) and os.path.exists(output_path)
            
        except Exception as e:
            logger.debug(f"Re-encoding with params failed: {e}")
            return False
    
    def _reencode_from_source(self, source_video: str, output_gif: str, target_size_mb: float,
                              force_guardrails: bool = False, previous_result_size_mb: Optional[float] = None,
                              seen_sizes: Optional[set] = None) -> bool:
        """
        Re-encode GIF from source video with multiple parallel strategies to find best result.
        This is preferred over re-encoding an already-compressed GIF.
        
        Args:
            source_video: Path to source video file
            output_gif: Path to output GIF file
            target_size_mb: Target size in MB
            force_guardrails: Force guardrail heuristics even if clip duration is short
            previous_result_size_mb: Optional previous result size for progress detection
            seen_sizes: Optional set of seen result sizes (rounded) to detect no progress
        
        Returns:
            True if re-encoding succeeded and file is under or reasonably close to target
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
            source_duration = float(video_info.get('duration', 0.0) or 0.0)
            
            # Get current GIF info to understand what we're optimizing
            gif_info = get_gif_info(output_gif) if os.path.exists(output_gif) else {}
            current_size_mb = os.path.getsize(output_gif) / 1024 / 1024 if os.path.exists(output_gif) else target_size_mb * 2
            
            # Calculate base parameters
            current_params = {
                'width': gif_info.get('width', video_info.get('width', 360)),
                'height': gif_info.get('height', video_info.get('width', 360)),
                'fps': gif_info.get('fps', 20),
                'colors': 256,
                'lossy': 0
            }
            
            base_params = self._calculate_required_parameters(
                current_size_mb, target_size_mb, current_params, gif_info, gif_path=output_gif
            )
            guardrails_cfg = getattr(self, 'long_clip_guardrails', {}) or {}
            segment_duration = float(gif_info.get('duration', source_duration) or source_duration)
            segment_fps = base_params.get('fps', 20)
            guardrails_active = self._is_guardrail_context(
                segment_duration,
                segment_fps,
                force_guardrails=force_guardrails
            )
            if guardrails_active:
                logger.info(
                    "Long-clip guardrails active for Stage 2 re-encoding (segment duration=%.1fs, fps=%.1f)",
                    segment_duration,
                    segment_fps
                )
            
            # IMPROVED: Create 3 parameter strategies for parallel execution
            # FIX: Adapt parameters if previous attempt produced identical results
            size_tolerance_mb = 1.0
            adapt_aggressively = False
            
            # Check if we need more aggressive parameters (previous result was identical)
            if previous_result_size_mb is not None:
                size_diff = abs(current_size_mb - previous_result_size_mb)
                if size_diff < size_tolerance_mb:
                    logger.debug(
                        f"Previous result identical to current ({current_size_mb:.2f}MB), "
                        f"using more aggressive parameters"
                    )
                    adapt_aggressively = True
            
            strategies = [
                {
                    'name': 'balanced',
                    'width': base_params['width'],
                    'fps': base_params['fps'],
                    'colors': base_params['colors'],
                    'lossy': base_params.get('lossy', 0),
                },
                {
                    'name': 'conservative',
                    'width': int(base_params['width'] * (1.1 if not adapt_aggressively else 1.05)),  # Less conservative if adapting
                    'fps': min(base_params['fps'] + (2 if not adapt_aggressively else 1), 30),
                    'colors': min(base_params['colors'] + (32 if not adapt_aggressively else 16), 256),
                    'lossy': base_params.get('lossy', 0),
                },
                {
                    'name': 'aggressive',
                    'width': int(base_params['width'] * (0.9 if not adapt_aggressively else 0.85)),  # More aggressive if adapting
                    'fps': max(base_params['fps'] - (2 if not adapt_aggressively else 3), 10),
                    'colors': max(base_params['colors'] - (32 if not adapt_aggressively else 48), 64),
                    'lossy': base_params.get('lossy', 0),
                },
            ]
            if guardrails_active:
                preferred_order = ['balanced', 'aggressive', 'conservative']
                ordered = []
                for name in preferred_order:
                    ordered.extend([s for s in strategies if s['name'] == name and s not in ordered])
                max_strategies = max(1, int(guardrails_cfg.get('reencode_strategy_limit', len(ordered)) or len(ordered)))
                strategies = ordered[:max_strategies]
            
            # Run strategies in parallel using ThreadPoolExecutor (max 3 workers to avoid overload)
            from concurrent.futures import ThreadPoolExecutor, as_completed
            results = []
            temp_files = []
            
            max_workers = 3
            if guardrails_active:
                max_workers = max(1, int(guardrails_cfg.get('reencode_max_workers', 1) or 1))
            logger.debug("Running %d parallel re-encoding strategies...", max_workers)
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for strategy in strategies:
                    temp_output = create_unique_temp_filename(f"reencode_{strategy['name']}", ".gif", self.temp_dir)
                    temp_files.append(temp_output)
                    future = executor.submit(
                        self._reencode_with_params,
                        source_video, temp_output, target_size_mb, strategy, video_info
                    )
                    futures[future] = (temp_output, strategy['name'])
                
                # Collect results with timeout
                for future in as_completed(futures, timeout=120):
                    temp_output, name = futures[future]
                    try:
                        if future.result() and os.path.exists(temp_output):
                            size_mb = os.path.getsize(temp_output) / 1024 / 1024
                            results.append((temp_output, size_mb, name))
                            logger.debug(f"Strategy '{name}': {size_mb:.2f}MB")
                    except Exception as e:
                        logger.debug(f"Strategy {name} failed: {e}")
            
            # Pick best result: closest to target without going too far under (prefer quality)
            # FIX: Filter out results that are identical to previous or seen sizes
            best = None
            # FIX: Get original size before we overwrite the file
            original_size_mb = os.path.getsize(output_gif) / 1024 / 1024 if os.path.exists(output_gif) else float('inf')
            
            # Track seen sizes for this call
            if seen_sizes is None:
                seen_sizes = set()
            
            for temp_path, size_mb, name in results:
                # Skip results that are identical to previous attempt or seen sizes
                size_rounded = round(size_mb / size_tolerance_mb) * size_tolerance_mb
                if previous_result_size_mb is not None:
                    prev_rounded = round(previous_result_size_mb / size_tolerance_mb) * size_tolerance_mb
                    if abs(size_rounded - prev_rounded) < size_tolerance_mb:
                        logger.debug(f"Strategy '{name}': Skipping (identical to previous: {size_mb:.2f}MB)")
                        continue
                
                if size_rounded in seen_sizes:
                    logger.debug(f"Strategy '{name}': Skipping (seen size before: {size_mb:.2f}MB)")
                    continue
                
                if size_mb <= target_size_mb:
                    # Meets target - prefer largest (best quality) that still meets target
                    if not best or size_mb > best[1]:
                        best = (temp_path, size_mb, name)
                elif not best or size_mb < best[1]:
                    # Over target - prefer smallest
                    best = (temp_path, size_mb, name)
            
            # Apply best result
            if best:
                # Check if best result is identical to previous (no progress)
                best_size_rounded = round(best[1] / size_tolerance_mb) * size_tolerance_mb
                if previous_result_size_mb is not None:
                    prev_rounded = round(previous_result_size_mb / size_tolerance_mb) * size_tolerance_mb
                    if abs(best_size_rounded - prev_rounded) < size_tolerance_mb:
                        logger.warning(
                            f"Best strategy '{best[2]}' produced identical result ({best[1]:.2f}MB), "
                            f"no progress made"
                        )
                        # Still apply it, but caller will detect this and stop retrying
                
                try:
                    safe_file_operation(shutil.copy2, best[0], output_gif)
                    logger.info(f"Selected '{best[2]}' strategy: {best[1]:.2f}MB (target: {target_size_mb:.2f}MB)")
                    success = best[1] <= target_size_mb
                    
                    # Add to seen sizes
                    seen_sizes.add(best_size_rounded)
                    
                    # Cleanup all temp files
                    for temp_file in temp_files:
                        try:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                        except Exception:
                            pass
                    
                    # FIX: If result is better than original but still over target, return True
                    # to allow Stage 3 optimization to continue. Only return False if result
                    # is worse than original or doesn't exist.
                    if not success:
                        # Check if this is an improvement over the original
                        if best[1] < original_size_mb:
                            # Improved but still over target - allow Stage 3 to optimize further
                            logger.debug(f"Stage 2 improved size ({best[1]:.2f}MB < {original_size_mb:.2f}MB) but still over target, continuing to Stage 3")
                            return True
                    
                    return success
                except Exception as e:
                    logger.debug(f"Failed to apply best result: {e}")
            
            # Cleanup temp files on failure
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception:
                    pass
            
            return False
            
        except Exception as e:
            logger.debug(f"Source re-encoding failed: {e}")
            return False
    
    def _guardrail_context_for_gif(self, gif_path: str, force_guardrails: bool,
                                   cached_info: Optional[Dict[str, Any]]) -> Tuple[bool, Dict[str, Any]]:
        gif_info = cached_info
        if gif_info is None:
            try:
                gif_info = get_gif_info(gif_path)
            except Exception:
                gif_info = {}
        duration = float(gif_info.get('duration', 0.0) or 0.0)
        fps = float(gif_info.get('fps', 0.0) or 0.0)
        if (not fps or fps <= 0) and duration > 0:
            try:
                frame_count = float(gif_info.get('frame_count', 0.0) or 0.0)
                if frame_count > 0:
                    fps = frame_count / duration
            except Exception:
                fps = 0.0
        active = self._is_guardrail_context(duration, fps, force_guardrails=force_guardrails)
        return active, gif_info
    
    def _is_guardrail_context(self, duration: float, fps: float, force_guardrails: bool = False) -> bool:
        guardrails_cfg = getattr(self, 'long_clip_guardrails', {}) or {}
        if not guardrails_cfg.get('enabled', False):
            return False
        if force_guardrails:
            return True
        duration = float(duration or 0.0)
        fps = float(fps or 0.0)
        frame_budget_threshold = float(guardrails_cfg.get('frame_budget', float('inf')) or float('inf'))
        duration_threshold = float(guardrails_cfg.get('duration_seconds', float('inf')) or float('inf'))
        frame_count = duration * fps
        if duration >= duration_threshold or frame_count >= frame_budget_threshold:
            return True
        duration_ratio = float(guardrails_cfg.get('segment_duration_ratio', 0.0) or 0.0)
        frame_ratio = float(guardrails_cfg.get('segment_frame_budget_ratio', 0.0) or 0.0)
        if duration_ratio > 0 and duration_threshold < float('inf') and duration >= duration_threshold * duration_ratio:
            return True
        if frame_ratio > 0 and frame_budget_threshold < float('inf') and frame_count >= frame_budget_threshold * frame_ratio:
            return True
        return False
    
    def _stage_gifsicle_lossy_compression(self, gif_path: str, target_bytes: int, target_size_mb: float, source_video: Optional[str] = None) -> bool:
        """
        Stage 3: Gifsicle lossy compression with calculated parameters.
        This replaces FFmpeg re-encoding which often makes files larger.
        
        Args:
            gif_path: Path to GIF file
            target_bytes: Target size in bytes
            target_size_mb: Target size in MB
            source_video: Optional source video path (None for GIF-only inputs)
        
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
            current_size_mb, target_size_mb, current_params, gif_info, gif_path=gif_path
        )
        
        logger.info(f"Calculated compression parameters: colors={required_params['colors']}, lossy={required_params['lossy']}, width={required_params['width']}")
        
        over_ratio = (current_size_mb / target_size_mb) if target_size_mb > 0 else float('inf')
        
        # Detect GIF-only early so we can enforce floors in all optimization paths
        is_gif_only = source_video is None or not os.path.exists(source_video)
        quality_floors = self.config_helper.get_quality_floors()
        allow_aggressive = self.config_helper.get_optimization_config().get('allow_aggressive_compression', False)
        
        # Enforce GIF-only floors on required_params BEFORE any optimization strategies
        if is_gif_only:
            min_width = quality_floors.get('min_width', 360) if quality_floors.get('enforce', True) else quality_floors.get('min_width_aggressive', 360)
            min_colors = 160  # Minimum colors for GIF-only to prevent banding
            max_lossy = 100  # Cap lossy for GIF-only
            min_scale = 0.85  # Don't scale below 85% for GIF-only
            
            # Adjust required_params to respect floors
            if required_params['width'] < min_width:
                required_params['width'] = min_width
                required_params['height'] = int(current_params['height'] * (min_width / current_params['width']))
            if required_params['colors'] < min_colors:
                required_params['colors'] = min_colors
            if required_params['lossy'] > max_lossy:
                required_params['lossy'] = max_lossy
            # Ensure scale respects minimum
            calculated_scale = required_params['width'] / float(current_params['width'])
            if calculated_scale < min_scale:
                required_params['width'] = int(current_params['width'] * min_scale)
                required_params['height'] = int(current_params['height'] * min_scale)
                # Re-apply width floor after scale adjustment
                if required_params['width'] < min_width:
                    required_params['width'] = min_width
                    required_params['height'] = int(current_params['height'] * (min_width / current_params['width']))
            
            logger.info(f"GIF-only floors applied: colors={required_params['colors']}, lossy={required_params['lossy']}, width={required_params['width']}")
        
        # For GIF-only inputs (no source video) that are heavily over target (>15%), 
        # try FFmpeg palette re-encode fallback before aggressive gifsicle compression
        if is_gif_only and over_ratio > 1.15:
            logger.info(f"GIF-only input {over_ratio*100:.1f}% over target, attempting FFmpeg palette re-encode fallback...")
            # Use quality-preserving parameters for the fallback
            quality_floors = self.config_helper.get_quality_floors()
            min_width = quality_floors.get('min_width', 360) if quality_floors.get('enforce', True) else quality_floors.get('min_width_aggressive', 360)
            min_fps = quality_floors.get('min_fps', 15) if quality_floors.get('enforce', True) else quality_floors.get('min_fps_aggressive', 15)
            
            # Calculate conservative parameters that preserve quality
            fallback_width = max(min_width, int(current_params['width'] * 0.92))  # Max 8% reduction
            fallback_fps = max(min_fps, int(current_params['fps'] * 0.95))  # Max 5% FPS reduction
            fallback_colors = max(192, required_params.get('colors', 256))  # Preserve more colors
            
            temp_output = create_unique_temp_filename("ffmpeg_fallback", ".gif", self.temp_dir)
            if self._ffmpeg_palette_reencode(
                gif_path, temp_output, fallback_width, fallback_fps, 
                max_colors=fallback_colors, mpdecimate_frac=0.3,
                stats_mode='diff', dither='floyd_steinberg'
            ):
                if os.path.exists(temp_output):
                    fallback_size = os.path.getsize(temp_output)
                    fallback_size_mb = fallback_size / 1024 / 1024
                    if fallback_size <= target_bytes:
                        try:
                            safe_file_operation(os.replace, temp_output, gif_path)
                            logger.info(f"✓ Target met with FFmpeg palette fallback: {fallback_size_mb:.2f}MB")
                            return True
                        except Exception as e:
                            logger.warning(f"Failed to replace file after FFmpeg fallback: {e}")
                    elif fallback_size < current_bytes:
                        # Improved but still over target - use as baseline for further optimization
                        try:
                            safe_file_operation(os.replace, temp_output, gif_path)
                            current_bytes = fallback_size
                            current_size_mb = fallback_size_mb
                            logger.info(f"FFmpeg fallback improved to {fallback_size_mb:.2f}MB, continuing with gifsicle...")
                        except Exception as e:
                            logger.warning(f"Failed to replace file after FFmpeg fallback: {e}")
                    else:
                        # Fallback made it worse, clean up and continue with gifsicle
                        try:
                            os.remove(temp_output)
                        except:
                            pass
                        logger.debug("FFmpeg fallback did not improve size, continuing with gifsicle...")
        
        # Try advanced strategies first if enabled
        if self.multi_pass_enabled and over_ratio > 1.2:
            logger.info("Attempting multi-pass optimization...")
            result = self._gifsicle_multi_pass_optimize(gif_path, target_bytes, target_size_mb)
            if result and os.path.exists(result):
                result_size = os.path.getsize(result)
                if result_size <= target_bytes:
                    try:
                        safe_file_operation(os.replace, result, gif_path)
                        logger.info("✓ Target met with multi-pass optimization!")
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to replace file after multi-pass: {e}")
                elif result_size < current_bytes:
                    # Use as starting point for further optimization
                    try:
                        safe_file_operation(os.replace, result, gif_path)
                        current_bytes = result_size
                        current_size_mb = current_bytes / 1024 / 1024
                        # Note: gif_path remains unchanged since os.replace moves file to gif_path
                        logger.info(f"Multi-pass improved file to {current_size_mb:.2f}MB, continuing...")
                    except Exception as e:
                        logger.warning(f"Failed to replace file after multi-pass: {e}")
        
        if self.progressive_refinement and over_ratio > 1.15:
            logger.info("Attempting progressive refinement...")
            # Note: Progressive refinement uses strategy ranges which may include values below GIF-only floors,
            # but it's quality-aware and typically produces good results. The param combinations below will
            # enforce floors if progressive refinement doesn't meet the target.
            result = self._gifsicle_progressive_refine(gif_path, target_bytes, target_size_mb)
            if result and os.path.exists(result):
                result_size = os.path.getsize(result)
                if result_size <= target_bytes:
                    try:
                        safe_file_operation(os.replace, result, gif_path)
                        logger.info("✓ Target met with progressive refinement!")
                        return True
                    except Exception as e:
                        logger.warning(f"Failed to replace file after progressive refinement: {e}")
        
        # Use adaptive strategy to generate parameter combinations
        strategy = self._adaptive_gifsicle_strategy(current_size_mb, target_size_mb, gif_info)
        
        # Initialize GIF-only floor variables (used in param combination generation)
        # Note: is_gif_only, quality_floors, and allow_aggressive are already set above
        min_colors = 128  # Default minimum colors
        max_lossy = 120  # Default max lossy
        min_scale = 0.7  # Default minimum scale
        min_width = quality_floors.get('min_width', 360) if quality_floors.get('enforce', True) else quality_floors.get('min_width_aggressive', 360)
        
        if is_gif_only:
            # For GIF-only inputs, enforce stricter quality floors
            min_colors = 160  # Minimum colors for GIF-only to prevent banding
            max_lossy = 100  # Cap lossy for GIF-only
            min_scale = 0.85  # Don't scale below 85% for GIF-only
            # required_params already adjusted above, but ensure they're still correct
            if required_params['width'] < min_width:
                required_params['width'] = min_width
                required_params['height'] = int(current_params['height'] * (min_width / current_params['width']))
            if required_params['colors'] < min_colors:
                required_params['colors'] = min_colors
            if required_params['lossy'] > max_lossy:
                required_params['lossy'] = max_lossy
        
        # Try multiple parameter combinations using adaptive strategy
        param_combinations = []
        
        # Primary: use calculated parameters
        primary_scale = required_params['width'] / float(current_params['width'])
        if is_gif_only and primary_scale < min_scale:
            primary_scale = min_scale
            required_params['width'] = int(current_params['width'] * primary_scale)
            required_params['height'] = int(current_params['height'] * primary_scale)
            # Re-apply width floor after scale adjustment
            if required_params['width'] < min_width:
                required_params['width'] = min_width
                required_params['height'] = int(current_params['height'] * (min_width / current_params['width']))
                primary_scale = required_params['width'] / float(current_params['width'])
        
        param_combinations.append({
            'colors': required_params['colors'],
            'lossy': required_params['lossy'],
            'scale': primary_scale,
            'optimize': strategy['optimize'],
            'resize_method': strategy['resize_method'],
            'name': 'primary'
        })
        
        # Generate smart parameter combinations based on adaptive strategy
        # Use orthogonal combinations: test independent parameter axes
        if self.adaptive_strategy:
            # Near-target fine tuning (within 5-15% over target)
            near_target_lower = 1.05
            near_target_upper = 1.15
            if target_size_mb > 0 and near_target_lower <= over_ratio <= near_target_upper:
                base_scale = required_params['width'] / float(max(1, current_params['width']))
                # Try correlated combinations (known-good pairs)
                for colors in strategy['colors_range'][:2]:  # First 2 colors
                    for lossy in strategy['lossy_range'][:2]:  # First 2 lossy
                        fine_tune_scale = max(0.7, min(1.0, base_scale * 0.95))
                        # Enforce floors for GIF-only
                        if is_gif_only:
                            colors = max(colors, min_colors)
                            lossy = min(lossy, max_lossy)
                            fine_tune_scale = max(fine_tune_scale, min_scale)
                        param_combinations.append({
                            'colors': colors,
                            'lossy': lossy,
                            'scale': fine_tune_scale,
                            'optimize': strategy['optimize'],
                            'resize_method': strategy['resize_method'],
                            'name': f'near-target ({strategy["strategy"]})'
                        })
                logger.info(
                    "  Scheduling near-target gifsicle compression (%.1f%% over target)",
                    (over_ratio - 1.0) * 100
                )
            else:
                # For other over-ratios, use strategy-based combinations
                # Test key combinations from the strategy
                for i, colors in enumerate(strategy['colors_range'][:3]):  # Top 3
                    for j, lossy in enumerate(strategy['lossy_range'][:3]):  # Top 3
                        if i + j < 4:  # Limit combinations
                            scale_idx = min(i + j, len(strategy['scale_range']) - 1)
                            scale = strategy['scale_range'][scale_idx]
                            # Enforce floors for GIF-only
                            if is_gif_only:
                                colors = max(colors, min_colors)
                                lossy = min(lossy, max_lossy)
                                scale = max(scale, min_scale)
                            param_combinations.append({
                                'colors': colors,
                                'lossy': lossy,
                                'scale': scale,
                                'optimize': strategy['optimize'],
                                'resize_method': strategy['resize_method'],
                                'name': f'{strategy["strategy"]} combo'
                            })
        else:
            # Fallback to original logic if adaptive strategy disabled
            if 1.15 < over_ratio <= 1.35:
                moderate_colors = max(128, required_params['colors'] - 20)
                moderate_lossy = min(120, required_params['lossy'] + 15)
                moderate_scale = required_params['width'] / float(current_params['width']) * 0.94
                if is_gif_only:
                    moderate_colors = max(moderate_colors, min_colors)
                    moderate_lossy = min(moderate_lossy, max_lossy)
                    moderate_scale = max(moderate_scale, min_scale)
                param_combinations.append({
                    'colors': moderate_colors,
                    'lossy': moderate_lossy,
                    'scale': moderate_scale,
                    'optimize': strategy['optimize'],
                    'resize_method': strategy['resize_method'],
                    'name': 'moderate'
                })
            
            if 1.35 < over_ratio <= 1.50:
                mod_agg_colors = max(128, required_params['colors'] - 32)
                mod_agg_lossy = min(120, required_params['lossy'] + 20)
                mod_agg_scale = required_params['width'] / float(current_params['width']) * 0.90
                if is_gif_only:
                    mod_agg_colors = max(mod_agg_colors, min_colors)
                    mod_agg_lossy = min(mod_agg_lossy, max_lossy)
                    mod_agg_scale = max(mod_agg_scale, min_scale)
                param_combinations.append({
                    'colors': mod_agg_colors,
                    'lossy': mod_agg_lossy,
                    'scale': mod_agg_scale,
                    'optimize': strategy['optimize'],
                    'resize_method': strategy['resize_method'],
                    'name': 'moderate-aggressive'
                })
            
            if current_size_mb > target_size_mb * 2.0:
                agg_colors = max(128, required_params['colors'] - 48)
                agg_lossy = min(120, required_params['lossy'] + 30)
                agg_scale = max(0.75, required_params['width'] / float(current_params['width']) * 0.85)
                if is_gif_only:
                    agg_colors = max(agg_colors, min_colors)
                    agg_lossy = min(agg_lossy, max_lossy)
                    agg_scale = max(agg_scale, min_scale)
                param_combinations.append({
                    'colors': agg_colors,
                    'lossy': agg_lossy,
                    'scale': agg_scale,
                    'optimize': strategy['optimize'],
                    'resize_method': strategy['resize_method'],
                    'name': 'aggressive (for large files)'
                })
        
        logger.info(f"Trying {len(param_combinations)} compression parameter combination(s)...")
        
        best_result = None
        best_size = current_bytes
        temp_files_to_cleanup = []
        timeout_failures = 0
        
        try:
            for i, params in enumerate(param_combinations):
                if self._shutdown_checker():
                    break
                
                param_name = params.get('name', f'combination {i+1}')
                logger.info(f"  Attempt {i+1}/{len(param_combinations)} ({param_name}): colors={params['colors']}, lossy={params['lossy']}, scale={params['scale']:.2f}")
                
                temp_output = create_unique_temp_filename(f"gifsicle_lossy_{i}", ".gif", self.temp_dir)
                temp_files_to_cleanup.append(temp_output)
                
                run_success = self._run_gifsicle(
                    gif_path, temp_output,
                    colors=params.get('colors'),
                    lossy=params.get('lossy'),
                    scale=params.get('scale'),
                    optimize=params.get('optimize'),
                    resize_method=params.get('resize_method')
                )
                
                if run_success:
                    timeout_failures = 0
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
                        
                        # Early exit if we're very close to target (within 5%)
                        # This prevents over-optimization and preserves quality
                        if temp_size <= target_bytes * 1.05:
                            if temp_size < best_size:
                                best_size = temp_size
                                best_result = temp_output
                            logger.info(f"  Result is within 5% of target, stopping early to preserve quality")
                            break
                        
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
                    if self.last_gifsicle_timeout:
                        timeout_failures += 1
                        logger.warning(f"  Result {i+1}: gifsicle timed out (consecutive timeouts: {timeout_failures})")
                        if timeout_failures >= 2:
                            logger.warning("  Aborting gifsicle lossy search after repeated timeouts")
                            break
                    else:
                        timeout_failures = 0
                    logger.warning(f"  Result {i+1}: gifsicle compression failed")
            
            # Use best result if available and better than current
            if best_result and os.path.exists(best_result) and best_size < current_bytes:
                try:
                    # Verify file replacement will work
                    if not os.path.exists(gif_path):
                        logger.warning(f"Stage 3: Target file does not exist for replacement: {gif_path}")
                        return False
                    
                    safe_file_operation(os.replace, best_result, gif_path)
                    
                    # Verify replacement succeeded
                    if os.path.exists(gif_path):
                        actual_size = os.path.getsize(gif_path)
                        if abs(actual_size - best_size) > 1024:  # Allow 1KB tolerance
                            logger.warning(f"Stage 3: File replacement size mismatch: expected {best_size}, got {actual_size}")
                    
                    best_size_mb = best_size / 1024 / 1024
                    reduction_mb = (current_bytes - best_size) / 1024 / 1024
                    reduction_pct = ((current_bytes - best_size) / current_bytes * 100) if current_bytes > 0 else 0
                    logger.info(f"Applied best gifsicle lossy compression: {best_size_mb:.2f}MB (reduced by {reduction_mb:.2f}MB, {reduction_pct:.1f}%)")
                    logger.debug(f"Stage 3: File replaced at {gif_path}, new size: {best_size_mb:.2f}MB")
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
    
    def _progressive_resolution_reduction(self, gif_path: str, target_bytes: int, target_size_mb: float,
                                         source_video: Optional[str] = None) -> bool:
        """
        Progressive resolution reduction: try multiple resolution steps to find optimal size.
        
        Returns:
            True if a better result was found and applied
        """
        try:
            if self._shutdown_checker():
                return False
            
            current_bytes = os.path.getsize(gif_path)
            current_size_mb = current_bytes / 1024 / 1024
            
            if current_size_mb <= target_size_mb:
                return True
            
            # Only run progressive reduction when the file is still significantly over target.
            over_ratio = (current_bytes - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
            progressive_threshold = 0.25  # 25% over target
            if over_ratio < progressive_threshold:
                logger.debug(
                    "Progressive resolution reduction skipped: over_ratio=%.2f%% < %.0f%% threshold",
                    over_ratio * 100,
                    progressive_threshold * 100,
                )
                return False
            
            # Get current GIF info
            try:
                gif_info = get_gif_info(gif_path)
            except Exception:
                return False
            
            current_width = gif_info.get('width', 360)
            current_fps = gif_info.get('fps', 20)
            current_colors = 256
            
            # Calculate progressive resolution steps
            # Try: 100%, 90%, 80%, 70%, 60%, 50% of current width
            resolution_steps = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
            quality_floors = self.config_helper.get_quality_floors()
            min_width = quality_floors.get('min_width_aggressive', 360)
            
            best_result = None
            best_size = current_bytes
            temp_files_to_cleanup = []
            
            try:
                for scale in resolution_steps:
                    if self._shutdown_checker():
                        break
                    
                    new_width = max(min_width, int(current_width * scale))
                    new_width = (new_width // 2) * 2  # Ensure even
                    
                    if new_width >= current_width:
                        continue  # Skip if not reducing
                    
                    # Calculate corresponding FPS reduction to maintain quality
                    # More aggressive resolution = slightly more aggressive FPS
                    fps_scale = 0.95 if scale < 0.7 else 0.98
                    new_fps = max(6, int(current_fps * fps_scale))
                    
                    # Create unique temp file for this iteration
                    temp_output = create_unique_temp_filename("progressive_res", ".gif", self.temp_dir)
                    temp_files_to_cleanup.append(temp_output)
                    
                    # Try this resolution
                    # Use improved dithering based on color count
                    if current_colors >= 192:
                        dither = 'floyd_steinberg'
                    elif current_colors >= 128:
                        dither = 'sierra2_4a'
                    else:
                        dither = 'bayer'
                    
                    success = self._ffmpeg_palette_reencode(
                        gif_path, temp_output,
                        new_width=new_width,
                        fps=new_fps,
                        max_colors=current_colors,
                        mpdecimate_frac=0.4,
                        stats_mode='diff',
                        dither=dither
                    )
                    
                    if success and os.path.exists(temp_output):
                        temp_size = os.path.getsize(temp_output)
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
                            # If we've met target, stop early
                            if temp_size <= target_bytes:
                                break
                
                # Apply best result
                if best_result and os.path.exists(best_result) and best_size < current_bytes:
                    try:
                        safe_file_operation(os.replace, best_result, gif_path)
                        best_size_mb = best_size / 1024 / 1024
                        reduction_mb = (current_bytes - best_size) / 1024 / 1024
                        logger.info(
                            f"Progressive resolution reduction applied: {best_size_mb:.2f}MB "
                            f"(reduced by {reduction_mb:.2f}MB)"
                        )
                        return True
                    except Exception as e:
                        logger.debug(f"Failed to apply progressive resolution result: {e}")
            finally:
                # Cleanup remaining temp files
                for temp_file in temp_files_to_cleanup:
                    if temp_file != best_result:
                        try:
                            if os.path.exists(temp_file):
                                os.remove(temp_file)
                        except Exception:
                            pass
            
            return False
        except Exception as e:
            logger.debug(f"Progressive resolution reduction exception: {e}")
            return False
    
    def _stage_ffmpeg_fallback_compression(self, gif_path: str, target_bytes: int, target_size_mb: float,
                                          gifsicle_just_applied: bool = False) -> bool:
        """
        Stage 3b: FFmpeg fallback compression when gifsicle fails or is unavailable.
        Uses aggressive re-encoding with calculated parameters to meet target size.
        
        Args:
            gif_path: Path to GIF file
            target_bytes: Target size in bytes
            target_size_mb: Target size in MB
            gifsicle_just_applied: True if gifsicle lossy compression was just applied
        
        Returns:
            True if compression was applied and file size reduced
        """
        try:
            if self._shutdown_checker():
                return False
            
            current_bytes = os.path.getsize(gif_path)
            current_size_mb = current_bytes / 1024 / 1024
            
            if current_size_mb <= target_size_mb:
                return True
            
            # Skip FFmpeg fallback if gifsicle was just applied and result is close to target
            # Re-encoding from an already-optimized GIF often increases file size
            if gifsicle_just_applied:
                over_ratio = (current_bytes - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
                if over_ratio < 0.20:  # Less than 20% over target
                    logger.info(
                        "Stage 3c: Skipping FFmpeg fallback - GIF was just optimized by gifsicle "
                        "and is close to target (%.2fMB, %.1f%% over). "
                        "Re-encoding from already-optimized GIF would likely increase file size.",
                        current_size_mb,
                        over_ratio * 100
                    )
                    return False
            
            # Get GIF info to calculate parameters
            try:
                gif_info = get_gif_info(gif_path)
            except Exception as e:
                logger.debug(f"Could not get GIF info for FFmpeg fallback: {e}")
                return False
            
            # Calculate required parameters for aggressive compression
            current_params = {
                'width': gif_info.get('width', 360),
                'height': gif_info.get('height', 360),
                'fps': gif_info.get('fps', 20),
                'colors': 256,
                'lossy': 0
            }
            
            # Use motion analysis for adaptive FPS reduction
            original_fps = current_params['fps']
            try:
                motion_analysis = analyze_motion(gif_path, original_fps, self.temp_dir)
                motion_level = motion_analysis.get('motion_level', 'medium')
                optimal_fps = motion_analysis.get('optimal_fps', original_fps)
                
                # Adjust FPS based on motion level for better compression
                # Low motion = can reduce FPS more aggressively
                if motion_level == 'low':
                    # Static scenes: reduce FPS more
                    fps_adjustment = 0.7
                elif motion_level == 'medium':
                    # Moderate motion: moderate FPS reduction
                    fps_adjustment = 0.85
                else:
                    # High motion: minimal FPS reduction to preserve quality
                    fps_adjustment = 0.95
                
                # Apply motion-aware FPS adjustment
                current_params['fps'] = max(6, int(original_fps * fps_adjustment))
                logger.debug(f"Motion-aware FPS adjustment: {original_fps} -> {current_params['fps']} (motion: {motion_level})")
            except Exception as e:
                logger.debug(f"Motion analysis failed, using default FPS: {e}")
            
            required_params = self._calculate_required_parameters(
                current_size_mb, target_size_mb, current_params, gif_info, gif_path=gif_path
            )
            
            # Use less aggressive mpdecimate to preserve quality and reduce jerkiness
            # Reduced maximum from 0.5 to 0.3 to maintain smoother motion
            mpdecimate_frac = 0.25  # Gentler duplicate removal
            if current_size_mb / target_size_mb > 3.0:
                mpdecimate_frac = 0.3  # Still relatively gentle even for very large files
            
            # Use improved dithering based on color count for better quality
            colors = required_params['colors']
            if colors >= 192:
                dither = 'floyd_steinberg'  # Best for many colors
            elif colors >= 128:
                dither = 'sierra2_4a'  # Good balance
            else:
                dither = 'bayer'  # Simpler for few colors
            
            # Create temp output
            with temp_file_context("ffmpeg_fallback", ".gif", self.temp_dir) as temp_output:
                success = self._ffmpeg_palette_reencode(
                    gif_path, temp_output,
                    new_width=required_params['width'],
                    fps=required_params['fps'],
                    max_colors=required_params['colors'],
                    mpdecimate_frac=mpdecimate_frac,
                    stats_mode='diff',
                    dither=dither
                )
                
                if success and os.path.exists(temp_output):
                    temp_size = os.path.getsize(temp_output)
                    temp_size_mb = temp_size / 1024 / 1024
                    
                    # Only apply if it's better than current
                    if temp_size < current_bytes:
                        try:
                            safe_file_operation(os.replace, temp_output, gif_path)
                            reduction_mb = (current_bytes - temp_size) / 1024 / 1024
                            reduction_pct = ((current_bytes - temp_size) / current_bytes * 100) if current_bytes > 0 else 0
                            logger.info(
                                f"FFmpeg fallback compression applied: {temp_size_mb:.2f}MB "
                                f"(reduced by {reduction_mb:.2f}MB, {reduction_pct:.1f}%)"
                            )
                            return True
                        except Exception as e:
                            logger.warning(f"Failed to apply FFmpeg fallback result: {e}")
                    else:
                        logger.debug(f"FFmpeg fallback result ({temp_size_mb:.2f}MB) not better than current ({current_size_mb:.2f}MB)")
            
            return False
        except Exception as e:
            logger.debug(f"FFmpeg fallback compression exception: {e}")
            return False
    
    def _stage_pil_compression(self, gif_path: str, target_bytes: int, target_size_mb: float) -> bool:
        """
        Stage 3c: PIL-based compression using adaptive palette quantization and frame optimization.
        
        Returns:
            True if compression was applied and file size reduced
        """
        try:
            if self._shutdown_checker():
                return False
            
            current_bytes = os.path.getsize(gif_path)
            current_size_mb = current_bytes / 1024 / 1024
            
            if current_size_mb <= target_size_mb:
                return True
            
            # Calculate how much over target the file is
            over_ratio = (current_bytes - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
            
            # Only enable duplicate removal when file is significantly over target (>50% over)
            # This prevents framerate degradation for files that are only moderately over target
            remove_duplicates = over_ratio > 0.5
            
            # Calculate target colors based on size reduction needed
            size_ratio = target_size_mb / current_size_mb
            # Estimate colors needed: more aggressive reduction = fewer colors
            if size_ratio < 0.3:
                target_colors = 64  # Very aggressive
            elif size_ratio < 0.5:
                target_colors = 128  # Aggressive
            elif size_ratio < 0.7:
                target_colors = 192  # Moderate
            else:
                target_colors = 224  # Conservative
            
            # Try PIL compression with calculated colors
            with temp_file_context("pil_compress", ".gif", self.temp_dir) as temp_output:
                success, reduction_pct = PILCompressor.compress_gif(
                    gif_path, temp_output,
                    max_colors=target_colors,
                    remove_duplicates=remove_duplicates,
                    optimize_disposal=True
                )
                
                if success and os.path.exists(temp_output):
                    temp_size = os.path.getsize(temp_output)
                    temp_size_mb = temp_size / 1024 / 1024
                    
                    # Only apply if it's better than current
                    if temp_size < current_bytes:
                        try:
                            safe_file_operation(os.replace, temp_output, gif_path)
                            reduction_mb = (current_bytes - temp_size) / 1024 / 1024
                            logger.info(
                                f"PIL compression applied: {temp_size_mb:.2f}MB "
                                f"(reduced by {reduction_mb:.2f}MB, {reduction_pct:.1f}%)"
                            )
                            return True
                        except Exception as e:
                            logger.warning(f"Failed to apply PIL compression result: {e}")
                    else:
                        logger.debug(f"PIL compression result ({temp_size_mb:.2f}MB) not better than current ({current_size_mb:.2f}MB)")
            
            return False
        except Exception as e:
            logger.debug(f"PIL compression exception: {e}")
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
            try:
                timeout_sec = min(timeout_sec, int(max(1, self.stage_time_budgets.get('lossless', timeout_sec))))
            except Exception:
                timeout_sec = min(timeout_sec, 30 if self.fast_mode else 60)
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
        """Bounded gifsicle search for near-target cases - optimized with adaptive search"""
        try:
            # Get configuration
            near_target_config = self.config_helper.get_optimization_config().get('near_target', {})
            max_runs = near_target_config.get('max_attempts', self.near_target_max_runs)
            fine_tune_threshold = near_target_config.get('fine_tune_threshold_percent', 10) / 100.0
            
            # Verify input file exists and log details
            if not os.path.exists(input_path):
                logger.warning(f"Bounded search: Input file does not exist: {input_path}")
                return False
            
            current_size = os.path.getsize(input_path)
            current_size_mb = current_size / (1024 * 1024)
            target_mb = target_bytes / (1024 * 1024)
            over_ratio = (current_size - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
            
            logger.debug(f"Bounded search: Input file {os.path.basename(input_path)} is {current_size_mb:.2f}MB (target: {target_mb:.2f}MB, {over_ratio*100:.1f}% over)")
            
            original_size = current_size
            best: Tuple[int, str] = (original_size, '')
            temp_files_to_cleanup = []
            runs = 0
            
            # Track results to avoid redundant tests: (scale, lossy) -> size
            # When lossy > 0, colors have no effect, so we only track scale+lossy
            result_cache: Dict[Tuple[float, int], int] = {}
            
            try:
                # Skip color-only attempts when we're close to target (≤15% over) since:
                # 1. We'll need lossy compression anyway to get under target
                # 2. When lossy > 0, colors have no effect
                # 3. Color-only attempts on already-optimized files often produce identical results
                skip_color_only = over_ratio <= 0.15
                
                if not skip_color_only:
                    # Quick color-only test (max 2 attempts to detect redundancy) - only for files far from target
                    color_only_sizes = []
                    for colors in [256, 224]:  # Only test 2 colors to detect redundancy
                        if self._shutdown_checker() or runs >= max_runs:
                            break
                        
                        runs += 1
                        temp = create_unique_temp_filename(f"near_c{colors}", ".gif", self.temp_dir)
                        temp_files_to_cleanup.append(temp)
                        
                        if self._run_gifsicle(input_path, temp, colors=colors, lossy=0, scale=1.0):
                            size = os.path.getsize(temp)
                            size_mb = size / (1024 * 1024)
                            
                            status = "✓ under target" if size < target_bytes else f"{((size - target_bytes) / target_bytes * 100):.1f}% over"
                            logger.debug(f"Attempt {runs}: colors={colors}, lossy=0, scale=1.0 → {size_mb:.2f}MB ({status}, target: {target_mb:.2f}MB)")
                            
                            if size < target_bytes:
                                safe_file_operation(shutil.copy2, temp, best_output_path)
                                for tf in temp_files_to_cleanup:
                                    try:
                                        if os.path.exists(tf):
                                            os.remove(tf)
                                    except Exception:
                                        pass
                                logger.debug(f"Bounded search succeeded in color-only phase after {runs} attempts")
                                return True
                            
                            if size < best[0]:
                                if best[1] and best[1] in temp_files_to_cleanup:
                                    temp_files_to_cleanup.remove(best[1])
                                    try:
                                        if os.path.exists(best[1]):
                                            os.remove(best[1])
                                    except Exception:
                                        pass
                                best = (size, temp)
                            
                            color_only_sizes.append(size)
                            if len(color_only_sizes) >= 2 and color_only_sizes[0] == color_only_sizes[1]:
                                logger.debug("Color-only attempts producing identical sizes, skipping remaining")
                                break
                else:
                    logger.debug(f"Skipping color-only attempts (file is {over_ratio*100:.1f}% over target, will use lossy compression)")
                
                # Adaptive search: Generate combinations ordered by effectiveness
                # Key insight: When lossy > 0, colors have no effect, so skip color variations
                # When starting from an already-optimized file (close to target), use conservative adjustments
                if 0 < over_ratio <= fine_tune_threshold:
                    # Very close (≤10%): Use targeted adjustments to get just under target
                    # Since we're starting from an already-optimized file, we need small but effective changes
                    # Try slightly more aggressive lossy first, then adjust scale if needed
                    lossy_steps = [70, 75, 65, 80, 60]  # Higher lossy values to reduce size
                    scale_steps = [0.88, 0.89, 0.87, 0.90, 0.86]  # Small scale reductions
                    max_runs = min(max_runs, 10)  # Reduce max attempts for fine-grained
                    logger.info(f"Using targeted fine-grained search ({over_ratio*100:.1f}% over target, starting from optimized file)")
                elif 0 < over_ratio <= 0.15:
                    # Close (10-15%): Moderate adjustments
                    lossy_steps = [65, 70, 60, 75, 55]
                    scale_steps = [0.89, 0.90, 0.88, 0.91, 0.87, 0.92]
                    logger.info(f"Using moderate search ({over_ratio*100:.1f}% over target)")
                elif 0 < over_ratio <= 0.25:
                    # Moderately over (15-25%): More aggressive settings
                    lossy_steps = [70, 75, 65, 80, 60]
                    scale_steps = [0.88, 0.87, 0.89, 0.86, 0.90, 0.85]
                    logger.info(f"Using aggressive search ({over_ratio*100:.1f}% over target)")
                else:
                    # Way over (>25%): Most aggressive settings
                    lossy_steps = [75, 80, 70, 85, 65]
                    scale_steps = [0.85, 0.84, 0.86, 0.83, 0.87, 0.82]
                    logger.info(f"Using most aggressive search ({over_ratio*100:.1f}% over target)")
                
                # Generate combinations ordered by effectiveness (most aggressive first)
                # Skip color variations when lossy > 0 (colors have no effect)
                combinations = []
                for scale in scale_steps:
                    for lossy in lossy_steps:
                        # Check cache first
                        cache_key = (scale, lossy)
                        if cache_key in result_cache:
                            cached_size = result_cache[cache_key]
                            if cached_size < target_bytes:
                                # Found a cached result that meets target
                                logger.debug(f"Cache hit: scale={scale:.2f}, lossy={lossy} → {cached_size/(1024*1024):.2f}MB")
                                # We'd need to regenerate, but for now just skip
                                continue
                        # Only test one color value when lossy > 0 (256 is default, others produce same size)
                        combinations.append((scale, 256, lossy))
                
                # Binary search refinement for fine-tuning near target
                def binary_search_refine(best_scale: float, best_lossy: int, best_size: int) -> Optional[Tuple[float, int, str]]:
                    """Binary search to fine-tune parameters near target"""
                    if best_size >= target_bytes * 1.05:  # Only refine if >5% over
                        return None
                    
                    # Binary search on scale
                    scale_low = best_scale * 0.98
                    scale_high = best_scale * 1.02
                    scale_mid = best_scale
                    
                    # Binary search on lossy
                    lossy_low = max(0, best_lossy - 5)
                    lossy_high = min(200, best_lossy + 5)
                    lossy_mid = best_lossy
                    
                    best_refined = None
                    best_refined_size = best_size
                    
                    for _ in range(3):  # Max 3 binary search iterations
                        if self._shutdown_checker() or runs >= max_runs:
                            break
                        
                        # Test midpoint
                        temp_refine = create_unique_temp_filename(f"bs_refine_s{scale_mid:.3f}_l{lossy_mid}", ".gif", self.temp_dir)
                        temp_files_to_cleanup.append(temp_refine)
                        runs += 1
                        
                        if self._run_gifsicle(input_path, temp_refine, lossy=lossy_mid, scale=scale_mid):
                            size_refine = os.path.getsize(temp_refine)
                            result_cache[(scale_mid, lossy_mid)] = size_refine
                            
                            if size_refine < best_refined_size:
                                if best_refined:
                                    try:
                                        os.remove(best_refined)
                                    except:
                                        pass
                                best_refined = temp_refine
                                best_refined_size = size_refine
                            
                            if size_refine <= target_bytes:
                                return (scale_mid, lossy_mid, temp_refine)
                            
                            # Adjust search range
                            if size_refine > target_bytes:
                                # Need more compression
                                scale_low = scale_mid
                                lossy_low = lossy_mid
                            else:
                                # Can use less compression
                                scale_high = scale_mid
                                lossy_high = lossy_mid
                            
                            scale_mid = (scale_low + scale_high) / 2
                            lossy_mid = (lossy_low + lossy_high) // 2
                    
                    if best_refined and best_refined_size < best_size:
                        return (scale_mid, lossy_mid, best_refined)
                    return None
                
                # Limit to remaining attempts
                remaining_attempts = max_runs - runs
                if remaining_attempts > 0:
                    combinations = combinations[:remaining_attempts]
                
                if combinations:
                    from concurrent.futures import ThreadPoolExecutor, as_completed
                    
                    # Use fewer workers for better early stopping
                    max_workers = min(3, len(combinations), max(2, (os.cpu_count() or 4) // 2))
                    logger.debug(f"Running {len(combinations)} optimized parameter combinations with {max_workers} parallel workers")
                    
                    indexed_combinations = [(i, params) for i, params in enumerate(combinations)]
                    found_solution = False
                    executor_shutdown = False
                    
                    def run_combination(indexed_item):
                        """Run a single gifsicle combination and return result"""
                        if executor_shutdown or self._shutdown_checker():
                            return None
                        
                        idx, (scale, colors, lossy) = indexed_item
                        attempt_num = runs + idx + 1
                        
                        # Check cache: if we've tested this scale+lossy combo, skip
                        cache_key = (scale, lossy)
                        if cache_key in result_cache:
                            return {
                                'success': True,
                                'size': result_cache[cache_key],
                                'temp': None,  # Cached result, no temp file
                                'scale': scale,
                                'colors': colors,
                                'lossy': lossy,
                                'attempt': attempt_num,
                                'cached': True
                            }
                        
                        temp = create_unique_temp_filename(f"near_s{int(scale*100)}_l{lossy}", ".gif", self.temp_dir)
                        
                        if self._run_gifsicle(input_path, temp, colors=colors, lossy=lossy, scale=scale):
                            if os.path.exists(temp):
                                size = os.path.getsize(temp)
                                result_cache[cache_key] = size  # Cache result
                                return {
                                    'success': True,
                                    'size': size,
                                    'temp': temp,
                                    'scale': scale,
                                    'colors': colors,
                                    'lossy': lossy,
                                    'attempt': attempt_num,
                                    'cached': False
                                }
                        return {
                            'success': False,
                            'temp': temp,
                            'scale': scale,
                            'colors': colors,
                            'lossy': lossy,
                            'attempt': attempt_num,
                            'cached': False
                        }
                    
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        # Submit tasks in batches for better early stopping
                        future_to_params = {}
                        submitted = 0
                        batch_size = max_workers * 2  # Submit 2 batches ahead
                        
                        # Process results as they complete
                        while submitted < len(indexed_combinations) or future_to_params:
                            # Submit next batch if not all submitted and solution not found
                            if not found_solution and submitted < len(indexed_combinations):
                                for i in range(submitted, min(submitted + batch_size, len(indexed_combinations))):
                                    if self._shutdown_checker():
                                        break
                                    item = indexed_combinations[i]
                                    future = executor.submit(run_combination, item)
                                    future_to_params[future] = item
                                    submitted += 1
                            
                            if not future_to_params:
                                break
                            
                            # Wait for at least one to complete
                            for future in as_completed(future_to_params):
                                if found_solution or self._shutdown_checker():
                                    executor_shutdown = True
                                    # Cancel all remaining
                                    for f in list(future_to_params.keys()):
                                        f.cancel()
                                    future_to_params.clear()
                                    break
                                
                                result = future.result()
                                if result is None:
                                    future_to_params.pop(future, None)
                                    continue
                                
                                # Only process if not cached (cached results are for reference only)
                                if not result.get('cached', False):
                                    if result['temp']:
                                        temp_files_to_cleanup.append(result['temp'])
                                    runs += 1
                                
                                if result['success']:
                                    size = result['size']
                                    size_mb = size / (1024 * 1024)
                                    target_mb = target_bytes / (1024 * 1024)
                                    
                                    # Only extract metadata for final result or when logging important milestones
                                    metadata_str = ""
                                    if size < target_bytes or (size < best[0] and size < target_bytes * 1.1):
                                        try:
                                            temp_path = result['temp'] if result['temp'] else None
                                            if temp_path and os.path.exists(temp_path):
                                                gif_info = get_gif_info(temp_path)
                                                width = gif_info.get('width', 0)
                                                height = gif_info.get('height', 0)
                                                frame_count = gif_info.get('frame_count', 0)
                                                fps = gif_info.get('fps', 0.0)
                                                duration = gif_info.get('duration', 0.0)
                                                metadata_str = f", {width}x{height}, {frame_count} frames, {fps:.1f}fps, {duration:.2f}s"
                                        except Exception:
                                            pass
                                    
                                    status = "✓ under target" if size < target_bytes else f"{((size - target_bytes) / target_bytes * 100):.1f}% over"
                                    logger.debug(
                                        f"Attempt {result['attempt']}: scale={result['scale']:.2f}, lossy={result['lossy']} → "
                                        f"{size_mb:.2f}MB ({status}, target: {target_mb:.2f}MB){metadata_str}"
                                    )
                                    
                                    # STRICT: Only accept results strictly under target
                                    if size < target_bytes:
                                        if result['temp'] and os.path.exists(result['temp']):
                                            safe_file_operation(shutil.copy2, result['temp'], best_output_path)
                                        # Cancel all remaining tasks immediately
                                        executor_shutdown = True
                                        for f in list(future_to_params.keys()):
                                            f.cancel()
                                        future_to_params.clear()
                                        found_solution = True
                                        logger.debug(f"Bounded search succeeded after {result['attempt']} attempts")
                                        break
                                    
                                    if size < best[0] and result['temp']:
                                        # Remove previous best from cleanup list
                                        if best[1] and best[1] in temp_files_to_cleanup:
                                            temp_files_to_cleanup.remove(best[1])
                                            try:
                                                if os.path.exists(best[1]):
                                                    os.remove(best[1])
                                            except Exception:
                                                pass
                                        best = (size, result['temp'])
                                
                                future_to_params.pop(future, None)
                                
                                if found_solution:
                                    break
                            
                            if found_solution:
                                break
                    
                    if found_solution:
                        # Cleanup all temp files
                        for tf in temp_files_to_cleanup:
                            try:
                                if os.path.exists(tf):
                                    os.remove(tf)
                            except Exception:
                                pass
                        return True
                
                # Binary search refinement if we're close to target
                if best[0] < original_size and best[0] > target_bytes and best[0] <= target_bytes * 1.1:
                    # Extract best parameters from the result
                    # We need to find which combination produced the best result
                    # For now, use a reasonable estimate based on over_ratio
                    best_scale_estimate = 0.90 if over_ratio > 0.15 else 0.95
                    best_lossy_estimate = 70 if over_ratio > 0.15 else 60
                    
                    logger.debug(f"Attempting binary search refinement (best: {best[0]/(1024*1024):.2f}MB, target: {target_bytes/(1024*1024):.2f}MB)")
                    refine_result = binary_search_refine(best_scale_estimate, best_lossy_estimate, best[0])
                    if refine_result:
                        refined_scale, refined_lossy, refined_temp = refine_result
                        refined_size = os.path.getsize(refined_temp)
                        if refined_size < best[0]:
                            # Remove old best
                            if best[1] and best[1] in temp_files_to_cleanup:
                                temp_files_to_cleanup.remove(best[1])
                                try:
                                    if os.path.exists(best[1]):
                                        os.remove(best[1])
                                except:
                                    pass
                            best = (refined_size, refined_temp)
                            temp_files_to_cleanup.append(refined_temp)
                            logger.debug(f"Binary search refinement improved result to {refined_size/(1024*1024):.2f}MB")
                
                # NO TOLERANCE: Only return best if it's strictly under target
                if best[1] and os.path.exists(best[1]):
                    if best[0] < target_bytes:
                        safe_file_operation(shutil.copy2, best[1], best_output_path)
                        # Extract metadata only for final result
                        try:
                            gif_info = get_gif_info(best[1])
                            width = gif_info.get('width', 0)
                            height = gif_info.get('height', 0)
                            frame_count = gif_info.get('frame_count', 0)
                            fps = gif_info.get('fps', 0.0)
                            duration = gif_info.get('duration', 0.0)
                            logger.debug(f"Bounded search found acceptable result ({best[0] / (1024*1024):.2f}MB) after {runs} attempts, {width}x{height}, {frame_count} frames, {fps:.1f}fps, {duration:.2f}s")
                        except Exception:
                            logger.debug(f"Bounded search found acceptable result ({best[0] / (1024*1024):.2f}MB) after {runs} attempts")
                        # Cleanup all temp files
                        for tf in temp_files_to_cleanup:
                            try:
                                if os.path.exists(tf):
                                    os.remove(tf)
                            except Exception:
                                pass
                        return True
                
                # Log if we exhausted attempts without success
                if runs >= max_runs:
                    logger.debug(f"Bounded search exhausted {max_runs} attempts, best result: {best[0] / (1024*1024):.2f}MB (target: {target_bytes / (1024*1024):.2f}MB)")
                
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
    
    def _run_gifsicle(self, input_path: str, output_path: str, colors: int = None, lossy: int = None, 
                     scale: float = None, optimize: int = None, resize_method: str = None,
                     resize_colors: bool = None, resize_dither: bool = None, interlace: bool = None,
                     optimize_disposal: bool = None, optimize_background: bool = None) -> bool:
        """
        Execute gifsicle with given parameters.
        
        Args:
            input_path: Input GIF file path
            output_path: Output GIF file path
            colors: Number of colors (2-256)
            lossy: Lossy compression level (0-200)
            scale: Scale factor (0.0-1.0)
            optimize: Optimization level (1-3)
            resize_method: Resize method (sample, mix, scale, lanczos3)
            resize_colors: Enable color reduction during resize
            resize_dither: Enable dithering during resize
            interlace: Enable interlacing
            optimize_disposal: Optimize frame disposal methods
            optimize_background: Optimize background color
        """
        timeout_sec = 45 if self.fast_mode else 120
        try:
            gifsicle_budget = int(max(1, self.stage_time_budgets.get('gifsicle', timeout_sec)))
            timeout_sec = min(timeout_sec, gifsicle_budget)
        except Exception:
            timeout_sec = 45 if self.fast_mode else 120
        try:
            # Track timeout status so callers can react (e.g., bail out after repeated timeouts)
            self.last_gifsicle_timeout = False
            # Verify input file exists
            if not os.path.exists(input_path):
                logger.warning(f"gifsicle: Input file does not exist: {input_path}")
                return False
            
            # Check gifsicle availability
            if not self._is_tool_available("gifsicle"):
                logger.warning("gifsicle: Tool not available or not working")
                return False
            
            # Get defaults from config
            default_optimize = optimize if optimize is not None else max(1, min(3, self.gifsicle_optimize_level))
            default_resize_method = resize_method if resize_method is not None else self.advanced_options.get('resize_method', 'mix')
            default_resize_colors = resize_colors if resize_colors is not None else self.advanced_options.get('resize_colors', True)
            default_resize_dither = resize_dither if resize_dither is not None else self.advanced_options.get('resize_dither', True)
            default_interlace = interlace if interlace is not None else self.advanced_options.get('interlace', False)
            default_optimize_disposal = optimize_disposal if optimize_disposal is not None else self.advanced_options.get('optimize_disposal', True)
            default_optimize_background = optimize_background if optimize_background is not None else self.advanced_options.get('optimize_background', True)
            
            cmd = [
                "gifsicle",
                f"--optimize={default_optimize}",
                "--careful",
                "--no-comments",
                "--no-extensions",
                "--no-names",
            ]
            
            # Scale with optional resize method
            if scale and abs(scale - 1.0) > 1e-6:
                if default_resize_method and default_resize_method in ['sample', 'mix', 'scale', 'lanczos3']:
                    cmd.extend(["--resize-method", default_resize_method])
                # --resize-colors requires an integer argument (number of colors)
                # Use the colors parameter if available, otherwise skip this option
                if default_resize_colors and colors and colors < 256:
                    cmd.extend(["--resize-colors", str(max(2, min(256, colors)))])
                # Note: --resize-dither is not a valid gifsicle option, removed
                cmd.extend(["--scale", f"{scale:.4f}"])
            
            # Color reduction
            if colors and colors < 256:
                cmd.extend(["--colors", str(max(2, min(256, colors)))])
            
            # Lossy compression
            if lossy and lossy > 0:
                lossy_value = max(1, min(200, int(lossy)))
                cmd.append(f"--lossy={lossy_value}")
            
            # Interlacing
            if default_interlace:
                cmd.append("--interlace")
            
            # Note: Frame disposal and background optimization are handled by --optimize
            # Additional disposal/background flags would require more complex gifsicle usage
            
            cmd.extend([input_path, "--output", output_path])
            
            logger.debug(f"gifsicle command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=timeout_sec
            )
            
            # Detailed error logging
            if result.returncode != 0:
                stderr_preview = result.stderr[:500] if result.stderr else "No stderr output"
                stdout_preview = result.stdout[:500] if result.stdout else "No stdout output"
                logger.warning(
                    f"gifsicle failed with return code {result.returncode}\n"
                    f"  stderr: {stderr_preview}\n"
                    f"  stdout: {stdout_preview}"
                )
                return False
            
            if not os.path.exists(output_path):
                logger.warning(f"gifsicle: Output file was not created: {output_path}")
                return False
            
            # Verify output file is valid and not empty
            output_size = os.path.getsize(output_path)
            if output_size == 0:
                logger.warning(f"gifsicle: Output file is empty: {output_path}")
                return False
            
            return True
        except subprocess.TimeoutExpired:
            self.last_gifsicle_timeout = True
            logger.warning(f"gifsicle: Command timed out after {timeout_sec}s")
            return False
        except FileNotFoundError:
            logger.warning("gifsicle: Command not found. Is gifsicle installed?")
            return False
        except Exception as e:
            logger.warning(f"gifsicle run failed: {e}", exc_info=True)
            return False
    
    def _adaptive_gifsicle_strategy(self, current_size_mb: float, target_size_mb: float, 
                                   gif_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select adaptive parameter strategy based on file characteristics.
        
        Args:
            current_size_mb: Current file size in MB
            target_size_mb: Target file size in MB
            gif_info: GIF information dictionary
        
        Returns:
            Dictionary with strategy parameters
        """
        if target_size_mb <= 0:
            target_size_mb = 10.0
        
        over_ratio = (current_size_mb / target_size_mb) if target_size_mb > 0 else 1.0
        file_size_mb = current_size_mb
        
        # Determine strategy based on over-ratio and file size
        if over_ratio <= 1.1:
            # Very close to target - conservative approach
            strategy_name = 'conservative'
        elif over_ratio <= 1.5:
            # Moderate overage - balanced approach
            strategy_name = 'balanced'
        else:
            # Far over target - aggressive approach
            strategy_name = 'aggressive'
        
        # Adjust based on file size
        if file_size_mb < 5.0:
            # Small files - can be more conservative
            if strategy_name == 'aggressive':
                strategy_name = 'balanced'
        elif file_size_mb > 15.0:
            # Large files - may need more aggressive
            if strategy_name == 'conservative':
                strategy_name = 'balanced'
        
        # Get preset from config
        preset = self.parameter_presets.get(strategy_name, self.parameter_presets.get('balanced', {}))
        
        return {
            'strategy': strategy_name,
            'optimize': preset.get('optimize', 2),
            'colors_range': preset.get('colors_range', [192, 160, 128]),
            'lossy_range': preset.get('lossy_range', [40, 60, 80, 100]),
            'scale_range': preset.get('scale_range', [0.95, 0.92, 0.88, 0.85]),
            'resize_method': preset.get('resize_method', 'mix')
        }
    
    def _gifsicle_multi_pass_optimize(self, gif_path: str, target_bytes: int, 
                                     target_size_mb: float) -> Optional[str]:
        """
        Multi-pass optimization pipeline for gifsicle.
        
        Pass 1: Lossless optimization
        Pass 2: Color reduction
        Pass 3: Lossy compression
        Pass 4: Resolution scaling
        Pass 5: Combined optimization
        
        Returns:
            Path to optimized file if successful, None otherwise
        """
        if not self.multi_pass_enabled:
            return None
        
        if not self._is_tool_available("gifsicle"):
            return None
        
        current_bytes = os.path.getsize(gif_path)
        current_size_mb = current_bytes / 1024 / 1024
        gif_info = get_gif_info(gif_path)
        
        temp_files = []
        best_result = None
        best_size = current_bytes
        
        # Cap multi-pass attempts and track failures/regressions
        MAX_MULTI_PASS_ATTEMPTS = 3  # Total attempts across all passes
        attempt_count = 0
        consecutive_failures = 0
        consecutive_regressions = 0
        MAX_CONSECUTIVE_FAILURES = 2  # Break after 2 consecutive failures
        seen_parameter_combos = set()  # Track parameter combinations to avoid repeats
        abort_multi_pass = False
        mp_start = time.monotonic()
        mp_budget = int(max(1, self.stage_time_budgets.get('gifsicle', 10)))

        def _should_stop(reason: str) -> bool:
            if self._shutdown_checker():
                logger.info(f"Multi-pass: Stopping due to shutdown [{reason}]")
                return True
            if time.monotonic() - mp_start > mp_budget:
                logger.warning(f"Multi-pass: Time budget exceeded ({mp_budget}s) [{reason}]")
                return True
            return False

        def _cleanup_and_return() -> Optional[str]:
            for tf in temp_files:
                if tf != best_result:
                    try:
                        if os.path.exists(tf):
                            os.remove(tf)
                    except Exception:
                        pass
            return best_result if best_result and best_size < current_bytes else None
        
        try:
            # Pass 1: Lossless optimization
            logger.debug("Multi-pass: Pass 1 - Lossless optimization")
            if abort_multi_pass or _should_stop("pass1_start") or attempt_count >= MAX_MULTI_PASS_ATTEMPTS:
                abort_multi_pass = True
            else:
                attempt_count += 1
                temp1 = create_unique_temp_filename("mp1_lossless", ".gif", self.temp_dir)
                temp_files.append(temp1)
                if self._run_gifsicle(gif_path, temp1, optimize=3, resize_method='lanczos3'):
                    size1 = os.path.getsize(temp1)
                    consecutive_failures = 0  # Reset on success
                    regressed = size1 > best_size
                    if size1 < best_size:
                        best_size = size1
                        best_result = temp1
                    consecutive_regressions = 0 if size1 <= best_size else consecutive_regressions + int(regressed)
                    if size1 <= target_bytes:
                        logger.info("Multi-pass: Target met in Pass 1 (lossless)")
                        return temp1
                    if consecutive_regressions >= 2:
                        logger.warning("Multi-pass: Stopping after consecutive regressions in Pass 1")
                        abort_multi_pass = True
                else:
                    consecutive_failures += 1
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.warning(f"Multi-pass: {consecutive_failures} consecutive failures, breaking")
                        abort_multi_pass = True
            if abort_multi_pass:
                return _cleanup_and_return()
            
            # Pass 2: Progressive color reduction (reduced from 5 to 3 attempts)
            logger.debug("Multi-pass: Pass 2 - Color reduction")
            input_for_pass2 = best_result if best_result else gif_path
            pass4_met_target = False
            for colors in [256, 192, 128]:  # Reduced from [256, 224, 192, 160, 128] to 3 attempts
                if abort_multi_pass or _should_stop("pass2") or attempt_count >= MAX_MULTI_PASS_ATTEMPTS:
                    abort_multi_pass = True
                    break
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.warning(f"Multi-pass Pass 2: Too many consecutive failures, breaking")
                    break
                # Track parameter combo to avoid repeats
                param_key = ('colors', colors)
                if param_key in seen_parameter_combos:
                    logger.debug(f"Multi-pass Pass 2: Skipping duplicate parameter combo colors={colors}")
                    continue
                attempt_count += 1
                seen_parameter_combos.add(param_key)
                temp2 = create_unique_temp_filename(f"mp2_c{colors}", ".gif", self.temp_dir)
                temp_files.append(temp2)
                if self._run_gifsicle(input_for_pass2, temp2, colors=colors, optimize=3):
                    size2 = os.path.getsize(temp2)
                    consecutive_failures = 0  # Reset on success
                    regressed = size2 > best_size
                    # Check quality if early exit is enabled
                    if self.enable_early_exit:
                        temp_gif_info = get_gif_info(temp2)
                        quality_score = self._calculate_quality_score(temp_gif_info, colors, 0, 1.0)
                        if quality_score < self.min_quality_threshold:
                            logger.warning(f"Multi-pass Pass 2: Rejecting colors={colors}, quality {quality_score:.1f} < threshold {self.min_quality_threshold}")
                            try:
                                os.remove(temp2)
                            except:
                                pass
                            consecutive_failures += 1
                            continue
                    # Break if size worsens significantly (avoid repeating same grid)
                    if size2 > best_size * 1.1:  # >10% worse
                        logger.debug(f"Multi-pass Pass 2: Size worsened ({size2/(1024*1024):.2f}MB > {best_size/(1024*1024):.2f}MB), skipping remaining colors")
                        consecutive_failures += 1
                        regressed = True
                        abort_multi_pass = abort_multi_pass or consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                        break
                    if size2 < best_size:
                        if best_result and best_result != input_for_pass2:
                            try:
                                os.remove(best_result)
                            except:
                                pass
                        best_size = size2
                        best_result = temp2
                    consecutive_regressions = 0 if size2 <= best_size else consecutive_regressions + int(regressed)
                    if consecutive_regressions >= 2:
                        logger.warning("Multi-pass: Stopping after consecutive regressions in Pass 2")
                        abort_multi_pass = True
                    if size2 <= target_bytes:
                        logger.info(f"Multi-pass: Target met in Pass 2 (colors={colors})")
                        return temp2
                else:
                    consecutive_failures += 1
                    abort_multi_pass = abort_multi_pass or consecutive_failures >= MAX_CONSECUTIVE_FAILURES

                if abort_multi_pass:
                    break
            
            if abort_multi_pass:
                return _cleanup_and_return()
            
            # Pass 3: Incremental lossy compression (reduced from 6 to 4 attempts)
            logger.debug("Multi-pass: Pass 3 - Lossy compression")
            input_for_pass3 = best_result if best_result else gif_path
            for lossy in [20, 60, 100, 120]:  # Reduced from [20, 40, 60, 80, 100, 120] to 4 attempts
                if abort_multi_pass or _should_stop("pass3") or attempt_count >= MAX_MULTI_PASS_ATTEMPTS:
                    abort_multi_pass = True
                    break
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.warning(f"Multi-pass Pass 3: Too many consecutive failures, breaking")
                    break
                # Track parameter combo to avoid repeats
                param_key = ('lossy', lossy)
                if param_key in seen_parameter_combos:
                    logger.debug(f"Multi-pass Pass 3: Skipping duplicate parameter combo lossy={lossy}")
                    continue
                attempt_count += 1
                seen_parameter_combos.add(param_key)
                temp3 = create_unique_temp_filename(f"mp3_l{lossy}", ".gif", self.temp_dir)
                temp_files.append(temp3)
                if self._run_gifsicle(input_for_pass3, temp3, lossy=lossy, optimize=2):
                    size3 = os.path.getsize(temp3)
                    consecutive_failures = 0  # Reset on success
                    regressed = size3 > best_size
                    # Check quality if early exit is enabled
                    if self.enable_early_exit:
                        temp_gif_info = get_gif_info(temp3)
                        colors_used = temp_gif_info.get('colors', 256)
                        quality_score = self._calculate_quality_score(temp_gif_info, colors_used, lossy, 1.0)
                        if quality_score < self.min_quality_threshold:
                            logger.warning(f"Multi-pass Pass 3: Rejecting lossy={lossy}, quality {quality_score:.1f} < threshold {self.min_quality_threshold}")
                            try:
                                os.remove(temp3)
                            except:
                                pass
                            consecutive_failures += 1
                            continue
                    # Break if size worsens significantly (avoid repeating same grid)
                    if size3 > best_size * 1.1:  # >10% worse
                        logger.debug(f"Multi-pass Pass 3: Size worsened ({size3/(1024*1024):.2f}MB > {best_size/(1024*1024):.2f}MB), skipping remaining lossy values")
                        consecutive_failures += 1
                        regressed = True
                        abort_multi_pass = abort_multi_pass or consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                        break
                    if size3 < best_size:
                        if best_result and best_result != input_for_pass3:
                            try:
                                os.remove(best_result)
                            except:
                                pass
                        best_size = size3
                        best_result = temp3
                    consecutive_regressions = 0 if size3 <= best_size else consecutive_regressions + int(regressed)
                    if consecutive_regressions >= 2:
                        logger.warning("Multi-pass: Stopping after consecutive regressions in Pass 3")
                        abort_multi_pass = True
                    if size3 <= target_bytes:
                        logger.info(f"Multi-pass: Target met in Pass 3 (lossy={lossy})")
                        return temp3
                else:
                    consecutive_failures += 1
                    abort_multi_pass = abort_multi_pass or consecutive_failures >= MAX_CONSECUTIVE_FAILURES

                if abort_multi_pass:
                    break
            
            if abort_multi_pass:
                return _cleanup_and_return()
            
            # Pass 4: Progressive resolution scaling
            logger.debug("Multi-pass: Pass 4 - Resolution scaling")
            input_for_pass4 = best_result if best_result else gif_path
            for scale in [0.98, 0.95, 0.92, 0.88, 0.85]:
                if abort_multi_pass or _should_stop("pass4") or attempt_count >= MAX_MULTI_PASS_ATTEMPTS:
                    abort_multi_pass = True
                    break
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    logger.warning(f"Multi-pass Pass 4: Too many consecutive failures, breaking")
                    break
                # Track parameter combo to avoid repeats
                param_key = ('scale', scale)
                if param_key in seen_parameter_combos:
                    logger.debug(f"Multi-pass Pass 4: Skipping duplicate parameter combo scale={scale:.2f}")
                    continue
                attempt_count += 1
                seen_parameter_combos.add(param_key)
                temp4 = create_unique_temp_filename(f"mp4_s{scale:.2f}", ".gif", self.temp_dir)
                temp_files.append(temp4)
                if self._run_gifsicle(input_for_pass4, temp4, scale=scale, resize_method='lanczos3', optimize=2):
                    size4 = os.path.getsize(temp4)
                    consecutive_failures = 0  # Reset on success
                    regressed = size4 > best_size
                    # Check quality if early exit is enabled
                    if self.enable_early_exit:
                        temp_gif_info = get_gif_info(temp4)
                        colors_used = temp_gif_info.get('colors', 256)
                        lossy_used = 0  # Pass 4 doesn't use lossy
                        quality_score = self._calculate_quality_score(temp_gif_info, colors_used, lossy_used, scale)
                        if quality_score < self.min_quality_threshold:
                            logger.warning(f"Multi-pass Pass 4: Rejecting scale={scale:.2f}, quality {quality_score:.1f} < threshold {self.min_quality_threshold}")
                            try:
                                os.remove(temp4)
                            except:
                                pass
                            consecutive_failures += 1
                            continue
                    # Break if size worsens significantly (avoid repeating same grid)
                    if size4 > best_size * 1.1:  # >10% worse
                        logger.debug(f"Multi-pass Pass 4: Size worsened ({size4/(1024*1024):.2f}MB > {best_size/(1024*1024):.2f}MB), skipping remaining scales")
                        consecutive_failures += 1
                        regressed = True
                        abort_multi_pass = abort_multi_pass or consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                        break
                    if size4 < best_size:
                        if best_result and best_result != input_for_pass4:
                            try:
                                os.remove(best_result)
                            except:
                                pass
                        best_size = size4
                        best_result = temp4
                    consecutive_regressions = 0 if size4 <= best_size else consecutive_regressions + int(regressed)
                    if consecutive_regressions >= 2:
                        logger.warning("Multi-pass: Stopping after consecutive regressions in Pass 4")
                        abort_multi_pass = True
                    if size4 <= target_bytes:
                        logger.info(f"Multi-pass: Target met in Pass 4 (scale={scale})")
                        pass4_met_target = True
                        # Skip Pass 5 if Pass 4 met target
                        break
                else:
                    consecutive_failures += 1
                    abort_multi_pass = abort_multi_pass or consecutive_failures >= MAX_CONSECUTIVE_FAILURES
                
                if abort_multi_pass:
                    break
            
            if abort_multi_pass:
                return _cleanup_and_return()
            
            # Pass 5: Combined optimization with best parameters from previous passes
            # Skip if Pass 4 already met target or too many attempts/failures
            if pass4_met_target:
                logger.debug("Multi-pass: Skipping Pass 5 (target already met in Pass 4)")
            elif abort_multi_pass or _should_stop("pass5") or attempt_count >= MAX_MULTI_PASS_ATTEMPTS or consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                logger.debug(f"Multi-pass: Skipping Pass 5 (attempts={attempt_count}/{MAX_MULTI_PASS_ATTEMPTS}, failures={consecutive_failures})")
            elif best_result:
                logger.debug("Multi-pass: Pass 5 - Combined optimization")
                attempt_count += 1
                strategy = self._adaptive_gifsicle_strategy(best_size / 1024 / 1024, target_size_mb, gif_info)
                temp5 = create_unique_temp_filename("mp5_combined", ".gif", self.temp_dir)
                temp_files.append(temp5)
                
                # Try best combination
                best_colors = strategy['colors_range'][0] if strategy['colors_range'] else 192
                best_lossy = strategy['lossy_range'][1] if len(strategy['lossy_range']) > 1 else 60
                best_scale = strategy['scale_range'][1] if len(strategy['scale_range']) > 1 else 0.92
                
                if self._run_gifsicle(best_result, temp5, colors=best_colors, lossy=best_lossy, 
                                     scale=best_scale, optimize=strategy['optimize'],
                                     resize_method=strategy['resize_method']):
                    size5 = os.path.getsize(temp5)
                    consecutive_failures = 0  # Reset on success
                    # Check quality if early exit is enabled
                    if self.enable_early_exit:
                        temp_gif_info = get_gif_info(temp5)
                        quality_score = self._calculate_quality_score(temp_gif_info, best_colors, best_lossy, best_scale)
                        if quality_score < self.min_quality_threshold:
                            logger.warning(f"Multi-pass Pass 5: Rejecting combined optimization, quality {quality_score:.1f} < threshold {self.min_quality_threshold}")
                            try:
                                os.remove(temp5)
                            except:
                                pass
                        else:
                            if size5 < best_size:
                                if best_result:
                                    try:
                                        os.remove(best_result)
                                    except:
                                        pass
                                best_size = size5
                                best_result = temp5
                            consecutive_regressions = 0 if size5 <= best_size else consecutive_regressions + 1
                            if consecutive_regressions >= 2:
                                logger.warning("Multi-pass: Stopping after consecutive regressions in Pass 5 (quality-aware)")
                                abort_multi_pass = True
                            if size5 <= target_bytes:
                                logger.info("Multi-pass: Target met in Pass 5 (combined)")
                                return temp5
                    else:
                        if size5 < best_size:
                            if best_result:
                                try:
                                    os.remove(best_result)
                                except:
                                    pass
                            best_size = size5
                            best_result = temp5
                            consecutive_regressions = 0 if size5 <= best_size else consecutive_regressions + 1
                            if consecutive_regressions >= 2:
                                logger.warning("Multi-pass: Stopping after consecutive regressions in Pass 5")
                                abort_multi_pass = True
                        if size5 <= target_bytes:
                            logger.info("Multi-pass: Target met in Pass 5 (combined)")
                            return temp5
                else:
                    consecutive_failures += 1
            
            if abort_multi_pass:
                return _cleanup_and_return()
            
            # Cleanup temp files except best result
            for tf in temp_files:
                if tf != best_result:
                    try:
                        if os.path.exists(tf):
                            os.remove(tf)
                    except:
                        pass
            
            return best_result if best_result and best_size < current_bytes else None
            
        except Exception as e:
            logger.debug(f"Multi-pass optimization failed: {e}")
            # Cleanup on error
            for tf in temp_files:
                try:
                    if os.path.exists(tf):
                        os.remove(tf)
                except:
                    pass
            return None
    
    def _gifsicle_progressive_refine(self, gif_path: str, target_bytes: int, 
                                    target_size_mb: float) -> Optional[str]:
        """
        Progressive refinement strategy: coarse search then fine search.
        
        Returns:
            Path to optimized file if successful, None otherwise
        """
        if not self.progressive_refinement:
            return None
        
        if not self._is_tool_available("gifsicle"):
            return None
        
        # Check if file exists before trying to get its size
        if not os.path.exists(gif_path):
            logger.warning(f"GIF file does not exist for progressive refinement: {gif_path}")
            return None
        
        current_bytes = os.path.getsize(gif_path)
        current_size_mb = current_bytes / 1024 / 1024
        gif_info = get_gif_info(gif_path)
        
        if current_bytes <= target_bytes:
            return gif_path
        
        over_ratio = (current_size_mb / target_size_mb) if target_size_mb > 0 else 1.0
        
        # Step 1: Coarse search - try broad parameter ranges
        logger.debug("Progressive refine: Coarse search phase")
        strategy = self._adaptive_gifsicle_strategy(current_size_mb, target_size_mb, gif_info)
        
        temp_files = []
        best_result = None
        best_size = current_bytes
        promising_params = []
        
        # Cap progressive refine attempts and track failures
        MAX_PROGRESSIVE_REFINE_ATTEMPTS = 10  # Total attempts across coarse and fine search
        attempt_count = 0
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 2  # Break after 2 consecutive failures
        seen_parameter_combos = set()  # Track parameter combinations to avoid repeats
        pr_start = time.monotonic()
        pr_budget = int(max(1, self.stage_time_budgets.get('gifsicle', 10)))
        stop_refine = False

        def _refine_should_stop(reason: str) -> bool:
            if self._shutdown_checker():
                logger.info(f"Progressive refine: stopping due to shutdown [{reason}]")
                return True
            if time.monotonic() - pr_start > pr_budget:
                logger.warning(f"Progressive refine: time budget exceeded ({pr_budget}s) [{reason}]")
                return True
            return False
        
        try:
            # Coarse search: test wide parameter ranges (reduced combinations)
            # Use every 3rd value instead of every 2nd to reduce attempts
            colors_step = max(1, len(strategy['colors_range']) // 3) if len(strategy['colors_range']) > 3 else 2
            lossy_step = max(1, len(strategy['lossy_range']) // 3) if len(strategy['lossy_range']) > 3 else 2
            scale_step = max(1, len(strategy['scale_range']) // 3) if len(strategy['scale_range']) > 3 else 2
            
            best_quality_score = 0.0
            
            for colors in strategy['colors_range'][::colors_step]:
                for lossy in strategy['lossy_range'][::lossy_step]:
                    for scale in strategy['scale_range'][::scale_step]:
                        if _refine_should_stop("coarse") or attempt_count >= MAX_PROGRESSIVE_REFINE_ATTEMPTS:
                            stop_refine = True
                            break
                        if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                            logger.warning(f"Progressive refine: Too many consecutive failures ({consecutive_failures}), breaking coarse search")
                            stop_refine = True
                            break
                        # Track parameter combo to avoid repeats
                        param_key = (colors, lossy, scale)
                        if param_key in seen_parameter_combos:
                            logger.debug(f"Progressive refine: Skipping duplicate parameter combo c={colors} l={lossy} s={scale:.2f}")
                            continue
                        attempt_count += 1
                        seen_parameter_combos.add(param_key)
                        
                        temp = create_unique_temp_filename(f"pr_coarse_c{colors}_l{lossy}_s{scale:.2f}", ".gif", self.temp_dir)
                        temp_files.append(temp)
                        
                        if self._run_gifsicle(gif_path, temp, colors=colors, lossy=lossy, 
                                            scale=scale, optimize=strategy['optimize'],
                                            resize_method=strategy['resize_method']):
                            size = os.path.getsize(temp)
                            consecutive_failures = 0  # Reset on success
                            
                            # Check quality if early exit is enabled
                            quality_score = None
                            if self.enable_early_exit:
                                temp_gif_info = get_gif_info(temp)
                                quality_score = self._calculate_quality_score(temp_gif_info, colors, lossy, scale)
                                if quality_score < self.min_quality_threshold:
                                    logger.warning(f"Progressive refine coarse: Rejecting c={colors} l={lossy} s={scale:.2f}, quality {quality_score:.1f} < threshold {self.min_quality_threshold}")
                                    try:
                                        os.remove(temp)
                                    except:
                                        pass
                                    consecutive_failures += 1
                                    continue
                                best_quality_score = max(best_quality_score, quality_score)
                            
                            # Break if size worsens significantly (avoid repeating same grid)
                            if size > best_size * 1.15:  # >15% worse
                                logger.debug(f"Progressive refine: Size worsened ({size/(1024*1024):.2f}MB > {best_size/(1024*1024):.2f}MB), skipping remaining combinations")
                                consecutive_failures += 1
                                # Don't break here, continue to find better combinations
                            
                            if size < best_size:
                                if best_result:
                                    try:
                                        os.remove(best_result)
                                    except:
                                        pass
                                best_size = size
                                best_result = temp
                            
                            # Track promising parameters (within 20% of best)
                            if size <= best_size * 1.2:
                                promising_params.append({
                                    'colors': colors,
                                    'lossy': lossy,
                                    'scale': scale,
                                    'size': size,
                                    'quality': quality_score if quality_score is not None else 10.0
                                })
                            
                            if size <= target_bytes:
                                logger.info("Progressive refine: Target met in coarse search")
                                # Cleanup
                                for tf in temp_files:
                                    if tf != temp:
                                        try:
                                            os.remove(tf)
                                        except:
                                            pass
                                return temp
                        else:
                            consecutive_failures += 1
                        if stop_refine:
                            break
                    if stop_refine:
                        break
                if stop_refine:
                    break
            
            # Step 2: Fine search - narrow down to promising region
            # Check quality before fine search - skip if best quality < threshold
            if self.enable_early_exit and best_quality_score < self.min_quality_threshold:
                logger.warning(f"Progressive refine: Skipping fine search, best quality {best_quality_score:.1f} < threshold {self.min_quality_threshold}")
                return best_result if best_result and best_size < current_bytes else None
            
            if promising_params and best_result and attempt_count < MAX_PROGRESSIVE_REFINE_ATTEMPTS and consecutive_failures < MAX_CONSECUTIVE_FAILURES:
                logger.debug("Progressive refine: Fine search phase")
                # Sort by size and take top 2 (reduced from 3)
                promising_params.sort(key=lambda x: x['size'])
                top_params = promising_params[:2]
                
                for params in top_params:
                    if _refine_should_stop("fine_start") or attempt_count >= MAX_PROGRESSIVE_REFINE_ATTEMPTS:
                        stop_refine = True
                        break
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        logger.warning(f"Progressive refine fine: Too many consecutive failures ({consecutive_failures}), breaking")
                        break
                    
                    # Try variations around promising parameters
                    for color_delta in [-16, 0, 16]:
                        for lossy_delta in [-10, 0, 10]:
                            for scale_delta in [-0.02, 0, 0.02]:
                                if _refine_should_stop("fine_loop") or attempt_count >= MAX_PROGRESSIVE_REFINE_ATTEMPTS or consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                                    stop_refine = True
                                    break
                                colors = max(64, min(256, params['colors'] + color_delta))
                                lossy = max(0, min(200, params['lossy'] + lossy_delta))
                                scale = max(0.7, min(1.0, params['scale'] + scale_delta))
                                
                                # Track parameter combo to avoid repeats
                                param_key = (colors, lossy, scale)
                                if param_key in seen_parameter_combos:
                                    continue
                                attempt_count += 1
                                seen_parameter_combos.add(param_key)
                                
                                temp = create_unique_temp_filename(f"pr_fine_c{colors}_l{lossy}_s{scale:.2f}", ".gif", self.temp_dir)
                                temp_files.append(temp)
                                
                                if self._run_gifsicle(best_result, temp, colors=colors, lossy=lossy,
                                                    scale=scale, optimize=strategy['optimize'],
                                                    resize_method=strategy['resize_method']):
                                    size = os.path.getsize(temp)
                                    consecutive_failures = 0  # Reset on success
                                    
                                    # Check quality if early exit is enabled
                                    if self.enable_early_exit:
                                        temp_gif_info = get_gif_info(temp)
                                        quality_score = self._calculate_quality_score(temp_gif_info, colors, lossy, scale)
                                        if quality_score < self.min_quality_threshold:
                                            logger.warning(f"Progressive refine fine: Rejecting c={colors} l={lossy} s={scale:.2f}, quality {quality_score:.1f} < threshold {self.min_quality_threshold}")
                                            try:
                                                os.remove(temp)
                                            except:
                                                pass
                                            consecutive_failures += 1
                                            continue
                                    
                                    # Break if size worsens significantly (avoid repeating same grid)
                                    if size > best_size * 1.15:  # >15% worse
                                        consecutive_failures += 1
                                        continue
                                    
                                    if size < best_size:
                                        if best_result:
                                            try:
                                                os.remove(best_result)
                                            except:
                                                pass
                                        best_size = size
                                        best_result = temp
                                else:
                                    consecutive_failures += 1
                                    
                                    if size <= target_bytes:
                                        logger.info("Progressive refine: Target met in fine search")
                                        # Cleanup
                                        for tf in temp_files:
                                            if tf != temp:
                                                try:
                                                    os.remove(tf)
                                                except:
                                                    pass
                                        return temp
                                
                                if stop_refine:
                                    break
                            if stop_refine:
                                break
                        if stop_refine:
                            break
            
            # Cleanup
            for tf in temp_files:
                if tf != best_result:
                    try:
                        if os.path.exists(tf):
                            os.remove(tf)
                    except:
                        pass
            
            return best_result if best_result and best_size < current_bytes else None
            
        except Exception as e:
            logger.debug(f"Progressive refinement failed: {e}")
            # Cleanup on error
            for tf in temp_files:
                try:
                    if os.path.exists(tf):
                        os.remove(tf)
                except:
                    pass
            return None
    
    def _calculate_quality_score(self, gif_info: Dict[str, Any], colors: int, lossy: int, 
                                 scale: float) -> float:
        """
        Calculate quality score for a parameter combination.
        Higher score = better quality.
        
        Returns:
            Quality score (0-10)
        """
        # Base score from resolution (scale)
        resolution_score = scale * 5.0  # 0-5 points
        
        # Color score
        color_score = (colors / 256.0) * 3.0  # 0-3 points
        
        # Lossy penalty (lower lossy = better quality)
        lossy_penalty = (lossy / 200.0) * 2.0  # 0-2 points penalty
        lossy_score = 2.0 - lossy_penalty  # 2-0 points
        
        total_score = resolution_score + color_score + lossy_score
        return min(10.0, max(0.0, total_score))
    
    def _gifsicle_quality_aware_optimize(self, gif_path: str, target_bytes: int,
                                        target_size_mb: float, min_quality: float = 5.0) -> Optional[str]:
        """
        Quality-aware optimization that prefers higher quality when multiple results meet target.
        
        Args:
            gif_path: Input GIF path
            target_bytes: Target size in bytes
            target_size_mb: Target size in MB
            min_quality: Minimum quality score threshold
        
        Returns:
            Path to optimized file if successful, None otherwise
        """
        if not self._is_tool_available("gifsicle"):
            return None
        
        current_bytes = os.path.getsize(gif_path)
        gif_info = get_gif_info(gif_path)
        strategy = self._adaptive_gifsicle_strategy(current_bytes / 1024 / 1024, target_size_mb, gif_info)
        
        temp_files = []
        candidates = []  # List of (path, size, quality_score)
        
        try:
            # Generate parameter combinations
            for colors in strategy['colors_range']:
                for lossy in strategy['lossy_range']:
                    for scale in strategy['scale_range']:
                        if self._shutdown_checker():
                            break
                        
                        quality = self._calculate_quality_score(gif_info, colors, lossy, scale)
                        if quality < min_quality:
                            continue  # Skip low quality combinations
                        
                        temp = create_unique_temp_filename(f"qa_c{colors}_l{lossy}_s{scale:.2f}", ".gif", self.temp_dir)
                        temp_files.append(temp)
                        
                        if self._run_gifsicle(gif_path, temp, colors=colors, lossy=lossy,
                                            scale=scale, optimize=strategy['optimize'],
                                            resize_method=strategy['resize_method']):
                            size = os.path.getsize(temp)
                            
                            # Only consider candidates that meet target or are close
                            if size <= target_bytes * 1.1:  # Within 10% of target
                                candidates.append({
                                    'path': temp,
                                    'size': size,
                                    'quality': quality,
                                    'colors': colors,
                                    'lossy': lossy,
                                    'scale': scale
                                })
            
            if not candidates:
                # Cleanup
                for tf in temp_files:
                    try:
                        if os.path.exists(tf):
                            os.remove(tf)
                    except:
                        pass
                return None
            
            # Sort candidates: first by meeting target, then by quality
            candidates.sort(key=lambda x: (x['size'] > target_bytes, -x['quality'], x['size']))
            
            best = candidates[0]
            
            # Cleanup other candidates
            for candidate in candidates[1:]:
                try:
                    if os.path.exists(candidate['path']):
                        os.remove(candidate['path'])
                except:
                    pass
            
            # Cleanup other temp files
            for tf in temp_files:
                if tf != best['path']:
                    try:
                        if os.path.exists(tf):
                            os.remove(tf)
                    except:
                        pass
            
            logger.info(f"Quality-aware: Selected params (colors={best['colors']}, lossy={best['lossy']}, "
                       f"scale={best['scale']:.2f}) with quality={best['quality']:.2f}")
            
            return best['path'] if best['size'] < current_bytes else None
            
        except Exception as e:
            logger.debug(f"Quality-aware optimization failed: {e}")
            # Cleanup on error
            for tf in temp_files:
                try:
                    if os.path.exists(tf):
                        os.remove(tf)
                except:
                    pass
            return None
    
    def _gifsicle_multi_resolution(self, gif_path: str, target_bytes: int,
                                  target_size_mb: float) -> Optional[str]:
        """
        Multi-resolution strategy: test multiple resolution targets simultaneously.
        
        Returns:
            Path to optimized file if successful, None otherwise
        """
        if not self._is_tool_available("gifsicle"):
            return None
        
        gif_info = get_gif_info(gif_path)
        current_width = gif_info.get('width', 360)
        current_height = gif_info.get('height', 360)
        
        # Test multiple resolution targets
        resolution_targets = [
            (int(current_width * 0.95), int(current_height * 0.95)),  # 95%
            (int(current_width * 0.90), int(current_height * 0.90)),  # 90%
            (int(current_width * 0.85), int(current_height * 0.85)),  # 85%
            (int(current_width * 0.80), int(current_height * 0.80)),  # 80%
        ]
        
        temp_files = []
        candidates = []
        
        try:
            strategy = self._adaptive_gifsicle_strategy(
                os.path.getsize(gif_path) / 1024 / 1024, target_size_mb, gif_info
            )
            
            for target_width, target_height in resolution_targets:
                if self._shutdown_checker():
                    break
                
                # Calculate scale factor
                scale = min(target_width / current_width, target_height / current_height)
                
                # Try with moderate compression
                colors = strategy['colors_range'][1] if len(strategy['colors_range']) > 1 else 160
                lossy = strategy['lossy_range'][1] if len(strategy['lossy_range']) > 1 else 60
                
                temp = create_unique_temp_filename(f"mr_w{target_width}_h{target_height}", ".gif", self.temp_dir)
                temp_files.append(temp)
                
                if self._run_gifsicle(gif_path, temp, colors=colors, lossy=lossy,
                                    scale=scale, optimize=strategy['optimize'],
                                    resize_method=strategy['resize_method']):
                    size = os.path.getsize(temp)
                    if size <= target_bytes * 1.1:  # Within 10% of target
                        quality = self._calculate_quality_score(gif_info, colors, lossy, scale)
                        candidates.append({
                            'path': temp,
                            'size': size,
                            'quality': quality,
                            'width': target_width,
                            'height': target_height
                        })
            
            if not candidates:
                # Cleanup
                for tf in temp_files:
                    try:
                        if os.path.exists(tf):
                            os.remove(tf)
                    except:
                        pass
                return None
            
            # Prefer higher resolution when multiple meet target
            candidates.sort(key=lambda x: (x['size'] > target_bytes, -x['width'], -x['quality'], x['size']))
            
            best = candidates[0]
            
            # Cleanup other candidates
            for candidate in candidates[1:]:
                try:
                    if os.path.exists(candidate['path']):
                        os.remove(candidate['path'])
                except:
                    pass
            
            # Cleanup other temp files
            for tf in temp_files:
                if tf != best['path']:
                    try:
                        if os.path.exists(tf):
                            os.remove(tf)
                    except:
                        pass
            
            logger.info(f"Multi-resolution: Selected {best['width']}x{best['height']} "
                       f"(quality={best['quality']:.2f}, size={best['size']/1024/1024:.2f}MB)")
            
            return best['path']
            
        except Exception as e:
            logger.debug(f"Multi-resolution optimization failed: {e}")
            # Cleanup on error
            for tf in temp_files:
                try:
                    if os.path.exists(tf):
                        os.remove(tf)
                except:
                    pass
            return None
    
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
        """Build scale filter with proper aspect ratio preservation and high quality flags"""
        # Use high-quality scaling flags for better output quality
        # lanczos: high-quality resampling algorithm
        # accurate_rnd: more accurate rounding
        # full_chroma_int: full chroma interpolation
        if width == -1 and height == -1:
            return "scale=iw:ih:flags=lanczos+accurate_rnd+full_chroma_int"
        elif height == -1:
            return f"scale={width}:-2:flags=lanczos+accurate_rnd+full_chroma_int"
        else:
            # Use force_original_aspect_ratio=decrease to preserve aspect ratio when both dimensions are specified
            return f"scale={width}:{height}:flags=lanczos+accurate_rnd+full_chroma_int:force_original_aspect_ratio=decrease"
    
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
            
            # Detect if input is a GIF (not source video) - GIF inputs need more aggressive compression
            # since they're already optimized and re-encoding often increases size
            is_gif_input = input_path.lower().endswith('.gif')
            
            # Determine mpdecimate frac
            if mpdecimate_frac is None:
                try:
                    similarity_analysis = analyze_frame_similarity(input_path, self.temp_dir)
                    mpdecimate_frac = similarity_analysis.get('optimal_frac', 0.3)
                except Exception:
                    mpdecimate_frac = 0.3
            
            # For GIF inputs, use more aggressive compression parameters
            # since we're re-encoding from an already-optimized GIF
            adjusted_max_colors = max_colors
            adjusted_fps = fps
            original_width = new_width
            if is_gif_input:
                # FIX: Much more aggressive parameters for GIF-to-GIF re-encoding
                # Re-encoding already-optimized GIFs often increases size, so be very conservative
                mpdecimate_frac = max(0.3, min(0.5, mpdecimate_frac * 1.5))  # More aggressive duplicate removal
                # Reduce colors much more aggressively for GIF inputs
                adjusted_max_colors = min(max_colors, int(max_colors * 0.6))  # Reduce colors by 40% (was 20%)
                # Reduce FPS more aggressively
                adjusted_fps = max(6, int(fps * 0.75))  # Reduce FPS by 25% (was 10%)
                # Also reduce width more aggressively
                new_width = int(new_width * 0.9)  # Reduce width by 10%
            
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
            
            # Early detection: if input is a GIF and parameters won't reduce size significantly, skip
            if is_gif_input:
                # Estimate if re-encoding will help based on current parameters
                # GIF-to-GIF re-encoding often increases size, so be conservative
                # FIX: Improved estimation formula that accounts for width reduction and more aggressive parameters
                width_reduction_factor = (new_width / 360.0) if new_width < 360 else 1.0  # Assume 360 is original
                estimated_reduction = (
                    (mpdecimate_frac * 0.4) +  # mpdecimate contribution (increased from 0.3)
                    ((256 - adjusted_max_colors) / 256 * 0.3) +  # Color reduction contribution (increased from 0.2)
                    ((1.0 - width_reduction_factor) * 0.3)  # Width reduction contribution (new)
                )
                # FIX: Increased threshold from 15% to 25% - GIF-to-GIF re-encoding is risky
                if estimated_reduction < 0.25:  # Less than 25% estimated reduction (was 15%)
                    logger.debug(
                        f"Skipping FFmpeg re-encode from GIF input - estimated reduction too small "
                        f"(frac={mpdecimate_frac:.2f}, colors={adjusted_max_colors}, width={new_width}, estimated={estimated_reduction:.1%})"
                    )
                    return False
            
            # Get palette path (use adjusted parameters for GIF inputs)
            palette_path = self._get_palette_cache_key(
                input_path, 'reencode', new_width, -1, adjusted_fps, adjusted_max_colors,
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
                    f'fps={adjusted_fps}',
                    self._build_scale_filter(new_width, -1)
                ]
                vf_palette = ','.join(pre + [f'palettegen=max_colors={int(adjusted_max_colors)}:stats_mode={stats_mode}'])
                
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
            
            # Apply palette (use adjusted parameters for GIF inputs)
            pre = [
                f'mpdecimate=hi=512:lo=256:frac={mpdecimate_frac:.2f}',
                f'fps={adjusted_fps}',
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
                    logger.warning(f"FFmpeg palette re-encode output {output_size_mb:.2f}MB is >10% larger than input {input_size_mb:.2f}MB. "
                                  f"Skipping re-encode to avoid increasing file size.")
                    # Clean up the oversized output
                    try:
                        os.remove(output_path)
                    except Exception:
                        pass
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

