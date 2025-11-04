"""
Adaptive Parameter Adjustment Module
Handles dynamic adjustment of encoding parameters to meet bitrate requirements
"""

import logging
import math
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass

try:
    from .config_manager import ConfigManager
    from .bitrate_validator import BitrateValidator, ValidationResult, AdjustmentPlan
except ImportError:
    from config_manager import ConfigManager
    from bitrate_validator import BitrateValidator, ValidationResult, AdjustmentPlan

logger = logging.getLogger(__name__)


@dataclass
class ParameterAdjustment:
    """Result of parameter adjustment"""
    success: bool
    adjusted_params: Dict[str, Any]
    original_params: Dict[str, Any]
    adjustment_type: str  # 'resolution', 'fps', 'combined', 'none'
    quality_impact: str  # 'minimal', 'moderate', 'significant'
    bitrate_improvement: int  # kbps gained
    message: str


class AdaptiveParameterAdjuster:
    """Adjusts encoding parameters to meet bitrate floor requirements"""
    
    def __init__(self, config: ConfigManager, bitrate_validator: Optional[BitrateValidator] = None):
        self.config = config
        self.bitrate_validator = bitrate_validator or BitrateValidator(config)
        
        # Load configuration settings
        self._load_adjustment_settings()
        
        logger.debug("AdaptiveParameterAdjuster initialized")
    
    def _load_adjustment_settings(self):
        """Load adjustment configuration settings"""
        # Safety margins for bitrate calculations
        self.safety_margin = self.config.get('video_compression.bitrate_validation.safety_margin', 1.1)
        
        # Resolution adjustment settings
        self.min_resolution_width = self.config.get('video_compression.bitrate_validation.min_resolution.width', 320)
        self.min_resolution_height = self.config.get('video_compression.bitrate_validation.min_resolution.height', 180)
        
        # FPS adjustment settings - use config default of 20 to match video_compression.yaml
        self.min_fps = self.config.get('video_compression.bitrate_validation.min_fps', 20)
        self.fps_reduction_steps = self.config.get('video_compression.bitrate_validation.fps_reduction_steps', [0.8, 0.6, 0.5])
        
        # Log the loaded configuration values for debugging
        logger.debug(f"Adaptive Parameter Adjuster initialized with min_fps: {self.min_fps}")
        logger.debug(f"FPS reduction steps: {self.fps_reduction_steps}")
        
        # Quality thresholds
        self.quality_thresholds = {
            'minimal': 0.8,    # 80% of original quality metric
            'moderate': 0.6,   # 60% of original quality metric
            'significant': 0.4  # 40% of original quality metric
        }
        
        # Fallback resolution cascade
        self.fallback_resolutions = self._get_fallback_resolution_cascade()
    
    def reload_configuration(self):
        """Reload configuration values - useful when config files change"""
        logger.info("Reloading adaptive parameter adjuster configuration")
        
        # Store old values for comparison
        old_min_fps = self.min_fps
        old_fps_steps = self.fps_reduction_steps.copy()
        
        # Reload values
        self.min_fps = self.config.get('video_compression.bitrate_validation.min_fps', 20)
        self.fps_reduction_steps = self.config.get('video_compression.bitrate_validation.fps_reduction_steps', [0.8, 0.6, 0.5])
        self.min_resolution_width = self.config.get('video_compression.bitrate_validation.min_resolution.width', 320)
        self.min_resolution_height = self.config.get('video_compression.bitrate_validation.min_resolution.height', 180)
        
        # Log changes
        if old_min_fps != self.min_fps:
            logger.info(f"Min FPS configuration changed: {old_min_fps} → {self.min_fps}")
        if old_fps_steps != self.fps_reduction_steps:
            logger.info(f"FPS reduction steps changed: {old_fps_steps} → {self.fps_reduction_steps}")
        
        # Reload fallback resolutions in case they changed
        self.fallback_resolutions = self._get_fallback_resolution_cascade()
    
    def _get_fallback_resolution_cascade(self) -> List[Tuple[int, int]]:
        """Get ordered list of fallback resolutions for extreme compression"""
        config_resolutions = self.config.get('video_compression.bitrate_validation.fallback_resolutions', [])
        
        # Default cascade from highest to lowest quality
        default_cascade = [
            (1280, 720),   # 720p
            (854, 480),    # 480p
            (640, 360),    # 360p
            (480, 270),    # 270p
            (426, 240),    # 240p
            (320, 180),    # Ultra-low for extreme cases
        ]
        
        if config_resolutions:
            try:
                return [(res[0], res[1]) for res in config_resolutions]
            except (IndexError, TypeError):
                logger.warning("Invalid fallback_resolutions format in config, using defaults")
                return default_cascade
        
        return default_cascade
    
    def adjust_for_bitrate_floor(self, params: Dict[str, Any], min_bitrate: int, 
                                video_info: Dict[str, Any], target_size_mb: float) -> ParameterAdjustment:
        """
        Adjust parameters to meet minimum bitrate requirements
        
        Args:
            params: Current encoding parameters
            min_bitrate: Minimum required bitrate in kbps
            video_info: Video information (duration, etc.)
            target_size_mb: Target file size in MB
            
        Returns:
            ParameterAdjustment with results
        """
        original_params = params.copy()
        current_bitrate = params.get('bitrate', 1000)
        
        # Check if adjustment is needed
        if current_bitrate >= min_bitrate:
            return ParameterAdjustment(
                success=True,
                adjusted_params=params,
                original_params=original_params,
                adjustment_type='none',
                quality_impact='minimal',
                bitrate_improvement=0,
                message=f"No adjustment needed. Current bitrate {current_bitrate}kbps meets minimum {min_bitrate}kbps"
            )
        
        logger.info(f"Adjusting parameters: current bitrate {current_bitrate}kbps < minimum {min_bitrate}kbps")
        
        # Try resolution reduction first
        resolution_result = self._try_resolution_adjustment(params, min_bitrate, video_info, target_size_mb)
        if resolution_result.success:
            return resolution_result
        
        # Try FPS reduction
        fps_result = self._try_fps_adjustment(params, min_bitrate, video_info, target_size_mb)
        if fps_result.success:
            return fps_result
        
        # Try combined approach
        combined_result = self._try_combined_adjustment(params, min_bitrate, video_info, target_size_mb)
        if combined_result.success:
            return combined_result
        
        # If all adjustments fail, return failure with best attempt
        best_attempt = max([resolution_result, fps_result, combined_result], 
                          key=lambda x: x.adjusted_params.get('bitrate', 0))
        
        return ParameterAdjustment(
            success=False,
            adjusted_params=best_attempt.adjusted_params,
            original_params=original_params,
            adjustment_type='failed',
            quality_impact='significant',
            bitrate_improvement=best_attempt.adjusted_params.get('bitrate', 0) - current_bitrate,
            message=f"Unable to meet minimum bitrate {min_bitrate}kbps. Best achieved: {best_attempt.adjusted_params.get('bitrate', 0)}kbps"
        )
    
    def _try_resolution_adjustment(self, params: Dict[str, Any], min_bitrate: int,
                                 video_info: Dict[str, Any], target_size_mb: float) -> ParameterAdjustment:
        """Try to meet bitrate requirements through resolution reduction"""
        original_params = params.copy()
        current_width = params.get('width', 1920)
        current_height = params.get('height', 1080)
        
        # Try each fallback resolution
        for width, height in self.fallback_resolutions:
            if width >= current_width and height >= current_height:
                continue  # Skip if not actually reducing resolution
            
            # Calculate new bitrate with reduced resolution
            adjusted_params = params.copy()
            adjusted_params.update({
                'width': width,
                'height': height
            })
            
            new_bitrate = self._calculate_bitrate_with_safety_margin(
                target_size_mb, video_info.get('duration_seconds', 60), 
                params.get('audio_bitrate', 64)
            )
            adjusted_params['bitrate'] = new_bitrate
            
            if new_bitrate >= min_bitrate:
                # Calculate quality impact
                resolution_ratio = (width * height) / (current_width * current_height)
                quality_impact = self._assess_quality_impact(resolution_ratio, 'resolution')
                
                return ParameterAdjustment(
                    success=True,
                    adjusted_params=adjusted_params,
                    original_params=original_params,
                    adjustment_type='resolution',
                    quality_impact=quality_impact,
                    bitrate_improvement=new_bitrate - params.get('bitrate', 1000),
                    message=f"Resolution adjusted to {width}x{height} to achieve {new_bitrate}kbps"
                )
        
        # Return best attempt even if it doesn't meet minimum
        if self.fallback_resolutions:
            width, height = self.fallback_resolutions[-1]  # Smallest resolution
            adjusted_params = params.copy()
            adjusted_params.update({
                'width': width,
                'height': height,
                'bitrate': self._calculate_bitrate_with_safety_margin(
                    target_size_mb, video_info.get('duration_seconds', 60),
                    params.get('audio_bitrate', 64)
                )
            })
            
            return ParameterAdjustment(
                success=False,
                adjusted_params=adjusted_params,
                original_params=original_params,
                adjustment_type='resolution',
                quality_impact='significant',
                bitrate_improvement=adjusted_params['bitrate'] - params.get('bitrate', 1000),
                message=f"Resolution reduced to minimum {width}x{height} but still below minimum bitrate"
            )
        
        return ParameterAdjustment(
            success=False,
            adjusted_params=params,
            original_params=original_params,
            adjustment_type='resolution',
            quality_impact='minimal',
            bitrate_improvement=0,
            message="No resolution adjustment possible"
        )
    
    def _try_fps_adjustment(self, params: Dict[str, Any], min_bitrate: int,
                          video_info: Dict[str, Any], target_size_mb: float) -> ParameterAdjustment:
        """Try to meet bitrate requirements through FPS reduction"""
        original_params = params.copy()
        current_fps = params.get('fps', 30)
        current_bitrate = params.get('bitrate', 1000)
        
        # Try each FPS reduction step
        for fps_factor in self.fps_reduction_steps:
            new_fps = max(self.min_fps, current_fps * fps_factor)
            
            if new_fps >= current_fps:
                continue  # Skip if not actually reducing FPS
            
            # Calculate bitrate improvement from FPS reduction
            # Lower FPS means we can allocate more bits per frame
            fps_ratio = current_fps / new_fps
            improved_bitrate = int(current_bitrate * fps_ratio)
            
            # Apply safety margin
            safe_bitrate = int(improved_bitrate / self.safety_margin)
            
            adjusted_params = params.copy()
            adjusted_params.update({
                'fps': new_fps,
                'bitrate': safe_bitrate
            })
            
            if safe_bitrate >= min_bitrate:
                # Calculate quality impact
                fps_ratio_impact = new_fps / current_fps
                quality_impact = self._assess_quality_impact(fps_ratio_impact, 'fps')
                
                return ParameterAdjustment(
                    success=True,
                    adjusted_params=adjusted_params,
                    original_params=original_params,
                    adjustment_type='fps',
                    quality_impact=quality_impact,
                    bitrate_improvement=safe_bitrate - current_bitrate,
                    message=f"FPS reduced to {new_fps:.1f} to achieve {safe_bitrate}kbps"
                )
        
        # Return best attempt
        if self.fps_reduction_steps:
            best_fps_factor = min(self.fps_reduction_steps)
            new_fps = max(self.min_fps, current_fps * best_fps_factor)
            fps_ratio = current_fps / new_fps
            improved_bitrate = int(current_bitrate * fps_ratio)
            safe_bitrate = int(improved_bitrate / self.safety_margin)
            
            adjusted_params = params.copy()
            adjusted_params.update({
                'fps': new_fps,
                'bitrate': safe_bitrate
            })
            
            return ParameterAdjustment(
                success=False,
                adjusted_params=adjusted_params,
                original_params=original_params,
                adjustment_type='fps',
                quality_impact='significant',
                bitrate_improvement=safe_bitrate - current_bitrate,
                message=f"FPS reduced to minimum {new_fps:.1f} but still below minimum bitrate"
            )
        
        return ParameterAdjustment(
            success=False,
            adjusted_params=params,
            original_params=original_params,
            adjustment_type='fps',
            quality_impact='minimal',
            bitrate_improvement=0,
            message="No FPS adjustment possible"
        )
    
    def _try_combined_adjustment(self, params: Dict[str, Any], min_bitrate: int,
                               video_info: Dict[str, Any], target_size_mb: float) -> ParameterAdjustment:
        """Try combined resolution and FPS adjustment"""
        original_params = params.copy()
        current_width = params.get('width', 1920)
        current_height = params.get('height', 1080)
        current_fps = params.get('fps', 30)
        current_bitrate = params.get('bitrate', 1000)
        
        best_result = None
        best_bitrate = 0
        
        # Try combinations of resolution and FPS reductions
        for width, height in self.fallback_resolutions:
            if width >= current_width and height >= current_height:
                continue
            
            for fps_factor in self.fps_reduction_steps:
                new_fps = max(self.min_fps, current_fps * fps_factor)
                
                if new_fps >= current_fps:
                    continue
                
                # Calculate combined improvement
                resolution_ratio = (width * height) / (current_width * current_height)
                fps_ratio = current_fps / new_fps
                
                # Combined effect: resolution reduction allows higher quality per pixel,
                # FPS reduction allows more bits per frame
                combined_improvement = fps_ratio * (1.0 / math.sqrt(resolution_ratio))
                improved_bitrate = int(current_bitrate * combined_improvement)
                safe_bitrate = int(improved_bitrate / self.safety_margin)
                
                adjusted_params = params.copy()
                adjusted_params.update({
                    'width': width,
                    'height': height,
                    'fps': new_fps,
                    'bitrate': safe_bitrate
                })
                
                if safe_bitrate >= min_bitrate:
                    # Calculate combined quality impact
                    combined_ratio = resolution_ratio * (new_fps / current_fps)
                    quality_impact = self._assess_quality_impact(combined_ratio, 'combined')
                    
                    return ParameterAdjustment(
                        success=True,
                        adjusted_params=adjusted_params,
                        original_params=original_params,
                        adjustment_type='combined',
                        quality_impact=quality_impact,
                        bitrate_improvement=safe_bitrate - current_bitrate,
                        message=f"Combined adjustment: {width}x{height}@{new_fps:.1f}fps to achieve {safe_bitrate}kbps"
                    )
                
                # Track best attempt
                if safe_bitrate > best_bitrate:
                    best_bitrate = safe_bitrate
                    best_result = ParameterAdjustment(
                        success=False,
                        adjusted_params=adjusted_params,
                        original_params=original_params,
                        adjustment_type='combined',
                        quality_impact='significant',
                        bitrate_improvement=safe_bitrate - current_bitrate,
                        message=f"Best combined attempt: {width}x{height}@{new_fps:.1f}fps achieved {safe_bitrate}kbps"
                    )
        
        return best_result or ParameterAdjustment(
            success=False,
            adjusted_params=params,
            original_params=original_params,
            adjustment_type='combined',
            quality_impact='minimal',
            bitrate_improvement=0,
            message="No combined adjustment possible"
        )
    
    def get_fallback_resolutions(self, original_res: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Get fallback resolutions smaller than the original
        
        Args:
            original_res: Original (width, height) tuple
            
        Returns:
            List of (width, height) tuples in descending order of quality
        """
        original_width, original_height = original_res
        original_pixels = original_width * original_height
        
        # Filter to only resolutions smaller than original
        fallback_options = [
            (w, h) for w, h in self.fallback_resolutions
            if w * h < original_pixels and w <= original_width and h <= original_height
        ]
        
        # Sort by pixel count (descending - highest quality first)
        fallback_options.sort(key=lambda res: res[0] * res[1], reverse=True)
        
        return fallback_options
    
    def calculate_optimal_fps_reduction(self, current_fps: float, bitrate_deficit: float) -> float:
        """
        Calculate optimal FPS reduction to address bitrate deficit
        
        Args:
            current_fps: Current frame rate
            bitrate_deficit: How much bitrate improvement is needed (ratio)
            
        Returns:
            Optimal new FPS value
        """
        # Calculate required FPS reduction to meet deficit
        # bitrate_deficit is the ratio of needed improvement (e.g., 1.5 means need 50% more bitrate)
        required_fps_ratio = 1.0 / bitrate_deficit
        target_fps = current_fps * required_fps_ratio
        
        # Ensure we don't go below minimum
        target_fps = max(self.min_fps, target_fps)
        
        # Round to reasonable FPS values
        common_fps_values = [60, 50, 30, 25, 24, 20, 15, 12, 10]
        
        # Find the closest common FPS value that's not higher than target
        for fps in common_fps_values:
            if fps <= target_fps and fps <= current_fps:
                return float(fps)
        
        return max(self.min_fps, target_fps)
    
    def _calculate_bitrate_with_safety_margin(self, target_size_mb: float, 
                                            duration_s: float, audio_kbps: int) -> int:
        """Calculate target video bitrate with safety margin applied"""
        # Handle zero or negative duration
        if duration_s <= 0:
            duration_s = 60  # Default to 60 seconds
        
        total_bits = target_size_mb * 8 * 1024 * 1024
        available_bits = total_bits - (audio_kbps * 1000 * duration_s)
        video_kbps = int(available_bits / (duration_s * 1000))
        
        # Apply safety margin
        safe_video_kbps = int(video_kbps / self.safety_margin)
        
        return max(64, safe_video_kbps)  # Minimum 64kbps
    
    def _assess_quality_impact(self, ratio: float, adjustment_type: str) -> str:
        """Assess quality impact based on reduction ratio and type"""
        if adjustment_type == 'resolution':
            # For resolution, ratio is pixel count ratio
            if ratio >= self.quality_thresholds['minimal']:
                return 'minimal'
            elif ratio >= self.quality_thresholds['moderate']:
                return 'moderate'
            else:
                return 'significant'
        
        elif adjustment_type == 'fps':
            # For FPS, ratio is frame rate ratio
            if ratio >= 0.8:  # 80% or more of original FPS
                return 'minimal'
            elif ratio >= 0.6:  # 60% or more of original FPS
                return 'moderate'
            else:
                return 'significant'
        
        elif adjustment_type == 'combined':
            # For combined, use more conservative thresholds
            if ratio >= 0.7:
                return 'minimal'
            elif ratio >= 0.5:
                return 'moderate'
            else:
                return 'significant'
        
        return 'moderate'  # Default
    
    def validate_adjusted_parameters(self, params: Dict[str, Any], encoder: str) -> ValidationResult:
        """Validate that adjusted parameters meet encoder requirements"""
        bitrate = params.get('bitrate', 1000)
        return self.bitrate_validator.validate_bitrate(bitrate, encoder)
    
    def log_adjustment_result(self, result: ParameterAdjustment, context: str = ""):
        """Log parameter adjustment result with appropriate level"""
        prefix = f"[{context}] " if context else ""
        
        if result.success:
            logger.info(f"{prefix}Parameter adjustment successful: {result.message}")
            logger.debug(f"{prefix}Quality impact: {result.quality_impact}, Bitrate improvement: +{result.bitrate_improvement}kbps")
        else:
            logger.warning(f"{prefix}Parameter adjustment failed: {result.message}")
            if result.bitrate_improvement > 0:
                logger.info(f"{prefix}Partial improvement achieved: +{result.bitrate_improvement}kbps")