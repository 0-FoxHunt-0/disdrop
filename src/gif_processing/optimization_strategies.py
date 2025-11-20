"""
Optimization Strategies
Implements strategy pattern for different GIF optimization approaches
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class OptimizationStrategy(ABC):
    """Base class for optimization strategies"""
    
    def __init__(self, config_helper, temp_dir: str, shutdown_checker=None):
        """
        Initialize strategy.
        
        Args:
            config_helper: GifConfigHelper instance
            temp_dir: Temporary directory path
            shutdown_checker: Optional shutdown checker callback
        """
        self.config = config_helper
        self.temp_dir = temp_dir
        self._shutdown_checker = shutdown_checker or (lambda: False)
    
    @abstractmethod
    def select_parameters(self, current_size_mb: float, target_size_mb: float, 
                         base_params: Dict[str, Any], gif_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select optimization parameters based on strategy.
        
        Args:
            current_size_mb: Current file size in MB
            target_size_mb: Target file size in MB
            base_params: Base parameters from config
            gif_info: GIF information (width, height, fps, etc.)
        
        Returns:
            Dictionary with optimization parameters
        """
        pass
    
    @abstractmethod
    def get_parameter_sequence(self, target_size_mb: float, base_params: Dict[str, Any], 
                              gif_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get sequence of parameter sets to try in order.
        
        Args:
            target_size_mb: Target file size in MB
            base_params: Base parameters from config
            gif_info: GIF information
        
        Returns:
            List of parameter dictionaries to try
        """
        pass


class QualityFirstStrategy(OptimizationStrategy):
    """Strategy that prioritizes quality over size reduction"""
    
    def select_parameters(self, current_size_mb: float, target_size_mb: float,
                         base_params: Dict[str, Any], gif_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select parameters that maximize quality while meeting size target.
        """
        params = base_params.copy()
        size_ratio = current_size_mb / target_size_mb if target_size_mb > 0 else 1.0
        
        # Quality-first: minimize quality loss
        if size_ratio > 1.5:
            # Need significant reduction - reduce resolution first
            params['width'] = max(int(gif_info.get('width', 360) * 0.85), 360)
            params['fps'] = max(int(base_params.get('fps', 20) * 0.9), 15)
            params['colors'] = max(int(base_params.get('colors', 256) * 0.9), 192)
        elif size_ratio > 1.2:
            # Moderate reduction - slight quality adjustments
            params['width'] = max(int(gif_info.get('width', 360) * 0.92), 360)
            params['fps'] = max(int(base_params.get('fps', 20) * 0.95), 15)
            params['colors'] = max(int(base_params.get('colors', 256) * 0.95), 224)
        else:
            # Small reduction - minimal quality impact
            params['width'] = max(int(gif_info.get('width', 360) * 0.96), 360)
            params['fps'] = base_params.get('fps', 20)
            params['colors'] = base_params.get('colors', 256)
        
        # Prefer high-quality dithering
        params['dither'] = 'floyd_steinberg'
        params['lossy'] = min(base_params.get('lossy', 60), 40)  # Lower lossy for quality
        
        return params
    
    def get_parameter_sequence(self, target_size_mb: float, base_params: Dict[str, Any],
                              gif_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get sequence prioritizing quality"""
        quality_stages = self.config.get_quality_stages()
        sequence = []
        
        # Start with highest quality stages
        for stage in quality_stages[:3]:  # Top 3 quality stages
            params = base_params.copy()
            params.update({
                'width': int(gif_info.get('width', 360) * stage.get('resolution_multiplier', 1.0)),
                'fps': int(base_params.get('fps', 20) * stage.get('fps_multiplier', 1.0)),
                'colors': stage.get('colors', 256),
                'dither': stage.get('dither', 'floyd_steinberg'),
                'lossy': stage.get('lossy', 0)
            })
            sequence.append(params)
        
        return sequence


class SizeFirstStrategy(OptimizationStrategy):
    """Strategy that prioritizes size reduction over quality"""
    
    def select_parameters(self, current_size_mb: float, target_size_mb: float,
                         base_params: Dict[str, Any], gif_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select parameters that minimize size while maintaining minimum quality.
        """
        params = base_params.copy()
        size_ratio = current_size_mb / target_size_mb if target_size_mb > 0 else 1.0
        
        # Size-first: aggressive reduction
        quality_floors = self.config.get_quality_floors()
        min_width = quality_floors.get('min_width', 360)
        min_fps = quality_floors.get('min_fps', 15)
        
        if size_ratio > 2.0:
            # Very aggressive reduction
            params['width'] = min_width
            params['fps'] = min_fps
            params['colors'] = 96
            params['lossy'] = 120
        elif size_ratio > 1.5:
            # Aggressive reduction
            params['width'] = max(int(gif_info.get('width', 360) * 0.75), min_width)
            params['fps'] = max(int(base_params.get('fps', 20) * 0.8), min_fps)
            params['colors'] = 128
            params['lossy'] = 100
        else:
            # Moderate reduction
            params['width'] = max(int(gif_info.get('width', 360) * 0.85), min_width)
            params['fps'] = max(int(base_params.get('fps', 20) * 0.85), min_fps)
            params['colors'] = 160
            params['lossy'] = 80
        
        # Use simpler dithering for better compression
        params['dither'] = 'bayer'
        
        return params
    
    def get_parameter_sequence(self, target_size_mb: float, base_params: Dict[str, Any],
                              gif_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get sequence prioritizing size reduction"""
        quality_stages = self.config.get_quality_stages()
        sequence = []
        
        # Start with lower quality stages for size reduction
        for stage in reversed(quality_stages[-3:]):  # Bottom 3 quality stages
            params = base_params.copy()
            params.update({
                'width': int(gif_info.get('width', 360) * stage.get('resolution_multiplier', 0.8)),
                'fps': int(base_params.get('fps', 20) * stage.get('fps_multiplier', 0.8)),
                'colors': stage.get('colors', 128),
                'dither': stage.get('dither', 'bayer'),
                'lossy': stage.get('lossy', 80)
            })
            sequence.append(params)
        
        return sequence


class BalancedStrategy(OptimizationStrategy):
    """Strategy that balances quality and size"""
    
    def select_parameters(self, current_size_mb: float, target_size_mb: float,
                         base_params: Dict[str, Any], gif_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select parameters that balance quality and size.
        """
        params = base_params.copy()
        size_ratio = current_size_mb / target_size_mb if target_size_mb > 0 else 1.0
        
        # Balanced approach
        quality_floors = self.config.get_quality_floors()
        min_width = quality_floors.get('min_width', 360)
        min_fps = quality_floors.get('min_fps', 15)
        
        if size_ratio > 1.5:
            # Need reduction - balanced approach
            params['width'] = max(int(gif_info.get('width', 360) * 0.88), min_width)
            params['fps'] = max(int(base_params.get('fps', 20) * 0.9), min_fps)
            params['colors'] = max(int(base_params.get('colors', 256) * 0.85), 160)
            params['lossy'] = 60
        elif size_ratio > 1.2:
            # Moderate reduction
            params['width'] = max(int(gif_info.get('width', 360) * 0.92), min_width)
            params['fps'] = max(int(base_params.get('fps', 20) * 0.95), min_fps)
            params['colors'] = max(int(base_params.get('colors', 256) * 0.9), 192)
            params['lossy'] = 40
        else:
            # Small reduction
            params['width'] = max(int(gif_info.get('width', 360) * 0.96), min_width)
            params['fps'] = base_params.get('fps', 20)
            params['colors'] = base_params.get('colors', 256)
            params['lossy'] = 20
        
        # Balanced dithering
        params['dither'] = 'floyd_steinberg' if params['colors'] > 192 else 'bayer'
        
        return params
    
    def get_parameter_sequence(self, target_size_mb: float, base_params: Dict[str, Any],
                              gif_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get sequence with balanced approach"""
        quality_stages = self.config.get_quality_stages()
        sequence = []
        
        # Use middle quality stages for balance
        mid_start = len(quality_stages) // 2 - 1
        for stage in quality_stages[mid_start:mid_start + 3]:
            params = base_params.copy()
            params.update({
                'width': int(gif_info.get('width', 360) * stage.get('resolution_multiplier', 0.9)),
                'fps': int(base_params.get('fps', 20) * stage.get('fps_multiplier', 0.9)),
                'colors': stage.get('colors', 192),
                'dither': stage.get('dither', 'floyd_steinberg'),
                'lossy': stage.get('lossy', 40)
            })
            sequence.append(params)
        
        return sequence


class AdaptiveStrategy(OptimizationStrategy):
    """Strategy that adapts based on content analysis"""
    
    def __init__(self, config_helper, temp_dir: str, shutdown_checker=None, analysis_results: Optional[Dict[str, Any]] = None):
        """
        Initialize adaptive strategy with analysis results.
        
        Args:
            analysis_results: Optional pre-computed analysis results
        """
        super().__init__(config_helper, temp_dir, shutdown_checker)
        self.analysis_results = analysis_results or {}
    
    def select_parameters(self, current_size_mb: float, target_size_mb: float,
                         base_params: Dict[str, Any], gif_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Select parameters based on content analysis.
        """
        params = base_params.copy()
        size_ratio = current_size_mb / target_size_mb if target_size_mb > 0 else 1.0
        
        # Use analysis results if available
        motion_level = self.analysis_results.get('motion_level', 'medium')
        complexity_level = self.analysis_results.get('complexity_level', 'medium')
        
        # Adjust based on motion
        if motion_level == 'low':
            # Low motion - can reduce FPS more
            fps_mult = 0.85
        elif motion_level == 'high':
            # High motion - preserve FPS
            fps_mult = 0.95
        else:
            fps_mult = 0.9
        
        # Adjust based on complexity
        if complexity_level == 'low':
            # Low complexity - can reduce colors more
            color_mult = 0.8
            dither = 'bayer'
        elif complexity_level == 'high':
            # High complexity - preserve colors
            color_mult = 0.95
            dither = 'floyd_steinberg'
        else:
            color_mult = 0.9
            dither = 'floyd_steinberg'
        
        # Apply size-based adjustments
        if size_ratio > 1.5:
            params['width'] = max(int(gif_info.get('width', 360) * 0.85), 360)
            params['fps'] = max(int(base_params.get('fps', 20) * fps_mult * 0.9), 15)
            params['colors'] = max(int(base_params.get('colors', 256) * color_mult * 0.9), 128)
            params['lossy'] = 60
        elif size_ratio > 1.2:
            params['width'] = max(int(gif_info.get('width', 360) * 0.92), 360)
            params['fps'] = max(int(base_params.get('fps', 20) * fps_mult), 15)
            params['colors'] = max(int(base_params.get('colors', 256) * color_mult), 192)
            params['lossy'] = 40
        else:
            params['width'] = max(int(gif_info.get('width', 360) * 0.96), 360)
            params['fps'] = max(int(base_params.get('fps', 20) * fps_mult), 15)
            params['colors'] = max(int(base_params.get('colors', 256) * color_mult), 224)
            params['lossy'] = 20
        
        params['dither'] = dither
        
        return params
    
    def get_parameter_sequence(self, target_size_mb: float, base_params: Dict[str, Any],
                              gif_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Get sequence based on content analysis"""
        # Use adaptive parameters
        sequence = []
        
        # Try multiple parameter sets based on analysis
        motion_level = self.analysis_results.get('motion_level', 'medium')
        complexity_level = self.analysis_results.get('complexity_level', 'medium')
        
        # Generate parameter sets
        for size_mult in [0.95, 0.90, 0.85, 0.80]:
            params = base_params.copy()
            
            # Adjust FPS based on motion
            if motion_level == 'low':
                fps_mult = 0.85
            elif motion_level == 'high':
                fps_mult = 0.95
            else:
                fps_mult = 0.9
            
            # Adjust colors based on complexity
            if complexity_level == 'low':
                color_mult = 0.8
            elif complexity_level == 'high':
                color_mult = 0.95
            else:
                color_mult = 0.9
            
            params.update({
                'width': max(int(gif_info.get('width', 360) * size_mult), 360),
                'fps': max(int(base_params.get('fps', 20) * fps_mult), 15),
                'colors': max(int(base_params.get('colors', 256) * color_mult), 128),
                'dither': 'floyd_steinberg' if complexity_level == 'high' else 'bayer',
                'lossy': int(60 * (1.0 - size_mult))
            })
            sequence.append(params)
        
        return sequence


def get_strategy(strategy_name: str, config_helper, temp_dir: str, 
                shutdown_checker=None, analysis_results: Optional[Dict[str, Any]] = None) -> OptimizationStrategy:
    """
    Factory function to get optimization strategy.
    
    Args:
        strategy_name: Name of strategy ('quality', 'size', 'balanced', 'adaptive')
        config_helper: GifConfigHelper instance
        temp_dir: Temporary directory path
        shutdown_checker: Optional shutdown checker callback
        analysis_results: Optional analysis results for adaptive strategy
    
    Returns:
        OptimizationStrategy instance
    """
    strategies = {
        'quality': QualityFirstStrategy,
        'size': SizeFirstStrategy,
        'balanced': BalancedStrategy,
        'adaptive': AdaptiveStrategy
    }
    
    strategy_class = strategies.get(strategy_name.lower(), BalancedStrategy)
    
    if strategy_name.lower() == 'adaptive':
        return strategy_class(config_helper, temp_dir, shutdown_checker, analysis_results)
    else:
        return strategy_class(config_helper, temp_dir, shutdown_checker)

