
from dataclasses import dataclass
from enum import Enum
from typing import Dict, NamedTuple, Optional

class ProcessingStatus(Enum):
    SUCCESS = "success"
    DIMENSION_ERROR = "dimension_error"
    PALETTE_ERROR = "palette_error"
    CONVERSION_ERROR = "conversion_error"
    OPTIMIZATION_ERROR = "optimization_error"
    FILE_ERROR = "file_error"
    SIZE_THRESHOLD_EXCEEDED = "size_threshold_exceeded"


@dataclass
class OptimizationConfig:
    scale_factor: float
    colors: int
    lossy_value: int
    dither_mode: str = 'bayer'
    bayer_scale: int = 3


# @dataclass
# class OptimizationResult:
#     fps: int
#     size: float
#     path: Optional[str]
#     status: ProcessingStatus
#     message: str = ""
#     settings: Optional[OptimizationConfig] = None

class OptimizationResult(NamedTuple):
    """Enhanced result tracking with size and quality metrics"""
    size: float
    target_met: bool
    settings: Dict
    quality_score: float
    warning: Optional[str] = None
    
# Add new quality control constants
QUALITY_THRESHOLDS = {
    'min_scale_factor': 0.2,  # Never scale below 20% of original size
    'min_colors': 32,         # Minimum allowable colors
    'max_lossy': 100,         # Maximum lossy compression value
    'min_dimension': 64,      # Minimum dimension in pixels
    'min_fps': 5,            # Minimum frames per second
    'acceptable_overshoot': 1.1,  # Allow 10% over target size
}

# Add optimization strategy constants
OPTIMIZATION_STRATEGIES = {
    'quality_steps': [
        # format: (scale_factor, colors, lossy, dither_mode)
        (1.0, 256, 20, 'floyd_steinberg'),   # High quality
        (0.9, 192, 30, 'floyd_steinberg'),   # Good quality
        (0.8, 128, 40, 'sierra2'),           # Balanced
        (0.7, 96, 60, 'bayer'),              # Reduced quality
        (0.5, 64, 80, 'bayer'),              # Low quality
        (0.3, 48, 90, 'bayer'),              # Very low quality
        (0.2, 32, 100, 'bayer')              # Minimum quality
    ],
    'frame_reduction_steps': [1, 2, 3, 4],   # Frame skipping steps
    'temporal_strategies': ['background', 'previous', 'none']
}

class OptimizationController:
    """Controls optimization process with quality-based progression"""

    def __init__(self, target_size_mb: float):
        self.target_size_mb = target_size_mb
        self.best_result = None
        self.quality_score = float('inf')
        self.attempts = []
        self.min_quality_threshold = 0.15  # Minimum acceptable quality score

    def evaluate_result(self, result_size: float, settings: Dict) -> OptimizationResult:
        """Evaluate optimization result with quality metrics"""
        quality_score = self._calculate_quality_score(settings)
        target_met = result_size <= self.target_size_mb

        # Track progression
        self.attempts.append({
            'size': result_size,
            'quality': quality_score,
            'settings': settings.copy()
        })

        warning = None
        if quality_score < self.min_quality_threshold:
            warning = "Quality threshold reached - further optimization may significantly degrade quality"

        result = OptimizationResult(
            size=result_size,
            target_met=target_met,
            settings=settings,
            quality_score=quality_score,
            warning=warning
        )

        # Track best result considering both size and quality
        if not self.best_result or self._is_better_result(result, self.best_result):
            self.best_result = result

        return result

    def _is_better_result(self, new: OptimizationResult, current: OptimizationResult) -> bool:
        """Determine if new result is better than current best"""
        # If one meets target and other doesn't, prefer the one meeting target
        if new.target_met != current.target_met:
            return new.target_met

        if new.target_met and current.target_met:
            # Both meet target, prefer higher quality
            return new.quality_score > current.quality_score
        else:
            # Neither meets target, prefer better size/quality balance
            new_score = self._calculate_balance_score(new)
            current_score = self._calculate_balance_score(current)
            return new_score > current_score

    def _calculate_balance_score(self, result: OptimizationResult) -> float:
        """Calculate balanced score between size and quality"""
        size_ratio = self.target_size_mb / result.size
        return (size_ratio * 0.7) + (result.quality_score * 0.3)

    def get_next_settings(self, current_result: OptimizationResult) -> Optional[Dict]:
        """Get next optimization settings based on results history"""
        current_size = current_result.size
        size_ratio = current_size / self.target_size_mb

        # If we're close to target, make small adjustments
        if 1.0 < size_ratio < 1.2:
            return self._fine_tune_settings(current_result.settings)

        # Analyze optimization trend
        trend = self._analyze_optimization_trend()

        # Use trend to determine next settings
        return self._get_progressive_settings(current_result.settings, size_ratio, trend)

    def _analyze_optimization_trend(self) -> Dict:
        """Analyze the trend of previous optimization attempts"""
        if len(self.attempts) < 2:
            return {'size_improvement': 1.0, 'quality_loss': 0.0}

        recent_attempts = self.attempts[-3:]  # Look at last 3 attempts

        size_improvements = []
        quality_losses = []

        for i in range(1, len(recent_attempts)):
            prev = recent_attempts[i-1]
            curr = recent_attempts[i]

            size_improvement = (prev['size'] - curr['size']) / prev['size']
            quality_loss = (prev['quality'] -
                            curr['quality']) / prev['quality']

            size_improvements.append(size_improvement)
            quality_losses.append(quality_loss)

        return {
            'size_improvement': sum(size_improvements) / len(size_improvements) if size_improvements else 1.0,
            'quality_loss': sum(quality_losses) / len(quality_losses) if quality_losses else 0.0
        }

    def _get_progressive_settings(self, current: Dict, size_ratio: float, trend: Dict) -> Dict:
        """Get next settings based on optimization trend"""
        new_settings = current.copy()

        # Determine aggression level based on trend
        if trend['size_improvement'] < 0.1 and trend['quality_loss'] > 0.2:
            # Poor results, be more aggressive
            reduction_factor = 0.8
        else:
            # Good progress, be more conservative
            reduction_factor = 0.9

        # Scale factor adjustment
        if current['scale_factor'] > 0.2:
            new_settings['scale_factor'] = max(0.2,
                                               current['scale_factor'] * (reduction_factor ** (size_ratio - 1)))

        # Color reduction
        if current['colors'] > 32:
            color_reduction = int(current['colors'] * reduction_factor)
            new_settings['colors'] = max(
                32, color_reduction - (color_reduction % 8))

        # Lossy value adjustment
        if current['lossy_value'] < 100:
            new_settings['lossy_value'] = min(100,
                                              int(current['lossy_value'] + (100 - current['lossy_value']) * 0.3))

        # Change dithering method if needed
        if trend['quality_loss'] > 0.3:
            new_settings['dither_mode'] = 'bayer'

        return new_settings

    def should_continue(self, current_settings: Dict) -> bool:
        """Determine if optimization should continue"""
        if len(self.attempts) > 0:
            latest_result = self.attempts[-1]

            # Stop if we're making minimal progress
            if len(self.attempts) > 3:
                last_three = self.attempts[-3:]
                size_changes = [abs(a['size'] - b['size']) / a['size']
                                for a, b in zip(last_three[:-1], last_three[1:])]

                if all(change < 0.05 for change in size_changes):  # Less than 5% change
                    return False

            # Stop if quality would become too low
            if latest_result['quality'] < self.min_quality_threshold:
                return False

        # Check minimum thresholds
        if (current_settings['scale_factor'] <= 0.2 or
                current_settings['colors'] <= 32):
            return False

        return True

    # ... rest of the existing OptimizationController code ...
