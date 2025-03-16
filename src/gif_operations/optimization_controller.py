from dataclasses import dataclass
from enum import Enum
from typing import Dict, NamedTuple, Optional, List, Tuple, Any
import logging
import math


class ProcessingStatus(Enum):
    SUCCESS = "success"
    DIMENSION_ERROR = "dimension_error"
    PALETTE_ERROR = "palette_error"
    CONVERSION_ERROR = "conversion_error"
    OPTIMIZATION_ERROR = "optimization_error"
    FILE_ERROR = "file_error"
    SIZE_THRESHOLD_EXCEEDED = "size_threshold_exceeded"
    QUALITY_THRESHOLD_REACHED = "quality_threshold_reached"


@dataclass
class OptimizationConfig:
    """Enhanced optimization configuration with more parameters."""
    scale_factor: float
    colors: int
    lossy_value: int
    dither_mode: str = 'bayer'
    bayer_scale: int = 3
    frame_skip: int = 1
    palette_stats_mode: str = 'diff'
    use_temporal_dithering: bool = False
    use_complex_filtering: bool = False


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
    processing_time: float = 0.0


# Add new quality control constants with enhanced defaults
QUALITY_THRESHOLDS = {
    'min_scale_factor': 0.15,      # Never scale below 15% of original size
    'min_colors': 32,              # Minimum allowable colors
    'max_lossy': 120,              # Maximum lossy compression value
    'min_dimension': 64,           # Minimum dimension in pixels
    'min_fps': 5,                  # Minimum frames per second
    'acceptable_overshoot': 1,     # Allow no overshoot
    'max_frame_skip': 4,           # Maximum frame skipping
    'fps_reduction_threshold': 15  # Don't reduce fps below this for videos with motion
}

# Add optimization strategy constants with more options
OPTIMIZATION_STRATEGIES = {
    'quality_steps': [
        # format: (scale_factor, colors, lossy, dither_mode, palette_stats, use_temporal)
        (1.0, 256, 20, 'floyd_steinberg', 'diff', True),   # High quality
        (0.9, 192, 30, 'floyd_steinberg', 'diff', True),   # Good quality
        (0.8, 128, 40, 'sierra2', 'diff', False),          # Balanced
        (0.7, 96, 60, 'bayer', 'full', False),             # Reduced quality
        (0.5, 64, 80, 'bayer', 'full', False),             # Low quality
        (0.3, 48, 90, 'bayer', 'full', False),             # Very low quality
        (0.2, 32, 120, 'bayer', 'full', False)             # Minimum quality
    ],
    # Frame skipping steps
    'frame_reduction_steps': [1, 2, 3, 4],
    # Dithering algorithms
    'dither_modes': ['floyd_steinberg', 'sierra2', 'bayer', 'none'],
    # Palette generation strategies
    'palette_stats_modes': ['diff', 'full', 'single'],
    # Bayer dithering scales
    'bayer_scales': [2, 3, 5]
}

# Content-aware optimization presets
CONTENT_PRESETS = {
    'animation': {  # For cartoon/anime style content
        'colors_priority': 'high',
        'frame_skip_priority': 'low',
        'dither_priority': 'high',
        'scale_priority': 'medium'
    },
    'real_world': {  # For photographic/movie content
        'colors_priority': 'medium',
        'frame_skip_priority': 'medium',
        'dither_priority': 'medium',
        'scale_priority': 'high'
    },
    'graphics': {  # For UI captures, games, etc.
        'colors_priority': 'high',
        'frame_skip_priority': 'medium',
        'dither_priority': 'low',
        'scale_priority': 'high'
    },
    'text_heavy': {  # For content with a lot of text
        'colors_priority': 'medium',
        'frame_skip_priority': 'high',
        'dither_priority': 'low',
        'scale_priority': 'medium'
    }
}


class OptimizationController:
    """Controls optimization process with quality-based progression and adaptive strategies"""

    def __init__(self, target_size_mb: float):
        self.target_size_mb = target_size_mb
        self.best_result = None
        self.quality_score = float('inf')
        self.attempts = []
        self.min_quality_threshold = 0.15  # Minimum acceptable quality score
        self.logger = logging.getLogger('gif_optimizer')
        self.content_type = 'real_world'  # Default content type
        self.initial_size = 0
        self.initial_dimensions = (0, 0)
        self.has_text = False
        self.has_motion = True
        self.aspect_ratio = 1.0

    def set_content_type(self, content_type: str):
        """Set the content type for optimized settings."""
        if content_type in CONTENT_PRESETS:
            self.content_type = content_type
            self.logger.info(f"Content type set to: {content_type}")
        else:
            self.logger.warning(
                f"Unknown content type: {content_type}, using 'real_world'")
            self.content_type = 'real_world'

    def set_source_properties(self, size_mb: float, dimensions: Tuple[int, int], has_text: bool = False, has_motion: bool = True):
        """Set source file properties for better optimization decisions."""
        self.initial_size = size_mb
        self.initial_dimensions = dimensions
        self.has_text = has_text
        self.has_motion = has_motion

        # Calculate aspect ratio
        if dimensions[1] > 0:
            self.aspect_ratio = dimensions[0] / dimensions[1]
        else:
            self.aspect_ratio = 1.0

        self.logger.debug(
            f"Source properties: {size_mb:.2f}MB, {dimensions}, text: {has_text}, motion: {has_motion}")

    def evaluate_result(self, result_size: float, settings: Dict, processing_time: float = 0.0) -> OptimizationResult:
        """Evaluate optimization result with quality metrics and performance data."""
        quality_score = self._calculate_quality_score(settings)
        target_met = result_size <= self.target_size_mb

        # Check if we're within acceptable overshoot
        if not target_met and result_size <= self.target_size_mb * QUALITY_THRESHOLDS['acceptable_overshoot']:
            target_met = True
            self.logger.debug(
                f"Accepting size within threshold: {result_size:.2f}MB <= {self.target_size_mb * QUALITY_THRESHOLDS['acceptable_overshoot']:.2f}MB")

        # Track progression
        self.attempts.append({
            'size': result_size,
            'quality': quality_score,
            'settings': settings.copy(),
            'processing_time': processing_time
        })

        warning = None
        if quality_score < self.min_quality_threshold:
            warning = "Quality threshold reached - further optimization may significantly degrade quality"

        result = OptimizationResult(
            size=result_size,
            target_met=target_met,
            settings=settings,
            quality_score=quality_score,
            warning=warning,
            processing_time=processing_time
        )

        # Store as best result if better than previous best
        if self.best_result is None or self._is_better_result(result, self.best_result):
            self.best_result = result
            self.logger.debug(
                f"New best result: {result_size:.2f}MB, quality: {quality_score:.2f}")

        return result

    def _is_better_result(self, new: OptimizationResult, current: OptimizationResult) -> bool:
        """Determine if the new result is better than the current best."""
        # If current doesn't meet target but new does, new is better
        if not current.target_met and new.target_met:
            return True

        # If both meet target, prefer higher quality
        if current.target_met and new.target_met:
            # If quality scores are similar, prefer smaller size
            if abs(new.quality_score - current.quality_score) < 0.05:
                return new.size < current.size
            return new.quality_score > current.quality_score

        # If neither meets target, prefer higher balanced score
        if not current.target_met and not new.target_met:
            current_balance = self._calculate_balance_score(current)
            new_balance = self._calculate_balance_score(new)
            return new_balance > current_balance

        # Current meets target but new doesn't
        return False

    def _calculate_balance_score(self, result: OptimizationResult) -> float:
        """Calculate a balanced score combining size and quality."""
        # How close to target (lower is better)
        size_ratio = result.size / self.target_size_mb

        # If significantly over target, heavily penalize
        if size_ratio > 2.0:
            size_score = 0.1
        else:
            # Size score decreases as size increases beyond target
            size_score = 1.0 if size_ratio <= 1.0 else 1.0 / size_ratio

        # Combined score - quality matters more when close to target
        balance = (size_score * 0.6) + (result.quality_score * 0.4)
        return balance

    def _calculate_quality_score(self, settings: Dict) -> float:
        """Calculate quality score based on settings and content type."""
        # Base factors that impact quality
        scale_factor = settings.get('scale_factor', 1.0)
        colors = settings.get('colors', 256)
        lossy = settings.get('lossy_value', 0)
        frame_skip = settings.get('frame_skip', 1)

        # Get content type priorities
        content_preset = CONTENT_PRESETS.get(
            self.content_type, CONTENT_PRESETS['real_world'])

        # Calculate individual quality components
        scale_quality = self._calculate_scale_quality(scale_factor)
        color_quality = self._calculate_color_quality(colors)
        lossy_quality = self._calculate_lossy_quality(lossy)
        temporal_quality = self._calculate_temporal_quality(frame_skip)

        # Apply content-specific weighting
        priority_weights = {
            'high': 1.5,
            'medium': 1.0,
            'low': 0.5
        }

        scale_weight = priority_weights[content_preset['scale_priority']]
        color_weight = priority_weights[content_preset['colors_priority']]
        frame_weight = priority_weights[content_preset['frame_skip_priority']]
        dither_weight = priority_weights[content_preset['dither_priority']]

        # Special case adjustments
        if self.has_text:
            # Text needs good resolution and color fidelity
            scale_weight *= 1.3
            color_weight *= 1.2

        if not self.has_motion:
            # Static content can afford frame skipping
            frame_weight *= 0.7

        if self.aspect_ratio > 2.0 or self.aspect_ratio < 0.5:
            # Extreme aspect ratios (very wide or tall) need special handling
            scale_weight *= 1.2

        # Calculate weighted score
        total_weight = scale_weight + color_weight + dither_weight + frame_weight
        weighted_score = (
            (scale_quality * scale_weight) +
            (color_quality * color_weight) +
            (lossy_quality * dither_weight) +
            (temporal_quality * frame_weight)
        ) / total_weight

        return max(0.0, min(1.0, weighted_score))  # Clamp between 0 and 1

    def _calculate_scale_quality(self, scale_factor: float) -> float:
        """Calculate quality impact of scaling."""
        # Exponential quality reduction as scale decreases
        # 1.0 = full quality, approaching 0 as scale approaches min_scale_factor
        min_scale = QUALITY_THRESHOLDS['min_scale_factor']
        if scale_factor <= min_scale:
            return 0.0

        # Higher exponent = steeper quality drop-off
        exponent = 1.5
        normalized = (scale_factor - min_scale) / (1.0 - min_scale)
        return math.pow(normalized, exponent)

    def _calculate_color_quality(self, colors: int) -> float:
        """Calculate quality impact of color reduction."""
        # Logarithmic quality reduction as colors decrease
        min_colors = QUALITY_THRESHOLDS['min_colors']
        max_colors = 256

        if colors <= min_colors:
            return 0.0
        if colors >= max_colors:
            return 1.0

        # Log scale feels more natural for color perception
        return math.log(colors / min_colors) / math.log(max_colors / min_colors)

    def _calculate_lossy_quality(self, lossy_value: int) -> float:
        """Calculate quality impact of lossy compression."""
        max_lossy = QUALITY_THRESHOLDS['max_lossy']

        if lossy_value >= max_lossy:
            return 0.0
        if lossy_value <= 0:
            return 1.0

        # Linear quality reduction - lossy directly impacts quality
        return 1.0 - (lossy_value / max_lossy)

    def _calculate_temporal_quality(self, frame_skip: int) -> float:
        """Calculate quality impact of frame skipping."""
        max_skip = QUALITY_THRESHOLDS['max_frame_skip']

        if frame_skip >= max_skip:
            return 0.0
        if frame_skip <= 1:
            return 1.0

        # Steeper quality reduction for higher frame skips
        # Frame skipping has more impact when content has motion
        exponent = 2.0 if self.has_motion else 1.2
        return math.pow(1.0 - ((frame_skip - 1) / (max_skip - 1)), exponent)

    def get_next_settings(self, current_result: OptimizationResult) -> Optional[Dict]:
        """Get next settings to try based on current result and optimization trends."""
        if not self.attempts:
            return self._get_initial_settings()

        # Extract current settings and result info
        current_settings = current_result.settings
        current_size = current_result.size
        size_ratio = current_size / self.target_size_mb

        # Analyze trends from previous attempts
        trend = self._analyze_optimization_trend()

        # If target met, we're done
        if current_result.target_met:
            self.logger.debug(
                "Target size met - no further optimization needed")
            return None

        # If quality threshold reached, stop
        if current_result.quality_score <= self.min_quality_threshold:
            self.logger.warning(
                "Quality threshold reached - stopping optimization")
            return None

        # Get progressive settings based on size ratio and trend analysis
        next_settings = self._get_progressive_settings(
            current_settings, size_ratio, trend)

        # Check if the next settings are worth trying
        if self._settings_too_similar(next_settings, [a['settings'] for a in self.attempts]):
            self.logger.debug(
                "Next settings too similar to previous attempts - trying alternate approach")
            next_settings = self._get_alternate_settings(
                current_settings, size_ratio)

        # If we still need significant size reduction, try more aggressive options
        if size_ratio > 2.0 and len(self.attempts) >= 3:
            self.logger.debug(
                f"Size still too large ({current_size:.2f}MB vs {self.target_size_mb:.2f}MB) - using aggressive optimization")
            next_settings = self._get_aggressive_settings(
                current_settings, size_ratio)

        return next_settings

    def _get_initial_settings(self) -> Dict:
        """Get initial optimization settings based on content type."""
        preset = CONTENT_PRESETS.get(
            self.content_type, CONTENT_PRESETS['real_world'])

        # Adjust initial scaling based on dimensions and target size
        scale_factor = 1.0
        if max(self.initial_dimensions) > 800:
            # Start with scaling for large images
            scale_factor = 0.8

        # Adjust initial colors based on content type
        colors = 256
        if self.content_type == 'graphics':
            colors = 192
        elif self.content_type == 'text_heavy':
            colors = 128

        # Determine dither mode based on content
        dither_mode = 'floyd_steinberg'
        if self.content_type == 'graphics' or self.content_type == 'text_heavy':
            dither_mode = 'bayer'

        # Determine initial lossy value based on size ratio
        size_ratio = self.initial_size / self.target_size_mb
        lossy_value = 0
        if size_ratio > 1.0:
            lossy_value = min(60, int(20 * size_ratio))

        # Determine frame skip based on content
        frame_skip = 1
        if not self.has_motion and size_ratio > 2.0:
            frame_skip = 2

        return {
            'scale_factor': scale_factor,
            'colors': colors,
            'lossy_value': lossy_value,
            'dither_mode': dither_mode,
            'bayer_scale': 3,
            'frame_skip': frame_skip,
            'palette_stats_mode': 'diff',
            'use_temporal_dithering': self.has_motion,
            'use_complex_filtering': False
        }

    def _settings_too_similar(self, settings: Dict, previous_settings: List[Dict]) -> bool:
        """Check if new settings are too similar to previously tried settings."""
        for prev in previous_settings:
            differences = 0

            # Check key parameters
            if abs(prev.get('scale_factor', 1.0) - settings.get('scale_factor', 1.0)) > 0.1:
                differences += 1
            if abs(prev.get('colors', 256) - settings.get('colors', 256)) > 32:
                differences += 1
            if abs(prev.get('lossy_value', 0) - settings.get('lossy_value', 0)) > 15:
                differences += 1
            if prev.get('frame_skip', 1) != settings.get('frame_skip', 1):
                differences += 1
            if prev.get('dither_mode', '') != settings.get('dither_mode', ''):
                differences += 1

            # If less than 2 significant differences, consider too similar
            if differences < 2:
                return True

        return False

    def _analyze_optimization_trend(self) -> Dict:
        """Analyze trend of previous optimization attempts."""
        if len(self.attempts) < 2:
            return {'size_elasticity': {}, 'quality_elasticity': {}, 'most_effective': None}

        # Calculate elasticity (sensitivity) for each parameter
        size_elasticity = {}
        quality_elasticity = {}

        # Compare each attempt with the previous one
        for i in range(1, len(self.attempts)):
            prev = self.attempts[i-1]
            curr = self.attempts[i]

            prev_settings = prev['settings']
            curr_settings = curr['settings']

            # Calculate relative changes
            size_change = (curr['size'] - prev['size']) / \
                prev['size'] if prev['size'] > 0 else 0
            quality_change = (curr['quality'] - prev['quality']) / \
                prev['quality'] if prev['quality'] > 0 else 0

            # Check scale factor impact
            if 'scale_factor' in prev_settings and 'scale_factor' in curr_settings:
                param_change = (
                    curr_settings['scale_factor'] - prev_settings['scale_factor']) / prev_settings['scale_factor']
                if abs(param_change) > 0.01:  # Only consider significant changes
                    size_elasticity['scale_factor'] = size_elasticity.get(
                        'scale_factor', []) + [size_change / param_change]
                    quality_elasticity['scale_factor'] = quality_elasticity.get(
                        'scale_factor', []) + [quality_change / param_change]

            # Check colors impact
            if 'colors' in prev_settings and 'colors' in curr_settings:
                param_change = (
                    curr_settings['colors'] - prev_settings['colors']) / prev_settings['colors']
                if abs(param_change) > 0.01:
                    size_elasticity['colors'] = size_elasticity.get(
                        'colors', []) + [size_change / param_change]
                    quality_elasticity['colors'] = quality_elasticity.get(
                        'colors', []) + [quality_change / param_change]

            # Check lossy impact
            if 'lossy_value' in prev_settings and 'lossy_value' in curr_settings:
                # Handle case where lossy was 0
                prev_lossy = max(1, prev_settings['lossy_value'])
                param_change = (
                    curr_settings['lossy_value'] - prev_settings['lossy_value']) / prev_lossy
                if abs(param_change) > 0.01:
                    size_elasticity['lossy_value'] = size_elasticity.get(
                        'lossy_value', []) + [size_change / param_change]
                    quality_elasticity['lossy_value'] = quality_elasticity.get(
                        'lossy_value', []) + [quality_change / param_change]

            # Check frame_skip impact
            if 'frame_skip' in prev_settings and 'frame_skip' in curr_settings:
                # Use absolute change for frame_skip
                param_change = curr_settings['frame_skip'] - \
                    prev_settings['frame_skip']
                if param_change != 0:
                    size_elasticity['frame_skip'] = size_elasticity.get(
                        'frame_skip', []) + [size_change / param_change]
                    quality_elasticity['frame_skip'] = quality_elasticity.get(
                        'frame_skip', []) + [quality_change / param_change]

        # Average the elasticities
        avg_size_elasticity = {k: sum(v)/len(v)
                               for k, v in size_elasticity.items() if v}
        avg_quality_elasticity = {k: sum(v)/len(v)
                                  for k, v in quality_elasticity.items() if v}

        # Determine most effective parameter (highest size reduction for lowest quality loss)
        effectiveness = {}
        for param in avg_size_elasticity:
            if param in avg_quality_elasticity:
                # More negative size elasticity is better
                # More positive (or less negative) quality elasticity is better
                effectiveness[param] = -avg_size_elasticity[param] / \
                    (1.0 - avg_quality_elasticity[param])

        most_effective = max(effectiveness.items(), key=lambda x: x[1])[
            0] if effectiveness else None

        return {
            'size_elasticity': avg_size_elasticity,
            'quality_elasticity': avg_quality_elasticity,
            'most_effective': most_effective
        }

    def _get_progressive_settings(self, current: Dict, size_ratio: float, trend: Dict) -> Dict:
        """Get progressive settings based on current settings, size ratio and trend analysis."""
        # Start with a copy of current settings
        next_settings = current.copy()

        # Calculate how much more reduction we need
        reduction_needed = size_ratio - 1.0

        # Use trend data to determine which parameter to adjust
        most_effective = trend.get('most_effective')

        if most_effective:
            # Adjust the most effective parameter more aggressively
            if most_effective == 'scale_factor':
                # Reduce scale more when we need more reduction
                scale_reduction = min(0.2, reduction_needed * 0.3)
                next_settings['scale_factor'] = max(
                    QUALITY_THRESHOLDS['min_scale_factor'],
                    current['scale_factor'] - scale_reduction
                )

            elif most_effective == 'colors':
                # Reduce colors exponentially based on reduction needed
                color_factor = 1.0 - min(0.5, reduction_needed * 0.3)
                next_settings['colors'] = max(
                    QUALITY_THRESHOLDS['min_colors'],
                    int(current['colors'] * color_factor)
                )

            elif most_effective == 'lossy_value':
                # Increase lossy value based on reduction needed
                lossy_increase = min(30, int(reduction_needed * 40))
                next_settings['lossy_value'] = min(
                    QUALITY_THRESHOLDS['max_lossy'],
                    current['lossy_value'] + lossy_increase
                )

            elif most_effective == 'frame_skip':
                # Increase frame skip if it's proven effective
                if current['frame_skip'] < QUALITY_THRESHOLDS['max_frame_skip']:
                    next_settings['frame_skip'] = current['frame_skip'] + 1
        else:
            # Without trend data, adjust multiple parameters in smaller increments

            # Scale adjustment
            if size_ratio > 2.0:
                # Large files need more aggressive scaling
                next_settings['scale_factor'] = max(
                    QUALITY_THRESHOLDS['min_scale_factor'],
                    current['scale_factor'] * 0.7
                )
            elif size_ratio > 1.5:
                next_settings['scale_factor'] = max(
                    QUALITY_THRESHOLDS['min_scale_factor'],
                    current['scale_factor'] * 0.8
                )
            elif size_ratio > 1.2:
                next_settings['scale_factor'] = max(
                    QUALITY_THRESHOLDS['min_scale_factor'],
                    current['scale_factor'] * 0.9
                )

            # Color adjustment
            if size_ratio > 2.0:
                next_settings['colors'] = max(
                    QUALITY_THRESHOLDS['min_colors'], int(current['colors'] * 0.5))
            elif size_ratio > 1.5:
                next_settings['colors'] = max(
                    QUALITY_THRESHOLDS['min_colors'], int(current['colors'] * 0.7))
            elif size_ratio > 1.2:
                next_settings['colors'] = max(
                    QUALITY_THRESHOLDS['min_colors'], int(current['colors'] * 0.8))

            # Lossy adjustment
            if size_ratio > 2.0:
                next_settings['lossy_value'] = min(
                    QUALITY_THRESHOLDS['max_lossy'], current['lossy_value'] + 40)
            elif size_ratio > 1.5:
                next_settings['lossy_value'] = min(
                    QUALITY_THRESHOLDS['max_lossy'], current['lossy_value'] + 25)
            elif size_ratio > 1.2:
                next_settings['lossy_value'] = min(
                    QUALITY_THRESHOLDS['max_lossy'], current['lossy_value'] + 15)

        # Try different dither mode if we've made several attempts
        if len(self.attempts) > 2:
            current_dither = current.get('dither_mode', 'bayer')
            dither_modes = OPTIMIZATION_STRATEGIES['dither_modes']
            current_index = dither_modes.index(
                current_dither) if current_dither in dither_modes else 0
            next_index = (current_index + 1) % len(dither_modes)
            next_settings['dither_mode'] = dither_modes[next_index]

        # For text-heavy content, prioritize resolution over colors
        if self.has_text and size_ratio < 3.0:
            # Restore some of the scale factor at the expense of colors
            next_settings['scale_factor'] = min(
                1.0, next_settings['scale_factor'] * 1.2)
            next_settings['colors'] = max(
                QUALITY_THRESHOLDS['min_colors'], int(next_settings['colors'] * 0.8))

        return next_settings

    def _get_alternate_settings(self, current: Dict, size_ratio: float) -> Dict:
        """Get alternate settings when normal progression isn't effective."""
        next_settings = current.copy()

        # Try a different approach - if we've been adjusting colors and scale, try frame skipping
        if current['frame_skip'] == 1 and self.has_motion and size_ratio > 1.3:
            next_settings['frame_skip'] = 2
            # Restore some quality in other areas
            next_settings['scale_factor'] = min(
                1.0, current['scale_factor'] * 1.2)
            next_settings['colors'] = min(256, int(current['colors'] * 1.2))

        # Try different palette stats mode
        current_mode = current.get('palette_stats_mode', 'diff')
        palette_modes = OPTIMIZATION_STRATEGIES['palette_stats_modes']
        current_index = palette_modes.index(
            current_mode) if current_mode in palette_modes else 0
        next_index = (current_index + 1) % len(palette_modes)
        next_settings['palette_stats_mode'] = palette_modes[next_index]

        # Try temporal dithering if we haven't
        next_settings['use_temporal_dithering'] = not current.get(
            'use_temporal_dithering', False)

        return next_settings

    def _get_aggressive_settings(self, current: Dict, size_ratio: float) -> Dict:
        """Get aggressive settings for significant size reduction."""
        # Start with predetermined aggressive settings
        aggressive = {
            'scale_factor': 0.3,
            'colors': 48,
            'lossy_value': 100,
            'dither_mode': 'bayer',
            'bayer_scale': 2,
            'frame_skip': min(3, QUALITY_THRESHOLDS['max_frame_skip']),
            'palette_stats_mode': 'full',
            'use_temporal_dithering': False,
            'use_complex_filtering': False
        }

        # Adjust based on content type
        if self.has_text:
            # Text needs better resolution
            aggressive['scale_factor'] = 0.5
            aggressive['colors'] = 32

        if not self.has_motion:
            # Without motion, we can skip more frames
            aggressive['frame_skip'] = QUALITY_THRESHOLDS['max_frame_skip']

        return aggressive

    def should_continue(self, current_settings: Dict) -> bool:
        """Determine if optimization should continue or if we've reached diminishing returns."""
        # Stop if we've hit minimum thresholds
        if current_settings.get('scale_factor', 1.0) <= QUALITY_THRESHOLDS['min_scale_factor']:
            self.logger.debug("Reached minimum scale factor threshold")
            return False

        if current_settings.get('colors', 256) <= QUALITY_THRESHOLDS['min_colors']:
            self.logger.debug("Reached minimum colors threshold")
            return False

        if current_settings.get('lossy_value', 0) >= QUALITY_THRESHOLDS['max_lossy']:
            self.logger.debug("Reached maximum lossy value threshold")
            return False

        # Stop after a reasonable number of attempts
        if len(self.attempts) >= 8:
            self.logger.debug(
                "Reached maximum number of optimization attempts")
            return False

        # Check for diminishing returns
        if len(self.attempts) >= 3:
            last_results = self.attempts[-3:]
            size_improvements = [
                abs(last_results[i]['size'] - last_results[i-1]
                    ['size']) / last_results[i-1]['size']
                for i in range(1, len(last_results))
            ]

            # If recent improvements are minimal, stop
            if all(improvement < 0.05 for improvement in size_improvements):
                self.logger.debug(
                    "Optimization showing diminishing returns - stopping")
                return False

        return True
