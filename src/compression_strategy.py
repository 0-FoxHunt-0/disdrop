"""
Smart Compression Strategy Module
Implements intelligent compression parameter selection with FPS constraint validation
and alternative strategies when FPS reduction is insufficient.
"""

import logging
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class CompressionStrategy(Enum):
    """Available compression strategies"""
    QUALITY_FIRST = "quality_first"
    BALANCED = "balanced"
    SIZE_FIRST = "size_first"
    AGGRESSIVE = "aggressive"


class ImpactLevel(Enum):
    """FPS reduction impact levels"""
    MINIMAL = "minimal"
    MODERATE = "moderate"
    SIGNIFICANT = "significant"
    SEVERE = "severe"


@dataclass
class CompressionConstraints:
    """Configuration constraints for compression"""
    min_fps: float
    max_file_size_mb: float
    min_resolution: Tuple[int, int]
    fps_reduction_steps: List[float]
    prefer_quality_over_fps: bool = True


@dataclass
class CompressionParams:
    """Compression parameters"""
    target_fps: float
    target_bitrate: int
    resolution: Tuple[int, int]
    quality_factor: float
    fps_reduction_applied: bool = False
    alternative_strategy_used: bool = False
    strategy_reason: str = ""


@dataclass
class FPSImpactAssessment:
    """Assessment of FPS reduction impact"""
    original_fps: float
    target_fps: float
    reduction_percent: float
    impact_level: ImpactLevel
    quality_impact_description: str
    recommendation: str
    respects_minimum: bool


@dataclass
class AlternativeStrategy:
    """Alternative compression strategy"""
    strategy_type: str
    description: str
    parameters: Dict[str, Any]
    expected_impact: str
    feasible: bool
    reason: str


class SmartCompressionStrategy:
    """
    Smart compression strategy engine that respects FPS constraints
    and implements alternative strategies when needed.
    """
    
    def __init__(self, config_manager):
        self.config = config_manager
        self.logger = logging.getLogger(__name__)
    
    def get_compression_constraints(self) -> CompressionConstraints:
        """Get current compression constraints from configuration"""
        return CompressionConstraints(
            min_fps=self.config.get('video_compression.bitrate_validation.min_fps', 20),
            max_file_size_mb=self.config.get('video_compression.max_file_size_mb', 10),
            min_resolution=(
                self.config.get('video_compression.bitrate_validation.min_resolution.width', 320),
                self.config.get('video_compression.bitrate_validation.min_resolution.height', 180)
            ),
            fps_reduction_steps=self.config.get('video_compression.bitrate_validation.fps_reduction_steps', [0.8, 0.6, 0.5]),
            prefer_quality_over_fps=self.config.get('video_compression.prefer_quality_over_fps', True)
        )
    
    def validate_fps_constraint(self, target_fps: float, constraints: CompressionConstraints) -> bool:
        """Validate that target FPS meets configuration constraints"""
        return target_fps >= constraints.min_fps
    
    def assess_fps_reduction_impact(self, original_fps: float, target_fps: float, 
                                  constraints: CompressionConstraints) -> FPSImpactAssessment:
        """Assess the impact of FPS reduction on video quality"""
        reduction_percent = ((original_fps - target_fps) / original_fps) * 100 if original_fps > 0 else 0
        
        # Determine impact level
        if reduction_percent <= 20:
            impact_level = ImpactLevel.MINIMAL
            quality_impact = "negligible quality impact expected"
            recommendation = "acceptable reduction"
        elif reduction_percent <= 40:
            impact_level = ImpactLevel.MODERATE
            quality_impact = "slight quality impact, should maintain smoothness"
            recommendation = "consider alternative strategies if possible"
        elif reduction_percent <= 60:
            impact_level = ImpactLevel.SIGNIFICANT
            quality_impact = "noticeable quality impact, may affect smoothness"
            recommendation = "strongly consider alternative strategies"
        else:
            impact_level = ImpactLevel.SEVERE
            quality_impact = "substantial quality impact, choppy playback likely"
            recommendation = "use alternative strategies or increase target size"
        
        respects_minimum = target_fps >= constraints.min_fps
        
        return FPSImpactAssessment(
            original_fps=original_fps,
            target_fps=target_fps,
            reduction_percent=reduction_percent,
            impact_level=impact_level,
            quality_impact_description=quality_impact,
            recommendation=recommendation,
            respects_minimum=respects_minimum
        )
    
    def select_compression_parameters(self, video_info: Dict[str, Any], 
                                    target_size_mb: float,
                                    strategy: CompressionStrategy = CompressionStrategy.BALANCED) -> CompressionParams:
        """
        Select optimal compression parameters respecting FPS constraints
        and preferring quality over aggressive FPS reduction
        """
        constraints = self.get_compression_constraints()
        original_fps = video_info.get('fps', 30.0)
        motion_level = video_info.get('motion_level', 'medium')
        duration = video_info.get('duration', 0)
        
        self.logger.info(f"=== SMART COMPRESSION PARAMETER SELECTION ===")
        self.logger.info(f"Input: {original_fps:.1f}fps, motion={motion_level}, duration={duration:.1f}s")
        self.logger.info(f"Target: {target_size_mb:.2f}MB, strategy={strategy.value}")
        self.logger.info(f"Constraints: min_fps={constraints.min_fps}, prefer_quality={constraints.prefer_quality_over_fps}")
        
        # Start with motion-based FPS selection
        if motion_level == 'high':
            base_fps = min(original_fps, 60)  # Preserve high FPS for motion
        elif motion_level == 'low':
            base_fps = 24  # Low motion can use cinematic 24fps
        else:
            base_fps = 30  # Standard for medium motion
        
        # Apply strategy-specific adjustments
        if strategy == CompressionStrategy.QUALITY_FIRST:
            # Minimize FPS reduction, prefer other parameters
            target_fps = max(base_fps, constraints.min_fps)
            strategy_reason = "Quality-first strategy: preserving FPS, will adjust bitrate/resolution instead"
        elif strategy == CompressionStrategy.SIZE_FIRST:
            # More aggressive FPS reduction allowed
            target_fps = self._apply_fps_reduction_steps(base_fps, constraints)
            strategy_reason = "Size-first strategy: applying FPS reduction to meet target size"
        else:  # BALANCED or AGGRESSIVE
            # Balanced approach with constraint validation
            target_fps = self._calculate_balanced_fps(base_fps, constraints, video_info, target_size_mb)
            strategy_reason = "Balanced strategy: optimizing FPS while respecting constraints"
        
        # Validate against constraints
        if not self.validate_fps_constraint(target_fps, constraints):
            self.logger.warning(f"Target FPS {target_fps:.1f} below minimum {constraints.min_fps}, clamping")
            target_fps = constraints.min_fps
            strategy_reason += " (clamped to minimum FPS)"
        
        # Assess impact
        impact = self.assess_fps_reduction_impact(original_fps, target_fps, constraints)
        
        # Calculate other parameters
        target_bitrate = self._calculate_target_bitrate(video_info, target_size_mb, target_fps)
        resolution = self._calculate_target_resolution(video_info, target_size_mb, target_fps)
        quality_factor = self._calculate_quality_factor(strategy, impact)
        
        params = CompressionParams(
            target_fps=target_fps,
            target_bitrate=target_bitrate,
            resolution=resolution,
            quality_factor=quality_factor,
            fps_reduction_applied=(target_fps < original_fps),
            strategy_reason=strategy_reason
        )
        
        self.logger.info(f"Selected parameters: {target_fps:.1f}fps, {target_bitrate}kbps, {resolution[0]}x{resolution[1]}")
        self.logger.info(f"FPS impact: {impact.impact_level.value} ({impact.reduction_percent:.1f}% reduction)")
        self.logger.info(f"Recommendation: {impact.recommendation}")
        
        return params
    
    def _apply_fps_reduction_steps(self, base_fps: float, constraints: CompressionConstraints) -> float:
        """Apply configured FPS reduction steps"""
        for step in constraints.fps_reduction_steps:
            reduced_fps = base_fps * step
            if reduced_fps >= constraints.min_fps:
                return reduced_fps
        return constraints.min_fps
    
    def _calculate_balanced_fps(self, base_fps: float, constraints: CompressionConstraints,
                              video_info: Dict[str, Any], target_size_mb: float) -> float:
        """Calculate balanced FPS considering compression difficulty"""
        original_size_mb = video_info.get('size_bytes', 0) / (1024 * 1024) if video_info.get('size_bytes') else 0
        compression_ratio = original_size_mb / max(target_size_mb, 0.1) if original_size_mb > 0 else 1.0
        
        # More aggressive FPS reduction for higher compression ratios
        if compression_ratio > 10:
            # Extreme compression needed
            target_fps = max(base_fps * 0.6, constraints.min_fps)
        elif compression_ratio > 5:
            # High compression needed
            target_fps = max(base_fps * 0.8, constraints.min_fps)
        else:
            # Moderate compression
            target_fps = max(base_fps * 0.9, constraints.min_fps)
        
        return target_fps
    
    def _calculate_target_bitrate(self, video_info: Dict[str, Any], target_size_mb: float, fps: float) -> int:
        """Calculate target bitrate based on size constraints"""
        duration = video_info.get('duration', 0)
        audio_bitrate = 96  # Assume 96kbps audio
        
        total_bits = target_size_mb * 8 * 1024 * 1024
        video_bits = total_bits - (audio_bitrate * 1000 * duration)
        video_bitrate = max(int(video_bits / duration / 1000), 64)  # Minimum 64kbps
        
        return video_bitrate
    
    def _calculate_target_resolution(self, video_info: Dict[str, Any], target_size_mb: float, fps: float) -> Tuple[int, int]:
        """Calculate target resolution based on constraints"""
        original_width = video_info.get('width', 1920)
        original_height = video_info.get('height', 1080)
        
        # For now, return original resolution - this can be enhanced later
        # with resolution reduction strategies
        return (original_width, original_height)
    
    def _calculate_quality_factor(self, strategy: CompressionStrategy, impact: FPSImpactAssessment) -> float:
        """Calculate quality factor based on strategy and FPS impact"""
        base_quality = 0.8  # Base quality factor
        
        if strategy == CompressionStrategy.QUALITY_FIRST:
            return min(base_quality + 0.1, 1.0)
        elif strategy == CompressionStrategy.SIZE_FIRST:
            return max(base_quality - 0.2, 0.3)
        elif impact.impact_level == ImpactLevel.SEVERE:
            # Compensate for severe FPS reduction with better quality
            return min(base_quality + 0.15, 1.0)
        else:
            return base_quality
    
    def get_alternative_strategies(self, video_info: Dict[str, Any], target_size_mb: float,
                                 current_params: CompressionParams) -> List[AlternativeStrategy]:
        """
        Generate alternative compression strategies when FPS reduction is insufficient
        """
        alternatives = []
        constraints = self.get_compression_constraints()
        
        # Strategy 1: Resolution reduction
        if current_params.fps_reduction_applied:
            resolution_alt = self._create_resolution_reduction_strategy(
                video_info, target_size_mb, current_params
            )
            alternatives.append(resolution_alt)
        
        # Strategy 2: Quality parameter adjustment
        quality_alt = self._create_quality_adjustment_strategy(
            video_info, target_size_mb, current_params
        )
        alternatives.append(quality_alt)
        
        # Strategy 3: Bitrate optimization
        bitrate_alt = self._create_bitrate_optimization_strategy(
            video_info, target_size_mb, current_params
        )
        alternatives.append(bitrate_alt)
        
        # Strategy 4: Segmentation (last resort)
        if target_size_mb < video_info.get('size_bytes', 0) / (1024 * 1024) / 3:  # If very aggressive compression needed
            segmentation_alt = self._create_segmentation_strategy(
                video_info, target_size_mb, current_params
            )
            alternatives.append(segmentation_alt)
        
        # Filter to only feasible alternatives
        feasible_alternatives = [alt for alt in alternatives if alt.feasible]
        
        self.logger.info(f"Generated {len(feasible_alternatives)} feasible alternative strategies")
        for alt in feasible_alternatives:
            self.logger.info(f"  - {alt.strategy_type}: {alt.description}")
        
        return feasible_alternatives
    
    def _create_resolution_reduction_strategy(self, video_info: Dict[str, Any], 
                                            target_size_mb: float, 
                                            current_params: CompressionParams) -> AlternativeStrategy:
        """Create resolution reduction alternative strategy"""
        original_width = video_info.get('width', 1920)
        original_height = video_info.get('height', 1080)
        constraints = self.get_compression_constraints()
        
        # Calculate reduced resolution (e.g., 80% of original)
        new_width = int(original_width * 0.8)
        new_height = int(original_height * 0.8)
        
        # Ensure it meets minimum constraints
        min_width, min_height = constraints.min_resolution
        feasible = new_width >= min_width and new_height >= min_height
        
        if not feasible:
            new_width, new_height = min_width, min_height
            reason = f"Reduced to minimum resolution {min_width}x{min_height}"
        else:
            reason = f"Reduced to {new_width}x{new_height} (80% of original)"
        
        return AlternativeStrategy(
            strategy_type="resolution_reduction",
            description=f"Reduce resolution to {new_width}x{new_height} instead of aggressive FPS reduction",
            parameters={
                'width': new_width,
                'height': new_height,
                'fps': video_info.get('fps', 30.0)  # Restore original FPS
            },
            expected_impact="Maintains smooth motion at lower resolution",
            feasible=feasible,
            reason=reason
        )
    
    def _create_quality_adjustment_strategy(self, video_info: Dict[str, Any],
                                          target_size_mb: float,
                                          current_params: CompressionParams) -> AlternativeStrategy:
        """Create quality parameter adjustment strategy"""
        return AlternativeStrategy(
            strategy_type="quality_adjustment",
            description="Adjust quality parameters (CRF, preset) instead of FPS reduction",
            parameters={
                'crf': 28,  # Higher CRF for smaller size
                'preset': 'slower',  # Slower preset for better compression
                'fps': video_info.get('fps', 30.0)  # Restore original FPS
            },
            expected_impact="Better compression efficiency with preserved motion smoothness",
            feasible=True,
            reason="Quality adjustment is always feasible"
        )
    
    def _create_bitrate_optimization_strategy(self, video_info: Dict[str, Any],
                                            target_size_mb: float,
                                            current_params: CompressionParams) -> AlternativeStrategy:
        """Create bitrate optimization strategy"""
        # Calculate more aggressive bitrate reduction
        duration = video_info.get('duration', 0)
        audio_bitrate = 64  # Reduce audio bitrate
        
        total_bits = target_size_mb * 8 * 1024 * 1024
        video_bits = total_bits - (audio_bitrate * 1000 * duration)
        optimized_bitrate = max(int(video_bits / duration / 1000), 32)
        
        return AlternativeStrategy(
            strategy_type="bitrate_optimization",
            description=f"Optimize bitrate allocation: {optimized_bitrate}kbps video + {audio_bitrate}kbps audio",
            parameters={
                'video_bitrate': optimized_bitrate,
                'audio_bitrate': audio_bitrate,
                'fps': video_info.get('fps', 30.0)  # Restore original FPS
            },
            expected_impact="More efficient bitrate allocation preserving motion quality",
            feasible=optimized_bitrate >= 32,
            reason=f"Optimized bitrate: {optimized_bitrate}kbps"
        )
    
    def _create_segmentation_strategy(self, video_info: Dict[str, Any],
                                    target_size_mb: float,
                                    current_params: CompressionParams) -> AlternativeStrategy:
        """Create video segmentation strategy"""
        duration = video_info.get('duration', 0)
        
        # Calculate number of segments needed
        segment_duration = 120  # 2 minutes per segment
        num_segments = max(2, int(duration / segment_duration))
        
        return AlternativeStrategy(
            strategy_type="segmentation",
            description=f"Split video into {num_segments} segments of ~{segment_duration}s each",
            parameters={
                'num_segments': num_segments,
                'segment_duration': segment_duration,
                'fps': video_info.get('fps', 30.0)  # Restore original FPS
            },
            expected_impact="Maintains quality and motion smoothness by splitting into smaller files",
            feasible=duration > 60,  # Only feasible for videos longer than 1 minute
            reason=f"Segmentation into {num_segments} parts"
        )
    
    def notify_aggressive_compression_required(self, video_info: Dict[str, Any], 
                                             target_size_mb: float,
                                             alternatives: List[AlternativeStrategy]) -> None:
        """Notify user when aggressive compression is required"""
        original_size_mb = video_info.get('size_bytes', 0) / (1024 * 1024) if video_info.get('size_bytes') else 0
        compression_ratio = original_size_mb / max(target_size_mb, 0.1) if original_size_mb > 0 else 1.0
        
        self.logger.warning("=== AGGRESSIVE COMPRESSION REQUIRED ===")
        self.logger.warning(f"Target compression ratio: {compression_ratio:.1f}x")
        self.logger.warning(f"Original size: {original_size_mb:.2f}MB → Target: {target_size_mb:.2f}MB")
        
        if compression_ratio > 10:
            self.logger.warning("EXTREME compression required - significant quality loss expected")
        elif compression_ratio > 5:
            self.logger.warning("HIGH compression required - noticeable quality loss expected")
        else:
            self.logger.warning("MODERATE compression required - some quality loss expected")
        
        if alternatives:
            self.logger.warning("Consider these alternative strategies:")
            for alt in alternatives:
                self.logger.warning(f"  - {alt.strategy_type}: {alt.description}")
        else:
            self.logger.warning("No alternative strategies available - aggressive compression will be applied")
        
        self.logger.warning("=== END COMPRESSION WARNING ===")
    
    def validate_compression_parameters(self, params: CompressionParams, 
                                      constraints: CompressionConstraints) -> List[str]:
        """Validate compression parameters against configuration constraints"""
        issues = []
        
        # Validate FPS constraint
        if params.target_fps < constraints.min_fps:
            issues.append(f"Target FPS {params.target_fps:.1f} below minimum {constraints.min_fps}")
        
        # Validate resolution constraint
        min_width, min_height = constraints.min_resolution
        if params.resolution[0] < min_width:
            issues.append(f"Target width {params.resolution[0]} below minimum {min_width}")
        if params.resolution[1] < min_height:
            issues.append(f"Target height {params.resolution[1]} below minimum {min_height}")
        
        # Validate bitrate
        if params.target_bitrate < 32:
            issues.append(f"Target bitrate {params.target_bitrate}kbps very low, may cause encoding issues")
        
        return issues