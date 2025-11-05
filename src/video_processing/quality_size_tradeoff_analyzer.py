"""
Quality-Size Trade-off Analyzer
Implements intelligent analysis and user notification for quality vs size decisions
"""

import math
from typing import Dict, Any, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from .content_analysis_engine import ContentType, ContentAnalysis
    from .content_aware_encoding_profiles import EncodingProfile, BitrateAllocation
except ImportError:
    from content_analysis_engine import ContentType, ContentAnalysis
    from content_aware_encoding_profiles import EncodingProfile, BitrateAllocation

logger = logging.getLogger(__name__)


class TradeoffStrategy(Enum):
    """Trade-off strategy options."""
    MAXIMIZE_QUALITY = "maximize_quality"
    BALANCED = "balanced"
    MINIMIZE_SIZE = "minimize_size"
    CUSTOM = "custom"


class QualityLevel(Enum):
    """Quality level classifications."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    UNACCEPTABLE = "unacceptable"


@dataclass
class QualitySizeOption:
    """A quality-size trade-off option."""
    option_id: str
    description: str
    
    # Size and quality metrics
    estimated_size_mb: float
    predicted_quality_score: float
    quality_level: QualityLevel
    
    # Encoding parameters
    video_bitrate_kbps: int
    audio_bitrate_kbps: int
    resolution: Tuple[int, int]
    fps: float
    codec: str
    preset: str
    
    # Trade-off analysis
    size_efficiency: float  # Quality per MB
    quality_loss_vs_best: float  # Quality loss compared to best option
    size_savings_vs_best: float  # Size savings compared to best quality
    
    # User-facing information
    pros: List[str]
    cons: List[str]
    recommended_for: List[str]
    
    # Technical details
    encoding_time_estimate: float  # Estimated encoding time in seconds
    compatibility_score: float  # Device/platform compatibility (0-1)
    
    
@dataclass
class TradeoffAnalysis:
    """Complete trade-off analysis results."""
    content_analysis: ContentAnalysis
    target_size_mb: float
    duration_seconds: float
    
    # Available options
    options: List[QualitySizeOption]
    recommended_option_id: str
    
    # Analysis insights
    size_constraint_severity: float  # How tight the size constraint is (0-1)
    quality_feasibility: float  # How feasible good quality is (0-1)
    complexity_challenge: float  # How challenging the content is to compress (0-1)
    
    # User notifications
    warnings: List[str]
    suggestions: List[str]
    technical_notes: List[str]
    
    # Alternative strategies
    alternative_approaches: List[Dict[str, Any]]


class QualitySizeTradeoffAnalyzer:
    """Analyzes quality vs size trade-offs and generates user-friendly options."""
    
    def __init__(self, config_manager):
        self.config = config_manager
        
        # Configuration
        self.min_acceptable_quality = self.config.get('tradeoff_analysis.min_acceptable_quality', 0.6)
        self.quality_thresholds = {
            QualityLevel.EXCELLENT: 0.9,
            QualityLevel.GOOD: 0.75,
            QualityLevel.ACCEPTABLE: 0.6,
            QualityLevel.POOR: 0.4,
            QualityLevel.UNACCEPTABLE: 0.0
        }
        
        # Encoding time estimates (relative to real-time)
        self.encoding_time_factors = {
            'ultrafast': 0.3,
            'superfast': 0.5,
            'veryfast': 0.8,
            'faster': 1.2,
            'fast': 1.8,
            'medium': 3.0,
            'slow': 5.0,
            'slower': 8.0,
            'veryslow': 12.0
        }
    
    def analyze_tradeoffs(self, content_analysis: ContentAnalysis, target_size_mb: float,
                         duration_seconds: float, original_resolution: Tuple[int, int],
                         strategy: TradeoffStrategy = TradeoffStrategy.BALANCED) -> TradeoffAnalysis:
        """
        Analyze quality-size trade-offs and generate options.
        
        Args:
            content_analysis: Content analysis results
            target_size_mb: Target file size in MB
            duration_seconds: Video duration in seconds
            original_resolution: Original video resolution (width, height)
            strategy: Preferred trade-off strategy
            
        Returns:
            Complete TradeoffAnalysis with options and recommendations
        """
        logger.info(f"Analyzing quality-size trade-offs for {target_size_mb:.1f}MB target")
        
        # Assess constraint severity
        size_constraint_severity = self._assess_size_constraint_severity(
            content_analysis, target_size_mb, duration_seconds, original_resolution
        )
        
        # Assess quality feasibility
        quality_feasibility = self._assess_quality_feasibility(
            content_analysis, target_size_mb, duration_seconds
        )
        
        # Assess complexity challenge
        complexity_challenge = self._assess_complexity_challenge(content_analysis)
        
        # Generate quality-size options
        options = self._generate_quality_size_options(
            content_analysis, target_size_mb, duration_seconds, original_resolution
        )
        
        # Select recommended option based on strategy
        recommended_option_id = self._select_recommended_option(options, strategy)
        
        # Generate user notifications
        warnings, suggestions, technical_notes = self._generate_user_notifications(
            content_analysis, options, size_constraint_severity, quality_feasibility
        )
        
        # Generate alternative approaches
        alternative_approaches = self._generate_alternative_approaches(
            content_analysis, target_size_mb, duration_seconds, size_constraint_severity
        )
        
        return TradeoffAnalysis(
            content_analysis=content_analysis,
            target_size_mb=target_size_mb,
            duration_seconds=duration_seconds,
            options=options,
            recommended_option_id=recommended_option_id,
            size_constraint_severity=size_constraint_severity,
            quality_feasibility=quality_feasibility,
            complexity_challenge=complexity_challenge,
            warnings=warnings,
            suggestions=suggestions,
            technical_notes=technical_notes,
            alternative_approaches=alternative_approaches
        )
    
    def _assess_size_constraint_severity(self, content_analysis: ContentAnalysis,
                                       target_size_mb: float, duration_seconds: float,
                                       original_resolution: Tuple[int, int]) -> float:
        """Assess how severe the size constraint is (0=easy, 1=very tight)."""
        
        # Calculate baseline bitrate needed for acceptable quality
        width, height = original_resolution
        pixels_per_second = width * height * 30  # Assume 30fps baseline
        
        # Minimum bits per pixel for acceptable quality based on content type
        min_bpp_map = {
            ContentType.ANIMATION: 0.015,
            ContentType.LIVE_ACTION: 0.025,
            ContentType.SCREEN_RECORDING: 0.008,
            ContentType.GAMING: 0.020,
            ContentType.MIXED: 0.022
        }
        
        min_bpp = min_bpp_map.get(content_analysis.content_type, 0.025)
        
        # Adjust for content complexity
        complexity_multiplier = 1.0 + (content_analysis.motion_complexity + 
                                     content_analysis.spatial_complexity) / 20.0
        adjusted_min_bpp = min_bpp * complexity_multiplier
        
        # Calculate minimum bitrate needed
        min_bitrate_kbps = (pixels_per_second * adjusted_min_bpp) / 1000
        
        # Calculate available bitrate
        total_bits = target_size_mb * 8 * 1024 * 1024
        available_bitrate_kbps = total_bits / (duration_seconds * 1000)
        
        # Calculate severity (how much below minimum we are)
        if available_bitrate_kbps >= min_bitrate_kbps:
            severity = 0.0  # No constraint
        else:
            ratio = available_bitrate_kbps / min_bitrate_kbps
            severity = 1.0 - ratio  # Higher severity for lower ratios
        
        return max(0.0, min(severity, 1.0))
    
    def _assess_quality_feasibility(self, content_analysis: ContentAnalysis,
                                   target_size_mb: float, duration_seconds: float) -> float:
        """Assess how feasible good quality is (0=impossible, 1=easy)."""
        
        # Base feasibility on content type compressibility
        compressibility_map = {
            ContentType.ANIMATION: 0.9,      # Very compressible
            ContentType.SCREEN_RECORDING: 0.95,  # Extremely compressible
            ContentType.LIVE_ACTION: 0.6,    # Moderately compressible
            ContentType.GAMING: 0.7,         # Fairly compressible
            ContentType.MIXED: 0.65          # Mixed compressibility
        }
        
        base_feasibility = compressibility_map.get(content_analysis.content_type, 0.6)
        
        # Adjust for content complexity
        complexity_penalty = (content_analysis.motion_complexity + 
                            content_analysis.spatial_complexity) / 20.0 * 0.3
        
        # Adjust for noise (noisy content is harder to compress)
        noise_penalty = content_analysis.noise_level / 10.0 * 0.2
        
        # Adjust for temporal stability (stable content is easier)
        stability_bonus = content_analysis.temporal_stability / 10.0 * 0.2
        
        feasibility = base_feasibility - complexity_penalty - noise_penalty + stability_bonus
        
        return max(0.0, min(feasibility, 1.0))
    
    def _assess_complexity_challenge(self, content_analysis: ContentAnalysis) -> float:
        """Assess how challenging the content is to compress (0=easy, 1=very hard)."""
        
        # Combine various complexity factors
        motion_challenge = content_analysis.motion_complexity / 10.0
        spatial_challenge = content_analysis.spatial_complexity / 10.0
        noise_challenge = content_analysis.noise_level / 10.0
        scene_change_challenge = min(content_analysis.scene_count / 50.0, 1.0)
        
        # Weight the factors
        overall_challenge = (
            motion_challenge * 0.3 +
            spatial_challenge * 0.3 +
            noise_challenge * 0.2 +
            scene_change_challenge * 0.2
        )
        
        return max(0.0, min(overall_challenge, 1.0))
    
    def _generate_quality_size_options(self, content_analysis: ContentAnalysis,
                                     target_size_mb: float, duration_seconds: float,
                                     original_resolution: Tuple[int, int]) -> List[QualitySizeOption]:
        """Generate multiple quality-size trade-off options."""
        
        options = []
        
        # Define resolution options (width, height, scale_factor)
        width, height = original_resolution
        resolution_options = [
            (width, height, 1.0, "Original"),
            (min(width, 1920), min(height, 1080), min(1920/width, 1080/height, 1.0), "1080p"),
            (min(width, 1280), min(height, 720), min(1280/width, 720/height, 1.0), "720p"),
            (min(width, 854), min(height, 480), min(854/width, 480/height, 1.0), "480p")
        ]
        
        # Define quality presets
        quality_presets = [
            ("highest", "veryslow", 18, 1.4, "Maximum Quality"),
            ("high", "slow", 20, 1.2, "High Quality"),
            ("balanced", "medium", 23, 1.0, "Balanced"),
            ("fast", "fast", 26, 0.8, "Fast Encode"),
            ("smallest", "veryfast", 28, 0.6, "Minimum Size")
        ]
        
        option_id = 0
        
        for res_width, res_height, scale_factor, res_name in resolution_options:
            for preset_name, ffmpeg_preset, base_crf, bitrate_factor, preset_desc in quality_presets:
                option_id += 1
                
                # Calculate bitrate allocation
                pixels_per_second = res_width * res_height * 30  # Assume 30fps
                
                # Base bitrate calculation
                base_bpp = self._get_base_bpp_for_content(content_analysis.content_type)
                base_bitrate = (pixels_per_second * base_bpp * bitrate_factor) / 1000
                
                # Adjust for content complexity
                complexity_factor = 1.0 + (content_analysis.motion_complexity + 
                                          content_analysis.spatial_complexity) / 20.0
                adjusted_bitrate = base_bitrate * complexity_factor
                
                # Allocate between video and audio
                audio_bitrate = min(128, max(32, int(adjusted_bitrate * 0.1)))
                video_bitrate = int(adjusted_bitrate - audio_bitrate)
                
                # Estimate file size
                total_bitrate_kbps = video_bitrate + audio_bitrate
                estimated_size_mb = (total_bitrate_kbps * duration_seconds) / (8 * 1024)
                
                # Skip if way over target (unless it's the smallest option)
                if estimated_size_mb > target_size_mb * 1.5 and preset_name != "smallest":
                    continue
                
                # Predict quality
                quality_score = self._predict_quality_score(
                    content_analysis, video_bitrate, res_width, res_height, ffmpeg_preset
                )
                
                # Determine quality level
                quality_level = self._classify_quality_level(quality_score)
                
                # Calculate efficiency metrics
                size_efficiency = quality_score / max(estimated_size_mb, 0.1)
                
                # Generate pros and cons
                pros, cons = self._generate_pros_cons(
                    preset_name, res_name, quality_score, estimated_size_mb, target_size_mb
                )
                
                # Generate recommendations
                recommended_for = self._generate_recommendations(
                    content_analysis, preset_name, res_name, quality_score
                )
                
                # Estimate encoding time
                encoding_time = self._estimate_encoding_time(
                    duration_seconds, ffmpeg_preset, res_width, res_height
                )
                
                # Calculate compatibility score
                compatibility = self._calculate_compatibility_score(res_width, res_height, ffmpeg_preset)
                
                option = QualitySizeOption(
                    option_id=f"option_{option_id}",
                    description=f"{preset_desc} ({res_name})",
                    estimated_size_mb=estimated_size_mb,
                    predicted_quality_score=quality_score,
                    quality_level=quality_level,
                    video_bitrate_kbps=video_bitrate,
                    audio_bitrate_kbps=audio_bitrate,
                    resolution=(res_width, res_height),
                    fps=30.0,  # Simplified
                    codec="libx264",  # Simplified
                    preset=ffmpeg_preset,
                    size_efficiency=size_efficiency,
                    quality_loss_vs_best=0.0,  # Will be calculated later
                    size_savings_vs_best=0.0,  # Will be calculated later
                    pros=pros,
                    cons=cons,
                    recommended_for=recommended_for,
                    encoding_time_estimate=encoding_time,
                    compatibility_score=compatibility
                )
                
                options.append(option)
        
        # Calculate relative metrics
        if options:
            best_quality = max(opt.predicted_quality_score for opt in options)
            smallest_size = min(opt.estimated_size_mb for opt in options)
            
            for option in options:
                option.quality_loss_vs_best = best_quality - option.predicted_quality_score
                option.size_savings_vs_best = option.estimated_size_mb - smallest_size
        
        # Sort by size efficiency (quality per MB)
        options.sort(key=lambda x: x.size_efficiency, reverse=True)
        
        return options[:8]  # Limit to top 8 options
    
    def _get_base_bpp_for_content(self, content_type: ContentType) -> float:
        """Get base bits-per-pixel for content type."""
        bpp_map = {
            ContentType.ANIMATION: 0.012,
            ContentType.LIVE_ACTION: 0.020,
            ContentType.SCREEN_RECORDING: 0.006,
            ContentType.GAMING: 0.016,
            ContentType.MIXED: 0.018
        }
        return bpp_map.get(content_type, 0.020)
    
    def _predict_quality_score(self, content_analysis: ContentAnalysis, video_bitrate_kbps: int,
                              width: int, height: int, preset: str) -> float:
        """Predict quality score based on encoding parameters."""
        
        # Calculate bits per pixel
        pixels_per_second = width * height * 30  # Assume 30fps
        bpp = (video_bitrate_kbps * 1000) / pixels_per_second
        
        # Base quality from BPP
        base_quality = min(bpp / 0.03, 1.0)  # Normalize around 0.03 BPP
        
        # Adjust for preset quality
        preset_quality_map = {
            'ultrafast': 0.6, 'superfast': 0.7, 'veryfast': 0.8, 'faster': 0.85,
            'fast': 0.9, 'medium': 0.95, 'slow': 1.0, 'slower': 1.05, 'veryslow': 1.1
        }
        preset_factor = preset_quality_map.get(preset, 0.95)
        
        # Adjust for content complexity
        complexity_penalty = (content_analysis.motion_complexity + 
                            content_analysis.spatial_complexity) / 20.0 * 0.2
        
        # Adjust for content type
        content_type_factor = {
            ContentType.ANIMATION: 1.1,      # Animation compresses well
            ContentType.SCREEN_RECORDING: 1.2,  # Screen content compresses very well
            ContentType.LIVE_ACTION: 1.0,    # Baseline
            ContentType.GAMING: 0.95,        # Gaming can be challenging
            ContentType.MIXED: 0.98          # Mixed content slightly challenging
        }.get(content_analysis.content_type, 1.0)
        
        predicted_quality = base_quality * preset_factor * content_type_factor - complexity_penalty
        
        return max(0.0, min(predicted_quality, 1.0))
    
    def _classify_quality_level(self, quality_score: float) -> QualityLevel:
        """Classify quality score into quality level."""
        for level, threshold in self.quality_thresholds.items():
            if quality_score >= threshold:
                return level
        return QualityLevel.UNACCEPTABLE
    
    def _generate_pros_cons(self, preset_name: str, res_name: str, quality_score: float,
                           estimated_size_mb: float, target_size_mb: float) -> Tuple[List[str], List[str]]:
        """Generate pros and cons for an option."""
        
        pros = []
        cons = []
        
        # Size-related pros/cons
        size_ratio = estimated_size_mb / target_size_mb
        if size_ratio <= 0.8:
            pros.append(f"Well under size limit ({estimated_size_mb:.1f}MB vs {target_size_mb:.1f}MB target)")
        elif size_ratio <= 1.0:
            pros.append(f"Meets size target ({estimated_size_mb:.1f}MB)")
        else:
            cons.append(f"Exceeds size limit ({estimated_size_mb:.1f}MB vs {target_size_mb:.1f}MB target)")
        
        # Quality-related pros/cons
        if quality_score >= 0.9:
            pros.append("Excellent visual quality")
        elif quality_score >= 0.75:
            pros.append("Good visual quality")
        elif quality_score >= 0.6:
            pros.append("Acceptable quality for most uses")
        else:
            cons.append("Quality may be noticeably reduced")
        
        # Preset-related pros/cons
        if preset_name == "highest":
            pros.append("Maximum quality encoding")
            cons.append("Very slow encoding time")
        elif preset_name == "high":
            pros.append("High quality with reasonable encoding time")
        elif preset_name == "balanced":
            pros.append("Good balance of quality, size, and speed")
        elif preset_name == "fast":
            pros.append("Fast encoding")
            cons.append("Some quality trade-offs for speed")
        elif preset_name == "smallest":
            pros.append("Smallest possible file size")
            cons.append("Significant quality reduction")
        
        # Resolution-related pros/cons
        if res_name == "Original":
            pros.append("Preserves original resolution")
        elif res_name == "1080p":
            pros.append("High definition quality")
        elif res_name == "720p":
            pros.append("Good quality, widely compatible")
        elif res_name == "480p":
            pros.append("Very small file size")
            cons.append("Lower resolution may affect detail")
        
        return pros, cons
    
    def _generate_recommendations(self, content_analysis: ContentAnalysis, preset_name: str,
                                res_name: str, quality_score: float) -> List[str]:
        """Generate recommendations for when to use this option."""
        
        recommendations = []
        
        # Content-type specific recommendations
        if content_analysis.content_type == ContentType.ANIMATION:
            if preset_name in ["highest", "high"]:
                recommendations.append("Animation with fine details")
            elif preset_name == "balanced":
                recommendations.append("Standard animation content")
        
        elif content_analysis.content_type == ContentType.LIVE_ACTION:
            if preset_name == "highest":
                recommendations.append("Professional video production")
            elif preset_name == "high":
                recommendations.append("High-quality personal videos")
            elif preset_name == "balanced":
                recommendations.append("General video sharing")
        
        elif content_analysis.content_type == ContentType.SCREEN_RECORDING:
            if quality_score >= 0.8:
                recommendations.append("Tutorials with text/UI elements")
            else:
                recommendations.append("Quick screen captures")
        
        # Size-based recommendations
        if preset_name == "smallest":
            recommendations.append("When file size is critical")
            recommendations.append("Mobile/bandwidth-limited sharing")
        
        # Quality-based recommendations
        if quality_score >= 0.9:
            recommendations.append("Archival/master copies")
            recommendations.append("Professional presentation")
        elif quality_score >= 0.75:
            recommendations.append("Social media sharing")
            recommendations.append("General viewing")
        elif quality_score >= 0.6:
            recommendations.append("Quick sharing/preview")
        
        return recommendations
    
    def _estimate_encoding_time(self, duration_seconds: float, preset: str,
                               width: int, height: int) -> float:
        """Estimate encoding time in seconds."""
        
        # Base time factor from preset
        time_factor = self.encoding_time_factors.get(preset, 3.0)
        
        # Adjust for resolution
        pixels = width * height
        resolution_factor = pixels / (1920 * 1080)  # Normalize to 1080p
        
        # Estimate total time
        estimated_time = duration_seconds * time_factor * resolution_factor
        
        return estimated_time
    
    def _calculate_compatibility_score(self, width: int, height: int, preset: str) -> float:
        """Calculate device/platform compatibility score."""
        
        compatibility = 1.0
        
        # Resolution compatibility
        if width > 1920 or height > 1080:
            compatibility -= 0.1  # Some older devices may struggle
        
        # Preset compatibility (slower presets may have compatibility issues)
        if preset in ['veryslow', 'slower']:
            compatibility -= 0.05  # Some players may not handle advanced features well
        
        return max(0.5, compatibility)
    
    def _select_recommended_option(self, options: List[QualitySizeOption],
                                  strategy: TradeoffStrategy) -> str:
        """Select the recommended option based on strategy."""
        
        if not options:
            return ""
        
        if strategy == TradeoffStrategy.MAXIMIZE_QUALITY:
            # Choose highest quality that's reasonably close to target size
            viable_options = [opt for opt in options if opt.estimated_size_mb <= opt.estimated_size_mb * 1.2]
            if viable_options:
                return max(viable_options, key=lambda x: x.predicted_quality_score).option_id
            else:
                return max(options, key=lambda x: x.predicted_quality_score).option_id
        
        elif strategy == TradeoffStrategy.MINIMIZE_SIZE:
            return min(options, key=lambda x: x.estimated_size_mb).option_id
        
        elif strategy == TradeoffStrategy.BALANCED:
            # Choose best size efficiency (quality per MB)
            return max(options, key=lambda x: x.size_efficiency).option_id
        
        else:  # CUSTOM or fallback
            return max(options, key=lambda x: x.size_efficiency).option_id
    
    def _generate_user_notifications(self, content_analysis: ContentAnalysis,
                                   options: List[QualitySizeOption],
                                   size_constraint_severity: float,
                                   quality_feasibility: float) -> Tuple[List[str], List[str], List[str]]:
        """Generate user warnings, suggestions, and technical notes."""
        
        warnings = []
        suggestions = []
        technical_notes = []
        
        # Size constraint warnings
        if size_constraint_severity > 0.8:
            warnings.append("Size constraint is very tight - significant quality reduction may be necessary")
        elif size_constraint_severity > 0.6:
            warnings.append("Size constraint is challenging - some quality trade-offs will be required")
        
        # Quality feasibility warnings
        if quality_feasibility < 0.4:
            warnings.append("Content is difficult to compress - achieving good quality may not be possible")
        elif quality_feasibility < 0.6:
            warnings.append("Content complexity may limit achievable quality")
        
        # Content-specific suggestions
        if content_analysis.content_type == ContentType.LIVE_ACTION and content_analysis.noise_level > 6.0:
            suggestions.append("Consider denoising the source video before compression for better results")
        
        if content_analysis.motion_complexity > 8.0:
            suggestions.append("High motion content detected - consider reducing frame rate if acceptable")
        
        if content_analysis.scene_count > 30:
            suggestions.append("Many scene changes detected - consider splitting into segments for better compression")
        
        # Option-based suggestions
        best_quality_option = max(options, key=lambda x: x.predicted_quality_score) if options else None
        smallest_option = min(options, key=lambda x: x.estimated_size_mb) if options else None
        
        if best_quality_option and smallest_option:
            quality_diff = best_quality_option.predicted_quality_score - smallest_option.predicted_quality_score
            size_diff = best_quality_option.estimated_size_mb - smallest_option.estimated_size_mb
            
            if quality_diff > 0.3 and size_diff > 2.0:
                suggestions.append(f"Consider if {size_diff:.1f}MB extra size is worth {quality_diff*100:.0f}% better quality")
        
        # Technical notes
        if content_analysis.content_type == ContentType.ANIMATION:
            technical_notes.append("Animation content typically compresses very efficiently")
        elif content_analysis.content_type == ContentType.SCREEN_RECORDING:
            technical_notes.append("Screen recordings with text benefit from higher quality settings")
        
        if content_analysis.spatial_complexity > 8.0:
            technical_notes.append("High spatial complexity detected - may benefit from higher bitrate allocation")
        
        return warnings, suggestions, technical_notes
    
    def _generate_alternative_approaches(self, content_analysis: ContentAnalysis,
                                       target_size_mb: float, duration_seconds: float,
                                       size_constraint_severity: float) -> List[Dict[str, Any]]:
        """Generate alternative approaches when standard options aren't sufficient."""
        
        alternatives = []
        
        # Suggest segmentation for very long videos
        if duration_seconds > 300 and size_constraint_severity > 0.7:  # 5+ minutes
            alternatives.append({
                'approach': 'video_segmentation',
                'title': 'Split into Multiple Parts',
                'description': f'Split the {duration_seconds/60:.1f}-minute video into shorter segments',
                'benefits': ['Better compression efficiency', 'Easier to share', 'More manageable file sizes'],
                'considerations': ['Multiple files to manage', 'May interrupt viewing experience']
            })
        
        # Suggest preprocessing for noisy content
        if content_analysis.noise_level > 6.0:
            alternatives.append({
                'approach': 'preprocessing',
                'title': 'Denoise Source Video',
                'description': 'Apply denoising to source video before compression',
                'benefits': ['Significantly better compression', 'Cleaner final result', 'Smaller file sizes'],
                'considerations': ['Additional processing time', 'May soften some details']
            })
        
        # Suggest two-pass encoding for critical quality
        if size_constraint_severity > 0.6 and content_analysis.spatial_complexity > 7.0:
            alternatives.append({
                'approach': 'two_pass_encoding',
                'title': 'Two-Pass Encoding',
                'description': 'Use two-pass encoding for optimal bitrate distribution',
                'benefits': ['Better quality at same file size', 'More efficient bitrate usage'],
                'considerations': ['Approximately double encoding time']
            })
        
        # Suggest format alternatives
        if content_analysis.content_type in [ContentType.ANIMATION, ContentType.SCREEN_RECORDING]:
            alternatives.append({
                'approach': 'format_change',
                'title': 'Consider Alternative Formats',
                'description': 'Use HEVC/H.265 codec for better compression efficiency',
                'benefits': ['30-50% smaller files', 'Better quality at same size'],
                'considerations': ['Slower encoding', 'May have compatibility issues on older devices']
            })
        
        # Suggest quality vs size reconsideration
        if size_constraint_severity > 0.8:
            alternatives.append({
                'approach': 'target_adjustment',
                'title': 'Reconsider Size Target',
                'description': f'Current {target_size_mb:.1f}MB target may be too restrictive for good quality',
                'benefits': ['Significantly better quality possible', 'Less compression artifacts'],
                'considerations': ['Larger file size', 'May not meet original size requirements']
            })
        
        return alternatives
    
    def generate_user_notification_summary(self, analysis: TradeoffAnalysis) -> Dict[str, Any]:
        """Generate a user-friendly summary of the trade-off analysis."""
        
        recommended_option = None
        for option in analysis.options:
            if option.option_id == analysis.recommended_option_id:
                recommended_option = option
                break
        
        summary = {
            'constraint_assessment': self._get_constraint_assessment_text(analysis.size_constraint_severity),
            'quality_outlook': self._get_quality_outlook_text(analysis.quality_feasibility),
            'recommended_option': {
                'description': recommended_option.description if recommended_option else "No suitable option",
                'size': f"{recommended_option.estimated_size_mb:.1f}MB" if recommended_option else "N/A",
                'quality': recommended_option.quality_level.value if recommended_option else "unknown",
                'encoding_time': f"{recommended_option.encoding_time_estimate/60:.1f} minutes" if recommended_option else "N/A"
            } if recommended_option else None,
            'key_warnings': analysis.warnings[:2],  # Top 2 warnings
            'key_suggestions': analysis.suggestions[:2],  # Top 2 suggestions
            'has_alternatives': len(analysis.alternative_approaches) > 0,
            'alternative_count': len(analysis.alternative_approaches)
        }
        
        return summary
    
    def _get_constraint_assessment_text(self, severity: float) -> str:
        """Get user-friendly text for constraint severity."""
        if severity < 0.3:
            return "Size target is achievable with good quality"
        elif severity < 0.6:
            return "Size target is challenging but manageable"
        elif severity < 0.8:
            return "Size target is tight - quality trade-offs required"
        else:
            return "Size target is very restrictive - significant quality impact expected"
    
    def _get_quality_outlook_text(self, feasibility: float) -> str:
        """Get user-friendly text for quality feasibility."""
        if feasibility > 0.8:
            return "Excellent quality is achievable"
        elif feasibility > 0.6:
            return "Good quality is likely achievable"
        elif feasibility > 0.4:
            return "Acceptable quality should be possible"
        else:
            return "Quality may be limited due to content complexity"