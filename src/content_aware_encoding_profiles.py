"""
Content-Aware Encoding Profiles
Implements dynamic encoding parameter selection based on content analysis
"""

import math
from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

try:
    from .content_analysis_engine import ContentType, ContentAnalysis
except ImportError:
    from content_analysis_engine import ContentType, ContentAnalysis

logger = logging.getLogger(__name__)


@dataclass
class EncodingProfile:
    """Encoding profile with content-specific parameters."""
    name: str
    description: str
    
    # Core encoding parameters
    codec_preference: List[str]  # Ordered list of preferred codecs
    preset: str
    tune: str
    crf_base: int  # Base CRF value
    
    # Bitrate allocation strategy
    bitrate_efficiency_factor: float  # Multiplier for bitrate calculations
    audio_bitrate_ratio: float  # Ratio of total bitrate for audio
    
    # Quality parameters
    min_quality_threshold: float  # Minimum acceptable quality score
    quality_vs_size_preference: float  # 0.0 = size priority, 1.0 = quality priority
    
    # Advanced parameters
    gop_size_factor: float  # Multiplier for GOP size
    b_frame_strategy: str  # 'conservative', 'balanced', 'aggressive'
    motion_estimation: str  # 'fast', 'good', 'best'
    
    # Preprocessing filters
    enable_denoising: bool
    enable_deband: bool
    enable_sharpening: bool
    
    # Perceptual optimizations
    psychovisual_tuning: Dict[str, Any]
    
    # Content-specific adjustments
    motion_sensitivity: float  # How much to adjust for motion complexity
    spatial_sensitivity: float  # How much to adjust for spatial complexity
    
    
@dataclass
class BitrateAllocation:
    """Bitrate allocation strategy based on content complexity."""
    video_bitrate_kbps: int
    audio_bitrate_kbps: int
    total_bitrate_kbps: int
    
    # Allocation reasoning
    complexity_factor: float
    motion_adjustment: float
    spatial_adjustment: float
    content_type_adjustment: float
    
    # Quality predictions
    predicted_quality_score: float
    confidence_level: float


@dataclass
class DynamicParameters:
    """Dynamically adjusted parameters based on content analysis."""
    adjusted_crf: int
    adjusted_preset: str
    adjusted_gop_size: int
    adjusted_b_frames: int
    
    # Filter chain
    video_filters: List[str]
    
    # Reasoning for adjustments
    adjustments_made: List[str]
    reasoning: Dict[str, str]


class ContentAwareEncodingProfiles:
    """Content-aware encoding profile system."""
    
    def __init__(self, config_manager):
        self.config = config_manager
        
        # Initialize predefined profiles
        self.profiles = self._initialize_profiles()
        
        # Configuration
        self.enable_dynamic_adjustment = self.config.get('encoding_profiles.enable_dynamic_adjustment', True)
        self.quality_vs_speed_preference = self.config.get('encoding_profiles.quality_vs_speed', 0.7)
        
    def _initialize_profiles(self) -> Dict[str, EncodingProfile]:
        """Initialize predefined encoding profiles for different content types."""
        
        profiles = {}
        
        # Animation profiles
        profiles['animation_standard'] = EncodingProfile(
            name='animation_standard',
            description='Standard animation encoding with flat color optimization',
            codec_preference=['libx264', 'libx265'],
            preset='medium',
            tune='animation',
            crf_base=20,
            bitrate_efficiency_factor=0.8,  # Animation compresses well
            audio_bitrate_ratio=0.1,
            min_quality_threshold=0.85,
            quality_vs_size_preference=0.7,
            gop_size_factor=1.2,  # Longer GOPs for animation
            b_frame_strategy='aggressive',
            motion_estimation='good',
            enable_denoising=False,  # Animation shouldn't need denoising
            enable_deband=True,  # Help with banding in gradients
            enable_sharpening=False,
            psychovisual_tuning={
                'psy_rd': 0.8,  # Lower for animation
                'psy_trellis': 0.1,
                'aq_mode': 2,
                'aq_strength': 0.8
            },
            motion_sensitivity=0.6,  # Less sensitive to motion in animation
            spatial_sensitivity=0.9   # More sensitive to spatial detail
        )
        
        profiles['animation_high_detail'] = EncodingProfile(
            name='animation_high_detail',
            description='High-detail animation with complex artwork',
            codec_preference=['libx265', 'libx264'],
            preset='slow',
            tune='animation',
            crf_base=18,
            bitrate_efficiency_factor=1.1,
            audio_bitrate_ratio=0.08,
            min_quality_threshold=0.9,
            quality_vs_size_preference=0.8,
            gop_size_factor=1.0,
            b_frame_strategy='balanced',
            motion_estimation='best',
            enable_denoising=False,
            enable_deband=True,
            enable_sharpening=True,
            psychovisual_tuning={
                'psy_rd': 1.0,
                'psy_trellis': 0.2,
                'aq_mode': 3,
                'aq_strength': 1.0
            },
            motion_sensitivity=0.5,
            spatial_sensitivity=1.2
        )
        
        profiles['animation_high_motion'] = EncodingProfile(
            name='animation_high_motion',
            description='High-motion animation (action scenes, effects)',
            codec_preference=['libx264', 'libx265'],
            preset='fast',
            tune='animation',
            crf_base=22,
            bitrate_efficiency_factor=1.2,
            audio_bitrate_ratio=0.12,
            min_quality_threshold=0.8,
            quality_vs_size_preference=0.6,
            gop_size_factor=0.8,  # Shorter GOPs for motion
            b_frame_strategy='conservative',
            motion_estimation='fast',
            enable_denoising=False,
            enable_deband=True,
            enable_sharpening=False,
            psychovisual_tuning={
                'psy_rd': 0.6,
                'psy_trellis': 0.05,
                'aq_mode': 1,
                'aq_strength': 0.6
            },
            motion_sensitivity=1.2,
            spatial_sensitivity=0.7
        )
        
        # Live action profiles
        profiles['live_action_standard'] = EncodingProfile(
            name='live_action_standard',
            description='Standard live action content',
            codec_preference=['libx264', 'libx265'],
            preset='medium',
            tune='film',
            crf_base=23,
            bitrate_efficiency_factor=1.0,
            audio_bitrate_ratio=0.12,
            min_quality_threshold=0.8,
            quality_vs_size_preference=0.6,
            gop_size_factor=1.0,
            b_frame_strategy='balanced',
            motion_estimation='good',
            enable_denoising=True,
            enable_deband=False,
            enable_sharpening=False,
            psychovisual_tuning={
                'psy_rd': 1.0,
                'psy_trellis': 0.15,
                'aq_mode': 2,
                'aq_strength': 1.0
            },
            motion_sensitivity=1.0,
            spatial_sensitivity=1.0
        )
        
        profiles['live_action_high_motion'] = EncodingProfile(
            name='live_action_high_motion',
            description='High-motion live action (sports, action)',
            codec_preference=['libx264', 'libx265'],
            preset='fast',
            tune='film',
            crf_base=24,
            bitrate_efficiency_factor=1.3,
            audio_bitrate_ratio=0.1,
            min_quality_threshold=0.75,
            quality_vs_size_preference=0.5,
            gop_size_factor=0.7,
            b_frame_strategy='conservative',
            motion_estimation='fast',
            enable_denoising=True,
            enable_deband=False,
            enable_sharpening=False,
            psychovisual_tuning={
                'psy_rd': 0.8,
                'psy_trellis': 0.1,
                'aq_mode': 1,
                'aq_strength': 0.8
            },
            motion_sensitivity=1.4,
            spatial_sensitivity=0.8
        )
        
        profiles['live_action_high_detail'] = EncodingProfile(
            name='live_action_high_detail',
            description='High-detail live action (nature, architecture)',
            codec_preference=['libx265', 'libx264'],
            preset='slow',
            tune='film',
            crf_base=21,
            bitrate_efficiency_factor=1.2,
            audio_bitrate_ratio=0.08,
            min_quality_threshold=0.85,
            quality_vs_size_preference=0.8,
            gop_size_factor=1.1,
            b_frame_strategy='aggressive',
            motion_estimation='best',
            enable_denoising=True,
            enable_deband=False,
            enable_sharpening=True,
            psychovisual_tuning={
                'psy_rd': 1.2,
                'psy_trellis': 0.2,
                'aq_mode': 3,
                'aq_strength': 1.2
            },
            motion_sensitivity=0.8,
            spatial_sensitivity=1.3
        )
        
        profiles['live_action_complex'] = EncodingProfile(
            name='live_action_complex',
            description='Complex live action (high motion + high detail)',
            codec_preference=['libx265', 'libx264'],
            preset='medium',
            tune='film',
            crf_base=22,
            bitrate_efficiency_factor=1.4,
            audio_bitrate_ratio=0.1,
            min_quality_threshold=0.8,
            quality_vs_size_preference=0.7,
            gop_size_factor=0.9,
            b_frame_strategy='balanced',
            motion_estimation='good',
            enable_denoising=True,
            enable_deband=False,
            enable_sharpening=True,
            psychovisual_tuning={
                'psy_rd': 1.0,
                'psy_trellis': 0.15,
                'aq_mode': 2,
                'aq_strength': 1.0
            },
            motion_sensitivity=1.2,
            spatial_sensitivity=1.2
        )
        
        # Screen recording profiles
        profiles['screen_recording'] = EncodingProfile(
            name='screen_recording',
            description='Screen recordings with text and UI elements',
            codec_preference=['libx264', 'libx265'],
            preset='fast',
            tune='stillimage',
            crf_base=18,  # Lower CRF for text clarity
            bitrate_efficiency_factor=0.7,  # Screen content compresses very well
            audio_bitrate_ratio=0.15,  # Often has voice-over
            min_quality_threshold=0.9,  # High quality needed for text
            quality_vs_size_preference=0.8,
            gop_size_factor=2.0,  # Much longer GOPs for screen content
            b_frame_strategy='aggressive',
            motion_estimation='fast',
            enable_denoising=False,
            enable_deband=False,
            enable_sharpening=True,  # Important for text clarity
            psychovisual_tuning={
                'psy_rd': 0.5,  # Lower for screen content
                'psy_trellis': 0.0,
                'aq_mode': 0,  # Disable adaptive quantization
                'aq_strength': 0.0
            },
            motion_sensitivity=0.3,  # Screen recordings usually have low motion
            spatial_sensitivity=1.5   # Very sensitive to spatial detail for text
        )
        
        # Gaming profiles
        profiles['gaming_standard'] = EncodingProfile(
            name='gaming_standard',
            description='Standard gaming content',
            codec_preference=['libx264', 'libx265'],
            preset='fast',
            tune='film',
            crf_base=22,
            bitrate_efficiency_factor=1.1,
            audio_bitrate_ratio=0.12,
            min_quality_threshold=0.8,
            quality_vs_size_preference=0.6,
            gop_size_factor=0.8,
            b_frame_strategy='balanced',
            motion_estimation='good',
            enable_denoising=False,
            enable_deband=True,
            enable_sharpening=True,
            psychovisual_tuning={
                'psy_rd': 0.9,
                'psy_trellis': 0.1,
                'aq_mode': 2,
                'aq_strength': 0.9
            },
            motion_sensitivity=1.1,
            spatial_sensitivity=1.1
        )
        
        profiles['gaming_high_action'] = EncodingProfile(
            name='gaming_high_action',
            description='High-action gaming with rapid scene changes',
            codec_preference=['libx264', 'libx265'],
            preset='veryfast',
            tune='film',
            crf_base=24,
            bitrate_efficiency_factor=1.4,
            audio_bitrate_ratio=0.1,
            min_quality_threshold=0.75,
            quality_vs_size_preference=0.5,
            gop_size_factor=0.6,
            b_frame_strategy='conservative',
            motion_estimation='fast',
            enable_denoising=False,
            enable_deband=True,
            enable_sharpening=False,
            psychovisual_tuning={
                'psy_rd': 0.7,
                'psy_trellis': 0.05,
                'aq_mode': 1,
                'aq_strength': 0.7
            },
            motion_sensitivity=1.5,
            spatial_sensitivity=0.9
        )
        
        # Mixed content profiles
        profiles['mixed_standard'] = EncodingProfile(
            name='mixed_standard',
            description='Mixed content (animation + live action)',
            codec_preference=['libx264', 'libx265'],
            preset='medium',
            tune='film',
            crf_base=22,
            bitrate_efficiency_factor=1.1,
            audio_bitrate_ratio=0.12,
            min_quality_threshold=0.8,
            quality_vs_size_preference=0.65,
            gop_size_factor=1.0,
            b_frame_strategy='balanced',
            motion_estimation='good',
            enable_denoising=True,
            enable_deband=True,
            enable_sharpening=False,
            psychovisual_tuning={
                'psy_rd': 0.9,
                'psy_trellis': 0.12,
                'aq_mode': 2,
                'aq_strength': 0.9
            },
            motion_sensitivity=1.0,
            spatial_sensitivity=1.0
        )
        
        profiles['mixed_complex'] = EncodingProfile(
            name='mixed_complex',
            description='Complex mixed content with varying characteristics',
            codec_preference=['libx265', 'libx264'],
            preset='slow',
            tune='film',
            crf_base=21,
            bitrate_efficiency_factor=1.3,
            audio_bitrate_ratio=0.1,
            min_quality_threshold=0.82,
            quality_vs_size_preference=0.75,
            gop_size_factor=0.9,
            b_frame_strategy='balanced',
            motion_estimation='best',
            enable_denoising=True,
            enable_deband=True,
            enable_sharpening=True,
            psychovisual_tuning={
                'psy_rd': 1.0,
                'psy_trellis': 0.15,
                'aq_mode': 3,
                'aq_strength': 1.0
            },
            motion_sensitivity=1.1,
            spatial_sensitivity=1.1
        )
        
        return profiles
    
    def select_encoding_profile(self, content_analysis: ContentAnalysis, 
                               quality_target: float = 0.8) -> EncodingProfile:
        """
        Select the best encoding profile based on content analysis.
        
        Args:
            content_analysis: Results from content analysis
            quality_target: Target quality level (0.0-1.0)
            
        Returns:
            Selected EncodingProfile
        """
        profile_name = content_analysis.recommended_encoding_profile
        
        # Get base profile
        if profile_name in self.profiles:
            base_profile = self.profiles[profile_name]
        else:
            logger.warning(f"Unknown profile '{profile_name}', using live_action_standard")
            base_profile = self.profiles['live_action_standard']
        
        # Adjust profile based on quality target
        if quality_target != 0.8:  # Default target
            adjusted_profile = self._adjust_profile_for_quality_target(base_profile, quality_target)
            return adjusted_profile
        
        return base_profile
    
    def calculate_optimal_bitrate_distribution(self, content_analysis: ContentAnalysis, 
                                             target_size_mb: float, duration_seconds: float,
                                             profile: EncodingProfile) -> BitrateAllocation:
        """
        Calculate optimal bitrate distribution based on content complexity.
        
        Args:
            content_analysis: Content analysis results
            target_size_mb: Target file size in MB
            duration_seconds: Video duration in seconds
            profile: Selected encoding profile
            
        Returns:
            BitrateAllocation with optimized bitrate distribution
        """
        # Calculate base bitrate budget
        total_bits = target_size_mb * 8 * 1024 * 1024
        base_bitrate_kbps = total_bits / (duration_seconds * 1000)
        
        # Apply content complexity adjustments
        complexity_factor = self._calculate_complexity_factor(content_analysis, profile)
        
        # Calculate motion and spatial adjustments
        motion_adjustment = self._calculate_motion_adjustment(
            content_analysis.motion_complexity, profile.motion_sensitivity
        )
        spatial_adjustment = self._calculate_spatial_adjustment(
            content_analysis.spatial_complexity, profile.spatial_sensitivity
        )
        
        # Content type specific adjustment
        content_type_adjustment = self._get_content_type_adjustment(content_analysis.content_type)
        
        # Apply all adjustments
        adjusted_bitrate = base_bitrate_kbps * complexity_factor * profile.bitrate_efficiency_factor
        adjusted_bitrate *= (1.0 + motion_adjustment + spatial_adjustment + content_type_adjustment)
        
        # Allocate between video and audio
        audio_bitrate = int(adjusted_bitrate * profile.audio_bitrate_ratio)
        audio_bitrate = max(32, min(audio_bitrate, 192))  # Clamp audio bitrate
        
        video_bitrate = int(adjusted_bitrate - audio_bitrate)
        video_bitrate = max(100, video_bitrate)  # Minimum video bitrate
        
        # Predict quality
        predicted_quality = self._predict_quality_score(
            content_analysis, video_bitrate, profile
        )
        
        # Calculate confidence based on content analysis reliability
        confidence = self._calculate_confidence_level(content_analysis)
        
        return BitrateAllocation(
            video_bitrate_kbps=video_bitrate,
            audio_bitrate_kbps=audio_bitrate,
            total_bitrate_kbps=video_bitrate + audio_bitrate,
            complexity_factor=complexity_factor,
            motion_adjustment=motion_adjustment,
            spatial_adjustment=spatial_adjustment,
            content_type_adjustment=content_type_adjustment,
            predicted_quality_score=predicted_quality,
            confidence_level=confidence
        )
    
    def generate_dynamic_parameters(self, content_analysis: ContentAnalysis, 
                                   profile: EncodingProfile,
                                   bitrate_allocation: BitrateAllocation) -> DynamicParameters:
        """
        Generate dynamically adjusted encoding parameters.
        
        Args:
            content_analysis: Content analysis results
            profile: Base encoding profile
            bitrate_allocation: Calculated bitrate allocation
            
        Returns:
            DynamicParameters with adjusted encoding settings
        """
        adjustments_made = []
        reasoning = {}
        
        # Adjust CRF based on content complexity
        base_crf = profile.crf_base
        crf_adjustment = 0
        
        if content_analysis.spatial_complexity > 8.0:
            crf_adjustment -= 2  # Lower CRF for high detail
            adjustments_made.append("CRF reduced for high spatial complexity")
            reasoning["crf"] = f"Reduced CRF by 2 due to high spatial complexity ({content_analysis.spatial_complexity:.1f})"
        elif content_analysis.spatial_complexity < 3.0:
            crf_adjustment += 1  # Higher CRF for low detail
            adjustments_made.append("CRF increased for low spatial complexity")
            reasoning["crf"] = f"Increased CRF by 1 due to low spatial complexity ({content_analysis.spatial_complexity:.1f})"
        
        if content_analysis.noise_level > 6.0:
            crf_adjustment += 1  # Higher CRF for noisy content
            adjustments_made.append("CRF increased for noisy content")
            reasoning["crf"] += f", increased by 1 for high noise level ({content_analysis.noise_level:.1f})"
        
        adjusted_crf = max(15, min(base_crf + crf_adjustment, 35))
        
        # Adjust preset based on complexity and quality requirements
        adjusted_preset = profile.preset
        if content_analysis.motion_complexity > 8.0 and profile.quality_vs_size_preference < 0.7:
            # Use faster preset for high motion when size is priority
            preset_map = {'slow': 'medium', 'medium': 'fast', 'fast': 'veryfast'}
            if profile.preset in preset_map:
                adjusted_preset = preset_map[profile.preset]
                adjustments_made.append("Preset accelerated for high motion content")
                reasoning["preset"] = f"Changed from {profile.preset} to {adjusted_preset} for high motion ({content_analysis.motion_complexity:.1f})"
        elif content_analysis.spatial_complexity > 8.0 and profile.quality_vs_size_preference > 0.7:
            # Use slower preset for high detail when quality is priority
            preset_map = {'veryfast': 'fast', 'fast': 'medium', 'medium': 'slow'}
            if profile.preset in preset_map:
                adjusted_preset = preset_map[profile.preset]
                adjustments_made.append("Preset slowed for high detail content")
                reasoning["preset"] = f"Changed from {profile.preset} to {adjusted_preset} for high spatial complexity ({content_analysis.spatial_complexity:.1f})"
        
        # Adjust GOP size
        base_gop = int(30 * profile.gop_size_factor)  # Assume 30fps base
        gop_adjustment = 0
        
        if content_analysis.motion_complexity > 7.0:
            gop_adjustment = -int(base_gop * 0.3)  # Shorter GOP for motion
            adjustments_made.append("GOP size reduced for high motion")
            reasoning["gop"] = f"Reduced GOP size by 30% for high motion ({content_analysis.motion_complexity:.1f})"
        elif content_analysis.temporal_stability > 8.0:
            gop_adjustment = int(base_gop * 0.5)  # Longer GOP for stable content
            adjustments_made.append("GOP size increased for stable content")
            reasoning["gop"] = f"Increased GOP size by 50% for high temporal stability ({content_analysis.temporal_stability:.1f})"
        
        adjusted_gop_size = max(15, min(base_gop + gop_adjustment, 300))
        
        # Adjust B-frames
        b_frame_map = {'conservative': 2, 'balanced': 4, 'aggressive': 8}
        base_b_frames = b_frame_map.get(profile.b_frame_strategy, 4)
        
        adjusted_b_frames = base_b_frames
        if content_analysis.motion_complexity > 8.0:
            adjusted_b_frames = max(1, base_b_frames - 2)  # Fewer B-frames for motion
            adjustments_made.append("B-frames reduced for high motion")
            reasoning["b_frames"] = f"Reduced B-frames from {base_b_frames} to {adjusted_b_frames} for high motion"
        elif content_analysis.content_type == ContentType.SCREEN_RECORDING:
            adjusted_b_frames = min(8, base_b_frames + 2)  # More B-frames for screen content
            adjustments_made.append("B-frames increased for screen recording")
            reasoning["b_frames"] = f"Increased B-frames from {base_b_frames} to {adjusted_b_frames} for screen recording"
        
        # Build video filter chain
        video_filters = []
        
        # Scaling filter (always present)
        video_filters.append("scale=-2:720:flags=lanczos")  # Placeholder - would be set by caller
        
        # Denoising
        if profile.enable_denoising and content_analysis.noise_level > 3.0:
            denoise_strength = min(content_analysis.noise_level / 10.0, 0.8)
            video_filters.append(f"hqdn3d={denoise_strength}:{denoise_strength}:6:6")
            adjustments_made.append("Denoising enabled")
            reasoning["denoising"] = f"Applied denoising (strength {denoise_strength:.1f}) for noise level {content_analysis.noise_level:.1f}"
        
        # Debanding
        if profile.enable_deband and (content_analysis.content_type == ContentType.ANIMATION or 
                                     content_analysis.color_complexity < 4.0):
            video_filters.append("gradfun")
            adjustments_made.append("Debanding enabled")
            reasoning["debanding"] = "Applied debanding for animation/low color complexity content"
        
        # Sharpening
        if profile.enable_sharpening and (content_analysis.content_type == ContentType.SCREEN_RECORDING or
                                         content_analysis.edge_density > 0.15):
            sharpen_strength = 0.5 if content_analysis.content_type == ContentType.SCREEN_RECORDING else 0.3
            video_filters.append(f"unsharp=5:5:{sharpen_strength}:5:5:0.0")
            adjustments_made.append("Sharpening enabled")
            reasoning["sharpening"] = f"Applied sharpening (strength {sharpen_strength}) for edge enhancement"
        
        return DynamicParameters(
            adjusted_crf=adjusted_crf,
            adjusted_preset=adjusted_preset,
            adjusted_gop_size=adjusted_gop_size,
            adjusted_b_frames=adjusted_b_frames,
            video_filters=video_filters,
            adjustments_made=adjustments_made,
            reasoning=reasoning
        )
    
    def _adjust_profile_for_quality_target(self, base_profile: EncodingProfile, 
                                          quality_target: float) -> EncodingProfile:
        """Adjust profile parameters based on quality target."""
        # Create a copy of the profile
        import copy
        adjusted_profile = copy.deepcopy(base_profile)
        
        # Adjust CRF based on quality target
        if quality_target > 0.9:
            adjusted_profile.crf_base = max(15, base_profile.crf_base - 3)
            adjusted_profile.preset = 'slow' if base_profile.preset in ['fast', 'medium'] else base_profile.preset
        elif quality_target < 0.6:
            adjusted_profile.crf_base = min(30, base_profile.crf_base + 3)
            adjusted_profile.preset = 'fast' if base_profile.preset in ['slow', 'medium'] else base_profile.preset
        
        # Adjust quality vs size preference
        adjusted_profile.quality_vs_size_preference = quality_target
        
        return adjusted_profile
    
    def _calculate_complexity_factor(self, content_analysis: ContentAnalysis, 
                                   profile: EncodingProfile) -> float:
        """Calculate overall complexity factor for bitrate adjustment."""
        # Combine motion and spatial complexity with profile sensitivities
        motion_impact = content_analysis.motion_complexity * profile.motion_sensitivity
        spatial_impact = content_analysis.spatial_complexity * profile.spatial_sensitivity
        
        # Normalize to 0-10 scale
        combined_complexity = (motion_impact + spatial_impact) / 2.0
        
        # Convert to multiplier (1.0 = baseline, higher = more bitrate needed)
        complexity_factor = 0.7 + (combined_complexity / 10.0) * 0.6  # Range: 0.7 to 1.3
        
        return complexity_factor
    
    def _calculate_motion_adjustment(self, motion_complexity: float, 
                                   motion_sensitivity: float) -> float:
        """Calculate motion-based bitrate adjustment."""
        if motion_complexity > 7.0:
            return (motion_complexity - 7.0) * motion_sensitivity * 0.1
        elif motion_complexity < 3.0:
            return -(3.0 - motion_complexity) * motion_sensitivity * 0.05
        return 0.0
    
    def _calculate_spatial_adjustment(self, spatial_complexity: float, 
                                    spatial_sensitivity: float) -> float:
        """Calculate spatial complexity-based bitrate adjustment."""
        if spatial_complexity > 7.0:
            return (spatial_complexity - 7.0) * spatial_sensitivity * 0.08
        elif spatial_complexity < 3.0:
            return -(3.0 - spatial_complexity) * spatial_sensitivity * 0.04
        return 0.0
    
    def _get_content_type_adjustment(self, content_type: ContentType) -> float:
        """Get content type specific bitrate adjustment."""
        adjustments = {
            ContentType.ANIMATION: -0.1,  # Animation compresses well
            ContentType.LIVE_ACTION: 0.0,  # Baseline
            ContentType.SCREEN_RECORDING: -0.2,  # Screen content compresses very well
            ContentType.GAMING: 0.05,  # Gaming content can be complex
            ContentType.MIXED: 0.02   # Mixed content slightly more complex
        }
        return adjustments.get(content_type, 0.0)
    
    def _predict_quality_score(self, content_analysis: ContentAnalysis, 
                              video_bitrate_kbps: int, profile: EncodingProfile) -> float:
        """Predict quality score based on content and bitrate allocation."""
        # This is a simplified prediction model
        # In practice, you'd use machine learning or empirical data
        
        # Base quality from bitrate
        base_quality = min(video_bitrate_kbps / 2000.0, 1.0)  # Normalize around 2Mbps
        
        # Adjust for content complexity
        complexity_penalty = (content_analysis.motion_complexity + content_analysis.spatial_complexity) / 20.0
        complexity_penalty *= 0.2  # Limit impact
        
        # Adjust for profile quality preference
        profile_bonus = profile.quality_vs_size_preference * 0.1
        
        predicted_quality = base_quality - complexity_penalty + profile_bonus
        return max(0.0, min(predicted_quality, 1.0))
    
    def _calculate_confidence_level(self, content_analysis: ContentAnalysis) -> float:
        """Calculate confidence level in the analysis and predictions."""
        # Higher confidence for content with clear characteristics
        confidence = 0.7  # Base confidence
        
        # Increase confidence for clear content types
        if content_analysis.content_type in [ContentType.ANIMATION, ContentType.SCREEN_RECORDING]:
            confidence += 0.2
        
        # Increase confidence for stable content
        if content_analysis.temporal_stability > 7.0:
            confidence += 0.1
        
        # Decrease confidence for mixed or complex content
        if content_analysis.content_type == ContentType.MIXED:
            confidence -= 0.1
        
        if content_analysis.scene_count > 20:  # Many scene changes
            confidence -= 0.1
        
        return max(0.3, min(confidence, 0.95))
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available profile names."""
        return list(self.profiles.keys())
    
    def get_profile_description(self, profile_name: str) -> str:
        """Get description of a specific profile."""
        if profile_name in self.profiles:
            return self.profiles[profile_name].description
        return "Unknown profile"
    
    def optimize_for_content_type(self, base_params: Dict[str, Any], 
                                 content_analysis: ContentAnalysis) -> Dict[str, Any]:
        """
        Optimize encoding parameters for specific content type.
        
        Args:
            base_params: Base encoding parameters
            content_analysis: Content analysis results
            
        Returns:
            Optimized parameters dictionary
        """
        profile = self.select_encoding_profile(content_analysis)
        
        # Apply profile-specific optimizations
        optimized_params = base_params.copy()
        
        # Update codec preference
        if 'encoder' not in optimized_params and profile.codec_preference:
            optimized_params['encoder'] = profile.codec_preference[0]
        
        # Update preset and tune
        optimized_params['preset'] = profile.preset
        optimized_params['tune'] = profile.tune
        
        # Apply psychovisual tuning
        for key, value in profile.psychovisual_tuning.items():
            optimized_params[key] = value
        
        return optimized_params