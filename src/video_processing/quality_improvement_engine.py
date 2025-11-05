"""
Quality Improvement Engine
Implements intelligent quality enhancement when quality checks fail
Enhanced with comprehensive failure analysis and root cause identification
"""

import logging
from typing import Dict, Any, Optional, Tuple, List, NamedTuple
from dataclasses import dataclass
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class FailureType(Enum):
    """Types of quality failures that can be detected."""
    LOW_VMAF = "low_vmaf"
    LOW_SSIM = "low_ssim"
    HIGH_BLOCKINESS = "high_blockiness"
    HIGH_BANDING = "high_banding"
    BLUR_ARTIFACTS = "blur_artifacts"
    NOISE_ARTIFACTS = "noise_artifacts"
    COMPRESSION_ARTIFACTS = "compression_artifacts"
    TEMPORAL_ARTIFACTS = "temporal_artifacts"
    COMBINED_QUALITY = "combined_quality"
    ARTIFACTS_ONLY = "artifacts_only"


class RootCause(Enum):
    """Root causes of quality failures."""
    INSUFFICIENT_BITRATE = "insufficient_bitrate"
    AGGRESSIVE_COMPRESSION = "aggressive_compression"
    POOR_ENCODER_SETTINGS = "poor_encoder_settings"
    RESOLUTION_MISMATCH = "resolution_mismatch"
    TEMPORAL_COMPRESSION = "temporal_compression"
    CODEC_LIMITATIONS = "codec_limitations"
    CONTENT_COMPLEXITY = "content_complexity"
    PREPROCESSING_ISSUES = "preprocessing_issues"


class ImprovementStrategy(Enum):
    """Available improvement strategies."""
    INCREASE_BITRATE = "increase_bitrate"
    BETTER_ENCODING = "better_encoding"
    ADVANCED_CODEC = "advanced_codec"
    TEMPORAL_OPTIMIZATION = "temporal_optimization"
    PERCEPTUAL_TUNING = "perceptual_tuning"
    RESOLUTION_OPTIMIZATION = "resolution_optimization"


@dataclass
class FailureAnalysis:
    """Comprehensive analysis of why quality evaluation failed."""
    failure_types: List[FailureType]
    root_causes: List[RootCause]
    vmaf_score: Optional[float]
    ssim_score: Optional[float]
    blockiness_score: Optional[float]
    banding_score: Optional[float]
    blur_score: Optional[float]
    noise_score: Optional[float]
    primary_issue: FailureType
    primary_root_cause: RootCause
    severity: float  # 0.0 to 1.0, higher = more severe
    confidence: float  # 0.0 to 1.0, confidence in analysis
    recommended_strategies: List[ImprovementStrategy]
    size_headroom_mb: float
    can_improve: bool
    detailed_analysis: Dict[str, Any]
    improvement_potential: float  # 0.0 to 1.0, potential for improvement


@dataclass
class ImprovementPlan:
    """Plan for improving quality parameters."""
    strategy: ImprovementStrategy
    bitrate_increase_factor: float
    preset_change: Optional[str]
    codec_upgrade: Optional[str]
    temporal_params: Dict[str, Any]
    expected_quality_gain: float
    estimated_size_increase_mb: float
    feasible: bool
    justification: str


@dataclass
class CompressionParams:
    """Current compression parameters."""
    bitrate: int
    width: int
    height: int
    fps: float
    encoder: str
    preset: str
    crf: Optional[int]
    tune: Optional[str]
    gop_size: Optional[int]
    b_frames: Optional[int]
    audio_bitrate: int = 64  # Default to 64kbps (acceptable minimum for AAC)


class QualityImprovementEngine:
    """Engine for improving quality when quality checks fail."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        
        # Load improvement configuration
        self.max_improvement_iterations = self._get_config('max_improvement_iterations', 3)
        self.bitrate_increase_steps = self._get_config('bitrate_increase_steps', [1.2, 1.5, 2.0])
        self.quality_improvement_threshold = self._get_config('quality_improvement_threshold', 5.0)
        self.size_tolerance_factor = self._get_config('size_tolerance_factor', 0.95)
        
        # Encoder capabilities
        self.encoder_presets = {
            'libx264': ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow'],
            'h264_amf': ['speed', 'balanced', 'quality'],
            'h264_nvenc': ['p1', 'p2', 'p3', 'p4', 'p5', 'p6', 'p7']
        }
        
        self.codec_upgrades = {
            'libx264': 'libx265',
            'h264_amf': 'hevc_amf',
            'h264_nvenc': 'hevc_nvenc'
        }
        
        logger.info("Quality Improvement Engine initialized")
    
    @staticmethod
    def params_dict_to_compression_params(params_dict: Dict[str, Any]) -> CompressionParams:
        """Convert params dictionary to CompressionParams, preserving audio_bitrate."""
        # Extract audio_bitrate, defaulting to 64kbps if not present
        audio_bitrate = params_dict.get('audio_bitrate', 64)
        # Ensure minimum of 64kbps
        audio_bitrate = max(int(audio_bitrate), 64)
        
        return CompressionParams(
            bitrate=int(params_dict.get('bitrate', 1000)),
            width=int(params_dict.get('width', 1920)),
            height=int(params_dict.get('height', 1080)),
            fps=float(params_dict.get('fps', 30.0)),
            encoder=str(params_dict.get('encoder', 'libx264')),
            preset=str(params_dict.get('preset', 'medium')),
            crf=params_dict.get('crf'),
            tune=params_dict.get('tune'),
            gop_size=params_dict.get('gop_size'),
            b_frames=params_dict.get('b_frames'),
            audio_bitrate=audio_bitrate
        )
    
    @staticmethod
    def compression_params_to_params_dict(params: CompressionParams) -> Dict[str, Any]:
        """Convert CompressionParams to params dictionary, including audio_bitrate."""
        result = {
            'bitrate': params.bitrate,
            'width': params.width,
            'height': params.height,
            'fps': params.fps,
            'encoder': params.encoder,
            'preset': params.preset,
            'audio_bitrate': max(params.audio_bitrate, 64)  # Ensure minimum
        }
        
        # Add optional fields if present
        if params.crf is not None:
            result['crf'] = params.crf
        if params.tune is not None:
            result['tune'] = params.tune
        if params.gop_size is not None:
            result['gop_size'] = params.gop_size
        if params.b_frames is not None:
            result['b_frames'] = params.b_frames
        
        return result
    
    def _get_config(self, key: str, default: Any) -> Any:
        """Get configuration value with fallback to default."""
        if self.config:
            return self.config.get(f'quality_improvement.{key}', default)
        return default
    
    def analyze_quality_failure(
        self, 
        quality_result: Dict[str, Any], 
        artifact_result: Dict[str, Any],
        current_size_mb: float,
        size_limit_mb: float,
        compression_params: Optional[Dict[str, Any]] = None,
        video_characteristics: Optional[Dict[str, Any]] = None
    ) -> FailureAnalysis:
        """Comprehensive analysis of why quality evaluation failed.
        
        Args:
            quality_result: Results from VMAF/SSIM evaluation
            artifact_result: Results from artifact detection
            current_size_mb: Current output file size
            size_limit_mb: Maximum allowed file size
            compression_params: Current compression parameters
            video_characteristics: Video content characteristics
            
        Returns:
            FailureAnalysis with comprehensive failure analysis and improvement strategies
        """
        logger.info("Performing comprehensive quality failure analysis")
        
        # Extract all available scores
        vmaf_score = quality_result.get('vmaf_score')
        ssim_score = quality_result.get('ssim_score')
        blockiness_score = artifact_result.get('blockiness_score')
        banding_score = artifact_result.get('banding_score')
        
        # Extract additional artifact scores if available
        blur_score = artifact_result.get('blur_score')
        noise_score = artifact_result.get('noise_score')
        
        # Perform detailed failure type analysis
        failure_types = self._analyze_failure_types(
            vmaf_score, ssim_score, blockiness_score, banding_score, blur_score, noise_score
        )
        
        # Perform root cause analysis
        root_causes = self._analyze_root_causes(
            failure_types, compression_params, video_characteristics, 
            vmaf_score, ssim_score, blockiness_score, banding_score
        )
        
        # Determine primary issue and root cause
        primary_issue, primary_root_cause = self._determine_primary_issue_and_cause(
            failure_types, root_causes, vmaf_score, ssim_score, blockiness_score, banding_score
        )
        
        # Calculate severity and confidence
        severity = self._calculate_severity(failure_types, vmaf_score, ssim_score, blockiness_score, banding_score)
        confidence = self._calculate_analysis_confidence(quality_result, artifact_result, compression_params)
        
        # Calculate size headroom and improvement potential
        size_headroom_mb = size_limit_mb - current_size_mb
        can_improve = size_headroom_mb > 0.1  # Need at least 100KB headroom
        improvement_potential = self._calculate_improvement_potential(
            failure_types, root_causes, size_headroom_mb, compression_params
        )
        
        # Generate detailed analysis
        detailed_analysis = self._generate_detailed_analysis(
            failure_types, root_causes, compression_params, video_characteristics,
            vmaf_score, ssim_score, blockiness_score, banding_score
        )
        
        # Recommend improvement strategies based on comprehensive analysis
        recommended_strategies = self._recommend_strategies_comprehensive(
            failure_types, root_causes, severity, size_headroom_mb, improvement_potential
        )
        
        analysis = FailureAnalysis(
            failure_types=failure_types,
            root_causes=root_causes,
            vmaf_score=vmaf_score,
            ssim_score=ssim_score,
            blockiness_score=blockiness_score,
            banding_score=banding_score,
            blur_score=blur_score,
            noise_score=noise_score,
            primary_issue=primary_issue,
            primary_root_cause=primary_root_cause,
            severity=severity,
            confidence=confidence,
            recommended_strategies=recommended_strategies,
            size_headroom_mb=size_headroom_mb,
            can_improve=can_improve,
            detailed_analysis=detailed_analysis,
            improvement_potential=improvement_potential
        )
        
        logger.info(f"Quality failure analysis complete: primary_issue={primary_issue.value}, "
                   f"root_cause={primary_root_cause.value}, severity={severity:.2f}, "
                   f"confidence={confidence:.2f}, improvement_potential={improvement_potential:.2f}")
        
        return analysis
    
    def _analyze_failure_types(
        self,
        vmaf_score: Optional[float],
        ssim_score: Optional[float],
        blockiness_score: Optional[float],
        banding_score: Optional[float],
        blur_score: Optional[float],
        noise_score: Optional[float]
    ) -> List[FailureType]:
        """Analyze and identify all quality failure types."""
        failure_types = []
        
        # VMAF analysis (perceptual quality)
        if vmaf_score is not None:
            if vmaf_score < 60:
                failure_types.append(FailureType.LOW_VMAF)
            elif vmaf_score < 80:
                failure_types.append(FailureType.LOW_VMAF)
        
        # SSIM analysis (structural similarity)
        if ssim_score is not None:
            if ssim_score < 0.90:
                failure_types.append(FailureType.LOW_SSIM)
            elif ssim_score < 0.94:
                failure_types.append(FailureType.LOW_SSIM)
        
        # Artifact-specific analysis
        if blockiness_score is not None and blockiness_score > 0.12:
            failure_types.append(FailureType.HIGH_BLOCKINESS)
            failure_types.append(FailureType.COMPRESSION_ARTIFACTS)
        
        if banding_score is not None and banding_score > 0.10:
            failure_types.append(FailureType.HIGH_BANDING)
            failure_types.append(FailureType.COMPRESSION_ARTIFACTS)
        
        if blur_score is not None and blur_score > 0.15:
            failure_types.append(FailureType.BLUR_ARTIFACTS)
        
        if noise_score is not None and noise_score > 0.20:
            failure_types.append(FailureType.NOISE_ARTIFACTS)
        
        # Temporal artifacts (inferred from multiple issues)
        if len([ft for ft in failure_types if ft in [FailureType.HIGH_BLOCKINESS, FailureType.BLUR_ARTIFACTS]]) >= 2:
            failure_types.append(FailureType.TEMPORAL_ARTIFACTS)
        
        # Combined quality issues
        if len(failure_types) >= 3:
            failure_types.append(FailureType.COMBINED_QUALITY)
        elif len(failure_types) == 0:
            failure_types.append(FailureType.COMBINED_QUALITY)
        
        return list(set(failure_types))  # Remove duplicates
    
    def _analyze_root_causes(
        self,
        failure_types: List[FailureType],
        compression_params: Optional[Dict[str, Any]],
        video_characteristics: Optional[Dict[str, Any]],
        vmaf_score: Optional[float],
        ssim_score: Optional[float],
        blockiness_score: Optional[float],
        banding_score: Optional[float]
    ) -> List[RootCause]:
        """Analyze root causes of quality failures."""
        root_causes = []
        
        # Analyze compression parameters if available
        if compression_params:
            bitrate = compression_params.get('bitrate', 0)
            resolution = compression_params.get('resolution', (0, 0))
            fps = compression_params.get('fps', 30)
            encoder = compression_params.get('encoder', '')
            preset = compression_params.get('preset', '')
            
            # Bitrate analysis
            if bitrate < 500:
                root_causes.append(RootCause.INSUFFICIENT_BITRATE)
            elif bitrate < 1000 and (vmaf_score and vmaf_score < 70):
                root_causes.append(RootCause.INSUFFICIENT_BITRATE)
            
            # Encoder settings analysis
            if encoder == 'libx264' and preset in ['ultrafast', 'superfast']:
                root_causes.append(RootCause.POOR_ENCODER_SETTINGS)
            elif 'nvenc' in encoder and preset in ['p1', 'p2']:
                root_causes.append(RootCause.POOR_ENCODER_SETTINGS)
            
            # Aggressive compression detection
            if video_characteristics:
                original_bitrate = video_characteristics.get('original_bitrate', 0)
                if original_bitrate > 0 and bitrate < original_bitrate * 0.1:
                    root_causes.append(RootCause.AGGRESSIVE_COMPRESSION)
        
        # Artifact-based root cause analysis
        if FailureType.HIGH_BLOCKINESS in failure_types:
            root_causes.extend([RootCause.INSUFFICIENT_BITRATE, RootCause.AGGRESSIVE_COMPRESSION])
        
        if FailureType.HIGH_BANDING in failure_types:
            root_causes.extend([RootCause.CODEC_LIMITATIONS, RootCause.AGGRESSIVE_COMPRESSION])
        
        if FailureType.BLUR_ARTIFACTS in failure_types:
            root_causes.extend([RootCause.PREPROCESSING_ISSUES, RootCause.POOR_ENCODER_SETTINGS])
        
        if FailureType.TEMPORAL_ARTIFACTS in failure_types:
            root_causes.append(RootCause.TEMPORAL_COMPRESSION)
        
        # Content complexity analysis
        if video_characteristics:
            motion_level = video_characteristics.get('motion_level', 'medium')
            complexity = video_characteristics.get('complexity', 'medium')
            
            if motion_level == 'high' and (vmaf_score and vmaf_score < 75):
                root_causes.append(RootCause.CONTENT_COMPLEXITY)
            
            if complexity == 'high' and (ssim_score and ssim_score < 0.92):
                root_causes.append(RootCause.CONTENT_COMPLEXITY)
        
        return list(set(root_causes))  # Remove duplicates
    
    def _determine_primary_issue_and_cause(
        self,
        failure_types: List[FailureType],
        root_causes: List[RootCause],
        vmaf_score: Optional[float],
        ssim_score: Optional[float],
        blockiness_score: Optional[float],
        banding_score: Optional[float]
    ) -> Tuple[FailureType, RootCause]:
        """Determine the primary quality issue and root cause."""
        
        # Calculate severity for each failure type
        issue_severities = []
        
        for failure_type in failure_types:
            if failure_type == FailureType.LOW_VMAF and vmaf_score is not None:
                severity = max(0, (80 - vmaf_score) / 80)
                issue_severities.append((failure_type, severity))
            elif failure_type == FailureType.LOW_SSIM and ssim_score is not None:
                severity = max(0, (0.94 - ssim_score) / 0.94)
                issue_severities.append((failure_type, severity))
            elif failure_type == FailureType.HIGH_BLOCKINESS and blockiness_score is not None:
                severity = min(1.0, (blockiness_score - 0.12) / 0.88)
                issue_severities.append((failure_type, severity))
            elif failure_type == FailureType.HIGH_BANDING and banding_score is not None:
                severity = min(1.0, (banding_score - 0.10) / 0.90)
                issue_severities.append((failure_type, severity))
            else:
                # Default severity for other types
                issue_severities.append((failure_type, 0.5))
        
        # Get primary issue
        if issue_severities:
            primary_issue = max(issue_severities, key=lambda x: x[1])[0]
        else:
            primary_issue = FailureType.COMBINED_QUALITY
        
        # Determine primary root cause based on primary issue
        primary_root_cause = RootCause.INSUFFICIENT_BITRATE  # Default
        
        if primary_issue == FailureType.HIGH_BLOCKINESS:
            if RootCause.INSUFFICIENT_BITRATE in root_causes:
                primary_root_cause = RootCause.INSUFFICIENT_BITRATE
            elif RootCause.AGGRESSIVE_COMPRESSION in root_causes:
                primary_root_cause = RootCause.AGGRESSIVE_COMPRESSION
        elif primary_issue == FailureType.HIGH_BANDING:
            if RootCause.CODEC_LIMITATIONS in root_causes:
                primary_root_cause = RootCause.CODEC_LIMITATIONS
            elif RootCause.AGGRESSIVE_COMPRESSION in root_causes:
                primary_root_cause = RootCause.AGGRESSIVE_COMPRESSION
        elif primary_issue in [FailureType.LOW_VMAF, FailureType.LOW_SSIM]:
            if RootCause.INSUFFICIENT_BITRATE in root_causes:
                primary_root_cause = RootCause.INSUFFICIENT_BITRATE
            elif RootCause.POOR_ENCODER_SETTINGS in root_causes:
                primary_root_cause = RootCause.POOR_ENCODER_SETTINGS
        
        # Use first available root cause if primary not found
        if primary_root_cause not in root_causes and root_causes:
            primary_root_cause = root_causes[0]
        
        return primary_issue, primary_root_cause
    
    def _calculate_severity(
        self,
        failure_types: List[FailureType],
        vmaf_score: Optional[float],
        ssim_score: Optional[float],
        blockiness_score: Optional[float],
        banding_score: Optional[float]
    ) -> float:
        """Calculate overall severity of quality failures."""
        severities = []
        
        # VMAF severity
        if vmaf_score is not None:
            if vmaf_score < 40:
                severities.append(1.0)  # Critical
            elif vmaf_score < 60:
                severities.append(0.8)  # High
            elif vmaf_score < 80:
                severities.append(0.6)  # Medium
            else:
                severities.append(0.3)  # Low
        
        # SSIM severity
        if ssim_score is not None:
            if ssim_score < 0.85:
                severities.append(1.0)  # Critical
            elif ssim_score < 0.90:
                severities.append(0.8)  # High
            elif ssim_score < 0.94:
                severities.append(0.6)  # Medium
            else:
                severities.append(0.3)  # Low
        
        # Artifact severities
        if blockiness_score is not None and blockiness_score > 0.12:
            artifact_severity = min(1.0, (blockiness_score - 0.12) / 0.88)
            severities.append(artifact_severity)
        
        if banding_score is not None and banding_score > 0.10:
            artifact_severity = min(1.0, (banding_score - 0.10) / 0.90)
            severities.append(artifact_severity)
        
        # Combined failure type penalty
        if len(failure_types) >= 3:
            severities.append(0.8)  # Multiple issues increase severity
        
        if not severities:
            return 0.5  # Default moderate severity
        
        # Use weighted average with emphasis on worst issues
        severities.sort(reverse=True)
        if len(severities) == 1:
            return severities[0]
        elif len(severities) == 2:
            return severities[0] * 0.7 + severities[1] * 0.3
        else:
            return severities[0] * 0.5 + severities[1] * 0.3 + sum(severities[2:]) * 0.2 / len(severities[2:])
    
    def _calculate_analysis_confidence(
        self,
        quality_result: Dict[str, Any],
        artifact_result: Dict[str, Any],
        compression_params: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate confidence in the failure analysis."""
        confidence_factors = []
        
        # Quality metrics availability
        if quality_result.get('vmaf_score') is not None:
            confidence_factors.append(0.9)  # High confidence in VMAF
        if quality_result.get('ssim_score') is not None:
            confidence_factors.append(0.8)  # Good confidence in SSIM
        
        # Artifact detection availability
        if artifact_result.get('blockiness_score') is not None:
            confidence_factors.append(0.7)
        if artifact_result.get('banding_score') is not None:
            confidence_factors.append(0.7)
        
        # Parameter information availability
        if compression_params:
            confidence_factors.append(0.6)
        
        # Quality evaluation success
        eval_success = quality_result.get('evaluation_success', False)
        if eval_success:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.3)
        
        if not confidence_factors:
            return 0.3  # Low confidence with no data
        
        # Calculate weighted average
        return sum(confidence_factors) / len(confidence_factors)
    
    def _calculate_improvement_potential(
        self,
        failure_types: List[FailureType],
        root_causes: List[RootCause],
        size_headroom_mb: float,
        compression_params: Optional[Dict[str, Any]]
    ) -> float:
        """Calculate potential for quality improvement."""
        potential_factors = []
        
        # Size headroom factor
        if size_headroom_mb > 2.0:
            potential_factors.append(0.9)  # High potential with lots of headroom
        elif size_headroom_mb > 0.5:
            potential_factors.append(0.7)  # Good potential
        elif size_headroom_mb > 0.1:
            potential_factors.append(0.4)  # Limited potential
        else:
            potential_factors.append(0.1)  # Very limited potential
        
        # Root cause addressability
        addressable_causes = [
            RootCause.INSUFFICIENT_BITRATE,
            RootCause.POOR_ENCODER_SETTINGS,
            RootCause.AGGRESSIVE_COMPRESSION
        ]
        
        addressable_count = sum(1 for cause in root_causes if cause in addressable_causes)
        if addressable_count > 0:
            potential_factors.append(0.8)
        
        # Failure type addressability
        addressable_failures = [
            FailureType.LOW_VMAF,
            FailureType.LOW_SSIM,
            FailureType.HIGH_BLOCKINESS,
            FailureType.HIGH_BANDING
        ]
        
        addressable_failure_count = sum(1 for ft in failure_types if ft in addressable_failures)
        if addressable_failure_count > 0:
            potential_factors.append(0.7)
        
        # Encoder capability factor
        if compression_params:
            encoder = compression_params.get('encoder', '')
            if encoder in ['libx264', 'libx265']:
                potential_factors.append(0.8)  # Software encoders have more options
            elif 'nvenc' in encoder or 'amf' in encoder:
                potential_factors.append(0.6)  # Hardware encoders have fewer options
        
        if not potential_factors:
            return 0.3  # Default low potential
        
        return sum(potential_factors) / len(potential_factors)
    
    def _generate_detailed_analysis(
        self,
        failure_types: List[FailureType],
        root_causes: List[RootCause],
        compression_params: Optional[Dict[str, Any]],
        video_characteristics: Optional[Dict[str, Any]],
        vmaf_score: Optional[float],
        ssim_score: Optional[float],
        blockiness_score: Optional[float],
        banding_score: Optional[float]
    ) -> Dict[str, Any]:
        """Generate detailed analysis information."""
        analysis = {
            'failure_summary': {
                'total_failure_types': len(failure_types),
                'total_root_causes': len(root_causes),
                'has_quality_metrics': vmaf_score is not None or ssim_score is not None,
                'has_artifact_metrics': blockiness_score is not None or banding_score is not None
            },
            'quality_assessment': {},
            'artifact_assessment': {},
            'parameter_assessment': {},
            'recommendations': []
        }
        
        # Quality metrics assessment
        if vmaf_score is not None:
            if vmaf_score < 40:
                quality_level = 'critical'
            elif vmaf_score < 60:
                quality_level = 'poor'
            elif vmaf_score < 80:
                quality_level = 'acceptable'
            else:
                quality_level = 'good'
            
            analysis['quality_assessment']['vmaf'] = {
                'score': vmaf_score,
                'level': quality_level,
                'target': 80,
                'deficit': max(0, 80 - vmaf_score)
            }
        
        if ssim_score is not None:
            if ssim_score < 0.85:
                quality_level = 'critical'
            elif ssim_score < 0.90:
                quality_level = 'poor'
            elif ssim_score < 0.94:
                quality_level = 'acceptable'
            else:
                quality_level = 'good'
            
            analysis['quality_assessment']['ssim'] = {
                'score': ssim_score,
                'level': quality_level,
                'target': 0.94,
                'deficit': max(0, 0.94 - ssim_score)
            }
        
        # Artifact assessment
        if blockiness_score is not None:
            analysis['artifact_assessment']['blockiness'] = {
                'score': blockiness_score,
                'threshold': 0.12,
                'severity': 'high' if blockiness_score > 0.20 else 'medium' if blockiness_score > 0.12 else 'low'
            }
        
        if banding_score is not None:
            analysis['artifact_assessment']['banding'] = {
                'score': banding_score,
                'threshold': 0.10,
                'severity': 'high' if banding_score > 0.18 else 'medium' if banding_score > 0.10 else 'low'
            }
        
        # Parameter assessment
        if compression_params:
            bitrate = compression_params.get('bitrate', 0)
            encoder = compression_params.get('encoder', '')
            preset = compression_params.get('preset', '')
            
            analysis['parameter_assessment'] = {
                'bitrate': {
                    'current': bitrate,
                    'adequacy': 'low' if bitrate < 500 else 'medium' if bitrate < 1500 else 'high'
                },
                'encoder': {
                    'type': encoder,
                    'category': 'software' if encoder in ['libx264', 'libx265'] else 'hardware'
                },
                'preset': {
                    'current': preset,
                    'quality_focus': preset in ['slow', 'slower', 'veryslow', 'quality', 'p6', 'p7']
                }
            }
        
        # Generate specific recommendations
        for root_cause in root_causes:
            if root_cause == RootCause.INSUFFICIENT_BITRATE:
                analysis['recommendations'].append({
                    'type': 'bitrate_increase',
                    'description': 'Increase video bitrate to improve quality',
                    'priority': 'high'
                })
            elif root_cause == RootCause.POOR_ENCODER_SETTINGS:
                analysis['recommendations'].append({
                    'type': 'encoder_optimization',
                    'description': 'Use quality-focused encoder preset',
                    'priority': 'medium'
                })
            elif root_cause == RootCause.AGGRESSIVE_COMPRESSION:
                analysis['recommendations'].append({
                    'type': 'compression_adjustment',
                    'description': 'Reduce compression aggressiveness',
                    'priority': 'high'
                })
        
        return analysis
    
    def _recommend_strategies_comprehensive(
        self,
        failure_types: List[FailureType],
        root_causes: List[RootCause],
        severity: float,
        size_headroom_mb: float,
        improvement_potential: float
    ) -> List[ImprovementStrategy]:
        """Recommend improvement strategies based on comprehensive failure analysis."""
        
        strategies = []
        
        # Root cause-based strategy selection (primary approach)
        if RootCause.INSUFFICIENT_BITRATE in root_causes and size_headroom_mb > 0.5:
            strategies.append(ImprovementStrategy.INCREASE_BITRATE)
        
        if RootCause.POOR_ENCODER_SETTINGS in root_causes:
            strategies.append(ImprovementStrategy.BETTER_ENCODING)
        
        if RootCause.CODEC_LIMITATIONS in root_causes:
            strategies.append(ImprovementStrategy.ADVANCED_CODEC)
        
        if RootCause.TEMPORAL_COMPRESSION in root_causes:
            strategies.append(ImprovementStrategy.TEMPORAL_OPTIMIZATION)
        
        if RootCause.AGGRESSIVE_COMPRESSION in root_causes:
            strategies.extend([
                ImprovementStrategy.INCREASE_BITRATE,
                ImprovementStrategy.BETTER_ENCODING
            ])
        
        # Failure type-based strategy selection (secondary approach)
        if FailureType.HIGH_BLOCKINESS in failure_types:
            strategies.extend([
                ImprovementStrategy.INCREASE_BITRATE,
                ImprovementStrategy.TEMPORAL_OPTIMIZATION
            ])
        
        if FailureType.HIGH_BANDING in failure_types:
            strategies.extend([
                ImprovementStrategy.ADVANCED_CODEC,
                ImprovementStrategy.PERCEPTUAL_TUNING
            ])
        
        if FailureType.BLUR_ARTIFACTS in failure_types:
            strategies.extend([
                ImprovementStrategy.BETTER_ENCODING,
                ImprovementStrategy.PERCEPTUAL_TUNING
            ])
        
        if FailureType.LOW_VMAF in failure_types or FailureType.LOW_SSIM in failure_types:
            strategies.extend([
                ImprovementStrategy.INCREASE_BITRATE,
                ImprovementStrategy.BETTER_ENCODING
            ])
        
        # Severity-based strategy enhancement
        if severity > 0.8:  # Critical quality issues
            strategies.extend([
                ImprovementStrategy.ADVANCED_CODEC,
                ImprovementStrategy.TEMPORAL_OPTIMIZATION,
                ImprovementStrategy.PERCEPTUAL_TUNING
            ])
        elif severity > 0.6:  # High quality issues
            strategies.extend([
                ImprovementStrategy.BETTER_ENCODING,
                ImprovementStrategy.TEMPORAL_OPTIMIZATION
            ])
        
        # Improvement potential-based filtering
        if improvement_potential < 0.3:
            # Low potential - focus on most effective strategies
            high_impact_strategies = [
                ImprovementStrategy.INCREASE_BITRATE,
                ImprovementStrategy.ADVANCED_CODEC
            ]
            strategies = [s for s in strategies if s in high_impact_strategies]
        
        # Size headroom constraints
        if size_headroom_mb < 0.2:
            # Very limited headroom - exclude bitrate increase
            strategies = [s for s in strategies if s != ImprovementStrategy.INCREASE_BITRATE]
        
        # Ensure we always have at least one strategy if improvement is possible
        if not strategies and size_headroom_mb > 0.1:
            if size_headroom_mb > 0.5:
                strategies.append(ImprovementStrategy.INCREASE_BITRATE)
            else:
                strategies.append(ImprovementStrategy.BETTER_ENCODING)
        
        # Remove duplicates while preserving priority order
        seen = set()
        unique_strategies = []
        for strategy in strategies:
            if strategy not in seen:
                seen.add(strategy)
                unique_strategies.append(strategy)
        
        # Limit to top 4 strategies to avoid overwhelming
        return unique_strategies[:4]
    
    def optimize_parameters_with_constraints(
        self,
        input_path: str,
        analysis: FailureAnalysis,
        current_params: CompressionParams,
        size_limit_mb: float,
        duration_seconds: float,
        video_characteristics: Optional[Dict[str, Any]] = None,
        use_multi_dimensional: bool = True
    ) -> Optional[CompressionParams]:
        """
        Optimize compression parameters with size constraints and quality improvement focus.
        
        This is the main entry point for task 3.3 - quality-size optimization engine.
        It implements multi-dimensional parameter optimization, constraint satisfaction,
        iterative improvement with convergence detection, and fallback strategies.
        
        Args:
            input_path: Path to input video
            analysis: Quality failure analysis
            current_params: Current compression parameters
            size_limit_mb: Maximum allowed file size
            duration_seconds: Video duration
            video_characteristics: Video content characteristics
            use_multi_dimensional: Whether to use multi-dimensional optimization
            
        Returns:
            Optimized compression parameters or None if optimization fails
        """
        logger.info("Starting constraint-based parameter optimization")
        
        # Strategy 1: Multi-dimensional optimization (primary approach)
        if use_multi_dimensional and analysis.can_improve:
            optimized_params = self.optimize_with_quality_size_engine(
                input_path=input_path,
                analysis=analysis,
                current_params=current_params,
                size_limit_mb=size_limit_mb,
                duration_seconds=duration_seconds,
                video_characteristics=video_characteristics
            )
            
            if optimized_params:
                logger.info("Multi-dimensional optimization successful")
                return optimized_params
            else:
                logger.warning("Multi-dimensional optimization failed, falling back to iterative approach")
        
        # Strategy 2: Iterative improvement with convergence detection (fallback)
        return self._iterative_improvement_with_convergence(
            analysis=analysis,
            current_params=current_params,
            size_limit_mb=size_limit_mb,
            duration_seconds=duration_seconds
        )
    
    def _iterative_improvement_with_convergence(
        self,
        analysis: FailureAnalysis,
        current_params: CompressionParams,
        size_limit_mb: float,
        duration_seconds: float,
        max_iterations: int = 5
    ) -> Optional[CompressionParams]:
        """
        Perform iterative improvement with convergence detection as fallback strategy.
        
        Args:
            analysis: Quality failure analysis
            current_params: Current compression parameters
            size_limit_mb: Maximum allowed file size
            duration_seconds: Video duration
            max_iterations: Maximum number of improvement iterations
            
        Returns:
            Improved compression parameters or None if no improvement possible
        """
        logger.info("Starting iterative improvement with convergence detection")
        
        best_params = current_params
        best_quality_estimate = analysis.vmaf_score or 0.0
        improvement_history = []
        convergence_threshold = self.quality_improvement_threshold
        
        for iteration in range(max_iterations):
            logger.info(f"Improvement iteration {iteration + 1}/{max_iterations}")
            
            # Generate improvement plan for current iteration
            improvement_plan = self.generate_improvement_plan(
                analysis=analysis,
                current_params=best_params,
                duration_seconds=duration_seconds,
                iteration=iteration
            )
            
            if not improvement_plan or not improvement_plan.feasible:
                logger.info(f"No feasible improvement plan at iteration {iteration + 1}")
                break
            
            # Apply improvement plan
            candidate_params = self.apply_improvement_plan(improvement_plan, best_params)
            
            # Estimate quality improvement and size impact
            estimated_quality = best_quality_estimate + improvement_plan.expected_quality_gain
            estimated_size_increase = improvement_plan.estimated_size_increase_mb
            
            # Check size constraint satisfaction
            current_size_estimate = (best_params.bitrate * duration_seconds) / (8 * 1024)
            new_size_estimate = current_size_estimate + estimated_size_increase
            
            if new_size_estimate > size_limit_mb:
                logger.warning(f"Size constraint violation: {new_size_estimate:.2f}MB > {size_limit_mb:.2f}MB")
                
                # Try to adjust parameters to fit size constraint
                adjusted_params = self._adjust_params_for_size_constraint(
                    candidate_params, size_limit_mb, duration_seconds
                )
                
                if adjusted_params:
                    candidate_params = adjusted_params
                    # Recalculate estimates
                    new_size_estimate = (candidate_params.bitrate * duration_seconds) / (8 * 1024)
                    estimated_quality = best_quality_estimate + (improvement_plan.expected_quality_gain * 0.7)
                else:
                    logger.warning("Cannot adjust parameters to satisfy size constraint")
                    break
            
            # Check for convergence
            quality_improvement = estimated_quality - best_quality_estimate
            improvement_history.append(quality_improvement)
            
            if quality_improvement < convergence_threshold:
                logger.info(f"Convergence detected: improvement {quality_improvement:.2f} < threshold {convergence_threshold:.2f}")
                
                # Check if we've been converging for multiple iterations
                if len(improvement_history) >= 3:
                    recent_improvements = improvement_history[-3:]
                    if all(imp < convergence_threshold for imp in recent_improvements):
                        logger.info("Converged after multiple low-improvement iterations")
                        break
            
            # Update best parameters
            best_params = candidate_params
            best_quality_estimate = estimated_quality
            
            logger.info(f"Iteration {iteration + 1} complete: "
                       f"estimated quality={estimated_quality:.1f}, "
                       f"estimated size={new_size_estimate:.2f}MB, "
                       f"improvement={quality_improvement:.2f}")
        
        # Check if we achieved meaningful improvement
        total_improvement = sum(improvement_history)
        if total_improvement > convergence_threshold:
            logger.info(f"Iterative improvement successful: total improvement={total_improvement:.2f}")
            return best_params
        else:
            logger.warning(f"Iterative improvement failed: total improvement={total_improvement:.2f}")
            return None
    
    def _adjust_params_for_size_constraint(
        self,
        params: CompressionParams,
        size_limit_mb: float,
        duration_seconds: float
    ) -> Optional[CompressionParams]:
        """
        Adjust parameters to satisfy size constraint while minimizing quality loss.
        
        Args:
            params: Compression parameters to adjust
            size_limit_mb: Maximum allowed file size
            duration_seconds: Video duration
            
        Returns:
            Adjusted parameters or None if constraint cannot be satisfied
        """
        # Calculate maximum allowed bitrate
        max_bitrate = int((size_limit_mb * 8 * 1024 * 0.95) / duration_seconds)  # 95% utilization
        
        if params.bitrate <= max_bitrate:
            return params  # Already satisfies constraint
        
        # Create adjusted parameters
        adjusted_params = CompressionParams(
            bitrate=max_bitrate,
            width=params.width,
            height=params.height,
            fps=params.fps,
            encoder=params.encoder,
            preset=params.preset,
            crf=params.crf,
            tune=params.tune,
            gop_size=params.gop_size,
            b_frames=params.b_frames,
            audio_bitrate=max(params.audio_bitrate, 64)  # Preserve audio_bitrate, ensure minimum
        )
        
        # If still too high, try additional adjustments
        if max_bitrate < 500:  # Minimum reasonable bitrate
            logger.warning(f"Required bitrate {max_bitrate}kbps too low for acceptable quality")
            return None
        
        logger.info(f"Adjusted bitrate from {params.bitrate}k to {max_bitrate}k for size constraint")
        return adjusted_params

    def generate_improvement_plan(
        self, 
        analysis: FailureAnalysis, 
        current_params: CompressionParams,
        duration_seconds: float,
        iteration: int = 0
    ) -> Optional[ImprovementPlan]:
        """Generate a specific improvement plan based on failure analysis.
        
        Args:
            analysis: Quality failure analysis
            current_params: Current compression parameters
            duration_seconds: Video duration for bitrate calculations
            iteration: Current improvement iteration (0-based)
            
        Returns:
            ImprovementPlan or None if no improvement is possible
        """
        if not analysis.can_improve or iteration >= len(analysis.recommended_strategies):
            return None
        
        strategy = analysis.recommended_strategies[iteration]
        logger.info(f"Generating improvement plan for strategy: {strategy.value} (iteration {iteration + 1})")
        
        if strategy == ImprovementStrategy.INCREASE_BITRATE:
            return self._plan_bitrate_increase(analysis, current_params, duration_seconds, iteration)
        
        elif strategy == ImprovementStrategy.BETTER_ENCODING:
            return self._plan_better_encoding(analysis, current_params)
        
        elif strategy == ImprovementStrategy.ADVANCED_CODEC:
            return self._plan_codec_upgrade(analysis, current_params)
        
        elif strategy == ImprovementStrategy.TEMPORAL_OPTIMIZATION:
            return self._plan_temporal_optimization(analysis, current_params)
        
        elif strategy == ImprovementStrategy.PERCEPTUAL_TUNING:
            return self._plan_perceptual_tuning(analysis, current_params)
        
        else:
            logger.warning(f"Unknown improvement strategy: {strategy}")
            return None
    
    def _plan_bitrate_increase(
        self, 
        analysis: FailureAnalysis, 
        current_params: CompressionParams,
        duration_seconds: float,
        iteration: int
    ) -> ImprovementPlan:
        """Plan intelligent bitrate increase to improve quality."""
        
        # Analyze current bitrate adequacy
        current_bitrate = current_params.bitrate
        resolution_factor = (current_params.width * current_params.height) / (1920 * 1080)  # Normalize to 1080p
        fps_factor = current_params.fps / 30.0  # Normalize to 30fps
        
        # Calculate recommended bitrate based on content and quality targets
        base_recommended_bitrate = self._calculate_recommended_bitrate(
            analysis, current_params, resolution_factor, fps_factor
        )
        
        # Calculate increase factor based on severity and iteration
        base_factors = self.bitrate_increase_steps
        if iteration < len(base_factors):
            increase_factor = base_factors[iteration]
        else:
            increase_factor = base_factors[-1] * (1 + 0.2 * (iteration - len(base_factors) + 1))
        
        # Adjust factor based on severity and root causes
        severity_multiplier = 1.0 + (analysis.severity * 0.5)
        
        # Additional multiplier for specific root causes
        root_cause_multiplier = 1.0
        if RootCause.INSUFFICIENT_BITRATE in analysis.root_causes:
            root_cause_multiplier = 1.3
        elif RootCause.AGGRESSIVE_COMPRESSION in analysis.root_causes:
            root_cause_multiplier = 1.5
        
        final_factor = increase_factor * severity_multiplier * root_cause_multiplier
        
        # Calculate target bitrate using multiple approaches
        factor_based_bitrate = int(current_bitrate * final_factor)
        recommended_bitrate = max(base_recommended_bitrate, factor_based_bitrate)
        
        # Calculate maximum feasible bitrate based on size headroom
        max_additional_bitrate = int((analysis.size_headroom_mb * self.size_tolerance_factor * 8 * 1024) / duration_seconds)
        feasible_bitrate = min(recommended_bitrate, current_bitrate + max_additional_bitrate)
        
        # Ensure minimum meaningful increase
        min_increase = max(100, int(current_bitrate * 0.1))  # At least 10% or 100kbps
        if feasible_bitrate < current_bitrate + min_increase:
            feasible_bitrate = current_bitrate + min_increase
        
        # Estimate quality gain using improved model
        expected_quality_gain = self._estimate_quality_gain_from_bitrate(
            current_bitrate, feasible_bitrate, analysis, current_params
        )
        
        # Estimate size increase
        estimated_size_increase = (feasible_bitrate - current_bitrate) * duration_seconds / (8 * 1024)
        
        feasible = (feasible_bitrate > current_bitrate and 
                   estimated_size_increase <= analysis.size_headroom_mb and
                   feasible_bitrate <= current_bitrate + max_additional_bitrate)
        
        justification = (f"Increase bitrate from {current_bitrate}k to {feasible_bitrate}k "
                        f"(+{((feasible_bitrate/current_bitrate - 1) * 100):.1f}%) to address "
                        f"{analysis.primary_root_cause.value}. Expected VMAF gain: +{expected_quality_gain:.1f}")
        
        return ImprovementPlan(
            strategy=ImprovementStrategy.INCREASE_BITRATE,
            bitrate_increase_factor=feasible_bitrate / current_bitrate,
            preset_change=None,
            codec_upgrade=None,
            temporal_params={},
            expected_quality_gain=expected_quality_gain,
            estimated_size_increase_mb=estimated_size_increase,
            feasible=feasible,
            justification=justification
        )
    
    def _calculate_recommended_bitrate(
        self,
        analysis: FailureAnalysis,
        current_params: CompressionParams,
        resolution_factor: float,
        fps_factor: float
    ) -> int:
        """Calculate recommended bitrate based on quality targets and content."""
        
        # Base bitrate recommendations (kbps) for 1080p30
        base_bitrates = {
            'low_quality': 1500,
            'medium_quality': 3000,
            'high_quality': 6000,
            'premium_quality': 10000
        }
        
        # Determine target quality level based on current scores
        if analysis.vmaf_score is not None:
            if analysis.vmaf_score < 60:
                target_level = 'premium_quality'
            elif analysis.vmaf_score < 75:
                target_level = 'high_quality'
            elif analysis.vmaf_score < 85:
                target_level = 'medium_quality'
            else:
                target_level = 'low_quality'
        else:
            # Default to medium quality if no VMAF score
            target_level = 'medium_quality'
        
        # Adjust for severity
        if analysis.severity > 0.8:
            target_level = 'premium_quality'
        elif analysis.severity > 0.6 and target_level == 'low_quality':
            target_level = 'medium_quality'
        
        base_bitrate = base_bitrates[target_level]
        
        # Scale for resolution and fps
        recommended_bitrate = int(base_bitrate * resolution_factor * fps_factor)
        
        # Adjust for encoder efficiency
        encoder_efficiency = {
            'libx264': 1.0,
            'libx265': 0.7,  # HEVC is more efficient
            'h264_nvenc': 1.2,  # Hardware encoders need more bitrate
            'h264_amf': 1.2,
            'hevc_nvenc': 0.8,
            'hevc_amf': 0.8
        }
        
        efficiency_factor = encoder_efficiency.get(current_params.encoder, 1.0)
        recommended_bitrate = int(recommended_bitrate * efficiency_factor)
        
        return max(recommended_bitrate, current_params.bitrate)  # Never recommend lower than current
    
    def _estimate_quality_gain_from_bitrate(
        self,
        current_bitrate: int,
        new_bitrate: int,
        analysis: FailureAnalysis,
        current_params: CompressionParams
    ) -> float:
        """Estimate VMAF quality gain from bitrate increase."""
        
        if new_bitrate <= current_bitrate:
            return 0.0
        
        bitrate_ratio = new_bitrate / current_bitrate
        
        # Base quality gain model (logarithmic relationship)
        base_gain = 15.0 * np.log(bitrate_ratio)  # Logarithmic scaling
        
        # Adjust based on current quality level
        if analysis.vmaf_score is not None:
            if analysis.vmaf_score < 50:
                # Low quality videos benefit more from bitrate increases
                quality_multiplier = 1.5
            elif analysis.vmaf_score < 70:
                quality_multiplier = 1.2
            elif analysis.vmaf_score < 85:
                quality_multiplier = 1.0
            else:
                # High quality videos benefit less
                quality_multiplier = 0.7
        else:
            quality_multiplier = 1.0
        
        # Adjust based on encoder efficiency
        encoder_efficiency = {
            'libx264': 1.0,
            'libx265': 1.2,  # HEVC provides better quality per bitrate
            'h264_nvenc': 0.8,  # Hardware encoders less efficient
            'h264_amf': 0.8,
            'hevc_nvenc': 1.1,
            'hevc_amf': 1.1
        }
        
        encoder_factor = encoder_efficiency.get(current_params.encoder, 1.0)
        
        # Adjust based on failure types
        failure_multiplier = 1.0
        if FailureType.HIGH_BLOCKINESS in analysis.failure_types:
            failure_multiplier = 1.3  # Blockiness responds well to bitrate increases
        elif FailureType.HIGH_BANDING in analysis.failure_types:
            failure_multiplier = 0.8  # Banding less responsive to bitrate alone
        
        estimated_gain = base_gain * quality_multiplier * encoder_factor * failure_multiplier
        
        # Cap the estimated gain to realistic values
        return min(estimated_gain, 25.0)  # Maximum 25 VMAF point gain
    
    def _plan_better_encoding(self, analysis: FailureAnalysis, current_params: CompressionParams) -> ImprovementPlan:
        """Plan advanced encoding optimizations for better quality."""
        
        current_encoder = current_params.encoder
        current_preset = current_params.preset
        
        # Enhanced encoder preset mappings with quality rankings
        encoder_presets_quality = {
            'libx264': {
                'ultrafast': 1, 'superfast': 2, 'veryfast': 3, 'faster': 4, 
                'fast': 5, 'medium': 6, 'slow': 7, 'slower': 8, 'veryslow': 9
            },
            'libx265': {
                'ultrafast': 1, 'superfast': 2, 'veryfast': 3, 'faster': 4,
                'fast': 5, 'medium': 6, 'slow': 7, 'slower': 8, 'veryslow': 9
            },
            'h264_nvenc': {
                'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4, 'p5': 5, 'p6': 6, 'p7': 7
            },
            'hevc_nvenc': {
                'p1': 1, 'p2': 2, 'p3': 3, 'p4': 4, 'p5': 5, 'p6': 6, 'p7': 7
            },
            'h264_amf': {
                'speed': 1, 'balanced': 2, 'quality': 3
            },
            'hevc_amf': {
                'speed': 1, 'balanced': 2, 'quality': 3
            }
        }
        
        if current_encoder not in encoder_presets_quality:
            return ImprovementPlan(
                strategy=ImprovementStrategy.BETTER_ENCODING,
                bitrate_increase_factor=1.0,
                preset_change=None,
                codec_upgrade=None,
                temporal_params={},
                expected_quality_gain=0.0,
                estimated_size_increase_mb=0.0,
                feasible=False,
                justification=f"Encoder {current_encoder} not supported for preset optimization"
            )
        
        preset_quality_map = encoder_presets_quality[current_encoder]
        current_quality_level = preset_quality_map.get(current_preset, 3)  # Default to mid-level
        
        # Determine target quality level based on analysis
        target_quality_level = self._determine_target_preset_quality(
            analysis, current_quality_level, len(preset_quality_map)
        )
        
        # Find the best preset for target quality level
        better_preset = None
        for preset, quality_level in preset_quality_map.items():
            if quality_level == target_quality_level:
                better_preset = preset
                break
        
        # If no exact match, find the closest higher quality preset
        if not better_preset:
            for preset, quality_level in preset_quality_map.items():
                if quality_level > current_quality_level:
                    if not better_preset or quality_level < preset_quality_map[better_preset]:
                        better_preset = preset
        
        if better_preset and better_preset != current_preset:
            # Calculate expected quality gain based on preset improvement
            quality_improvement = target_quality_level - current_quality_level
            base_gain = quality_improvement * 1.5  # Base gain per quality level
            
            # Adjust based on encoder type
            if current_encoder in ['libx264', 'libx265']:
                encoder_multiplier = 1.2  # Software encoders benefit more from preset changes
            else:
                encoder_multiplier = 0.8  # Hardware encoders have less preset impact
            
            # Adjust based on failure types
            failure_multiplier = 1.0
            if FailureType.HIGH_BLOCKINESS in analysis.failure_types:
                failure_multiplier = 1.3  # Preset changes help with blockiness
            elif FailureType.BLUR_ARTIFACTS in analysis.failure_types:
                failure_multiplier = 1.4  # Preset changes help with blur
            
            expected_quality_gain = base_gain * encoder_multiplier * failure_multiplier
            expected_quality_gain = min(expected_quality_gain, 8.0)  # Cap at 8 VMAF points
            
            # Estimate encoding time increase (slower presets take more time)
            time_increase_factor = 1.0 + (quality_improvement * 0.3)
            
            # Minimal size increase for preset changes
            estimated_size_increase = 0.05 + (quality_improvement * 0.02)
            
            justification = (f"Upgrade encoding preset from '{current_preset}' to '{better_preset}' "
                           f"(quality level {current_quality_level} → {target_quality_level}) to address "
                           f"{analysis.primary_issue.value}. Expected VMAF gain: +{expected_quality_gain:.1f}")
            
            return ImprovementPlan(
                strategy=ImprovementStrategy.BETTER_ENCODING,
                bitrate_increase_factor=1.0,
                preset_change=better_preset,
                codec_upgrade=None,
                temporal_params={'encoding_time_factor': time_increase_factor},
                expected_quality_gain=expected_quality_gain,
                estimated_size_increase_mb=estimated_size_increase,
                feasible=True,
                justification=justification
            )
        
        # If no preset improvement possible, try advanced encoding options
        advanced_options = self._plan_advanced_encoding_options(analysis, current_params)
        if advanced_options:
            return advanced_options
        
        return ImprovementPlan(
            strategy=ImprovementStrategy.BETTER_ENCODING,
            bitrate_increase_factor=1.0,
            preset_change=None,
            codec_upgrade=None,
            temporal_params={},
            expected_quality_gain=0.0,
            estimated_size_increase_mb=0.0,
            feasible=False,
            justification="Already using optimal encoding settings for this encoder"
        )
    
    def _determine_target_preset_quality(
        self, 
        analysis: FailureAnalysis, 
        current_level: int, 
        max_level: int
    ) -> int:
        """Determine target preset quality level based on failure analysis."""
        
        # Base improvement based on severity
        if analysis.severity > 0.8:
            improvement = 3  # Aggressive improvement for critical issues
        elif analysis.severity > 0.6:
            improvement = 2  # Moderate improvement for high issues
        else:
            improvement = 1  # Conservative improvement for medium issues
        
        # Adjust based on specific failure types
        if FailureType.HIGH_BLOCKINESS in analysis.failure_types:
            improvement += 1  # Preset changes help significantly with blockiness
        
        if FailureType.BLUR_ARTIFACTS in analysis.failure_types:
            improvement += 1  # Preset changes help with blur
        
        # Adjust based on root causes
        if RootCause.POOR_ENCODER_SETTINGS in analysis.root_causes:
            improvement += 2  # Direct indication that settings need improvement
        
        target_level = min(current_level + improvement, max_level)
        return max(target_level, current_level + 1)  # Ensure at least some improvement
    
    def _plan_advanced_encoding_options(
        self, 
        analysis: FailureAnalysis, 
        current_params: CompressionParams
    ) -> Optional[ImprovementPlan]:
        """Plan advanced encoding options when preset changes aren't available."""
        
        temporal_params = {}
        expected_gain = 0.0
        
        # x264/x265 specific optimizations
        if current_params.encoder in ['libx264', 'libx265']:
            # Motion estimation improvements
            if FailureType.HIGH_BLOCKINESS in analysis.failure_types:
                temporal_params['me_method'] = 'umh'  # Better motion estimation
                temporal_params['subme'] = 9  # Higher subpixel motion estimation
                expected_gain += 2.0
            
            # Psychovisual optimizations
            if FailureType.LOW_VMAF in analysis.failure_types:
                temporal_params['psy_rd'] = '1.0:0.2'  # Psychovisual rate-distortion
                temporal_params['aq_mode'] = 3  # Advanced adaptive quantization
                expected_gain += 1.5
            
            # Deblocking filter optimization
            if FailureType.HIGH_BLOCKINESS in analysis.failure_types:
                temporal_params['deblock'] = '1:1'  # Stronger deblocking
                expected_gain += 1.0
            
            # Rate control improvements
            if RootCause.INSUFFICIENT_BITRATE in analysis.root_causes:
                temporal_params['rc_lookahead'] = 60  # Better rate control lookahead
                expected_gain += 1.0
        
        # Hardware encoder optimizations
        elif 'nvenc' in current_params.encoder:
            if FailureType.HIGH_BLOCKINESS in analysis.failure_types:
                temporal_params['spatial_aq'] = True
                temporal_params['temporal_aq'] = True
                expected_gain += 1.5
            
            if FailureType.LOW_VMAF in analysis.failure_types:
                temporal_params['rc_mode'] = 'vbr'  # Variable bitrate for better quality
                expected_gain += 1.0
        
        elif 'amf' in current_params.encoder:
            if FailureType.HIGH_BLOCKINESS in analysis.failure_types:
                temporal_params['quality_preset'] = 'quality'
                expected_gain += 1.0
        
        if temporal_params and expected_gain > 0:
            justification = (f"Apply advanced encoding optimizations for {current_params.encoder} "
                           f"to address {analysis.primary_issue.value}. "
                           f"Expected VMAF gain: +{expected_gain:.1f}")
            
            return ImprovementPlan(
                strategy=ImprovementStrategy.BETTER_ENCODING,
                bitrate_increase_factor=1.0,
                preset_change=None,
                codec_upgrade=None,
                temporal_params=temporal_params,
                expected_quality_gain=expected_gain,
                estimated_size_increase_mb=0.1,
                feasible=True,
                justification=justification
            )
        
        return None
    
    def _plan_codec_upgrade(self, analysis: FailureAnalysis, current_params: CompressionParams) -> ImprovementPlan:
        """Plan intelligent codec upgrade for better compression efficiency."""
        
        current_encoder = current_params.encoder
        
        # Enhanced codec upgrade mappings with efficiency ratings
        codec_upgrades_advanced = {
            'libx264': {
                'primary': 'libx265',
                'efficiency_gain': 0.25,  # 25% better compression
                'quality_gain_base': 8.0,
                'encoding_time_factor': 3.0
            },
            'h264_nvenc': {
                'primary': 'hevc_nvenc',
                'efficiency_gain': 0.20,  # 20% better compression
                'quality_gain_base': 6.0,
                'encoding_time_factor': 1.5
            },
            'h264_amf': {
                'primary': 'hevc_amf',
                'efficiency_gain': 0.20,
                'quality_gain_base': 6.0,
                'encoding_time_factor': 1.5
            },
            'h264_qsv': {
                'primary': 'hevc_qsv',
                'efficiency_gain': 0.18,
                'quality_gain_base': 5.5,
                'encoding_time_factor': 1.4
            }
        }
        
        # Check for alternative codec options
        alternative_codecs = {
            'libx264': ['libx265', 'libaom-av1'],  # AV1 as future option
            'libx265': ['libaom-av1'],
            'h264_nvenc': ['hevc_nvenc', 'av1_nvenc'],
            'hevc_nvenc': ['av1_nvenc']
        }
        
        upgrade_info = codec_upgrades_advanced.get(current_encoder)
        
        if not upgrade_info:
            # Try alternative approach for unsupported encoders
            alternatives = alternative_codecs.get(current_encoder, [])
            if alternatives:
                # Use first available alternative
                upgrade_codec = alternatives[0]
                upgrade_info = {
                    'primary': upgrade_codec,
                    'efficiency_gain': 0.15,  # Conservative estimate
                    'quality_gain_base': 4.0,
                    'encoding_time_factor': 2.0
                }
            else:
                return ImprovementPlan(
                    strategy=ImprovementStrategy.ADVANCED_CODEC,
                    bitrate_increase_factor=1.0,
                    preset_change=None,
                    codec_upgrade=None,
                    temporal_params={},
                    expected_quality_gain=0.0,
                    estimated_size_increase_mb=0.0,
                    feasible=False,
                    justification=f"No codec upgrade available for {current_encoder}"
                )
        
        upgrade_codec = upgrade_info['primary']
        efficiency_gain = upgrade_info['efficiency_gain']
        base_quality_gain = upgrade_info['quality_gain_base']
        encoding_time_factor = upgrade_info['encoding_time_factor']
        
        # Calculate expected quality gain based on failure analysis
        quality_gain_multiplier = 1.0
        
        # Adjust based on failure types that benefit from better codecs
        if FailureType.HIGH_BANDING in analysis.failure_types:
            quality_gain_multiplier += 0.5  # HEVC handles gradients better
        
        if FailureType.HIGH_BLOCKINESS in analysis.failure_types:
            quality_gain_multiplier += 0.3  # Better compression reduces blockiness
        
        if FailureType.LOW_VMAF in analysis.failure_types and analysis.vmaf_score and analysis.vmaf_score < 70:
            quality_gain_multiplier += 0.4  # Significant improvement for low quality
        
        # Adjust based on severity
        severity_multiplier = 1.0 + (analysis.severity * 0.5)
        
        expected_quality_gain = base_quality_gain * quality_gain_multiplier * severity_multiplier
        expected_quality_gain = min(expected_quality_gain, 15.0)  # Cap at 15 VMAF points
        
        # Calculate size impact (negative means size reduction)
        current_bitrate = current_params.bitrate
        duration_estimate = 60  # Assume 60 seconds for estimation
        
        # HEVC typically reduces size at same quality, or improves quality at same size
        size_reduction_mb = (current_bitrate * duration_estimate * efficiency_gain) / (8 * 1024)
        estimated_size_change = -size_reduction_mb  # Negative for size reduction
        
        # Determine optimal preset for new codec
        optimal_preset = self._determine_optimal_preset_for_codec(upgrade_codec, analysis)
        
        # Additional temporal parameters for new codec
        temporal_params = {
            'encoding_time_factor': encoding_time_factor,
            'efficiency_gain': efficiency_gain
        }
        
        # Add codec-specific optimizations
        if upgrade_codec == 'libx265':
            temporal_params.update({
                'x265_params': 'rd=4:psy-rd=2.0:aq-mode=3',  # High quality settings
                'tune': 'psnr' if FailureType.LOW_SSIM in analysis.failure_types else None
            })
        elif 'hevc_nvenc' in upgrade_codec:
            temporal_params.update({
                'rc_mode': 'vbr_hq',  # High quality variable bitrate
                'spatial_aq': True,
                'temporal_aq': True
            })
        elif 'hevc_amf' in upgrade_codec:
            temporal_params.update({
                'quality_preset': 'quality',
                'rate_control_mode': 'vbr_peak'
            })
        
        # Check feasibility based on encoding time constraints
        feasible = True
        feasibility_notes = []
        
        if encoding_time_factor > 4.0:
            feasible = False
            feasibility_notes.append("encoding time increase too high")
        
        if analysis.size_headroom_mb < 0.1 and estimated_size_change > 0:
            feasible = False
            feasibility_notes.append("insufficient size headroom")
        
        justification_parts = [
            f"Upgrade codec from '{current_encoder}' to '{upgrade_codec}'",
            f"for {efficiency_gain*100:.0f}% better compression efficiency"
        ]
        
        if optimal_preset:
            justification_parts.append(f"using '{optimal_preset}' preset")
        
        justification_parts.append(f"Expected VMAF gain: +{expected_quality_gain:.1f}")
        
        if estimated_size_change < 0:
            justification_parts.append(f"with {abs(estimated_size_change):.1f}MB size reduction")
        
        if not feasible:
            justification_parts.append(f"(Limited by: {', '.join(feasibility_notes)})")
        
        justification = " ".join(justification_parts)
        
        return ImprovementPlan(
            strategy=ImprovementStrategy.ADVANCED_CODEC,
            bitrate_increase_factor=1.0,
            preset_change=optimal_preset,
            codec_upgrade=upgrade_codec,
            temporal_params=temporal_params,
            expected_quality_gain=expected_quality_gain,
            estimated_size_increase_mb=estimated_size_change,
            feasible=feasible,
            justification=justification
        )
    
    def _determine_optimal_preset_for_codec(self, codec: str, analysis: FailureAnalysis) -> Optional[str]:
        """Determine optimal preset for the target codec based on quality requirements."""
        
        preset_recommendations = {
            'libx265': {
                'high_quality': 'slow',
                'balanced': 'medium',
                'fast': 'fast'
            },
            'hevc_nvenc': {
                'high_quality': 'p7',
                'balanced': 'p5',
                'fast': 'p3'
            },
            'hevc_amf': {
                'high_quality': 'quality',
                'balanced': 'balanced',
                'fast': 'speed'
            }
        }
        
        if codec not in preset_recommendations:
            return None
        
        # Determine quality requirement based on analysis
        if analysis.severity > 0.7 or FailureType.LOW_VMAF in analysis.failure_types:
            quality_level = 'high_quality'
        elif analysis.severity > 0.4:
            quality_level = 'balanced'
        else:
            quality_level = 'fast'
        
        return preset_recommendations[codec].get(quality_level)
    
    def _plan_temporal_optimization(self, analysis: FailureAnalysis, current_params: CompressionParams) -> ImprovementPlan:
        """Plan advanced temporal encoding optimizations."""
        
        temporal_params = {}
        expected_gain = 0.0
        optimization_details = []
        
        # GOP (Group of Pictures) optimization
        current_gop = current_params.gop_size or int(current_params.fps * 2)
        optimal_gop = self._calculate_optimal_gop_size(analysis, current_params)
        
        if optimal_gop != current_gop:
            temporal_params['gop_size'] = optimal_gop
            expected_gain += 1.0
            optimization_details.append(f"GOP: {current_gop} → {optimal_gop}")
        
        # B-frame optimization
        current_b_frames = current_params.b_frames or 0
        optimal_b_frames = self._calculate_optimal_b_frames(analysis, current_params)
        
        if optimal_b_frames != current_b_frames:
            temporal_params['b_frames'] = optimal_b_frames
            expected_gain += 1.5
            optimization_details.append(f"B-frames: {current_b_frames} → {optimal_b_frames}")
        
        # Reference frame optimization
        if current_params.encoder in ['libx264', 'libx265']:
            optimal_ref_frames = self._calculate_optimal_ref_frames(analysis, current_params)
            if optimal_ref_frames:
                temporal_params['ref_frames'] = optimal_ref_frames
                expected_gain += 0.8
                optimization_details.append(f"Ref frames: {optimal_ref_frames}")
        
        # Motion estimation optimization
        if FailureType.HIGH_BLOCKINESS in analysis.failure_types or FailureType.TEMPORAL_ARTIFACTS in analysis.failure_types:
            me_optimizations = self._plan_motion_estimation_optimization(current_params)
            temporal_params.update(me_optimizations)
            if me_optimizations:
                expected_gain += 1.2
                optimization_details.append("Enhanced motion estimation")
        
        # Rate control optimization
        if RootCause.INSUFFICIENT_BITRATE in analysis.root_causes:
            rc_optimizations = self._plan_rate_control_optimization(analysis, current_params)
            temporal_params.update(rc_optimizations)
            if rc_optimizations:
                expected_gain += 1.0
                optimization_details.append("Improved rate control")
        
        # Adaptive quantization optimization
        if FailureType.HIGH_BANDING in analysis.failure_types or FailureType.LOW_VMAF in analysis.failure_types:
            aq_optimizations = self._plan_adaptive_quantization(analysis, current_params)
            temporal_params.update(aq_optimizations)
            if aq_optimizations:
                expected_gain += 0.8
                optimization_details.append("Adaptive quantization")
        
        # Deblocking filter optimization
        if FailureType.HIGH_BLOCKINESS in analysis.failure_types:
            deblock_optimizations = self._plan_deblocking_optimization(analysis, current_params)
            temporal_params.update(deblock_optimizations)
            if deblock_optimizations:
                expected_gain += 0.6
                optimization_details.append("Deblocking filter")
        
        # Calculate final expected gain with diminishing returns
        if expected_gain > 0:
            # Apply diminishing returns for multiple optimizations
            num_optimizations = len(optimization_details)
            if num_optimizations > 1:
                diminishing_factor = 1.0 - (num_optimizations - 1) * 0.1
                expected_gain *= max(diminishing_factor, 0.7)
            
            # Adjust based on severity
            severity_multiplier = 1.0 + (analysis.severity * 0.3)
            expected_gain *= severity_multiplier
            
            # Cap the gain
            expected_gain = min(expected_gain, 6.0)
        
        # Estimate size impact (temporal optimizations usually have minimal impact)
        estimated_size_increase = len(temporal_params) * 0.02  # Small increase per optimization
        
        feasible = len(temporal_params) > 0
        
        if feasible:
            justification = (f"Apply temporal optimizations: {', '.join(optimization_details)}. "
                           f"Expected VMAF gain: +{expected_gain:.1f}")
        else:
            justification = "No beneficial temporal optimizations available for current configuration"
        
        return ImprovementPlan(
            strategy=ImprovementStrategy.TEMPORAL_OPTIMIZATION,
            bitrate_increase_factor=1.0,
            preset_change=None,
            codec_upgrade=None,
            temporal_params=temporal_params,
            expected_quality_gain=expected_gain,
            estimated_size_increase_mb=estimated_size_increase,
            feasible=feasible,
            justification=justification
        )
    
    def _calculate_optimal_gop_size(self, analysis: FailureAnalysis, current_params: CompressionParams) -> int:
        """Calculate optimal GOP size based on content and quality requirements."""
        
        fps = current_params.fps
        
        # Base GOP size recommendations
        if FailureType.TEMPORAL_ARTIFACTS in analysis.failure_types:
            # Shorter GOP for temporal issues
            base_gop = int(fps * 1.0)  # 1 second
        elif FailureType.HIGH_BLOCKINESS in analysis.failure_types:
            # Medium GOP for blockiness issues
            base_gop = int(fps * 1.5)  # 1.5 seconds
        else:
            # Standard GOP for general quality
            base_gop = int(fps * 2.0)  # 2 seconds
        
        # Adjust based on frame rate
        if fps >= 60:
            base_gop = min(base_gop, 120)  # Cap at 120 for high fps
        elif fps <= 24:
            base_gop = max(base_gop, 24)   # Minimum 24 for low fps
        
        # Ensure GOP is reasonable
        return max(12, min(base_gop, 300))
    
    def _calculate_optimal_b_frames(self, analysis: FailureAnalysis, current_params: CompressionParams) -> int:
        """Calculate optimal number of B-frames."""
        
        encoder = current_params.encoder
        
        # Encoder-specific B-frame recommendations
        if encoder in ['libx264', 'libx265']:
            if FailureType.HIGH_BLOCKINESS in analysis.failure_types:
                return 4  # More B-frames help with compression efficiency
            elif FailureType.TEMPORAL_ARTIFACTS in analysis.failure_types:
                return 2  # Fewer B-frames for temporal stability
            else:
                return 3  # Balanced default
        elif 'nvenc' in encoder:
            # Hardware encoders typically support fewer B-frames
            if FailureType.HIGH_BLOCKINESS in analysis.failure_types:
                return 3
            else:
                return 2
        elif 'amf' in encoder:
            # AMD hardware encoder
            return 2
        else:
            return 2  # Conservative default
    
    def _calculate_optimal_ref_frames(self, analysis: FailureAnalysis, current_params: CompressionParams) -> Optional[int]:
        """Calculate optimal number of reference frames."""
        
        if FailureType.HIGH_BLOCKINESS in analysis.failure_types:
            return 6  # More reference frames for better compression
        elif FailureType.TEMPORAL_ARTIFACTS in analysis.failure_types:
            return 4  # Moderate reference frames for stability
        elif analysis.severity > 0.7:
            return 5  # High quality requirement
        else:
            return None  # Use encoder default
    
    def _plan_motion_estimation_optimization(self, current_params: CompressionParams) -> Dict[str, Any]:
        """Plan motion estimation optimizations."""
        
        optimizations = {}
        encoder = current_params.encoder
        
        if encoder == 'libx264':
            optimizations.update({
                'me_method': 'umh',      # Uneven multi-hexagon search
                'subme': 9,              # High subpixel motion estimation
                'me_range': 24,          # Extended motion search range
                'trellis': 2             # Trellis quantization
            })
        elif encoder == 'libx265':
            optimizations.update({
                'me': 'star',            # Star motion estimation
                'subme': 4,              # High subpixel refinement
                'me_range': 32,          # Extended search range
                'rd': 4                  # High RD optimization level
            })
        elif 'nvenc' in encoder:
            optimizations.update({
                'rc_mode': 'vbr_hq',     # High quality VBR
                'multipass': 'fullres',  # Full resolution multipass
                'spatial_aq': True,      # Spatial adaptive quantization
                'temporal_aq': True      # Temporal adaptive quantization
            })
        
        return optimizations
    
    def _plan_rate_control_optimization(self, analysis: FailureAnalysis, current_params: CompressionParams) -> Dict[str, Any]:
        """Plan rate control optimizations."""
        
        optimizations = {}
        encoder = current_params.encoder
        
        if encoder in ['libx264', 'libx265']:
            optimizations.update({
                'rc_lookahead': 60,      # Extended lookahead
                'mbtree': True,          # Macroblock tree rate control
                'qcomp': 0.6             # Quantizer compression
            })
            
            if analysis.severity > 0.7:
                optimizations['vbv_bufsize'] = current_params.bitrate * 2  # Larger VBV buffer
                optimizations['vbv_maxrate'] = int(current_params.bitrate * 1.5)  # Higher max rate
        
        elif 'nvenc' in encoder:
            optimizations.update({
                'rc_mode': 'vbr_hq',
                'qmin': 15,              # Minimum quantizer
                'qmax': 35,              # Maximum quantizer
                'init_qp': 23            # Initial quantizer
            })
        
        return optimizations
    
    def _plan_adaptive_quantization(self, analysis: FailureAnalysis, current_params: CompressionParams) -> Dict[str, Any]:
        """Plan adaptive quantization optimizations."""
        
        optimizations = {}
        encoder = current_params.encoder
        
        if encoder == 'libx264':
            if FailureType.HIGH_BANDING in analysis.failure_types:
                optimizations.update({
                    'aq_mode': 3,        # Advanced AQ mode
                    'aq_strength': 0.8,  # Strong AQ
                    'psy_rd': '1.0:0.2'  # Psychovisual optimization
                })
            else:
                optimizations.update({
                    'aq_mode': 2,        # Variance AQ
                    'aq_strength': 1.0   # Default strength
                })
        
        elif encoder == 'libx265':
            optimizations.update({
                'aq_mode': 4,            # Advanced AQ mode for x265
                'aq_strength': 0.8,
                'psy_rd': 2.0,           # Psychovisual RD optimization
                'psy_rdoq': 1.0          # Psychovisual RDOQ
            })
        
        return optimizations
    
    def _plan_deblocking_optimization(self, analysis: FailureAnalysis, current_params: CompressionParams) -> Dict[str, Any]:
        """Plan deblocking filter optimizations."""
        
        optimizations = {}
        encoder = current_params.encoder
        
        if encoder in ['libx264', 'libx265']:
            if analysis.blockiness_score and analysis.blockiness_score > 0.2:
                # Strong deblocking for high blockiness
                optimizations['deblock'] = '2:1'
            elif analysis.blockiness_score and analysis.blockiness_score > 0.12:
                # Moderate deblocking
                optimizations['deblock'] = '1:1'
            else:
                # Light deblocking
                optimizations['deblock'] = '1:0'
        
        return optimizations
    
    def _plan_perceptual_tuning(self, analysis: FailureAnalysis, current_params: CompressionParams) -> ImprovementPlan:
        """Plan perceptual quality tuning."""
        
        temporal_params = {}
        
        # Add psychovisual optimizations for x264
        if current_params.encoder == 'libx264':
            temporal_params['psy_rd'] = '1.0:0.15'  # Psychovisual rate-distortion optimization
            temporal_params['aq_mode'] = 2  # Adaptive quantization mode 2
            temporal_params['aq_strength'] = 0.8  # Adaptive quantization strength
        
        # Tune for specific content if banding is detected
        if FailureType.HIGH_BANDING in analysis.failure_types:
            if current_params.tune != 'grain':
                temporal_params['tune'] = 'grain'  # Better for content with grain/noise
        
        expected_quality_gain = 1.5 + (analysis.severity * 1.0)  # 1.5-2.5 VMAF points
        estimated_size_increase = 0.02  # Very minimal size impact
        
        justification = "Apply perceptual tuning optimizations for better visual quality"
        
        return ImprovementPlan(
            strategy=ImprovementStrategy.PERCEPTUAL_TUNING,
            bitrate_increase_factor=1.0,
            preset_change=None,
            codec_upgrade=None,
            temporal_params=temporal_params,
            expected_quality_gain=expected_quality_gain,
            estimated_size_increase_mb=estimated_size_increase,
            feasible=True,
            justification=justification
        )
    
    def apply_improvement_plan(
        self, 
        plan: ImprovementPlan, 
        current_params: CompressionParams
    ) -> CompressionParams:
        """Apply an improvement plan to compression parameters.
        
        Args:
            plan: The improvement plan to apply
            current_params: Current compression parameters
            
        Returns:
            Updated compression parameters
        """
        if not plan.feasible:
            logger.warning(f"Cannot apply infeasible improvement plan: {plan.justification}")
            return current_params
        
        logger.info(f"Applying improvement plan: {plan.justification}")
        
        # Create new parameters based on current ones
        # Preserve audio_bitrate, ensuring minimum of 64kbps
        preserved_audio_bitrate = max(current_params.audio_bitrate, 64)
        
        new_params = CompressionParams(
            bitrate=int(current_params.bitrate * plan.bitrate_increase_factor),
            width=current_params.width,
            height=current_params.height,
            fps=current_params.fps,
            encoder=plan.codec_upgrade or current_params.encoder,
            preset=plan.preset_change or current_params.preset,
            crf=current_params.crf,
            tune=current_params.tune,
            gop_size=current_params.gop_size,
            b_frames=current_params.b_frames,
            audio_bitrate=preserved_audio_bitrate
        )
        
        # Apply temporal parameters
        for key, value in plan.temporal_params.items():
            if hasattr(new_params, key):
                setattr(new_params, key, value)
        
        logger.info(f"Parameters updated: bitrate={new_params.bitrate}k, "
                   f"encoder={new_params.encoder}, preset={new_params.preset}")
        
        return new_params
    
    def optimize_with_quality_size_engine(
        self,
        input_path: str,
        analysis: FailureAnalysis,
        current_params: CompressionParams,
        size_limit_mb: float,
        duration_seconds: float,
        video_characteristics: Optional[Dict[str, Any]] = None
    ) -> Optional[CompressionParams]:
        """
        Use the quality-size optimization engine for multi-dimensional parameter optimization.
        
        Args:
            input_path: Path to input video
            analysis: Quality failure analysis
            current_params: Current compression parameters
            size_limit_mb: Maximum allowed file size
            duration_seconds: Video duration for calculations
            video_characteristics: Video content characteristics
            
        Returns:
            Optimized compression parameters or None if optimization fails
        """
        try:
            # Try relative import first, then absolute import
            try:
                from .quality_size_optimization_engine import (
                    QualitySizeOptimizationEngine, OptimizationConstraints, 
                    ParameterSpace, OptimizationPoint
                )
            except ImportError:
                from quality_size_optimization_engine import (
                    QualitySizeOptimizationEngine, OptimizationConstraints, 
                    ParameterSpace, OptimizationPoint
                )
            
            logger.info("Starting multi-dimensional quality-size optimization")
            
            # Create optimization engine
            optimization_engine = QualitySizeOptimizationEngine(
                config_manager=self.config,
                quality_improvement_engine=self
            )
            
            # Define optimization constraints
            constraints = OptimizationConstraints(
                max_size_mb=size_limit_mb,
                min_quality_vmaf=analysis.vmaf_score or 70.0,
                min_quality_ssim=analysis.ssim_score or 0.90,
                max_iterations=self._get_config('optimization.max_iterations', 10),
                convergence_threshold=self._get_config('optimization.convergence_threshold', 1.0),
                size_tolerance_mb=self._get_config('optimization.size_tolerance_mb', 0.1),
                quality_improvement_threshold=self.quality_improvement_threshold,
                time_budget_seconds=self._get_config('optimization.time_budget_seconds', 600)
            )
            
            # Define parameter space based on current parameters and analysis
            parameter_space = self._create_parameter_space(current_params, analysis, size_limit_mb, duration_seconds)
            
            # Create initial optimization point from current parameters
            initial_point = OptimizationPoint(
                bitrate=current_params.bitrate,
                crf=current_params.crf,
                preset=current_params.preset,
                encoder=current_params.encoder,
                resolution_factor=1.0,
                fps_factor=1.0
            )
            
            # Run optimization
            result = optimization_engine.optimize_quality_size_tradeoff(
                input_path=input_path,
                constraints=constraints,
                parameter_space=parameter_space,
                initial_point=initial_point,
                video_characteristics=video_characteristics
            )
            
            if result.best_point and result.best_point.feasible:
                # Convert optimization point back to compression parameters
                optimized_params = self._optimization_point_to_compression_params(
                    result.best_point, current_params
                )
                
                logger.info(f"Multi-dimensional optimization successful: "
                           f"quality={result.best_point.quality_score:.1f}, "
                           f"size={result.best_point.size_mb:.2f}MB, "
                           f"iterations={result.total_iterations}, "
                           f"convergence={result.convergence_status.value}")
                
                return optimized_params
            else:
                logger.warning("Multi-dimensional optimization failed to find feasible solution")
                return None
                
        except ImportError:
            logger.error("Quality-size optimization engine not available")
            return None
        except Exception as e:
            logger.error(f"Multi-dimensional optimization failed: {e}")
            return None
    
    def _create_parameter_space(
        self,
        current_params: CompressionParams,
        analysis: FailureAnalysis,
        size_limit_mb: float,
        duration_seconds: float
    ) -> 'ParameterSpace':
        """Create parameter space for optimization based on current parameters and constraints."""
        
        try:
            from .quality_size_optimization_engine import ParameterSpace
        except ImportError:
            from quality_size_optimization_engine import ParameterSpace
        
        # Calculate bitrate range based on size constraints and quality requirements
        min_bitrate = max(500, current_params.bitrate // 2)  # At least 500kbps
        max_bitrate_from_size = int((size_limit_mb * 8 * 1024 * 0.9) / duration_seconds)  # 90% size utilization
        max_bitrate = min(max_bitrate_from_size, current_params.bitrate * 3)  # Max 3x current
        
        # Adjust range based on analysis
        if analysis.severity > 0.8:
            # High severity - allow more aggressive bitrate increases
            max_bitrate = min(max_bitrate_from_size, current_params.bitrate * 4)
        elif analysis.size_headroom_mb < 1.0:
            # Limited headroom - be more conservative
            max_bitrate = min(max_bitrate, current_params.bitrate * 1.5)
        
        # CRF range based on current encoder
        if current_params.encoder in ['libx264', 'libx265']:
            crf_range = (18, 32)
        else:
            crf_range = (20, 35)  # Hardware encoders typically use different range
        
        # Preset options based on encoder
        if current_params.encoder == 'libx264':
            preset_options = ['fast', 'medium', 'slow', 'slower']
        elif current_params.encoder == 'libx265':
            preset_options = ['fast', 'medium', 'slow']
        elif 'nvenc' in current_params.encoder:
            preset_options = ['p3', 'p4', 'p5', 'p6']
        elif 'amf' in current_params.encoder:
            preset_options = ['speed', 'balanced', 'quality']
        else:
            preset_options = ['medium', 'slow']
        
        # Encoder options - prefer current encoder but allow upgrades
        encoder_options = [current_params.encoder]
        if current_params.encoder == 'libx264' and 'libx265' in self.codec_upgrades:
            encoder_options.append('libx265')
        elif 'h264_nvenc' in current_params.encoder and 'hevc_nvenc' in self.codec_upgrades:
            encoder_options.append('hevc_nvenc')
        
        # Resolution factors - allow some reduction for size constraints
        resolution_factors = [1.0, 0.9, 0.8]
        if analysis.size_headroom_mb > 2.0:
            resolution_factors = [1.0]  # Keep full resolution if we have headroom
        
        # FPS factors - allow reduction for size constraints
        fps_factors = [1.0, 0.9, 0.8]
        if analysis.size_headroom_mb > 2.0:
            fps_factors = [1.0]  # Keep full FPS if we have headroom
        
        return ParameterSpace(
            bitrate_range=(min_bitrate, max_bitrate),
            crf_range=crf_range,
            preset_options=preset_options,
            encoder_options=encoder_options,
            resolution_factors=resolution_factors,
            fps_factors=fps_factors
        )
    
    def _optimization_point_to_compression_params(
        self,
        point: 'OptimizationPoint',
        base_params: CompressionParams
    ) -> CompressionParams:
        """Convert optimization point back to compression parameters."""
        
        # Calculate adjusted resolution
        new_width = int(base_params.width * point.resolution_factor)
        new_height = int(base_params.height * point.resolution_factor)
        
        # Ensure even dimensions for video encoding
        new_width = new_width - (new_width % 2)
        new_height = new_height - (new_height % 2)
        
        # Calculate adjusted FPS
        new_fps = base_params.fps * point.fps_factor
        
        return CompressionParams(
            bitrate=point.bitrate,
            width=new_width,
            height=new_height,
            fps=new_fps,
            encoder=point.encoder,
            preset=point.preset,
            crf=point.crf,
            tune=base_params.tune,
            gop_size=base_params.gop_size,
            b_frames=base_params.b_frames,
            audio_bitrate=max(base_params.audio_bitrate, 64)  # Preserve audio_bitrate, ensure minimum
        )