"""
Unified Quality Predictor
Provides multiple strategies for predicting video quality metrics (VMAF/SSIM)

This module consolidates functionality from:
- quality_prediction_models.py (statistical prediction from lightweight metrics)
- fast_quality_estimator.py (fast estimation via sampling)
- evaluation_result_predictor.py (risk-based prediction from video characteristics)
"""

import os
import logging
import time
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum

try:
    from .lightweight_quality_metrics import LightweightQualityResult, LightweightQualityMetrics
    from .sampling_engine import SamplingEngine
    from ..logger_setup import get_logger
except ImportError:
    # Fallback for direct execution
    from lightweight_quality_metrics import LightweightQualityResult, LightweightQualityMetrics
    from sampling_engine import SamplingEngine
    from logger_setup import get_logger

logger = get_logger(__name__)


class PredictionStrategy(Enum):
    """Available prediction strategies."""
    STATISTICAL = "statistical"  # From lightweight quality metrics
    FAST_ESTIMATION = "fast_estimation"  # Via sampling + lightweight metrics
    CHARACTERISTICS_BASED = "characteristics_based"  # From video characteristics
    AUTO = "auto"  # Automatically select best strategy


class PredictionConfidence(Enum):
    """Confidence levels for predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


@dataclass
class ModelFeatures:
    """Feature vector for quality prediction models."""
    # Blur features
    blur_score: float
    laplacian_variance: float
    sobel_variance: float
    fft_blur_score: float
    
    # Noise features
    noise_score: float
    noise_variance: float
    snr_estimate: float
    grain_score: float
    
    # Complexity features
    complexity_score: float
    edge_density: float
    texture_complexity: float
    color_variance: float
    gradient_magnitude: float
    
    # Motion features
    motion_score: float
    optical_flow_magnitude: float
    frame_difference_score: float
    temporal_stability: float
    
    # Meta features
    frame_count: int
    processing_time: float


@dataclass
class VideoCharacteristics:
    """Video characteristics for prediction."""
    duration: float
    width: int
    height: int
    fps: float
    bitrate: Optional[float]
    codec: str
    file_size_mb: float
    complexity_score: float
    motion_score: float
    scene_changes: int


@dataclass
class QualityPrediction:
    """Unified quality prediction result."""
    predicted_vmaf: Optional[float]
    predicted_ssim: Optional[float]
    vmaf_confidence: float
    ssim_confidence: float
    overall_confidence: float
    confidence_level: PredictionConfidence
    strategy: PredictionStrategy
    prediction_time: float
    should_skip_evaluation: bool
    prediction_basis: str
    estimated_evaluation_time: float
    risk_factors: List[str]
    feature_importance: Dict[str, float]
    model_version: str = "unified_1.0"


class QualityPredictor:
    """Unified quality predictor with multiple strategies."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        
        # Configuration
        self.confidence_threshold = self._get_config('confidence_threshold', 0.7)
        self.skip_full_evaluation_threshold = self._get_config('skip_full_evaluation_threshold', 0.8)
        self.max_sample_duration = self._get_config('max_sample_duration', 30.0)
        self.enable_content_analysis = self._get_config('enable_content_analysis', True)
        
        # Initialize component modules
        self.lightweight_metrics = LightweightQualityMetrics(config_manager)
        self.sampling_engine = SamplingEngine(config_manager)
        
        # Initialize models
        self.vmaf_model = self._initialize_vmaf_model()
        self.ssim_model = self._initialize_ssim_model()
        self.confidence_model = self._initialize_confidence_model()
        self.feature_weights = self._initialize_feature_weights()
        
        # Characteristics-based model weights
        self.vmaf_char_weights = self._load_vmaf_char_weights()
        self.ssim_char_weights = self._load_ssim_char_weights()
        
        # Historical data for learning
        self.historical_results: List[Dict[str, Any]] = []
        self.max_historical_results = self._get_config('max_historical_results', 100)
        
        logger.info("Unified quality predictor initialized")
    
    def _get_config(self, key: str, default: Any) -> Any:
        """Get configuration value with fallback to default."""
        if self.config:
            return self.config.get(f'quality_predictor.{key}', default)
        return default
    
    def predict_quality(
        self,
        video_path: str,
        strategy: PredictionStrategy = PredictionStrategy.AUTO,
        lightweight_result: Optional[LightweightQualityResult] = None,
        video_characteristics: Optional[VideoCharacteristics] = None,
        target_duration: Optional[float] = None
    ) -> QualityPrediction:
        """
        Predict quality scores using the specified strategy.
        
        Args:
            video_path: Path to video file
            strategy: Prediction strategy to use
            lightweight_result: Pre-computed lightweight quality result (for STATISTICAL strategy)
            video_characteristics: Pre-computed video characteristics (for CHARACTERISTICS_BASED strategy)
            target_duration: Optional duration limit for sampling (for FAST_ESTIMATION strategy)
            
        Returns:
            QualityPrediction with predicted scores and confidence
        """
        start_time = time.time()
        
        # Auto-select strategy if needed
        if strategy == PredictionStrategy.AUTO:
            strategy = self._select_best_strategy(video_path, lightweight_result, video_characteristics)
        
        logger.debug(f"Predicting quality using strategy: {strategy.value}")
        
        # Execute prediction based on strategy
        if strategy == PredictionStrategy.STATISTICAL:
            prediction = self._predict_statistical(video_path, lightweight_result)
        elif strategy == PredictionStrategy.FAST_ESTIMATION:
            prediction = self._predict_fast_estimation(video_path, target_duration)
        elif strategy == PredictionStrategy.CHARACTERISTICS_BASED:
            prediction = self._predict_from_characteristics(video_path, video_characteristics)
        else:
            # Fallback to statistical
            prediction = self._predict_statistical(video_path, lightweight_result)
        
        prediction.prediction_time = time.time() - start_time
        prediction.strategy = strategy
        
        logger.info(f"Quality prediction complete: VMAF≈{prediction.predicted_vmaf:.1f}, "
                   f"SSIM≈{prediction.predicted_ssim:.3f}, confidence={prediction.overall_confidence:.2f}, "
                   f"strategy={strategy.value}, time={prediction.prediction_time:.2f}s")
        
        return prediction
    
    def _select_best_strategy(
        self,
        video_path: str,
        lightweight_result: Optional[LightweightQualityResult],
        video_characteristics: Optional[VideoCharacteristics]
    ) -> PredictionStrategy:
        """Automatically select the best prediction strategy based on available data."""
        # If we have lightweight result, use statistical (most accurate)
        if lightweight_result:
            return PredictionStrategy.STATISTICAL
        
        # If we have video characteristics, use characteristics-based
        if video_characteristics:
            return PredictionStrategy.CHARACTERISTICS_BASED
        
        # Otherwise, use fast estimation (will compute lightweight metrics)
        return PredictionStrategy.FAST_ESTIMATION
    
    def _predict_statistical(
        self,
        video_path: str,
        lightweight_result: Optional[LightweightQualityResult]
    ) -> QualityPrediction:
        """Predict quality using statistical models from lightweight metrics."""
        # Compute lightweight metrics if not provided
        if lightweight_result is None:
            lightweight_result = self.lightweight_metrics.analyze_video_quality(video_path)
        
        # Extract features
        features = self._extract_features(lightweight_result)
        
        # Predict VMAF score
        predicted_vmaf, vmaf_confidence = self._predict_vmaf_score(features)
        
        # Predict SSIM score
        predicted_ssim, ssim_confidence = self._predict_ssim_score(features)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(features, vmaf_confidence, ssim_confidence)
        confidence_level = self._map_confidence_level(overall_confidence)
        
        # Calculate feature importance
        feature_importance = self._calculate_feature_importance(features)
        
        # Determine if evaluation should be skipped
        should_skip = self._should_skip_evaluation(overall_confidence, predicted_vmaf, predicted_ssim)
        
        # Estimate evaluation time
        estimated_time = self._estimate_evaluation_time_from_features(features)
        
        # Risk factors
        risk_factors = self._identify_risk_factors_from_features(features)
        
        return QualityPrediction(
            predicted_vmaf=predicted_vmaf,
            predicted_ssim=predicted_ssim,
            vmaf_confidence=vmaf_confidence,
            ssim_confidence=ssim_confidence,
            overall_confidence=overall_confidence,
            confidence_level=confidence_level,
            strategy=PredictionStrategy.STATISTICAL,
            prediction_time=0.0,  # Will be set by caller
            should_skip_evaluation=should_skip,
            prediction_basis="statistical_model_from_lightweight_metrics",
            estimated_evaluation_time=estimated_time,
            risk_factors=risk_factors,
            feature_importance=feature_importance
        )
    
    def _predict_fast_estimation(
        self,
        video_path: str,
        target_duration: Optional[float]
    ) -> QualityPrediction:
        """Predict quality using fast estimation via sampling."""
        # Generate representative samples
        sample_duration = min(target_duration or self.max_sample_duration, 
                             self._get_video_duration(video_path) or 60.0)
        
        video_segments = self.sampling_engine.generate_representative_samples(video_path, sample_duration)
        
        if not video_segments:
            return self._create_fallback_prediction()
        
        # Extract timestamps from segments
        sample_timestamps = [seg.start_time + seg.duration / 2 for seg in video_segments]
        
        # Analyze quality using lightweight metrics
        lightweight_result = self.lightweight_metrics.analyze_video_quality(video_path, sample_timestamps)
        
        # Use statistical prediction on the sampled data
        prediction = self._predict_statistical(video_path, lightweight_result)
        prediction.strategy = PredictionStrategy.FAST_ESTIMATION
        prediction.prediction_basis = f"fast_estimation_via_sampling_{len(sample_timestamps)}_samples"
        
        return prediction
    
    def _predict_from_characteristics(
        self,
        video_path: str,
        video_characteristics: Optional[VideoCharacteristics]
    ) -> QualityPrediction:
        """Predict quality from video characteristics."""
        # Analyze video characteristics if not provided
        if video_characteristics is None:
            video_characteristics = self._analyze_video_characteristics(video_path)
            if video_characteristics is None:
                return self._create_fallback_prediction()
        
        # Predict VMAF score
        predicted_vmaf = self._predict_vmaf_from_characteristics(video_characteristics)
        
        # Predict SSIM score
        predicted_ssim = self._predict_ssim_from_characteristics(video_characteristics)
        
        # Calculate confidence
        confidence, confidence_score = self._calculate_prediction_confidence(video_characteristics)
        confidence_level = self._map_confidence_level(confidence_score)
        
        # Determine if evaluation should be skipped
        should_skip = self._should_skip_evaluation(confidence_score, predicted_vmaf, predicted_ssim, video_characteristics)
        
        # Estimate evaluation time
        estimated_time = self._estimate_evaluation_time_from_characteristics(video_characteristics)
        
        # Risk factors
        risk_factors = self._identify_risk_factors_from_characteristics(video_characteristics)
        
        # Find similar historical results
        similar_count = len(self._find_similar_historical_results(video_characteristics))
        
        if similar_count >= 3:
            prediction_basis = f"historical_data_{similar_count}_similar"
        elif similar_count >= 1:
            prediction_basis = f"partial_historical_data_{similar_count}_similar"
        else:
            prediction_basis = "heuristic_model_from_characteristics"
        
        return QualityPrediction(
            predicted_vmaf=predicted_vmaf,
            predicted_ssim=predicted_ssim,
            vmaf_confidence=confidence_score,
            ssim_confidence=confidence_score,
            overall_confidence=confidence_score,
            confidence_level=confidence_level,
            strategy=PredictionStrategy.CHARACTERISTICS_BASED,
            prediction_time=0.0,  # Will be set by caller
            should_skip_evaluation=should_skip,
            prediction_basis=prediction_basis,
            estimated_evaluation_time=estimated_time,
            risk_factors=risk_factors,
            feature_importance={}
        )
    
    # Statistical prediction methods (from quality_prediction_models.py)
    
    def _extract_features(self, lightweight_result: LightweightQualityResult) -> ModelFeatures:
        """Extract feature vector from lightweight quality result."""
        blur = lightweight_result.blur_metrics
        noise = lightweight_result.noise_metrics
        complexity = lightweight_result.complexity_metrics
        motion = lightweight_result.motion_metrics
        
        return ModelFeatures(
            # Blur features
            blur_score=blur.overall_blur_score,
            laplacian_variance=blur.laplacian_variance,
            sobel_variance=blur.sobel_variance,
            fft_blur_score=blur.fft_blur_score,
            
            # Noise features
            noise_score=noise.overall_noise_score,
            noise_variance=noise.noise_variance,
            snr_estimate=noise.snr_estimate,
            grain_score=noise.grain_score,
            
            # Complexity features
            complexity_score=complexity.overall_complexity_score,
            edge_density=complexity.edge_density,
            texture_complexity=complexity.texture_complexity,
            color_variance=complexity.color_variance,
            gradient_magnitude=complexity.gradient_magnitude,
            
            # Motion features
            motion_score=motion.overall_motion_score,
            optical_flow_magnitude=motion.optical_flow_magnitude,
            frame_difference_score=motion.frame_difference_score,
            temporal_stability=motion.temporal_stability,
            
            # Meta features
            frame_count=lightweight_result.frame_count,
            processing_time=lightweight_result.processing_time
        )
    
    def _predict_vmaf_score(self, features: ModelFeatures) -> Tuple[float, float]:
        """Predict VMAF score using statistical model."""
        model = self.vmaf_model
        
        # Linear regression model with feature interactions
        predicted_vmaf = (
            model['intercept'] +
            model['blur_coeff'] * features.blur_score +
            model['noise_coeff'] * (1.0 - features.noise_score) +  # Invert noise
            model['complexity_coeff'] * features.complexity_score +
            model['motion_coeff'] * features.temporal_stability +
            model['laplacian_coeff'] * min(1.0, features.laplacian_variance / 100.0) +
            model['snr_coeff'] * min(1.0, features.snr_estimate / 40.0) +
            model['edge_coeff'] * features.edge_density +
            
            # Feature interactions
            model['blur_noise_interaction'] * features.blur_score * (1.0 - features.noise_score) +
            model['complexity_motion_interaction'] * features.complexity_score * features.temporal_stability
        )
        
        # Clamp to valid VMAF range
        predicted_vmaf = max(0.0, min(100.0, predicted_vmaf))
        
        # Calculate confidence
        confidence = self._calculate_vmaf_confidence(features, predicted_vmaf)
        
        return predicted_vmaf, confidence
    
    def _predict_ssim_score(self, features: ModelFeatures) -> Tuple[float, float]:
        """Predict SSIM score using statistical model."""
        model = self.ssim_model
        
        # SSIM prediction model
        predicted_ssim = (
            model['intercept'] +
            model['blur_coeff'] * features.blur_score +
            model['noise_coeff'] * (1.0 - features.noise_score) +
            model['texture_coeff'] * features.texture_complexity +
            model['gradient_coeff'] * min(1.0, features.gradient_magnitude / 100.0) +
            model['stability_coeff'] * features.temporal_stability +
            model['fft_coeff'] * features.fft_blur_score +
            
            # SSIM-specific interactions
            model['structure_interaction'] * features.blur_score * features.texture_complexity +
            model['luminance_interaction'] * (1.0 - features.noise_score) * features.temporal_stability
        )
        
        # Clamp to valid SSIM range
        predicted_ssim = max(0.0, min(1.0, predicted_ssim))
        
        # Calculate confidence
        confidence = self._calculate_ssim_confidence(features, predicted_ssim)
        
        return predicted_ssim, confidence
    
    def _calculate_vmaf_confidence(self, features: ModelFeatures, predicted_vmaf: float) -> float:
        """Calculate confidence in VMAF prediction."""
        confidence_factors = []
        
        # Blur confidence
        blur_conf = min(1.0, features.blur_score + 0.3)
        confidence_factors.append(blur_conf)
        
        # Noise confidence
        noise_conf = 1.0 - features.noise_score
        confidence_factors.append(noise_conf)
        
        # Frame count confidence
        frame_conf = min(1.0, features.frame_count / 10.0)
        confidence_factors.append(frame_conf)
        
        # Prediction range confidence
        if 30 <= predicted_vmaf <= 80:
            range_conf = 1.0
        elif 20 <= predicted_vmaf <= 90:
            range_conf = 0.8
        else:
            range_conf = 0.6
        confidence_factors.append(range_conf)
        
        # Feature consistency
        feature_consistency = self._calculate_feature_consistency(features)
        confidence_factors.append(feature_consistency)
        
        # Weighted average
        weights = [0.25, 0.25, 0.15, 0.20, 0.15]
        confidence = sum(w * c for w, c in zip(weights, confidence_factors))
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_ssim_confidence(self, features: ModelFeatures, predicted_ssim: float) -> float:
        """Calculate confidence in SSIM prediction."""
        confidence_factors = []
        
        # Structural features confidence
        struct_conf = (features.blur_score + features.texture_complexity) / 2.0
        confidence_factors.append(struct_conf)
        
        # Noise confidence
        noise_conf = 1.0 - features.noise_score
        confidence_factors.append(noise_conf)
        
        # Temporal stability confidence
        temporal_conf = features.temporal_stability
        confidence_factors.append(temporal_conf)
        
        # Frame count confidence
        frame_conf = min(1.0, features.frame_count / 10.0)
        confidence_factors.append(frame_conf)
        
        # SSIM range confidence
        if predicted_ssim >= 0.85:
            range_conf = 1.0
        elif predicted_ssim >= 0.70:
            range_conf = 0.8
        else:
            range_conf = 0.6
        confidence_factors.append(range_conf)
        
        # Weighted average
        weights = [0.25, 0.25, 0.20, 0.15, 0.15]
        confidence = sum(w * c for w, c in zip(weights, confidence_factors))
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_overall_confidence(
        self,
        features: ModelFeatures,
        vmaf_confidence: float,
        ssim_confidence: float
    ) -> float:
        """Calculate overall confidence in quality predictions."""
        # Base confidence from individual metrics
        base_confidence = (vmaf_confidence + ssim_confidence) / 2.0
        
        # Adjust based on feature quality
        feature_quality_factors = [
            min(1.0, features.frame_count / 5.0),
            features.temporal_stability,
            1.0 - abs(features.processing_time - 2.0) / 10.0,
        ]
        
        feature_quality = sum(feature_quality_factors) / len(feature_quality_factors)
        
        # Combine base confidence with feature quality
        overall_confidence = base_confidence * 0.7 + feature_quality * 0.3
        
        return max(0.0, min(1.0, overall_confidence))
    
    def _calculate_feature_consistency(self, features: ModelFeatures) -> float:
        """Calculate consistency between different feature measurements."""
        # Check consistency between different blur measurements
        blur_features = [
            features.blur_score,
            min(1.0, features.laplacian_variance / 100.0),
            features.fft_blur_score
        ]
        blur_std = self._calculate_std(blur_features)
        blur_consistency = 1.0 - min(1.0, blur_std) if blur_std > 0 else 1.0
        
        # Check consistency between noise measurements
        noise_features = [
            features.noise_score,
            min(1.0, features.noise_variance / 0.1),
            max(0.0, min(1.0, 1.0 - features.snr_estimate / 40.0))
        ]
        noise_std = self._calculate_std(noise_features)
        noise_consistency = 1.0 - min(1.0, noise_std) if noise_std > 0 else 1.0
        
        # Overall consistency
        consistency = (blur_consistency + noise_consistency) / 2.0
        
        return max(0.0, min(1.0, consistency))
    
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
    
    def _calculate_feature_importance(self, features: ModelFeatures) -> Dict[str, float]:
        """Calculate feature importance for this specific prediction."""
        # Base importance from model weights
        importance = self.feature_weights.copy()
        
        # Adjust importance based on feature values
        if features.blur_score > 0.7:
            importance['blur_features'] *= 1.2
        
        if features.noise_score > 0.3:
            importance['noise_features'] *= 1.3
        
        if features.motion_score > 0.5:
            importance['motion_features'] *= 1.1
        
        # Normalize to sum to 1.0
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    # Characteristics-based prediction methods (from evaluation_result_predictor.py)
    
    def _analyze_video_characteristics(self, video_path: str) -> Optional[VideoCharacteristics]:
        """Analyze video to extract characteristics for prediction."""
        try:
            video_info = self._get_video_info(video_path)
            if not video_info:
                return None
            
            # Calculate file size
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            
            # Analyze content complexity if enabled
            if self.enable_content_analysis:
                complexity_score, motion_score, scene_changes = self._analyze_content_complexity(video_path, video_info)
            else:
                complexity_score = 0.5
                motion_score = 0.5
                scene_changes = max(1, int(video_info['duration'] / 10))
            
            return VideoCharacteristics(
                duration=video_info['duration'],
                width=video_info['width'],
                height=video_info['height'],
                fps=video_info['fps'],
                bitrate=video_info.get('bitrate'),
                codec=video_info.get('codec', 'unknown'),
                file_size_mb=file_size_mb,
                complexity_score=complexity_score,
                motion_score=motion_score,
                scene_changes=scene_changes
            )
        except Exception as e:
            logger.error(f"Failed to analyze video characteristics: {e}")
            return None
    
    def _get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Get video information using ffprobe."""
        try:
            import subprocess
            import json
            
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return None
            
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                return None
            
            # Extract information
            duration = float(data.get('format', {}).get('duration', 0))
            fps_str = video_stream.get('r_frame_rate', '30/1')
            
            # Parse fps
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den) if float(den) != 0 else 30.0
            else:
                fps = float(fps_str)
            
            # Extract bitrate
            bitrate = None
            if 'bit_rate' in video_stream:
                bitrate = float(video_stream['bit_rate'])
            elif 'bit_rate' in data.get('format', {}):
                bitrate = float(data['format']['bit_rate'])
            
            return {
                'duration': duration,
                'fps': fps,
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'bitrate': bitrate,
                'codec': video_stream.get('codec_name', 'unknown')
            }
        except Exception as e:
            logger.debug(f"Error getting video info: {e}")
            return None
    
    def _analyze_content_complexity(self, video_path: str, video_info: Dict[str, Any]) -> Tuple[float, float, int]:
        """Analyze video content complexity and motion."""
        try:
            # Estimate complexity based on resolution and bitrate
            pixel_count = video_info['width'] * video_info['height']
            
            # Complexity heuristics
            if pixel_count > 1920 * 1080:
                base_complexity = 0.7
            elif pixel_count > 1280 * 720:
                base_complexity = 0.5
            else:
                base_complexity = 0.3
            
            # Adjust based on bitrate if available
            if video_info.get('bitrate'):
                bpp = video_info['bitrate'] / (pixel_count * video_info['fps'])
                if bpp < 0.1:
                    base_complexity += 0.2
                elif bpp > 0.5:
                    base_complexity -= 0.1
            
            # Motion estimation based on fps
            if video_info['fps'] > 50:
                motion_score = 0.8
            elif video_info['fps'] > 30:
                motion_score = 0.6
            else:
                motion_score = 0.4
            
            # Scene changes estimation
            scene_changes = max(1, int(video_info['duration'] / 8))
            
            complexity_score = max(0.0, min(1.0, base_complexity))
            motion_score = max(0.0, min(1.0, motion_score))
            
            return complexity_score, motion_score, scene_changes
        except Exception as e:
            logger.debug(f"Content analysis failed: {e}")
            return 0.5, 0.5, 1
    
    def _predict_vmaf_from_characteristics(self, video_chars: VideoCharacteristics) -> Optional[float]:
        """Predict VMAF score from video characteristics."""
        try:
            weights = self.vmaf_char_weights
            
            # Base prediction
            predicted = weights['baseline']
            
            # Resolution factor
            pixel_count = video_chars.width * video_chars.height
            predicted += (pixel_count / 1000000) * weights['resolution_factor']
            
            # Bitrate factor
            if video_chars.bitrate:
                predicted += (video_chars.bitrate / 1000000) * weights['bitrate_factor']
            
            # Complexity penalty
            predicted += video_chars.complexity_score * weights['complexity_penalty']
            
            # Motion penalty
            predicted += video_chars.motion_score * weights['motion_penalty']
            
            # FPS factor
            predicted += video_chars.fps * weights['fps_factor']
            
            # Codec bonus
            if video_chars.codec.lower() in ['hevc', 'h265', 'av1']:
                predicted += weights['codec_bonus']
            
            # Duration penalty
            if video_chars.duration > 300:
                predicted += video_chars.duration * weights['duration_penalty']
            
            # Clamp to valid VMAF range
            return max(0.0, min(100.0, predicted))
        except Exception as e:
            logger.debug(f"VMAF prediction from characteristics failed: {e}")
            return None
    
    def _predict_ssim_from_characteristics(self, video_chars: VideoCharacteristics) -> Optional[float]:
        """Predict SSIM score from video characteristics."""
        try:
            weights = self.ssim_char_weights
            
            # Base prediction
            predicted = weights['baseline']
            
            # Resolution factor
            pixel_count = video_chars.width * video_chars.height
            predicted += (pixel_count / 1000000) * weights['resolution_factor']
            
            # Bitrate factor
            if video_chars.bitrate:
                predicted += (video_chars.bitrate / 1000000) * weights['bitrate_factor']
            
            # Complexity penalty
            predicted += video_chars.complexity_score * weights['complexity_penalty']
            
            # Motion penalty
            predicted += video_chars.motion_score * weights['motion_penalty']
            
            # FPS factor
            predicted += video_chars.fps * weights['fps_factor']
            
            # Codec bonus
            if video_chars.codec.lower() in ['hevc', 'h265', 'av1']:
                predicted += weights['codec_bonus']
            
            # Duration penalty
            if video_chars.duration > 300:
                predicted += video_chars.duration * weights['duration_penalty']
            
            # Clamp to valid SSIM range
            return max(0.0, min(1.0, predicted))
        except Exception as e:
            logger.debug(f"SSIM prediction from characteristics failed: {e}")
            return None
    
    def _calculate_prediction_confidence(
        self,
        video_chars: VideoCharacteristics
    ) -> Tuple[PredictionConfidence, float]:
        """Calculate confidence in prediction based on video characteristics."""
        confidence_factors = []
        
        # Resolution confidence
        pixel_count = video_chars.width * video_chars.height
        if pixel_count >= 1920 * 1080:
            confidence_factors.append(0.9)
        elif pixel_count >= 1280 * 720:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Bitrate confidence
        if video_chars.bitrate:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # Complexity confidence
        complexity_confidence = 1.0 - video_chars.complexity_score * 0.5
        confidence_factors.append(complexity_confidence)
        
        # Motion confidence
        motion_confidence = 1.0 - video_chars.motion_score * 0.3
        confidence_factors.append(motion_confidence)
        
        # Duration confidence
        if video_chars.duration < 60:
            confidence_factors.append(0.9)
        elif video_chars.duration < 300:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Historical data confidence
        similar_results = self._find_similar_historical_results(video_chars)
        if len(similar_results) >= 3:
            confidence_factors.append(0.9)
        elif len(similar_results) >= 1:
            confidence_factors.append(0.6)
        else:
            confidence_factors.append(0.3)
        
        # Calculate overall confidence
        overall_confidence = sum(confidence_factors) / len(confidence_factors)
        
        # Map to confidence levels
        if overall_confidence >= 0.8:
            confidence_level = PredictionConfidence.VERY_HIGH
        elif overall_confidence >= 0.7:
            confidence_level = PredictionConfidence.HIGH
        elif overall_confidence >= 0.5:
            confidence_level = PredictionConfidence.MEDIUM
        else:
            confidence_level = PredictionConfidence.LOW
        
        return confidence_level, overall_confidence
    
    def _should_skip_evaluation(
        self,
        confidence_score: float,
        predicted_vmaf: Optional[float],
        predicted_ssim: Optional[float],
        video_chars: Optional[VideoCharacteristics] = None
    ) -> bool:
        """Determine if evaluation should be skipped based on prediction confidence."""
        # Skip if confidence is very high and predicted quality is good
        if confidence_score >= self.skip_full_evaluation_threshold:
            if predicted_vmaf and predicted_ssim:
                if predicted_vmaf >= 75.0 and predicted_ssim >= 0.90:
                    # Additional checks for characteristics-based
                    if video_chars:
                        if (video_chars.complexity_score < 0.3 and
                            video_chars.motion_score < 0.4 and
                            video_chars.duration < 120):
                            return True
                    else:
                        return True
        
        return False
    
    def _estimate_evaluation_time_from_features(self, features: ModelFeatures) -> float:
        """Estimate evaluation time from features."""
        # Base time estimation
        base_time = 30.0
        
        # Adjust based on frame count
        frame_factor = min(2.0, features.frame_count / 100.0)
        
        # Adjust based on complexity
        complexity_factor = 1.0 + features.complexity_score * 0.5
        
        estimated_time = base_time * frame_factor * complexity_factor
        
        return min(300.0, estimated_time)  # Cap at 5 minutes
    
    def _estimate_evaluation_time_from_characteristics(self, video_chars: VideoCharacteristics) -> float:
        """Estimate evaluation time from video characteristics."""
        base_time = 30.0
        
        # Duration factor
        duration_factor = min(3.0, video_chars.duration / 60.0)
        
        # Resolution factor
        pixel_count = video_chars.width * video_chars.height
        resolution_factor = pixel_count / (1920 * 1080)
        
        # Complexity factor
        complexity_factor = 1.0 + video_chars.complexity_score * 0.5
        
        estimated_time = base_time * duration_factor * resolution_factor * complexity_factor
        
        return min(300.0, estimated_time)
    
    def _identify_risk_factors_from_features(self, features: ModelFeatures) -> List[str]:
        """Identify risk factors from features."""
        risk_factors = []
        
        if features.complexity_score > 0.7:
            risk_factors.append("high_complexity_content")
        
        if features.motion_score > 0.8:
            risk_factors.append("high_motion_content")
        
        if features.blur_score > 0.6:
            risk_factors.append("high_blur")
        
        if features.noise_score > 0.4:
            risk_factors.append("high_noise")
        
        return risk_factors
    
    def _identify_risk_factors_from_characteristics(self, video_chars: VideoCharacteristics) -> List[str]:
        """Identify risk factors from video characteristics."""
        risk_factors = []
        
        if video_chars.complexity_score > 0.7:
            risk_factors.append("high_complexity_content")
        
        if video_chars.motion_score > 0.8:
            risk_factors.append("high_motion_content")
        
        if video_chars.width * video_chars.height > 3840 * 2160:
            risk_factors.append("very_high_resolution")
        
        if video_chars.duration > 600:
            risk_factors.append("very_long_duration")
        
        if video_chars.bitrate and video_chars.bitrate < 500000:
            risk_factors.append("very_low_bitrate")
        
        if video_chars.scene_changes > video_chars.duration / 3:
            risk_factors.append("frequent_scene_changes")
        
        if video_chars.codec == 'unknown':
            risk_factors.append("unknown_codec")
        
        return risk_factors
    
    def _find_similar_historical_results(self, video_chars: VideoCharacteristics) -> List[Dict[str, Any]]:
        """Find similar historical results for comparison."""
        similar_results = []
        
        for result in self.historical_results:
            similarity = self._calculate_similarity(video_chars, result.get('video_chars'))
            if similarity and similarity > 0.7:
                similar_results.append(result)
        
        return similar_results
    
    def _calculate_similarity(
        self,
        chars1: VideoCharacteristics,
        chars2: Optional[VideoCharacteristics]
    ) -> Optional[float]:
        """Calculate similarity between two video characteristics."""
        if chars2 is None:
            return None
        
        similarity_factors = []
        
        # Resolution similarity
        res1 = chars1.width * chars1.height
        res2 = chars2.width * chars2.height
        res_similarity = 1.0 - abs(res1 - res2) / max(res1, res2)
        similarity_factors.append(res_similarity)
        
        # Duration similarity
        dur_similarity = 1.0 - abs(chars1.duration - chars2.duration) / max(chars1.duration, chars2.duration)
        similarity_factors.append(dur_similarity)
        
        # Complexity similarity
        complexity_similarity = 1.0 - abs(chars1.complexity_score - chars2.complexity_score)
        similarity_factors.append(complexity_similarity)
        
        # Motion similarity
        motion_similarity = 1.0 - abs(chars1.motion_score - chars2.motion_score)
        similarity_factors.append(motion_similarity)
        
        # FPS similarity
        fps_similarity = 1.0 - abs(chars1.fps - chars2.fps) / max(chars1.fps, chars2.fps)
        similarity_factors.append(fps_similarity)
        
        return sum(similarity_factors) / len(similarity_factors)
    
    def _get_video_duration(self, video_path: str) -> Optional[float]:
        """Get video duration."""
        video_info = self._get_video_info(video_path)
        return video_info.get('duration') if video_info else None
    
    def _map_confidence_level(self, confidence_score: float) -> PredictionConfidence:
        """Map confidence score to confidence level."""
        if confidence_score >= 0.8:
            return PredictionConfidence.VERY_HIGH
        elif confidence_score >= 0.7:
            return PredictionConfidence.HIGH
        elif confidence_score >= 0.5:
            return PredictionConfidence.MEDIUM
        else:
            return PredictionConfidence.LOW
    
    def _create_fallback_prediction(self) -> QualityPrediction:
        """Create a fallback prediction when analysis fails."""
        return QualityPrediction(
            predicted_vmaf=50.0,
            predicted_ssim=0.80,
            vmaf_confidence=0.1,
            ssim_confidence=0.1,
            overall_confidence=0.1,
            confidence_level=PredictionConfidence.LOW,
            strategy=PredictionStrategy.STATISTICAL,
            prediction_time=0.0,
            should_skip_evaluation=False,
            prediction_basis="fallback_conservative_estimate",
            estimated_evaluation_time=60.0,
            risk_factors=["prediction_failed"],
            feature_importance={}
        )
    
    # Model initialization methods
    
    def _initialize_vmaf_model(self) -> Dict[str, float]:
        """Initialize VMAF prediction model coefficients."""
        return {
            'intercept': 45.0,
            'blur_coeff': 35.0,
            'noise_coeff': 25.0,
            'complexity_coeff': 8.0,
            'motion_coeff': 12.0,
            'laplacian_coeff': 15.0,
            'snr_coeff': 20.0,
            'edge_coeff': 10.0,
            'blur_noise_interaction': 15.0,
            'complexity_motion_interaction': 5.0
        }
    
    def _initialize_ssim_model(self) -> Dict[str, float]:
        """Initialize SSIM prediction model coefficients."""
        return {
            'intercept': 0.75,
            'blur_coeff': 0.15,
            'noise_coeff': 0.12,
            'texture_coeff': 0.08,
            'gradient_coeff': 0.06,
            'stability_coeff': 0.10,
            'fft_coeff': 0.08,
            'structure_interaction': 0.05,
            'luminance_interaction': 0.04
        }
    
    def _initialize_confidence_model(self) -> Dict[str, float]:
        """Initialize confidence calculation model."""
        return {
            'base_confidence': 0.6,
            'blur_weight': 0.25,
            'noise_weight': 0.25,
            'consistency_weight': 0.20,
            'frame_count_weight': 0.15,
            'range_weight': 0.15
        }
    
    def _initialize_feature_weights(self) -> Dict[str, float]:
        """Initialize feature importance weights."""
        return {
            'blur_features': 0.35,
            'noise_features': 0.25,
            'complexity_features': 0.20,
            'motion_features': 0.15,
            'meta_features': 0.05
        }
    
    def _load_vmaf_char_weights(self) -> Dict[str, float]:
        """Load VMAF prediction weights for characteristics-based prediction."""
        if self.config:
            weights = self.config.get('quality_predictor.vmaf_char_weights', {})
            if weights:
                return weights
        
        return {
            'baseline': 75.0,
            'resolution_factor': 0.015,
            'bitrate_factor': 0.008,
            'complexity_penalty': -15.0,
            'motion_penalty': -10.0,
            'fps_factor': 0.1,
            'codec_bonus': 5.0,
            'duration_penalty': -0.05
        }
    
    def _load_ssim_char_weights(self) -> Dict[str, float]:
        """Load SSIM prediction weights for characteristics-based prediction."""
        if self.config:
            weights = self.config.get('quality_predictor.ssim_char_weights', {})
            if weights:
                return weights
        
        return {
            'baseline': 0.92,
            'resolution_factor': 0.00008,
            'bitrate_factor': 0.000005,
            'complexity_penalty': -0.08,
            'motion_penalty': -0.05,
            'fps_factor': 0.0005,
            'codec_bonus': 0.02,
            'duration_penalty': -0.0001
        }
    
    # Learning and feedback methods
    
    def record_evaluation_result(
        self,
        video_chars: Optional[VideoCharacteristics],
        actual_vmaf: Optional[float],
        actual_ssim: Optional[float],
        evaluation_time: float,
        evaluation_success: bool
    ) -> None:
        """Record actual evaluation result for learning."""
        result = {
            'video_chars': video_chars,
            'actual_vmaf': actual_vmaf,
            'actual_ssim': actual_ssim,
            'evaluation_time': evaluation_time,
            'evaluation_success': evaluation_success,
            'timestamp': time.time()
        }
        
        self.historical_results.append(result)
        
        # Limit historical results
        if len(self.historical_results) > self.max_historical_results:
            self.historical_results = self.historical_results[-self.max_historical_results:]
        
        logger.debug(f"Recorded evaluation result: VMAF={actual_vmaf}, SSIM={actual_ssim}, "
                    f"time={evaluation_time:.2f}s, success={evaluation_success}")
    
    def get_prediction_accuracy(self) -> Dict[str, Any]:
        """Get accuracy statistics for predictions."""
        if not self.historical_results:
            return {
                'total_predictions': 0,
                'vmaf_accuracy': None,
                'ssim_accuracy': None,
                'time_accuracy': None
            }
        
        vmaf_errors = []
        ssim_errors = []
        time_errors = []
        
        for result in self.historical_results:
            if result.get('evaluation_success'):
                video_chars = result.get('video_chars')
                if video_chars:
                    prediction = self._predict_from_characteristics(None, video_chars)
                    
                    # VMAF accuracy
                    if result.get('actual_vmaf') and prediction.predicted_vmaf:
                        error = abs(result['actual_vmaf'] - prediction.predicted_vmaf)
                        vmaf_errors.append(error)
                    
                    # SSIM accuracy
                    if result.get('actual_ssim') and prediction.predicted_ssim:
                        error = abs(result['actual_ssim'] - prediction.predicted_ssim)
                        ssim_errors.append(error)
                    
                    # Time accuracy
                    time_error = abs(result.get('evaluation_time', 0) - prediction.estimated_evaluation_time)
                    time_errors.append(time_error)
        
        return {
            'total_predictions': len(self.historical_results),
            'vmaf_accuracy': {
                'mean_absolute_error': sum(vmaf_errors) / len(vmaf_errors) if vmaf_errors else None,
                'predictions_count': len(vmaf_errors)
            },
            'ssim_accuracy': {
                'mean_absolute_error': sum(ssim_errors) / len(ssim_errors) if ssim_errors else None,
                'predictions_count': len(ssim_errors)
            },
            'time_accuracy': {
                'mean_absolute_error': sum(time_errors) / len(time_errors) if time_errors else None,
                'predictions_count': len(time_errors)
            }
        }

