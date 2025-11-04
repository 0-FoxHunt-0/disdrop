"""
Quality Prediction Models
Statistical models to predict VMAF and SSIM from fast metrics
"""

import os
import logging
import time
import math
import pickle
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from dataclasses import dataclass
import numpy as np
from pathlib import Path

from lightweight_quality_metrics import LightweightQualityResult, BlurMetrics, NoiseMetrics, ComplexityMetrics, MotionMetrics

logger = logging.getLogger(__name__)


@dataclass
class QualityPrediction:
    """Quality prediction result with confidence scoring."""
    predicted_vmaf: float
    predicted_ssim: float
    vmaf_confidence: float
    ssim_confidence: float
    overall_confidence: float
    model_version: str
    prediction_time: float
    feature_importance: Dict[str, float]


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


class QualityPredictionModels:
    """Statistical models for predicting VMAF and SSIM from lightweight metrics."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        
        # Model configuration
        self.model_version = "1.0"
        self.confidence_threshold = self._get_config('confidence_threshold', 0.7)
        self.skip_full_evaluation_threshold = self._get_config('skip_full_evaluation_threshold', 0.8)
        
        # Initialize models
        self.vmaf_model = self._initialize_vmaf_model()
        self.ssim_model = self._initialize_ssim_model()
        self.confidence_model = self._initialize_confidence_model()
        
        # Feature importance weights (learned from empirical data)
        self.feature_weights = self._initialize_feature_weights()
        
        logger.info(f"Quality Prediction Models initialized (version {self.model_version})")
    
    def _get_config(self, key: str, default: Any) -> Any:
        """Get configuration value with fallback to default."""
        if self.config:
            return self.config.get(f'quality_prediction_models.{key}', default)
        return default
    
    def predict_quality_scores(self, lightweight_result: LightweightQualityResult) -> QualityPrediction:
        """Predict VMAF and SSIM scores from lightweight quality metrics.
        
        Args:
            lightweight_result: Result from lightweight quality analysis
            
        Returns:
            QualityPrediction with predicted scores and confidence
        """
        start_time = time.time()
        
        logger.debug("Predicting quality scores from lightweight metrics")
        
        # Extract features
        features = self._extract_features(lightweight_result)
        
        # Predict VMAF score
        predicted_vmaf, vmaf_confidence = self._predict_vmaf_score(features)
        
        # Predict SSIM score
        predicted_ssim, ssim_confidence = self._predict_ssim_score(features)
        
        # Calculate overall confidence
        overall_confidence = self._calculate_overall_confidence(
            features, vmaf_confidence, ssim_confidence
        )
        
        # Calculate feature importance for this prediction
        feature_importance = self._calculate_feature_importance(features)
        
        prediction_time = time.time() - start_time
        
        prediction = QualityPrediction(
            predicted_vmaf=predicted_vmaf,
            predicted_ssim=predicted_ssim,
            vmaf_confidence=vmaf_confidence,
            ssim_confidence=ssim_confidence,
            overall_confidence=overall_confidence,
            model_version=self.model_version,
            prediction_time=prediction_time,
            feature_importance=feature_importance
        )
        
        logger.info(f"Quality prediction: VMAF≈{predicted_vmaf:.1f} (conf={vmaf_confidence:.2f}), "
                   f"SSIM≈{predicted_ssim:.3f} (conf={ssim_confidence:.2f}), "
                   f"overall_conf={overall_confidence:.2f}")
        
        return prediction
    
    def should_skip_expensive_evaluation(self, prediction: QualityPrediction) -> bool:
        """Determine if expensive full evaluation should be skipped.
        
        Args:
            prediction: Quality prediction result
            
        Returns:
            True if full evaluation can be skipped
        """
        # Skip if high confidence and good predicted quality
        high_confidence = prediction.overall_confidence >= self.skip_full_evaluation_threshold
        good_quality = (
            prediction.predicted_vmaf >= 75.0 and 
            prediction.predicted_ssim >= 0.90
        )
        
        # Also consider individual metric confidence
        reliable_predictions = (
            prediction.vmaf_confidence >= 0.7 and 
            prediction.ssim_confidence >= 0.7
        )
        
        should_skip = high_confidence and good_quality and reliable_predictions
        
        logger.debug(f"Skip evaluation decision: {should_skip} "
                    f"(conf={prediction.overall_confidence:.2f}, "
                    f"vmaf={prediction.predicted_vmaf:.1f}, "
                    f"ssim={prediction.predicted_ssim:.3f})")
        
        return should_skip
    
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
        """Predict VMAF score using statistical model.
        
        Args:
            features: Extracted feature vector
            
        Returns:
            Tuple of (predicted_vmaf, confidence)
        """
        model = self.vmaf_model
        
        # Linear regression model with feature interactions
        predicted_vmaf = (
            model['intercept'] +
            model['blur_coeff'] * features.blur_score +
            model['noise_coeff'] * (1.0 - features.noise_score) +  # Invert noise (less noise = higher quality)
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
        
        # Calculate confidence based on feature reliability
        confidence = self._calculate_vmaf_confidence(features, predicted_vmaf)
        
        return predicted_vmaf, confidence
    
    def _predict_ssim_score(self, features: ModelFeatures) -> Tuple[float, float]:
        """Predict SSIM score using statistical model.
        
        Args:
            features: Extracted feature vector
            
        Returns:
            Tuple of (predicted_ssim, confidence)
        """
        model = self.ssim_model
        
        # SSIM prediction model (more sensitive to structural similarity)
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
        
        # Blur confidence (higher blur score = more confident)
        blur_conf = min(1.0, features.blur_score + 0.3)
        confidence_factors.append(blur_conf)
        
        # Noise confidence (lower noise = more confident)
        noise_conf = 1.0 - features.noise_score
        confidence_factors.append(noise_conf)
        
        # Frame count confidence (more frames = more confident)
        frame_conf = min(1.0, features.frame_count / 10.0)
        confidence_factors.append(frame_conf)
        
        # Prediction range confidence (mid-range predictions are more reliable)
        if 30 <= predicted_vmaf <= 80:
            range_conf = 1.0
        elif 20 <= predicted_vmaf <= 90:
            range_conf = 0.8
        else:
            range_conf = 0.6
        confidence_factors.append(range_conf)
        
        # Feature consistency confidence
        feature_consistency = self._calculate_feature_consistency(features)
        confidence_factors.append(feature_consistency)
        
        # Weighted average of confidence factors
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
        
        # SSIM range confidence (SSIM is typically high for good quality)
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
            min(1.0, features.frame_count / 5.0),  # At least 5 frames for good confidence
            features.temporal_stability,           # Stable video = more reliable
            1.0 - abs(features.processing_time - 2.0) / 10.0,  # Reasonable processing time
        ]
        
        feature_quality = np.mean(feature_quality_factors)
        
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
        blur_consistency = 1.0 - np.std(blur_features) if len(blur_features) > 1 else 1.0
        
        # Check consistency between noise measurements
        noise_features = [
            features.noise_score,
            min(1.0, features.noise_variance / 0.1),
            max(0.0, min(1.0, 1.0 - features.snr_estimate / 40.0))
        ]
        noise_consistency = 1.0 - np.std(noise_features) if len(noise_features) > 1 else 1.0
        
        # Overall consistency
        consistency = (blur_consistency + noise_consistency) / 2.0
        
        return max(0.0, min(1.0, consistency))
    
    def _calculate_feature_importance(self, features: ModelFeatures) -> Dict[str, float]:
        """Calculate feature importance for this specific prediction."""
        
        # Base importance from model weights
        importance = self.feature_weights.copy()
        
        # Adjust importance based on feature values
        # High blur score increases blur importance
        if features.blur_score > 0.7:
            importance['blur_features'] *= 1.2
        
        # High noise increases noise importance
        if features.noise_score > 0.3:
            importance['noise_features'] *= 1.3
        
        # High motion increases temporal importance
        if features.motion_score > 0.5:
            importance['motion_features'] *= 1.1
        
        # Normalize to sum to 1.0
        total_importance = sum(importance.values())
        if total_importance > 0:
            importance = {k: v / total_importance for k, v in importance.items()}
        
        return importance
    
    def _initialize_vmaf_model(self) -> Dict[str, float]:
        """Initialize VMAF prediction model coefficients."""
        # These coefficients are empirically derived and would ideally be learned from training data
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
        # SSIM model coefficients (more focused on structural similarity)
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
    
    def update_model_from_feedback(
        self, 
        features: ModelFeatures, 
        actual_vmaf: float, 
        actual_ssim: float,
        predicted_vmaf: float,
        predicted_ssim: float
    ) -> None:
        """Update model coefficients based on actual vs predicted results.
        
        This would be used in a production system to continuously improve predictions.
        
        Args:
            features: Feature vector used for prediction
            actual_vmaf: Actual VMAF score from full evaluation
            actual_ssim: Actual SSIM score from full evaluation
            predicted_vmaf: Predicted VMAF score
            predicted_ssim: Predicted SSIM score
        """
        logger.debug(f"Model feedback: VMAF error={abs(actual_vmaf - predicted_vmaf):.2f}, "
                    f"SSIM error={abs(actual_ssim - predicted_ssim):.4f}")
        
        # In a full implementation, this would update model coefficients
        # using techniques like online learning or periodic retraining
        
        # For now, just log the feedback for future model improvement
        feedback_data = {
            'features': features,
            'actual_vmaf': actual_vmaf,
            'actual_ssim': actual_ssim,
            'predicted_vmaf': predicted_vmaf,
            'predicted_ssim': predicted_ssim,
            'vmaf_error': abs(actual_vmaf - predicted_vmaf),
            'ssim_error': abs(actual_ssim - predicted_ssim)
        }
        
        # Store feedback for model retraining (implementation-specific)
        self._store_feedback_data(feedback_data)
    
    def _store_feedback_data(self, feedback_data: Dict[str, Any]) -> None:
        """Store feedback data for future model improvement."""
        # In a production system, this would store data to a database or file
        # for periodic model retraining
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current models."""
        return {
            'version': self.model_version,
            'vmaf_model_coefficients': len(self.vmaf_model),
            'ssim_model_coefficients': len(self.ssim_model),
            'feature_weights': self.feature_weights,
            'confidence_threshold': self.confidence_threshold,
            'skip_evaluation_threshold': self.skip_full_evaluation_threshold
        }