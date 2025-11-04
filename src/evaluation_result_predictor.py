"""
Evaluation Result Predictor
Predicts quality evaluation results based on video characteristics and content analysis
"""

import logging
import os
import subprocess
import json
import time
import math
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class PredictionConfidence(Enum):
    """Confidence levels for predictions."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


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
    """Predicted quality evaluation result."""
    predicted_vmaf: Optional[float]
    predicted_ssim: Optional[float]
    confidence: PredictionConfidence
    confidence_score: float
    should_skip_evaluation: bool
    prediction_basis: str
    estimated_evaluation_time: float
    risk_factors: List[str]


@dataclass
class HistoricalResult:
    """Historical evaluation result for learning."""
    video_chars: VideoCharacteristics
    actual_vmaf: Optional[float]
    actual_ssim: Optional[float]
    evaluation_time: float
    evaluation_success: bool
    timestamp: float


class EvaluationResultPredictor:
    """Predicts quality evaluation results to enable early termination and smart scheduling."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        
        # Configuration
        self.skip_threshold = self._get_config('skip_threshold', 0.8)
        self.confidence_threshold = self._get_config('confidence_threshold', 0.7)
        self.max_historical_results = self._get_config('max_historical_results', 100)
        self.enable_content_analysis = self._get_config('enable_content_analysis', True)
        
        # Historical data for learning
        self.historical_results: List[HistoricalResult] = []
        
        # Prediction models (simple heuristic-based for now)
        self.vmaf_model_weights = self._load_vmaf_model_weights()
        self.ssim_model_weights = self._load_ssim_model_weights()
        
        logger.info("Evaluation result predictor initialized")
    
    def _get_config(self, key: str, default: Any) -> Any:
        """Get configuration value with fallback to default."""
        if self.config:
            return self.config.get(f'quality_evaluation.result_prediction.{key}', default)
        return default
    
    def _load_vmaf_model_weights(self) -> Dict[str, float]:
        """Load VMAF prediction model weights."""
        if self.config:
            weights = self.config.get('quality_evaluation.result_prediction.vmaf_model_weights', {})
            if weights:
                return weights
        
        # Default heuristic weights based on empirical observations
        return {
            'baseline': 75.0,  # Base VMAF score
            'resolution_factor': 0.015,  # Higher resolution generally better
            'bitrate_factor': 0.008,  # Higher bitrate generally better
            'complexity_penalty': -15.0,  # Complex content harder to compress
            'motion_penalty': -10.0,  # High motion content harder to compress
            'fps_factor': 0.1,  # Higher FPS slightly better
            'codec_bonus': 5.0,  # Modern codecs (HEVC) bonus
            'duration_penalty': -0.05  # Longer videos slightly harder
        }
    
    def _load_ssim_model_weights(self) -> Dict[str, float]:
        """Load SSIM prediction model weights."""
        if self.config:
            weights = self.config.get('quality_evaluation.result_prediction.ssim_model_weights', {})
            if weights:
                return weights
        
        # Default heuristic weights
        return {
            'baseline': 0.92,  # Base SSIM score
            'resolution_factor': 0.00008,  # Higher resolution generally better
            'bitrate_factor': 0.000005,  # Higher bitrate generally better
            'complexity_penalty': -0.08,  # Complex content penalty
            'motion_penalty': -0.05,  # High motion penalty
            'fps_factor': 0.0005,  # Higher FPS slightly better
            'codec_bonus': 0.02,  # Modern codecs bonus
            'duration_penalty': -0.0001  # Longer videos slightly harder
        }
    
    def analyze_video_characteristics(self, video_path: str) -> Optional[VideoCharacteristics]:
        """Analyze video to extract characteristics for prediction.
        
        Args:
            video_path: Path to video file
            
        Returns:
            VideoCharacteristics object or None if analysis fails
        """
        try:
            # Get basic video info using ffprobe
            video_info = self._get_video_info(video_path)
            if not video_info:
                return None
            
            # Calculate file size
            file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
            
            # Analyze content complexity if enabled
            if self.enable_content_analysis:
                complexity_score, motion_score, scene_changes = self._analyze_content_complexity(video_path)
            else:
                complexity_score = 0.5  # Default medium complexity
                motion_score = 0.5  # Default medium motion
                scene_changes = max(1, int(video_info['duration'] / 10))  # Estimate scene changes
            
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
    
    def _analyze_content_complexity(self, video_path: str) -> Tuple[float, float, int]:
        """Analyze video content complexity and motion.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Tuple of (complexity_score, motion_score, scene_changes)
        """
        try:
            # Use ffmpeg to analyze content complexity
            # This is a simplified analysis - in production you might use more sophisticated methods
            
            # Analyze spatial complexity using scene detection
            scene_cmd = [
                'ffmpeg', '-i', video_path, '-vf', 'select=gt(scene\\,0.3)', 
                '-f', 'null', '-', '-v', 'quiet', '-stats'
            ]
            
            # For now, use simplified heuristics based on file properties
            video_info = self._get_video_info(video_path)
            if not video_info:
                return 0.5, 0.5, 1
            
            # Estimate complexity based on resolution and bitrate
            pixel_count = video_info['width'] * video_info['height']
            
            # Complexity heuristics
            if pixel_count > 1920 * 1080:
                base_complexity = 0.7  # High resolution = potentially complex
            elif pixel_count > 1280 * 720:
                base_complexity = 0.5
            else:
                base_complexity = 0.3
            
            # Adjust based on bitrate if available
            if video_info['bitrate']:
                bpp = video_info['bitrate'] / (pixel_count * video_info['fps'])
                if bpp < 0.1:
                    base_complexity += 0.2  # Low bpp suggests complex content
                elif bpp > 0.5:
                    base_complexity -= 0.1  # High bpp suggests simple content
            
            # Motion estimation based on fps and duration
            if video_info['fps'] > 50:
                motion_score = 0.8  # High fps suggests motion
            elif video_info['fps'] > 30:
                motion_score = 0.6
            else:
                motion_score = 0.4
            
            # Scene changes estimation
            scene_changes = max(1, int(video_info['duration'] / 8))  # Rough estimate
            
            complexity_score = max(0.0, min(1.0, base_complexity))
            motion_score = max(0.0, min(1.0, motion_score))
            
            return complexity_score, motion_score, scene_changes
        
        except Exception as e:
            logger.debug(f"Content analysis failed: {e}")
            return 0.5, 0.5, 1  # Default values
    
    def predict_quality_scores(self, video_chars: VideoCharacteristics) -> QualityPrediction:
        """Predict quality scores based on video characteristics.
        
        Args:
            video_chars: Video characteristics
            
        Returns:
            QualityPrediction with predicted scores and confidence
        """
        # Predict VMAF score
        predicted_vmaf = self._predict_vmaf_score(video_chars)
        
        # Predict SSIM score
        predicted_ssim = self._predict_ssim_score(video_chars)
        
        # Calculate confidence based on prediction reliability
        confidence, confidence_score = self._calculate_prediction_confidence(video_chars)
        
        # Determine if evaluation should be skipped
        should_skip = self._should_skip_evaluation(confidence_score, video_chars)
        
        # Estimate evaluation time
        estimated_time = self._estimate_evaluation_time(video_chars)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(video_chars)
        
        # Determine prediction basis
        prediction_basis = self._get_prediction_basis(video_chars)
        
        return QualityPrediction(
            predicted_vmaf=predicted_vmaf,
            predicted_ssim=predicted_ssim,
            confidence=confidence,
            confidence_score=confidence_score,
            should_skip_evaluation=should_skip,
            prediction_basis=prediction_basis,
            estimated_evaluation_time=estimated_time,
            risk_factors=risk_factors
        )
    
    def _predict_vmaf_score(self, video_chars: VideoCharacteristics) -> Optional[float]:
        """Predict VMAF score using heuristic model."""
        try:
            weights = self.vmaf_model_weights
            
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
            
            # Codec bonus for modern codecs
            if video_chars.codec.lower() in ['hevc', 'h265', 'av1']:
                predicted += weights['codec_bonus']
            
            # Duration penalty for very long videos
            if video_chars.duration > 300:  # 5 minutes
                predicted += video_chars.duration * weights['duration_penalty']
            
            # Clamp to valid VMAF range
            return max(0.0, min(100.0, predicted))
        
        except Exception as e:
            logger.debug(f"VMAF prediction failed: {e}")
            return None
    
    def _predict_ssim_score(self, video_chars: VideoCharacteristics) -> Optional[float]:
        """Predict SSIM score using heuristic model."""
        try:
            weights = self.ssim_model_weights
            
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
            logger.debug(f"SSIM prediction failed: {e}")
            return None
    
    def _calculate_prediction_confidence(self, video_chars: VideoCharacteristics) -> Tuple[PredictionConfidence, float]:
        """Calculate confidence in prediction based on video characteristics."""
        confidence_factors = []
        
        # Resolution confidence (higher resolution = more predictable)
        pixel_count = video_chars.width * video_chars.height
        if pixel_count >= 1920 * 1080:
            confidence_factors.append(0.9)
        elif pixel_count >= 1280 * 720:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.5)
        
        # Bitrate confidence (known bitrate = more predictable)
        if video_chars.bitrate:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.4)
        
        # Complexity confidence (lower complexity = more predictable)
        complexity_confidence = 1.0 - video_chars.complexity_score * 0.5
        confidence_factors.append(complexity_confidence)
        
        # Motion confidence (lower motion = more predictable)
        motion_confidence = 1.0 - video_chars.motion_score * 0.3
        confidence_factors.append(motion_confidence)
        
        # Duration confidence (shorter videos = more predictable)
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
    
    def _should_skip_evaluation(self, confidence_score: float, video_chars: VideoCharacteristics) -> bool:
        """Determine if evaluation should be skipped based on prediction confidence."""
        # Skip if confidence is very high and video characteristics suggest predictable result
        if confidence_score >= self.skip_threshold:
            # Additional checks for skip decision
            if (video_chars.complexity_score < 0.3 and 
                video_chars.motion_score < 0.4 and 
                video_chars.duration < 120):
                return True
        
        return False
    
    def _estimate_evaluation_time(self, video_chars: VideoCharacteristics) -> float:
        """Estimate time required for quality evaluation."""
        # Base time estimation
        base_time = 30.0  # Base 30 seconds
        
        # Duration factor (longer videos take more time)
        duration_factor = min(3.0, video_chars.duration / 60.0)  # Cap at 3x for very long videos
        
        # Resolution factor (higher resolution takes more time)
        pixel_count = video_chars.width * video_chars.height
        resolution_factor = pixel_count / (1920 * 1080)  # Normalize to 1080p
        
        # Complexity factor (complex content takes more time)
        complexity_factor = 1.0 + video_chars.complexity_score * 0.5
        
        estimated_time = base_time * duration_factor * resolution_factor * complexity_factor
        
        return min(300.0, estimated_time)  # Cap at 5 minutes
    
    def _identify_risk_factors(self, video_chars: VideoCharacteristics) -> List[str]:
        """Identify risk factors that might cause evaluation to fail."""
        risk_factors = []
        
        # High complexity content
        if video_chars.complexity_score > 0.7:
            risk_factors.append("high_complexity_content")
        
        # High motion content
        if video_chars.motion_score > 0.8:
            risk_factors.append("high_motion_content")
        
        # Very high resolution
        if video_chars.width * video_chars.height > 3840 * 2160:
            risk_factors.append("very_high_resolution")
        
        # Very long duration
        if video_chars.duration > 600:  # 10 minutes
            risk_factors.append("very_long_duration")
        
        # Low bitrate (might indicate quality issues)
        if video_chars.bitrate and video_chars.bitrate < 500000:  # 500 kbps
            risk_factors.append("very_low_bitrate")
        
        # Many scene changes
        if video_chars.scene_changes > video_chars.duration / 3:
            risk_factors.append("frequent_scene_changes")
        
        # Unknown codec
        if video_chars.codec == 'unknown':
            risk_factors.append("unknown_codec")
        
        return risk_factors
    
    def _get_prediction_basis(self, video_chars: VideoCharacteristics) -> str:
        """Get description of prediction basis."""
        similar_count = len(self._find_similar_historical_results(video_chars))
        
        if similar_count >= 3:
            return f"historical_data_{similar_count}_similar"
        elif similar_count >= 1:
            return f"partial_historical_data_{similar_count}_similar"
        else:
            return "heuristic_model"
    
    def _find_similar_historical_results(self, video_chars: VideoCharacteristics) -> List[HistoricalResult]:
        """Find similar historical results for comparison."""
        similar_results = []
        
        for result in self.historical_results:
            similarity_score = self._calculate_similarity(video_chars, result.video_chars)
            if similarity_score > 0.7:  # 70% similarity threshold
                similar_results.append(result)
        
        return similar_results
    
    def _calculate_similarity(self, chars1: VideoCharacteristics, chars2: VideoCharacteristics) -> float:
        """Calculate similarity between two video characteristics."""
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
    
    def record_evaluation_result(
        self, 
        video_chars: VideoCharacteristics,
        actual_vmaf: Optional[float],
        actual_ssim: Optional[float],
        evaluation_time: float,
        evaluation_success: bool
    ) -> None:
        """Record actual evaluation result for learning.
        
        Args:
            video_chars: Video characteristics
            actual_vmaf: Actual VMAF score
            actual_ssim: Actual SSIM score
            evaluation_time: Time taken for evaluation
            evaluation_success: Whether evaluation succeeded
        """
        result = HistoricalResult(
            video_chars=video_chars,
            actual_vmaf=actual_vmaf,
            actual_ssim=actual_ssim,
            evaluation_time=evaluation_time,
            evaluation_success=evaluation_success,
            timestamp=time.time()
        )
        
        self.historical_results.append(result)
        
        # Limit historical results to prevent memory growth
        if len(self.historical_results) > self.max_historical_results:
            self.historical_results = self.historical_results[-self.max_historical_results:]
        
        logger.debug(f"Recorded evaluation result: VMAF={actual_vmaf}, SSIM={actual_ssim}, "
                    f"time={evaluation_time:.2f}s, success={evaluation_success}")
    
    def get_prediction_accuracy(self) -> Dict[str, Any]:
        """Get accuracy statistics for predictions.
        
        Returns:
            Dictionary with accuracy statistics
        """
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
            if result.evaluation_success:
                # Calculate prediction for this result
                prediction = self.predict_quality_scores(result.video_chars)
                
                # VMAF accuracy
                if result.actual_vmaf is not None and prediction.predicted_vmaf is not None:
                    error = abs(result.actual_vmaf - prediction.predicted_vmaf)
                    vmaf_errors.append(error)
                
                # SSIM accuracy
                if result.actual_ssim is not None and prediction.predicted_ssim is not None:
                    error = abs(result.actual_ssim - prediction.predicted_ssim)
                    ssim_errors.append(error)
                
                # Time accuracy
                time_error = abs(result.evaluation_time - prediction.estimated_evaluation_time)
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