"""
Fast Quality Estimator
Provides rapid quality assessment using sampling and lightweight metrics
"""

import os
import subprocess
import json
import logging
import time
import hashlib
from typing import Dict, Any, Optional, Tuple, List, NamedTuple
from dataclasses import dataclass
from pathlib import Path
import tempfile

from sampling_engine import SamplingEngine, VideoSegment, FrameInfo
from lightweight_quality_metrics import LightweightQualityMetrics, LightweightQualityResult
from quality_prediction_models import QualityPredictionModels, QualityPrediction

logger = logging.getLogger(__name__)


@dataclass
class FrameSample:
    """Information about a sampled frame."""
    timestamp: float
    frame_number: int
    complexity_score: float
    blur_score: float
    noise_score: float


@dataclass
class QualityMetrics:
    """Lightweight quality metrics computed from samples."""
    average_blur: float
    average_noise: float
    average_complexity: float
    motion_score: float
    scene_changes: int
    temporal_consistency: float


@dataclass
class QualityEstimate:
    """Fast quality estimation result."""
    predicted_vmaf: float
    predicted_ssim: float
    confidence: float
    computation_time: float
    sample_coverage: float
    should_run_full_evaluation: bool
    fast_metrics: QualityMetrics
    sample_count: int


class FastQualityEstimator:
    """Provides rapid quality assessment without expensive full evaluations."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        
        # Configuration
        self.max_sample_duration = self._get_config('max_sample_duration', 30.0)
        self.samples_per_minute = self._get_config('samples_per_minute', 4)
        self.min_samples = self._get_config('min_samples', 10)
        self.max_samples = self._get_config('max_samples', 50)
        self.confidence_threshold = self._get_config('confidence_threshold', 0.7)
        self.skip_full_evaluation_threshold = self._get_config('skip_full_evaluation_threshold', 0.8)
        
        # Initialize component modules
        self.sampling_engine = SamplingEngine(config_manager)
        self.quality_metrics = LightweightQualityMetrics(config_manager)
        self.prediction_models = QualityPredictionModels(config_manager)
        
        logger.info("Fast Quality Estimator initialized with integrated modules")
    
    def _get_config(self, key: str, default: Any) -> Any:
        """Get configuration value with fallback to default."""
        if self.config:
            return self.config.get(f'fast_quality_estimation.{key}', default)
        return default
    
    def estimate_quality_fast(
        self, 
        video_path: str, 
        target_duration: Optional[float] = None
    ) -> QualityEstimate:
        """Provide rapid quality assessment using sampling and lightweight metrics.
        
        Args:
            video_path: Path to video file
            target_duration: Optional duration limit for sampling
            
        Returns:
            QualityEstimate with predicted scores and confidence
        """
        start_time = time.time()
        
        logger.info(f"Starting fast quality estimation for: {os.path.basename(video_path)}")
        
        # Get video info
        video_info = self._get_video_info(video_path)
        if not video_info:
            return self._create_fallback_estimate(time.time() - start_time)
        
        duration = video_info.get('duration', 0)
        
        # Determine sampling strategy
        sample_duration = min(target_duration or self.max_sample_duration, duration)
        
        logger.debug(f"Sampling strategy: {sample_duration:.1f}s from {duration:.1f}s total")
        
        # Generate representative samples using sampling engine
        video_segments = self.sampling_engine.generate_representative_samples(
            video_path, sample_duration
        )
        
        if not video_segments:
            return self._create_fallback_estimate(time.time() - start_time)
        
        # Extract timestamps from segments for quality analysis
        sample_timestamps = []
        for segment in video_segments:
            # Sample from middle of each segment
            mid_timestamp = segment.start_time + segment.duration / 2
            sample_timestamps.append(mid_timestamp)
        
        # Analyze quality using lightweight metrics
        quality_result = self.quality_metrics.analyze_video_quality(
            video_path, sample_timestamps
        )
        
        # Predict quality scores using statistical models
        prediction = self.prediction_models.predict_quality_scores(quality_result)
        
        # Determine if full evaluation should be skipped
        should_skip_full = self.prediction_models.should_skip_expensive_evaluation(prediction)
        
        computation_time = time.time() - start_time
        sample_coverage = sample_duration / duration if duration > 0 else 0
        
        # Convert to legacy QualityEstimate format for compatibility
        estimate = QualityEstimate(
            predicted_vmaf=prediction.predicted_vmaf,
            predicted_ssim=prediction.predicted_ssim,
            confidence=prediction.overall_confidence,
            computation_time=computation_time,
            sample_coverage=sample_coverage,
            should_run_full_evaluation=not should_skip_full,
            fast_metrics=self._convert_to_legacy_metrics(quality_result),
            sample_count=quality_result.frame_count
        )
        
        logger.info(f"Fast estimation complete: VMAF≈{prediction.predicted_vmaf:.1f}, "
                   f"SSIM≈{prediction.predicted_ssim:.3f}, "
                   f"confidence={prediction.overall_confidence:.2f}, "
                   f"time={computation_time:.2f}s, skip_full={should_skip_full}")
        
        return estimate
    
    def _get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Get basic video information using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.warning(f"ffprobe failed for {video_path}: {result.stderr}")
                return None
            
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                logger.warning(f"No video stream found in {video_path}")
                return None
            
            # Extract relevant info
            duration = float(data.get('format', {}).get('duration', 0))
            fps_str = video_stream.get('r_frame_rate', '30/1')
            
            # Parse fps fraction
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den) if float(den) != 0 else 30.0
            else:
                fps = float(fps_str)
            
            return {
                'duration': duration,
                'fps': fps,
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0))
            }
            
        except Exception as e:
            logger.error(f"Error getting video info for {video_path}: {e}")
            return None
    
    def _calculate_sample_count(self, duration: float, sample_duration: float) -> int:
        """Calculate optimal number of samples based on video duration."""
        
        # Base sample count on duration
        samples_needed = int(sample_duration * self.samples_per_minute / 60)
        
        # Ensure within bounds
        samples_needed = max(self.min_samples, min(self.max_samples, samples_needed))
        
        # Adjust for very short videos
        if duration < 10:
            samples_needed = min(samples_needed, int(duration * 2))  # 2 samples per second max
        
        return samples_needed
    
    def sample_frames_for_analysis(
        self, 
        video_path: str, 
        num_samples: int,
        sample_duration: Optional[float] = None
    ) -> List[FrameSample]:
        """Extract representative frame samples for quality analysis.
        
        Args:
            video_path: Path to video file
            num_samples: Number of frames to sample
            sample_duration: Duration to sample from (None = full video)
            
        Returns:
            List of FrameSample objects with analysis data
        """
        logger.debug(f"Extracting {num_samples} frame samples from {os.path.basename(video_path)}")
        
        samples = []
        
        try:
            # Create temporary directory for frame extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                
                # Calculate sampling timestamps
                timestamps = self._generate_sample_timestamps(video_path, num_samples, sample_duration)
                
                for i, timestamp in enumerate(timestamps):
                    frame_path = os.path.join(temp_dir, f"frame_{i:04d}.png")
                    
                    # Extract frame at timestamp
                    if self._extract_frame(video_path, timestamp, frame_path):
                        # Analyze frame
                        sample = self._analyze_frame(frame_path, timestamp, i)
                        if sample:
                            samples.append(sample)
                    
                    # Limit processing time
                    if len(samples) >= num_samples:
                        break
        
        except Exception as e:
            logger.error(f"Error sampling frames from {video_path}: {e}")
        
        logger.debug(f"Successfully extracted {len(samples)} frame samples")
        return samples
    
    def _generate_sample_timestamps(
        self, 
        video_path: str, 
        num_samples: int,
        sample_duration: Optional[float]
    ) -> List[float]:
        """Generate timestamps for frame sampling."""
        
        video_info = self._get_video_info(video_path)
        if not video_info:
            return []
        
        total_duration = video_info['duration']
        effective_duration = min(sample_duration or total_duration, total_duration)
        
        if num_samples <= 1:
            return [effective_duration / 2]  # Middle of video
        
        # Generate evenly spaced timestamps with some randomization
        timestamps = []
        interval = effective_duration / num_samples
        
        for i in range(num_samples):
            # Base timestamp
            base_time = i * interval + interval / 2
            
            # Add small random offset (±10% of interval) for better coverage
            import random
            offset = random.uniform(-interval * 0.1, interval * 0.1)
            timestamp = max(0, min(effective_duration - 1, base_time + offset))
            
            timestamps.append(timestamp)
        
        return sorted(timestamps)
    
    def _extract_frame(self, video_path: str, timestamp: float, output_path: str) -> bool:
        """Extract a single frame at the specified timestamp."""
        try:
            cmd = [
                'ffmpeg', '-v', 'quiet', '-ss', str(timestamp), '-i', video_path,
                '-vframes', '1', '-y', output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            return result.returncode == 0 and os.path.exists(output_path)
            
        except Exception as e:
            logger.debug(f"Error extracting frame at {timestamp}s: {e}")
            return False
    
    def _analyze_frame(self, frame_path: str, timestamp: float, frame_number: int) -> Optional[FrameSample]:
        """Analyze a single frame for quality metrics."""
        try:
            # Use ffmpeg filters to compute frame metrics
            cmd = [
                'ffmpeg', '-v', 'quiet', '-i', frame_path,
                '-vf', 'idet,blackdetect,cropdetect', '-f', 'null', '-'
            ]
            
            # For now, use simplified analysis
            # In a full implementation, you'd use computer vision libraries like OpenCV
            
            # Placeholder analysis - replace with actual image analysis
            complexity_score = 0.5  # Would compute spatial complexity
            blur_score = 0.3       # Would compute blur/sharpness
            noise_score = 0.2      # Would compute noise level
            
            return FrameSample(
                timestamp=timestamp,
                frame_number=frame_number,
                complexity_score=complexity_score,
                blur_score=blur_score,
                noise_score=noise_score
            )
            
        except Exception as e:
            logger.debug(f"Error analyzing frame {frame_path}: {e}")
            return None
    
    def compute_lightweight_metrics(self, samples: List[FrameSample], fps: float) -> QualityMetrics:
        """Compute lightweight quality metrics from frame samples.
        
        Args:
            samples: List of analyzed frame samples
            fps: Video frame rate
            
        Returns:
            QualityMetrics with aggregated analysis
        """
        if not samples:
            return QualityMetrics(0, 0, 0, 0, 0, 0)
        
        # Aggregate frame-level metrics
        total_blur = sum(s.blur_score for s in samples)
        total_noise = sum(s.noise_score for s in samples)
        total_complexity = sum(s.complexity_score for s in samples)
        
        average_blur = total_blur / len(samples)
        average_noise = total_noise / len(samples)
        average_complexity = total_complexity / len(samples)
        
        # Compute temporal metrics
        motion_score = self._compute_motion_score(samples)
        scene_changes = self._detect_scene_changes(samples)
        temporal_consistency = self._compute_temporal_consistency(samples)
        
        return QualityMetrics(
            average_blur=average_blur,
            average_noise=average_noise,
            average_complexity=average_complexity,
            motion_score=motion_score,
            scene_changes=scene_changes,
            temporal_consistency=temporal_consistency
        )
    
    def _compute_motion_score(self, samples: List[FrameSample]) -> float:
        """Compute motion score from frame complexity changes."""
        if len(samples) < 2:
            return 0.0
        
        complexity_changes = []
        for i in range(1, len(samples)):
            change = abs(samples[i].complexity_score - samples[i-1].complexity_score)
            complexity_changes.append(change)
        
        return sum(complexity_changes) / len(complexity_changes) if complexity_changes else 0.0
    
    def _detect_scene_changes(self, samples: List[FrameSample]) -> int:
        """Detect scene changes from complexity variations."""
        if len(samples) < 3:
            return 0
        
        scene_changes = 0
        threshold = 0.3  # Complexity change threshold for scene detection
        
        for i in range(1, len(samples) - 1):
            prev_complexity = samples[i-1].complexity_score
            curr_complexity = samples[i].complexity_score
            next_complexity = samples[i+1].complexity_score
            
            # Look for significant complexity changes
            change1 = abs(curr_complexity - prev_complexity)
            change2 = abs(next_complexity - curr_complexity)
            
            if change1 > threshold or change2 > threshold:
                scene_changes += 1
        
        return scene_changes
    
    def _compute_temporal_consistency(self, samples: List[FrameSample]) -> float:
        """Compute temporal consistency score."""
        if len(samples) < 2:
            return 1.0
        
        # Measure consistency in blur and noise across frames
        blur_variance = self._compute_variance([s.blur_score for s in samples])
        noise_variance = self._compute_variance([s.noise_score for s in samples])
        
        # Lower variance = higher consistency
        consistency = 1.0 / (1.0 + blur_variance + noise_variance)
        return min(1.0, consistency)
    
    def _compute_variance(self, values: List[float]) -> float:
        """Compute variance of a list of values."""
        if len(values) < 2:
            return 0.0
        
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance
    
    def _predict_vmaf(self, metrics: QualityMetrics) -> float:
        """Predict VMAF score from lightweight metrics."""
        model = self.vmaf_model
        
        predicted = (
            model['baseline'] +
            model['blur_weight'] * metrics.average_blur +
            model['noise_weight'] * metrics.average_noise +
            model['complexity_weight'] * metrics.average_complexity +
            model['motion_weight'] * metrics.motion_score +
            model['consistency_weight'] * metrics.temporal_consistency
        )
        
        # Clamp to valid VMAF range
        return max(0.0, min(100.0, predicted))
    
    def _predict_ssim(self, metrics: QualityMetrics) -> float:
        """Predict SSIM score from lightweight metrics."""
        model = self.ssim_model
        
        predicted = (
            model['baseline'] +
            model['blur_weight'] * metrics.average_blur +
            model['noise_weight'] * metrics.average_noise +
            model['complexity_weight'] * metrics.average_complexity +
            model['motion_weight'] * metrics.motion_score +
            model['consistency_weight'] * metrics.temporal_consistency
        )
        
        # Clamp to valid SSIM range
        return max(0.0, min(1.0, predicted))
    
    def _calculate_confidence(
        self, 
        samples: List[FrameSample], 
        sample_duration: float,
        total_duration: float
    ) -> float:
        """Calculate confidence in the quality estimate."""
        
        # Base confidence on sample coverage
        coverage_factor = min(1.0, sample_duration / min(total_duration, 60.0))  # Cap at 1 minute
        
        # Adjust for sample count
        sample_factor = min(1.0, len(samples) / self.min_samples)
        
        # Adjust for temporal consistency (more consistent = higher confidence)
        if samples:
            metrics = self.compute_lightweight_metrics(samples, 30.0)  # Assume 30fps for consistency calc
            consistency_factor = metrics.temporal_consistency
        else:
            consistency_factor = 0.0
        
        # Combine factors
        confidence = (coverage_factor * 0.4 + sample_factor * 0.3 + consistency_factor * 0.3)
        
        return max(0.0, min(1.0, confidence))
    
    def _convert_to_legacy_metrics(self, quality_result: LightweightQualityResult) -> QualityMetrics:
        """Convert new quality metrics to legacy format for compatibility."""
        return QualityMetrics(
            average_blur=quality_result.blur_metrics.overall_blur_score,
            average_noise=quality_result.noise_metrics.overall_noise_score,
            average_complexity=quality_result.complexity_metrics.overall_complexity_score,
            motion_score=quality_result.motion_metrics.overall_motion_score,
            scene_changes=0,  # Not directly available in new format
            temporal_consistency=quality_result.motion_metrics.temporal_stability
        )
    
    def _create_fallback_estimate(self, computation_time: float) -> QualityEstimate:
        """Create a fallback estimate when analysis fails."""
        return QualityEstimate(
            predicted_vmaf=50.0,  # Conservative estimate
            predicted_ssim=0.80,  # Conservative estimate
            confidence=0.1,       # Low confidence
            computation_time=computation_time,
            sample_coverage=0.0,
            should_run_full_evaluation=True,  # Always run full evaluation on failure
            fast_metrics=QualityMetrics(0, 0, 0, 0, 0, 0),
            sample_count=0
        )