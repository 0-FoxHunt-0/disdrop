"""
Lightweight Quality Metrics
Implements fast quality assessment algorithms for efficient video analysis
"""

import os
import subprocess
import json
import logging
import time
import math
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from dataclasses import dataclass
import tempfile
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageStat

logger = logging.getLogger(__name__)


@dataclass
class BlurMetrics:
    """Blur detection metrics."""
    laplacian_variance: float
    sobel_variance: float
    fft_blur_score: float
    overall_blur_score: float
    confidence: float


@dataclass
class NoiseMetrics:
    """Noise level analysis metrics."""
    noise_variance: float
    snr_estimate: float
    grain_score: float
    overall_noise_score: float
    confidence: float


@dataclass
class ComplexityMetrics:
    """Spatial complexity metrics."""
    edge_density: float
    texture_complexity: float
    color_variance: float
    gradient_magnitude: float
    overall_complexity_score: float


@dataclass
class MotionMetrics:
    """Motion detection metrics."""
    optical_flow_magnitude: float
    frame_difference_score: float
    motion_vector_consistency: float
    temporal_stability: float
    overall_motion_score: float


@dataclass
class LightweightQualityResult:
    """Combined lightweight quality assessment result."""
    blur_metrics: BlurMetrics
    noise_metrics: NoiseMetrics
    complexity_metrics: ComplexityMetrics
    motion_metrics: MotionMetrics
    overall_quality_score: float
    processing_time: float
    frame_count: int


class LightweightQualityMetrics:
    """Fast quality assessment using lightweight algorithms."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        
        # Configuration parameters
        self.blur_threshold = self._get_config('blur_threshold', 100.0)
        self.noise_threshold = self._get_config('noise_threshold', 0.1)
        self.complexity_threshold = self._get_config('complexity_threshold', 0.3)
        self.motion_threshold = self._get_config('motion_threshold', 0.2)
        
        # Processing limits
        self.max_analysis_frames = self._get_config('max_analysis_frames', 20)
        self.frame_resize_width = self._get_config('frame_resize_width', 480)
        self.max_processing_time = self._get_config('max_processing_time', 30.0)
        
        logger.info("Lightweight Quality Metrics initialized")
    
    def _get_config(self, key: str, default: Any) -> Any:
        """Get configuration value with fallback to default."""
        if self.config:
            return self.config.get(f'lightweight_quality_metrics.{key}', default)
        return default
    
    def analyze_video_quality(self, video_path: str, sample_timestamps: Optional[List[float]] = None) -> LightweightQualityResult:
        """Analyze video quality using lightweight metrics.
        
        Args:
            video_path: Path to video file
            sample_timestamps: Optional list of timestamps to analyze
            
        Returns:
            LightweightQualityResult with comprehensive quality assessment
        """
        start_time = time.time()
        
        logger.info(f"Starting lightweight quality analysis for {os.path.basename(video_path)}")
        
        # Extract frames for analysis
        frames = self._extract_analysis_frames(video_path, sample_timestamps)
        
        if not frames:
            logger.error("No frames extracted for analysis")
            return self._create_fallback_result(time.time() - start_time)
        
        logger.debug(f"Analyzing {len(frames)} frames")
        
        # Analyze each metric type
        blur_metrics = self._analyze_blur_metrics(frames)
        noise_metrics = self._analyze_noise_metrics(frames)
        complexity_metrics = self._analyze_complexity_metrics(frames)
        motion_metrics = self._analyze_motion_metrics(frames)
        
        # Calculate overall quality score
        overall_score = self._calculate_overall_quality_score(
            blur_metrics, noise_metrics, complexity_metrics, motion_metrics
        )
        
        processing_time = time.time() - start_time
        
        result = LightweightQualityResult(
            blur_metrics=blur_metrics,
            noise_metrics=noise_metrics,
            complexity_metrics=complexity_metrics,
            motion_metrics=motion_metrics,
            overall_quality_score=overall_score,
            processing_time=processing_time,
            frame_count=len(frames)
        )
        
        logger.info(f"Quality analysis complete: overall_score={overall_score:.2f}, "
                   f"blur={blur_metrics.overall_blur_score:.2f}, "
                   f"noise={noise_metrics.overall_noise_score:.2f}, "
                   f"complexity={complexity_metrics.overall_complexity_score:.2f}, "
                   f"motion={motion_metrics.overall_motion_score:.2f}, "
                   f"time={processing_time:.2f}s")
        
        return result
    
    def _extract_analysis_frames(self, video_path: str, sample_timestamps: Optional[List[float]]) -> List[np.ndarray]:
        """Extract frames for quality analysis."""
        frames = []
        
        try:
            if sample_timestamps:
                # Use provided timestamps
                for timestamp in sample_timestamps[:self.max_analysis_frames]:
                    frame = self._extract_frame_at_timestamp(video_path, timestamp)
                    if frame is not None:
                        frames.append(frame)
            else:
                # Extract frames at regular intervals
                video_info = self._get_video_info(video_path)
                if video_info:
                    duration = video_info['duration']
                    num_frames = min(self.max_analysis_frames, max(5, int(duration / 10)))
                    
                    for i in range(num_frames):
                        timestamp = (i + 0.5) * duration / num_frames
                        frame = self._extract_frame_at_timestamp(video_path, timestamp)
                        if frame is not None:
                            frames.append(frame)
        
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
        
        return frames
    
    def _extract_frame_at_timestamp(self, video_path: str, timestamp: float) -> Optional[np.ndarray]:
        """Extract a single frame at the specified timestamp."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Extract frame using ffmpeg
            cmd = [
                'ffmpeg', '-v', 'quiet', '-ss', str(timestamp), '-i', video_path,
                '-vframes', '1', '-vf', f'scale={self.frame_resize_width}:-1',
                '-y', temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            
            if result.returncode == 0 and os.path.exists(temp_path):
                # Load frame with OpenCV
                frame = cv2.imread(temp_path)
                if frame is not None:
                    return frame
            
        except Exception as e:
            logger.debug(f"Frame extraction failed at {timestamp}s: {e}")
        
        finally:
            # Cleanup
            if 'temp_path' in locals() and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass
        
        return None
    
    def _analyze_blur_metrics(self, frames: List[np.ndarray]) -> BlurMetrics:
        """Analyze blur using multiple detection algorithms."""
        laplacian_scores = []
        sobel_scores = []
        fft_scores = []
        
        for frame in frames:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Laplacian variance (most common blur metric)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            laplacian_scores.append(laplacian_var)
            
            # Sobel gradient magnitude
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            sobel_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            sobel_scores.append(sobel_magnitude.var())
            
            # FFT-based blur detection
            fft_score = self._compute_fft_blur_score(gray)
            fft_scores.append(fft_score)
        
        # Aggregate scores
        avg_laplacian = np.mean(laplacian_scores) if laplacian_scores else 0
        avg_sobel = np.mean(sobel_scores) if sobel_scores else 0
        avg_fft = np.mean(fft_scores) if fft_scores else 0
        
        # Normalize scores (empirical scaling)
        laplacian_normalized = min(1.0, avg_laplacian / self.blur_threshold)
        sobel_normalized = min(1.0, avg_sobel / (self.blur_threshold * 2))
        fft_normalized = avg_fft  # Already 0-1
        
        # Calculate overall blur score (higher = sharper)
        overall_blur = (laplacian_normalized * 0.5 + sobel_normalized * 0.3 + fft_normalized * 0.2)
        
        # Calculate confidence based on consistency
        scores = [laplacian_normalized, sobel_normalized, fft_normalized]
        confidence = 1.0 - (np.std(scores) if len(scores) > 1 else 0.0)
        
        return BlurMetrics(
            laplacian_variance=avg_laplacian,
            sobel_variance=avg_sobel,
            fft_blur_score=avg_fft,
            overall_blur_score=overall_blur,
            confidence=max(0.0, min(1.0, confidence))
        )
    
    def _compute_fft_blur_score(self, gray_image: np.ndarray) -> float:
        """Compute blur score using FFT analysis."""
        try:
            # Apply FFT
            fft = np.fft.fft2(gray_image)
            fft_shift = np.fft.fftshift(fft)
            magnitude_spectrum = np.abs(fft_shift)
            
            # Calculate high-frequency content
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            
            # Define high-frequency region (outer 30% of spectrum)
            mask = np.zeros((h, w), dtype=bool)
            y, x = np.ogrid[:h, :w]
            distance = np.sqrt((x - center_w)**2 + (y - center_h)**2)
            max_distance = min(center_h, center_w)
            mask[distance > max_distance * 0.7] = True
            
            # Calculate ratio of high-frequency to total energy
            high_freq_energy = np.sum(magnitude_spectrum[mask])
            total_energy = np.sum(magnitude_spectrum)
            
            if total_energy > 0:
                hf_ratio = high_freq_energy / total_energy
                return min(1.0, hf_ratio * 10)  # Scale to 0-1 range
            
        except Exception as e:
            logger.debug(f"FFT blur analysis failed: {e}")
        
        return 0.5  # Default score
    
    def _analyze_noise_metrics(self, frames: List[np.ndarray]) -> NoiseMetrics:
        """Analyze noise levels using multiple methods."""
        noise_variances = []
        snr_estimates = []
        grain_scores = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Noise variance estimation using Laplacian
            noise_var = self._estimate_noise_variance(gray)
            noise_variances.append(noise_var)
            
            # SNR estimation
            snr = self._estimate_snr(gray)
            snr_estimates.append(snr)
            
            # Film grain detection
            grain = self._detect_film_grain(gray)
            grain_scores.append(grain)
        
        # Aggregate scores
        avg_noise_var = np.mean(noise_variances) if noise_variances else 0
        avg_snr = np.mean(snr_estimates) if snr_estimates else 0
        avg_grain = np.mean(grain_scores) if grain_scores else 0
        
        # Normalize scores
        noise_normalized = min(1.0, avg_noise_var / self.noise_threshold)
        snr_normalized = min(1.0, max(0.0, avg_snr / 40.0))  # Assume 40dB is good SNR
        grain_normalized = avg_grain  # Already 0-1
        
        # Calculate overall noise score (lower = less noise)
        overall_noise = (noise_normalized * 0.4 + (1.0 - snr_normalized) * 0.4 + grain_normalized * 0.2)
        
        # Calculate confidence
        scores = [noise_normalized, 1.0 - snr_normalized, grain_normalized]
        confidence = 1.0 - (np.std(scores) if len(scores) > 1 else 0.0)
        
        return NoiseMetrics(
            noise_variance=avg_noise_var,
            snr_estimate=avg_snr,
            grain_score=avg_grain,
            overall_noise_score=overall_noise,
            confidence=max(0.0, min(1.0, confidence))
        )
    
    def _estimate_noise_variance(self, gray_image: np.ndarray) -> float:
        """Estimate noise variance using Laplacian method."""
        try:
            # Apply Laplacian filter
            laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
            
            # Estimate noise variance (Immerkær method)
            noise_var = np.var(laplacian) / 36.0  # Normalization factor for 3x3 Laplacian
            
            return noise_var
        
        except Exception:
            return 0.0
    
    def _estimate_snr(self, gray_image: np.ndarray) -> float:
        """Estimate Signal-to-Noise Ratio."""
        try:
            # Simple SNR estimation using signal variance vs noise variance
            signal_var = np.var(gray_image.astype(np.float64))
            noise_var = self._estimate_noise_variance(gray_image)
            
            if noise_var > 0:
                snr_linear = signal_var / noise_var
                snr_db = 10 * np.log10(snr_linear) if snr_linear > 0 else 0
                return max(0.0, snr_db)
            
        except Exception:
            pass
        
        return 20.0  # Default SNR
    
    def _detect_film_grain(self, gray_image: np.ndarray) -> float:
        """Detect film grain or digital noise patterns."""
        try:
            # Apply high-pass filter to isolate noise
            kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            filtered = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
            
            # Calculate grain score based on high-frequency content
            grain_energy = np.mean(np.abs(filtered))
            
            # Normalize to 0-1 range (empirical scaling)
            grain_score = min(1.0, grain_energy / 50.0)
            
            return grain_score
        
        except Exception:
            return 0.0
    
    def _analyze_complexity_metrics(self, frames: List[np.ndarray]) -> ComplexityMetrics:
        """Analyze spatial complexity using multiple metrics."""
        edge_densities = []
        texture_complexities = []
        color_variances = []
        gradient_magnitudes = []
        
        for frame in frames:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Edge density using Canny
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            edge_densities.append(edge_density)
            
            # Texture complexity using Local Binary Patterns approximation
            texture_complexity = self._compute_texture_complexity(gray)
            texture_complexities.append(texture_complexity)
            
            # Color variance
            color_var = self._compute_color_variance(frame)
            color_variances.append(color_var)
            
            # Gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_mag = np.mean(np.sqrt(grad_x**2 + grad_y**2))
            gradient_magnitudes.append(gradient_mag)
        
        # Aggregate scores
        avg_edge_density = np.mean(edge_densities) if edge_densities else 0
        avg_texture = np.mean(texture_complexities) if texture_complexities else 0
        avg_color_var = np.mean(color_variances) if color_variances else 0
        avg_gradient = np.mean(gradient_magnitudes) if gradient_magnitudes else 0
        
        # Normalize scores
        edge_normalized = min(1.0, avg_edge_density * 10)  # Scale edge density
        texture_normalized = avg_texture  # Already 0-1
        color_normalized = min(1.0, avg_color_var / 1000.0)  # Empirical scaling
        gradient_normalized = min(1.0, avg_gradient / 100.0)  # Empirical scaling
        
        # Calculate overall complexity score
        overall_complexity = (
            edge_normalized * 0.3 + 
            texture_normalized * 0.3 + 
            color_normalized * 0.2 + 
            gradient_normalized * 0.2
        )
        
        return ComplexityMetrics(
            edge_density=avg_edge_density,
            texture_complexity=avg_texture,
            color_variance=avg_color_var,
            gradient_magnitude=avg_gradient,
            overall_complexity_score=overall_complexity
        )
    
    def _compute_texture_complexity(self, gray_image: np.ndarray) -> float:
        """Compute texture complexity using simplified LBP-like method."""
        try:
            # Simple texture measure using local variance
            kernel = np.ones((3, 3), np.float32) / 9
            local_mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray_image.astype(np.float32) - local_mean)**2, -1, kernel)
            
            # Average local variance as texture complexity
            texture_score = np.mean(local_variance)
            
            # Normalize to 0-1 range
            return min(1.0, texture_score / 1000.0)
        
        except Exception:
            return 0.0
    
    def _compute_color_variance(self, color_image: np.ndarray) -> float:
        """Compute color variance across channels."""
        try:
            # Calculate variance for each channel
            b_var = np.var(color_image[:, :, 0].astype(np.float64))
            g_var = np.var(color_image[:, :, 1].astype(np.float64))
            r_var = np.var(color_image[:, :, 2].astype(np.float64))
            
            # Average variance across channels
            return (b_var + g_var + r_var) / 3.0
        
        except Exception:
            return 0.0
    
    def _analyze_motion_metrics(self, frames: List[np.ndarray]) -> MotionMetrics:
        """Analyze motion using temporal analysis."""
        if len(frames) < 2:
            return MotionMetrics(0, 0, 0, 1.0, 0)
        
        optical_flow_mags = []
        frame_diff_scores = []
        motion_consistencies = []
        
        for i in range(1, len(frames)):
            prev_gray = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            
            # Optical flow magnitude (simplified)
            flow_mag = self._compute_optical_flow_magnitude(prev_gray, curr_gray)
            optical_flow_mags.append(flow_mag)
            
            # Frame difference
            diff_score = self._compute_frame_difference(prev_gray, curr_gray)
            frame_diff_scores.append(diff_score)
            
            # Motion vector consistency (simplified)
            consistency = self._compute_motion_consistency(prev_gray, curr_gray)
            motion_consistencies.append(consistency)
        
        # Aggregate scores
        avg_flow = np.mean(optical_flow_mags) if optical_flow_mags else 0
        avg_diff = np.mean(frame_diff_scores) if frame_diff_scores else 0
        avg_consistency = np.mean(motion_consistencies) if motion_consistencies else 1.0
        
        # Normalize scores
        flow_normalized = min(1.0, avg_flow / 50.0)  # Empirical scaling
        diff_normalized = min(1.0, avg_diff / 100.0)  # Empirical scaling
        
        # Calculate overall motion score
        overall_motion = (flow_normalized * 0.5 + diff_normalized * 0.5)
        
        # Temporal stability (inverse of motion)
        temporal_stability = 1.0 - overall_motion
        
        return MotionMetrics(
            optical_flow_magnitude=avg_flow,
            frame_difference_score=avg_diff,
            motion_vector_consistency=avg_consistency,
            temporal_stability=temporal_stability,
            overall_motion_score=overall_motion
        )
    
    def _compute_optical_flow_magnitude(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """Compute simplified optical flow magnitude."""
        try:
            # Use Lucas-Kanade optical flow on corner points
            corners = cv2.goodFeaturesToTrack(prev_gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
            
            if corners is not None and len(corners) > 0:
                # Calculate optical flow
                flow, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, corners, None)
                
                # Calculate flow magnitudes
                if flow is not None and status is not None:
                    good_flow = flow[status == 1]
                    good_corners = corners[status == 1]
                    
                    if len(good_flow) > 0:
                        flow_vectors = good_flow - good_corners.reshape(-1, 2)
                        magnitudes = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)
                        return np.mean(magnitudes)
            
        except Exception as e:
            logger.debug(f"Optical flow computation failed: {e}")
        
        return 0.0
    
    def _compute_frame_difference(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """Compute frame difference score."""
        try:
            # Simple frame difference
            diff = cv2.absdiff(prev_gray, curr_gray)
            return np.mean(diff)
        
        except Exception:
            return 0.0
    
    def _compute_motion_consistency(self, prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
        """Compute motion vector consistency (simplified)."""
        try:
            # Use block-based motion estimation approximation
            # This is a simplified version - real implementation would be more complex
            
            # Divide image into blocks and compute local motion
            h, w = prev_gray.shape
            block_size = 16
            consistencies = []
            
            for y in range(0, h - block_size, block_size):
                for x in range(0, w - block_size, block_size):
                    block1 = prev_gray[y:y+block_size, x:x+block_size]
                    block2 = curr_gray[y:y+block_size, x:x+block_size]
                    
                    # Simple correlation-based consistency
                    correlation = cv2.matchTemplate(block1, block2, cv2.TM_CCOEFF_NORMED)
                    if correlation.size > 0:
                        consistencies.append(np.max(correlation))
            
            return np.mean(consistencies) if consistencies else 0.5
        
        except Exception:
            return 0.5
    
    def _calculate_overall_quality_score(
        self, 
        blur_metrics: BlurMetrics,
        noise_metrics: NoiseMetrics,
        complexity_metrics: ComplexityMetrics,
        motion_metrics: MotionMetrics
    ) -> float:
        """Calculate overall quality score from individual metrics."""
        
        # Weight the different quality aspects
        blur_weight = 0.35      # Sharpness is very important
        noise_weight = 0.25     # Low noise is important
        complexity_weight = 0.20  # Content complexity affects perception
        motion_weight = 0.20    # Motion artifacts matter
        
        # Calculate weighted score
        # Note: noise score is inverted (lower noise = higher quality)
        overall_score = (
            blur_metrics.overall_blur_score * blur_weight +
            (1.0 - noise_metrics.overall_noise_score) * noise_weight +
            complexity_metrics.overall_complexity_score * complexity_weight +
            motion_metrics.temporal_stability * motion_weight
        )
        
        return max(0.0, min(1.0, overall_score))
    
    def _get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Get basic video information using ffprobe."""
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
            
            duration = float(data.get('format', {}).get('duration', 0))
            fps_str = video_stream.get('r_frame_rate', '30/1')
            
            # Parse fps
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
            logger.debug(f"Error getting video info: {e}")
            return None
    
    def _create_fallback_result(self, processing_time: float) -> LightweightQualityResult:
        """Create fallback result when analysis fails."""
        return LightweightQualityResult(
            blur_metrics=BlurMetrics(0, 0, 0, 0.5, 0.1),
            noise_metrics=NoiseMetrics(0, 20, 0, 0.5, 0.1),
            complexity_metrics=ComplexityMetrics(0, 0, 0, 0, 0.5),
            motion_metrics=MotionMetrics(0, 0, 0.5, 0.5, 0.5),
            overall_quality_score=0.5,
            processing_time=processing_time,
            frame_count=0
        )