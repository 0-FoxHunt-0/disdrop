"""
Artifact Detection Module
Implements blockiness and banding detection using OpenCV
"""

import os
import logging
import random
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ArtifactDetector:
    """Detect compression artifacts (blockiness, banding) in video frames."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        self._load_config()
    
    def _load_config(self):
        """Load artifact detection configuration from config manager."""
        if not self.config:
            self.base_blockiness_threshold = 0.65
            self.base_banding_threshold = 0.50
            self.ssim_relative_enabled = True
            self.ssim_reference = 0.94
            self.adjustment_factor = 5.0
            self.max_adjustment = 10.0
            self.num_keyframes = 5
            self.num_random = 3
            return
        
        artifact_cfg = self.config.get('video_compression.quality_evaluation.artifact_detection', {})
        base_thresholds = artifact_cfg.get('base_thresholds', {})
        self.base_blockiness_threshold = base_thresholds.get('blockiness', 0.65)
        self.base_banding_threshold = base_thresholds.get('banding', 0.50)
        
        ssim_adj = artifact_cfg.get('ssim_relative_adjustment', {})
        self.ssim_relative_enabled = ssim_adj.get('enabled', True)
        self.ssim_reference = ssim_adj.get('ssim_reference', 0.94)
        self.adjustment_factor = ssim_adj.get('adjustment_factor', 5.0)
        self.max_adjustment = ssim_adj.get('max_adjustment', 10.0)
        
        frame_sampling = artifact_cfg.get('frame_sampling', {})
        self.num_keyframes = frame_sampling.get('num_keyframes', 5)
        self.num_random = frame_sampling.get('num_random', 3)
    
    def _calculate_ssim_adjusted_thresholds(self, ssim_score: Optional[float], 
                                           base_blockiness: float, 
                                           base_banding: float) -> Tuple[float, float]:
        """Calculate SSIM-adjusted artifact thresholds.
        
        When SSIM is high (indicating good quality), we relax artifact thresholds
        to prevent false positives. This is based on the observation that high SSIM
        scores (e.g., >0.94) indicate good quality even if artifact metrics are elevated.
        
        Args:
            ssim_score: SSIM score (0-1), or None if not available
            base_blockiness: Base blockiness threshold
            base_banding: Base banding threshold
        
        Returns:
            Tuple of (adjusted_blockiness_threshold, adjusted_banding_threshold)
        """
        if not self.ssim_relative_enabled or ssim_score is None:
            return base_blockiness, base_banding
        
        # Validate SSIM score range (should be 0-1)
        if ssim_score < 0 or ssim_score > 1:
            logger.warning(f"Invalid SSIM score {ssim_score}, expected 0-1. Using base thresholds.")
            return base_blockiness, base_banding
        
        # If SSIM is above reference, adjust thresholds proportionally
        if ssim_score >= self.ssim_reference:
            # Calculate adjustment: higher SSIM = more lenient thresholds
            # Adjustment scales from 1.0 (at reference) to max_adjustment (at SSIM=1.0)
            ssim_excess = ssim_score - self.ssim_reference
            ssim_range = 1.0 - self.ssim_reference  # Typically 0.06 (0.94 to 1.0)
            
            if ssim_range > 0:
                adjustment_ratio = 1.0 + (ssim_excess / ssim_range) * (self.adjustment_factor - 1.0)
                adjustment_ratio = min(adjustment_ratio, self.max_adjustment)
            else:
                adjustment_ratio = self.max_adjustment
            
            adjusted_blockiness = base_blockiness * adjustment_ratio
            adjusted_banding = base_banding * adjustment_ratio
            
            logger.debug(f"SSIM-adjusted thresholds: SSIM={ssim_score:.4f}, "
                        f"blockiness={base_blockiness:.4f}->{adjusted_blockiness:.4f}, "
                        f"banding={base_banding:.4f}->{adjusted_banding:.4f}")
            
            return adjusted_blockiness, adjusted_banding
        
        return base_blockiness, base_banding
    
    def detect_artifacts(self, video_path: str, 
                        blockiness_threshold: Optional[float] = None,
                        banding_threshold: Optional[float] = None,
                        ssim_score: Optional[float] = None,
                        num_keyframes: Optional[int] = None,
                        num_random: Optional[int] = None) -> Dict[str, Any]:
        """Detect compression artifacts in video by sampling frames.
        
        Args:
            video_path: Path to video file
            blockiness_threshold: Max acceptable blockiness score (0-1). If None, uses config base threshold.
            banding_threshold: Max acceptable banding score (0-1). If None, uses config base threshold.
            ssim_score: SSIM score (0-1) for threshold adjustment. If provided and SSIM-relative adjustment
                       is enabled, thresholds are relaxed when SSIM is high.
            num_keyframes: Number of evenly-spaced keyframes to sample. If None, uses config default.
            num_random: Number of random frames to sample. If None, uses config default.
        
        Returns:
            Dict with keys:
                - blockiness_score: float (avg)
                - banding_score: float (avg)
                - passes: bool (both under thresholds)
                - details: frame-level scores and threshold info
        """
        # Use provided thresholds or config defaults
        base_blockiness = blockiness_threshold if blockiness_threshold is not None else self.base_blockiness_threshold
        base_banding = banding_threshold if banding_threshold is not None else self.base_banding_threshold
        
        # Calculate SSIM-adjusted thresholds if SSIM is provided
        final_blockiness, final_banding = self._calculate_ssim_adjusted_thresholds(
            ssim_score, base_blockiness, base_banding
        )
        
        # Use provided sampling params or config defaults
        num_key = num_keyframes if num_keyframes is not None else self.num_keyframes
        num_rand = num_random if num_random is not None else self.num_random
        
        result = {
            'blockiness_score': None,
            'banding_score': None,
            'passes': False,
            'details': {
                'blockiness_threshold': final_blockiness,
                'banding_threshold': final_banding,
                'base_blockiness_threshold': base_blockiness,
                'base_banding_threshold': base_banding,
                'ssim_score': ssim_score,
                'ssim_adjusted': ssim_score is not None and self.ssim_relative_enabled and ssim_score >= self.ssim_reference,
                'frame_scores': []
            }
        }
        
        try:
            if not os.path.exists(video_path):
                result['details']['error'] = f"Video not found: {video_path}"
                return result
            
            # Sample frames
            frames = self._sample_frames(video_path, num_key, num_rand)
            if not frames:
                result['details']['error'] = "Could not sample any frames"
                return result
            
            # Compute artifact scores for each frame
            blockiness_scores = []
            banding_scores = []
            
            for frame_idx, frame in frames:
                block_score = self._compute_blockiness(frame)
                band_score = self._compute_banding(frame)
                
                blockiness_scores.append(block_score)
                banding_scores.append(band_score)
                
                result['details']['frame_scores'].append({
                    'frame_idx': frame_idx,
                    'blockiness': block_score,
                    'banding': band_score
                })
            
            # Average scores
            avg_blockiness = sum(blockiness_scores) / len(blockiness_scores)
            avg_banding = sum(banding_scores) / len(banding_scores)
            
            result['blockiness_score'] = avg_blockiness
            result['banding_score'] = avg_banding
            
            # Check thresholds using SSIM-adjusted values
            blockiness_pass = avg_blockiness <= final_blockiness
            banding_pass = avg_banding <= final_banding
            result['passes'] = blockiness_pass and banding_pass
            
            result['details']['blockiness_pass'] = blockiness_pass
            result['details']['banding_pass'] = banding_pass
            
            ssim_info = f", SSIM={ssim_score:.4f}" if ssim_score is not None else ""
            adj_info = " (SSIM-adjusted)" if result['details']['ssim_adjusted'] else ""
            logger.info(f"Artifact detection: blockiness={avg_blockiness:.4f} (thresh={final_blockiness:.4f}{adj_info}), "
                       f"banding={avg_banding:.4f} (thresh={final_banding:.4f}{adj_info}){ssim_info}, passes={result['passes']}")
            
        except Exception as e:
            logger.error(f"Artifact detection failed: {e}")
            result['details']['error'] = str(e)
        
        return result
    
    def _sample_frames(self, video_path: str, num_keyframes: int, 
                      num_random: int) -> List[Tuple[int, np.ndarray]]:
        """Sample frames from video: evenly-spaced keyframes + random frames.
        
        Returns list of (frame_index, frame_bgr) tuples.
        """
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Could not open video: {video_path}")
                return frames
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return frames
            
            # Evenly-spaced keyframes
            keyframe_indices = []
            if num_keyframes > 0:
                step = max(1, total_frames // (num_keyframes + 1))
                keyframe_indices = [step * (i + 1) for i in range(num_keyframes)]
                keyframe_indices = [idx for idx in keyframe_indices if idx < total_frames]
            
            # Random frames (avoid duplicates with keyframes)
            random_indices = []
            if num_random > 0:
                candidate_indices = [i for i in range(total_frames) if i not in keyframe_indices]
                if candidate_indices:
                    num_to_sample = min(num_random, len(candidate_indices))
                    random_indices = random.sample(candidate_indices, num_to_sample)
            
            # Combine and sort
            all_indices = sorted(set(keyframe_indices + random_indices))
            
            # Extract frames
            for frame_idx in all_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append((frame_idx, frame))
            
            cap.release()
            
        except Exception as e:
            logger.warning(f"Frame sampling error: {e}")
        
        return frames
    
    def _compute_blockiness(self, frame: np.ndarray) -> float:
        """Compute blockiness score using grid-edge energy detection.
        
        Measures energy at 8x8 and 16x16 block boundaries (common in video codecs).
        
        Returns score in [0, 1] where higher = more blocky.
        """
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            h, w = gray.shape
            
            # Compute horizontal and vertical gradients
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Energy at block boundaries (8x8 and 16x16)
            boundary_energy = 0.0
            total_energy = 0.0
            
            # Check 8x8 boundaries
            for i in range(8, h, 8):
                if i < h:
                    boundary_energy += np.sum(np.abs(grad_y[i, :]))
            for j in range(8, w, 8):
                if j < w:
                    boundary_energy += np.sum(np.abs(grad_x[:, j]))
            
            # Check 16x16 boundaries (weighted more)
            for i in range(16, h, 16):
                if i < h:
                    boundary_energy += np.sum(np.abs(grad_y[i, :])) * 1.5
            for j in range(16, w, 16):
                if j < w:
                    boundary_energy += np.sum(np.abs(grad_x[:, j])) * 1.5
            
            # Total gradient energy
            total_energy = np.sum(np.abs(grad_x)) + np.sum(np.abs(grad_y))
            
            if total_energy > 0:
                # Ratio of boundary energy to total energy
                blockiness_ratio = boundary_energy / total_energy
                # Normalize to [0, 1] range (empirically, ratio ~0.3 for blocky content)
                blockiness_score = min(1.0, blockiness_ratio * 3.0)
            else:
                blockiness_score = 0.0
            
            return blockiness_score
            
        except Exception as e:
            logger.warning(f"Blockiness computation error: {e}")
            return 0.0
    
    def _compute_banding(self, frame: np.ndarray) -> float:
        """Compute banding score using gradient step detection.
        
        Banding creates visible steps in smooth gradients (e.g., sky, walls).
        
        Returns score in [0, 1] where higher = more banding.
        """
        try:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame
            
            # Apply slight blur to reduce noise
            blurred = cv2.GaussianBlur(gray, (5, 5), 1.0)
            
            # Compute gradient magnitude
            grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            
            # Histogram of gradient magnitudes
            hist, bin_edges = np.histogram(grad_mag, bins=50, range=(0, 255))
            
            # Banding manifests as peaks in low-gradient regions (discrete steps)
            # Focus on gradients in [5, 30] range (visible but weak gradients)
            low_grad_bins = hist[:15]  # First 15 bins cover 0-~75 gradient range
            
            if np.sum(low_grad_bins) > 0:
                # Measure variance in low-gradient histogram (peaks = banding)
                low_grad_variance = np.var(low_grad_bins)
                # Normalize (empirically, variance ~50-100 indicates banding)
                banding_score = min(1.0, low_grad_variance / 100.0)
            else:
                banding_score = 0.0
            
            return banding_score
            
        except Exception as e:
            logger.warning(f"Banding computation error: {e}")
            return 0.0

