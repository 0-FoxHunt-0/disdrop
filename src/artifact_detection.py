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
    
    def detect_artifacts(self, video_path: str, 
                        blockiness_threshold: float = 0.12,
                        banding_threshold: float = 0.10,
                        num_keyframes: int = 5,
                        num_random: int = 3) -> Dict[str, Any]:
        """Detect compression artifacts in video by sampling frames.
        
        Args:
            video_path: Path to video file
            blockiness_threshold: Max acceptable blockiness score (0-1)
            banding_threshold: Max acceptable banding score (0-1)
            num_keyframes: Number of evenly-spaced keyframes to sample
            num_random: Number of random frames to sample
        
        Returns:
            Dict with keys:
                - blockiness_score: float (avg)
                - banding_score: float (avg)
                - passes: bool (both under thresholds)
                - details: frame-level scores
        """
        result = {
            'blockiness_score': None,
            'banding_score': None,
            'passes': False,
            'details': {
                'blockiness_threshold': blockiness_threshold,
                'banding_threshold': banding_threshold,
                'frame_scores': []
            }
        }
        
        try:
            if not os.path.exists(video_path):
                result['details']['error'] = f"Video not found: {video_path}"
                return result
            
            # Sample frames
            frames = self._sample_frames(video_path, num_keyframes, num_random)
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
            
            # Check thresholds
            blockiness_pass = avg_blockiness <= blockiness_threshold
            banding_pass = avg_banding <= banding_threshold
            result['passes'] = blockiness_pass and banding_pass
            
            result['details']['blockiness_pass'] = blockiness_pass
            result['details']['banding_pass'] = banding_pass
            
            logger.info(f"Artifact detection: blockiness={avg_blockiness:.4f} (thresh={blockiness_threshold}), "
                       f"banding={avg_banding:.4f} (thresh={banding_threshold}), passes={result['passes']}")
            
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

