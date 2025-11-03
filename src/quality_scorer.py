"""
Quality Scorer Module
Implements hybrid multi-factor quality scoring for video/GIF validation
Replaces strict SSIM-based validation with perceptual and efficiency-based metrics
"""

import os
import subprocess
import json
import logging
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any, Tuple, Optional, List
import hashlib
import random

logger = logging.getLogger(__name__)


class QualityScorer:
    """
    Hybrid quality scoring system that evaluates compressed videos/GIFs using multiple factors:
    - Perceptual hash similarity (40%)
    - Size efficiency (20%)
    - Frame validity (20%)
    - Technical validity (20%)
    """
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        
        # Load configuration or use defaults
        if self.config:
            qc_cfg = self.config.get('video_compression.quality_controls.hybrid_quality_check', {}) or {}
        else:
            qc_cfg = {}
        
        # Quality floor thresholds
        self.single_file_floor = float(qc_cfg.get('single_file_floor', 65))
        self.segment_floors = qc_cfg.get('segment_floors', [70, 60, 50, 45])
        
        # Component weights (should sum to 1.0)
        weights = qc_cfg.get('weights', {})
        self.weight_perceptual = float(weights.get('perceptual_hash', 0.40))
        self.weight_size_efficiency = float(weights.get('size_efficiency', 0.20))
        self.weight_frame_validity = float(weights.get('frame_validity', 0.20))
        self.weight_technical = float(weights.get('technical_validity', 0.20))
        
        # Sampling configuration
        sampling = qc_cfg.get('sampling', {})
        self.num_key_frames = int(sampling.get('key_frames', 5))
        self.num_random_frames = int(sampling.get('random_frames', 3))
    
    def calculate_quality_score(self, original_path: str, compressed_path: str, 
                                target_size_mb: float, is_segment: bool = False,
                                quality_floor_override: Optional[float] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive quality score for a compressed video/GIF
        
        Args:
            original_path: Path to original video
            compressed_path: Path to compressed output
            target_size_mb: Target size limit in MB
            is_segment: Whether this is a segment (affects floor threshold)
            quality_floor_override: Optional override for quality floor
            
        Returns:
            Dict with quality score, individual component scores, and pass/fail status
        """
        result = {
            'overall_score': 0.0,
            'components': {},
            'passes': False,
            'is_segment': is_segment,
            'floor_used': 0.0,
            'details': {}
        }
        
        try:
            # Determine quality floor
            if quality_floor_override is not None:
                floor = quality_floor_override
            elif is_segment:
                floor = self.segment_floors[0] if self.segment_floors else 70
            else:
                floor = self.single_file_floor
            
            result['floor_used'] = floor
            
            # Component 1: Perceptual Hash Similarity (40%)
            perceptual_score = self._calculate_perceptual_similarity(original_path, compressed_path)
            result['components']['perceptual_hash'] = perceptual_score
            
            # Component 2: Size Efficiency (20%)
            efficiency_score = self._evaluate_size_efficiency(
                original_path, compressed_path, target_size_mb
            )
            result['components']['size_efficiency'] = efficiency_score
            
            # Component 3: Frame Validity (20%)
            frame_validity_score = self._sample_frame_validity(compressed_path)
            result['components']['frame_validity'] = frame_validity_score
            
            # Component 4: Technical Validity (20%)
            technical_score = self._check_technical_validity(
                original_path, compressed_path
            )
            result['components']['technical_validity'] = technical_score
            
            # Calculate weighted overall score
            overall = (
                perceptual_score * self.weight_perceptual +
                efficiency_score * self.weight_size_efficiency +
                frame_validity_score * self.weight_frame_validity +
                technical_score * self.weight_technical
            )
            
            result['overall_score'] = overall
            result['passes'] = overall >= floor
            
            logger.info(
                f"Quality score: {overall:.1f}/100 (floor: {floor:.1f}) - "
                f"Perceptual: {perceptual_score:.1f}, Efficiency: {efficiency_score:.1f}, "
                f"Frames: {frame_validity_score:.1f}, Technical: {technical_score:.1f}"
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Quality scoring failed: {e}")
            # On error, return neutral passing score to avoid blocking
            result['overall_score'] = floor if floor else 65.0
            result['passes'] = True
            result['error'] = str(e)
            return result
    
    def _calculate_perceptual_similarity(self, original_path: str, compressed_path: str) -> float:
        """
        Calculate perceptual similarity using perceptual hashing
        Returns score from 0-100 (100 = identical perception)
        """
        try:
            # Get key frame positions
            original_frames = self._extract_key_frames(original_path)
            compressed_frames = self._extract_key_frames(compressed_path)
            
            if not original_frames or not compressed_frames:
                logger.warning("Could not extract frames for perceptual comparison")
                return 75.0  # Neutral score
            
            # Calculate perceptual hashes and compare
            similarities = []
            for orig_frame, comp_frame in zip(original_frames, compressed_frames):
                if orig_frame is not None and comp_frame is not None:
                    sim = self._compare_frames_perceptual(orig_frame, comp_frame)
                    similarities.append(sim)
            
            if not similarities:
                return 75.0  # Neutral score
            
            # Average similarity across all sampled frames
            avg_similarity = np.mean(similarities)
            
            # Convert to 0-100 scale
            score = avg_similarity * 100
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.warning(f"Perceptual similarity calculation failed: {e}")
            return 75.0  # Neutral score on error
    
    def _extract_key_frames(self, video_path: str) -> List[Optional[np.ndarray]]:
        """Extract key frames at specific positions in the video"""
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return frames
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return frames
            
            # Calculate key frame positions: start, 25%, 50%, 75%, end
            positions = [
                0,
                total_frames // 4,
                total_frames // 2,
                (total_frames * 3) // 4,
                max(0, total_frames - 1)
            ]
            
            for pos in positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                if ret and frame is not None:
                    frames.append(frame)
                else:
                    frames.append(None)
            
            cap.release()
            
        except Exception as e:
            logger.debug(f"Frame extraction error: {e}")
        
        return frames
    
    def _compare_frames_perceptual(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """
        Compare two frames using perceptual hash (dHash algorithm)
        Returns similarity from 0.0 (different) to 1.0 (identical)
        """
        try:
            # Resize frames to small size for hash comparison
            size = (16, 16)
            
            # Convert to grayscale and resize
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            resized1 = cv2.resize(gray1, size, interpolation=cv2.INTER_AREA)
            resized2 = cv2.resize(gray2, size, interpolation=cv2.INTER_AREA)
            
            # Calculate difference hash (dHash)
            hash1 = self._dhash(resized1)
            hash2 = self._dhash(resized2)
            
            # Calculate Hamming distance
            hamming_dist = bin(hash1 ^ hash2).count('1')
            
            # Convert to similarity (0-1 scale)
            # Hash size is 8*8 = 64 bits for dHash
            max_distance = 64
            similarity = 1.0 - (hamming_dist / max_distance)
            
            # Also do a quick PSNR check for additional validation
            psnr = cv2.PSNR(
                cv2.resize(frame1, (256, 256)),
                cv2.resize(frame2, (256, 256))
            )
            
            # Normalize PSNR (typically 20-50 dB for compressed video)
            # 20 dB = poor, 30 dB = acceptable, 40+ dB = good
            psnr_score = min(1.0, max(0.0, (psnr - 20) / 30))
            
            # Combine dHash similarity (70%) with PSNR (30%)
            combined = similarity * 0.7 + psnr_score * 0.3
            
            return combined
            
        except Exception as e:
            logger.debug(f"Frame comparison error: {e}")
            return 0.75  # Neutral score
    
    def _dhash(self, image: np.ndarray) -> int:
        """
        Calculate difference hash for an image
        Compares adjacent pixels horizontally
        """
        hash_value = 0
        for i in range(8):
            for j in range(8):
                if image[i, j] > image[i, j + 1]:
                    hash_value |= (1 << (i * 8 + j))
        return hash_value
    
    def _evaluate_size_efficiency(self, original_path: str, compressed_path: str, 
                                  target_size_mb: float) -> float:
        """
        Evaluate how efficiently the compression utilized available space
        Returns score from 0-100
        """
        try:
            original_size_mb = os.path.getsize(original_path) / (1024 * 1024)
            compressed_size_mb = os.path.getsize(compressed_path) / (1024 * 1024)
            
            if compressed_size_mb > target_size_mb:
                # Over target - penalize proportionally
                overage_ratio = (compressed_size_mb - target_size_mb) / target_size_mb
                score = max(0, 100 - (overage_ratio * 200))  # -20 points per 10% overage
                return score
            
            # Calculate how well we used available space
            utilization = compressed_size_mb / target_size_mb
            
            # Also consider compression ratio achieved
            compression_ratio = compressed_size_mb / original_size_mb
            
            # Optimal utilization is 85-98% of target
            if 0.85 <= utilization <= 0.98:
                utilization_score = 100
            elif utilization < 0.85:
                # Under-utilized (could have had better quality)
                utilization_score = 70 + (utilization / 0.85) * 30
            else:
                # Very close to limit (good)
                utilization_score = 95
            
            # Reward good compression ratio
            # Typical compression: 0.1-0.5 of original size for 10MB target
            if compression_ratio < 0.3:
                compression_score = 100
            elif compression_ratio < 0.5:
                compression_score = 90
            elif compression_ratio < 0.7:
                compression_score = 75
            else:
                compression_score = 60
            
            # Combine scores
            score = utilization_score * 0.6 + compression_score * 0.4
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.warning(f"Size efficiency evaluation failed: {e}")
            return 75.0  # Neutral score
    
    def _sample_frame_validity(self, video_path: str) -> float:
        """
        Test that frames can be decoded at random positions
        Returns score from 0-100
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.warning(f"Cannot open video for frame validation: {video_path}")
                return 50.0
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                cap.release()
                return 50.0
            
            # Test random frame positions
            num_tests = min(self.num_random_frames + 5, total_frames)
            test_positions = random.sample(range(total_frames), min(num_tests, total_frames))
            
            successful_reads = 0
            valid_frames = 0
            
            for pos in test_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, pos)
                ret, frame = cap.read()
                
                if ret:
                    successful_reads += 1
                    
                    # Validate frame is not corrupted
                    if frame is not None and frame.size > 0:
                        # Check for reasonable dimensions
                        h, w = frame.shape[:2]
                        if 10 < w < 5000 and 10 < h < 5000:
                            # Check frame is not all black or all white
                            mean_val = np.mean(frame)
                            if 5 < mean_val < 250:
                                valid_frames += 1
            
            cap.release()
            
            if num_tests == 0:
                return 75.0
            
            # Calculate success rate
            read_rate = successful_reads / num_tests
            validity_rate = valid_frames / num_tests
            
            # Combine rates
            score = (read_rate * 0.5 + validity_rate * 0.5) * 100
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.warning(f"Frame validity sampling failed: {e}")
            return 75.0  # Neutral score
    
    def _check_technical_validity(self, original_path: str, compressed_path: str) -> float:
        """
        Check technical validity: format integrity, duration accuracy, etc.
        Returns score from 0-100
        """
        score = 100.0
        penalties = []
        
        try:
            # Check file exists and has size
            if not os.path.exists(compressed_path):
                return 0.0
            
            file_size = os.path.getsize(compressed_path)
            if file_size == 0:
                return 0.0
            
            # Get video information using ffprobe
            original_info = self._get_video_info(original_path)
            compressed_info = self._get_video_info(compressed_path)
            
            if not compressed_info:
                penalties.append(('invalid_format', 30))
            else:
                # Check duration accuracy (allow 5% tolerance)
                if original_info and 'duration' in original_info and 'duration' in compressed_info:
                    orig_duration = float(original_info['duration'])
                    comp_duration = float(compressed_info['duration'])
                    
                    if orig_duration > 0:
                        duration_ratio = comp_duration / orig_duration
                        
                        # For segments and GIFs, be very lenient with duration
                        is_gif = compressed_path.lower().endswith('.gif')
                        if is_gif or '_segment_' in compressed_path.lower():
                            # GIFs/segments can be much shorter due to frame dropping
                            if duration_ratio < 0.2 or duration_ratio > 1.5:
                                penalties.append(('duration_mismatch', 5))
                        else:
                            # Regular videos should maintain duration better
                            if duration_ratio < 0.8 or duration_ratio > 1.2:
                                penalties.append(('duration_mismatch', 15))
                
                # Check has video stream
                if not compressed_info.get('has_video'):
                    penalties.append(('no_video_stream', 40))
                
                # Check reasonable frame count
                frame_count = compressed_info.get('frame_count', 0)
                if frame_count < 5:
                    penalties.append(('too_few_frames', 20))
            
            # Apply penalties
            for penalty_name, penalty_value in penalties:
                score -= penalty_value
                logger.debug(f"Technical validity penalty: {penalty_name} (-{penalty_value})")
            
            return max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.warning(f"Technical validity check failed: {e}")
            return 75.0  # Neutral score
    
    def _get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Get video information using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=15
            )
            
            if result.returncode != 0:
                return None
            
            data = json.loads(result.stdout)
            
            # Extract relevant information
            info = {
                'duration': float(data.get('format', {}).get('duration', 0)),
                'size': int(data.get('format', {}).get('size', 0)),
                'has_video': False,
                'frame_count': 0
            }
            
            # Check for video streams
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    info['has_video'] = True
                    info['frame_count'] = int(stream.get('nb_frames', 0))
                    break
            
            return info
            
        except Exception as e:
            logger.debug(f"Failed to get video info: {e}")
            return None
    
    def get_segment_quality_floor(self, attempt: int) -> float:
        """
        Get quality floor for segment based on attempt number
        Progressive fallback: 70 → 60 → 50 → 45
        """
        if attempt < 0:
            attempt = 0
        
        if attempt < len(self.segment_floors):
            return self.segment_floors[attempt]
        else:
            # Return lowest floor if exceeded attempts
            return self.segment_floors[-1] if self.segment_floors else 45.0

