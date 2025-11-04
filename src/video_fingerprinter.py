"""
Video fingerprinting module for content-based hashing and cache key generation.

This module provides functionality to create unique fingerprints for videos based on
their content, enabling efficient caching and similarity detection.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np
try:
    from .ffmpeg_utils import FFmpegUtils
    from .logger_setup import get_logger
except ImportError:
    # Fallback for direct execution
    try:
        from ffmpeg_utils import FFmpegUtils
        from logger_setup import get_logger
    except ImportError:
        # Create minimal fallbacks for testing
        class FFmpegUtils:
            def run_command(self, cmd, capture_output=False):
                class Result:
                    returncode = 1
                    stdout = ""
                return Result()
        
        import logging
        def get_logger(name):
            return logging.getLogger(name)

logger = get_logger(__name__)


@dataclass
class VideoFingerprint:
    """Represents a video's content-based fingerprint."""
    content_hash: str
    perceptual_hash: str
    duration: float
    resolution: Tuple[int, int]
    frame_count: int
    file_size: int
    metadata_hash: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert fingerprint to dictionary for serialization."""
        return {
            'content_hash': self.content_hash,
            'perceptual_hash': self.perceptual_hash,
            'duration': self.duration,
            'resolution': self.resolution,
            'frame_count': self.frame_count,
            'file_size': self.file_size,
            'metadata_hash': self.metadata_hash
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoFingerprint':
        """Create fingerprint from dictionary."""
        return cls(
            content_hash=data['content_hash'],
            perceptual_hash=data['perceptual_hash'],
            duration=data['duration'],
            resolution=tuple(data['resolution']),
            frame_count=data['frame_count'],
            file_size=data['file_size'],
            metadata_hash=data['metadata_hash']
        )


@dataclass
class CompressionParams:
    """Represents compression parameters for cache key generation."""
    codec: str
    bitrate: Optional[int]
    resolution: Optional[Tuple[int, int]]
    fps: Optional[float]
    preset: Optional[str]
    crf: Optional[int]
    additional_params: Dict[str, Any]
    
    def to_cache_key_part(self) -> str:
        """Generate cache key component from compression parameters."""
        params_dict = {
            'codec': self.codec,
            'bitrate': self.bitrate,
            'resolution': self.resolution,
            'fps': self.fps,
            'preset': self.preset,
            'crf': self.crf,
            'additional': sorted(self.additional_params.items()) if self.additional_params else None
        }
        
        # Create deterministic string representation
        params_str = json.dumps(params_dict, sort_keys=True, default=str)
        return hashlib.sha256(params_str.encode()).hexdigest()[:16]


class VideoFingerprinter:
    """Generates content-based fingerprints for videos."""
    
    def __init__(self, ffmpeg_utils: Optional[FFmpegUtils] = None):
        """Initialize the video fingerprinter."""
        self.ffmpeg_utils = ffmpeg_utils or FFmpegUtils()
        self._sample_frames = 10  # Number of frames to sample for perceptual hashing
        self._hash_size = 8  # Size of perceptual hash
    
    def generate_fingerprint(self, video_path: str) -> VideoFingerprint:
        """
        Generate a comprehensive fingerprint for a video file.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            VideoFingerprint containing content and perceptual hashes
        """
        logger.debug(f"Generating fingerprint for video: {video_path}")
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Get basic file information
        file_size = os.path.getsize(video_path)
        
        # Extract video metadata
        metadata = self._extract_video_metadata(video_path)
        duration = metadata.get('duration', 0.0)
        resolution = metadata.get('resolution', (0, 0))
        frame_count = metadata.get('frame_count', 0)
        
        # Generate content hash (based on file content)
        content_hash = self._generate_content_hash(video_path)
        
        # Generate perceptual hash (based on visual content)
        perceptual_hash = self._generate_perceptual_hash(video_path)
        
        # Generate metadata hash
        metadata_hash = self._generate_metadata_hash(metadata)
        
        fingerprint = VideoFingerprint(
            content_hash=content_hash,
            perceptual_hash=perceptual_hash,
            duration=duration,
            resolution=resolution,
            frame_count=frame_count,
            file_size=file_size,
            metadata_hash=metadata_hash
        )
        
        logger.debug(f"Generated fingerprint: content={content_hash[:8]}..., perceptual={perceptual_hash[:8]}...")
        return fingerprint
    
    def _extract_video_metadata(self, video_path: str) -> Dict[str, Any]:
        """Extract video metadata using FFmpeg."""
        try:
            # Use FFprobe to get video information
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = self.ffmpeg_utils.run_command(cmd, capture_output=True)
            if result.returncode != 0:
                logger.warning(f"Failed to extract metadata from {video_path}")
                return {}
            
            metadata = json.loads(result.stdout)
            
            # Extract video stream information
            video_stream = None
            for stream in metadata.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                logger.warning(f"No video stream found in {video_path}")
                return {}
            
            # Extract relevant information
            duration = float(metadata.get('format', {}).get('duration', 0))
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            frame_count = int(video_stream.get('nb_frames', 0))
            
            # If frame count is not available, estimate from duration and fps
            if frame_count == 0 and duration > 0:
                fps_str = video_stream.get('r_frame_rate', '0/1')
                if '/' in fps_str:
                    num, den = fps_str.split('/')
                    fps = float(num) / float(den) if float(den) > 0 else 0
                    frame_count = int(duration * fps)
            
            return {
                'duration': duration,
                'resolution': (width, height),
                'frame_count': frame_count,
                'codec': video_stream.get('codec_name', ''),
                'bitrate': int(video_stream.get('bit_rate', 0)),
                'fps': video_stream.get('r_frame_rate', ''),
                'pixel_format': video_stream.get('pix_fmt', '')
            }
            
        except Exception as e:
            logger.error(f"Error extracting metadata from {video_path}: {e}")
            return {}
    
    def _generate_content_hash(self, video_path: str) -> str:
        """Generate hash based on file content."""
        hasher = hashlib.sha256()
        
        try:
            with open(video_path, 'rb') as f:
                # Read file in chunks to handle large files
                for chunk in iter(lambda: f.read(8192), b""):
                    hasher.update(chunk)
            
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating content hash for {video_path}: {e}")
            return ""
    
    def _generate_perceptual_hash(self, video_path: str) -> str:
        """Generate perceptual hash based on visual content."""
        try:
            # Extract sample frames from the video
            frames = self._extract_sample_frames(video_path)
            if not frames:
                logger.warning(f"No frames extracted for perceptual hash: {video_path}")
                return ""
            
            # Compute average hash for each frame
            frame_hashes = []
            for frame in frames:
                frame_hash = self._compute_frame_hash(frame)
                frame_hashes.append(frame_hash)
            
            # Combine frame hashes into a single perceptual hash
            combined_hash = ''.join(frame_hashes)
            
            # Create final hash from combined frame hashes
            hasher = hashlib.md5()
            hasher.update(combined_hash.encode())
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating perceptual hash for {video_path}: {e}")
            return ""
    
    def _extract_sample_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract sample frames from video for perceptual hashing."""
        frames = []
        
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Could not open video: {video_path}")
                return frames
            
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames == 0:
                logger.warning(f"Video has no frames: {video_path}")
                return frames
            
            # Calculate frame indices to sample
            sample_indices = []
            if total_frames <= self._sample_frames:
                sample_indices = list(range(total_frames))
            else:
                step = total_frames // self._sample_frames
                sample_indices = [i * step for i in range(self._sample_frames)]
            
            # Extract frames at calculated indices
            for frame_idx in sample_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret and frame is not None:
                    # Convert to grayscale and resize for consistent hashing
                    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    resized_frame = cv2.resize(gray_frame, (64, 64))
                    frames.append(resized_frame)
            
            cap.release()
            
        except Exception as e:
            logger.error(f"Error extracting sample frames from {video_path}: {e}")
        
        return frames
    
    def _compute_frame_hash(self, frame: np.ndarray) -> str:
        """Compute perceptual hash for a single frame using average hash algorithm."""
        try:
            # Resize to hash size
            resized = cv2.resize(frame, (self._hash_size, self._hash_size))
            
            # Calculate average pixel value
            avg_pixel = np.mean(resized)
            
            # Create binary hash based on average
            binary_hash = ''
            for row in resized:
                for pixel in row:
                    binary_hash += '1' if pixel > avg_pixel else '0'
            
            # Convert binary to hexadecimal
            hex_hash = hex(int(binary_hash, 2))[2:].zfill(16)
            return hex_hash
            
        except Exception as e:
            logger.error(f"Error computing frame hash: {e}")
            return "0" * 16
    
    def _generate_metadata_hash(self, metadata: Dict[str, Any]) -> str:
        """Generate hash from video metadata."""
        try:
            # Create deterministic string from metadata
            metadata_str = json.dumps(metadata, sort_keys=True, default=str)
            hasher = hashlib.md5()
            hasher.update(metadata_str.encode())
            return hasher.hexdigest()
            
        except Exception as e:
            logger.error(f"Error generating metadata hash: {e}")
            return ""
    
    def calculate_similarity(self, hash1: str, hash2: str) -> float:
        """
        Calculate similarity between two perceptual hashes.
        
        Args:
            hash1: First perceptual hash
            hash2: Second perceptual hash
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        if not hash1 or not hash2 or len(hash1) != len(hash2):
            return 0.0
        
        try:
            # Convert hex hashes to binary
            bin1 = bin(int(hash1, 16))[2:].zfill(len(hash1) * 4)
            bin2 = bin(int(hash2, 16))[2:].zfill(len(hash2) * 4)
            
            # Calculate Hamming distance
            hamming_distance = sum(c1 != c2 for c1, c2 in zip(bin1, bin2))
            
            # Convert to similarity score
            max_distance = len(bin1)
            similarity = 1.0 - (hamming_distance / max_distance)
            
            return similarity
            
        except Exception as e:
            logger.error(f"Error calculating hash similarity: {e}")
            return 0.0


class CacheKeyGenerator:
    """Generates cache keys for video compression results."""
    
    def __init__(self, fingerprinter: Optional[VideoFingerprinter] = None):
        """Initialize the cache key generator."""
        self.fingerprinter = fingerprinter or VideoFingerprinter()
        self._collision_counter = {}
    
    def generate_cache_key(self, video_path: str, params: CompressionParams) -> str:
        """
        Generate a unique cache key for video and compression parameters.
        
        Args:
            video_path: Path to the video file
            params: Compression parameters
            
        Returns:
            Unique cache key string
        """
        try:
            # Generate video fingerprint
            fingerprint = self.fingerprinter.generate_fingerprint(video_path)
            
            # Create base key from content hash and parameters
            params_key = params.to_cache_key_part()
            base_key = f"{fingerprint.content_hash[:16]}_{params_key}"
            
            # Handle potential collisions
            final_key = self._handle_collision(base_key)
            
            logger.debug(f"Generated cache key: {final_key}")
            return final_key
            
        except Exception as e:
            logger.error(f"Error generating cache key for {video_path}: {e}")
            # Fallback to simple hash
            fallback_data = f"{video_path}_{params.to_cache_key_part()}"
            return hashlib.sha256(fallback_data.encode()).hexdigest()[:32]
    
    def generate_similarity_key(self, video_path: str) -> str:
        """
        Generate a key for similarity-based cache lookups.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Similarity key based on perceptual hash
        """
        try:
            fingerprint = self.fingerprinter.generate_fingerprint(video_path)
            return fingerprint.perceptual_hash
            
        except Exception as e:
            logger.error(f"Error generating similarity key for {video_path}: {e}")
            return ""
    
    def _handle_collision(self, base_key: str) -> str:
        """Handle cache key collisions by adding a counter."""
        if base_key not in self._collision_counter:
            self._collision_counter[base_key] = 0
            return base_key
        
        # Increment collision counter and append to key
        self._collision_counter[base_key] += 1
        collision_suffix = f"_c{self._collision_counter[base_key]}"
        return base_key + collision_suffix
    
    def extract_fingerprint_from_key(self, cache_key: str) -> Optional[str]:
        """
        Extract the content hash portion from a cache key.
        
        Args:
            cache_key: Cache key string
            
        Returns:
            Content hash if extractable, None otherwise
        """
        try:
            # Cache key format: {content_hash}_{params_hash}[_c{collision_num}]
            parts = cache_key.split('_')
            if len(parts) >= 2:
                return parts[0]
            return None
            
        except Exception as e:
            logger.error(f"Error extracting fingerprint from cache key {cache_key}: {e}")
            return None