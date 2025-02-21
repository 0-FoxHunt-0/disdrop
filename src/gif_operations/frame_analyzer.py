import asyncio
from pathlib import Path
from typing import List, NamedTuple, Dict
from cachetools import TTLCache
import cv2
import numpy as np
import threading
import logging

logger = logging.getLogger(__name__)


class FrameAnalysis(NamedTuple):
    """Frame analysis results"""
    similarity_score: float
    motion_score: float
    color_count: int
    brightness: float
    complexity: float


class FrameAnalyzer:
    """Analyzes GIF frames for optimal compression strategy"""

    def __init__(self):
        self.cache = TTLCache(maxsize=100, ttl=300)
        self._lock = threading.Lock()

    async def analyze_file(self, file_path: Path) -> FrameAnalysis:
        """Analyze file characteristics for optimization"""
        cache_key = f"{file_path}:{file_path.stat().st_mtime}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        frames = await self._load_frames(file_path)

        similarity = self._calculate_frame_similarity(frames)
        motion = self._analyze_motion(frames)
        colors = self._analyze_color_distribution(frames)
        brightness = self._calculate_brightness(frames)
        complexity = self._calculate_image_complexity(frames)

        analysis = FrameAnalysis(
            similarity_score=similarity,
            motion_score=motion,
            color_count=colors,
            brightness=brightness,
            complexity=complexity
        )

        self.cache[cache_key] = analysis
        return analysis

    async def analyze_frame(self, frame: np.ndarray) -> Dict[str, float]:
        """Analyze a single frame."""
        try:
            # Convert to grayscale if not already
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Basic metrics
            brightness = np.mean(frame) / 255.0
            complexity = self._calculate_frame_complexity(frame)

            return {
                'brightness': brightness,
                'complexity': complexity
            }
        except Exception as e:
            logger.error(f"Frame analysis error: {e}")
            return {'brightness': 0.5, 'complexity': 0.5}  # Safe defaults

    def _calculate_frame_complexity(self, frame: np.ndarray) -> float:
        """Calculate frame complexity using edge detection and entropy."""
        try:
            # Edge detection
            edges = cv2.Canny(frame, 100, 200)
            edge_density = np.count_nonzero(edges) / edges.size

            # Calculate entropy
            hist = cv2.calcHist([frame], [0], None, [256], [0, 256])
            hist = hist / hist.sum()
            entropy = -np.sum(hist * np.log2(hist + 1e-7))
            normalized_entropy = entropy / 8.0  # Max entropy for 8-bit image

            # Combine metrics
            return (edge_density * 0.6 + normalized_entropy * 0.4)
        except Exception as e:
            logger.error(f"Complexity calculation error: {e}")
            return 0.5  # Safe default

    def _calculate_frame_similarity(self, frames: List[np.ndarray]) -> float:
        """Calculate similarity between consecutive frames"""
        if len(frames) < 2:
            return 1.0

        similarities = []
        for i in range(len(frames) - 1):
            diff = np.abs(frames[i+1] - frames[i]).mean()
            similarities.append(1 - (diff / 255))

        return np.mean(similarities)

    def _analyze_motion(self, frames: List[np.ndarray]) -> float:
        """Analyze motion between frames"""
        if len(frames) < 2:
            return 0.0

        motion_scores = []
        for i in range(len(frames) - 1):
            flow = cv2.calcOpticalFlowFarneback(
                frames[i], frames[i+1], None,
                0.5, 3, 15, 3, 5, 1.2, 0)
            motion_scores.append(np.abs(flow).mean())

        return np.mean(motion_scores)

    async def _load_frames(self, file_path: Path) -> List[np.ndarray]:
        """Load frames from GIF/video with efficient memory handling"""
        frames = []
        cap = cv2.VideoCapture(str(file_path))
        try:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            # Process in chunks to save memory
            chunk_size = min(30, total_frames)

            for i in range(0, total_frames, chunk_size):
                chunk_frames = []
                for _ in range(chunk_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    # Convert to grayscale for analysis
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    chunk_frames.append(gray)

                frames.extend(chunk_frames)
                # Allow other tasks to run
                await asyncio.sleep(0)

        finally:
            cap.release()

        return frames

    def _analyze_color_distribution(self, frames: List[np.ndarray]) -> int:
        """Analyze color distribution and count unique colors"""
        unique_colors = set()
        # Sample every 10th frame
        sample_frames = frames[::max(1, len(frames) // 10)]

        for frame in sample_frames:
            # Convert to RGB for better color analysis
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            # Downsample for efficiency
            downsampled = cv2.resize(rgb, (160, 90))
            colors = map(tuple, downsampled.reshape(-1, 3))
            unique_colors.update(colors)

        return len(unique_colors)

    def _calculate_brightness(self, frames: List[np.ndarray]) -> float:
        """Calculate average brightness of frames"""
        brightnesses = []
        for frame in frames:
            mean_brightness = np.mean(frame)
            brightnesses.append(mean_brightness / 255.0)
        return np.mean(brightnesses)

    def _calculate_image_complexity(self, frames: List[np.ndarray]) -> float:
        """Calculate image complexity using edge detection"""
        complexities = []
        for frame in frames:
            edges = cv2.Canny(frame, 100, 200)
            complexity = np.count_nonzero(
                edges) / (frame.shape[0] * frame.shape[1])
            complexities.append(complexity)
        return np.mean(complexities)
