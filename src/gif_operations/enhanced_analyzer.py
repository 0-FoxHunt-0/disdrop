import asyncio
from pathlib import Path
from typing import List, Dict, NamedTuple, Optional, Tuple
import cv2
import numpy as np
from PIL import Image, ImageChops
import threading
import logging
from cachetools import TTLCache
from concurrent.futures import ThreadPoolExecutor
import math

logger = logging.getLogger(__name__)


class EnhancedGIFAnalysis(NamedTuple):
    """Comprehensive GIF analysis results for optimized processing"""
    motion_score: float              # 0-1 score indicating motion amount
    complexity_score: float          # 0-1 score indicating visual complexity
    color_richness: float            # 0-1 score indicating color importance
    perceptual_quality: float        # Perceptual quality estimate
    temporal_coherence: float        # Frame-to-frame coherence measure
    optimal_fps: int                 # Recommended FPS based on motion
    optimal_colors: int              # Recommended color count
    optimal_dither: str              # Recommended dither method
    dimensions: Tuple[int, int]      # Width and height
    frame_count: int                 # Number of frames
    duration_ms: int                 # Duration in milliseconds
    has_transparency: bool           # Whether transparency is used


class EnhancedAnalyzer:
    """Advanced GIF analysis for high-quality optimization"""

    def __init__(self, max_workers: int = 4):
        self.cache = TTLCache(maxsize=200, ttl=600)  # 10 minute cache
        self._lock = threading.Lock()
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Define dither methods with quality scores
        self.dither_methods = {
            'none': 0.0,              # No dithering - lowest quality, smallest size
            'bayer:bayer_scale=2': 0.3,  # Bayer dithering with small pattern
            'bayer:bayer_scale=3': 0.5,  # Medium Bayer pattern
            'bayer:bayer_scale=4': 0.7,  # Large Bayer pattern - good balance
            'floyd_steinberg': 0.9,   # Floyd-Steinberg - highest quality, largest size
        }

    async def analyze_gif(self, file_path: Path) -> EnhancedGIFAnalysis:
        """Perform comprehensive GIF analysis with caching"""
        cache_key = f"{file_path}:{file_path.stat().st_mtime}"

        with self._lock:
            if cache_key in self.cache:
                return self.cache[cache_key]

        try:
            # Extract basic info using PIL
            with Image.open(file_path) as img:
                if not getattr(img, "is_animated", False):
                    # Handle static GIFs
                    return self._create_static_analysis(img)

                # Get basic properties
                frame_count = getattr(img, "n_frames", 1)
                width, height = img.size
                has_transparency = img.info.get(
                    'transparency') is not None or 'transparency' in img.info

                # Calculate total duration
                duration_ms = 0
                for i in range(frame_count):
                    img.seek(i)
                    # Default 100ms
                    duration_ms += img.info.get('duration', 100)

                # Extract frames for detailed analysis
                frames = []
                img.seek(0)
                for i in range(frame_count):
                    # Sample frames (skip some if too many)
                    if frame_count > 50 and i % math.ceil(frame_count / 50) != 0:
                        img.seek(i)
                        continue

                    frame = np.array(img.convert('RGB'))
                    frames.append(frame)
                    try:
                        img.seek(i + 1)
                    except EOFError:
                        break

            # Perform detailed analysis with frames
            motion_score = await self._analyze_motion(frames)
            complexity_score = self._calculate_complexity(frames)
            color_richness = self._analyze_color_importance(frames)
            perceptual_quality = self._estimate_perceptual_quality(frames)
            temporal_coherence = self._calculate_temporal_coherence(frames)

            # Calculate optimal parameters
            optimal_fps = self._calculate_optimal_fps(
                motion_score, frame_count, duration_ms)
            optimal_colors = self._calculate_optimal_colors(
                color_richness, complexity_score)
            optimal_dither = self._select_optimal_dither(
                complexity_score, color_richness)

            # Create analysis result
            analysis = EnhancedGIFAnalysis(
                motion_score=motion_score,
                complexity_score=complexity_score,
                color_richness=color_richness,
                perceptual_quality=perceptual_quality,
                temporal_coherence=temporal_coherence,
                optimal_fps=optimal_fps,
                optimal_colors=optimal_colors,
                optimal_dither=optimal_dither,
                dimensions=(width, height),
                frame_count=frame_count,
                duration_ms=duration_ms,
                has_transparency=has_transparency
            )

            # Cache the result
            with self._lock:
                self.cache[cache_key] = analysis

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing GIF {file_path}: {e}")
            # Return fallback analysis with conservative values
            return self._create_fallback_analysis()

    def _create_static_analysis(self, img: Image.Image) -> EnhancedGIFAnalysis:
        """Create analysis for static GIFs"""
        width, height = img.size
        has_transparency = img.info.get(
            'transparency') is not None or 'transparency' in img.info

        # Convert to numpy for analysis
        np_img = np.array(img.convert('RGB'))
        complexity = self._calculate_static_complexity(np_img)
        color_richness = self._analyze_static_color_importance(np_img)

        return EnhancedGIFAnalysis(
            motion_score=0.0,  # Static image
            complexity_score=complexity,
            color_richness=color_richness,
            perceptual_quality=0.9,  # Static images generally preserve quality well
            temporal_coherence=1.0,  # Perfect coherence (static)
            optimal_fps=0,  # Not applicable
            optimal_colors=self._calculate_optimal_colors(
                color_richness, complexity),
            optimal_dither=self._select_optimal_dither(
                complexity, color_richness),
            dimensions=(width, height),
            frame_count=1,
            duration_ms=0,
            has_transparency=has_transparency
        )

    def _create_fallback_analysis(self) -> EnhancedGIFAnalysis:
        """Create fallback analysis with conservative values"""
        return EnhancedGIFAnalysis(
            motion_score=0.5,
            complexity_score=0.5,
            color_richness=0.5,
            perceptual_quality=0.5,
            temporal_coherence=0.5,
            optimal_fps=15,
            optimal_colors=128,
            optimal_dither='bayer:bayer_scale=3',
            dimensions=(0, 0),
            frame_count=0,
            duration_ms=0,
            has_transparency=False
        )

    async def _analyze_motion(self, frames: List[np.ndarray]) -> float:
        """Analyze motion between frames using optical flow"""
        if len(frames) < 2:
            return 0.0

        # Use thread pool for CPU-intensive operations
        loop = asyncio.get_event_loop()
        motion_scores = []

        # Process frame pairs in chunks to avoid memory issues
        for i in range(len(frames) - 1):
            # Use background thread for CPU-intensive operations
            score = await loop.run_in_executor(
                self.executor,
                self._calculate_frame_motion,
                frames[i], frames[i+1]
            )
            motion_scores.append(score)

            # Allow other async tasks to run
            if i % 5 == 0:
                await asyncio.sleep(0)

        # Normalize motion score between 0 and 1
        avg_motion = np.mean(motion_scores) if motion_scores else 0
        # Scale factor determined empirically
        normalized_motion = min(1.0, avg_motion / 0.05)

        return normalized_motion

    def _calculate_frame_motion(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        """Calculate motion between two frames"""
        try:
            # Convert to grayscale if needed
            if len(frame1.shape) == 3:
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
            else:
                gray1, gray2 = frame1, frame2

            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                gray1, gray2, None,
                0.5, 3, 15, 3, 5, 1.2, 0
            )

            # Calculate magnitude
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # Get mean motion
            return float(np.mean(mag))
        except Exception as e:
            logger.error(f"Error calculating frame motion: {e}")
            return 0.1  # Conservative default

    def _calculate_complexity(self, frames: List[np.ndarray]) -> float:
        """Calculate visual complexity across frames"""
        if not frames:
            return 0.5

        complexities = []
        sample_frames = frames[::max(1, len(frames) // min(10, len(frames)))]

        for frame in sample_frames:
            # Convert to grayscale
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            else:
                gray = frame

            # Calculate edge density
            edges = cv2.Canny(gray, 100, 200)
            edge_density = np.count_nonzero(edges) / edges.size

            # Calculate entropy
            entropy = self._calculate_entropy(gray)

            # Combined score
            complexity = 0.6 * edge_density + 0.4 * entropy
            complexities.append(complexity)

        return float(np.mean(complexities))

    def _calculate_static_complexity(self, image: np.ndarray) -> float:
        """Calculate complexity for a single frame"""
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Calculate edge density
        edges = cv2.Canny(gray, 100, 200)
        edge_density = np.count_nonzero(edges) / edges.size

        # Calculate entropy
        entropy = self._calculate_entropy(gray)

        # Combined score
        return float(0.6 * edge_density + 0.4 * entropy)

    def _calculate_entropy(self, image: np.ndarray) -> float:
        """Calculate normalized entropy of an image"""
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist / hist.sum()
        non_zero = hist > 0
        if np.any(non_zero):
            entropy = -np.sum(hist[non_zero] * np.log2(hist[non_zero]))
            return float(entropy / 8.0)  # Normalize by max entropy (8 bits)
        return 0.0

    def _analyze_color_importance(self, frames: List[np.ndarray]) -> float:
        """Analyze color importance in the GIF"""
        if not frames:
            return 0.5

        color_scores = []
        sample_frames = frames[::max(1, len(frames) // min(10, len(frames)))]

        for frame in sample_frames:
            if len(frame.shape) != 3:
                continue

            # Split channels
            b, g, r = cv2.split(frame)

            # Calculate channel variances
            var_r = np.var(r) / 255.0
            var_g = np.var(g) / 255.0
            var_b = np.var(b) / 255.0

            # Calculate grayscale version
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            var_gray = np.var(gray) / 255.0

            # Color importance is high when color variance is much higher than grayscale variance
            if var_gray > 0:
                color_ratio = (var_r + var_g + var_b) / (3 * var_gray)
                color_importance = min(
                    1.0, max(0.0, (color_ratio - 1.0) * 2.0))
            else:
                color_importance = 0.5

            color_scores.append(color_importance)

        return float(np.mean(color_scores) if color_scores else 0.5)

    def _analyze_static_color_importance(self, image: np.ndarray) -> float:
        """Analyze color importance for a single image"""
        if len(image.shape) != 3:
            return 0.3  # Conservative default for grayscale

        # Split channels
        b, g, r = cv2.split(image)

        # Calculate channel variances
        var_r = np.var(r) / 255.0
        var_g = np.var(g) / 255.0
        var_b = np.var(b) / 255.0

        # Calculate grayscale version
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        var_gray = np.var(gray) / 255.0

        # Color importance is high when color variance is much higher than grayscale variance
        if var_gray > 0:
            color_ratio = (var_r + var_g + var_b) / (3 * var_gray)
            return float(min(1.0, max(0.0, (color_ratio - 1.0) * 2.0)))
        return 0.5  # Default value

    def _estimate_perceptual_quality(self, frames: List[np.ndarray]) -> float:
        """Estimate perceptual quality of the GIF"""
        if not frames:
            return 0.5

        # Calculate average SSIM between consecutive frames
        if len(frames) < 2:
            return 0.9  # Single frame has high quality

        ssim_scores = []
        for i in range(len(frames) - 1):
            if i % 5 == 0:  # Sample every 5th frame pair for efficiency
                try:
                    frame1 = frames[i]
                    frame2 = frames[i+1]

                    # Convert to grayscale for SSIM
                    if len(frame1.shape) == 3:
                        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
                    else:
                        gray1, gray2 = frame1, frame2

                    # Calculate SSIM
                    ssim = self._calculate_ssim(gray1, gray2)
                    ssim_scores.append(ssim)
                except Exception as e:
                    logger.error(f"Error calculating SSIM: {e}")

        # Higher SSIM indicates higher quality baseline
        return float(np.mean(ssim_scores) if ssim_scores else 0.5)

    def _calculate_ssim(self, img1: np.ndarray, img2: np.ndarray) -> float:
        """Calculate SSIM between two grayscale images"""
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)

        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
            ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        return float(np.mean(ssim_map))

    def _calculate_temporal_coherence(self, frames: List[np.ndarray]) -> float:
        """Calculate temporal coherence between frames"""
        if len(frames) < 2:
            return 1.0  # Perfect coherence for single frame

        # Calculate mean absolute difference between consecutive frames
        diffs = []
        for i in range(len(frames) - 1):
            if i % 3 == 0:  # Sample for efficiency
                frame1 = frames[i]
                frame2 = frames[i+1]

                # Convert to grayscale
                if len(frame1.shape) == 3:
                    gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
                    gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
                else:
                    gray1, gray2 = frame1, frame2

                # Calculate difference
                diff = np.abs(gray1.astype(np.float32) -
                              gray2.astype(np.float32))
                mean_diff = np.mean(diff) / 255.0
                diffs.append(mean_diff)

        # Higher difference = lower coherence
        mean_diff = np.mean(diffs) if diffs else 0
        # Scale the coherence score
        coherence = max(0.0, 1.0 - mean_diff * 2.0)

        return float(coherence)

    def _calculate_optimal_fps(self, motion_score: float, frame_count: int, duration_ms: int) -> int:
        """Calculate optimal FPS based on motion"""
        if frame_count <= 1 or duration_ms == 0:
            return 0

        # Calculate original FPS
        original_fps = (frame_count * 1000) / duration_ms

        # Adjust based on motion
        if motion_score < 0.2:  # Low motion
            optimal_fps = max(8, original_fps * 0.6)
        elif motion_score < 0.5:  # Medium motion
            optimal_fps = max(12, original_fps * 0.8)
        else:  # High motion
            optimal_fps = max(15, original_fps * 0.9)

        # Cap at reasonable values
        optimal_fps = min(30, optimal_fps)

        return int(round(optimal_fps))

    def _calculate_optimal_colors(self, color_richness: float, complexity_score: float) -> int:
        """Calculate optimal color count based on color richness and complexity"""
        # Base color count on color richness and complexity
        if color_richness > 0.7 and complexity_score > 0.7:
            # High color richness and complexity
            optimal_colors = 192
        elif color_richness > 0.5 or complexity_score > 0.6:
            # Medium color richness or high complexity
            optimal_colors = 128
        elif color_richness > 0.3:
            # Moderate color richness
            optimal_colors = 96
        else:
            # Low color richness
            optimal_colors = 64

        return optimal_colors

    def _select_optimal_dither(self, complexity_score: float, color_richness: float) -> str:
        """Select optimal dither method based on image characteristics"""
        # Combined score for dither selection
        dither_score = 0.6 * complexity_score + 0.4 * color_richness

        # Select dither method based on score
        if dither_score < 0.2:
            return 'none'  # No dithering for very simple content
        elif dither_score < 0.4:
            return 'bayer:bayer_scale=2'  # Simple Bayer for basic content
        elif dither_score < 0.7:
            return 'bayer:bayer_scale=3'  # Medium Bayer for normal content
        elif dither_score < 0.85:
            return 'bayer:bayer_scale=4'  # Stronger Bayer for complex content
        else:
            return 'floyd_steinberg'  # Floyd-Steinberg for very complex content
