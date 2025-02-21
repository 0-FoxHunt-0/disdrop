import math
from pathlib import Path
import logging
from typing import Dict, Tuple, Optional, List, Union
import threading
import time
import cv2
import numpy as np
from PIL import Image, ImageChops
import json
import subprocess
import shutil
from dataclasses import dataclass
from enum import Enum

from src.default_config import TEMP_FILE_DIR
from src.gif_operations.processing_stats import ProcessingStats
from src.logging_system import run_ffmpeg_command
from src.video_optimization import VideoProcessor

from .memory import MemoryManager
from .ffmpeg_handler import ffmpeg_handler


class ProcessingStatus(Enum):
    SUCCESS = "success"
    DIMENSION_ERROR = "dimension_error"
    PALETTE_ERROR = "palette_error"
    CONVERSION_ERROR = "conversion_error"
    OPTIMIZATION_ERROR = "optimization_error"
    FILE_ERROR = "file_error"
    SIZE_THRESHOLD_EXCEEDED = "size_threshold_exceeded"


@dataclass
class OptimizationConfig:
    scale_factor: float
    colors: int
    lossy_value: int
    dither_mode: str = 'bayer'
    bayer_scale: int = 3


@dataclass
class OptimizationResult:
    fps: int
    size: float
    path: Optional[str]
    status: ProcessingStatus
    message: str = ""
    settings: Optional[OptimizationConfig] = None


class GIFOptimizer:
    """Base class for GIF optimization."""

    def __init__(self, compression_settings: Dict = None):
        # Initialize loggers
        self.dev_logger = logging.getLogger('developer')
        self.dev_logger.setLevel(logging.DEBUG)

        # Core attributes
        self.compression_settings = compression_settings
        self.failed_files = []

        # Initialize components
        self.ffmpeg = ffmpeg_handler
        self.memory_manager = MemoryManager()
        self.stats = ProcessingStats()

        # Thread management
        self._shutdown_event = threading.Event()
        self._processing_cancelled = threading.Event()
        self._shutdown_initiated = False

        # Locks
        self._lock = threading.Lock()
        self._file_locks = {}
        self._file_locks_lock = threading.Lock()

    def optimize_gif(self, input_path: Path, output_path: Path, target_size_mb: float) -> Tuple[float, bool]:
        """Main optimization method."""
        try:
            initial_size = self.get_file_size(input_path)
            if initial_size <= target_size_mb:
                shutil.copy2(input_path, output_path)
                return initial_size, True

            # Get dimensions
            dimensions = self._get_dimensions_with_retry(input_path)
            if not dimensions:
                return initial_size, False

            # Initial optimization settings
            settings = self._get_initial_settings(input_path, dimensions)

            # Progressive optimization
            size, success = self._progressive_optimize(
                input_path, output_path, target_size_mb, settings, dimensions)

            # Record stats
            self.stats.add_result(
                input_path, initial_size, size, success)

            return size, success

        except Exception as e:
            self.dev_logger.error(f"Optimization failed: {str(e)}")
            return initial_size, False

    def _should_exit(self) -> bool:
        """Check if processing should stop."""
        return (self._shutdown_event.is_set() or
                self._processing_cancelled.is_set() or
                self._shutdown_initiated)

    def _get_dimensions_with_retry(self, file_path: Path, max_retries: int = 3) -> Optional[Tuple[int, int]]:
        """Get file dimensions with retry logic."""
        for attempt in range(max_retries):
            try:
                return self.ffmpeg.get_dimensions(file_path)
            except Exception as e:
                if attempt == max_retries - 1:
                    self.dev_logger.error(f"Failed to get dimensions: {e}")
                    return None
                time.sleep(0.5 * (attempt + 1))

    def _get_initial_settings(self, input_path: Path, dimensions: Tuple[int, int]) -> OptimizationConfig:
        """Get initial optimization settings."""
        # Base settings
        settings = OptimizationConfig(
            scale_factor=1.0,
            colors=256,
            lossy_value=20,
            dither_mode='floyd_steinberg',
            bayer_scale=3
        )

        # Adjust based on image characteristics
        return self._adjust_settings_for_input(settings, input_path, dimensions)

    def _adjust_settings_for_input(self, settings: OptimizationConfig,
                                   input_path: Path,
                                   dimensions: Tuple[int, int]) -> OptimizationConfig:
        """Adjust optimization settings based on input characteristics."""
        max_dimension = max(dimensions)
        file_size = self.get_file_size(input_path)

        # Adjust scale factor based on dimensions
        if max_dimension > 1920:
            settings.scale_factor = 0.75
        elif max_dimension > 1280:
            settings.scale_factor = 0.85

        # Adjust colors based on file size
        if file_size > 50:
            settings.colors = 192
        elif file_size > 100:
            settings.colors = 128

        return settings

    def _progressive_optimize(self, input_path: Path, output_path: Path,
                              target_size: float, settings: OptimizationConfig,
                              dimensions: Tuple[int, int]) -> Tuple[float, bool]:
        """Progressive optimization with quality preservation."""
        best_result = {'size': float('inf'), 'settings': None}
        current_settings = settings
        timestamp = int(time.time() * 1000)
        temp_files = []

        try:
            for attempt in range(8):  # Max 8 optimization attempts
                if self._should_exit():
                    break

                # Create temp file directly in TEMP_FILE_DIR
                temp_output = Path(
                    TEMP_FILE_DIR) / f"temp_{timestamp}_attempt_{attempt}_{input_path.stem}.gif"
                temp_files.append(temp_output)

                result_size = self._apply_optimization(
                    input_path, temp_output, current_settings, dimensions)

                try:
                    if result_size < best_result['size']:
                        if 'path' in best_result and best_result['path']:
                            Path(best_result['path']).unlink(missing_ok=True)
                        best_result = {
                            'size': result_size,
                            'settings': current_settings,
                            'path': str(temp_output)
                        }

                    if result_size <= target_size:
                        shutil.move(str(temp_output), str(output_path))
                        return result_size, True
                    else:
                        temp_output.unlink(missing_ok=True)

                    # Update settings for next attempt
                    current_settings = self._get_next_settings(
                        current_settings, result_size, target_size)

                except Exception as e:
                    self.dev_logger.error(
                        f"Optimization error in attempt {attempt}: {e}")
                    temp_output.unlink(missing_ok=True)

        except Exception as e:
            self.dev_logger.error(f"Optimization error: {e}")
            return float('inf'), False
        finally:
            # Clean up all temp files
            for temp_file in temp_files:
                try:
                    if temp_file.exists():
                        temp_file.unlink()
                except Exception as e:
                    self.dev_logger.error(f"Failed to cleanup temp file: {e}")

        # Use best result if target not met
        if best_result['path'] and Path(best_result['path']).exists():
            try:
                shutil.move(best_result['path'], str(output_path))
                return best_result['size'], best_result['size'] <= target_size
            except Exception as e:
                self.dev_logger.error(f"Failed to move best result: {e}")

        return float('inf'), False

    def _apply_optimization(self, input_path: Path, output_path: Path,
                            settings: OptimizationConfig,
                            dimensions: Tuple[int, int]) -> float:
        """Apply optimization settings to create GIF."""
        try:
            if self.ffmpeg.create_optimized_gif(
                input_path, output_path, 30, dimensions, {
                    'colors': settings.colors,
                    'scale_factor': settings.scale_factor,
                    'dither_mode': settings.dither_mode,
                    'bayer_scale': settings.bayer_scale
                }
            ):
                return self.get_file_size(output_path)
            return float('inf')
        except Exception as e:
            self.dev_logger.error(f"Optimization error: {e}")
            return float('inf')

    @staticmethod
    def get_file_size(file_path: Path) -> float:
        """Get file size in MB."""
        return file_path.stat().st_size / (1024 * 1024)

    def cleanup(self):
        """Cleanup resources."""
        self._shutdown_initiated = True
        self._shutdown_event.set()
        self.memory_manager.cleanup()


class DynamicGIFOptimizer(GIFOptimizer):
    """Dynamic GIF optimizer with adaptive quality settings."""

    def __init__(self, compression_settings=None):
        super().__init__(compression_settings)
        # Historical optimization results for learning
        self.optimization_history = {}
        # Quality thresholds
        self.quality_thresholds = {
            'high': 0.85,
            'medium': 0.70,
            'low': 0.50
        }
        # Initialize adaptive parameters
        self.param_bounds = {
            'scale_factor': (0.2, 1.0),
            'colors': (32, 256),
            'lossy_value': (0, 100),
            'dither_modes': ['floyd_steinberg', 'bayer', 'sierra2_4a']
        }

    def optimize_gif(self, input_path: Path, output_path: Path, target_size_mb: float) -> Tuple[float, bool]:
        """Optimize GIF with dynamic quality adjustment."""
        try:
            original_size = self.get_file_size(input_path)
            current_best_size = original_size
            best_output = None

            if original_size <= target_size_mb:
                shutil.copy2(input_path, output_path)
                return original_size, True

            # Analyze input characteristics
            image_complexity = self._analyze_image_complexity(input_path)
            motion_complexity = self._analyze_motion_complexity(input_path)
            color_count = self._analyze_color_distribution(input_path)

            # Get similar historical results
            similar_cases = self._find_similar_cases(
                original_size, target_size_mb, image_complexity, motion_complexity)

            # Initialize parameters based on analysis
            current_settings = self._get_initial_settings(
                original_size, target_size_mb, image_complexity,
                motion_complexity, color_count, similar_cases
            )

            best_result = {'size': float('inf'), 'settings': None}
            adaptation_count = 0
            max_adaptations = 5

            for attempt in range(8):  # Max 8 optimization attempts
                if self._should_exit():
                    break

                temp_output = Path(
                    TEMP_FILE_DIR) / f"temp_{timestamp}_attempt_{attempt}_{input_path.stem}.gif"
                temp_files.append(temp_output)

                result_size = self._apply_optimization(
                    input_path, temp_output, current_settings, dimensions)

                try:
                    # Add size validation
                    if result_size > original_size * 1.1:  # If result is 10% larger than original
                        self.dev_logger.warning(
                            f"Optimization attempt {attempt} produced larger file, reverting to previous settings")
                        temp_output.unlink(missing_ok=True)
                        # Revert to previous successful settings if available
                        if best_result['settings']:
                            current_settings = best_result['settings']
                        continue

                    if result_size < current_best_size:
                        if best_output and best_output.exists():
                            best_output.unlink()
                        best_output = temp_output
                        current_best_size = result_size
                        # Create a safe copy of the best result
                        shutil.copy2(temp_output, Path(
                            TEMP_FILE_DIR) / f"best_{input_path.stem}.gif")

                    # More conservative parameter adjustment
                    if result_size > target_size_mb:
                        # Only adjust one parameter at a time
                        if attempt % 3 == 0:
                            current_settings.scale_factor = max(
                                0.25, current_settings.scale_factor * 0.9)
                        elif attempt % 3 == 1:
                            current_settings.colors = max(
                                64, current_settings.colors - 32)
                        else:
                            current_settings.lossy_value = min(
                                100, current_settings.lossy_value + 10)

                except Exception as e:
                    self.dev_logger.error(
                        f"Error during optimization attempt {attempt}: {e}")
                    continue

            while adaptation_count < max_adaptations:
                result = self._try_optimization(
                    input_path, output_path, current_settings)

                if result['success']:
                    # Update optimization history
                    self._update_history(original_size, target_size_mb,
                                         image_complexity, motion_complexity,
                                         current_settings, result)

                    if result['size'] <= target_size_mb:
                        return result['size'], True

                    if result['size'] < best_result['size']:
                        best_result = result

                # Adapt parameters based on result
                current_settings = self._adapt_parameters(
                    current_settings, result, target_size_mb,
                    image_complexity, motion_complexity
                )
                adaptation_count += 1

            # Use best result if target not met
            if best_result['settings']:
                final_result = self._try_optimization(
                    input_path, output_path, best_result['settings'])
                return final_result['size'], final_result['size'] <= target_size_mb

            return original_size, False

        except Exception as e:
            self.dev_logger.error(f"Dynamic optimization failed: {str(e)}")
            return original_size, False

    def _analyze_image_complexity(self, image_path: Path) -> float:
        """Analyze image complexity using entropy and edge detection."""
        try:
            with Image.open(image_path) as img:
                # Convert to grayscale for analysis
                gray = img.convert('L')
                # Calculate entropy
                histogram = gray.histogram()
                total_pixels = sum(histogram)
                entropy = 0
                for pixel_count in histogram:
                    if pixel_count > 0:
                        probability = pixel_count / total_pixels
                        entropy -= probability * math.log2(probability)
                return entropy / 8.0  # Normalize to 0-1
        except Exception:
            return 0.5  # Default to medium complexity

    def _analyze_motion_complexity(self, gif_path: Path) -> float:
        """Analyze motion complexity between frames."""
        try:
            with Image.open(gif_path) as img:
                if not getattr(img, 'is_animated', False):
                    return 0.0

                frames = []
                differences = []
                try:
                    while True:
                        frames.append(img.convert('L'))
                        img.seek(img.tell() + 1)
                except EOFError:
                    pass

                for i in range(len(frames) - 1):
                    diff = ImageChops.difference(frames[i], frames[i + 1])
                    diff_ratio = sum(diff.getdata()) / \
                        (diff.width * diff.height * 255)
                    differences.append(diff_ratio)

                return sum(differences) / len(differences) if differences else 0.0
        except Exception:
            return 0.5

    def _adapt_parameters(self, current_settings: OptimizationConfig,
                          result: dict, target_size: float,
                          image_complexity: float, motion_complexity: float) -> OptimizationConfig:
        """Adapt parameters based on optimization results and complexity metrics."""
        size_ratio = result['size'] / target_size

        # Dynamic adjustment factors
        quality_sensitivity = 1.0 - \
            (image_complexity * 0.5 + motion_complexity * 0.5)
        scale_factor = current_settings.scale_factor
        colors = current_settings.colors
        lossy_value = current_settings.lossy_value

        if size_ratio > 1.5:  # Much larger than target
            scale_factor *= max(0.7, 1 - (size_ratio - 1) * 0.2)
            colors = int(colors * 0.75)
            lossy_value = min(100, lossy_value + 20)
        elif size_ratio > 1.1:  # Slightly larger than target
            scale_factor *= max(0.85, 1 - (size_ratio - 1) * 0.1)
            colors = int(colors * 0.9)
            lossy_value = min(100, lossy_value + 10)
        elif size_ratio < 0.8:  # Much smaller than target
            scale_factor = min(1.0, scale_factor * 1.1)
            colors = min(256, int(colors * 1.2))
            lossy_value = max(0, lossy_value - 10)

        # Apply quality sensitivity
        scale_factor = max(self.param_bounds['scale_factor'][0],
                           min(self.param_bounds['scale_factor'][1],
                               scale_factor * (1 + quality_sensitivity * 0.1)))

        colors = max(self.param_bounds['colors'][0],
                     min(self.param_bounds['colors'][1],
                         int(colors * (1 + quality_sensitivity * 0.1))))

        return OptimizationConfig(
            scale_factor=scale_factor,
            colors=colors,
            lossy_value=lossy_value,
            dither_mode=self._select_dither_mode(
                image_complexity, motion_complexity)
        )

    def _select_dither_mode(self, image_complexity: float, motion_complexity: float) -> str:
        """Select appropriate dither mode based on content characteristics."""
        if image_complexity > 0.8:
            return 'floyd_steinberg'  # Better for complex images
        elif motion_complexity > 0.6:
            return 'sierra2_4a'  # Better for high motion
        else:
            return 'bayer'  # Good balance for simpler content

    def _update_history(self, original_size: float, target_size: float,
                        image_complexity: float, motion_complexity: float,
                        settings: OptimizationConfig, result: dict) -> None:
        """Update optimization history for learning."""
        key = f"{int(original_size)}MB_{int(target_size)}MB"
        if key not in self.optimization_history:
            self.optimization_history[key] = []

        self.optimization_history[key].append({
            'image_complexity': image_complexity,
            'motion_complexity': motion_complexity,
            'settings': settings,
            'result': result,
            'effectiveness': target_size / result['size'] if result['size'] > 0 else 0
        })

        # Keep only the most recent and effective results
        self.optimization_history[key] = sorted(
            self.optimization_history[key],
            key=lambda x: x['effectiveness'],
            reverse=True
        )[:5]

    def _analyze_color_distribution(self, input_path: Path) -> int:
        """Analyze color distribution in the GIF to determine optimal color count."""
        try:
            with Image.open(input_path) as img:
                # Get all frames
                frames = []
                try:
                    while True:
                        frames.append(img.copy())
                        img.seek(img.tell() + 1)
                except EOFError:
                    pass

                # Analyze color distribution across frames
                total_unique_colors = set()
                color_frequencies = {}

                for frame in frames:
                    # Convert to RGB for consistent color analysis
                    rgb_frame = frame.convert('RGB')
                    colors = rgb_frame.getcolors(
                        maxcolors=257)  # 257 to detect if > 256

                    if colors is None:  # More than 256 colors
                        return 256

                    for count, color in colors:
                        total_unique_colors.add(color)
                        color_frequencies[color] = color_frequencies.get(
                            color, 0) + count

                # Calculate significant colors (those that appear frequently)
                total_pixels = sum(freq for freq in color_frequencies.values())
                significant_threshold = total_pixels * 0.001  # 0.1% threshold
                significant_colors = sum(1 for freq in color_frequencies.values()
                                         if freq > significant_threshold)

                # Calculate optimal color count
                if significant_colors > 192:
                    return 256
                elif significant_colors > 128:
                    return 192
                elif significant_colors > 64:
                    return 128
                else:
                    return 64

        except Exception as e:
            self.dev_logger.warning(
                f"Color analysis failed: {e}, using default")
            return 256

    def _find_similar_cases(self, original_size: float, target_size: float,
                            image_complexity: float, motion_complexity: float) -> List[Dict]:
        """Find similar historical optimization cases."""
        key = f"{int(original_size)}MB_{int(target_size)}MB"
        if key not in self.optimization_history:
            return []

        similar_cases = []
        for case in self.optimization_history[key]:
            # Calculate similarity score based on complexity metrics
            complexity_diff = abs(case['image_complexity'] - image_complexity) + \
                abs(case['motion_complexity'] - motion_complexity)

            if complexity_diff < 0.3:  # Consider cases with similar complexity
                similar_cases.append(case)

        return sorted(similar_cases, key=lambda x: x['effectiveness'], reverse=True)

    def _get_initial_settings(self, original_size: float, target_size: float,
                              image_complexity: float, motion_complexity: float,
                              color_count: int, similar_cases: List[Dict]) -> OptimizationConfig:
        """Get initial settings based on file characteristics and history."""
        if similar_cases:
            # Use most effective historical case as base
            best_case = similar_cases[0]
            settings = best_case['settings']
            # Slightly adjust settings based on current characteristics
            settings.scale_factor *= (target_size / original_size) ** 0.5
            return settings

        # Calculate initial settings based on characteristics
        size_ratio = target_size / original_size

        # Base scale factor on size ratio and complexity
        scale_factor = min(1.0, max(0.3,
                                    size_ratio ** 0.5 *
                                    (1 - image_complexity * 0.3)
                                    ))

        # Adjust colors based on analyzed color count
        colors = min(color_count, 256)
        if size_ratio < 0.3:
            colors = min(colors, 192)

        # Set lossy value based on complexity
        lossy_value = int(20 + (1 - image_complexity) * 40)

        # Select dither mode based on content
        dither_mode = self._select_dither_mode(
            image_complexity, motion_complexity)

        return OptimizationConfig(
            scale_factor=scale_factor,
            colors=colors,
            lossy_value=lossy_value,
            dither_mode=dither_mode
        )

    def _try_optimization(self, input_path: Path, output_path: Path,
                          settings: OptimizationConfig) -> Dict:
        """Try optimization with given settings."""
        try:
            if self._apply_optimization(input_path, output_path, settings,
                                        self._get_dimensions_with_retry(input_path)):
                size = self.get_file_size(output_path)
                return {
                    'success': True,
                    'size': size,
                    'settings': settings
                }
        except Exception as e:
            self.dev_logger.error(f"Optimization attempt failed: {e}")

        return {
            'success': False,
            'size': float('inf'),
            'settings': settings
        }
