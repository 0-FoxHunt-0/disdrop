from itertools import chain
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
import os
import queue
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed, ProcessPoolExecutor
from dataclasses import dataclass
from enum import Enum
from functools import wraps
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Generator
import psutil
import traceback
import json
from cachetools import TTLCache
import cv2
import ctypes
import gc
from typing import NamedTuple
from functools import lru_cache
import asyncio
from contextlib import contextmanager, asynccontextmanager  # Fix the import error
import numpy as np
from PIL import Image, ImageSequence
import io

# Fix the imports to use absolute paths since this is imported from main.py
from src.default_config import (GIF_COMPRESSION, GIF_PASS_OVERS, GIF_SIZE_TO_SKIP,
                                INPUT_DIR, OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS,
                                TEMP_FILE_DIR)
from src.logging_system import log_function_call
from src.temp_file_manager import TempFileManager
from src.video_optimization import VideoProcessor
from src.utils.error_handler import VideoProcessingError
from src.utils.video_dimensions import get_video_dimensions
from .utils.resource_manager import ResourceMonitor, ResourceGuard
from .utils.ffmpeg_handler import run_ffmpeg_command, ffmpeg_handler

# Setup logger
logger = logging.getLogger(__name__)


class OptimizationConfig(NamedTuple):
    scale_factor: float
    colors: int
    lossy_value: int
    dither_mode: str = 'bayer'
    bayer_scale: int = 3


class ProcessingStatus(Enum):
    SUCCESS = "success"
    DIMENSION_ERROR = "dimension_error"
    PALETTE_ERROR = "palette_error"
    CONVERSION_ERROR = "conversion_error"
    OPTIMIZATION_ERROR = "optimization_error"
    FILE_ERROR = "file_error"
    SIZE_THRESHOLD_EXCEEDED = "size_threshold_exceeded"


@dataclass
class ProcessingResult:
    fps: int
    size: float
    path: Optional[str]
    status: ProcessingStatus
    message: str = ""
    settings: Optional[OptimizationConfig] = None


class FileProcessor:
    """Base class for file processing operations."""

    def __init__(self):
        self.file_size_cache = TTLCache(maxsize=1000, ttl=3600)

    @staticmethod
    def wait_for_file_completion(file_path: Union[str, Path], timeout: int = 30) -> bool:
        """Wait for file to be completely written and accessible."""
        file_path = Path(file_path)
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Try to open the file in read mode
                with open(file_path, 'rb') as f:
                    # Try to seek to end to ensure complete file access
                    f.seek(0, 2)
                # Force sync to filesystem
                if hasattr(os, 'sync'):
                    os.sync()
                # Get initial size
                initial_size = file_path.stat().st_size
                # Wait a small interval
                time.sleep(0.1)
                # Check if size is stable
                if initial_size == file_path.stat().st_size:
                    return True
            except (IOError, OSError):
                time.sleep(0.1)
                continue
        return False

    def get_file_size(self, file_path: Union[str, Path], force_refresh: bool = True) -> float:
        """Get file size in MB with improved reliability."""
        try:
            file_path = Path(file_path)

            if force_refresh:
                # Clear any cached size
                self.file_size_cache.pop(str(file_path), None)

                # Wait for file to be completely written
                if not self.wait_for_file_completion(file_path):
                    logging.warning(
                        f"File may not be completely written: {file_path}")

            # Get fresh size
            size = file_path.stat().st_size / (1024 * 1024)
            # Update cache
            self.file_size_cache[str(file_path)] = size
            return size
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return float('inf')
        except Exception as e:
            logging.error(f"Error getting file size: {e}")
            return float('inf')

    @staticmethod
    def ensure_directory(directory: Path) -> None:
        """Ensure directory exists."""
        directory.mkdir(parents=True, exist_ok=True)


class FFmpegHandler:
    """Handles FFmpeg-related operations."""

    def __init__(self):
        self._current_process = None
        self._process_lock = threading.Lock()
        self._current_processes = set()  # Track all active processes

    def _kill_current_process(self):
        """Safely kill the current FFmpeg process."""
        with self._process_lock:
            for process in self._current_processes:
                try:
                    if sys.platform == 'win32':
                        subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                    else:
                        os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except Exception as e:
                    logging.error(f"Error killing process {process.pid}: {e}")
            self._current_processes.clear()

    @staticmethod
    def create_optimized_gif(file_path: Path, output_path: Path,
                             fps: int, dimensions: Tuple[int, int], settings: Dict) -> bool:
        """Create optimized GIF in a single pass."""
        try:
            cmd = [
                'ffmpeg',
                '-i', str(file_path),
                '-vf',
                f'fps={fps},'
                f'scale={dimensions[0]}:{dimensions[1]}:flags=lanczos,'
                'split[s0][s1];'
                '[s0]palettegen=max_colors={colors}:stats_mode=diff[p];'
                '[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle'.format(
                    colors=settings.get('colors', 256)
                ),
                '-y',
                str(output_path)
            ]

            return run_ffmpeg_command(cmd)

        except Exception as e:
            logging.error(f"GIF creation failed: {str(e)}")
            return False


class DynamicGIFOptimizer:
    """Handles dynamic GIF optimization with adaptive settings."""

    def __init__(self):
        self.dev_logger = logging.getLogger('developer')
        self.dev_logger.setLevel(logging.DEBUG)
        self.base_settings = {
            'colors': 256,
            'lossy_value': 15,
            'scale_factor': 1.0
        }
        self.optimization_history = {}

    def optimize_gif(self, input_path: Path, output_path: Path, target_size_mb: float = 10.0) -> Tuple[float, bool]:
        """
        Dynamically optimize GIF using adaptive settings based on results.

        Args:
            input_path: Path to input GIF
            output_path: Path to output optimized GIF
            target_size_mb: Target size in MB (default 10MB)

        Returns:
            Tuple[float, bool]: (final_size, success)
        """
        input_size = self.get_file_size(input_path)
        self.dev_logger.info(
            f"Starting optimization: {input_size:.2f}MB → Target: {target_size_mb}MB")

        current_settings = self.base_settings.copy()
        best_result = {'size': float('inf'), 'settings': None}
        attempt = 0
        max_attempts = 8

        size_ratio = target_size_mb / input_size
        current_settings['scale_factor'] = self._calculate_initial_scale_factor(
            input_size, size_ratio)

        # Track used settings to avoid duplicates
        tried_settings = set()

        while attempt < max_attempts:
            attempt += 1
            settings_key = (
                current_settings['scale_factor'],
                current_settings['colors'],
                current_settings['lossy_value']
            )

            # Skip if we've already tried these settings
            if settings_key in tried_settings:
                current_settings = self._get_aggressive_settings(
                    current_settings, attempt)
                continue

            tried_settings.add(settings_key)

            # Single log block for attempt header and settings
            self.dev_logger.info(f"\n{'-'*50}")
            self.dev_logger.info(
                f"Optimization Attempt {attempt}/{max_attempts}")
            self.dev_logger.info(f"{'-'*50}")
            self.dev_logger.info("Settings:")

            # Get and log dimensions
            width = height = None
            if hasattr(VideoProcessor, '_get_dimensions'):
                try:
                    width, height = VideoProcessor()._get_dimensions(input_path)
                    width = int(width * current_settings['scale_factor'])
                    height = int(height * current_settings['scale_factor'])
                    self.dev_logger.info(f"- Size: {width}x{height}")
                except:
                    pass

            self.dev_logger.info(
                f"- Scale: {current_settings['scale_factor']:.3f}")
            self.dev_logger.info(f"- Colors: {current_settings['colors']}")
            self.dev_logger.info(f"- Lossy: {current_settings['lossy_value']}")

            # Apply optimization
            result_size = self._apply_optimization(
                input_path, output_path, current_settings)
            reduction = ((input_size - result_size) / input_size) * 100

            # Log results once
            self.dev_logger.info(
                f"Result: {result_size:.2f}MB ({reduction:.1f}% reduction)")

            # Track best result
            if result_size < best_result['size']:
                best_result = {
                    'size': result_size,
                    'settings': current_settings.copy(),
                    'attempt': attempt,
                    'reduction': reduction
                }

            # Check if target achieved
            if result_size <= target_size_mb:
                self.dev_logger.info(f"Target achieved: {result_size:.2f}MB")
                return result_size, True

            # Update settings with improved logic
            if attempt < max_attempts:
                new_settings = self._adjust_settings(
                    current_settings,
                    input_size,
                    result_size,
                    target_size_mb,
                    attempt,
                    best_result
                )

                if self._settings_similar(current_settings, new_settings):
                    new_settings = self._get_aggressive_settings(
                        current_settings, attempt)

                current_settings = new_settings

        # Use best result if target not achieved
        if best_result['settings']:
            self.dev_logger.info(
                f"\nUsing best settings from attempt {best_result['attempt']} "
                f"({best_result['reduction']:.1f}% reduction)"
            )
            final_size = self._apply_optimization(
                input_path, output_path, best_result['settings'])
            return final_size, final_size <= target_size_mb

        return input_size, False

    def _calculate_initial_scale_factor(self, input_size: float, size_ratio: float) -> float:
        """Calculate initial scale factor with more conservative scaling."""
        # Start with more conservative base scaling
        if input_size > 200:
            base_scale = min(1.0, max(0.5, (size_ratio ** 0.4)))
        elif input_size > 100:
            base_scale = min(1.0, max(0.6, (size_ratio ** 0.35)))
        elif input_size > 50:
            base_scale = min(1.0, max(0.7, (size_ratio ** 0.3)))
        else:
            base_scale = min(1.0, max(0.8, (size_ratio ** 0.25)))

        # Less aggressive scaling for high-ratio cases
        if size_ratio < 0.1:
            base_scale *= 0.9
        elif size_ratio < 0.2:
            base_scale *= 0.95

        # Never go below 50% of original size
        return round(max(0.5, base_scale), 3)

    def _apply_optimization(self, input_path: Path, output_path: Path, settings: Dict) -> float:
        """Enhanced optimization with frame analysis."""
        try:
            # Analyze frame characteristics first
            frame_info = self._analyze_frames(input_path)

            cmd = [
                'gifsicle',
                '--optimize=3',
                '--colors', str(settings['colors']),
                '--lossy=' + str(settings['lossy_value'])
            ]

            # Apply frame-specific optimizations
            if frame_info['similar_frames'] > 0.5:  # If more than 50% frames are similar
                cmd.extend(['--merge'])  # Merge similar frames

            if frame_info['disposal_method'] == 'background':
                cmd.append('--disposal=background')
            elif frame_info['disposal_method'] == 'previous':
                cmd.append('--disposal=previous')

            # Add scaling if needed
            if settings['scale_factor'] < 1.0:
                width, height = VideoProcessor()._get_dimensions(input_path)
                if width and height:
                    new_width = int(width * settings['scale_factor'])
                    new_height = int(height * settings['scale_factor'])
                    cmd.extend(['--resize', f'{new_width}x{new_height}'])

            # Add performance optimizations
            cmd.extend([
                '--no-conserve-memory',
                '--careful',
                '--threads=4'
            ])

            cmd.extend(['--batch', str(input_path), '-o', str(output_path)])

            if run_ffmpeg_command(cmd):
                return self.get_file_size(output_path)
            return float('inf')

        except Exception as e:
            self.dev_logger.error(f"Optimization error: {str(e)}")
            return float('inf')

    def _adjust_settings(self, current: Dict, input_size: float, result_size: float,
                         target_size: float, attempt: int, best_result: Dict) -> Dict:
        """Enhanced settings adjustment prioritizing quality."""
        new_settings = current.copy()
        size_ratio = result_size / target_size
        distance_from_target = abs(1 - size_ratio)

        # More conservative adjustment factors
        scale_adjust = min(0.95, max(0.8, 1 - (distance_from_target * 0.2)))
        colors_adjust = min(0.95, max(0.85, 1 - (distance_from_target * 0.15)))

        if size_ratio > 1:  # Still too large
            # Try color and lossy adjustments before scaling
            if attempt <= 2:
                # First try reducing colors if we're starting with max
                if new_settings['colors'] > 192:
                    new_settings['colors'] = 192
                # And adjust lossy compression
                new_settings['lossy_value'] = min(
                    100, new_settings['lossy_value'] + 15)
            elif attempt <= 4:
                # Only now start reducing scale, but more conservatively
                new_settings['scale_factor'] *= scale_adjust
                if new_settings['colors'] > 128:
                    new_settings['colors'] = 128
            else:
                # Last resort scaling
                new_settings['scale_factor'] *= scale_adjust ** 0.9
                new_settings['colors'] = max(
                    128, int(new_settings['colors'] * colors_adjust))

            # Ensure we never go below 50% of original size
            new_settings['scale_factor'] = max(
                0.5, new_settings['scale_factor'])

        else:  # Under target, try to improve quality
            if best_result['size'] < result_size:
                new_settings = self._get_quality_focused_settings(
                    current, attempt)
            else:
                # Try to recover quality
                new_settings['scale_factor'] = min(
                    1.0, new_settings['scale_factor'] * 1.05)
                new_settings['colors'] = min(256, new_settings['colors'] + 32)
                new_settings['lossy_value'] = max(
                    15, new_settings['lossy_value'] - 15)

        return new_settings

    def _get_aggressive_settings(self, current: Dict, attempt: int) -> Dict:
        """Get more aggressive settings when normal adjustments aren't enough."""
        aggressive = current.copy()

        # Start with scale and lossy adjustments
        if attempt <= 3:
            aggressive['scale_factor'] *= 0.85
            aggressive['lossy_value'] = min(80, aggressive['lossy_value'] + 20)
        elif attempt <= 5:
            aggressive['scale_factor'] *= 0.75
            aggressive['lossy_value'] = min(90, aggressive['lossy_value'] + 15)
        else:
            # Only reduce colors in later attempts
            aggressive['scale_factor'] *= 0.7
            aggressive['colors'] = max(128, aggressive['colors'] - 32)
            aggressive['lossy_value'] = min(
                100, aggressive['lossy_value'] + 10)

        return aggressive

    def _get_quality_focused_settings(self, current: Dict, attempt: int) -> Dict:
        """Get quality-focused settings for optimization."""
        settings = current.copy()

        # Prioritize color depth and lossy adjustments over scaling
        if attempt <= 2:
            settings['lossy_value'] = min(100, settings['lossy_value'] + 20)
        elif attempt <= 4:
            settings['colors'] = max(128, settings['colors'] - 32)
            settings['lossy_value'] = min(100, settings['lossy_value'] + 15)
        else:
            # Only reduce scale as a last resort
            settings['scale_factor'] = max(0.5, settings['scale_factor'] * 0.9)

        return settings

    @staticmethod
    def _settings_similar(settings1: Dict, settings2: Dict, threshold: float = 0.1) -> bool:
        """Check if two settings are very similar."""
        return (abs(settings1['scale_factor'] - settings2['scale_factor']) < threshold and
                abs(settings1['colors'] - settings2['colors']) < 32 and
                abs(settings1['lossy_value'] - settings2['lossy_value']) < 20)

    @staticmethod
    def get_file_size(file_path: Union[str, Path]) -> float:
        """Get file size in MB."""
        return Path(file_path).stat().st_size / (1024 * 1024)

    def _analyze_frames(self, gif_path: Path) -> Dict:
        """Analyze GIF frames for optimization hints."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'frame=pkt_size',
                '-of', 'json',
                str(gif_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            data = json.loads(result.stdout)

            frame_sizes = [frame.get('pkt_size', 0)
                           for frame in data.get('frames', [])]
            if not frame_sizes:
                return {
                    'similar_frames': 0.0,
                    'disposal_method': 'none'
                }

            # Calculate frame similarity ratio
            avg_size = sum(frame_sizes) / len(frame_sizes)
            similar_frames = sum(1 for size in frame_sizes if abs(
                size - avg_size) < avg_size * 0.1)
            similarity_ratio = similar_frames / len(frame_sizes)

            # Determine optimal disposal method
            if similarity_ratio > 0.7:  # Lots of similar frames
                disposal = 'background'
            elif similarity_ratio > 0.4:  # Some similarity
                disposal = 'previous'
            else:
                disposal = 'none'

            return {
                'similar_frames': similarity_ratio,
                'disposal_method': disposal
            }

        except Exception as e:
            self.dev_logger.warning(f"Frame analysis failed: {e}")
            return {
                'similar_frames': 0.0,
                'disposal_method': 'none'
            }


class GIFOptimizer(FileProcessor):
    """Handles GIF optimization operations."""

    def __init__(self, compression_settings: Dict = None):
        super().__init__()
        self.compression_settings = compression_settings or GIF_COMPRESSION
        self.failed_files = []
        self.dev_logger = logging.getLogger('developer')
        self.dynamic_optimizer = DynamicGIFOptimizer()
        self.base_settings = {
            'colors': 256,
            'lossy_value': 15,
            'scale_factor': 1.0
        }
        self._init_directories()

    def _init_directories(self) -> None:
        """Initialize required directories."""
        for directory in [INPUT_DIR, OUTPUT_DIR, TEMP_FILE_DIR]:
            try:
                directory.mkdir(parents=True, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create directory {directory}: {e}")
                raise


class GIFProcessor(GIFOptimizer):
    """Main GIF processing class combining optimization and conversion."""

    def __init__(self, compression_settings: Dict = None):
        # Initialize GIFOptimizer first
        super().__init__(compression_settings)

        # Initialize processor-specific attributes
        self._process_lock = threading.Lock()
        self.ffmpeg = FFmpegHandler()
        self._shutdown_event = threading.Event()
        self._processing_lock = threading.Lock()
        self._active_threads = set()
        self._threads_lock = threading.Lock()
        self._file_locks = {}
        self._file_locks_lock = threading.Lock()
        self._processing_cancelled = threading.Event()
        self.logging_lock = threading.Lock()
        self.processed_files = set()
        self.size_check_lock = threading.Lock()
        self.dynamic_optimizer = DynamicGIFOptimizer()
        self._processing_progress = {}
        self._progress_lock = threading.Lock()
        self.palette_cache = {}
        self.palette_lock = threading.Lock()
        self._processing_timeout = 600
        self.resource_manager = ResourceManager()
        self.retry_count = 3
        self.retry_delay = 1.0
        self._stats = {'processed': 0, 'failed': 0, 'retried': 0}
        self._stats_lock = threading.Lock()
        self.memory_manager = MemoryManager(threshold_mb=1500)
        self.batch_processor = BatchProcessor()
        self.stats_manager = ProcessingStats()
        self._file_cache = TTLCache(maxsize=100, ttl=300)
        self._cleanup_handlers = []  # Add this line
        self._shutdown_initiated = False
        signal.signal(signal.SIGINT, self._signal_handler)
        # Add resource monitor
        self.resource_monitor = ResourceMonitor()

        # Thread management
        self.worker_threads = []
        self.max_threads = 2  # Limit concurrent threads
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self._stop_workers = threading.Event()

        # Add immediate termination flag
        self._immediate_termination = threading.Event()
        # Register signal handlers for immediate termination
        signal.signal(signal.SIGINT, self._immediate_shutdown_handler)
        signal.signal(signal.SIGTERM, self._immediate_shutdown_handler)

        # Add async support
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        # Add async support
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # Add dimension cache
        self.dimension_cache = TTLCache(maxsize=100, ttl=300)

        # Add progressive optimization
        self.progressive_optimization = True

        # Add automatic quality adjustment
        self.quality_manager = QualityManager()

    def get_file_lock(self, file_path: str) -> threading.Lock:
        """Get or create a lock for a specific file."""
        with self._file_locks_lock:
            if file_path not in self._file_locks:
                self._file_locks[file_path] = threading.Lock()
            return self._file_locks[file_path]

    def _register_thread(self):
        """Register current thread as active."""
        with self._threads_lock:
            self._active_threads.add(threading.current_thread())

    def _unregister_thread(self):
        """Unregister current thread."""
        with self._threads_lock:
            self._active_threads.discard(threading.current_thread())

    def cleanup_resources(self) -> None:
        """Enhanced cleanup with improved resource management."""
        if self._shutdown_initiated:
            return

        self._shutdown_initiated = True
        self._shutdown_event.set()

        try:
            # Shutdown resource manager first
            if hasattr(self, 'resource_manager'):
                self.resource_manager.shutdown()

            # Kill FFmpeg processes
            if hasattr(self, 'ffmpeg'):
                self.ffmpeg._kill_current_process()

            # Wait for active threads
            cleanup_timeout = 30
            start_time = time.time()
            while time.time() - start_time < cleanup_timeout:
                with self._threads_lock:
                    if not self._active_threads:
                        break
                time.sleep(0.1)

            # Clear locks and resources
            with self._file_locks_lock:
                self._file_locks.clear()

            # Run cleanup handlers
            for handler in self._cleanup_handlers:
                try:
                    handler()
                except Exception as e:
                    self.dev_logger.error(f"Cleanup handler failed: {str(e)}")

            # Clean temp files
            self._cleanup_temp_directory()

            # Clear caches
            if hasattr(self, '_file_cache'):
                self._file_cache.clear()
            if hasattr(self, 'dimension_cache'):
                self.dimension_cache.clear()
            if hasattr(self, 'palette_cache'):
                self.palette_cache.clear()

        except Exception as e:
            self.dev_logger.error(f"Error during cleanup: {str(e)}")

        self._stop_workers.set()

        # Wait for threads to finish
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=5)

        self.worker_threads.clear()

        # Clear queues
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

    def _cleanup_temp_directory(self):
        """Clean up temporary directory."""
        try:
            temp_dir = Path(TEMP_FILE_DIR)
            if (temp_dir.exists()):
                for temp_file in temp_dir.glob("*"):
                    try:
                        if (temp_file.is_file()):
                            temp_file.unlink(missing_ok=True)
                    except Exception as e:
                        self.dev_logger.error(
                            f"Failed to cleanup {temp_file}: {str(e)}")
        except Exception as e:
            self.dev_logger.error(f"Error cleaning temp directory: {str(e)}")

    def register_cleanup_handler(self, handler: Callable) -> None:
        """Register a cleanup handler to be called during resource cleanup."""
        self._cleanup_handlers.append(handler)

    def _log_with_lock(self, level: str, message: str, file_id: str = "") -> None:
        """Thread-safe logging with deduplication."""
        with self.logging_lock:
            log_key = f"{file_id}:{message}"
            if log_key not in self.processed_files:
                if level == "info":
                    self.dev_logger.info(message)
                elif level == "error":
                    self.dev_logger.error(message)
                elif level == "warning":
                    self.dev_logger.warning(message)
                elif level == "success":
                    self.user_logger.success(message)
                self.processed_files.add(log_key)

    @log_function_call
    def create_gif(self, file_path: Path, output_path: Path,
                   fps: int, dimensions: Tuple[int, int]) -> bool:
        """Create optimized GIF with better performance."""
        try:
            # First pass - Create palette optimized GIF
            temp_palette = Path(TEMP_FILE_DIR) / \
                f"palette_{file_path.stem}.png"
            temp_output = Path(TEMP_FILE_DIR) / f"temp_{output_path.stem}.gif"

            try:
                # Generate optimized palette first
                palette_cmd = [
                    'ffmpeg', '-i', str(file_path),
                    '-vf', f'fps={fps},scale={dimensions[0]}:{dimensions[1]}:flags=lanczos,palettegen=max_colors=128:stats_mode=diff',
                    '-y', str(temp_palette)
                ]

                if not run_ffmpeg_command(palette_cmd):
                    return False

                # Create initial GIF with palette
                gif_cmd = [
                    'ffmpeg', '-i', str(file_path),
                    '-i', str(temp_palette),
                    '-lavfi',
                    f'fps={fps},'
                    f'scale={dimensions[0]}:{dimensions[1]}:flags=lanczos[x];'
                    f'[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
                    '-y',
                    str(temp_output)
                ]

                # Add performance optimization flags
                gif_cmd.extend([
                    '-threads', '4',
                    '-preset', 'faster',
                    '-movflags', '+faststart'
                ])

                if run_ffmpeg_command(gif_cmd):
                    # Verify output
                    if temp_output.exists() and temp_output.stat().st_size > 0:
                        gif_size = self.get_file_size(
                            temp_output, force_refresh=True)
                        self.dev_logger.success(
                            f"[{fps}fps] Generated GIF ({gif_size:.2f}MB) -> Optimizing..."
                        )
                        # Copy to final output
                        shutil.copy2(temp_output, output_path)
                        return True
                return False

            finally:
                # Cleanup temp files
                for temp_file in [temp_palette, temp_output]:
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except Exception as e:
                        self.dev_logger.error(
                            f"Failed to cleanup {temp_file}: {e}")

        except Exception as e:
            self.dev_logger.error(f"GIF creation failed: {str(e)}")
            return False

    def process_single_fps(self, args: Tuple) -> ProcessingResult:
        """Process a single optimization pass with optional settings override.

        Args:
            file_path: Path to the GIF or video file to process.
            output_path: Path to the output GIF file.
            is_video: Whether the input file is a video.
            pass_index: Index of the pass to use from GIF_PASS_OVERS.
            override_settings: Optional dictionary of settings to override.

        Returns:
            ProcessingResult: Result of the optimization pass.
        """

        file_path, output_path, is_video, fps, current_settings = args
        temp_dir = Path(TEMP_FILE_DIR)
        file_id = f"{Path(file_path).stem}_{fps}"

        # Ensure temp directory exists
        temp_dir.mkdir(parents=True, exist_ok=True)

        # Single temporary file for the optimized GIF
        final_gif = temp_dir / f"{Path(output_path).stem}_{fps}_optimized.gif"

        try:
            # Register file for cleanup
            TempFileManager.register(final_gif)

            if self._processing_cancelled.is_set():
                return ProcessingResult(fps, float('inf'), None,
                                        ProcessingStatus.SIZE_THRESHOLD_EXCEEDED,
                                        "Processing cancelled")

            if is_video:
                # Get dimensions
                from video_optimization import VideoProcessor
                width, height = VideoProcessor()._get_dimensions(file_path)
                if not width or not height:
                    return ProcessingResult(fps, float('inf'), None,
                                            ProcessingStatus.DIMENSION_ERROR,
                                            f"Could not determine dimensions for {Path(file_path).name}")

                # Apply scale factor
                scale_factor = current_settings.get('scale_factor', 1.0)
                new_width = int(width * scale_factor)
                new_height = int(height * scale_factor)

                # Create optimized GIF in one step
                if not self.ffmpeg.create_optimized_gif(
                    file_path, final_gif, fps, (new_width,
                                                new_height), current_settings
                ):
                    return ProcessingResult(fps, float('inf'), None,
                                            ProcessingStatus.CONVERSION_ERROR,
                                            f"GIF creation failed for {Path(file_path).name}")

            # Get size after creation
            gif_size = self.get_file_size(final_gif, force_refresh=True)

            # Log progress
            self._log_with_lock("info",
                                f"Created GIF: {gif_size:.2f}MB at {fps}fps", file_id)

            return ProcessingResult(fps, gif_size, str(final_gif),
                                    ProcessingStatus.SUCCESS,
                                    f"Processed successfully - {gif_size:.2f}MB")

        except Exception as e:
            self._log_with_lock("error",
                                f"Error processing {Path(file_path).name}: {str(e)}",
                                file_id)
            return ProcessingResult(fps, float('inf'), None,
                                    ProcessingStatus.OPTIMIZATION_ERROR,
                                    str(e))
        finally:
            # Cleanup temp files
            if final_gif.exists():
                try:
                    final_gif.unlink()
                    TempFileManager.unregister(final_gif)
                except Exception as e:
                    self._log_with_lock("error",
                                        f"Failed to cleanup {final_gif}: {str(e)}",
                                        file_id)

    def performance_monitor(func: Callable) -> Callable:
        """Decorator to monitor function performance."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            process = psutil.Process()
            start_memory = process.memory_info().rss / 1024 / 1024

            try:
                result = func(*args, **kwargs)
                end_time = time.time()
                end_memory = process.memory_info().rss / 1024 / 1024

                logging.debug(
                    f"Function: {func.__name__} | "
                    f"Time: {end_time - start_time:.2f}s | "
                    f"Memory: {end_memory - start_memory:.2f}MB"
                )
                return result
            except Exception as e:
                logging.error(
                    f"Error in {func.__name__}: {str(e)}\n{traceback.format_exc()}")
                raise
        return wrapper

    def process_file(self, file_path: Path, output_path: Path, is_video: bool) -> None:
        """Process a single file with improved dimension detection."""
        file_lock = self.get_file_lock(str(file_path))

        try:
            # Add timeout to lock acquisition
            if not file_lock.acquire(timeout=5):
                self.dev_logger.error(
                    f"Failed to acquire lock for {file_path.name}")
                self.failed_files.append(file_path)
                return

            if self._shutdown_event.is_set():
                return

            # Check if file already meets size requirements
            file_size = self.get_file_size(file_path)
            target_size = self.compression_settings.get(
                'min_size_mb', 15.0)

            if file_size <= target_size:
                self.dev_logger.info(
                    f"File {file_path.name} already meets size requirements "
                    f"({file_size:.2f}MB <= {target_size}MB)")
                if not output_path.exists():
                    shutil.copy2(file_path, output_path)
                    self.dev_logger.info(
                        f"Copied original file to: {output_path}")
                return

            # Get dimensions first
            dimensions = self._get_dimensions_with_retry(file_path)

            # Add debug logging for dimension validation
            self.dev_logger.debug(
                f"Validating dimensions for {file_path.name}: {dimensions}")

            # Fix dimension validation
            if (not dimensions or
                len(dimensions) != 2 or
                    not all(isinstance(d, int) and d > 0 for d in dimensions)):
                self.dev_logger.error(
                    f"Could not determine valid dimensions for {file_path.name}")
                self.failed_files.append(file_path)
                return

            width, height = dimensions
            if width <= 0 or height <= 0:
                self.dev_logger.error(
                    f"Invalid dimensions for {file_path.name}: {width}x{height}")
                self.failed_files.append(file_path)
                return

            if is_video:
                self._process_video(file_path, output_path, dimensions)
            else:
                self._process_gif(file_path, output_path, dimensions)

        except Exception as e:
            self.dev_logger.error(
                f"Error processing {file_path.name}: {str(e)}")
            self.failed_files.append(file_path)
        finally:
            try:
                file_lock.release()
            except Exception:
                pass
            self._cleanup_file_resources(str(file_path))

    def _process_video(self, file_path: Path, output_path: Path, dimensions: Tuple[int, int]) -> None:
        """Process video with improved strategy for large files."""
        temp_files = []
        success = False
        best_size = float('inf')
        best_temp_file = None
        logged_attempts = set()  # Add this to track logged attempts

        try:
            target_size = self.compression_settings.get('min_size_mb', 10.0)
            initial_size = self.get_file_size(file_path)

            # More aggressive initial scale for large files
            size_ratio = target_size / initial_size
            if initial_size > 90:  # Very large files
                scale_factor = min(1.0, max(0.3, (size_ratio ** 0.5) * 0.7))
                current_colors = 192  # Start with fewer colors
            else:
                scale_factor = min(1.0, max(0.4, (size_ratio ** 0.4)))
                current_colors = 256

            self.dev_logger.info(
                f"\n{'='*50}\n"
                f"Starting processing for: {file_path.name}\n"
                f"Initial size: {initial_size:.2f}MB\n"
                f"Target size: {target_size:.2f}MB\n"
                f"Dimensions: {dimensions[0]}x{dimensions[1]}\n"
                f"{'='*50}"
            )

            attempt = 0
            max_attempts = 8
            current_size = initial_size
            fps = self.compression_settings["fps_range"][0]
            prev_settings = set()  # Track used settings

            while attempt < max_attempts and current_size > target_size:
                attempt_num = attempt + 1

                # Only log attempt header if we haven't seen this attempt number
                if attempt_num not in logged_attempts:
                    self.dev_logger.info(
                        f"\n{'-'*50}\n"
                        f"Optimization Attempt {attempt_num}/{max_attempts}\n"
                        f"{'-'*50}"
                    )
                    logged_attempts.add(attempt_num)

                # Create unique temp file
                timestamp = int(time.time() * 1000)
                current_temp_file = Path(
                    TEMP_FILE_DIR) / f"temp_{timestamp}.gif"
                temp_files.append(current_temp_file)

                # Calculate dimensions
                new_width = int(dimensions[0] * scale_factor // 2 * 2)
                new_height = int(dimensions[1] * scale_factor // 2 * 2)

                # Calculate lossy value based on current size ratio
                current_ratio = current_size / target_size
                lossy_value = min(100, int(40 * current_ratio))

                # Skip if we've already tried these settings
                settings_key = (scale_factor, current_colors, lossy_value)
                if settings_key in prev_settings:
                    # Force more aggressive changes
                    scale_factor *= 0.7
                    current_colors = max(128, current_colors - 32)
                    continue

                prev_settings.add(settings_key)

                # Log settings only once per unique configuration
                self.dev_logger.info(
                    f"Settings:\n"
                    f"- Scale: {scale_factor:.3f}\n"
                    f"- Size: {new_width}x{new_height}\n"
                    f"- Colors: {current_colors}\n"
                    f"- Lossy: {lossy_value}"
                )

                # Optimized FFmpeg command with better compression
                combined_cmd = [
                    'ffmpeg', '-i', str(file_path),
                    '-vf', (f'fps={fps},'
                            f'scale={new_width}:{new_height}:flags=lanczos,'
                            f'split[x][y];[x]palettegen=max_colors={current_colors}:reserve_transparent=0[p];'
                            f'[y][p]paletteuse=dither=sierra2_4a:diff_mode=rectangle'),
                    '-y', str(current_temp_file)
                ]

                if not run_ffmpeg_command(combined_cmd):
                    self.dev_logger.error("Conversion failed")
                    break

                # Optimize in place with gifsicle
                optimize_cmd = [
                    'gifsicle',
                    '--optimize=3',
                    '--colors', str(current_colors),
                    '--lossy=' + str(lossy_value),
                    '--no-conserve-memory',
                    '--careful',
                    str(current_temp_file),
                    '--output', str(current_temp_file)
                ]

                run_ffmpeg_command(optimize_cmd)
                current_size = self.get_file_size(current_temp_file)

                # Track best result
                if current_size < best_size:
                    best_size = current_size
                    if best_temp_file:
                        try:
                            best_temp_file.unlink()
                        except Exception:
                            pass
                    best_temp_file = current_temp_file

                reduction = ((initial_size - current_size) /
                             initial_size) * 100
                self.dev_logger.info(
                    f"Result: {current_size:.2f}MB ({reduction:.1f}% reduction)"
                )

                if current_size <= target_size:
                    self.dev_logger.info(
                        f"\nTarget size achieved: {current_size:.2f}MB")
                    shutil.copy2(current_temp_file, output_path)
                    success = True
                    break

                # Smart parameter adjustment based on results
                if attempt < max_attempts - 1:
                    size_ratio = current_size / target_size

                    if size_ratio > 3:
                        # Very far from target, be aggressive
                        scale_factor *= 0.6
                        current_colors = max(128, current_colors - 64)
                    elif size_ratio > 2:
                        # Still too large
                        scale_factor *= 0.7
                        current_colors = max(128, current_colors - 32)
                    elif size_ratio > 1.5:
                        # Getting closer
                        scale_factor *= 0.8
                        if current_colors > 192:
                            current_colors = 192
                    else:
                        # Fine-tuning
                        scale_factor *= 0.9

                    # Ensure we don't get stuck
                    if scale_factor > 0.3 and attempt >= 2:
                        scale_factor = min(scale_factor, 0.3)

                    # Prevent going too low
                    scale_factor = max(0.25, scale_factor)
                    current_colors = max(64, current_colors)

                attempt += 1

            # If we didn't reach target but have a best result
            if not success and best_temp_file and best_size < initial_size:
                shutil.copy2(best_temp_file, output_path)
                self.dev_logger.info(
                    f"\nUsing best achieved result: {best_size:.2f}MB"
                )

            final_size = self.get_file_size(output_path)
            self.dev_logger.info(
                f"\nFinal Results:\n"
                f"Initial: {initial_size:.2f}MB\n"
                f"Final: {final_size:.2f}MB\n"
                f"Reduction: {((initial_size - final_size) / initial_size) * 100:.1f}%"
            )

        except Exception as e:
            self.dev_logger.error(f"Error: {str(e)}")
            self.failed_files.append(file_path)
        finally:
            for temp_file in temp_files:
                try:
                    if temp_file.exists() and temp_file != best_temp_file:
                        temp_file.unlink()
                except Exception as e:
                    self.dev_logger.error(f"Cleanup error: {e}")
            if best_temp_file and best_temp_file.exists():
                try:
                    best_temp_file.unlink()
                except Exception:
                    pass

    # ...rest of the code...

    @log_function_call
    def _compress_large_gif(self, gif_path: Path) -> None:
        """Additional compression for large GIFs."""
        temp_path = None
        try:
            timestamp = int(time.time() * 1000)
            temp_path = gif_path.with_name(
                f"temp_{timestamp}_{gif_path.stem}.gif")

            # More conservative optimization settings for better quality
            cmd = [
                'gifsicle',
                '--optimize=3',
                '--colors', '128',     # Keep more colors for quality
                '--lossy=80',          # Less aggressive lossy compression
                '--scale', '0.8',      # Less aggressive scaling
                '--no-conserve-memory',
                '--careful',           # More careful optimization
                '--threads=4',
                str(gif_path),
                '-o', str(temp_path)
            ]

            if run_ffmpeg_command(cmd) and temp_path.exists():
                original_size = self.get_file_size(gif_path)
                compressed_size = self.get_file_size(temp_path)

                if compressed_size < original_size:
                    try:
                        gif_path.unlink()
                        temp_path.replace(gif_path)
                        logger.success(
                            f"Additional compression succeeded: {original_size:.2f}MB -> {compressed_size:.2f}MB"
                        )
                    except Exception as e:
                        logger.error(f"Failed to replace file: {e}")
                else:
                    logger.skip(
                        f"Additional compression skipped - no size reduction achieved"
                    )

        except Exception as e:
            logger.error(f"Additional compression failed: {str(e)}")
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except Exception as e:
                    logger.error(f"Failed to cleanup temp file: {e}")

    def _ensure_worker_threads(self):
        """Ensure worker threads are running."""
        if not self.worker_threads:
            for _ in range(self.max_threads):
                thread = threading.Thread(
                    target=self._worker_thread, daemon=True)
                thread.start()
                self.worker_threads.append(thread)

    def _worker_thread(self):
        """Worker thread to process tasks from queue."""
        while not self._stop_workers.is_set():
            try:
                task = self.task_queue.get(timeout=1)
                if task:
                    file_path, temp_gif, fps, dimensions, config, target_size = task
                    result = self._process_single_config(
                        file_path, temp_gif, fps,
                        dimensions[0], dimensions[1],
                        config, target_size
                    )
                    self.result_queue.put(result)
                self.task_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                self.dev_logger.error(f"Worker thread error: {e}")

    def _process_single_config(self, file_path: Path, temp_gif: Path, fps: int,
                               width: int, height: int, settings: Dict, target_size: float) -> Optional[Dict]:
        """Process a single optimization configuration."""
        try:
            # Create initial GIF
            if self.ffmpeg.create_optimized_gif(
                file_path,
                temp_gif,
                fps,
                (int(width * settings['scale_factor']),
                 int(height * settings['scale_factor'])),
                settings
            ):
                size = self.get_file_size(temp_gif)

                # Only proceed with gifsicle if initial size is promising
                if size < min(90, target_size * 2):
                    optimized_gif = Path(TEMP_FILE_DIR) / \
                        f"{temp_gif.stem}_opt.gif"

                    cmd = [
                        'gifsicle',
                        '--optimize=3',
                        '--colors', str(settings['colors']),
                        '--lossy=' + str(settings['lossy_value']),
                        '--no-conserve-memory',
                        '--threads=2',
                        str(temp_gif),
                        '-o', str(optimized_gif)
                    ]

                    if run_ffmpeg_command(cmd) and optimized_gif.exists():
                        final_size = self.get_file_size(optimized_gif)
                        return {
                            'success': True,
                            'fps': fps,
                            'size': final_size,
                            'path': str(optimized_gif),
                            'settings': settings
                        }

        except Exception as e:
            self.dev_logger.error(f"Configuration processing error: {str(e)}")

        return None

    def _cleanup_temp_files(self, temp_files: list[Path]) -> None:
        """Clean up temporary files with improved error handling."""
        for temp_file in temp_files:
            try:
                if temp_file and temp_file.exists():
                    temp_file.unlink()
                TempFileManager.unregister(temp_file)
            except Exception as e:
                self.dev_logger.error(
                    f"Failed to cleanup {temp_file}: {str(e)}")

    def _cleanup_file_resources(self, file_path: str) -> None:
        """Clean up resources associated with a file."""
        with self._file_locks_lock:
            if file_path in self._file_locks:
                del self._file_locks[file_path]

        # Clear process queue
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break

        # Force garbage collection
        gc.collect()

    def _wrapped_process_single_fps(self, args: Tuple) -> ProcessingResult:
        """Wrapper for process_single_fps with thread registration."""
        self._register_thread()
        try:
            return self.process_single_fps(args)
        finally:
            self._unregister_thread()

    @performance_monitor
    def process_all(self) -> List[Path]:
        """Process all GIF files in input directory."""
        self.failed_files = []
        try:
            # Get all input files and sort by size
            input_files = []

            # Collect video files
            for video_format in SUPPORTED_VIDEO_FORMATS:
                input_files.extend(Path(INPUT_DIR).glob(f'*{video_format}'))

            # Collect GIF files
            input_files.extend(Path(INPUT_DIR).glob('*.gif'))

            # Sort files by size (smallest first)
            sorted_files = sorted(
                input_files,
                key=lambda f: f.stat().st_size
            )

            # Process sorted files
            for file_path in sorted_files:
                if self._should_exit():
                    self.dev_logger.info("Gracefully stopping processing...")
                    break

                output_gif = Path(OUTPUT_DIR) / f"{file_path.stem}.gif"
                if not output_gif.exists():
                    is_video = file_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS
                    self.process_file(file_path, output_gif, is_video=is_video)

        except KeyboardInterrupt:
            self.dev_logger.warning("Processing interrupted by user")
            self._shutdown_event.set()
        except Exception as e:
            self.dev_logger.error(f"Error in process_all: {str(e)}")
        finally:
            if self._should_exit():
                self.dev_logger.info("Cleaning up after graceful exit...")
            self.cleanup_resources()

        return self.failed_files

    def _process_gif(self, file_path: Path, output_path: Path, dimensions: Tuple[int, int]) -> None:
        """Process GIF with graceful exit support."""
        try:
            if self._should_exit():
                return

            file_size = self.get_file_size(file_path)
            if file_size > GIF_SIZE_TO_SKIP:
                self.dev_logger.warning(
                    f"Skipping {file_path.name} - {file_size:.1f}MB exceeds size limit")
                return None

            max_attempts = 3  # Add max_attempts definition
            for attempt in range(max_attempts):
                if self._should_exit():
                    self.dev_logger.info(
                        "Gracefully stopping GIF optimization...")
                    return

                # Try progressive optimization first
                if self.progressive_optimization:
                    try:
                        success, final_size = self.dynamic_optimizer.optimize_gif(
                            file_path, output_path, self.compression_settings['min_size_mb'])
                        if success:
                            self.dev_logger.info(
                                f"Progressive optimization succeeded: {final_size:.2f}MB")
                            return file_path
                    except Exception as e:
                        self.dev_logger.warning(
                            f"Progressive optimization failed: {e}")

                # Fall back to standard optimization
                try:
                    current_settings = self._get_optimized_configs(
                        file_size,
                        self.compression_settings['min_size_mb'],
                        30,  # Default FPS for GIFs
                        dimensions
                    )[0]  # Use best quality settings first

                    result = self._process_single_config(
                        file_path,
                        output_path,
                        self.compression_settings['fps_range'][0],
                        dimensions[0],
                        dimensions[1],
                        current_settings,
                        self.compression_settings['min_size_mb']
                    )

                    if result and result.get('success'):
                        return file_path

                except Exception as e:
                    self.dev_logger.error(
                        f"Standard optimization attempt {attempt + 1} failed: {e}")
                    if attempt == max_attempts - 1:
                        raise

                time.sleep(1)  # Brief pause between attempts

            return None

        except Exception as e:
            self.dev_logger.error(f"Error processing {file_path}: {str(e)}")
            return None

    def _update_progress(self, file_id: str, status: str) -> None:
        """Update processing progress for a file."""
        with self._progress_lock:
            self._processing_progress[file_id] = {
                'status': status,
                'timestamp': time.time()
            }

    def _check_timeout(self, file_id: str) -> bool:
        """Check if processing has timed out."""
        with self._progress_lock:
            if file_id in self._processing_progress:
                elapsed = time.time() - \
                    self._processing_progress[file_id]['timestamp']
                return elapsed > self._processing_timeout
        return False

    def _signal_handler(self, signum, frame):
        """Enhanced signal handler with proper cleanup."""
        if self._shutdown_initiated:
            return

        self.dev_logger.warning("\nReceived interrupt signal. Cleaning up...")
        try:
            self.cleanup_resources()
        finally:
            sys.exit(0)

    def _immediate_shutdown_handler(self, signum, frame):
        """Handle immediate shutdown when signal is received."""
        if self._shutdown_initiated:
            return

        self.dev_logger.warning(
            "\nReceived termination signal. Stopping all processes...")
        self._immediate_termination.set()
        self._shutdown_event.set()
        self._processing_cancelled.set()

        # Kill any running FFmpeg processes immediately
        self.ffmpeg._kill_current_process()

        # Clear queues immediately
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

        # Stop all worker threads
        self._stop_workers.set()

        # Cleanup resources
        self.cleanup_resources()
        sys.exit(0)

    def _should_exit(self) -> bool:
        """Enhanced exit check that includes immediate termination."""
        return (self._shutdown_event.is_set() or
                self._processing_cancelled.set() or
                self._shutdown_initiated or
                self._immediate_termination.is_set())

    def _process_single_config_with_retry(self, *args, **kwargs) -> Optional[dict]:
        """Process single configuration with retry logic."""
        for attempt in range(self.retry_count):
            try:
                return self._process_single_config(*args, **kwargs)
            except Exception as e:
                if attempt < self.retry_count - 1:
                    with self._stats_lock:
                        self._stats['retried'] += 1
                    time.sleep(self.retry_delay * (attempt + 1))
                else:
                    raise

    def _calculate_optimal_batch_size(self) -> int:
        """Calculate optimal batch size based on available memory."""
        available_memory = psutil.virtual_memory().available / (1024 * 1024)  # MB
        # Assume 500MB per process
        return max(1, min(10, int(available_memory / 500)))

    def _generate_optimization_configs(self, input_size: float, target_size: float) -> List[Dict]:
        """Generate optimized configuration list."""
        ratio = target_size / input_size
        configs = []

        # Dynamic scale factors based on size ratio
        scale_factors = [
            min(1.0, max(0.25, ratio ** 0.5)),
            min(0.75, max(0.2, ratio ** 0.6)),
            min(0.5, max(0.15, ratio ** 0.7)),
            min(0.25, max(0.1, ratio ** 0.8))
        ]

        for scale in scale_factors:
            for colors in [256, 192, 128, 64]:
                configs.append({
                    'scale_factor': scale,
                    'colors': colors,
                    'lossy_value': min(100, int(30 * (1/scale))),
                    # Priority for processing order
                    'priority': scale * (colors / 256)
                })

        # Sort by priority
        return sorted(configs, key=lambda x: x['priority'], reverse=True)

    def _check_early_exit(self, results: List[Dict]) -> bool:
        """Check if we can exit early with good enough results."""
        if not results:
            return False

        best_result = max(results, key=lambda x: (x['fps'], -x['size']))
        target_size = self.compression_settings.get('min_size_mb', 15.0)

        return (best_result['size'] <= target_size and
                best_result['fps'] >= self.compression_settings['fps_range'][0] + 2)

    def _get_dimensions_with_retry(self, file_path: Path, max_retries: int = 3) -> Tuple[Optional[int], Optional[int]]:
        """Get video dimensions with improved error handling and validation."""
        last_dimensions = None
        last_error = None
        file_path = Path(file_path)

        if not file_path.exists():
            self.dev_logger.error(f"File not found: {file_path}")
            return None, None

        # Try to process file even if it's being written
        if not self.wait_for_file_completion(file_path, timeout=10):
            self.dev_logger.warning(f"File may still be writing: {file_path}")

        methods = [
            (self._get_dimensions_ffprobe, "FFprobe"),
            (self._get_dimensions_opencv, "OpenCV"),
            (self._get_dimensions_ffmpeg, "FFmpeg")
        ]

        for method, method_name in methods:
            for attempt in range(max_retries):
                try:
                    dimensions = method(file_path)

                    # Store last valid dimensions as backup
                    if dimensions and len(dimensions) == 2 and all(isinstance(d, int) for d in dimensions):
                        last_dimensions = dimensions

                    if self._validate_dimensions(dimensions):
                        self.dev_logger.debug(
                            f"Got valid dimensions using {method_name}: {dimensions}")
                        return dimensions

                except Exception as e:
                    last_error = str(e)
                    if attempt == max_retries - 1:
                        self.dev_logger.debug(
                            f"{method_name} failed after {max_retries} attempts: {str(e)}")

                if attempt < max_retries - 1:
                    time.sleep(0.5 * (2 ** attempt))

        # If we have last_dimensions, try to use them as fallback
        if last_dimensions:
            if self._validate_dimensions(last_dimensions, strict=False):
                self.dev_logger.warning(
                    f"Using fallback dimensions: {last_dimensions}")
                return last_dimensions

        self.dev_logger.error(
            f"All dimension detection methods failed for {file_path}. Last error: {last_error}")
        return None, None

    def _validate_dimensions(self, dimensions: Tuple[Optional[int], Optional[int]], strict: bool = True) -> bool:
        """Validate dimensions with configurable strictness."""
        try:
            if not dimensions or len(dimensions) != 2:
                return False

            width, height = dimensions
            if not isinstance(width, int) or not isinstance(height, int):
                return False

            if width <= 0 or height <= 0:
                return False

            # Strict validation includes aspect ratio and size limits
            if strict:
                # Max 8K resolution
                if width > 7680 or height > 4320:
                    return False

                # Minimum dimensions (adjust as needed)
                if width < 16 or height < 16:
                    return False

                # Reasonable aspect ratio (between 1:10 and 10:1)
                aspect_ratio = width / height
                if aspect_ratio < 0.1 or aspect_ratio > 10:
                    return False

            return True

        except Exception as e:
            self.dev_logger.debug(f"Dimension validation error: {str(e)}")
            return False

    def _get_dimensions_ffprobe(self, file_path: Path) -> Tuple[int, int]:
        """Get dimensions using ffprobe with improved reliability."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,rotation,sample_aspect_ratio,display_aspect_ratio',
                '-of', 'json',
                str(file_path)
            ]

            output = subprocess.check_output(
                cmd, stderr=subprocess.PIPE, text=True)
            data = json.loads(output)

            if not data.get('streams'):
                raise ValueError("No video streams found")

            stream = data['streams'][0]

            # Get base dimensions
            width = int(stream.get('width', 0))
            height = int(stream.get('height', 0))

            if width <= 0 or height <= 0:
                raise ValueError("Invalid dimensions in stream")

            # Handle rotation
            rotation = int(stream.get('rotation', '0') or '0')
            if rotation in [90, 270]:
                width, height = height, width

            # Handle pixel aspect ratio correction
            sar = stream.get('sample_aspect_ratio', '1:1')
            if sar and sar != '1:1':
                try:
                    num, den = map(int, sar.split(':'))
                    if num > 0 and den > 0:
                        width = int(width * (num / den))
                except:
                    pass

            return width, height

        except json.JSONDecodeError as e:
            raise ValueError(f"FFprobe JSON error: {e}")
        except subprocess.CalledProcessError as e:
            raise ValueError(f"FFprobe process error: {e}")
        except Exception as e:
            raise ValueError(f"FFprobe error: {str(e)}")

    def _get_dimensions_opencv(self, file_path: Path) -> Tuple[int, int]:
        """Get dimensions using OpenCV with proper resource management."""
        cap = None
        try:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                raise ValueError("Failed to open video file")

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if width <= 0 or height <= 0:
                # Try reading first frame
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                else:
                    raise ValueError("Failed to read frame dimensions")

            # Verify dimensions are valid
            if width <= 0 or height <= 0:
                raise ValueError(f"Invalid dimensions: {width}x{height}")

            return (width, height)

        except Exception as e:
            raise ValueError(f"OpenCV failed: {str(e)}")
        finally:
            if cap is not None:
                cap.release()

    def _get_dimensions_ffmpeg(self, file_path: Path) -> Tuple[int, int]:
        """Get dimensions using FFmpeg as last resort."""
        try:
            cmd = ['ffmpeg', '-i', str(file_path)]
            output = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            ).stderr

            # Try multiple regex patterns
            patterns = [
                r'Stream.*Video.* (\d+)x(\d+)',
                r'Video: .* (\d+)x(\d+)',
                r', (\d+)x(\d+)[,\s]'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, output)
                if matches:
                    width, height = map(int, matches[0])
                    if width > 0 and height > 0:
                        return (width, height)

            raise ValueError("No valid dimensions found in FFmpeg output")
        except Exception as e:
            raise ValueError(f"FFmpeg dimension detection failed: {str(e)}")

    def _should_exit(self) -> bool:
        """Check if processing should stop."""
        return (self._shutdown_event.is_set() or
                self._processing_cancelled.is_set() or
                self._shutdown_initiated)

    @log_function_call
    def create_gif(self, file_path: Path, output_path: Path,
                   fps: int, dimensions: Tuple[int, int]) -> bool:
        """Create optimized GIF with enhanced compression."""
        try:
            # Initial size check
            original_size = self.get_file_size(file_path)
            if original_size > 1000:  # If source is larger than 1GB
                logger.error(
                    f"Source file too large: {original_size:.2f}MB")
                return False

            # Calculate optimal dimensions while maintaining aspect ratio
            max_dimension = max(dimensions[0], dimensions[1])
            scale_factor = 1.0
            if max_dimension > 1280:  # Limit max dimension to 1280px
                scale_factor = 1280 / max_dimension

            new_width = int(dimensions[0] * scale_factor // 2 * 2)
            new_height = int(dimensions[1] * scale_factor // 2 * 2)

            # First pass - Create palette optimized GIF
            temp_palette = Path(TEMP_FILE_DIR) / \
                f"palette_{file_path.stem}.png"
            temp_output = Path(TEMP_FILE_DIR) / f"temp_{output_path.stem}.gif"

            try:
                # Generate optimized palette
                palette_cmd = [
                    'ffmpeg', '-i', str(file_path),
                    '-vf', f'fps={fps},scale={new_width}:{new_height}:flags=lanczos,palettegen=max_colors=128:stats_mode=diff',
                    '-y', str(temp_palette)
                ]

                if not run_ffmpeg_command(palette_cmd):
                    return False

                # Create initial GIF with palette
                gif_cmd = [
                    'ffmpeg', '-i', str(file_path),
                    '-i', str(temp_palette),
                    '-lavfi', f'fps={fps},scale={new_width}:{new_height}:flags=lanczos [x];[x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
                    '-y', str(temp_output)
                ]

                if not run_ffmpeg_command(gif_cmd):
                    return False

                # Second pass - Optimize with gifsicle
                if temp_output.exists():
                    gifsicle_cmd = [
                        'gifsicle',
                        '--optimize=3',
                        '--colors', '128',
                        '--lossy=80',
                        '--no-conserve-memory',
                        str(temp_output),
                        '-o', str(output_path)
                    ]

                    if not run_ffmpeg_command(gifsicle_cmd):
                        return False

                    # Verify final size
                    final_size = self.get_file_size(output_path)
                    if final_size > 100:  # If still too large, try extreme compression
                        self._compress_large_gif(output_path)
                        final_size = self.get_file_size(output_path)

                    logger.success(
                        f"Generated GIF: Original={original_size:.2f}MB, Final={final_size:.2f}MB"
                    )
                    return True

                return False

            finally:
                # Cleanup temp files
                for temp_file in [temp_palette, temp_output]:
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                    except Exception as e:
                        logger.error(
                            f"Failed to cleanup {temp_file}: {e}")

        except Exception as e:
            logger.error(f"GIF creation failed: {str(e)}")
            return False

    @log_function_call
    def _compress_large_gif(self, gif_path: Path) -> None:
        """Additional compression for large GIFs."""
        try:
            temp_path = gif_path.with_name(f"temp_{gif_path.name}")

            # Extreme optimization settings
            cmd = [
                'gifsicle',
                '--optimize=3',
                '--colors', '64',        # Reduce colors
                '--lossy=100',           # Maximum compression
                '--scale', '0.7',        # Reduce dimensions
                '--no-conserve-memory',
                str(gif_path),
                '-o', str(temp_path)
            ]

            if run_ffmpeg_command(cmd) and temp_path.exists():
                # Replace original with compressed version
                temp_path.replace(gif_path)
            elif temp_path.exists():
                temp_path.unlink()

        except Exception as e:
            logger.error(f"Additional compression failed: {str(e)}")
            if temp_path.exists():
                temp_path.unlink()

    def cleanup_resources(self) -> None:
        """Enhanced cleanup with forceful file removal."""
        if self._shutdown_initiated:
            return

        self._shutdown_initiated = True
        self._shutdown_event.set()

        try:
            # Stop all processing first
            self._stop_workers.set()
            for thread in self.worker_threads:
                if thread.is_alive():
                    thread.join(timeout=5)

            # Kill any FFmpeg processes
            self._kill_ffmpeg_processes()

            # Forceful cleanup of temp files
            self._force_cleanup_temp_files()

        except Exception as e:
            self.dev_logger.error(f"Cleanup error: {e}")
            # Emergency cleanup as last resort
            self._emergency_cleanup()

    def _kill_ffmpeg_processes(self) -> None:
        """Kill all FFmpeg processes."""
        try:
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/F', '/IM', 'ffmpeg.exe'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                subprocess.run(['taskkill', '/F', '/IM', 'gifsicle.exe'],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                os.system("pkill -9 ffmpeg")
                os.system("pkill -9 gifsicle")
        except Exception as e:
            self.dev_logger.error(f"Failed to kill processes: {e}")

    def _force_cleanup_temp_files(self) -> None:
        """Force cleanup of temporary files and directory."""
        temp_dir = Path(TEMP_FILE_DIR)
        if not temp_dir.exists():
            return

        # Multiple attempts with process killing
        for attempt in range(3):
            try:
                for temp_file in temp_dir.glob("*"):
                    try:
                        if temp_file.is_file():
                            temp_file.unlink(missing_ok=True)
                    except PermissionError:
                        self._kill_ffmpeg_processes()
                        time.sleep(1)
                        temp_file.unlink(missing_ok=True)

                # Try to remove temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)
                break
            except Exception as e:
                self.dev_logger.error(
                    f"Cleanup attempt {attempt + 1} failed: {e}")
                if attempt == 2:  # Last attempt
                    self._emergency_cleanup()

    def _emergency_cleanup(self) -> None:
        """Emergency cleanup as last resort."""
        self.dev_logger.warning("Performing emergency cleanup...")
        try:
            # Kill all related processes
            self._kill_ffmpeg_processes()
            time.sleep(2)  # Wait for processes to die

            # Force delete temp directory
            temp_dir = Path(TEMP_FILE_DIR)
            if temp_dir.exists():
                if sys.platform == 'win32':
                    os.system(f'rmdir /S /Q "{temp_dir}"')
                else:
                    os.system(f'rm -rf "{temp_dir}"')
        except Exception as e:
            self.dev_logger.error(f"Emergency cleanup failed: {e}")

    def _get_video_frame_info(self, file_path: Path) -> Optional[Tuple[int, float]]:
        """Get total frames and duration of video.

        Args:
            file_path: Path to the video file

        Returns:
            Tuple[int, float]: (total_frames, duration_in_seconds) or None if failed
        """
        try:
            cmd = [
                'ffprobe', '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=nb_frames,duration',
                '-of', 'json',
                str(file_path)
            ]

            output = subprocess.check_output(cmd, text=True)
            data = json.loads(output)

            if 'streams' in data and data['streams']:
                stream = data['streams'][0]
                frames = int(stream.get('nb_frames', 0))
                duration = float(stream.get('duration', 0))

                if frames == 0 or duration == 0:  # If ffprobe couldn't get frame count, estimate it
                    cmd = [
                        'ffmpeg', '-i', str(file_path), '-map', '0:v:0', '-c', 'copy', '-f', 'null', '-']
                    output = subprocess.run(
                        cmd, capture_output=True, text=True).stderr
                    matches = re.search(r'frame=\s*(\d+)', output)
                    if matches:
                        frames = int(matches.group(1))
                        # Verify frame count is reasonable
                        if frames > 0 and frames < 1000000:  # Sanity check
                            return frames, duration
                        else:
                            logging.warning(
                                f"Suspicious frame count: {frames}, falling back to duration-based estimate")
                            # Estimate frames from duration and common framerates
                            for fps in [30, 25, 24, 60]:
                                estimated = int(duration * fps)
                                if abs(estimated - frames) < frames * 0.1:  # Within 10%
                                    return estimated, duration

                if frames > 0 and duration > 0:
                    return frames, duration

            self.dev_logger.warning(
                f"Could not get accurate frame info for {file_path}")
            return None

        except Exception as e:
            self.dev_logger.error(f"Error getting frame info: {str(e)}")
            return None

    def _get_optimized_configs(self, input_size: float, target_size: float,
                               avg_fps: float, dimensions: Tuple[int, int]) -> List[Dict]:
        """Generate optimized configurations based on input characteristics.

        Args:
            input_size: Original file size in MB
            target_size: Target file size in MB
            avg_fps: Average FPS of the source
            dimensions: (width, height) of the source

        Returns:
            List[Dict]: List of optimization configurations sorted by priority
        """
        configs = []
        size_ratio = target_size / input_size
        max_dimension = max(dimensions)

        # Base scale factor on both size ratio and resolution
        if max_dimension > 1920:
            scale_factors = [0.75, 0.5, 0.35]
        elif max_dimension > 1280:
            scale_factors = [0.85, 0.75, 0.5]
        else:
            scale_factors = [1.0, 0.85, 0.75]

        # Keep high color count initially
        color_configs = [256]
        if size_ratio < 0.3:  # Only add lower color options for very large files
            color_configs.extend([192, 128])

        # Create configs prioritizing quality
        for scale in scale_factors:
            for colors in color_configs:
                # Increased base lossy value
                lossy_value = min(100, int(40 * (1/scale)))
                configs.append({
                    'scale_factor': scale,
                    'colors': colors,
                    'lossy_value': lossy_value,
                    'priority': scale * (colors / 256)
                })

        return sorted(configs, key=lambda x: x['priority'], reverse=True)

    def _check_early_exit(self, results: List[Dict]) -> bool:
        """Check if we can exit early with good enough results."""
        if not results:
            return False

        best_result = max(results, key=lambda x: (x['fps'], -x['size']))
        target_size = self.compression_settings.get('min_size_mb', 15.0)

        return (best_result['size'] <= target_size and
                best_result['fps'] >= self.compression_settings['fps_range'][0] + 2)

    async def process_file_async(self, file_path: Path, output_path: Path, is_video: bool) -> None:
        """Async version of process_file with better resource management."""
        async with managed_process_pool() as pool:
            try:
                # Get file lock with timeout
                file_lock = self.get_file_lock(str(file_path))
                if not await asyncio.get_event_loop().run_in_executor(None, file_lock.acquire, True, 5):
                    self.dev_logger.error(
                        f"Failed to acquire lock for {file_path.name}")
                    self.failed_files.append(file_path)
                    return

                # Check file size and target
                file_size = await asyncio.get_event_loop().run_in_executor(None, self.get_file_size, file_path)
                target_size = self.compression_settings.get(
                    'min_size_mb', 15.0)

                if file_size <= target_size:
                    if not output_path.exists():
                        await asyncio.get_event_loop().run_in_executor(None, shutil.copy2, file_path, output_path)
                    return

                # Get dimensions
                dimensions = await asyncio.get_event_loop().run_in_executor(None, self._get_dimensions_with_retry, file_path)
                if not self._validate_dimensions(dimensions):
                    self.failed_files.append(file_path)
                    return

                # Process based on type
                if is_video:
                    await self._process_video_async(file_path, output_path, dimensions)
                else:
                    await self._process_gif_async(file_path, output_path, dimensions)

            except Exception as e:
                self.dev_logger.error(f"Async processing error: {str(e)}")
                self.failed_files.append(file_path)
            finally:
                if file_lock:
                    file_lock.release()

    def optimize_gif(self, input_path: Path, output_path: Path, target_size_mb: float = 15.0) -> Tuple[float, bool]:
        """Improved GIF optimization with progressive quality reduction."""
        # First try progressive optimization if enabled
        if self.progressive_optimization:
            try:
                # Get source FPS to limit maximum target FPS
                source_fps = self._get_source_fps(input_path)
                self.dev_logger.info(f"Source FPS: {source_fps}")

                # Adjust FPS range if needed
                fps_range = self.compression_settings['fps_range']
                if fps_range[1] > source_fps:
                    adjusted_range = (min(fps_range[0], source_fps), min(
                        fps_range[1], source_fps))
                    self.dev_logger.info(
                        f"Adjusting FPS range to match source: {adjusted_range}")
                    self.compression_settings['fps_range'] = adjusted_range

                result = self._progressive_optimize(
                    input_path, output_path, target_size_mb)
                if result.success:
                    return result.size, True
            except Exception as e:
                self.dev_logger.warning(
                    f"Progressive optimization failed: {e}")

        # Fall back to standard optimization
        try:
            file_size = self.get_file_size(input_path)
            dimensions = self._get_dimensions_with_retry(input_path)

            if not dimensions or not self._validate_dimensions(dimensions):
                return file_size, False

            # Get source colors
            source_colors = self._analyze_source_colors(input_path)

            # Start with high quality settings
            current_settings = self._get_optimized_configs(
                file_size,
                target_size_mb,
                self._get_source_fps(input_path),
                dimensions,
                source_colors
            )[0]

            result = self._process_single_config(
                input_path,
                output_path,
                min(self.compression_settings['fps_range']
                    [0], self._get_source_fps(input_path)),
                dimensions[0],
                dimensions[1],
                current_settings,
                target_size_mb
            )

            if result and result.get('success'):
                return result['size'], result['size'] <= target_size_mb

            return file_size, False

        except Exception as e:
            self.dev_logger.error(f"Standard optimization failed: {str(e)}")
            return file_size, False

    def _get_optimized_configs(self, input_size: float, target_size: float,
                               source_fps: float, dimensions: Tuple[int, int],
                               source_colors: int) -> List[Dict]:
        """Generate optimized configurations based on input characteristics."""
        configs = []
        size_ratio = target_size / input_size
        max_dimension = max(dimensions)

        # Scale factors based on dimensions and ratio
        if max_dimension > 1920:
            scale_factors = [0.75, 0.5, 0.35]
        elif max_dimension > 1280:
            scale_factors = [0.85, 0.75, 0.5]
        else:
            scale_factors = [1.0, 0.85, 0.75]

        # Use source colors as maximum
        color_configs = [min(256, source_colors)]
        if size_ratio < 0.3:
            color_configs.extend([
                min(192, int(source_colors * 0.75)),
                min(128, int(source_colors * 0.5))
            ])

        # Create configs prioritizing quality
        for scale in scale_factors:
            for colors in color_configs:
                configs.append({
                    'scale_factor': scale,
                    'colors': colors,
                    'lossy_value': min(100, int(40 * (1/scale))),
                    'priority': scale * (colors / source_colors)
                })

        return sorted(configs, key=lambda x: x['priority'], reverse=True)

    def _get_source_fps(self, file_path: Path) -> float:
        """Get the original framerate of the source file."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=r_frame_rate',
                '-of', 'json',
                str(file_path)
            ]

            output = subprocess.check_output(cmd, text=True)
            data = json.loads(output)

            if 'streams' in data and data['streams']:
                # Parse fractional framerate (e.g., "24000/1001")
                rate = data['streams'][0].get('r_frame_rate', '')
                if rate and '/' in rate:
                    num, den = map(float, rate.split('/'))
                    if den != 0:
                        return round(num / den, 2)
                elif rate:
                    return float(rate)

            # Fallback to manual frame counting
            frame_info = self._get_video_frame_info(file_path)
            if frame_info:
                frames, duration = frame_info
                if duration > 0:
                    return round(frames / duration, 2)

            return 30.0  # Default fallback

        except Exception as e:
            self.dev_logger.warning(f"Failed to get source FPS: {e}")
            return 30.0  # Safe default


class ProcessingResult(NamedTuple):
    """Improved result tracking."""
    success: bool
    size: float
    settings: OptimizationConfig
    error: Optional[str] = None


# Add cleanup handler improvements
def cleanup_with_retry(func: Callable) -> Callable:
    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    raise
                time.sleep(0.5 * (2 ** attempt))
    return wrapper


# Add memory monitoring improvements
class MemoryManager:
    """Manages memory usage and cleanup."""

    def __init__(self, threshold_mb: int = 1000):
        self.threshold_mb = threshold_mb
        self._last_cleanup = time.time()
        self._cleanup_interval = 30  # seconds
        self._memory_usage = []
        self._lock = threading.Lock()
        self._gc_threshold = 0.8  # 80% of threshold triggers cleanup

    def check_memory(self) -> bool:
        """Check if memory cleanup is needed with improved thresholds."""
        try:
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024 * 1024)

            with self._lock:
                self._memory_usage.append(current_memory)
                if len(self._memory_usage) > 10:
                    self._memory_usage.pop(0)

                avg_memory = sum(self._memory_usage) / len(self._memory_usage)

                should_cleanup = (
                    current_memory > self.threshold_mb or
                    avg_memory > self.threshold_mb * self._gc_threshold or
                    time.time() - self._last_cleanup > self._cleanup_interval
                )

                if should_cleanup:
                    self.cleanup()
                    return True
            return False
        except Exception as e:
            logging.error(f"Memory check error: {e}")
            return True  # Trigger cleanup on error

    def cleanup(self) -> None:
        """Enhanced memory cleanup with better resource management."""
        try:
            import gc
            gc.collect(generation=2)  # Full collection

            with self._lock:
                self._last_cleanup = time.time()
                self._memory_usage.clear()

            # Try to release memory back to OS on supported platforms
            if hasattr(gc, 'malloc_trim'):  # Python 3.7+ on Linux
                gc.malloc_trim()

            if sys.platform == 'win32':
                import ctypes
                ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
        except Exception as e:
            logging.error(f"Memory cleanup error: {e}")


class BatchProcessor:
    """Handles batch processing of files with memory management."""

    def __init__(self, max_batch_size: int = 5):
        self.max_batch_size = max_batch_size
        self.memory_manager = MemoryManager()
        self.queue = queue.Queue()
        self._results = []
        self._lock = threading.Lock()

    def process_batch(self, items: List[Any], processor_func: Callable) -> Generator:
        """Process items in batches with memory management."""
        for i in range(0, len(items), self.max_batch_size):
            batch = items[i:i + self.max_batch_size]
            futures = []

            with ThreadPoolExecutor(max_workers=min(len(batch), os.cpu_count() or 2)) as executor:
                for item in batch:
                    futures.append(executor.submit(processor_func, item))

                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=120)
                        yield result
                    except Exception as e:
                        logging.error(f"Batch processing error: {str(e)}")
                        continue

            self.memory_manager.check_memory()


class ProcessingStats:
    """Tracks processing statistics and performance metrics."""

    def __init__(self):
        self.stats = {
            'processed': 0,
            'failed': 0,
            'total_time': 0,
            'memory_usage': [],
            'errors': []
        }
        self._lock = threading.Lock()

    def record_success(self, fps: int, size: float, settings: Dict) -> None:
        """Record successful processing."""
        with self._lock:
            self.stats['processed'] += 1
            self.stats.setdefault('fps_distribution', {}).setdefault(fps, 0)
            self.stats['fps_distribution'][fps] += 1

    def record_error(self, error: str) -> None:
        """Record processing error."""
        with self._lock:
            self.stats['failed'] += 1
            self.stats['errors'].append(error)

    def save_stats(self) -> None:
        """Save processing statistics."""
        try:
            with open('processing_stats.json', 'w') as f:
                json.dump(self.stats, f, indent=2)
        except Exception:
            pass


class ResourceManager:
    """Enhanced resource manager with proper cleanup."""

    def __init__(self):
        self.cpu_count = os.cpu_count() or 1
        self.memory_threshold = 85
        self._executors = []
        self._executor = None
        self._shutdown = False
        self._tasks_lock = threading.Lock()
        self._init_lock = threading.Lock()

    def get_executor(self) -> ThreadPoolExecutor:
        """Get or create thread pool executor with tracking."""
        if self._shutdown:
            return None

        with self._init_lock:
            if not self._executor:
                self._executor = ThreadPoolExecutor(
                    max_workers=self.max_workers)
                self._executors.append(self._executor)
            return self._executor

    def shutdown(self):
        """Properly shutdown all executors."""
        self._shutdown = True
        with self._init_lock:
            if self._executor:
                try:
                    self._executor.shutdown(wait=False)
                except Exception as e:
                    logging.error(f"Error shutting down main executor: {e}")
                self._executor = None

            for executor in self._executors:
                try:
                    if not executor._shutdown:
                        executor.shutdown(wait=False)
                except Exception as e:
                    logging.error(f"Error shutting down executor: {e}")
            self._executors.clear()


@asynccontextmanager
async def managed_process_pool(max_workers: int = None):
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        yield pool
    finally:
        pool.shutdown(wait=True)


def _standalone_get_dimensions_ffprobe(file_path: Path) -> Tuple[int, int]:
    """Standalone ffprobe dimension detection."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height,rotation,sample_aspect_ratio',
            '-of', 'json',
            str(file_path)
        ]

        output = subprocess.check_output(
            cmd, stderr=subprocess.PIPE, text=True)
        data = json.loads(output)

        if not data.get('streams'):
            raise ValueError("No video streams found")

        stream = data['streams'][0]
        width = int(stream.get('width', 0))
        height = int(stream.get('height', 0))

        if width <= 0 or height <= 0:
            raise ValueError("Invalid dimensions in stream")

        # Handle rotation
        rotation = int(stream.get('rotation', '0') or '0')
        if rotation in [90, 270]:
            width, height = height, width

        return width, height

    except Exception as e:
        raise ValueError(f"FFprobe dimension detection failed: {e}")


def _standalone_get_dimensions_opencv(file_path: Path) -> Tuple[int, int]:
    """Standalone OpenCV dimension detection."""
    cap = None
    try:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid dimensions: {width}x{height}")

        return (width, height)

    except Exception as e:
        raise ValueError(f"OpenCV dimension detection failed: {e}")
    finally:
        if cap is not None:
            cap.release()


def _standalone_get_dimensions_ffmpeg(file_path: Path) -> Tuple[int, int]:
    """Standalone FFmpeg dimension detection."""
    try:
        cmd = ['ffmpeg', '-i', str(file_path)]
        output = subprocess.run(cmd, capture_output=True, text=True).stderr

        patterns = [
            r'Stream.*Video.* (\d+)x(\d+)',
            r'Video: .* (\d+)x(\d+)',
            r', (\d+)x(\d+)[,\s]'
        ]

        for pattern in patterns:
            matches = re.findall(pattern, output)
            if matches:
                width, height = map(int, matches[0])
                if width > 0 and height > 0:
                    return (width, height)

        raise ValueError("No valid dimensions found in FFmpeg output")
    except Exception as e:
        raise ValueError(f"FFmpeg dimension detection failed: {e}")


@lru_cache(maxsize=100)
def cached_dimension_detection(file_path: str) -> Tuple[int, int]:
    """Cached version of dimension detection using standalone functions."""
    try:
        methods = [
            ('ffprobe', _standalone_get_dimensions_ffprobe),
            ('opencv', _standalone_get_dimensions_opencv),
            ('ffmpeg', _standalone_get_dimensions_ffmpeg)
        ]

        for method_name, method in methods:
            try:
                dimensions = method(Path(file_path))
                if dimensions and all(isinstance(d, int) and d > 0 for d in dimensions):
                    return dimensions
            except Exception as e:
                logging.debug(f"{method_name} dimension detection failed: {e}")
                continue

        raise ValueError("All dimension detection methods failed")

    except Exception as e:
        logging.error(f"Dimension detection failed: {e}")
        return (0, 0)


def validate_compression_settings(settings: Dict[str, Any]) -> None:
    required_keys = ['fps_range', 'colors', 'lossy_value']
    if not all(key in settings for key in required_keys):
        raise ValueError(f"Missing required keys: {required_keys}")


def process_gifs(compression_settings: Optional[Dict[str, Any]] = None) -> List[Path]:
    if compression_settings is not None:
        validate_compression_settings(compression_settings)
    processor = GIFProcessor(compression_settings)
    return processor.process_all()


class QualityManager:
    """Manages quality settings adaptation based on results."""

    def __init__(self):
        self.quality_history = {}
        self._lock = threading.Lock()

    def get_settings(self, file_size: float, target_size: float) -> Dict:
        """Get optimized quality settings based on file size."""
        with self._lock:
            ratio = target_size / file_size
            return {
                'scale_factor': min(1.0, max(0.3, ratio ** 0.5)),
                'colors': 256 if ratio > 0.5 else 192,
                'lossy_value': min(100, int(50 * (1/ratio)))
            }

    def update_settings(self, settings: Dict, result_size: float, target_size: float) -> Dict:
        """Update settings based on optimization results."""
        with self._lock:
            new_settings = settings.copy()
            ratio = result_size / target_size

            if ratio > 1.5:
                new_settings['scale_factor'] *= 0.8
                new_settings['colors'] = min(new_settings['colors'], 192)
                new_settings['lossy_value'] = min(
                    100, new_settings['lossy_value'] + 20)
            elif ratio > 1.2:
                new_settings['scale_factor'] *= 0.9
                new_settings['lossy_value'] = min(
                    100, new_settings['lossy_value'] + 10)

            return new_settings


# Add new imports

# Add configuration constants
MAX_PARALLEL_PROCESSES = max(1, (os.cpu_count() or 2) - 1)
COMPRESSION_BATCH_SIZE = 3
FRAME_ANALYSIS_THRESHOLD = 0.1
MEMORY_LIMIT_MB = 1500
CACHE_SIZE = 100


class OptimizationStrategy(NamedTuple):
    """Enhanced optimization strategy configuration"""
    scale_factor: float
    colors: int
    lossy_value: int
    dither_mode: str
    bayer_scale: int
    frame_skip: int = 0
    compression_level: int = 3
    use_temporal: bool = True


class FrameAnalysis(NamedTuple):
    """Frame analysis results"""
    similarity_score: float
    motion_score: float
    color_count: int
    brightness: float
    complexity: float


class CacheManager:
    """Manages caching for optimization operations"""

    def __init__(self):
        self.frame_cache = TTLCache(maxsize=CACHE_SIZE, ttl=300)
        self.palette_cache = TTLCache(maxsize=CACHE_SIZE, ttl=600)
        self.dimension_cache = TTLCache(maxsize=CACHE_SIZE, ttl=900)
        self._lock = threading.Lock()

    @contextmanager
    def cached_operation(self, cache_key: str, cache_type: str = 'frame'):
        cache = getattr(self, f'{cache_type}_cache')
        with self._lock:
            if cache_key in cache:
                yield cache[cache_key]
                return
        result = yield None
        with self._lock:
            cache[cache_key] = result


class GIFOptimizer(FileProcessor):
    """Enhanced GIF optimization with better performance and results"""

    def __init__(self, compression_settings: Dict = None):
        super().__init__()
        self.compression_settings = compression_settings or GIF_COMPRESSION
        self.cache_manager = CacheManager()
        self.frame_analyzer = FrameAnalyzer()
        self.pool = ThreadPoolExecutor(max_workers=MAX_PARALLEL_PROCESSES)
        self._setup_optimizers()

    def _setup_optimizers(self):
        """Initialize optimization components"""
        self.optimizers = {
            'light': self._create_light_optimizer(),
            'balanced': self._create_balanced_optimizer(),
            'aggressive': self._create_aggressive_optimizer()
        }

    def _create_light_optimizer(self) -> OptimizationStrategy:
        return OptimizationStrategy(
            scale_factor=1.0,
            colors=256,
            lossy_value=20,
            dither_mode='floyd_steinberg',
            bayer_scale=2,
            frame_skip=0,
            use_temporal=True
        )

    def _create_balanced_optimizer(self) -> OptimizationStrategy:
        return OptimizationStrategy(
            scale_factor=0.85,
            colors=192,
            lossy_value=40,
            dither_mode='bayer',
            bayer_scale=3,
            frame_skip=1,
            use_temporal=True
        )

    def _create_aggressive_optimizer(self) -> OptimizationStrategy:
        return OptimizationStrategy(
            scale_factor=0.7,
            colors=128,
            lossy_value=60,
            dither_mode='bayer',
            bayer_scale=4,
            frame_skip=2,
            use_temporal=False
        )

    @log_function_call
    async def optimize_gif(self, input_path: Path, output_path: Path, target_size_mb: float) -> Tuple[float, bool]:
        """Optimized GIF compression with parallel processing and smart analysis"""
        try:
            # Analyze input file
            analysis = await self.frame_analyzer.analyze_file(input_path)

            # Choose optimization strategy based on analysis
            strategy = self._select_strategy(analysis, target_size_mb)

            # Process in parallel batches
            async with ProcessPoolExecutor(max_workers=MAX_PARALLEL_PROCESSES) as pool:
                tasks = self._create_optimization_tasks(input_path, strategy)
                results = await asyncio.gather(*[
                    pool.submit(self._process_batch, batch, strategy)
                    for batch in self._chunk_tasks(tasks, COMPRESSION_BATCH_SIZE)
                ])

            # Combine results and optimize final output
            final_size = await self._combine_and_optimize(
                results, output_path, strategy)

            success = final_size <= target_size_mb
            if not success:
                # Try progressive fallback if needed
                return await self._progressive_fallback(
                    input_path, output_path, target_size_mb)

            return final_size, success

        except Exception as e:
            logger.error(f"Optimization failed: {str(e)}")
            return float('inf'), False

    def _select_strategy(self, analysis: FrameAnalysis, target_size_mb: float) -> OptimizationStrategy:
        """Select optimal strategy based on frame analysis"""
        if analysis.complexity < 0.3 and analysis.motion_score < 0.2:
            return self.optimizers['light']
        elif analysis.complexity < 0.6 and analysis.motion_score < 0.5:
            return self.optimizers['balanced']
        return self.optimizers['aggressive']

    async def _process_batch(self, frames: List[np.ndarray], strategy: OptimizationStrategy) -> bytes:
        """Process a batch of frames with the given strategy"""
        # Implementation for batch processing
        # ...existing implementation...


class FrameAnalyzer:
    """Analyzes GIF frames for optimal compression strategy"""

    def __init__(self):
        self.cache = TTLCache(maxsize=100, ttl=300)

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


class GIFOptimizer(FileProcessor):
    # ...existing code...

    def _create_optimization_tasks(self, input_path: Path, strategy: OptimizationStrategy) -> List[Dict]:
        """Create optimization tasks for parallel processing"""
        tasks = []
        frame_count = self._get_frame_count(input_path)

        # Calculate optimal chunk size based on frame count and CPU cores
        chunk_size = max(1, frame_count // (MAX_PARALLEL_PROCESSES * 2))

        for start in range(0, frame_count, chunk_size):
            end = min(start + chunk_size, frame_count)
            tasks.append({
                'start_frame': start,
                'end_frame': end,
                'strategy': strategy,
                'input_path': input_path
            })

        return tasks

    def _chunk_tasks(self, tasks: List[Dict], batch_size: int) -> List[List[Dict]]:
        """Split tasks into batches for processing"""
        return [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

    async def _process_batch(self, frames: List[np.ndarray], strategy: OptimizationStrategy) -> bytes:
        """Process a batch of frames with the given strategy"""
        # Create temporary buffer for frames
        buffer = io.BytesIO()

        # Create PIL image sequence
        images = []
        for frame in frames:
            # Apply strategy settings
            processed = self._apply_optimization_strategy(frame, strategy)
            images.append(Image.fromarray(processed))

        # Save as GIF with optimization settings
        images[0].save(
            buffer,
            format='GIF',
            save_all=True,
            append_images=images[1:],
            optimize=True,
            duration=1000 // 30,  # 30fps default
            loop=0
        )

        return buffer.getvalue()

    def _apply_optimization_strategy(self, frame: np.ndarray, strategy: OptimizationStrategy) -> np.ndarray:
        """Apply optimization strategy to a single frame"""
        # Resize
        if strategy.scale_factor != 1.0:
            new_size = tuple(int(dim * strategy.scale_factor)
                             for dim in frame.shape[:2])
            frame = cv2.resize(
                frame, new_size[::-1], interpolation=cv2.INTER_LANCZOS4)

        # Convert to PIL for better color quantization
        pil_image = Image.fromarray(frame)

        # Apply dithering and color reduction
        if strategy.colors < 256:
            if strategy.dither_mode == 'floyd_steinberg':
                pil_image = pil_image.quantize(
                    colors=strategy.colors,
                    method=2,  # Floyd-Steinberg dithering
                    palette=Image.ADAPTIVE
                )
            else:  # Bayer dithering
                pil_image = pil_image.quantize(
                    colors=strategy.colors,
                    method=3,  # Ordered dithering
                    palette=Image.ADAPTIVE
                )

        return np.array(pil_image)

    async def _combine_and_optimize(self, results: List[bytes], output_path: Path, strategy: OptimizationStrategy) -> float:
        """Combine processed batches and apply final optimization"""
        # Combine all batches
        combined = io.BytesIO()
        images = []

        for result in results:
            gif = Image.open(io.BytesIO(result))
            for frame in ImageSequence.Iterator(gif):
                images.append(frame.copy())

        # Apply final optimization
        images[0].save(
            output_path,
            format='GIF',
            save_all=True,
            append_images=images[1:],
            optimize=True,
            duration=1000 // 30,
            loop=0
        )

        # Apply gifsicle optimization
        await self._apply_gifsicle_optimization(output_path, strategy)

        return self.get_file_size(output_path)

    async def _progressive_fallback(self, input_path: Path, output_path: Path, target_size_mb: float) -> Tuple[float, bool]:
        """Progressive fallback for when initial optimization fails"""
        strategies = [
            self.optimizers['balanced'],
            self.optimizers['aggressive'],
            OptimizationStrategy(
                scale_factor=0.5,
                colors=64,
                lossy_value=80,
                dither_mode='bayer',
                bayer_scale=5,
                frame_skip=3,
                compression_level=3,
                use_temporal=False
            )
        ]

        for strategy in strategies:
            try:
                size, success = await self.optimize_gif(input_path, output_path, target_size_mb)
                if success:
                    return size, True
            except Exception as e:
                logger.warning(f"Progressive fallback attempt failed: {e}")
                continue

        return float('inf'), False

    async def _apply_gifsicle_optimization(self, gif_path: Path, strategy: OptimizationStrategy) -> None:
        """Apply gifsicle optimization with strategy settings"""
        cmd = [
            'gifsicle',
            '--optimize=3',
            f'--colors={strategy.colors}',
            f'--lossy={strategy.lossy_value}',
            '--no-conserve-memory',
            '--careful'
        ]

        if strategy.frame_skip > 0:
            cmd.extend(
                ['--unoptimize', f'--delete-every={strategy.frame_skip}'])

        cmd.extend([str(gif_path), '--output', str(gif_path)])

        # Run optimization
        await asyncio.create_subprocess_exec(*cmd)


def process_gifs(compression_settings: Optional[Dict[str, Any]] = None) -> List[Path]:
    """Process GIFs with improved optimization system"""
    if compression_settings is not None:
        validate_compression_settings(compression_settings)

    optimizer = GIFOptimizer(compression_settings)

    async def process_all_async():
        return await optimizer.process_all()

    loop = asyncio.get_event_loop()
    return loop.run_until_complete(process_all_async())


# Add new quality control constants
QUALITY_THRESHOLDS = {
    'min_scale_factor': 0.2,  # Never scale below 20% of original size
    'min_colors': 32,         # Minimum allowable colors
    'max_lossy': 100,         # Maximum lossy compression value
    'min_dimension': 64,      # Minimum dimension in pixels
    'min_fps': 5,            # Minimum frames per second
    'acceptable_overshoot': 1.1,  # Allow 10% over target size
}

# Add optimization strategy constants
OPTIMIZATION_STRATEGIES = {
    'quality_steps': [
        # format: (scale_factor, colors, lossy, dither_mode)
        (1.0, 256, 20, 'floyd_steinberg'),   # High quality
        (0.9, 192, 30, 'floyd_steinberg'),   # Good quality
        (0.8, 128, 40, 'sierra2'),           # Balanced
        (0.7, 96, 60, 'bayer'),              # Reduced quality
        (0.5, 64, 80, 'bayer'),              # Low quality
        (0.3, 48, 90, 'bayer'),              # Very low quality
        (0.2, 32, 100, 'bayer')              # Minimum quality
    ],
    'frame_reduction_steps': [1, 2, 3, 4],   # Frame skipping steps
    'temporal_strategies': ['background', 'previous', 'none']
}


class OptimizationResult(NamedTuple):
    """Enhanced result tracking with size and quality metrics"""
    size: float
    target_met: bool
    settings: Dict
    quality_score: float
    warning: Optional[str] = None

# Add new optimization controller


class OptimizationController:
    """Controls optimization process with quality limits"""

    def __init__(self, target_size_mb: float):
        self.target_size_mb = target_size_mb
        self.best_result = None
        self.quality_score = float('inf')
        self.attempts = []

    def evaluate_result(self, result_size: float, settings: Dict) -> OptimizationResult:
        """Evaluate optimization result with quality metrics"""
        quality_score = self._calculate_quality_score(settings)
        target_met = result_size <= self.target_size_mb

        warning = None
        if not target_met and settings['scale_factor'] <= QUALITY_THRESHOLDS['min_scale_factor']:
            warning = "Cannot reach target size without going below minimum quality"

        result = OptimizationResult(
            size=result_size,
            target_met=target_met,
            settings=settings,
            quality_score=quality_score,
            warning=warning
        )

        # Track best result even if target not met
        if not self.best_result or result_size < self.best_result.size:
            self.best_result = result

        return result

    def _calculate_quality_score(self, settings: Dict) -> float:
        """Calculate quality score (0-1, higher is better)"""
        scale_weight = 0.4
        color_weight = 0.3
        lossy_weight = 0.3

        scale_score = settings['scale_factor']
        color_score = settings['colors'] / 256
        lossy_score = 1 - (settings['lossy_value'] / 100)

        return (scale_score * scale_weight +
                color_score * color_weight +
                lossy_score * lossy_weight)

    def should_continue(self, current_settings: Dict) -> bool:
        """Determine if optimization should continue"""
        # Stop if we've hit minimum thresholds
        if (current_settings['scale_factor'] <= QUALITY_THRESHOLDS['min_scale_factor'] or
                current_settings['colors'] <= QUALITY_THRESHOLDS['min_colors']):
            return False

        return True

    def get_next_settings(self, current_result: OptimizationResult) -> Optional[Dict]:
        """Get next optimization settings based on results"""
        current_size = current_result.size
        size_ratio = current_size / self.target_size_mb

        # If very close to target, make small adjustments
        if 1.0 < size_ratio < QUALITY_THRESHOLDS['acceptable_overshoot']:
            return self._fine_tune_settings(current_result.settings)

        # Find next quality step down
        return self._get_next_quality_step(current_result.settings, size_ratio)

    def _fine_tune_settings(self, settings: Dict) -> Dict:
        """Make small adjustments to nearly-good-enough settings"""
        new_settings = settings.copy()

        # Try small color reduction first
        if settings['colors'] > QUALITY_THRESHOLDS['min_colors'] + 16:
            new_settings['colors'] = max(
                QUALITY_THRESHOLDS['min_colors'],
                settings['colors'] - 16
            )
            return new_settings

        # Then try small lossy increase
        if settings['lossy_value'] < QUALITY_THRESHOLDS['max_lossy'] - 5:
            new_settings['lossy_value'] = min(
                QUALITY_THRESHOLDS['max_lossy'],
                settings['lossy_value'] + 5
            )
            return new_settings

        # Finally, try small scale reduction
        if settings['scale_factor'] > QUALITY_THRESHOLDS['min_scale_factor'] + 0.05:
            new_settings['scale_factor'] = max(
                QUALITY_THRESHOLDS['min_scale_factor'],
                settings['scale_factor'] - 0.05
            )
            return new_settings

        return None

    def _get_next_quality_step(self, settings: Dict, size_ratio: float) -> Optional[Dict]:
        """Get next quality step based on how far we are from target"""
        current_quality = next(
            (i for i, step in enumerate(OPTIMIZATION_STRATEGIES['quality_steps'])
             if step[0] == settings['scale_factor']
             and step[1] == settings['colors']),
            0
        )

        # Skip more steps if we're very far from target
        steps_to_skip = min(3, int(size_ratio - 1))
        next_quality = current_quality + steps_to_skip

        if next_quality < len(OPTIMIZATION_STRATEGIES['quality_steps']):
            scale, colors, lossy, dither = OPTIMIZATION_STRATEGIES['quality_steps'][next_quality]
            return {
                'scale_factor': scale,
                'colors': colors,
                'lossy_value': lossy,
                'dither_mode': dither
            }

        return None

# Update GIF optimization class


class GIFOptimizer(FileProcessor):
    """Enhanced GIF optimization with better quality control"""

    def __init__(self, compression_settings: Dict = None):
        # ...existing initialization...

        self.frame_analyzer = FrameAnalyzer()
        self.warnings = []

    def optimize_gif(self, input_path: Path, output_path: Path, target_size_mb: float) -> Tuple[float, bool]:
        """Main optimization method with quality control"""
        controller = OptimizationController(target_size_mb)

        # Start with highest quality settings
        current_settings = self._get_initial_settings(input_path)

        while True:
            # Apply current settings
            result_size = self._apply_optimization(
                input_path, output_path, current_settings)

            # Evaluate result
            result = controller.evaluate_result(result_size, current_settings)

            # Log progress
            self._log_optimization_progress(result)

            if result.target_met:
                return result_size, True

            if result.warning:
                self.warnings.append(result.warning)
                # Use best result if we can't meet target
                if controller.best_result:
                    best = controller.best_result
                    self._apply_optimization(
                        input_path, output_path, best.settings)
                    return best.size, False

            # Get next settings
            next_settings = controller.get_next_settings(result)
            if not next_settings or not controller.should_continue(next_settings):
                # Use best result if we can't continue
                if controller.best_result:
                    best = controller.best_result
                    self._apply_optimization(
                        input_path, output_path, best.settings)
                    return best.size, False
                return result_size, False

            current_settings = next_settings

    def _get_initial_settings(self, input_path: Path) -> Dict:
        """Get initial settings based on input analysis"""
        analysis = self.frame_analyzer.analyze_file(input_path)

        # Start with highest quality settings
        settings = {
            'scale_factor': 1.0,
            'colors': 256,
            'lossy_value': 20,
            'dither_mode': 'floyd_steinberg'
        }

        # Adjust based on analysis
        if analysis.complexity < 0.3:
            settings['colors'] = 192  # Less complex images need fewer colors

        if analysis.motion_score < 0.2:
            settings['fps'] = max(QUALITY_THRESHOLDS['min_fps'],
                                  int(analysis.motion_score * 30))

        return settings

    def _log_optimization_progress(self, result: OptimizationResult) -> None:
        """Log optimization progress with quality metrics"""
        quality_percent = int(result.quality_score * 100)
        self.dev_logger.info(
            f"Size: {result.size:.2f}MB, "
            f"Quality: {quality_percent}%, "
            f"Settings: scale={result.settings['scale_factor']:.2f}, "
            f"colors={result.settings['colors']}, "
            f"lossy={result.settings['lossy_value']}"
        )

        if result.warning:
            self.dev_logger.warning(result.warning)
