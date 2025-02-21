import asyncio
from asyncio.log import logger
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from dataclasses import dataclass
from enum import Enum
from functools import wraps
import gc
import json
import logging
import os
from pathlib import Path
import queue
import re
import shutil
import subprocess
import sys
import threading
import time
from tkinter import Image
import traceback
from typing import Callable, Dict, List, NamedTuple, Optional, Tuple

from cachetools import TTLCache
import cv2
import psutil

from src.default_config import GIF_SIZE_TO_SKIP, INPUT_DIR, OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS, TEMP_FILE_DIR
from src.gif_operations.ffmpeg_handler import FFmpegHandler
from src.gif_operations.memory import MemoryManager
from src.gif_operations.optimizer import GIFOptimizer, ProcessingStatus
from src.gif_operations.processing_stats import ProcessingStats
from src.gif_operations.quality_manager import QualityManager
from src.gif_operations.resource_manager import ResourceMonitor
from src.logging_system import log_function_call, run_ffmpeg_command
from src.temp_file_manager import TempFileManager
# Add DynamicGIFOptimizer to import
from src.gif_operations.optimizer import GIFOptimizer, DynamicGIFOptimizer
from src.gif_operations.base_processor import BaseProcessor


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


class OptimizationConfig(NamedTuple):
    scale_factor: float
    colors: int
    lossy_value: int
    dither_mode: str = 'bayer'
    bayer_scale: int = 3


class ProcessingResult(NamedTuple):
    """Improved result tracking."""
    success: bool
    size: float
    settings: OptimizationConfig
    error: Optional[str] = None


class ProcessingStatus(Enum):
    """Improved processing status enumeration."""
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


class GIFProcessor(BaseProcessor):
    """Main GIF processing class combining optimization and conversion."""

    def __init__(self, verbose: bool = False):
        # Initialize base class first
        super().__init__(verbose=verbose)

        # Initialize thread-safe locks and events
        self._shutdown_event = threading.Event()
        self._processing_cancelled = threading.Event()
        self._immediate_termination = threading.Event()
        self._shutdown_initiated = False
        self._file_locks = {}
        self._file_locks_lock = threading.Lock()
        self._threads_lock = threading.Lock()
        self._active_threads = set()
        self._stats_lock = threading.Lock()
        self._progress_lock = threading.Lock()
        self._processing_progress = {}
        self._stats = {'retried': 0}
        self._cleanup_handlers = []
        self.processed_files = set()
        self.logging_lock = threading.Lock()
        self.failed_files = []

        # Initialize compression settings with consistent target size
        self.compression_settings = {
            'min_size_mb': 10.0,  # Changed from 15.0 to 10.0
            'fps_range': (15, 30),
            'colors': 256,
            'quality': 85
        }

        # Initialize rest of components
        self.dynamic_optimizer = DynamicGIFOptimizer(self.compression_settings)
        self._init_worker_management()
        self._init_caches()
        self._init_components()
        self._init_async_support()

    def _init_worker_management(self):
        """Initialize worker thread management."""
        self.worker_threads = []
        self.max_threads = 2
        self.task_queue = queue.Queue()
        self.result_queue = queue.Queue()
        self._stop_workers = threading.Event()

    def _init_caches(self):
        """Initialize caches."""
        self._file_cache = TTLCache(maxsize=100, ttl=300)
        self.dimension_cache = TTLCache(maxsize=100, ttl=300)

    def _init_components(self):
        """Initialize processing components."""
        self.ffmpeg = FFmpegHandler()
        self.memory_manager = MemoryManager(threshold_mb=1500)
        self.stats = ProcessingStats()
        self.resource_monitor = ResourceMonitor()
        self.quality_manager = QualityManager()
        self.progressive_optimization = True
        self.retry_count = 3
        self.retry_delay = 1

    def _init_async_support(self):
        """Initialize async support."""
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

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
            # First kill any running processes
            self._kill_ffmpeg_processes()

            # Wait a moment for processes to die
            time.sleep(0.5)

            # Force close any file handles
            gc.collect()

            # Clear locks and resources
            with self._file_locks_lock:
                self._file_locks.clear()

            # Run cleanup handlers
            for handler in self._cleanup_handlers:
                try:
                    handler()
                except Exception as e:
                    self.dev_logger.error(f"Cleanup handler failed: {str(e)}")

            # Clean temp files with multiple attempts
            self._cleanup_temp_directory()

            # Clear caches
            self._clear_caches()

        except Exception as e:
            self.dev_logger.error(f"Error during cleanup: {str(e)}")
        finally:
            self._stop_workers.set()
            self._cleanup_threads()

    def _cleanup_temp_directory(self):
        """Clean up temporary directory with improved handling."""
        temp_dir = Path(TEMP_FILE_DIR)
        if not temp_dir.exists():
            return

        try:
            # Multiple attempts to clean files
            for attempt in range(3):
                remaining_files = []

                # Try to remove each file
                for item in temp_dir.glob("*"):
                    try:
                        if item.is_file():
                            try:
                                item.unlink(missing_ok=True)
                            except Exception:
                                remaining_files.append(item)
                        elif item.is_dir():
                            try:
                                shutil.rmtree(item, ignore_errors=True)
                            except Exception:
                                remaining_files.append(item)
                    except Exception as e:
                        self.dev_logger.debug(f"Failed to remove {item}: {e}")
                        remaining_files.append(item)

                if not remaining_files:
                    break

                # If files remain, try more aggressive cleanup
                if attempt < 2:
                    self._kill_ffmpeg_processes()
                    gc.collect()
                    time.sleep(1)

            # If files still remain, log them but continue
            if remaining_files:
                self.dev_logger.warning(
                    f"Could not remove {len(remaining_files)} files in temp directory")
                for file in remaining_files:
                    self.dev_logger.debug(f"Remaining file: {file}")

            # Don't try to remove the temp directory if files remain
            if not remaining_files:
                try:
                    temp_dir.rmdir()
                except Exception as e:
                    self.dev_logger.debug(
                        f"Could not remove temp directory: {e}")

            # Always ensure temp directory exists
            temp_dir.mkdir(parents=True, exist_ok=True)

        except Exception as e:
            self.dev_logger.error(f"Error cleaning temp directory: {str(e)}")
            # Ensure temp directory exists even if cleanup failed
            temp_dir.mkdir(parents=True, exist_ok=True)

    def _clear_caches(self):
        """Clear all caches."""
        if hasattr(self, '_file_cache'):
            self._file_cache.clear()
        if hasattr(self, 'dimension_cache'):
            self.dimension_cache.clear()
        if hasattr(self, 'palette_cache'):
            self.palette_cache.clear()

    def _cleanup_threads(self):
        """Clean up worker threads."""
        for thread in self.worker_threads:
            if thread.is_alive():
                thread.join(timeout=5)
        self.worker_threads.clear()

        # Clear queues
        self._clear_queues()

    def _clear_queues(self):
        """Clear all queues."""
        for queue_obj in [self.task_queue, self.result_queue]:
            while not queue_obj.empty():
                try:
                    queue_obj.get_nowait()
                except queue.Empty:
                    break

    def _kill_ffmpeg_processes(self):
        """Kill any running FFmpeg processes."""
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
    def create_gif(self, file_path: Path, output_path: Path, fps: int, dimensions: Tuple[int, int]) -> bool:
        """Create optimized GIF with enhanced logging."""
        try:
            # Early exit for large files
            MAX_INTERMEDIATE_SIZE_MB = 100  # Set a reasonable limit for intermediate files

            if self.verbose:
                self.dev_logger.info(f"""
=== Starting GIF Creation ===
Source: {file_path.name}
Target FPS: {fps}
Input Dimensions: {dimensions[0]}x{dimensions[1]}
Output Path: {output_path}
===========================""")

            original_size = self.get_file_size(file_path)
            if original_size > 1000:
                self.dev_logger.error(
                    f"Source file too large: {original_size:.2f}MB")
                return False

            # Calculate dimensions with more aggressive scaling for large files
            max_dimension = max(dimensions[0], dimensions[1])
            scale_factor = 1.0

            # More aggressive scaling based on file size
            if original_size > 100:
                scale_factor = min(scale_factor, 0.5)
            elif original_size > 50:
                scale_factor = min(scale_factor, 0.7)

            if max_dimension > 1280:
                scale_factor = min(scale_factor, 1280 / max_dimension)

            if self.verbose:
                self.dev_logger.info(
                    f"Using scale factor {scale_factor:.2f} based on size and dimensions")

            new_width = int(dimensions[0] * scale_factor // 2 * 2)
            new_height = int(dimensions[1] * scale_factor // 2 * 2)

            if self.verbose:
                self.dev_logger.info(
                    f"Output dimensions: {new_width}x{new_height}")

            # Generate temp file paths
            temp_palette = Path(TEMP_FILE_DIR) / \
                f"palette_{file_path.stem}.png"
            temp_output = Path(TEMP_FILE_DIR) / f"temp_{output_path.stem}.gif"

            try:
                if self.verbose:
                    self.dev_logger.info(
                        "=== Step 1: Generating color palette ===")

                palette_cmd = [
                    'ffmpeg', '-i', str(file_path),
                    '-vf', f'fps={fps},scale={new_width}:{new_height}:flags=lanczos,palettegen=max_colors=128:stats_mode=diff',
                    '-y', str(temp_palette)
                ]

                if self.verbose:
                    self.dev_logger.info(
                        f"Palette command: {' '.join(palette_cmd)}")

                if not run_ffmpeg_command(palette_cmd):
                    self.dev_logger.error("Palette generation failed")
                    return False

                palette_size = temp_palette.stat().st_size / 1024
                if self.verbose:
                    self.dev_logger.info(
                        f"Palette generated successfully ({palette_size:.1f}KB)")
                    self.dev_logger.info(
                        "\n=== Step 2: Creating initial GIF ===")

                gif_cmd = [
                    'ffmpeg', '-i', str(file_path),
                    '-i', str(temp_palette),
                    '-lavfi',
                    f'fps={fps},scale={new_width}:{new_height}:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
                    '-y', str(temp_output)
                ]

                if self.verbose:
                    self.dev_logger.info(
                        f"GIF creation command: {' '.join(gif_cmd)}")

                if not run_ffmpeg_command(gif_cmd):
                    self.dev_logger.error("Initial GIF creation failed")
                    return False

                # Add size check after initial GIF creation
                if temp_output.exists():
                    initial_size = self.get_file_size(temp_output)
                    if initial_size > MAX_INTERMEDIATE_SIZE_MB:
                        self.dev_logger.warning(  # Changed from error to warning
                            f"Initial GIF too large ({initial_size:.2f}MB > {MAX_INTERMEDIATE_SIZE_MB}MB), trying more aggressive compression")
                        # Try more aggressive settings
                        return self._create_gif_aggressive(file_path, output_path, fps, dimensions)

                    # Continue with normal optimization if size is reasonable
                    if self.verbose:
                        self.dev_logger.info(
                            f"Initial GIF created: {initial_size:.2f}MB")
                        self.dev_logger.info(
                            "\n=== Step 3: Optimizing with Gifsicle ===")

                    gifsicle_cmd = [
                        'gifsicle',
                        '--optimize=3',
                        '--colors', '128',
                        '--lossy=80',
                        '--no-conserve-memory',
                        str(temp_output),
                        '-o', str(output_path)
                    ]

                    if self.verbose:
                        self.dev_logger.info(
                            f"Optimization command: {' '.join(gifsicle_cmd)}")

                    if not run_ffmpeg_command(gifsicle_cmd):
                        self.dev_logger.error("Gifsicle optimization failed")
                        return False

                    final_size = self.get_file_size(output_path)
                    reduction = ((original_size - final_size) /
                                 original_size) * 100

                    if self.verbose:
                        self.dev_logger.info(f"""
=== GIF Creation Summary ===
Original Size: {original_size:.2f}MB
Initial GIF: {initial_size:.2f}MB
Final Size: {final_size:.2f}MB
Reduction: {reduction:.1f}%
FPS: {fps}
Colors: 128
Dimensions: {new_width}x{new_height}
=========================""")

                    # Changed this condition to use min_size_mb instead of hardcoded 100
                    if final_size > self.compression_settings['min_size_mb']:
                        if self.verbose:
                            self.dev_logger.info(
                                "\n=== Step 4: Additional Compression ===")
                        self._compress_large_gif(output_path)

                    # Final check against target size
                    final_size = self.get_file_size(output_path)
                    return final_size <= self.compression_settings['min_size_mb']

                return False

            finally:
                # Cleanup with logging
                if self.verbose:
                    self.dev_logger.info(
                        "\n=== Cleaning up temporary files ===")
                for temp_file in [temp_palette, temp_output]:
                    try:
                        if temp_file.exists():
                            temp_file.unlink()
                            if self.verbose:
                                self.dev_logger.info(f"Removed: {temp_file}")
                    except Exception as e:
                        self.dev_logger.error(
                            f"Failed to cleanup {temp_file}: {e}")

        except Exception as e:
            self.dev_logger.error(f"GIF creation failed: {str(e)}")
            return False

    def _create_gif_aggressive(self, file_path: Path, output_path: Path, fps: int, dimensions: Tuple[int, int]) -> bool:
        """Create GIF with dynamically adjusted aggressive compression settings."""
        original_size = self.get_file_size(file_path)
        best_result_size = float('inf')
        best_result_path = None
        target_size = self.compression_settings['min_size_mb']

        # Calculate compression ratio needed
        compression_ratio = original_size / target_size

        # Modified configs to preserve colors better
        configs = []
        if compression_ratio > 8:
            # Extremely large files - preserve colors but be aggressive with other params
            configs.extend([
                {'scale': 0.3, 'colors': 256,
                    'fps': min(fps, 12), 'lossy': 100},
                {'scale': 0.4, 'colors': 192, 'fps': min(fps, 15), 'lossy': 90}
            ])
        elif compression_ratio > 4:
            # Large files - still keep good color depth
            configs.extend([
                {'scale': 0.4, 'colors': 256,
                    'fps': min(fps, 15), 'lossy': 90},
                {'scale': 0.5, 'colors': 192, 'fps': min(fps, 20), 'lossy': 80}
            ])
        else:
            # Moderate compression - maintain maximum color quality
            configs.extend([
                {'scale': 0.6, 'colors': 256,
                    'fps': min(fps, 20), 'lossy': 80},
                {'scale': 0.7, 'colors': 256, 'fps': min(fps, 25), 'lossy': 70}
            ])

        try:
            # Create a unique temp directory for this process
            temp_dir = Path(TEMP_FILE_DIR) / \
                f"aggressive_{file_path.stem}_{int(time.time())}"
            temp_dir.mkdir(parents=True, exist_ok=True)

            for config in configs:
                if self._should_exit():
                    break

                try:
                    temp_palette = temp_dir / f"palette_{config['colors']}.png"
                    temp_output = temp_dir / f"output_{config['colors']}.gif"

                    new_width = int(dimensions[0] * config['scale'] // 2 * 2)
                    new_height = int(dimensions[1] * config['scale'] // 2 * 2)

                    # Ensure minimum dimensions
                    new_width = max(16, new_width)
                    new_height = max(16, new_height)

                    # Modified palette generation command to preserve color fidelity
                    palette_cmd = [
                        'ffmpeg', '-i', str(file_path),
                        '-vf', (f'fps={config["fps"]},scale={new_width}:{new_height}:flags=lanczos,'
                                f'palettegen=max_colors={config["colors"]}:stats_mode=full:reserve_transparent=0'),
                        '-y', str(temp_palette)
                    ]

                    if not run_ffmpeg_command(palette_cmd):
                        continue

                    # Modified paletteuse settings for better color preservation
                    gif_cmd = [
                        'ffmpeg', '-i', str(file_path),
                        '-i', str(temp_palette),
                        '-lavfi',
                        (f'fps={config["fps"]},scale={new_width}:{new_height}:flags=lanczos[x];'
                         f'[x][1:v]paletteuse=dither=floyd_steinberg:diff_mode=rectangle:new=1'),
                        '-y', str(temp_output)
                    ]

                    if not run_ffmpeg_command(gif_cmd):
                        continue

                    # Check if file was created and optimize with gifsicle
                    if temp_output.exists():
                        # Modified gifsicle optimization to focus on frame optimization rather than color reduction
                        gifsicle_cmd = [
                            'gifsicle',
                            '--optimize=3',
                            '--colors', str(config['colors']),
                            '--lossy=' + str(config['lossy']),
                            '--no-conserve-memory',
                            '--careful',  # Added careful flag
                            '--dither=none',  # Prevent additional dithering
                            str(temp_output),
                            '-o', str(temp_output)
                        ]

                        if run_ffmpeg_command(gifsicle_cmd):
                            current_size = self.get_file_size(temp_output)
                            self.dev_logger.info(
                                f"Aggressive attempt result: {current_size:.2f}MB"
                            )

                            # Update best result if this is better
                            if current_size < best_result_size:
                                if best_result_path and best_result_path != temp_output:
                                    try:
                                        best_result_path.unlink()
                                    except Exception:
                                        pass
                                best_result_size = current_size
                                best_result_path = temp_output
                                # Create a copy in the temp directory
                                best_copy = temp_dir / \
                                    f"best_{config['colors']}.gif"
                                shutil.copy2(temp_output, best_copy)
                                best_result_path = best_copy

                            # If we've reached target size, we can stop
                            if current_size <= target_size:
                                shutil.copy2(best_result_path, output_path)
                                return True

                except Exception as e:
                    self.dev_logger.error(
                        f"Error during aggressive compression attempt: {e}")
                    continue

            # If we have a best result but didn't reach target, use it anyway
            if best_result_path and best_result_path.exists():
                if best_result_size < original_size:
                    shutil.copy2(best_result_path, output_path)
                    self.dev_logger.warning(
                        f"Using best achieved result: {best_result_size:.2f}MB"
                    )
                    return True

            return False

        except Exception as e:
            self.dev_logger.error(f"Aggressive GIF creation failed: {str(e)}")
            return False
        finally:
            # Clean up temp directory and all its contents
            try:
                if temp_dir.exists():
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception as e:
                self.dev_logger.error(f"Failed to cleanup temp directory: {e}")

    def process_file(self, file_path: Path, output_path: Path, is_video: bool) -> bool:
        """Process a single file with improved error handling."""
        try:
            if not file_path.exists():
                self.dev_logger.error(
                    f"Input file does not exist: {file_path}")
                return False

            # Check file size before processing
            file_size = self.get_file_size(file_path)
            if file_size > GIF_SIZE_TO_SKIP:
                self.dev_logger.warning(
                    f"Skipping {file_path.name} - {file_size:.1f}MB exceeds size limit")
                return False

            try:
                if is_video:
                    # For videos, create temporary GIF with timestamp
                    timestamp = int(time.time() * 1000)
                    temp_gif = Path(TEMP_FILE_DIR) / \
                        f"temp_{timestamp}_{file_path.stem}.gif"

                    # Get dimensions with retry
                    dimensions = self._get_dimensions_with_retry(file_path)
                    if not dimensions or not self._validate_dimensions(dimensions):
                        self.dev_logger.error(
                            f"Invalid dimensions for {file_path}")
                        return False

                    self.dev_logger.info(
                        f"Creating GIF from video: {file_path.name}")
                    if not self.create_gif(file_path, temp_gif, 30, dimensions):
                        self.dev_logger.error(
                            f"Failed to create GIF from video: {file_path}")
                        return False

                    # Add size check after initial creation
                    if temp_gif.exists():
                        initial_size = self.get_file_size(temp_gif)
                        if initial_size > self.compression_settings['min_size_mb'] * 2:
                            self.dev_logger.info(
                                f"Initial GIF too large ({initial_size:.2f}MB), trying direct FFmpeg conversion")
                            temp_gif.unlink(missing_ok=True)
                            return self._compress_large_with_ffmpeg(file_path, output_path)

                        # Now optimize the temporary GIF
                        self.dev_logger.info(
                            f"Optimizing temporary GIF: {temp_gif}")
                        result = self.optimize_gif(
                            temp_gif, output_path, self.compression_settings['min_size_mb'])
                        temp_gif.unlink(missing_ok=True)
                        return result[1]

                else:
                    # Direct GIF optimization
                    self.dev_logger.info(
                        f"Optimizing GIF directly: {file_path}")
                    result = self.optimize_gif(
                        file_path, output_path, self.compression_settings['min_size_mb'])
                    return result[1]

            except Exception as e:
                self.dev_logger.error(f"Processing error: {e}")
                return False

        except Exception as e:
            self.dev_logger.error(f"Error processing {file_path}: {str(e)}")
            return False

    def _process_video(self, file_path: Path, output_path: Path, dimensions: Tuple[int, int]) -> None:
        """Process video with improved strategy for large files."""
        temp_files = []
        timestamp = int(time.time() * 1000)
        success = False
        best_size = float('inf')
        best_temp_file = None
        logged_attempts = set()  # Add this to track logged attempts

        initial_size = self.get_file_size(file_path)
        final_size = initial_size

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

                # Create temp files in main temp directory
                current_temp_file = Path(
                    TEMP_FILE_DIR) / f"temp_{timestamp}_{file_path.stem}.gif"
                temp_files.append(current_temp_file)

                # Calculate dimensions
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

            success = final_size <= self.compression_settings.get(
                'min_size_mb', 10.0)
            self.stats.add_result(file_path, initial_size, final_size, success)

        except Exception as e:
            self.dev_logger.error(f"Error: {str(e)}")
            self.stats.add_result(file_path, initial_size, initial_size, False)
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
                               width: int, height: int, settings: OptimizationConfig, target_size: float) -> Optional[Dict]:
        """Process a single optimization configuration."""
        try:
            # Create initial GIF
            if self.ffmpeg.create_optimized_gif(
                file_path,
                temp_gif,
                fps,
                (int(width * settings.scale_factor),
                 int(height * settings.scale_factor)),
                {
                    'colors': settings.colors,
                    'lossy_value': settings.lossy_value,
                    'dither_mode': settings.dither_mode,
                    'bayer_scale': settings.bayer_scale
                }
            ):
                size = self.get_file_size(temp_gif)

                # Only proceed with gifsicle if initial size is promising
                if size < min(90, target_size * 2):
                    optimized_gif = Path(TEMP_FILE_DIR) / \
                        f"{temp_gif.stem}_opt.gif"

                    cmd = [
                        'gifsicle',
                        '--optimize=3',
                        '--colors', str(settings.colors),
                        '--lossy=' + str(settings.lossy_value),
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
        """Process all files with improved reporting."""
        self.failed_files = []
        self.stats.reset()

        try:
            # Get all input files and sort by size
            input_files = []

            # Collect video files
            for video_format in SUPPORTED_VIDEO_FORMATS:
                input_files.extend(Path(INPUT_DIR).glob(f'*{video_format}'))

            # Collect GIF files
            input_files.extend(Path(INPUT_DIR).glob('*.gif'))

            if not input_files:
                self.dev_logger.info("No files found to process")
                return []

            # Sort files by size (smallest first)
            sorted_files = sorted(input_files, key=lambda f: f.stat().st_size)

            # Process sorted files
            total_files = len(sorted_files)
            for index, file_path in enumerate(sorted_files, 1):
                if self._should_exit():
                    self.dev_logger.info("Gracefully stopping processing...")
                    break

                self.dev_logger.info(
                    f"\nProcessing file {index}/{total_files}: {file_path.name}"
                )

                output_gif = Path(OUTPUT_DIR) / f"{file_path.stem}.gif"
                # Add this line
                target_size = self.compression_settings['min_size_mb']

                # Check if output exists and meets size requirements
                if output_gif.exists():
                    output_size = self.get_file_size(output_gif)
                    if output_size <= target_size:  # Use target_size consistently
                        self.dev_logger.info(
                            f"Skipping {file_path.name} - Output already exists and meets size requirement ({output_size:.2f}MB <= {target_size}MB)"
                        )
                        continue
                    else:
                        self.dev_logger.info(
                            f"Reprocessing {file_path.name} - Existing output ({output_size:.2f}MB) exceeds size limit ({target_size}MB)"
                        )
                        try:
                            output_gif.unlink()
                        except Exception as e:
                            self.dev_logger.error(
                                f"Failed to remove oversized output: {e}")
                            continue

                try:
                    is_video = file_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS
                    self.process_file(file_path, output_gif, is_video=is_video)

                    # Verify the output was created
                    if not output_gif.exists():
                        self.dev_logger.error(
                            f"Failed to create output for {file_path.name}")
                        self.failed_files.append(file_path)

                except Exception as e:
                    self.dev_logger.error(
                        f"Error processing {file_path.name}: {str(e)}")
                    self.failed_files.append(file_path)

                # Check memory usage and cleanup if needed
                self.memory_manager.check_memory()

        except KeyboardInterrupt:
            self.dev_logger.warning("\nProcessing interrupted by user")
            self._shutdown_event.set()
        except Exception as e:
            self.dev_logger.error(f"Error in process_all: {str(e)}")
            self.dev_logger.debug(f"Stack trace: {traceback.format_exc()}")
        finally:
            # Log processing summary
            self.dev_logger.info(self.stats.get_summary())

            if self._should_exit():
                self.dev_logger.info("Cleaning up after graceful exit...")
            self.cleanup_resources()

            # Report final status
            if self.failed_files:
                self.dev_logger.warning(
                    f"Failed to process {len(self.failed_files)} files"
                )
            else:
                self.dev_logger.success("All files processed successfully")

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

        self.dev_logger.warning("\nForce stopping all processes...")
        try:
            # Set all termination flags immediately
            self._immediate_termination.set()
            self._shutdown_event.set()
            self._processing_cancelled.set()
            self._shutdown_initiated = True

            # Kill all FFmpeg and Gifsicle processes immediately without waiting
            self._kill_all_processes(force=True)

            # Clear queues without waiting
            self._clear_queues_no_wait()

            # Stop all worker threads immediately
            self._stop_workers.set()
            self._terminate_threads()

        except Exception as e:
            self.dev_logger.error(f"Error during immediate shutdown: {e}")
        finally:
            sys.exit(1)  # Force exit without waiting

    def _kill_all_processes(self, force: bool = False):
        """Kill all running processes with optional force flag."""
        try:
            if sys.platform == 'win32':
                # Use taskkill with /F for force
                kill_flags = ['/F'] if force else []
                for proc in ['ffmpeg.exe', 'gifsicle.exe']:
                    subprocess.run(['taskkill'] + kill_flags + ['/IM', proc],
                                   stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
            else:
                # Use SIGKILL for force, SIGTERM otherwise
                sig = signal.SIGKILL if force else signal.SIGTERM
                os.system(f"pkill -{sig} ffmpeg")
                os.system(f"pkill -{sig} gifsicle")
        except Exception as e:
            self.dev_logger.error(f"Failed to kill processes: {e}")

    def _clear_queues_no_wait(self):
        """Clear queues without waiting."""
        for queue_obj in [self.task_queue, self.result_queue]:
            while True:
                try:
                    queue_obj.get_nowait()
                except queue.Empty:
                    break

    def _terminate_threads(self):
        """Terminate threads without waiting for completion."""
        for thread in list(self.worker_threads):
            if thread.is_alive():
                # Just flag the thread, don't wait
                self._stop_workers.set()
        self.worker_threads.clear()

    def _should_exit(self) -> bool:
        """Enhanced exit check with immediate response."""
        # Check immediate termination first
        if self._immediate_termination.is_set():
            return True
        return (self._shutdown_event.is_set() or
                self._processing_cancelled.is_set() or
                self._shutdown_initiated)

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

        # Remove the wait_for_file_completion check
        self.dev_logger.debug(f"Starting dimension detection for {file_path}")

        methods = [
            (self._get_dimensions_ffprobe, "FFprobe"),
            (self._get_dimensions_opencv, "OpenCV"),
            (self._get_dimensions_ffmpeg, "FFmpeg")
        ]

        for method, method_name in methods:
            for attempt in range(max_retries):
                try:
                    self.dev_logger.debug(
                        f"Trying {method_name} (attempt {attempt + 1})")
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
        """Optimize GIF until it reaches target size or max attempts are reached."""
        try:
            original_size = self.get_file_size(input_path)
            target_size_mb = min(
                target_size_mb, self.compression_settings['min_size_mb'])

            # Copy if already small enough
            if original_size <= target_size_mb:
                shutil.copy2(input_path, output_path)
                return original_size, True

            max_attempts = 5  # Maximum optimization attempts
            current_size = original_size
            current_input = input_path
            temp_output = None

            for attempt in range(max_attempts):
                if self._should_exit():
                    break

                temp_output = Path(TEMP_FILE_DIR) / \
                    f"temp_opt_{attempt}_{output_path.name}"

                # Skip gifsicle for large files (>100MB) and use FFmpeg directly
                if current_size > 100:
                    success = self._compress_large_with_ffmpeg(
                        current_input, temp_output)
                else:
                    success = self._compress_with_gifsicle(
                        current_input, temp_output)

                if not success or not temp_output.exists():
                    self.dev_logger.error(
                        f"Optimization attempt {attempt + 1} failed")
                    continue

                new_size = self.get_file_size(temp_output)
                self.dev_logger.info(
                    f"Attempt {attempt + 1}: {new_size:.2f}MB (Target: {target_size_mb:.2f}MB)")

                if new_size <= target_size_mb:
                    # Target reached
                    shutil.move(str(temp_output), str(output_path))
                    return new_size, True
                elif new_size < current_size:
                    # Better but not enough - continue optimizing
                    current_size = new_size
                    if current_input != input_path:
                        current_input.unlink()
                    current_input = temp_output
                    continue
                else:
                    # No improvement
                    temp_output.unlink()
                    break

            # If we have a better result but didn't reach target, use it
            if current_size < original_size and current_input != input_path:
                shutil.move(str(current_input), str(output_path))
                return current_size, False

            return original_size, False

        except Exception as e:
            self.dev_logger.error(f"Optimization error: {e}")
            return original_size, False
        finally:
            # Cleanup any remaining temp files
            try:
                if temp_output and temp_output.exists():
                    temp_output.unlink()
                if current_input != input_path and current_input.exists():
                    current_input.unlink()
            except Exception as e:
                self.dev_logger.error(f"Cleanup error: {e}")

    def _compress_with_gifsicle(self, input_path: Path, output_path: Path) -> bool:
        """Compress GIF using gifsicle with progressively aggressive settings."""
        try:
            cmd = [
                'gifsicle',
                '--optimize=3',
                '--colors', '128',
                '--lossy=80',
                '--no-conserve-memory',
                str(input_path),
                '-o', str(output_path)
            ]
            return run_ffmpeg_command(cmd)
        except Exception as e:
            self.dev_logger.error(f"Gifsicle compression failed: {e}")
            return False

    def _compress_large_with_ffmpeg(self, input_path: Path, output_path: Path) -> bool:
        """Compress large GIF using FFmpeg with aggressive settings."""
        try:
            # Get dimensions
            dimensions = self._get_dimensions_with_retry(input_path)
            if not dimensions:
                return False

            width, height = dimensions
            new_width = int(width * 0.7)  # Scale down to 70%
            new_height = int(height * 0.7)

            cmd = [
                'ffmpeg', '-i', str(input_path),
                '-vf', f'scale={new_width}:{new_height}:flags=lanczos,fps=15',
                '-gifflags', '-offsetting',
                '-y', str(output_path)
            ]
            return run_ffmpeg_command(cmd)
        except Exception as e:
            self.dev_logger.error(f"FFmpeg compression failed: {e}")
            return False

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

    def _progressive_optimize(self, input_path: Path, output_path: Path, target_size_mb: float) -> ProcessingResult:
        """Progressive optimization with quality reduction."""
        try:
            # Instead of creating a temp directory, just use temp files directly in TEMP_FILE_DIR
            timestamp = int(time.time() * 1000)
            temp_output = Path(TEMP_FILE_DIR) / \
                f"temp_opt_{timestamp}_{output_path.name}"

            original_size = self.get_file_size(input_path)
            dimensions = self._get_dimensions_with_retry(input_path)
            if not dimensions:
                return ProcessingResult(0, original_size, None, ProcessingStatus.DIMENSION_ERROR)

            # Start with best quality settings
            config = OptimizationConfig(
                scale_factor=1.0,
                colors=256,
                lossy_value=20,
                dither_mode='floyd_steinberg'
            )

            best_result = None
            for attempt in range(3):
                if self._should_exit():
                    break

                result = self._process_single_config(
                    input_path,
                    temp_output,
                    self.compression_settings['fps_range'][0],
                    dimensions[0],
                    dimensions[1],
                    config,
                    target_size_mb
                )

                # Add null check for result before accessing
                if result is None:
                    continue

                # Now safely access result dictionary
                if result.get('success'):
                    if result.get('size', float('inf')) <= target_size_mb:
                        return ProcessingResult(
                            result.get('fps', 0),
                            result.get('size', 0),
                            result.get('path'),
                            ProcessingStatus.SUCCESS,
                            settings=config
                        )

                    # Update best_result if we have a better size
                    if not best_result or result.get('size', float('inf')) < best_result.get('size', float('inf')):
                        best_result = result

                # Adjust settings for next attempt - safely handle None case
                config = self._adjust_config(config, result.get(
                    'size', float('inf')), target_size_mb)

            # Use best result if we have one
            if best_result:
                return ProcessingResult(
                    best_result.get('fps', 0),
                    best_result.get('size', 0),
                    best_result.get('path'),
                    ProcessingStatus.SUCCESS,
                    settings=config
                )

            return ProcessingResult(0, original_size, None, ProcessingStatus.OPTIMIZATION_ERROR)

        except Exception as e:
            self.dev_logger.error(f"Progressive optimization failed: {str(e)}")
            return ProcessingResult(0, original_size, None, ProcessingStatus.OPTIMIZATION_ERROR, str(e))

    def _analyze_source_colors(self, input_path: Path) -> int:
        """Analyze color distribution in the GIF with better preservation."""
        try:
            from PIL import Image
            with Image.open(input_path) as img:
                if img.mode != 'RGB':
                    img = img.convert('RGB')

                colors = img.getcolors(maxcolors=257)
                if not colors:  # More than 256 colors
                    return 256  # Always try to preserve maximum colors

                total_pixels = img.size[0] * img.size[1]
                threshold = total_pixels * 0.0005  # Reduced threshold for better preservation
                significant_colors = sum(
                    1 for count, _ in colors if count > threshold)

                # More generous color preservation
                if significant_colors > 128:
                    return 256
                elif significant_colors > 64:
                    return 192
                else:
                    return 128  # Minimum color count increased

        except Exception as e:
            self.dev_logger.warning(
                f"Color analysis failed: {e}, using maximum colors")
            return 256  # Default to maximum colors on error

    def _adjust_config(self, config: OptimizationConfig, current_size: float, target_size: float) -> OptimizationConfig:
        """Adjust optimization config based on results."""
        if current_size == float('inf'):
            # If previous attempt failed, be more aggressive
            return OptimizationConfig(
                scale_factor=max(0.5, config.scale_factor * 0.8),
                colors=min(config.colors, 128),
                lossy_value=min(100, config.lossy_value + 20)
            )

        size_ratio = current_size / target_size
        if size_ratio > 2:
            # Far from target, be aggressive
            return OptimizationConfig(
                scale_factor=max(0.4, config.scale_factor * 0.7),
                colors=max(64, config.colors - 64),
                lossy_value=min(100, config.lossy_value + 30)
            )
        elif size_ratio > 1.5:
            # Getting closer, be more careful
            return OptimizationConfig(
                scale_factor=max(0.6, config.scale_factor * 0.85),
                colors=max(128, config.colors - 32),
                lossy_value=min(100, config.lossy_value + 15)
            )
        else:
            # Fine tuning
            return OptimizationConfig(
                scale_factor=max(0.7, config.scale_factor * 0.9),
                colors=max(128, config.colors - 16),
                lossy_value=min(100, config.lossy_value + 10)
            )


@asynccontextmanager
async def managed_process_pool(max_workers: int = None):
    pool = ThreadPoolExecutor(max_workers=max_workers)
    try:
        yield pool
    finally:
        pool.shutdown(wait=True)

    def process_file(self, input_path: Path, output_path: Path, is_video: bool = False) -> bool:
        try:
            if not input_path.exists():
                logging.error(f"Input file does not exist: {input_path}")
                return False

            if is_video:
                # Convert video to GIF first
                temp_gif = self.temp_manager.get_temp_path('.gif')
                if not self.convert_video_to_gif(input_path, temp_gif):
                    return False
                input_path = temp_gif

            # Validate GIF before processing
            if not self.validate_gif(input_path):
                logging.error(f"Invalid or corrupted GIF file: {input_path}")
                return False

            # Process the GIF with error handling
            result = self.optimize_gif(input_path, output_path)

            if not result:
                logging.error(f"Failed to optimize GIF: {input_path}")
                return False

            return True

        except Exception as e:
            logging.error(f"Error processing file {input_path}: {str(e)}")
            return False

    def validate_gif(self, gif_path: Path) -> bool:
        try:
            # Basic validation using PIL
            with Image.open(gif_path) as img:
                return img.format == 'GIF' and getattr(img, 'is_animated', False)
        except Exception as e:
            logging.error(f"GIF validation failed for {gif_path}: {str(e)}")
            return False
