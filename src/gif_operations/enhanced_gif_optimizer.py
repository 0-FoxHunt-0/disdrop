import asyncio
import logging
import os
import shutil
import tempfile
import time
import uuid
import concurrent.futures
import subprocess
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Set, Callable
import threading
import re
import hashlib
import contextlib
import json
import gc

from src.default_config import TEMP_FILE_DIR
from src.gif_operations.base_processor import BaseProcessor
from src.gif_operations.file_processor import FileProcessor
from src.gif_operations.optimizers.adaptive_gif_optimizer import AdaptiveGIFOptimizer, OptimizationResult
from src.logging_system import ModernLogStyle, log_gif_progress, display_progress_update, create_live_progress_bar, ICONS

# Configure logger
logger = logging.getLogger(__name__)


class TempFileManager:
    """Context manager for handling temporary files in GIF optimization."""

    def __init__(self, prefix="gif_opt_", suffix=".gif", dir=TEMP_FILE_DIR):
        self.prefix = prefix
        self.suffix = suffix
        self.dir = Path(dir)
        self.temp_files = set()
        os.makedirs(self.dir, exist_ok=True)

    def create_temp_file(self, custom_prefix=None, custom_suffix=None) -> Path:
        """Create a new temporary file path and register it for cleanup"""
        prefix = custom_prefix or self.prefix
        suffix = custom_suffix or self.suffix
        temp_file = self.dir / f"{prefix}_{uuid.uuid4().hex}{suffix}"
        self.temp_files.add(temp_file)
        return temp_file

    def register_file(self, file_path: Path) -> None:
        """Register an existing file for cleanup"""
        self.temp_files.add(Path(file_path))

    def remove_from_tracking(self, file_path: Path) -> None:
        """Stop tracking a file (e.g., if it should be preserved)"""
        if file_path in self.temp_files:
            self.temp_files.remove(file_path)

    def cleanup(self, exclude: Set[Path] = None) -> None:
        """Remove all tracked temporary files except those in exclude set"""
        exclude = exclude or set()
        for file_path in self.temp_files:
            if file_path not in exclude and file_path.exists():
                try:
                    file_path.unlink()
                except Exception as e:
                    logger.warning(
                        f"Failed to remove temp file {file_path}: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()


class ProgressTracker:
    """Tracks and reports progress during GIF optimization."""

    def __init__(self, total_stages: int = 1, file_name: str = ""):
        self.total_stages = max(1, total_stages)
        self.current_stage = 0
        self.stage_progress = 0.0  # 0-100 for current stage
        self.file_name = file_name
        self.stage_names = {}
        self.start_time = time.time()
        self.stage_start_time = time.time()

    def set_total_stages(self, total_stages: int):
        """Set the total number of stages in the process."""
        self.total_stages = max(1, total_stages)

    def set_file_name(self, file_name: str):
        """Set the file name being processed."""
        self.file_name = file_name

    def set_stage_names(self, stage_names: Dict[int, str]):
        """Set descriptive names for each stage."""
        self.stage_names = stage_names

    def start_stage(self, stage: int, stage_name: str = ""):
        """Start a new stage in the process."""
        self.current_stage = stage
        self.stage_progress = 0.0
        self.stage_start_time = time.time()

        if stage_name:
            self.stage_names[stage] = stage_name

        self._report_progress()

    def update_progress(self, progress: float, message: str = ""):
        """Update progress for the current stage (0-100)."""
        self.stage_progress = min(100.0, max(0.0, progress))
        self._report_progress(message)

    def complete_stage(self, success: bool = True, message: str = ""):
        """Mark the current stage as complete."""
        self.stage_progress = 100.0
        self._report_progress(
            message, status="success" if success else "warning")

    def finish(self, success: bool = True, message: str = ""):
        """Mark the whole process as finished."""
        elapsed = time.time() - self.start_time

        status = "success" if success else "error"
        if not message:
            message = f"Process completed in {elapsed:.1f}s" if success else "Process failed"

        log_gif_progress(message, status)

    def _report_progress(self, message: str = "", status: str = "processing"):
        """Report current progress."""
        # Calculate overall progress
        stage_weight = 1.0 / self.total_stages
        overall_progress = ((self.current_stage - 1) * stage_weight +
                            (self.stage_progress / 100.0) * stage_weight) * 100.0

        # Get stage name if available
        stage_name = self.stage_names.get(
            self.current_stage, f"Stage {self.current_stage}")

        # Format progress message
        if not message:
            message = f"{stage_name}: {self.stage_progress:.0f}%"
        else:
            message = f"{stage_name}: {message}"

        # Add file name if available
        if self.file_name:
            message = f"{self.file_name} - {message}"

        # Add overall progress if multiple stages
        if self.total_stages > 1:
            message = f"{message} (Overall: {overall_progress:.0f}%)"

        # Log the progress
        log_gif_progress(message, status)


class EnhancedGIFOptimizer(BaseProcessor):
    """Enhanced GIF optimizer with perceptual quality preservation.

    This optimizer replaces the previous implementation with a completely new approach:
    1. Advanced content analysis for intelligent parameter selection
    2. Perceptual quality metrics to maintain visual fidelity
    3. Progressive optimization with quality-size tradeoff awareness
    4. Parallel processing for better performance
    5. Improved error handling and recovery
    6. LRU cache for optimization results to avoid redundant work
    """

    def __init__(self, max_workers: int = 4, compression_settings: Dict = None):
        """Initialize the enhanced GIF optimizer.

        Args:
            max_workers: Maximum number of worker threads for parallel processing
            compression_settings: Optional compression settings dictionary
        """
        super().__init__()
        self.max_workers = max_workers
        self.compression_settings = compression_settings or {}

        # Initialize core components
        try:
            self.adaptive_optimizer = AdaptiveGIFOptimizer(
                max_workers=max_workers)
        except Exception as e:
            logger.error(f"Failed to initialize AdaptiveGIFOptimizer: {e}")
            # Create a fallback optimizer
            self.adaptive_optimizer = None

        self.file_processor = FileProcessor()

        # Create thread pool for parallel operations
        self.executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers)

        # Initialize optimization result cache (filename hash -> result)
        self.optimization_cache = {}
        self.cache_lock = threading.RLock()

        # Initialize counters for optimization statistics
        self.stats = {
            'total_processed': 0,
            'cache_hits': 0,
            'successful_optimizations': 0,
            'failed_optimizations': 0,
            'bytes_saved': 0,
            'total_time': 0
        }
        self.stats_lock = threading.RLock()

        # Create temp directory if it doesn't exist
        os.makedirs(TEMP_FILE_DIR, exist_ok=True)

        # Track overall batch progress
        self.batch_progress = {
            'current': 0,
            'total': 0,
            'start_time': 0,
            'current_file': '',
            'current_stage': '',
            'error_count': 0
        }
        self.batch_progress_lock = threading.RLock()

        logger.info(
            f"Enhanced GIF Optimizer initialized with {max_workers} workers")

    def optimize_gif(self, input_path: Path, output_path: Path, target_size_mb: float = None) -> Tuple[float, bool]:
        """Main method to optimize a GIF file.

        Args:
            input_path: Path to the input GIF file
            output_path: Path where the optimized GIF should be saved
            target_size_mb: Target size in megabytes (optional)

        Returns:
            Tuple containing (final_size_mb, success)
        """
        start_time = time.time()

        # Update file being processed in batch progress tracker
        with self.batch_progress_lock:
            self.batch_progress['current_file'] = input_path.name
            self.batch_progress['current_stage'] = 'Initialization'
            display_progress_update(
                0, 100,
                description=f"Optimizing GIF",
                file_name=input_path.name,
                status="PROCESSING",
                start_time=start_time,
                is_new_line=True
            )

        # Create progress tracker for detailed progress reporting
        progress = ProgressTracker(total_stages=5, file_name=input_path.name)
        progress.set_stage_names({
            1: "Initialization",
            2: "Analysis",
            3: "Pre-optimization",
            4: "Main optimization",
            5: "Post-processing"
        })

        # Start initialization stage
        progress.start_stage(1, "Initialization")

        # Print additional output for better visibility
        print(f"[*] Starting optimization of {input_path.name}")

        try:
            # Log optimization start
            logger.info(f"Starting enhanced GIF optimization for {input_path}")
            log_gif_progress(
                f"Starting optimization of {input_path.name} (target: {target_size_mb:.2f}MB)", "starting")

            # Validate input
            if not input_path.exists():
                logger.error(f"Input file does not exist: {input_path}")
                log_gif_progress(
                    f"Input file does not exist: {input_path.name}", "error")

                # Update progress display
                with self.batch_progress_lock:
                    self.batch_progress['current_stage'] = 'Error'
                    self.batch_progress['error_count'] += 1
                    display_progress_update(
                        100, 100,
                        description=f"Failed - File not found",
                        file_name=input_path.name,
                        status="ERROR",
                        start_time=start_time
                    )

                return 0, False

            # Show progress update
            with self.batch_progress_lock:
                self.batch_progress['current_stage'] = 'Size Analysis'
                display_progress_update(
                    10, 100,
                    description=f"Analyzing file size",
                    file_name=input_path.name,
                    status="PROCESSING",
                    start_time=start_time
                )

            # Use configured size if not specified
            if target_size_mb is None:
                target_size_mb = self.compression_settings.get(
                    'target_size_mb', 10)
                logger.info(f"Using default target size: {target_size_mb}MB")

            # Get original size
            original_size = self.file_processor.get_file_size(input_path)
            logger.info(f"Original file size: {original_size:.2f}MB")

            # Progress update after size check
            with self.batch_progress_lock:
                display_progress_update(
                    20, 100,
                    description=f"Original size: {original_size:.2f}MB",
                    file_name=input_path.name,
                    status="PROCESSING",
                    start_time=start_time
                )

            # Skip optimization if already under target size
            if original_size <= target_size_mb:
                logger.info(
                    f"GIF already under target size: {original_size:.2f}MB (target: {target_size_mb:.2f}MB)")
                log_gif_progress(
                    f"File already below target size ({original_size:.2f}MB)", "skipped")
                shutil.copy2(input_path, output_path)

                # Extra user-visible output
                print(
                    f"[+] File {input_path.name} already meets size requirements ({original_size:.2f}MB)")

                # Final progress update
                with self.batch_progress_lock:
                    display_progress_update(
                        100, 100,
                        description=f"Already optimized",
                        file_name=input_path.name,
                        status="SUCCESS",
                        start_time=start_time
                    )

                with self.stats_lock:
                    self.stats['total_processed'] += 1
                    self.stats['successful_optimizations'] += 1
                    elapsed = time.time() - start_time
                    self.stats['total_time'] += elapsed

                return original_size, True

            # For very large files, try a special staged approach
            # This is for extremely large GIFs (>100MB)
            if original_size > 100:
                logger.info(
                    f"Extremely large file detected ({original_size:.2f}MB). Using staged optimization approach.")
                log_gif_progress(
                    f"Extremely large file detected. Using staged optimization...", "processing")

                # Update progress display for large file
                with self.batch_progress_lock:
                    self.batch_progress['current_stage'] = 'Large File Optimization'
                    display_progress_update(
                        25, 100,
                        description=f"Large file detected - Special optimization",
                        file_name=input_path.name,
                        status="PROCESSING",
                        start_time=start_time
                    )

                return self._optimize_extremely_large_gif(input_path, output_path, target_size_mb, original_size, start_time)

            # For large files, try parallel processing approach
            # Only for really big files (>20MB)
            if original_size > target_size_mb * 5 and original_size > 20:
                logger.info(
                    f"Large file detected ({original_size:.2f}MB). Trying parallel optimization approach.")
                log_gif_progress(
                    f"Large file detected. Trying parallel optimization...", "processing")

                parallel_result = self._optimize_in_parallel(
                    input_path, output_path, target_size_mb)

                if parallel_result.success and parallel_result.size_mb <= target_size_mb:
                    logger.info(
                        f"Parallel optimization successful: {original_size:.2f}MB → {parallel_result.size_mb:.2f}MB")
                    log_gif_progress(
                        f"Parallel optimization successful: {parallel_result.size_mb:.2f}MB",
                        "success"
                    )

                    # Update statistics
                    with self.stats_lock:
                        self.stats['total_processed'] += 1
                        self.stats['successful_optimizations'] += 1
                        self.stats['bytes_saved'] += (
                            original_size - parallel_result.size_mb) * 1024 * 1024
                        elapsed = time.time() - start_time
                        self.stats['total_time'] += elapsed

                    return parallel_result.size_mb, True
                else:
                    logger.info(
                        "Parallel optimization didn't meet target size or failed. Falling back to standard approach.")

            # Check if adaptive optimizer is available
            if self.adaptive_optimizer is None:
                logger.warning(
                    "AdaptiveGIFOptimizer not available, using fallback optimization")
                log_gif_progress(
                    "Using fallback optimization method", "processing")

                # Simple fallback optimization using gifsicle directly
                return self._apply_fallback_optimization(input_path, output_path, target_size_mb, original_size, start_time)

            # Check if we have a cached result for this file
            file_hash = self._get_file_hash(input_path)
            cached_result = self._get_cached_result(file_hash, target_size_mb)

            if cached_result and cached_result.get('success', False):
                log_gif_progress(
                    f"Using cached optimization result", "processing")
                cached_path = cached_result.get('path')

                if cached_path and os.path.exists(cached_path):
                    shutil.copy2(cached_path, output_path)
                    final_size = self.file_processor.get_file_size(output_path)

                    log_gif_progress(
                        f"Optimization complete (cached): {final_size:.2f}MB", "success")
                    logger.info(
                        f"Used cached result for {input_path.name}: {final_size:.2f}MB")

                    with self.stats_lock:
                        self.stats['total_processed'] += 1
                        self.stats['cache_hits'] += 1
                        self.stats['successful_optimizations'] += 1
                        self.stats['bytes_saved'] += (
                            original_size - final_size) * 1024 * 1024
                        elapsed = time.time() - start_time
                        self.stats['total_time'] += elapsed

                    return final_size, True

            # Create event loop for async operation
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            try:
                # Run the optimization process
                log_gif_progress(
                    f"Running adaptive GIF optimization", "processing")

                # Apply pre-optimization to reduce file size before main optimization
                pre_optimized_path = None
                if original_size > target_size_mb * 3:  # Only pre-optimize if significantly larger
                    log_gif_progress(
                        f"Applying preliminary optimization...", "processing")
                    pre_optimized_path = self._apply_preliminary_optimization(
                        input_path)

                    if pre_optimized_path and os.path.exists(pre_optimized_path):
                        pre_size = self.file_processor.get_file_size(
                            pre_optimized_path)
                        logger.info(
                            f"Preliminary optimization: {original_size:.2f}MB → {pre_size:.2f}MB")
                        input_for_optimizer = pre_optimized_path
                    else:
                        input_for_optimizer = input_path
                else:
                    input_for_optimizer = input_path

                # Run the main optimization with possible preflight input
                optimization_result = loop.run_until_complete(
                    self.adaptive_optimizer.optimize(
                        input_for_optimizer, output_path, target_size_mb
                    )
                )

                # If the adaptive optimizer didn't meet the target size, try increasingly aggressive optimization
                if optimization_result.success and optimization_result.size_mb > target_size_mb:
                    log_gif_progress(
                        f"Initial optimization didn't meet target size. Applying adaptive optimization...",
                        "processing"
                    )

                    # Calculate how far we are from the target size
                    current_size = optimization_result.size_mb
                    size_ratio = current_size / target_size_mb
                    reduction_needed = (
                        current_size - target_size_mb) / current_size

                    logger.info(f"Current size: {current_size:.2f}MB, Target: {target_size_mb:.2f}MB, "
                                f"Reduction needed: {reduction_needed:.2%}")

                    # Set up for adaptive optimization
                    temp_input = output_path
                    temp_output = Path(TEMP_FILE_DIR) / \
                        f"adaptive_{uuid.uuid4().hex}.gif"

                    # Analyze the GIF to determine its complexity
                    try:
                        # Use simple subprocess to get basic GIF info
                        info_cmd = ['gifsicle', '--info', str(temp_input)]
                        gif_info = subprocess.run(
                            info_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=False
                        ).stdout

                        # Extract frame count and color count if possible
                        frame_count = 1  # Default
                        if "images" in gif_info:
                            frame_match = gif_info.split(
                                "images")[0].strip().split()[-1]
                            try:
                                frame_count = int(frame_match)
                            except (ValueError, IndexError):
                                pass

                        color_count = 256  # Default to max
                        if "colors" in gif_info:
                            color_match = gif_info.split(
                                "colors")[0].strip().split()[-1]
                            try:
                                color_count = int(color_match)
                            except (ValueError, IndexError):
                                pass

                        # Enhanced analysis: check for color gradients and smooth transitions
                        # These benefit from dithering to maintain visual quality
                        has_gradients = False
                        sample_cmd = ['gifsicle', '--explode',
                                      '--output', TEMP_FILE_DIR, str(temp_input)]

                        try:
                            # Extract a sample of frames for analysis
                            sample_dir = Path(TEMP_FILE_DIR) / \
                                f"sample_{uuid.uuid4().hex}"
                            os.makedirs(sample_dir, exist_ok=True)

                            # Extract a subset of frames for analysis
                            # Analyze up to 5 frames
                            max_sample_frames = min(5, frame_count)
                            if frame_count > 1:
                                frame_indices = [
                                    int(i * (frame_count / max_sample_frames)) for i in range(max_sample_frames)]
                                sample_cmd = [
                                    'gifsicle',
                                    '--unoptimize',  # Ensure we can access full frame data
                                    str(temp_input)
                                ]

                                # Add specific frames to extract
                                for idx in frame_indices:
                                    sample_cmd.extend(['--select', f'#{idx}'])

                                sample_cmd.extend(
                                    ['--output', str(sample_dir / 'frame%03d.gif')])

                                subprocess.run(
                                    sample_cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    check=False
                                )

                                # Check extracted samples for gradient-like patterns
                                sample_files = list(sample_dir.glob('*.gif'))
                                if sample_files:
                                    # Simple heuristic: if a frame has many colors but few unique colors,
                                    # it likely contains gradients or smooth transitions
                                    # Check first few samples
                                    for sample in sample_files[:3]:
                                        sample_info = subprocess.run(
                                            ['gifsicle', '--info',
                                                str(sample)],
                                            stdout=subprocess.PIPE,
                                            stderr=subprocess.PIPE,
                                            text=True,
                                            check=False
                                        ).stdout

                                        if "colors" in sample_info:
                                            try:
                                                sample_colors = int(sample_info.split(
                                                    "colors")[0].strip().split()[-1])
                                                if sample_colors > 64:  # Frames with many colors may benefit from dithering
                                                    has_gradients = True
                                                    break
                                            except (ValueError, IndexError):
                                                pass

                            # Clean up sample directory
                            shutil.rmtree(sample_dir, ignore_errors=True)
                        except Exception as e:
                            logger.warning(
                                f"Enhanced frame analysis failed: {e}")
                            # Fall back to basic heuristic
                            has_gradients = color_count > 128

                        logger.info(
                            f"GIF analysis: ~{frame_count} frames, ~{color_count} colors, gradients: {has_gradients}")
                    except Exception as e:
                        logger.warning(
                            f"Failed to analyze GIF complexity: {e}")
                        frame_count = 1
                        color_count = 256
                        has_gradients = False

                    # Dynamic optimization strategy based on file characteristics and reduction needed
                    max_attempts = 8
                    attempt = 0
                    last_size = current_size

                    # Initial parameters are based on the reduction needed
                    # MODIFIED: Increased minimum colors to prevent black artifacts
                    colors = min(
                        256, max(64, int(color_count * (1 - reduction_needed))))
                    # MODIFIED: Decreased maximum lossy to prevent artifacts
                    lossy = min(80, int(reduction_needed * 150))
                    scale_factor = min(
                        1.0, max(0.5, 1.0 - (reduction_needed * 1.2)))

                    # Determine if dithering should be applied based on content analysis
                    # MODIFIED: Changed to ordered dithering which produces fewer black artifacts
                    use_dithering = has_gradients and colors < 192
                    dither_type = 'ordered'  # Changed from floyd-steinberg to ordered

                    # For animations with many frames, try frame reduction first
                    try_frame_reduction = frame_count > 10 and reduction_needed > 0.5
                    frame_skip = 0  # Default: don't skip frames

                    if try_frame_reduction:
                        # Calculate frame skip rate based on reduction needed
                        # Higher reduction needed = higher skip rate
                        frame_skip = max(0, min(2, int(reduction_needed * 4)))
                        if frame_skip > 0:
                            logger.info(
                                f"Will try frame reduction with skip rate: {frame_skip}")

                    # Adaptive optimization loop
                    while attempt < max_attempts and last_size > target_size_mb:
                        # Adjust parameters based on progress so far
                        if attempt > 0:
                            # Calculate how well we're progressing
                            progress_ratio = (
                                current_size - last_size) / current_size

                            # If we're making good progress, make smaller adjustments
                            # If poor progress, make more aggressive adjustments
                            if progress_ratio < 0.05:  # Less than 5% improvement
                                # More aggressive changes, but still preserving quality
                                colors = max(64, int(colors * 0.8))
                                lossy = min(80, lossy + 20)
                                scale_factor = max(0.5, scale_factor - 0.1)

                                # Try changing dither method if we're not making progress
                                if attempt % 2 == 1:
                                    if dither_type == 'ordered':
                                        dither_type = None  # Try no dithering
                                        use_dithering = False
                                    else:
                                        dither_type = 'ordered'
                                        use_dithering = True

                                # Try frame reduction for animations if we're struggling
                                if frame_count > 5 and attempt > 3 and frame_skip < 2:
                                    frame_skip += 1
                            else:
                                # More moderate changes
                                colors = max(64, int(colors * 0.9))
                                lossy = min(80, lossy + 15)
                                scale_factor = max(0.5, scale_factor - 0.05)

                        # Log the adaptive settings
                        dither_status = f"with {dither_type} dithering" if use_dithering else "no dithering"
                        frame_status = f", frame skip: {frame_skip}" if frame_skip > 0 else ""
                        logger.info(f"Adaptive optimization attempt {attempt+1}: colors={colors}, "
                                    f"lossy={lossy}, scale={scale_factor:.2f}, {dither_status}{frame_status}")
                        log_gif_progress(
                            f"Trying adaptive settings #{attempt+1}: quality ~{max(0, min(100, int(100 - (lossy/2))))}/100 {dither_status}",
                            "processing"
                        )

                        # Apply the adaptive settings
                        cmd = [
                            'gifsicle',
                            '--optimize=3',
                            f'--colors={colors}',
                            f'--lossy={lossy}',
                        ]

                        # Apply dithering when appropriate
                        if use_dithering and dither_type:
                            cmd.append(f'--dither={dither_type}')

                        # Apply scaling if needed
                        if scale_factor < 0.99:
                            cmd.extend(['--scale=' + f"{scale_factor:.2f}"])

                        # Apply frame reduction for animations if needed - fix the syntax for skip parameter
                        if frame_skip > 0 and frame_count > 5:
                            # Create a new approach using the correct gifsicle syntax
                            # First extract every nth frame based on skip rate
                            temp_skip_file = Path(
                                TEMP_FILE_DIR) / f"skip_frames_{uuid.uuid4().hex}.gif"

                            # Build a command that selects only every (frame_skip+1)th frame
                            skip_cmd = ['gifsicle', '--unoptimize']

                            # Add frame selection pattern based on frame skip
                            frame_indices = range(
                                0, frame_count, frame_skip + 1)
                            for idx in frame_indices:
                                skip_cmd.extend(['--select', f'#{idx}'])

                            skip_cmd.extend(
                                [str(temp_input), '-o', str(temp_skip_file)])

                            try:
                                logger.info(
                                    f"Applying frame skip with rate {frame_skip} (selecting every {frame_skip+1}th frame)")
                                process = subprocess.run(
                                    skip_cmd,
                                    stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE,
                                    text=True,
                                    check=False
                                )

                                if process.returncode == 0 and temp_skip_file.exists():
                                    # Use the skipped file as input for further optimization
                                    temp_input = temp_skip_file
                                else:
                                    logger.warning(
                                        f"Frame skipping failed: {process.stderr} - continuing with all frames")
                            except Exception as e:
                                logger.warning(
                                    f"Error during frame skipping: {e} - continuing with all frames")

                        cmd.extend([str(temp_input), '-o', str(temp_output)])

                        try:
                            process = subprocess.run(
                                cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=False
                            )

                            if process.returncode == 0 and temp_output.exists():
                                current_size = self.file_processor.get_file_size(
                                    temp_output)

                                # Check if we've reached the target size
                                if current_size <= target_size_mb:
                                    shutil.copy2(temp_output, output_path)
                                    quality_factor = max(
                                        0.1, min(1.0, (colors/256) * (1 - lossy/200) * scale_factor))

                                    # Decrease quality score slightly if we had to skip frames
                                    if frame_skip > 0:
                                        quality_factor *= max(0.5,
                                                              1.0 - (frame_skip * 0.15))

                                    # Increase quality score slightly if we used dithering
                                    if use_dithering and colors < 128:
                                        quality_factor = min(
                                            1.0, quality_factor * 1.1)

                                    # Create a mutable optimization result
                                    optimization_result = OptimizationResult(
                                        success=True,
                                        file_path=output_path,
                                        size_mb=current_size,
                                        # Approximate quality based on parameters
                                        quality_score=quality_factor,
                                        settings={
                                            "colors": colors,
                                            "lossy": lossy,
                                            "scale": scale_factor,
                                            "dithering": use_dithering,
                                            "frame_skip": frame_skip,
                                            "adaptive_attempt": attempt+1
                                        },
                                        error=None
                                    )
                                    logger.info(
                                        f"Target size reached with adaptive optimization attempt {attempt+1}")
                                    log_gif_progress(
                                        f"Target size reached with adaptive settings #{attempt+1}",
                                        "success"
                                    )
                                    break
                                else:
                                    # Track progress and continue with updated input
                                    logger.info(
                                        f"Attempt {attempt+1} result: {current_size:.2f}MB")
                                    last_size = current_size
                                    shutil.copy2(temp_output, temp_input)
                            else:
                                logger.error(
                                    f"Adaptive optimization attempt {attempt+1} failed: {process.stderr}")
                        except Exception as e:
                            logger.error(
                                f"Error in adaptive optimization attempt {attempt+1}: {e}")

                        attempt += 1

                    # Clean up temp file
                    if temp_output.exists():
                        try:
                            temp_output.unlink()
                        except Exception:
                            pass

                    # If we still haven't reached the target after adaptive attempts,
                    # but we've made significant progress, use our best result
                    if last_size > target_size_mb:
                        logger.info(f"Adaptive optimization improved file size but did not reach target: "
                                    f"{original_size:.2f}MB → {last_size:.2f}MB (target: {target_size_mb:.2f}MB)")

                        # No longer consider "near successes" as successes - all files must be under target size
                        log_gif_progress(
                            f"Failed to reach target size: {last_size:.2f}MB > {target_size_mb:.2f}MB",
                            "error"
                        )

                        # Create a mutable optimization result instead of modifying an existing one
                        optimization_result = OptimizationResult(
                            success=False,
                            file_path=None,
                            size_mb=last_size,
                            quality_score=0.0,
                            settings={},
                            error=f"Failed to reach target size after {max_attempts} optimization attempts"
                        )

            finally:
                # Clean up the event loop
                loop.close()

            # Handle the result
            if optimization_result.success and optimization_result.file_path:
                final_size = optimization_result.size_mb

                # Only consider it successful if it's actually under target size
                if final_size > target_size_mb:
                    # Create a new result object instead of modifying the existing one
                    optimization_result = OptimizationResult(
                        success=False,
                        file_path=optimization_result.file_path,
                        size_mb=final_size,
                        quality_score=optimization_result.quality_score,
                        settings=optimization_result.settings,
                        error="Size exceeds target after optimization"
                    )

                    # We should have caught this already, but just in case
                    logger.warning(
                        f"File incorrectly marked as success despite size {final_size:.2f}MB > {target_size_mb:.2f}MB")
                    log_gif_progress(
                        f"Failed to reach target size: {final_size:.2f}MB > {target_size_mb:.2f}MB",
                        "error"
                    )

                    # Try one last resort optimization before giving up
                    log_gif_progress(
                        f"Target size not reached. Applying last resort optimization...",
                        "warning"
                    )

                    # Create a more sophisticated last resort approach with multiple techniques
                    final_attempt_path = Path(
                        TEMP_FILE_DIR) / f"last_resort_{uuid.uuid4().hex}.gif"

                    # Try a series of advanced techniques to reach target size
                    last_resort_techniques = [
                        {
                            "name": "aggressive_lossy",
                            "cmd": [
                                'gifsicle',
                                '--optimize=3',
                                '--colors=64',  # Increased from 16 to reduce black artifacts
                                '--lossy=80',   # Reduced from 200 to reduce artifacts
                                '--scale=0.5',
                                '--dither=ordered',  # Changed to ordered dithering for fewer artifacts
                                '--no-warnings',
                                str(output_path),
                                '-o', str(final_attempt_path)
                            ]
                        },
                        {
                            "name": "frame_deduplication",
                            "cmd": [
                                'gifsicle',
                                '--optimize=3',
                                '--colors=80',  # Increased from 32 to reduce black artifacts
                                '--lossy=70',   # Reduced from 150 to reduce artifacts
                                '--scale=0.6',
                                '--dither=ordered',  # Changed to ordered dithering
                                '--no-warnings',
                                # Try to merge similar frames to reduce size
                                '--merge',
                                # Deduplication of identical frames
                                '--unoptimize', '--optimize=3',
                                str(output_path),
                                '-o', str(final_attempt_path)
                            ]
                        },
                        {
                            "name": "disposal_and_crop",
                            "cmd": [
                                'gifsicle',
                                '--optimize=3',
                                '--colors=80',  # Increased from 32 to reduce black artifacts
                                '--lossy=70',   # Reduced from 150 to reduce artifacts
                                '--scale=0.6',
                                '--dither=ordered',  # Changed to ordered dithering
                                '--no-warnings',
                                # Change disposal method to reduce size
                                '--disposal=bg',
                                # Try to crop any transparent borders
                                '--crop-transparency',
                                # Remove interlacing to save space
                                '--no-interlace',
                                str(output_path),
                                '-o', str(final_attempt_path)
                            ]
                        },
                        {
                            "name": "extreme_temporal",
                            "cmd": [
                                'gifsicle',
                                '--optimize=3',
                                '--colors=96',  # Increased from 24 to reduce black artifacts
                                '--lossy=70',   # Reduced from 150 to reduce artifacts
                                '--scale=0.5',
                                '--dither=ordered',  # Changed to ordered dithering
                                '--no-warnings',
                                # Skip every other frame to drastically reduce size - using proper syntax
                                '--unoptimize',  # Need to unoptimize to access all frames
                                # Using the proper frame selection approach instead of --skip
                                # Build a select pattern for every 2nd frame
                            ]
                        },
                        {
                            "name": "absolute_minimum",
                            "cmd": [
                                'gifsicle',
                                '--optimize=3',
                                '--colors=32',  # Increased from 8 to prevent severe black artifacts
                                '--lossy=80',   # Reduced from 200 to reduce artifacts
                                '--scale=0.4',
                                '--dither=ordered',  # Changed to ordered dithering
                                # Most aggressive settings as a last resort
                                '--no-interlace',
                                '--disposal=bg',
                                '--no-warnings',
                                str(output_path),
                                '-o', str(final_attempt_path)
                            ]
                        }
                    ]

                    # For the extreme_temporal technique, we need to add frame selection
                    # Get approx frame count for selection
                    try:
                        frame_info_cmd = ['gifsicle',
                                          '--info', str(output_path)]
                        frame_info = subprocess.run(
                            frame_info_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=False
                        ).stdout

                        frame_count = 0
                        if "images" in frame_info:
                            try:
                                frame_match = frame_info.split(
                                    "images")[0].strip().split()[-1]
                                frame_count = int(frame_match)
                            except (ValueError, IndexError):
                                frame_count = 100  # Default if we can't detect
                        else:
                            frame_count = 100  # Default if we can't detect

                        # Add selection for every 2nd frame
                        for i in range(0, frame_count, 2):
                            last_resort_techniques[3]["cmd"].extend(
                                ['--select', f'#{i}'])
                    except Exception as e:
                        logger.warning(f"Error setting frame selection: {e}")
                        # Add a reasonable default selection if we couldn't calculate
                        for i in range(0, 100, 2):
                            last_resort_techniques[3]["cmd"].extend(
                                ['--select', f'#{i}'])

                    # Add output paths to extreme_temporal technique
                    last_resort_techniques[3]["cmd"].extend(
                        [str(output_path), '-o', str(final_attempt_path)])

                    success = False
                    best_size = float('inf')
                    best_technique = None

                    # Try each technique until we find one that works
                    for technique in last_resort_techniques:
                        try:
                            logger.info(
                                f"Trying last resort technique: {technique['name']}")
                            log_gif_progress(
                                f"Trying {technique['name']} optimization...",
                                "processing"
                            )

                            # Remove any previous attempt
                            if final_attempt_path.exists():
                                final_attempt_path.unlink()

                            process = subprocess.run(
                                technique['cmd'],
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=False
                            )

                            if process.returncode == 0 and final_attempt_path.exists():
                                current_size = self.file_processor.get_file_size(
                                    final_attempt_path)
                                logger.info(
                                    f"Technique {technique['name']} result: {current_size:.2f}MB")

                                # Track the best result even if it's not below target
                                if current_size < best_size:
                                    best_size = current_size
                                    best_technique = technique['name']

                                # If we've reached target size, we can stop
                                if current_size <= target_size_mb:
                                    success = True
                                    logger.info(
                                        f"Target size reached with {technique['name']}")
                                    log_gif_progress(
                                        f"Target size reached with {technique['name']}: {current_size:.2f}MB",
                                        "success"
                                    )
                                    break
                        except Exception as e:
                            logger.error(
                                f"Error in {technique['name']} optimization: {e}")

                    # If any technique succeeded, copy the result
                    if success:
                        shutil.copy2(final_attempt_path, output_path)
                        final_size = best_size

                        # Update statistics for success
                        with self.stats_lock:
                            self.stats['total_processed'] += 1
                            self.stats['successful_optimizations'] += 1
                            self.stats['bytes_saved'] += (
                                original_size - final_size) * 1024 * 1024
                            elapsed = time.time() - start_time
                            self.stats['total_time'] += elapsed

                        return final_size, True
                    else:
                        # If we have a best technique that made progress but didn't hit target
                        if best_technique and best_size < float('inf'):
                            logger.warning(
                                f"Best last resort technique ({best_technique}) improved but didn't reach target: {best_size:.2f}MB > {target_size_mb:.2f}MB")

                        # Absolute failure - delete the output file and report error
                        log_gif_progress(
                            f"FAILED: Unable to reach target size even with extreme optimization: {best_size:.2f}MB > {target_size_mb:.2f}MB",
                            "error"
                        )

                        # Do not copy the file to output_path since it fails to meet requirements
                        if output_path.exists():
                            try:
                                output_path.unlink()
                            except Exception as e:
                                logger.error(
                                    f"Error removing failed output file: {e}")

                # Update cache with successful result
                self._cache_result(file_hash, target_size_mb, {
                    'success': True,
                    'path': str(output_path),
                    'size': final_size,
                    'settings': optimization_result.settings,
                    'quality': optimization_result.quality_score
                })

                # Final success message
                reduction_percent = (
                    (original_size - final_size) / original_size * 100) if original_size > 0 else 0
                logger.info(f"GIF optimization successful: {original_size:.2f}MB → {final_size:.2f}MB "
                            f"({reduction_percent:.1f}% reduction, quality: {optimization_result.quality_score:.2f})")

                # Log detailed settings
                logger.info(
                    f"Optimization settings: {optimization_result.settings}")

                log_gif_progress(
                    f"Optimization successful: {original_size:.2f}MB → {final_size:.2f}MB ({reduction_percent:.1f}% reduction)",
                    "success"
                )

                # Update statistics
                with self.stats_lock:
                    self.stats['total_processed'] += 1
                    self.stats['successful_optimizations'] += 1
                    self.stats['bytes_saved'] += (original_size -
                                                  final_size) * 1024 * 1024
                    elapsed = time.time() - start_time
                    self.stats['total_time'] += elapsed

                return final_size, True

            # If we get here, the optimization was not successful or the file is still too large
            # Try one last resort optimization before giving up
            log_gif_progress(
                f"Target size not reached. Applying last resort optimization...",
                "warning"
            )

            # Create a more sophisticated last resort approach with multiple techniques
            final_attempt_path = Path(
                TEMP_FILE_DIR) / f"last_resort_{uuid.uuid4().hex}.gif"

            # Try a series of advanced techniques to reach target size
            last_resort_techniques = [
                {
                    "name": "aggressive_lossy",
                    "cmd": [
                        'gifsicle',
                        '--optimize=3',
                        '--colors=64',  # Increased from 16 to reduce black artifacts
                        '--lossy=80',   # Reduced from 200 to reduce artifacts
                        '--scale=0.5',
                        '--dither=ordered',  # Changed to ordered dithering for fewer artifacts
                        '--no-warnings',
                        str(output_path),
                        '-o', str(final_attempt_path)
                    ]
                },
                {
                    "name": "frame_deduplication",
                    "cmd": [
                        'gifsicle',
                        '--optimize=3',
                        '--colors=80',  # Increased from 32 to reduce black artifacts
                        '--lossy=70',   # Reduced from 150 to reduce artifacts
                        '--scale=0.6',
                        '--dither=ordered',  # Changed to ordered dithering
                        '--no-warnings',
                        # Try to merge similar frames to reduce size
                        '--merge',
                        # Deduplication of identical frames
                        '--unoptimize', '--optimize=3',
                        str(output_path),
                        '-o', str(final_attempt_path)
                    ]
                },
                {
                    "name": "disposal_and_crop",
                    "cmd": [
                        'gifsicle',
                        '--optimize=3',
                        '--colors=80',  # Increased from 32 to reduce black artifacts
                        '--lossy=70',   # Reduced from 150 to reduce artifacts
                        '--scale=0.6',
                        '--dither=ordered',  # Changed to ordered dithering
                        '--no-warnings',
                        # Change disposal method to reduce size
                        '--disposal=bg',
                        # Try to crop any transparent borders
                        '--crop-transparency',
                        # Remove interlacing to save space
                        '--no-interlace',
                        str(output_path),
                        '-o', str(final_attempt_path)
                    ]
                },
                {
                    "name": "extreme_temporal",
                    "cmd": [
                        'gifsicle',
                        '--optimize=3',
                        '--colors=96',  # Increased from 24 to reduce black artifacts
                        '--lossy=70',   # Reduced from 150 to reduce artifacts
                        '--scale=0.5',
                        '--dither=ordered',  # Changed to ordered dithering
                        '--no-warnings',
                        # Skip every other frame to drastically reduce size - using proper syntax
                        '--unoptimize',  # Need to unoptimize to access all frames
                        # Using the proper frame selection approach instead of --skip
                        # Build a select pattern for every 2nd frame
                    ]
                },
                {
                    "name": "absolute_minimum",
                    "cmd": [
                        'gifsicle',
                        '--optimize=3',
                        '--colors=32',  # Increased from 8 to prevent severe black artifacts
                        '--lossy=80',   # Reduced from 200 to reduce artifacts
                        '--scale=0.4',
                        '--dither=ordered',  # Changed to ordered dithering
                        # Most aggressive settings as a last resort
                        '--no-interlace',
                        '--disposal=bg',
                        '--no-warnings',
                        str(output_path),
                        '-o', str(final_attempt_path)
                    ]
                }
            ]

            success = False
            best_size = float('inf')
            best_technique = None

            # Try each technique until we find one that works
            for technique in last_resort_techniques:
                try:
                    logger.info(
                        f"Trying last resort technique: {technique['name']}")
                    log_gif_progress(
                        f"Trying {technique['name']} optimization...",
                        "processing"
                    )

                    # Remove any previous attempt
                    if final_attempt_path.exists():
                        final_attempt_path.unlink()

                    process = subprocess.run(
                        technique['cmd'],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False
                    )

                    if process.returncode == 0 and final_attempt_path.exists():
                        current_size = self.file_processor.get_file_size(
                            final_attempt_path)
                        logger.info(
                            f"Technique {technique['name']} result: {current_size:.2f}MB")

                        # Track the best result even if it's not below target
                        if current_size < best_size:
                            best_size = current_size
                            best_technique = technique['name']

                        # If we've reached target size, we can stop
                        if current_size <= target_size_mb:
                            success = True
                            logger.info(
                                f"Target size reached with {technique['name']}")
                            log_gif_progress(
                                f"Target size reached with {technique['name']}: {current_size:.2f}MB",
                                "success"
                            )
                            break
                except Exception as e:
                    logger.error(
                        f"Error in {technique['name']} optimization: {e}")

            # If any technique succeeded, copy the result
            if success:
                shutil.copy2(final_attempt_path, output_path)
                final_size = best_size

                # Update statistics for success
                with self.stats_lock:
                    self.stats['total_processed'] += 1
                    self.stats['successful_optimizations'] += 1
                    self.stats['bytes_saved'] += (original_size -
                                                  final_size) * 1024 * 1024
                    elapsed = time.time() - start_time
                    self.stats['total_time'] += elapsed

                return final_size, True
            else:
                # If we have a best technique that made progress but didn't hit target
                if best_technique and best_size < float('inf'):
                    logger.warning(
                        f"Best last resort technique ({best_technique}) improved but didn't reach target: {best_size:.2f}MB > {target_size_mb:.2f}MB")

                # Absolute failure - delete the output file and report error
                log_gif_progress(
                    f"FAILED: Unable to reach target size even with extreme optimization: {best_size:.2f}MB > {target_size_mb:.2f}MB",
                    "error"
                )

                # Do not copy the file to output_path since it fails to meet requirements
                if output_path.exists():
                    try:
                        output_path.unlink()
                    except Exception as e:
                        logger.error(f"Error removing failed output file: {e}")

            # Clean up
            if final_attempt_path.exists():
                try:
                    final_attempt_path.unlink()
                except Exception:
                    pass

            # Update statistics for failure
            with self.stats_lock:
                self.stats['total_processed'] += 1
                self.stats['failed_optimizations'] += 1
                elapsed = time.time() - start_time
                self.stats['total_time'] += elapsed

            logger.error(
                f"GIF optimization failed: Unable to meet target size of {target_size_mb:.2f}MB")
            log_gif_progress(
                f"File optimization failed: Could not meet size target", "error")

            # Return failure
            return original_size, False
        except Exception as e:
            # Handle any unexpected errors
            logger.exception(f"Unexpected error in GIF optimization: {e}")
            log_gif_progress(f"Optimization error: {str(e)}", "error")

            # Do not copy the original file since it doesn't meet size requirements
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception as copy_error:
                    logger.error(
                        f"Failed to remove failed output file: {copy_error}")

            # Update statistics
            with self.stats_lock:
                self.stats['total_processed'] += 1
                self.stats['failed_optimizations'] += 1
                elapsed = time.time() - start_time
                self.stats['total_time'] += elapsed

            return self.file_processor.get_file_size(input_path), False

    def _apply_preliminary_optimization(self, input_path: Path) -> Optional[Path]:
        """Apply lightweight preliminary optimization to reduce file size for large GIFs

        Args:
            input_path: Path to the input GIF

        Returns:
            Path to pre-optimized file or None if failed
        """
        try:
            # Create a temporary output path
            pre_opt_path = Path(TEMP_FILE_DIR) / \
                f"preopt_{uuid.uuid4().hex}.gif"

            # Basic gifsicle command with moderate settings
            cmd = [
                'gifsicle',
                '--optimize=2',
                '--no-warnings',
                '--batch',
                str(input_path),
                '-o', str(pre_opt_path)
            ]

            # Run the command
            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )

            if process.returncode == 0 and pre_opt_path.exists():
                return pre_opt_path
            else:
                logger.warning(
                    f"Preliminary optimization failed: {process.stderr}")
                return None

        except Exception as e:
            logger.warning(f"Error in preliminary optimization: {e}")
            return None

    def optimize_gif_with_quality_target(self, input_path: Path, output_path: Path,
                                         quality_target: float = 0.8,
                                         max_size_mb: float = None) -> Tuple[float, bool]:
        """Optimize GIF targeting a specific quality level rather than size.

        Args:
            input_path: Path to the input GIF file
            output_path: Path where the optimized GIF should be saved
            quality_target: Target quality level (0.0-1.0) where 1.0 is perfect
            max_size_mb: Maximum size constraint (optional)

        Returns:
            Tuple containing (final_size_mb, success)
        """
        start_time = time.time()

        try:
            # Log the quality-based optimization
            logger.info(
                f"Starting quality-targeted optimization for {input_path} (quality: {quality_target:.2f})")
            log_gif_progress(
                f"Starting quality-targeted optimization (target: {quality_target:.2f})", "starting")

            # Validate input
            if not input_path.exists():
                logger.error(f"Input file does not exist: {input_path}")
                log_gif_progress(
                    f"Input file does not exist: {input_path.name}", "error")
                return 0, False

            # Get original size
            original_size = self.file_processor.get_file_size(input_path)

            # If max_size_mb is provided and the file is already smaller, just copy
            if max_size_mb and original_size <= max_size_mb:
                logger.info(
                    f"GIF already under maximum size: {original_size:.2f}MB (max: {max_size_mb:.2f}MB)")
                log_gif_progress(
                    f"File already under maximum size ({original_size:.2f}MB)", "skipped")
                shutil.copy2(input_path, output_path)
                return original_size, True

            # Implement quality-targeted optimization
            # This simulates a quality-preserving optimization approach using multiple quality levels
            # For now, we'll implement it by trying different optimization settings and selecting
            # the one that best meets our quality target

            # Define quality levels from highest to lowest
            # MODIFIED: Updated quality levels with better dithering settings to reduce artifacts
            quality_levels = [
                {'colors': 256, 'lossy': 0, 'scale': 1.0,
                    'dither': None, 'quality_score': 1.0},
                {'colors': 256, 'lossy': 10, 'scale': 1.0,
                    'dither': None, 'quality_score': 0.95},
                {'colors': 224, 'lossy': 20, 'scale': 1.0,
                    'dither': None, 'quality_score': 0.9},
                {'colors': 192, 'lossy': 30, 'scale': 1.0,
                    'dither': None, 'quality_score': 0.85},
                {'colors': 160, 'lossy': 40, 'scale': 1.0,
                    'dither': 'ordered', 'quality_score': 0.8},
                {'colors': 128, 'lossy': 50, 'scale': 0.9,
                    'dither': 'ordered', 'quality_score': 0.75},
                {'colors': 96, 'lossy': 60, 'scale': 0.9,
                    'dither': 'ordered', 'quality_score': 0.7},
                {'colors': 80, 'lossy': 70, 'scale': 0.8,
                    'dither': 'ordered', 'quality_score': 0.65},
                {'colors': 64, 'lossy': 80, 'scale': 0.8,
                    'dither': 'ordered', 'quality_score': 0.6},
                {'colors': 48, 'lossy': 90, 'scale': 0.7,
                    'dither': 'ordered', 'quality_score': 0.5},
                {'colors': 32, 'lossy': 100, 'scale': 0.6,
                    'dither': 'ordered', 'quality_score': 0.4}
            ]

            # Find the closest quality level that meets or exceeds our target
            selected_level = None
            for level in quality_levels:
                if level['quality_score'] >= quality_target:
                    selected_level = level
                    break

            if not selected_level:
                # Default to the highest quality if nothing matches
                selected_level = quality_levels[0]

            logger.info(f"Selected quality level: {selected_level['quality_score']:.2f} "
                        f"(colors: {selected_level['colors']}, lossy: {selected_level['lossy']})")
            log_gif_progress(
                f"Using quality settings: q={selected_level['quality_score']:.2f}", "processing")

            # Use these parameters to optimize
            temp_output = Path(TEMP_FILE_DIR) / \
                f"quality_{uuid.uuid4().hex}.gif"

            # Run gifsicle with the selected settings
            cmd = [
                'gifsicle',
                '--optimize=3',
                f'--colors={selected_level["colors"]}',
                f'--lossy={selected_level["lossy"]}',
            ]

            if selected_level.get('dither'):
                cmd.append(f'--dither={selected_level["dither"]}')

            if selected_level['scale'] < 1.0:
                cmd.extend(['--scale=' + f"{selected_level['scale']:.2f}"])

            cmd.extend([str(input_path), '-o', str(temp_output)])

            process = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )

            if process.returncode == 0 and temp_output.exists():
                result_size = self.file_processor.get_file_size(temp_output)

                # Check if it meets max_size constraint
                if max_size_mb and result_size > max_size_mb:
                    logger.warning(
                        f"Quality-targeted result exceeds max size: {result_size:.2f}MB > {max_size_mb:.2f}MB")
                    log_gif_progress(
                        f"Quality result exceeds size limit. Using size-based optimization.", "warning")

                    # Fall back to size-based optimization
                    os.unlink(temp_output)
                    return self.optimize_gif(input_path, output_path, max_size_mb)
                else:
                    # Success - copy the result
                    shutil.copy2(temp_output, output_path)
                    os.unlink(temp_output)

                    reduction_percent = (
                        (original_size - result_size) / original_size * 100) if original_size > 0 else 0
                    logger.info(f"Quality optimization successful: {original_size:.2f}MB → {result_size:.2f}MB "
                                f"({reduction_percent:.1f}% reduction, quality: {selected_level['quality_score']:.2f})")

                    log_gif_progress(
                        f"Quality optimization successful: {result_size:.2f}MB with quality {selected_level['quality_score']:.2f}",
                        "success"
                    )

                    # Update statistics
                    with self.stats_lock:
                        self.stats['total_processed'] += 1
                        self.stats['successful_optimizations'] += 1
                        self.stats['bytes_saved'] += (
                            original_size - result_size) * 1024 * 1024
                        elapsed = time.time() - start_time
                        self.stats['total_time'] += elapsed

                    return result_size, True
            else:
                logger.error(f"Quality optimization failed: {process.stderr}")
                log_gif_progress(
                    f"Quality optimization failed, trying standard approach", "warning")

                # Fall back to standard size-based optimization
                return self.optimize_gif(input_path, output_path, max_size_mb)

        except Exception as e:
            logger.exception(f"Error in quality-targeted optimization: {e}")
            log_gif_progress(
                f"Quality optimization error: {str(e)}, using standard approach", "error")

            # Fall back to standard size-based optimization
            return self.optimize_gif(input_path, output_path, max_size_mb)

    def batch_optimize(self, file_paths: List[Path], output_dir: Path,
                       target_size_mb: float = None,
                       parallel: bool = True,
                       prioritize_quality: bool = False) -> Dict[Path, Tuple[float, bool]]:
        """Optimize multiple GIFs in batch.

        Args:
            file_paths: List of paths to GIF files to optimize
            output_dir: Directory to save optimized files
            target_size_mb: Target size in MB (optional)
            parallel: Whether to process files in parallel (default: True)
            prioritize_quality: Whether to prioritize quality over file size (default: False)

        Returns:
            Dictionary mapping input paths to (size, success) tuples
        """
        start_time = time.time()

        # Ensure output_dir is a Path
        output_dir = Path(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        results = {}

        # Ensure all file_paths are Path objects
        file_paths = [Path(f) for f in file_paths]

        logger.info(f"Starting batch optimization of {len(file_paths)} files")
        quality_msg = " (quality-prioritized)" if prioritize_quality else ""
        log_gif_progress(
            f"Starting batch optimization of {len(file_paths)} files{quality_msg}", "starting")

        # Sort files by size (largest first) to optimize processing time
        try:
            file_paths.sort(key=lambda f: f.stat(
            ).st_size if f.exists() else 0, reverse=True)
            logger.info("Sorted files by size for optimal processing order")
        except Exception as e:
            logger.warning(f"Failed to sort files by size: {e}")

        if parallel and len(file_paths) > 1:
            # Process in parallel using ThreadPoolExecutor with dynamic worker limits
            # For many small files, use more workers; for fewer large files, use fewer workers
            total_files = len(file_paths)

            # Calculate memory requirements based on file sizes
            try:
                total_size_mb = sum(
                    f.stat().st_size for f in file_paths if f.exists()) / (1024 * 1024)
                avg_size_mb = total_size_mb / total_files if total_files > 0 else 0
                logger.info(
                    f"Batch processing {total_files} files with average size: {avg_size_mb:.2f}MB")

                # Adjust worker count based on average file size
                # Fewer workers for larger files to prevent memory issues
                if avg_size_mb > 50:  # Very large files
                    workers = max(1, min(2, self.max_workers))
                elif avg_size_mb > 20:  # Large files
                    workers = max(1, min(4, self.max_workers))
                else:  # Smaller files
                    workers = self.max_workers

                logger.info(
                    f"Using {workers} workers for parallel processing based on file sizes")
            except Exception as e:
                logger.warning(
                    f"Error calculating file sizes: {e}, using default worker count")
                workers = self.max_workers

            # Process in batches to prevent memory issues
            # Process up to 10 files at a time
            batch_size = min(10, total_files)

            # Define a function to process a single file with better error handling
            def process_single_file(input_path):
                output_path = output_dir / input_path.name
                start = time.time()
                try:
                    # Check if input file exists
                    if not input_path.exists():
                        logger.error(
                            f"Input file does not exist: {input_path}")
                        return input_path, (0, False)

                    # Check if output already exists and is recent
                    if output_path.exists():
                        try:
                            # If output is newer than input and meets size requirements, skip
                            input_mtime = input_path.stat().st_mtime
                            output_mtime = output_path.stat().st_mtime
                            output_size = output_path.stat().st_size / (1024 * 1024)

                            if output_mtime > input_mtime and (target_size_mb is None or output_size <= target_size_mb):
                                logger.info(
                                    f"Skipping {input_path.name} - already processed")
                                return input_path, (output_size, True)
                        except Exception as e:
                            logger.warning(
                                f"Error checking existing output file: {e}")

                    # Process the file
                    if prioritize_quality:
                        size, success = self.optimize_gif_prioritize_quality(
                            input_path, output_path, target_size_mb)
                    else:
                        size, success = self.optimize_gif(
                            input_path, output_path, target_size_mb)

                    # Log processing time for this file
                    elapsed = time.time() - start
                    logger.info(
                        f"Processed {input_path.name} in {elapsed:.2f}s: {size:.2f}MB, success={success}")

                    return input_path, (size, success)
                except Exception as e:
                    logger.error(f"Error processing {input_path}: {e}")
                    # Make sure we don't leave a partially processed file
                    if output_path.exists():
                        try:
                            output_path.unlink()
                        except Exception:
                            pass
                    return input_path, (0, False)
                finally:
                    # Force garbage collection after each file to reduce memory pressure
                    if hasattr(gc, 'collect'):
                        gc.collect()

            # Process files in batches
            completed = 0
            with tqdm(total=total_files, desc="Optimizing GIFs", unit="file") as progress_bar:
                for i in range(0, total_files, batch_size):
                    batch = file_paths[i:i+batch_size]
                    logger.info(
                        f"Processing batch {i//batch_size + 1}/{(total_files + batch_size - 1)//batch_size}")

                    # Process this batch in parallel
                    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                        future_to_path = {executor.submit(
                            process_single_file, input_path): input_path for input_path in batch}

                        for future in concurrent.futures.as_completed(future_to_path):
                            input_path, result = future.result()
                            results[input_path] = result
                            completed += 1

                            # Update progress
                            progress_pct = (completed / total_files) * 100
                            progress_bar.update(1)
                            log_gif_progress(
                                f"Processed {completed}/{total_files} files ({progress_pct:.1f}%)", "processing")

                    # Clean up temporary files between batches to prevent memory issues
                    self.temp_manager.cleanup()

                    # Pause briefly between batches to allow system resources to recover
                    if i + batch_size < total_files:
                        time.sleep(0.5)
        else:
            # Process sequentially with a progress bar
            with tqdm(total=len(file_paths), desc="Optimizing GIFs", unit="file") as progress_bar:
                for i, input_path in enumerate(file_paths):
                    output_path = output_dir / input_path.name
                    logger.info(
                        f"Processing file {i+1}/{len(file_paths)}: {input_path.name}")
                    log_gif_progress(
                        f"Processing file {i+1}/{len(file_paths)}: {input_path.name}", "processing")

                    try:
                        if prioritize_quality:
                            size, success = self.optimize_gif_prioritize_quality(
                                input_path, output_path, target_size_mb)
                        else:
                            size, success = self.optimize_gif(
                                input_path, output_path, target_size_mb)
                        results[input_path] = (size, success)
                    except Exception as e:
                        logger.error(f"Error processing {input_path}: {e}")
                        results[input_path] = (0, False)

                    progress_bar.update(1)

                    # Clean up temporary files after each file
                    self.temp_manager.cleanup()

        # Log final results
        success_count = sum(1 for _, success in results.values() if success)
        elapsed_time = time.time() - start_time

        # Calculate total size reduction
        try:
            original_total = sum(
                f.stat().st_size for f in file_paths if f.exists()) / (1024 * 1024)
            optimized_total = sum(
                size for size, _ in results.values()) if results else 0
            percent_reduction = ((original_total - optimized_total) /
                                 original_total * 100) if original_total > 0 else 0

            logger.info(
                f"Batch optimization complete: {success_count}/{len(file_paths)} successful in {elapsed_time:.1f}s")
            logger.info(
                f"Total size reduction: {original_total:.2f}MB → {optimized_total:.2f}MB ({percent_reduction:.1f}%)")

            log_gif_progress(
                f"Batch optimization complete: {success_count}/{len(file_paths)} successful in {elapsed_time:.1f}s. "
                f"Reduced by {percent_reduction:.1f}%",
                "success" if success_count == len(file_paths) else "warning")
        except Exception as e:
            logger.warning(f"Error calculating batch statistics: {e}")
            log_gif_progress(
                f"Batch optimization complete: {success_count}/{len(file_paths)} successful in {elapsed_time:.1f}s",
                "success" if success_count == len(file_paths) else "warning")

        return results

    def get_optimization_stats(self) -> Dict:
        """Get statistics about optimization operations.

        Returns:
            Dictionary containing optimization statistics
        """
        with self.stats_lock:
            stats_copy = self.stats.copy()

            # Add derived statistics
            if stats_copy['total_processed'] > 0:
                stats_copy['success_rate'] = (stats_copy['successful_optimizations'] /
                                              stats_copy['total_processed']) * 100

                if stats_copy['total_time'] > 0:
                    stats_copy['avg_time_per_file'] = stats_copy['total_time'] / \
                        stats_copy['total_processed']
                else:
                    stats_copy['avg_time_per_file'] = 0
            else:
                stats_copy['success_rate'] = 0
                stats_copy['avg_time_per_file'] = 0

            stats_copy['cache_size'] = len(self.optimization_cache)

            return stats_copy

    def cleanup(self) -> None:
        """Clean up temporary resources."""
        logger.info("Cleaning up temporary resources")

        # Try to clean up temp directory
        try:
            temp_files_count = 0
            temp_dir = Path(TEMP_FILE_DIR)

            if temp_dir.exists():
                # Define patterns to clean
                patterns = ["*.gif", "preopt_*.gif",
                            "quality_*.gif", "temp_*.gif"]

                for pattern in patterns:
                    for file_path in temp_dir.glob(pattern):
                        try:
                            file_path.unlink(missing_ok=True)
                            temp_files_count += 1
                        except Exception as e:
                            logger.warning(
                                f"Failed to remove temp file {file_path}: {e}")

            logger.info(
                f"Cleanup completed: removed {temp_files_count} temporary files")

            # Shut down thread pool
            self.executor.shutdown(wait=False)

            # Log statistics
            stats = self.get_optimization_stats()
            logger.info(f"Optimization statistics: {stats}")

            # Only log success rate if it exists in the stats dictionary
            if 'success_rate' in stats:
                logger.info(
                    f"Processed {stats['total_processed']} files with {stats['success_rate']:.1f}% success rate")
            else:
                logger.info(f"Processed {stats['total_processed']} files")

            if stats['bytes_saved'] > 0:
                mb_saved = stats['bytes_saved'] / (1024 * 1024)
                logger.info(f"Total savings: {mb_saved:.2f}MB")

        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def _get_file_hash(self, file_path: Path) -> str:
        """Generate a hash for a file based on path, size, modification time and a sample of content.

        This provides a more reliable cache key by including a content sample, while still 
        being much faster than computing a full content hash.

        Args:
            file_path: Path to the file

        Returns:
            String hash for the file
        """
        try:
            # Get file metadata
            stat = file_path.stat()
            file_size = stat.st_size
            mod_time = stat.st_mtime

            # Read samples from the beginning, middle, and end of the file
            # for more reliable change detection without reading the entire file
            # 8KB or file_size/3, whichever is smaller
            sample_size = min(8192, file_size // 3)
            content_hash = ""

            if file_size > 0 and sample_size > 0:
                # Compute hash from file samples
                try:
                    with open(file_path, 'rb') as f:
                        # Sample from beginning
                        beginning = f.read(sample_size)

                        # Sample from middle
                        if file_size > sample_size * 2:
                            f.seek(file_size // 2 - sample_size // 2)
                            middle = f.read(sample_size)
                        else:
                            middle = b''

                        # Sample from end
                        if file_size > sample_size:
                            f.seek(max(0, file_size - sample_size))
                            end = f.read(sample_size)
                        else:
                            end = b''

                        # Create a hash of the samples
                        hasher = hashlib.md5()
                        hasher.update(beginning)
                        hasher.update(middle)
                        hasher.update(end)
                        content_hash = hasher.hexdigest()
                except Exception as e:
                    logger.warning(
                        f"Error sampling file content for hash: {e}")
                    # Fall back to just using metadata if content access fails

            # Combine metadata and content sample hash
            return f"{file_path.absolute()}_{file_size}_{mod_time}_{content_hash}"
        except Exception as e:
            logger.warning(f"Error generating file hash: {e}")
            # Fall back to path as last resort
            return str(file_path.absolute())

    def _get_cached_result(self, file_hash: str, target_size_mb: float) -> Optional[Dict]:
        """Get cached optimization result for a file hash.

        Args:
            file_hash: Hash identifying the file
            target_size_mb: Target size in MB

        Returns:
            Cached result dictionary or None if not found
        """
        with self.cache_lock:
            cache_key = f"{file_hash}_{target_size_mb}"

            # Check if we have this exact cache key
            if cache_key in self.optimization_cache:
                cached_result = self.optimization_cache[cache_key]

                # Verify the cached path still exists before returning
                if 'path' in cached_result:
                    cached_path = cached_result.get('path')
                    if cached_path and os.path.exists(cached_path):
                        # Update last accessed time for LRU tracking
                        cached_result['last_accessed'] = time.time()
                        self.optimization_cache[cache_key] = cached_result
                        return cached_result
                    else:
                        # Path no longer exists, remove this entry
                        logger.info(
                            f"Removing stale cache entry with missing file: {cached_path}")
                        del self.optimization_cache[cache_key]
                        return None

            return None

    def _cache_result(self, file_hash: str, target_size_mb: float, result: Dict) -> None:
        """Cache an optimization result.

        Args:
            file_hash: Hash identifying the file
            target_size_mb: Target size in MB
            result: Result dictionary to cache
        """
        with self.cache_lock:
            # Add timestamp for LRU implementation
            result['cached_time'] = time.time()
            result['last_accessed'] = time.time()

            cache_key = f"{file_hash}_{target_size_mb}"
            self.optimization_cache[cache_key] = result

            # Implement LRU cache - limit cache size to prevent memory issues
            cache_max_size = 100
            if len(self.optimization_cache) > cache_max_size:
                # Remove least recently accessed entries
                entries = list(self.optimization_cache.items())
                # Sort by last_accessed time (oldest first)
                entries.sort(key=lambda x: x[1].get('last_accessed', 0))
                # Remove oldest entries to get back under limit
                entries_to_remove = len(entries) - cache_max_size
                for i in range(entries_to_remove):
                    key_to_remove = entries[i][0]
                    logger.debug(f"Removing LRU cache entry: {key_to_remove}")
                    del self.optimization_cache[key_to_remove]

    def clear_cache(self) -> None:
        """Clear the optimization cache."""
        with self.cache_lock:
            self.optimization_cache.clear()
            logger.info("Optimization cache cleared")

    def invalidate_cache_entry(self, file_path: Path) -> None:
        """Invalidate cache entries for a specific file path.

        Args:
            file_path: Path to the file whose cache entries should be invalidated
        """
        with self.cache_lock:
            # Find all cache keys that contain this file path
            keys_to_remove = []
            path_str = str(file_path.absolute())

            for key in self.optimization_cache.keys():
                if path_str in key:
                    keys_to_remove.append(key)

            # Remove the identified keys
            for key in keys_to_remove:
                del self.optimization_cache[key]

            if keys_to_remove:
                logger.info(
                    f"Invalidated {len(keys_to_remove)} cache entries for {file_path}")

    def _apply_fallback_optimization(self, input_path: Path, output_path: Path, target_size_mb: float, original_size: float, start_time: float, progress_tracker=None) -> Tuple[float, bool]:
        """Apply a fallback optimization method when adaptive_optimizer is None.

        Args:
            input_path: Path to the input GIF file
            output_path: Path where the optimized GIF should be saved
            target_size_mb: Target size in megabytes (optional)
            original_size: Original file size in megabytes
            start_time: Start time of the optimization process
            progress_tracker: Optional progress tracker for reporting progress

        Returns:
            Tuple containing (final_size_mb, success)
        """
        temp_files_to_clean = []  # Track temporary files to clean up
        try:
            # Ensure paths are Path objects
            input_path = Path(input_path)
            output_path = Path(output_path)

            # Log fallback optimization
            logger.info("Using enhanced fallback optimization method")
            log_gif_progress(
                "Using enhanced fallback optimization method", "processing")

            # Create a temporary file for the initial optimization
            temp_output_path = Path(TEMP_FILE_DIR) / \
                f"fallback_initial_{uuid.uuid4().hex}.gif"
            temp_files_to_clean.append(temp_output_path)

            # Initial optimization using gifsicle - prioritize scaling and improved dithering
            initial_cmd = [
                'gifsicle',
                '--optimize=3',
                '--colors=256',  # Maximum colors for best quality
                '--scale=0.8',   # Prioritize scaling down before other optimizations
                '--dither=bayer:bayer_scale=4',  # Better dithering for sharper images
                '--no-warnings',
                str(input_path.absolute()),
                '-o', str(temp_output_path.absolute())
            ]

            # Run the initial command
            process = subprocess.run(
                initial_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )

            if process.returncode != 0 or not temp_output_path.exists():
                logger.error(
                    f"Initial fallback optimization failed: {process.stderr}")
                # Fall back to simpler approach - prioritize scaling over color reduction
                simple_cmd = [
                    'gifsicle',
                    '--optimize=2',
                    '--colors=256',  # Keep maximum colors
                    '--scale=0.7',   # Use aggressive scaling instead of color reduction
                    '--dither=floyd-steinberg',  # Use better dithering
                    '--no-warnings',
                    str(input_path.absolute()),
                    '-o', str(output_path.absolute())
                ]

                subprocess.run(
                    simple_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False
                )

                if output_path.exists():
                    final_size = self.file_processor.get_file_size(output_path)
                    return final_size, final_size <= target_size_mb
                else:
                    # If all fails, copy the original
                    shutil.copy2(input_path, output_path)
                    return original_size, False

            # Check if initial optimization meets target
            optimized_size = self.file_processor.get_file_size(
                temp_output_path)

            if optimized_size <= target_size_mb:
                # Initial optimization was successful
                logger.info(
                    f"Initial fallback optimization successful: {original_size:.2f}MB → {optimized_size:.2f}MB")
                shutil.copy2(temp_output_path, output_path)

                # Update statistics
                with self.stats_lock:
                    self.stats['total_processed'] += 1
                    self.stats['successful_optimizations'] += 1
                    self.stats['bytes_saved'] += (original_size -
                                                  optimized_size) * 1024 * 1024
                    elapsed = time.time() - start_time
                    self.stats['total_time'] += elapsed

                return optimized_size, True

            # Need further optimization - try adaptive approach with color palette optimization
            logger.info(
                f"Initial optimization insufficient: {optimized_size:.2f}MB (target: {target_size_mb:.2f}MB)")
            log_gif_progress(
                "Applying adaptive color optimization...", "processing")

            # Use our new adaptive optimization method
            result_size, success = self._apply_adaptive_optimization(
                temp_output_path,
                output_path,
                target_size_mb,
                original_size,
                start_time
            )

            if success:
                return result_size, True

            # If adaptive also failed, handle the failure
            logger.error(
                f"All optimization attempts failed to meet target: {result_size:.2f}MB (target: {target_size_mb:.2f}MB)")
            log_gif_progress(
                f"Failed to meet target size: {result_size:.2f}MB", "error")

            # For strict requirements, don't return a file that doesn't meet the target
            if result_size > target_size_mb:
                logger.error(
                    f"Optimization failed to meet target size requirements")

                # Update statistics
                with self.stats_lock:
                    self.stats['total_processed'] += 1
                    self.stats['failed_optimizations'] += 1
                    elapsed = time.time() - start_time
                    self.stats['total_time'] += elapsed

                return result_size, False
            else:
                # This shouldn't happen since success would be True if we're under target
                return result_size, True

        except Exception as e:
            # Clean up temporary files before raising exception
            try:
                for temp_file in temp_files_to_clean:
                    if temp_file and temp_file.exists():
                        temp_file.unlink()
            except Exception:
                pass

            logger.exception(f"Error in fallback optimization: {e}")
            log_gif_progress(f"Fallback optimization error: {str(e)}", "error")

            # Update statistics
            with self.stats_lock:
                self.stats['total_processed'] += 1
                self.stats['failed_optimizations'] += 1
                elapsed = time.time() - start_time
                self.stats['total_time'] += elapsed

            return self.file_processor.get_file_size(input_path), False

    def _apply_adaptive_optimization(self, temp_input_path: Path, output_path: Path,
                                     target_size_mb: float, original_size: float, start_time: float) -> Tuple[float, bool]:
        """Apply adaptive optimization with intelligent color palette selection.

        Args:
            temp_input_path: Path to the input file (already pre-optimized)
            output_path: Path where the final optimized file should be saved
            target_size_mb: Target size in megabytes
            original_size: Original file size for statistics
            start_time: Start time for statistics

        Returns:
            Tuple containing (final_size_mb, success)
        """
        temp_files_to_clean = []  # Track temporary files to clean up
        color_map_path = None
        try:
            # Ensure paths are Path objects
            temp_input_path = Path(temp_input_path)
            output_path = Path(output_path)

            # Ensure input path is valid and exists
            if not temp_input_path.exists():
                logger.error(
                    f"Input file for adaptive optimization does not exist: {temp_input_path}")
                return 0, False

            # Create a copy of the input file to work with
            working_input_path = Path(
                TEMP_FILE_DIR) / f"adaptive_working_{uuid.uuid4().hex}.gif"
            temp_files_to_clean.append(working_input_path)
            try:
                shutil.copy2(temp_input_path, working_input_path)
                temp_input_path = working_input_path  # Use the copy for operations
            except Exception as e:
                logger.warning(f"Failed to create working copy: {e}")
                # Continue with original path if copy fails
                temp_input_path = temp_input_path

            # Get current size
            current_size = self.file_processor.get_file_size(temp_input_path)

            # Calculate how much more reduction is needed
            reduction_needed = (current_size - target_size_mb) / \
                current_size if current_size > target_size_mb else 0
            logger.info(f"Reduction needed: {reduction_needed:.2%}")

            # Detect frame count for better decision making
            frame_count = self._detect_frame_count(temp_input_path)
            logger.info(f"Detected {frame_count} frames in GIF")

            # Setup for adaptive optimization
            temp_current_path = temp_input_path

            # Store all results to allow selecting best quality under target size
            all_results = []

            # IMPROVED: Higher minimum for colors and lower maximum lossy value for better quality
            # Increase minimum colors to reduce black artifacts
            min_colors = max(128, min(256, int(256 * (1 - reduction_needed))))

            # Calculate max lossy based on reduction needed - be less aggressive
            # Reduce maximum lossy to prevent black artifacts
            max_lossy = min(80, int(60 * reduction_needed) + 20)

            # Set minimum scale factor to preserve quality
            min_scale = max(0.8, min(1.0, 1.0 - (reduction_needed / 2)))

            # Define quality stages from highest to lowest
            # Each dictionary defines a specific quality configuration to try
            quality_stages = [
                # Stage 1: Highest quality with minimal compression
                {
                    'colors': 256,
                    'lossy': max(5, min(15, int(15 * reduction_needed))),
                    'scale': 1.0,
                    'dither': None,  # No dithering for highest quality
                    'quality_level': 'highest'
                },
                # Stage 2: High quality with moderate compression
                {
                    'colors': max(192, min_colors),
                    'lossy': max(10, min(30, int(30 * reduction_needed))),
                    'scale': 1.0,
                    'dither': None,  # No dithering to prevent artifacts
                    'quality_level': 'high'
                },
                # Stage 3: Medium-high quality with more compression
                {
                    'colors': max(160, min_colors),
                    'lossy': max(20, min(50, int(50 * reduction_needed))),
                    'scale': min(1.0, max(min_scale, 0.95)),
                    'dither': 'ordered',  # Try ordered dithering which creates fewer artifacts
                    'quality_level': 'medium-high'
                },
                # Stage 4: Medium quality with stronger compression
                {
                    'colors': max(128, min_colors),
                    'lossy': max(30, min(60, int(60 * reduction_needed))),
                    'scale': min(1.0, max(min_scale, 0.9)),
                    'dither': 'ordered',  # Ordered dithering creates fewer artifacts than floyd-steinberg
                    'quality_level': 'medium'
                },
                # Stage 5: Medium-low quality (only used if needed)
                {
                    'colors': max(96, min_colors),
                    'lossy': max(40, min(max_lossy, int(70 * reduction_needed))),
                    'scale': min(1.0, max(min_scale, 0.85)),
                    'dither': 'ordered',  # Ordered dithering for less noticeable artifacts
                    'quality_level': 'medium-low'
                }
            ]

            # For very large reductions, add one more aggressive stage
            if reduction_needed > 0.7:
                quality_stages.append({
                    'colors': min_colors,
                    'lossy': max_lossy,
                    'scale': min_scale,
                    'dither': None,  # No dithering for low quality
                    'quality_level': 'low'
                })

            # Try each quality stage in order
            for stage_index, stage in enumerate(quality_stages):
                # Create a temporary file for this attempt
                temp_output_attempt = Path(
                    TEMP_FILE_DIR) / f"adaptive_{stage['quality_level']}_{uuid.uuid4().hex}.gif"
                temp_files_to_clean.append(temp_output_attempt)

                # Build optimization command
                colors = stage['colors']
                lossy = stage['lossy']
                scale_factor = stage['scale']
                dither = stage['dither']

                optimize_cmd = [
                    'gifsicle',
                    '--optimize=3',
                    f'--colors={colors}',
                    f'--lossy={lossy}',
                ]

                # Apply scaling if needed
                if scale_factor < 0.99:
                    optimize_cmd.append(f'--scale={scale_factor:.2f}')

                # Add dithering for better color transitions
                if dither:
                    optimize_cmd.append(f'--dither={dither}')

                # Add input and output paths with absolute paths
                optimize_cmd.extend([
                    '--no-warnings',
                    str(temp_current_path.absolute()),
                    '--output',
                    str(temp_output_attempt.absolute())
                ])

                # Log the current optimization attempt
                logger.info(
                    f"Trying {stage['quality_level']} quality: colors={colors}, lossy={lossy}, scale={scale_factor:.2f}")
                log_gif_progress(
                    f"Trying {stage['quality_level']} quality settings ({stage_index+1}/{len(quality_stages)})",
                    "processing"
                )

                # Run the optimization command
                try:
                    process = subprocess.run(
                        optimize_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        timeout=180  # 3 minutes timeout
                    )

                    if process.returncode == 0 and temp_output_attempt.exists():
                        result_size = self.file_processor.get_file_size(
                            temp_output_attempt)

                        # Create a result record
                        result = {
                            'stage': stage_index,
                            'quality_level': stage['quality_level'],
                            'size': result_size,
                            'path': temp_output_attempt,
                            'colors': colors,
                            'lossy': lossy,
                            'scale': scale_factor
                        }

                        # Add to our results list
                        all_results.append(result)

                        logger.info(
                            f"{stage['quality_level']} quality result: {result_size:.2f}MB (target: {target_size_mb:.2f}MB)")

                        # If we've reached the target and this is one of the first two stages (high quality),
                        # we can stop early with good quality
                        if result_size <= target_size_mb and stage_index <= 1:
                            logger.info(
                                f"Target size reached with high quality settings!")
                            break
                    else:
                        logger.error(
                            f"{stage['quality_level']} quality optimization failed: {process.stderr}")

                except Exception as e:
                    logger.error(
                        f"{stage['quality_level']} quality optimization failed: {e}")

            # Find the best result that meets the target size
            valid_results = [
                r for r in all_results if r['size'] <= target_size_mb]

            if valid_results:
                # Sort by quality level (lowest stage number = highest quality)
                valid_results.sort(key=lambda r: r['stage'])
                # Take the highest quality that meets target
                best_result = valid_results[0]

                logger.info(
                    f"Using {best_result['quality_level']} quality: {best_result['size']:.2f}MB with "
                    f"colors={best_result['colors']}, lossy={best_result['lossy']}, scale={best_result['scale']:.2f}"
                )
                log_gif_progress(
                    f"Optimization successful: {best_result['size']:.2f}MB with {best_result['quality_level']} quality",
                    "success"
                )

                # Copy the best result to the output
                shutil.copy2(best_result['path'], output_path)
                final_size = best_result['size']

                # Update statistics
                with self.stats_lock:
                    self.stats['total_processed'] += 1
                    self.stats['successful_optimizations'] += 1
                    self.stats['bytes_saved'] += (original_size -
                                                  final_size) * 1024 * 1024
                    elapsed = time.time() - start_time
                    self.stats['total_time'] += elapsed

                # Clean up temp files except the one we're returning
                for temp_file in temp_files_to_clean:
                    if temp_file.exists() and temp_file != best_result['path']:
                        try:
                            temp_file.unlink()
                        except:
                            pass

                # Also clean up the best result file if it's not needed
                if best_result['path'].exists() and best_result['path'] != output_path:
                    try:
                        best_result['path'].unlink()
                    except:
                        pass

                return final_size, True

            # If no good results were found, try one more approach with binary search
            logger.info(
                "No satisfactory results found, using binary search for optimal settings")

            # Taking the most recent temp file as our working input
            latest_result = max(
                all_results, key=lambda r: r['stage']) if all_results else None

            if latest_result and latest_result['path'].exists():
                if latest_result['size'] <= target_size_mb:
                    # Use this result if it somehow meets the target
                    shutil.copy2(latest_result['path'], output_path)
                    return latest_result['size'], True
                else:
                    # Use this as our starting point for a binary search
                    temp_current_path = latest_result['path']

                    # Binary search for lossy value
                    min_lossy = latest_result['lossy']
                    max_lossy_search = min(200, latest_result['lossy'] + 80)
                    best_lossy_result = None

                    for _ in range(5):  # Limit iterations
                        test_lossy = (min_lossy + max_lossy_search) // 2
                        test_output = Path(TEMP_FILE_DIR) / \
                            f"binary_search_{uuid.uuid4().hex}.gif"
                        temp_files_to_clean.append(test_output)

                        binary_cmd = [
                            'gifsicle',
                            '--optimize=3',
                            f'--colors={latest_result["colors"]}',
                            f'--lossy={test_lossy}',
                            '--dither=floyd-steinberg',
                            str(temp_current_path),
                            '-o', str(test_output)
                        ]

                        process = subprocess.run(
                            binary_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            text=True,
                            check=False,
                            timeout=60
                        )

                        if process.returncode == 0 and test_output.exists():
                            test_size = self.file_processor.get_file_size(
                                test_output)
                            logger.info(
                                f"Binary search with lossy={test_lossy}: {test_size:.2f}MB")

                            if test_size <= target_size_mb:
                                # Found a valid result
                                best_lossy_result = {
                                    'path': test_output,
                                    'size': test_size,
                                    'lossy': test_lossy
                                }
                                # Try for better quality
                                max_lossy_search = test_lossy - 1
                            else:
                                # Need more compression
                                min_lossy = test_lossy + 1
                                # Clean up this attempt
                                if test_output.exists():
                                    try:
                                        test_output.unlink()
                                    except:
                                        pass
                        else:
                            # Something went wrong, increase lossy
                            min_lossy = test_lossy + 10
                            if test_output.exists():
                                try:
                                    test_output.unlink()
                                except:
                                    pass

                        if best_lossy_result or min_lossy >= max_lossy_search:
                            break

                    if best_lossy_result:
                        # Use this result
                        shutil.copy2(best_lossy_result['path'], output_path)
                        logger.info(
                            f"Binary search successful: {best_lossy_result['size']:.2f}MB with lossy={best_lossy_result['lossy']}"
                        )
                        log_gif_progress(
                            f"Optimization successful via binary search: {best_lossy_result['size']:.2f}MB",
                            "success"
                        )

                        # Update statistics
                        with self.stats_lock:
                            self.stats['total_processed'] += 1
                            self.stats['successful_optimizations'] += 1
                            self.stats['bytes_saved'] += (
                                original_size - best_lossy_result['size']) * 1024 * 1024
                            elapsed = time.time() - start_time
                            self.stats['total_time'] += elapsed

                        # Clean up
                        for temp_file in temp_files_to_clean:
                            if temp_file.exists() and temp_file != best_lossy_result['path']:
                                try:
                                    temp_file.unlink()
                                except:
                                    pass

                        if best_lossy_result['path'].exists():
                            try:
                                best_lossy_result['path'].unlink()
                            except:
                                pass

                        return best_lossy_result['size'], True

            # If all else fails, use the most aggressive optimization
            logger.warning(
                "All optimization attempts failed to meet target size")
            log_gif_progress("Using last resort optimization", "warning")

            last_resort = Path(TEMP_FILE_DIR) / \
                f"last_resort_{uuid.uuid4().hex}.gif"
            temp_files_to_clean.append(last_resort)

            # Very aggressive settings
            last_cmd = [
                'gifsicle',
                '--optimize=3',
                '--colors=64',
                '--lossy=150',
                '--scale=0.7',
                '--dither=floyd-steinberg',  # Still keep dithering for quality
                str(temp_input_path),
                '-o', str(last_resort)
            ]

            process = subprocess.run(
                last_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False
            )

            if process.returncode == 0 and last_resort.exists():
                final_size = self.file_processor.get_file_size(last_resort)
                shutil.copy2(last_resort, output_path)

                success = final_size <= target_size_mb
                status = "success" if success else "error"
                message = "Optimization successful with minimal quality" if success else "Failed to reach target size"

                logger.info(
                    f"Last resort optimization result: {final_size:.2f}MB (target: {target_size_mb:.2f}MB)")
                log_gif_progress(f"{message}: {final_size:.2f}MB", status)

                # Update statistics
                with self.stats_lock:
                    self.stats['total_processed'] += 1
                    if success:
                        self.stats['successful_optimizations'] += 1
                        self.stats['bytes_saved'] += (
                            original_size - final_size) * 1024 * 1024
                    else:
                        self.stats['failed_optimizations'] += 1
                    elapsed = time.time() - start_time
                    self.stats['total_time'] += elapsed

                return final_size, success

            # If even that fails, return the original with failure
            logger.error("All optimization attempts failed")
            return original_size, False

        except Exception as e:
            logger.exception(f"Error in adaptive optimization: {e}")
            return original_size, False

        finally:
            # Clean up all temporary files
            for temp_file in temp_files_to_clean:
                if temp_file and temp_file.exists():
                    try:
                        temp_file.unlink()
                    except Exception as e:
                        logger.warning(
                            f"Failed to clean up temp file {temp_file}: {e}")

    def _optimize_extremely_large_gif(self, input_path: Path, output_path: Path,
                                      target_size_mb: float, original_size: float,
                                      start_time: float) -> Tuple[float, bool]:
        """Special optimization approach for extremely large GIF files (>100MB).

        This method applies a staged approach to optimization:
        1. Initial basic optimization to reduce size
        2. Frame reduction for animated GIFs
        3. Color reduction and lossy compression
        4. Final aggressive optimization if needed

        Args:
            input_path: Path to the input GIF file
            output_path: Path to save the optimized GIF
            target_size_mb: Target size in MB
            original_size: Original file size in MB
            start_time: Time when optimization started

        Returns:
            Tuple of (final_size_mb, success)
        """
        success = False
        final_size = original_size
        best_output = input_path  # Initialize with input_path as fallback
        temp_files_to_clean = []

        try:
            # Setup for staged approach
            temp_dir = Path(TEMP_FILE_DIR)
            stage1_output = temp_dir / f"stage1_{uuid.uuid4().hex}.gif"
            stage2_output = temp_dir / f"stage2_{uuid.uuid4().hex}.gif"
            stage3_output = temp_dir / f"stage3_{uuid.uuid4().hex}.gif"

            # Get preliminary info about the file
            try:
                info_cmd = ['gifsicle', '--info', str(input_path)]
                gif_info = subprocess.run(
                    info_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=30  # 30 seconds timeout for info
                ).stdout

                # Extract frame count and dimensions if possible
                frame_count = 1  # Default
                if "images" in gif_info:
                    try:
                        frame_match = gif_info.split(
                            "images")[0].strip().split()[-1]
                        frame_count = int(frame_match)
                    except (ValueError, IndexError):
                        pass

                logger.info(f"GIF info: approximately {frame_count} frames")
            except Exception as e:
                logger.warning(f"Could not get GIF info: {e}")
                frame_count = 1  # Default to single frame

            # STAGE 1: Initial size reduction with basic optimization
            logger.info("Stage 1: Initial size reduction")
            log_gif_progress("Stage 1: Initial size reduction", "processing")

            stage1_cmd = [
                'gifsicle',
                '--optimize=2',  # Less aggressive for first stage
                '--colors=256',  # Keep all colors initially
                '--no-warnings',
                str(input_path),
                '-o', str(stage1_output)
            ]

            try:
                subprocess.run(
                    stage1_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=180  # 3 minutes timeout
                )
            except subprocess.TimeoutExpired:
                logger.warning(
                    "Stage 1 optimization timed out, using more aggressive settings")
                # If timeout, try more aggressive settings
                stage1_cmd = [
                    'gifsicle',
                    '--optimize=1',  # Even less aggressive
                    '--colors=128',  # Reduce colors
                    '--no-warnings',
                    str(input_path),
                    '-o', str(stage1_output)
                ]
                try:
                    subprocess.run(
                        stage1_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        timeout=180  # 3 minutes timeout
                    )
                except Exception as e:
                    logger.error(f"Stage 1 retry failed: {e}")
                    return self._apply_fallback_optimization(input_path, output_path, target_size_mb, original_size, start_time)

            # Check if stage 1 was successful
            if not stage1_output.exists():
                logger.error(
                    "Stage 1 output file doesn't exist, falling back to alternative method")
                return self._apply_fallback_optimization(input_path, output_path, target_size_mb, original_size, start_time)

            stage1_size = self.file_processor.get_file_size(stage1_output)
            logger.info(
                f"Stage 1 complete: {original_size:.2f}MB → {stage1_size:.2f}MB")

            # If already at target size, we're done
            if stage1_size <= target_size_mb:
                logger.info(
                    f"Target size reached after Stage 1: {stage1_size:.2f}MB")
                shutil.copy2(stage1_output, output_path)
                return stage1_size, True

            # STAGE 2: Frame reduction for animated GIFs
            # Only apply if it's an animation with many frames
            if frame_count > 10:
                logger.info(
                    f"Stage 2: Frame reduction for {frame_count} frames")
                log_gif_progress("Stage 2: Frame reduction", "processing")

                # Calculate frame skip based on how many frames and how far we are from target
                reduction_factor = max(
                    0.5, min(0.9, target_size_mb / stage1_size))
                # 1-5 based on how much reduction needed
                frame_skip = max(1, int((1 - reduction_factor) * 5))

                # Build command to keep every nth frame
                stage2_cmd = ['gifsicle', '--unoptimize']

                # Select frames with appropriate skip rate
                frame_indices = list(range(0, frame_count, frame_skip + 1))
                if not frame_indices:
                    # Ensure we at least keep the first frame
                    frame_indices = [0]

                for idx in frame_indices:
                    stage2_cmd.extend(['--select', f'#{idx}'])

                stage2_cmd.extend([
                    '--optimize=2',
                    str(stage1_output),
                    '-o', str(stage2_output)
                ])

                try:
                    subprocess.run(
                        stage2_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        timeout=180  # 3 minutes timeout
                    )
                except Exception as e:
                    logger.error(f"Stage 2 frame reduction failed: {e}")
                    # Continue with stage1 output
                    stage2_output = stage1_output
            else:
                # Skip frame reduction for non-animations or few frames
                logger.info(
                    "Skipping frame reduction (not an animation or too few frames)")
                stage2_output = stage1_output

            # Check stage 2 results
            if stage2_output != stage1_output and stage2_output.exists():
                stage2_size = self.file_processor.get_file_size(stage2_output)
                logger.info(
                    f"Stage 2 complete: {stage1_size:.2f}MB → {stage2_size:.2f}MB")

                # If at target, we're done
                if stage2_size <= target_size_mb:
                    logger.info(
                        f"Target size reached after Stage 2: {stage2_size:.2f}MB")
                    shutil.copy2(stage2_output, output_path)
                    return stage2_size, True
            else:
                # If stage 2 failed or was skipped, use stage 1 output
                logger.info("Using Stage 1 output for next stage")
                stage2_output = stage1_output
                stage2_size = stage1_size

            # STAGE 3: Color reduction and lossy compression
            logger.info("Stage 3: Color reduction and lossy compression")
            log_gif_progress("Stage 3: Final optimization", "processing")

            # Calculate necessary reduction
            reduction_needed = (stage2_size - target_size_mb) / stage2_size

            # Set parameters based on how much reduction is needed
            colors = max(32, min(128, int(128 * (1 - reduction_needed))))
            lossy = min(120, int(reduction_needed * 200))
            scale_factor = max(0.5, min(1.0, 1.0 - (reduction_needed / 2)))

            stage3_cmd = [
                'gifsicle',
                '--optimize=3',
                f'--colors={colors}',
                f'--lossy={lossy}',
            ]

            if scale_factor < 0.99:
                stage3_cmd.append(f'--scale={scale_factor:.2f}')

            # Add dithering to maintain visual quality with fewer colors
            stage3_cmd.append('--dither=floyd-steinberg')

            stage3_cmd.extend([
                '--no-warnings',
                str(stage2_output),
                '-o', str(stage3_output)
            ])

            try:
                subprocess.run(
                    stage3_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=180  # 3 minutes timeout
                )
            except Exception as e:
                logger.error(f"Stage 3 optimization failed: {e}")
                # Try a more conservative approach if first attempt failed
                stage3_cmd = [
                    'gifsicle',
                    '--optimize=3',
                    '--colors=32',
                    '--lossy=80',
                    '--no-warnings',
                    str(stage2_output),
                    '-o', str(stage3_output)
                ]

                try:
                    subprocess.run(
                        stage3_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        timeout=180  # 3 minutes timeout
                    )
                except Exception as e:
                    logger.error(f"Stage 3 retry failed: {e}")
                    # Use previous stage output
                    stage3_output = stage2_output

            # Check final result
            if stage3_output != stage2_output and stage3_output.exists():
                final_size = self.file_processor.get_file_size(stage3_output)
                logger.info(
                    f"Stage 3 complete: {stage2_size:.2f}MB → {final_size:.2f}MB")

                # Update best output
                best_output = stage3_output

                # Check if we met the target
                success = final_size <= target_size_mb
                if success:
                    logger.info(
                        f"Staged optimization successful: {original_size:.2f}MB → {final_size:.2f}MB")
                    log_gif_progress(
                        f"Staged optimization successful: {final_size:.2f}MB",
                        "success"
                    )
                else:
                    # This is where we could add additional optimization for cases like 26.44MB when target is 10MB
                    logger.warning(
                        f"Staged optimization improved size but didn't meet target: {final_size:.2f}MB > {target_size_mb:.2f}MB")
                    log_gif_progress(
                        f"Optimization improved size but didn't meet target: {final_size:.2f}MB",
                        "warning"
                    )

                    # STAGE 4: Apply more aggressive optimization if we're within a reasonable range of the target
                    if final_size <= target_size_mb * 3:  # If we're within 3x of target, try harder
                        logger.info(
                            "Stage 4: Applying additional aggressive optimization to meet target")
                        log_gif_progress(
                            "Stage 4: Final aggressive optimization", "processing")

                        # Create a new temporary file for this final stage
                        stage4_output = temp_dir / \
                            f"stage4_{uuid.uuid4().hex}.gif"

                        # Calculate how much more reduction is needed
                        reduction_needed = (
                            final_size - target_size_mb) / final_size

                        # Very aggressive settings to reach target
                        colors = max(16, int(32 * (1 - reduction_needed)))
                        lossy = min(200, int(120 + reduction_needed * 80))
                        scale_factor = max(
                            0.5, min(0.9, 0.9 - (reduction_needed * 0.4)))

                        stage4_cmd = [
                            'gifsicle',
                            '--optimize=3',
                            f'--colors={colors}',
                            f'--lossy={lossy}',
                        ]

                        if scale_factor < 0.99:
                            stage4_cmd.append(f'--scale={scale_factor:.2f}')

                        stage4_cmd.append('--dither=floyd-steinberg')
                        stage4_cmd.extend([
                            '--no-warnings',
                            str(stage3_output),
                            '-o', str(stage4_output)
                        ])

                        try:
                            subprocess.run(
                                stage4_cmd,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                check=False,
                                timeout=180  # 3 minutes timeout
                            )

                            if stage4_output.exists():
                                stage4_size = self.file_processor.get_file_size(
                                    stage4_output)
                                logger.info(
                                    f"Stage 4 complete: {final_size:.2f}MB → {stage4_size:.2f}MB")

                                # Use this result if it's better
                                if stage4_size < final_size:
                                    final_size = stage4_size
                                    best_output = stage4_output
                                    success = final_size <= target_size_mb

                                    if success:
                                        logger.info(
                                            f"Additional optimization successful: {original_size:.2f}MB → {final_size:.2f}MB")
                                        log_gif_progress(
                                            f"Additional optimization successful: {final_size:.2f}MB",
                                            "success"
                                        )
                                    else:
                                        logger.warning(
                                            f"Reached best possible size: {final_size:.2f}MB > {target_size_mb:.2f}MB")
                                        log_gif_progress(
                                            f"Reached best possible size: {final_size:.2f}MB",
                                            "warning"
                                        )
                                else:
                                    # If stage4 wasn't better, clean up the file
                                    try:
                                        stage4_output.unlink()
                                    except Exception:
                                        pass
                        except Exception as e:
                            logger.error(f"Stage 4 optimization failed: {e}")

                # Copy the best result to output path if optimization was successful
                # Otherwise, save the best result to a special temp directory for later assessment
                if success:
                    # If successful, copy to output path
                    shutil.copy2(best_output, output_path)
                else:
                    # If we didn't meet the target size, save the best version to a special assessment directory
                    assessment_dir = Path(TEMP_FILE_DIR) / \
                        "optimization_assessment"
                    os.makedirs(assessment_dir, exist_ok=True)

                    # Create a descriptive filename with original size and achieved size
                    assessment_filename = f"{input_path.stem}_orig{original_size:.1f}MB_opt{final_size:.1f}MB{input_path.suffix}"
                    assessment_path = assessment_dir / assessment_filename

                    # Copy the best optimized version for assessment
                    shutil.copy2(best_output, assessment_path)
                    logger.info(
                        f"Saved best optimized version for assessment: {assessment_path}")
                    log_gif_progress(
                        f"Saved best version for assessment: {assessment_path}", "info")

                # Update statistics
                with self.stats_lock:
                    self.stats['total_processed'] += 1
                    if success:
                        self.stats['successful_optimizations'] += 1
                    else:
                        self.stats['failed_optimizations'] += 1
                    self.stats['bytes_saved'] += (original_size -
                                                  final_size) * 1024 * 1024
                    elapsed = time.time() - start_time
                    self.stats['total_time'] += elapsed

                # Clean up temp files except the best_output
                for temp_file in [stage1_output, stage2_output, stage3_output]:
                    try:
                        if temp_file.exists() and temp_file != best_output:
                            temp_file.unlink()
                    except Exception:
                        pass

                return final_size, success
            else:
                logger.error(
                    "Final optimization stage failed, using previous stage output")
                # Use the output from the previous successful stage
                previous_output = stage2_output
                previous_size = stage2_size

                # Check if we met the target
                success = previous_size <= target_size_mb

                # Copy the best result to output path if optimization was successful
                # Otherwise, save the best result to a special temp directory for later assessment
                if success:
                    # If successful, copy to output path
                    shutil.copy2(previous_output, output_path)
                else:
                    # If we didn't meet the target size, save the best version to a special assessment directory
                    assessment_dir = Path(TEMP_FILE_DIR) / \
                        "optimization_assessment"
                    os.makedirs(assessment_dir, exist_ok=True)

                    # Create a descriptive filename with original size and achieved size
                    assessment_filename = f"{input_path.stem}_orig{original_size:.1f}MB_opt{previous_size:.1f}MB{input_path.suffix}"
                    assessment_path = assessment_dir / assessment_filename

                    # Copy the best optimized version for assessment
                    shutil.copy2(previous_output, assessment_path)
                    logger.info(
                        f"Saved best optimized version for assessment: {assessment_path}")
                    log_gif_progress(
                        f"Saved best version for assessment: {assessment_path}", "info")

                # Update statistics
                with self.stats_lock:
                    self.stats['total_processed'] += 1
                    if success:
                        self.stats['successful_optimizations'] += 1
                    else:
                        self.stats['failed_optimizations'] += 1
                    self.stats['bytes_saved'] += (original_size -
                                                  previous_size) * 1024 * 1024
                    elapsed = time.time() - start_time
                    self.stats['total_time'] += elapsed

                if success:
                    logger.info(
                        f"Partial optimization successful: {original_size:.2f}MB → {previous_size:.2f}MB")
                    log_gif_progress(
                        f"Partial optimization successful: {previous_size:.2f}MB",
                        "success"
                    )
                else:
                    logger.warning(
                        f"Partial optimization didn't meet target: {previous_size:.2f}MB > {target_size_mb:.2f}MB")
                    log_gif_progress(
                        f"Partial optimization didn't meet target: {previous_size:.2f}MB",
                        "warning"
                    )

                # Clean up temp files except the best_output (previous_output)
                for temp_file in [stage1_output, stage2_output, stage3_output]:
                    try:
                        if temp_file.exists() and temp_file != previous_output:
                            temp_file.unlink()
                    except Exception:
                        pass

                return previous_size, success

        except Exception as e:
            logger.exception(f"Error in staged optimization: {e}")
            # Fall back to normal optimization method
            return self._apply_fallback_optimization(input_path, output_path, target_size_mb, original_size, start_time)

    def _detect_frames_with_gifsicle(self, gif_path: Path) -> int:
        """
        Detect frame count using gifsicle --info command

        Args:
            gif_path: Path to the GIF file

        Returns:
            Detected frame count or 0 if detection fails
        """
        try:
            # Use proper path handling
            info_cmd = ['gifsicle', '--info', str(gif_path.absolute())]

            # Set a reasonable timeout to prevent hanging
            process = subprocess.run(
                info_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=30  # 30 seconds timeout
            )

            if process.returncode != 0:
                logger.warning(
                    f"gifsicle info command failed: {process.stderr}")
                return 0

            gif_info = process.stdout

            # Extract frame count using robust pattern matching
            if "images" in gif_info:
                # Try different patterns to extract the value
                patterns = [
                    r'(\d+)\s+images',
                    r'images\s+(\d+)',
                    r'(\d+)(?=\s+image)'
                ]

                for pattern in patterns:
                    matches = re.search(pattern, gif_info)
                    if matches:
                        try:
                            return int(matches.group(1))
                        except (ValueError, IndexError):
                            continue

            # If we couldn't extract frames but see screens, try that
            if "screens" in gif_info:
                matches = re.search(r'(\d+)\s+screens', gif_info)
                if matches:
                    try:
                        return int(matches.group(1))
                    except (ValueError, IndexError):
                        pass

            logger.debug(
                f"Could not extract frame count from gifsicle output: {gif_info}")
            return 0

        except subprocess.TimeoutExpired:
            logger.warning(f"gifsicle info command timed out for {gif_path}")
            return 0
        except Exception as e:
            logger.warning(f"Error detecting frames with gifsicle: {e}")
            return 0

    def _safe_path_for_subprocess(self, path: Path) -> str:
        """
        Safely format a path for use in subprocess commands.
        Handles spaces and special characters appropriately.

        Args:
            path: Path object to format

        Returns:
            String representation of path safe for subprocess use
        """
        # Ensure path is absolute and properly normalized
        abs_path = path.absolute().resolve()

        # Convert to string and ensure proper quoting if needed
        path_str = str(abs_path)

        # Subprocess list format doesn't need extra quoting - each argument
        # is passed separately. Just return the string representation.
        return path_str

    def _run_subprocess(self, cmd: List[str], timeout: int = 180, check: bool = False) -> subprocess.CompletedProcess:
        """
        Run a subprocess command with proper error handling and logging.

        Args:
            cmd: Command to run as list of strings
            timeout: Timeout in seconds
            check: Whether to raise an exception on non-zero return code

        Returns:
            CompletedProcess instance with results

        Raises:
            Various exceptions if check=True and command fails
        """
        if not cmd:
            raise ValueError("Empty command provided")

        logger.debug(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=check,
                timeout=timeout
            )

            if result.returncode != 0:
                logger.warning(
                    f"Command failed with return code {result.returncode}: {result.stderr}")

            return result

        except subprocess.TimeoutExpired as e:
            logger.error(
                f"Command timed out after {timeout} seconds: {' '.join(cmd)}")
            raise
        except Exception as e:
            logger.error(f"Error running command: {e}")
            raise

    def _handle_already_optimized_file(self, input_path: Path, output_path: Path,
                                       original_size: float, target_size_mb: float,
                                       start_time: float) -> Tuple[float, bool]:
        """Handle case where file is already under target size.

        Args:
            input_path: Path to input file
            output_path: Path to output file
            original_size: Original file size in MB
            target_size_mb: Target size in MB
            start_time: Start time for statistics

        Returns:
            Tuple containing (final_size_mb, success)
        """
        logger.info(
            f"GIF already under target size: {original_size:.2f}MB (target: {target_size_mb:.2f}MB)")
        log_gif_progress(
            f"File already below target size ({original_size:.2f}MB)", "skipped")
        shutil.copy2(input_path, output_path)

        with self.stats_lock:
            self.stats['total_processed'] += 1
            self.stats['successful_optimizations'] += 1
            elapsed = time.time() - start_time
            self.stats['total_time'] += elapsed

        return original_size, True

    def _try_cached_optimization(self, input_path: Path, output_path: Path,
                                 target_size_mb: float, original_size: float,
                                 start_time: float) -> Optional[Tuple[float, bool]]:
        """Try to use cached optimization result if available.

        Args:
            input_path: Path to input file
            output_path: Path to output file
            target_size_mb: Target size in MB
            original_size: Original file size in MB
            start_time: Start time for statistics

        Returns:
            Optimization result tuple if cached result used, None otherwise
        """
        # Check if we have a cached result for this file
        file_hash = self._get_file_hash(input_path)
        cached_result = self._get_cached_result(file_hash, target_size_mb)

        if cached_result and cached_result.get('success', False):
            log_gif_progress(
                f"Using cached optimization result", "processing")
            cached_path = cached_result.get('path')

            if cached_path and os.path.exists(cached_path):
                shutil.copy2(cached_path, output_path)
                final_size = self.file_processor.get_file_size(output_path)

                log_gif_progress(
                    f"Optimization complete (cached): {final_size:.2f}MB", "success")
                logger.info(
                    f"Used cached result for {input_path.name}: {final_size:.2f}MB")

                with self.stats_lock:
                    self.stats['total_processed'] += 1
                    self.stats['cache_hits'] += 1
                    self.stats['successful_optimizations'] += 1
                    self.stats['bytes_saved'] += (
                        original_size - final_size) * 1024 * 1024
                    elapsed = time.time() - start_time
                    self.stats['total_time'] += elapsed

                return final_size, True

        return None

    def _apply_pre_optimization_if_needed(self, input_path: Path, target_size_mb: float,
                                          original_size: float) -> Path:
        """Apply preliminary optimization to reduce file size before main optimization if needed.

        Args:
            input_path: Path to input file
            target_size_mb: Target size in MB
            original_size: Original file size in MB

        Returns:
            Path to use for main optimization (either original or pre-optimized)
        """
        # Apply pre-optimization to reduce file size before main optimization
        if original_size > target_size_mb * 3:  # Only pre-optimize if significantly larger
            log_gif_progress(
                f"Applying preliminary optimization...", "processing")
            pre_optimized_path = self._apply_preliminary_optimization(
                input_path)

            if pre_optimized_path and os.path.exists(pre_optimized_path):
                pre_size = self.file_processor.get_file_size(
                    pre_optimized_path)
                logger.info(
                    f"Preliminary optimization: {original_size:.2f}MB → {pre_size:.2f}MB")
                return pre_optimized_path

        return input_path

    def _update_optimization_stats(self, success: bool, original_size: float, final_size: float, start_time: float, is_cache_hit: bool = False):
        """Update optimization statistics.

        Args:
            success: Whether optimization was successful
            original_size: Original file size in MB
            final_size: Final file size in MB
            start_time: Start time for calculating elapsed time
            is_cache_hit: Whether this was a cache hit
        """
        with self.stats_lock:
            self.stats['total_processed'] += 1

            if success:
                self.stats['successful_optimizations'] += 1
                self.stats['bytes_saved'] += (original_size -
                                              final_size) * 1024 * 1024
            else:
                self.stats['failed_optimizations'] += 1

            if is_cache_hit:
                self.stats['cache_hits'] += 1

            elapsed = time.time() - start_time
            self.stats['total_time'] += elapsed

    def _report_optimization_result(self, success: bool, original_size: float, final_size: float,
                                    target_size_mb: float, quality_score: float = None,
                                    settings: Dict = None):
        """Report optimization result via logs and progress messages.

        Args:
            success: Whether optimization was successful
            original_size: Original file size in MB
            final_size: Final file size in MB
            target_size_mb: Target size in MB
            quality_score: Quality score (optional)
            settings: Optimization settings used (optional)
        """
        if success:
            reduction_percent = (
                (original_size - final_size) / original_size * 100) if original_size > 0 else 0

            quality_info = f", quality: {quality_score:.2f}" if quality_score is not None else ""
            logger.info(f"GIF optimization successful: {original_size:.2f}MB → {final_size:.2f}MB "
                        f"({reduction_percent:.1f}% reduction{quality_info})")

            if settings:
                logger.info(f"Optimization settings: {settings}")

            log_gif_progress(
                f"Optimization successful: {original_size:.2f}MB → {final_size:.2f}MB ({reduction_percent:.1f}% reduction)",
                "success"
            )
        else:
            logger.error(
                f"GIF optimization failed: Unable to meet target size of {target_size_mb:.2f}MB")
            log_gif_progress(
                f"File optimization failed: Could not meet size target", "error")

    def _check_file_size_requirement(self, current_size: float, target_size_mb: float, file_path: Path) -> bool:
        """Check if file meets size requirements.

        Args:
            current_size: Current file size in MB
            target_size_mb: Target size in MB
            file_path: Path to the file being checked

        Returns:
            Whether the file meets size requirements
        """
        if current_size <= target_size_mb:
            logger.info(
                f"File meets target size: {current_size:.2f}MB ≤ {target_size_mb:.2f}MB")
            return True
        else:
            logger.warning(
                f"File exceeds target size: {current_size:.2f}MB > {target_size_mb:.2f}MB")
            log_gif_progress(
                f"Failed to reach target size: {current_size:.2f}MB > {target_size_mb:.2f}MB",
                "warning"
            )
            return False

    def _perform_main_optimization(self, input_path: Path, output_path: Path,
                                   target_size_mb: float, original_size: float,
                                   start_time: float, progress_tracker=None) -> Tuple[float, bool]:
        """Main optimization process with detailed progress reporting."""
        # Update progress
        if progress_tracker:
            progress_tracker.update_progress(0.1, "Starting main optimization")

        # Display detailed progress update
        with self.batch_progress_lock:
            self.batch_progress['current_stage'] = 'Main Optimization'
            display_progress_update(
                40, 100,
                description=f"Starting main optimization",
                file_name=input_path.name,
                status="OPTIMIZING",
                start_time=start_time
            )

        # Rest of the optimization code...
        # At various stages update progress:

        # Example progress update at 50%
        with self.batch_progress_lock:
            display_progress_update(
                50, 100,
                description=f"Applying optimizations",
                file_name=input_path.name,
                status="OPTIMIZING",
                start_time=start_time
            )

        # Original functionality continues...
        # (Add similar progress updates throughout the method)

        # Create temp file manager for this optimization run
        with TempFileManager(prefix="opt_") as temp_manager:
            try:
                if progress_tracker:
                    progress_tracker.update_progress(
                        20, "Setting up optimization pipeline")

                # Create event loop for async operation
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)

                try:
                    # Run the optimization process
                    log_gif_progress(
                        f"Running adaptive GIF optimization", "processing")

                    # Apply pre-optimization if needed
                    if original_size > target_size_mb * 3:  # Only pre-optimize if significantly larger
                        if progress_tracker:
                            progress_tracker.update_progress(
                                30, "Applying preliminary optimization")

                        log_gif_progress(
                            f"Applying preliminary optimization...", "processing")
                        pre_optimized_path = self._apply_preliminary_optimization(
                            input_path)

                        if pre_optimized_path and os.path.exists(pre_optimized_path):
                            temp_manager.register_file(pre_optimized_path)
                            pre_size = self.file_processor.get_file_size(
                                pre_optimized_path)

                            if progress_tracker:
                                progress_tracker.update_progress(
                                    35, f"Pre-optimization: {original_size:.2f}MB → {pre_size:.2f}MB")

                            logger.info(
                                f"Preliminary optimization: {original_size:.2f}MB → {pre_size:.2f}MB")
                            input_for_optimizer = pre_optimized_path
                        else:
                            input_for_optimizer = input_path
                            if progress_tracker:
                                progress_tracker.update_progress(
                                    35, "Preliminary optimization skipped")
                    else:
                        input_for_optimizer = input_path
                        if progress_tracker:
                            progress_tracker.update_progress(
                                35, "Preliminary optimization not needed")

                    if progress_tracker:
                        progress_tracker.update_progress(
                            40, "Running main optimization")

                    # Run the main optimization with the prepared input
                    optimization_result = loop.run_until_complete(
                        self.adaptive_optimizer.optimize(
                            input_for_optimizer, output_path, target_size_mb
                        )
                    )

                    if progress_tracker:
                        if optimization_result.success:
                            progress_tracker.update_progress(
                                50, f"Initial optimization complete: {optimization_result.size_mb:.2f}MB")
                        else:
                            progress_tracker.update_progress(
                                50, f"Initial optimization failed: {optimization_result.error}")

                    # If the adaptive optimizer didn't meet the target size, try increasingly aggressive optimization
                    if optimization_result.success and optimization_result.size_mb > target_size_mb:
                        if progress_tracker:
                            progress_tracker.update_progress(
                                60, "Target not met, applying additional optimization")

                        log_gif_progress(
                            f"Initial optimization didn't meet target size. Applying adaptive optimization...",
                            "processing"
                        )

                        # Additional optimization code here...
                        # This is a complex section that would need to be adapted with progress reporting
                        # For now, we'll just report that we're in this stage

                        if progress_tracker:
                            progress_tracker.update_progress(
                                80, "Applying adaptive settings to reach target size")

                finally:
                    # Clean up the event loop
                    loop.close()

                # Handle the result of optimization
                if optimization_result.success and optimization_result.file_path:
                    final_size = optimization_result.size_mb

                    # Only consider it successful if it's actually under target size
                    if final_size > target_size_mb:
                        if progress_tracker:
                            progress_tracker.update_progress(
                                85, f"Size still exceeds target: {final_size:.2f}MB > {target_size_mb:.2f}MB")

                        logger.warning(
                            f"File incorrectly marked as success despite size {final_size:.2f}MB > {target_size_mb:.2f}MB")
                        log_gif_progress(
                            f"Failed to reach target size: {final_size:.2f}MB > {target_size_mb:.2f}MB",
                            "error"
                        )

                        # Apply last resort optimization as a fallback
                        if progress_tracker:
                            progress_tracker.update_progress(
                                90, "Trying last resort optimization")

                        # This would call our last resort method
                        # For now, just report failure
                        self._update_optimization_stats(
                            False, original_size, final_size, start_time)

                        if progress_tracker:
                            progress_tracker.update_progress(
                                100, "Optimization failed to meet target size")

                        return final_size, False

                    # Update cache with successful result
                    file_hash = self._get_file_hash(input_path)
                    self._cache_result(file_hash, target_size_mb, {
                        'success': True,
                        'path': str(output_path),
                        'size': final_size,
                        'settings': optimization_result.settings,
                        'quality': optimization_result.quality_score
                    })

                    # Final success message and statistics
                    self._report_optimization_result(
                        True,
                        original_size,
                        final_size,
                        target_size_mb,
                        optimization_result.quality_score,
                        optimization_result.settings
                    )

                    self._update_optimization_stats(
                        True, original_size, final_size, start_time)

                    if progress_tracker:
                        reduction_percent = (
                            (original_size - final_size) / original_size * 100) if original_size > 0 else 0
                        progress_tracker.update_progress(
                            100,
                            f"Success: {original_size:.2f}MB → {final_size:.2f}MB ({reduction_percent:.1f}% reduction)"
                        )

                    return final_size, True

                else:
                    # Failed optimization
                    self._report_optimization_result(
                        False, original_size, original_size, target_size_mb)
                    self._update_optimization_stats(
                        False, original_size, original_size, start_time)

                    if progress_tracker:
                        progress_tracker.update_progress(
                            100, "Optimization failed")

                    return original_size, False

            except Exception as e:
                logger.exception(f"Error in main optimization process: {e}")

                if progress_tracker:
                    progress_tracker.update_progress(100, f"Error: {str(e)}")

                # Update failure statistics
                self._update_optimization_stats(
                    False, original_size, original_size, start_time)

                return original_size, False

    # Add new method for quality-prioritized optimization
    def optimize_gif_prioritize_quality(self, input_path: Path, output_path: Path, target_size_mb: float = None) -> Tuple[float, bool]:
        """
        Optimize a GIF while prioritizing quality, but still trying to meet the target size.

        This improved version focuses on maintaining visual quality first while still
        attempting to reach the target file size through smarter compression techniques.

        Args:
            input_path: Path to the input GIF file
            output_path: Path to save the optimized GIF
            target_size_mb: Target size in MB (defaults to config value if not specified)

        Returns:
            Tuple containing (final_size_mb, success)
        """
        logger = logging.getLogger('app')

        # Use default target size if not specified
        if target_size_mb is None:
            target_size_mb = 10.0  # Hard limit at 10MB

        # Ensure input file exists
        if not input_path.exists():
            logger.error(f"Input file does not exist: {input_path}")
            return 0, False

        # Get original file size
        original_size = self.file_processor.get_file_size(input_path)
        logger.info(
            f"Original size: {original_size:.2f}MB, target: {target_size_mb:.2f}MB")

        # If already under target size, just copy the file
        if original_size <= target_size_mb:
            try:
                shutil.copy2(input_path, output_path)
                logger.info(
                    f"File already under target size ({original_size:.2f}MB), no optimization needed")
                return original_size, True
            except Exception as e:
                logger.error(f"Error copying file: {e}")
                return 0, False

        # Create a temporary file manager for this operation
        with TempFileManager(prefix="quality_") as temp_manager:
            # Create working copy
            working_copy = temp_manager.create_temp_file()
            shutil.copy2(input_path, working_copy)

            # Progressive optimization steps focusing on quality - prioritizing scaling first
            # Prioritize scaling > lossiness > color reduction
            optimization_steps = [
                # Step 1: Just optimize without lossy compression
                {
                    "colors": 256,
                    "lossy": 0,
                    "optimize": 3,
                    "scale": 1.0
                },
                # Step 2: Scale down slightly with full colors
                {
                    "colors": 256,
                    "lossy": 0,
                    "optimize": 3,
                    "scale": 0.9
                },
                # Step 3: More scaling with full colors
                {
                    "colors": 256,
                    "lossy": 0,
                    "optimize": 3,
                    "scale": 0.8
                },
                # Step 4: Maximum scaling with full colors
                {
                    "colors": 256,
                    "lossy": 0,
                    "optimize": 3,
                    "scale": 0.7
                },
                # Step 5: Only add light lossy compression if scaling wasn't enough
                {
                    "colors": 256,
                    "lossy": 20,
                    "optimize": 3,
                    "scale": 0.7
                },
                # Step 6: Only reduce colors as a last resort
                {
                    "colors": 224,
                    "lossy": 20,
                    "optimize": 3,
                    "scale": 0.7
                }
            ]

            # Track the best result so far
            best_result = {"size": float(
                'inf'), "file": None, "settings": None}

            # Try each optimization step
            for i, settings in enumerate(optimization_steps):
                try:
                    result_file = temp_manager.create_temp_file(
                        custom_suffix=f"_step{i}.gif")

                    # Build gifsicle command with current settings
                    cmd = [
                        'gifsicle',
                        '--optimize=' + str(settings['optimize']),
                        '--colors', str(settings['colors'])
                    ]

                    # Add lossy option if specified
                    if settings['lossy'] > 0:
                        cmd.append('--lossy=' + str(settings['lossy']))

                    # Add scaling if specified
                    if 'scale' in settings:
                        cmd.append(f'--scale={settings["scale"]}')

                    # Add input and output files
                    cmd.extend(['-o', str(result_file), str(working_copy)])

                    # Run gifsicle
                    subprocess.run(cmd, check=True, capture_output=True)

                    # Check result size
                    if result_file.exists():
                        size_mb = result_file.stat().st_size / (1024 * 1024)
                        logger.info(
                            f"Step {i+1} result: {size_mb:.2f}MB (settings: {settings})")

                        # Update best result if this is the best so far
                        if size_mb <= target_size_mb and size_mb < best_result["size"]:
                            best_result = {
                                "size": size_mb, "file": result_file, "settings": settings}
                            logger.info(
                                f"New best result: {size_mb:.2f}MB (under target)")
                        elif size_mb < best_result["size"] and best_result["size"] > target_size_mb:
                            best_result = {
                                "size": size_mb, "file": result_file, "settings": settings}
                            logger.info(
                                f"New best result: {size_mb:.2f}MB (still over target)")

                        # If we're under target size, we can stop early
                        if size_mb <= target_size_mb:
                            logger.info(
                                f"Reached target size at step {i+1}, stopping optimization")
                            break
                except Exception as e:
                    logger.error(f"Error in optimization step {i+1}: {e}")

            # Use the best result if we found one
            if best_result["file"] is not None:
                try:
                    shutil.copy2(best_result["file"], output_path)
                    logger.info(
                        f"Optimized GIF: {best_result['size']:.2f}MB, settings: {best_result['settings']}")

                    # Check if we met the target
                    success = best_result["size"] <= target_size_mb
                    final_size = best_result["size"]

                    return final_size, success
                except Exception as e:
                    logger.error(f"Error copying best result: {e}")
                    return 0, False

            # If no good result found, try more extreme measures
            logger.warning(
                "No satisfactory result found with quality preservation, using fallback")

            # Fall back to more aggressive optimization
            return self._apply_fallback_optimization(input_path, output_path, target_size_mb, original_size, time.time())

    def _optimize_in_parallel(self, input_path: Path, output_path: Path, target_size_mb: float) -> OptimizationResult:
        """Optimize a GIF file using multiple parallel optimization techniques.

        This method tries different optimization techniques in parallel and picks the best result.

        Args:
            input_path: Path to the input GIF file
            output_path: Path where the optimized GIF should be saved
            target_size_mb: Target size in megabytes

        Returns:
            OptimizationResult containing the optimization result
        """
        logger.info(f"Starting parallel optimization for {input_path}")
        start_time = time.time()
        original_size = self.file_processor.get_file_size(input_path)

        # Define optimization techniques to try in parallel - prioritizing scaling and quality
        optimization_techniques = [
            {
                "name": "standard",
                "cmd": [
                    'gifsicle',
                    '--optimize=3',
                    '--colors=256',
                    str(input_path),
                ]
            },
            {
                "name": "scale_90",
                "cmd": [
                    'gifsicle',
                    '--optimize=3',
                    '--colors=256',
                    '--scale=0.9',
                    str(input_path),
                ]
            },
            {
                "name": "scale_80",
                "cmd": [
                    'gifsicle',
                    '--optimize=3',
                    '--colors=256',
                    '--scale=0.8',
                    str(input_path),
                ]
            },
            {
                "name": "scale_70",
                "cmd": [
                    'gifsicle',
                    '--optimize=3',
                    '--colors=256',
                    '--scale=0.7',
                    '--dither=bayer:bayer_scale=4',
                    str(input_path),
                ]
            },
            {
                "name": "scale_70_lossy",
                "cmd": [
                    'gifsicle',
                    '--optimize=3',
                    '--colors=256',
                    '--lossy=30',
                    '--scale=0.7',
                    '--dither=floyd-steinberg',
                    str(input_path),
                ]
            },
            {
                "name": "last_resort",
                "cmd": [
                    'gifsicle',
                    '--optimize=3',
                    '--colors=192',
                    '--lossy=50',
                    '--scale=0.7',
                    '--dither=floyd-steinberg',
                    str(input_path),
                ]
            }
        ]

        # Create a temporary output file for each technique
        temp_output_files = {}
        for technique in optimization_techniques:
            temp_output = self.temp_manager.create_temp_file(
                custom_prefix=f"parallel_{technique['name']}_")
            temp_output_files[technique["name"]] = temp_output

            # Add output path to the command
            if "frame_skip" not in technique:
                technique["cmd"].extend(['-o', str(temp_output)])

        # Define a function to run a single optimization technique
        def run_optimization(technique):
            try:
                if "frame_skip" in technique:
                    # Special handling for frame reduction
                    frame_skip = technique["frame_skip"]
                    temp_output = temp_output_files[technique["name"]]

                    # Extract every nth frame
                    skip_frames_cmd = technique["cmd"].copy()

                    # Add frame selection for every nth frame
                    for i in range(0, frame_count, frame_skip):
                        skip_frames_cmd.extend(['--select', f'#{i}'])

                    skip_frames_cmd.extend(['-o', str(temp_output)])

                    # Run the command
                    process = subprocess.run(
                        skip_frames_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        timeout=120  # 2 minute timeout
                    )
                else:
                    # Regular optimization technique
                    process = subprocess.run(
                        technique["cmd"],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                        check=False,
                        timeout=120  # 2 minute timeout
                    )

                temp_output = temp_output_files[technique["name"]]

                if process.returncode == 0 and temp_output.exists():
                    result_size = self.file_processor.get_file_size(
                        temp_output)
                    logger.info(
                        f"Parallel {technique['name']} result: {result_size:.2f}MB")
                    return {
                        "name": technique["name"],
                        "size": result_size,
                        "path": temp_output,
                        "success": True,
                        "quality_score": self._estimate_quality_score(technique)
                    }
                else:
                    logger.warning(
                        f"Parallel {technique['name']} optimization failed: {process.stderr}")
                    return {
                        "name": technique["name"],
                        "success": False
                    }
            except Exception as e:
                logger.error(f"Error in {technique['name']} optimization: {e}")
                return {
                    "name": technique["name"],
                    "success": False
                }

        # Run all optimization techniques in parallel
        all_results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(run_optimization, technique): technique["name"]
                       for technique in optimization_techniques}

            for future in concurrent.futures.as_completed(futures):
                technique_name = futures[future]
                try:
                    result = future.result()
                    if result["success"]:
                        all_results.append(result)
                except Exception as e:
                    logger.error(
                        f"Error getting result for {technique_name}: {e}")

        # Find the best result that meets the target size
        valid_results = [r for r in all_results if r["size"] <= target_size_mb]

        if valid_results:
            # Sort by quality score (highest first)
            valid_results.sort(key=lambda r: r["quality_score"], reverse=True)
            best_result = valid_results[0]

            # Copy the best result to the output path
            shutil.copy2(best_result["path"], output_path)

            logger.info(f"Selected {best_result['name']} optimization: {best_result['size']:.2f}MB with "
                        f"quality score {best_result['quality_score']:.2f}")

            # Create the final result object
            result = OptimizationResult(
                success=True,
                file_path=output_path,
                size_mb=best_result["size"],
                quality_score=best_result["quality_score"],
                settings={"technique": best_result["name"]},
                error=None
            )
        else:
            # If no technique met the target size, pick the smallest one
            if all_results:
                # Sort by size (smallest first)
                all_results.sort(key=lambda r: r["size"])
                best_result = all_results[0]

                # Copy the best result to the output path
                shutil.copy2(best_result["path"], output_path)

                logger.info(f"No technique met target size. Selected {best_result['name']} optimization: "
                            f"{best_result['size']:.2f}MB (target: {target_size_mb:.2f}MB)")

                # Create the result object (marked as not successful since it didn't meet target)
                result = OptimizationResult(
                    success=False,  # Doesn't meet target size
                    file_path=output_path,
                    size_mb=best_result["size"],
                    quality_score=best_result["quality_score"],
                    settings={"technique": best_result["name"]},
                    error="No technique met target size"
                )
            else:
                logger.error("All parallel optimization techniques failed")
                result = OptimizationResult(
                    success=False,
                    file_path=None,
                    size_mb=original_size,
                    quality_score=0.0,
                    settings={},
                    error="All parallel optimization techniques failed"
                )

        # Clean up temporary files
        self.temp_manager.batch_cleanup(list(temp_output_files.values()))

        # Log execution time
        elapsed = time.time() - start_time
        logger.info(
            f"Parallel optimization completed in {elapsed:.2f} seconds")

        return result

    def _estimate_quality_score(self, technique) -> float:
        """Estimate quality score based on optimization parameters."""
        # Base quality score
        quality = 1.0

        # Reduce quality based on color reduction
        if '--colors=' in ' '.join(technique["cmd"]):
            for part in technique["cmd"]:
                if part.startswith('--colors='):
                    colors = int(part.split('=')[1])
                    quality *= min(1.0, max(0.1, colors / 256))
                    break

        # Reduce quality based on lossy compression
        if '--lossy=' in ' '.join(technique["cmd"]):
            for part in technique["cmd"]:
                if part.startswith('--lossy='):
                    lossy = int(part.split('=')[1])
                    quality *= min(1.0, max(0.1, 1.0 - (lossy / 200)))
                    break

        # Reduce quality based on scaling
        if '--scale=' in ' '.join(technique["cmd"]):
            for part in technique["cmd"]:
                if part.startswith('--scale='):
                    scale = float(part.split('=')[1])
                    quality *= scale
                    break

        # Boost quality slightly if dithering is used (better visual quality with fewer colors)
        if '--dither=' in ' '.join(technique["cmd"]):
            quality = min(1.0, quality * 1.1)

        # Reduce quality if frame skip is applied
        if "frame_skip" in technique:
            quality *= max(0.5, 1.0 - (technique["frame_skip"] * 0.15))

        return max(0.1, min(1.0, quality))

    def _detect_frame_count(self, gif_path: Path, default_count: int = 1) -> int:
        """Detect the number of frames in a GIF using multiple fallback approaches.

        Args:
            gif_path: Path to the GIF file
            default_count: Default frame count to return if detection fails

        Returns:
            Detected frame count or default value if detection fails
        """
        # Add frame count caching to avoid repeatedly analyzing the same file
        file_hash = self._get_file_hash(gif_path)
        cache_key = f"frames_{file_hash}"

        with self.cache_lock:
            if cache_key in self.optimization_cache:
                # Use cached frame count if available
                frame_count = self.optimization_cache[cache_key]
                logger.debug(
                    f"Using cached frame count for {gif_path.name}: {frame_count}")
                # Update LRU status
                self.optimization_cache.move_to_end(cache_key)
                return frame_count

        if not gif_path.exists():
            logger.warning(
                f"Cannot detect frames: file {gif_path} doesn't exist")
            return default_count

        # Try multiple methods to detect frame count with fallbacks
        frame_count = self._detect_frames_with_gifsicle(gif_path)

        # If gifsicle method fails, try alternative method
        if frame_count <= 0:
            frame_count = self._detect_frames_alternative(gif_path)

        # If all methods fail, return the default
        if frame_count <= 0:
            logger.warning(
                f"Failed to detect frame count for {gif_path}, using default: {default_count}")
            return default_count

        # Cache the result for future use
        with self.cache_lock:
            self.optimization_cache[cache_key] = frame_count
            # Maintain LRU order
            self.optimization_cache.move_to_end(cache_key)
            # Remove oldest entry if cache is full
            if len(self.optimization_cache) > self.cache_max_size:
                self.optimization_cache.popitem(last=False)

        return frame_count

    def _detect_frames_alternative(self, gif_path: Path) -> int:
        """Alternative method to detect frame count by unpacking the GIF.

        Args:
            gif_path: Path to the GIF file

        Returns:
            Detected frame count or 0 if detection fails
        """
        try:
            # Create a temporary directory for exploded frames
            temp_dir = Path(TEMP_FILE_DIR) / f"frames_{uuid.uuid4().hex}"
            os.makedirs(temp_dir, exist_ok=True)

            try:
                # Try to explode the GIF into separate frames
                explode_cmd = [
                    'gifsicle',
                    '--explode',
                    '--output', str(temp_dir),
                    str(gif_path.absolute())
                ]

                # Use a timeout to prevent hanging on problematic files
                process = subprocess.run(
                    explode_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=False,
                    timeout=30  # 30 seconds timeout should be enough
                )

                # Count the number of files created
                frame_files = list(temp_dir.glob('*.gif'))
                frame_count = len(frame_files)

                # Log the result
                if frame_count > 0:
                    logger.debug(
                        f"Alternative frame detection found {frame_count} frames in {gif_path.name}")
                    return frame_count
                else:
                    logger.debug(
                        f"Alternative frame detection failed to find frames in {gif_path.name}")
                    return 0

            finally:
                # Clean up temp directory
                shutil.rmtree(temp_dir, ignore_errors=True)

        except subprocess.TimeoutExpired:
            logger.warning(
                f"Alternative frame detection timed out for {gif_path}")
            return 0
        except Exception as e:
            logger.warning(f"Error in alternative frame detection: {e}")
            return 0

    def _run_subprocess(self, cmd: List[str], timeout: int = 180, check: bool = False) -> subprocess.CompletedProcess:
        """Run a subprocess with improved error handling and resource management.

        Args:
            cmd: Command to run as a list of strings
            timeout: Timeout in seconds
            check: Whether to check the return code and raise an exception on failure

        Returns:
            CompletedProcess instance
        """
        try:
            # Set environment variables to limit memory usage in subprocess
            env = os.environ.copy()
            # Try to set resource limits for the subprocess
            with contextlib.ExitStack() as stack:
                try:
                    import resource
                    # Soft limit of 2GB memory per process
                    resource.setrlimit(resource.RLIMIT_AS,
                                       (2 * 1024 * 1024 * 1024, -1))
                except (ImportError, ValueError, resource.error):
                    pass

                # Execute the command with timeout
                return subprocess.run(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    check=check,
                    timeout=timeout,
                    env=env
                )
        except subprocess.TimeoutExpired as e:
            logger.error(
                f"Command timed out after {timeout} seconds: {' '.join(cmd)}")
            raise
        except Exception as e:
            logger.error(f"Error running command: {' '.join(cmd)}: {e}")
            raise
