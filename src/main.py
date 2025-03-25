import argparse
import logging
import signal
import sys
from pathlib import Path
import time
import traceback
import gc
import shutil
import subprocess
import os
import json
import threading
from concurrent.futures import ThreadPoolExecutor

# Use relative imports since we're in the src package
from .default_config import (GIF_PASS_OVERS, INPUT_DIR, LOG_DIR, OUTPUT_DIR,
                             TEMP_FILE_DIR, GIF_COMPRESSION, VIDEO_COMPRESSION)
# Use package-level import - move GIFProcessor inside main to avoid circular imports
from .gif_operations import DynamicGIFOptimizer
from .gif_operations import GIFProcessor
from .gif_operations.enhanced_gif_optimizer import EnhancedGIFOptimizer
from .gpu_acceleration import setup_gpu_acceleration, get_optimal_settings
from .logging_system import setup_application_logging, clear_ffmpeg_log, setup_ffmpeg_logging, run_ffmpeg_command, run_ffmpeg_command, log_gif_progress, ICONS, ThreadSafeProgressLogger, display_progress_update
from .temp_file_manager import TempFileManager
from .video_optimization import VideoProcessor
from .utils.error_handler import VideoProcessingError

# Application display banner
BANNER = r"""

      ::::::::: ::::::::::: ::::::::  :::::::::  :::::::::   ::::::::  ::::::::: 
     :+:    :+:    :+:    :+:    :+: :+:    :+: :+:    :+: :+:    :+: :+:    :+: 
    +:+    +:+    +:+    +:+        +:+    +:+ +:+    +:+ +:+    +:+ +:+    +:+  
   +#+    +:+    +#+    +#++:++#++ +#+    +:+ +#++:++#:  +#+    +:+ +#++:++#+    
  +#+    +#+    +#+           +#+ +#+    +#+ +#+    +#+ +#+    +#+ +#+           
 #+#    #+#    #+#    #+#    #+# #+#    #+# #+#    #+# #+#    #+# #+#            
######### ########### ########  #########  ###    ###  ########  ###             

"""

# Helper function to run ffprobe and capture its output


def run_ffprobe_command(command, timeout=None):
    """Run ffprobe command and capture its output while redirecting logs to the ffmpeg log file.

    Args:
        command: Command list to run
        timeout: Optional timeout in seconds

    Returns:
        str: Command output if successful, empty string otherwise
    """
    import tempfile
    import os
    import subprocess
    import json

    # Create a temporary file to store the output
    with tempfile.NamedTemporaryFile(delete=False, mode='w+', suffix='.json') as temp_file:
        temp_path = temp_file.name

    try:
        # ffprobe doesn't support -y option, so we need to handle output differently
        # Modify the command to output to file using > redirection via shell
        cmd_str = ' '.join(command) + f' > "{temp_path}"'

        # Use subprocess directly with shell=True for redirection
        proc = subprocess.Popen(
            cmd_str,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        try:
            _, stderr = proc.communicate(timeout=timeout)
            success = proc.returncode == 0
        except subprocess.TimeoutExpired:
            proc.kill()
            return ""

        # Read the output file if command was successful
        if success and os.path.exists(temp_path) and os.path.getsize(temp_path) > 0:
            with open(temp_path, 'r') as f:
                return f.read()
        return ""
    finally:
        # Clean up the temporary file
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass


def setup_signal_handlers(processor):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger = logging.getLogger('app')
        logger.warning(
            "\nReceived interrupt signal, initiating immediate shutdown...")
        processor._immediate_shutdown_handler(signum, frame)
        # Add sys.exit to ensure the program terminates after cleanup
        sys.exit(0)

    # Register for both SIGINT (Ctrl+C) and SIGTERM
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Video and GIF optimization tool')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug logging')
    parser.add_argument('--no-gpu', action='store_true',
                        help='Disable GPU acceleration')
    parser.add_argument('--input-dir', type=Path,
                        help='Custom input directory')
    parser.add_argument('--output-dir', type=Path,
                        help='Custom output directory')
    parser.add_argument('--verbose', action='store_true',
                        help='Enable verbose debug logging')
    parser.add_argument('--log', type=Path,
                        help='Custom log file')
    parser.add_argument('--no-video', action='store_true',
                        help='Skip video processing')
    parser.add_argument('--no-gif', action='store_true',
                        help='Skip GIF processing')
    parser.add_argument('--gpu', action='store_true',
                        help='Enable GPU acceleration for video processing')
    parser.add_argument('--clean-temp', action='store_true',
                        help='Clean temporary files after processing')
    parser.add_argument('--preserve-temp', action='store_true',
                        help='Preserve temporary files after processing')
    parser.add_argument('--target-size', type=float,
                        help='Target size in MB for processed files')
    parser.add_argument('--high-quality', action='store_true',
                        help='Prioritize quality while still respecting target file size limits')
    return parser.parse_args()


def verify_dependencies() -> bool:
    logger = logging.getLogger('app')
    required_commands = ['ffmpeg', 'ffprobe', 'gifsicle']
    missing_commands = [
        cmd for cmd in required_commands if not shutil.which(cmd)]

    if missing_commands:
        logger.error(f"Missing required dependencies: {
            ', '.join(missing_commands)}")
        return False
    return True


def process_failed_items(failed_files: list[Path], pass_over_index: int) -> list[Path]:
    logger = logging.getLogger('app')
    logger.info(f"Starting optimization pass {pass_over_index + 1}")

    # Update compression settings for this pass
    pass_over = GIF_PASS_OVERS[pass_over_index]
    GIF_COMPRESSION.update(pass_over)

    processor = GIFProcessor()
    dynamic_optimizer = DynamicGIFOptimizer()
    remaining_failed = []

    for file_path in failed_files:
        try:
            source_file = Path(file_path)
            output_path = OUTPUT_DIR / f"{source_file.stem}.gif"
            is_video = source_file.suffix.lower() in ['.mp4', '.mkv', '.avi']

            # Try dynamic optimizer first
            if dynamic_optimizer.optimize_gif(source_file, output_path, GIF_COMPRESSION['min_size_mb'])[1]:
                continue

            # Fall back to standard processor
            if not processor.process_file(source_file, output_path, is_video):
                remaining_failed.append(file_path)

        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            remaining_failed.append(file_path)

    # Reset compression settings to default
    GIF_COMPRESSION.update({
        'fps_range': (15, 10),
        'colors': 256,
        'lossy_value': 55,
        'min_size_mb': 10,
        'min_width': 120,
        'min_height': 120
    })

    return remaining_failed


def process_videos(input_dir: Path, output_dir: Path, logger=None, gpu_enabled=False, gpu_settings=None) -> None:
    """Process videos with enhanced error handling and progress tracking."""
    app_logger = logging.getLogger('app')
    logger = logger or app_logger  # Use the shared logger
    processor = VideoProcessor(
        use_gpu=gpu_enabled, gpu_settings=gpu_settings, logger=logger)  # Pass GPU status

    try:
        # Ensure directories exist
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)

        # Process videos with progress tracking
        total_processed = 0
        last_progress_time = time.time()

        def log_progress(current: int, total: int, file_name: str) -> None:
            nonlocal last_progress_time
            current_time = time.time()

            # Only update progress every 5 seconds
            if current_time - last_progress_time >= 5:
                progress = (current / total) * 100
                logger.info(f"Progress: {progress:.1f}% ({current}/{total})")
                logger.info(f"Currently processing: {file_name}")
                # Print to console for visibility
                print(
                    f"[VIDEO] Progress: {progress:.1f}% ({current}/{total}) - {file_name}")
                last_progress_time = current_time

        # Add progress callback to processor
        processor.set_progress_callback(log_progress)

        # Print start message
        print(f"[{ICONS['STARTING']}] VIDEO PROCESSING STARTED")

        # Process videos without timeout parameter
        failed_files = processor.process_videos(
            input_dir,
            output_dir,
            target_size_mb=15.0
        )

        # Report results
        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} videos:")
            print(
                f"[{ICONS['WARNING']}] Failed to process {len(failed_files)} videos:")
            for file in failed_files:
                logger.warning(f"  - {file}")
                print(f"  - {file}")
        else:
            logger.info("All videos processed successfully")
            print(f"[{ICONS['SUCCESS']}] All videos processed successfully")

    except Exception as e:
        logger.error(f"Video processing error: {str(e)}", exc_info=True)
        print(f"[{ICONS['ERROR']}] Video processing failed: {str(e)}")
        sys.exit(1)


def process_gifs(input_dir: Path, output_dir: Path, logger=None, target_size_mb=None, prioritize_quality=False):
    """Process GIF files in the input directory and save to output directory.

    Args:
        input_dir: Directory containing input GIF files
        output_dir: Directory to save processed GIFs
        logger: Optional logger to use
        target_size_mb: Target size in MB (optional)
        prioritize_quality: Whether to prioritize quality over file size (optional)

    Returns:
        Dict with processing results
    """
    app_logger = logging.getLogger('app')
    logger = logger or app_logger

    # Initialize enhanced optimizer
    # Use min of 8 workers or CPU count
    max_workers = min(8, os.cpu_count() or 4)
    optimizer = EnhancedGIFOptimizer(max_workers=max_workers)

    try:
        # Ensure directories exist
        input_dir.mkdir(exist_ok=True)
        output_dir.mkdir(exist_ok=True)

        # Find GIF files to process
        logger.info("Finding GIF files to process...")
        log_gif_progress("Finding GIF files to process...", "starting")

        gif_files = list(input_dir.glob('*.gif'))
        logger.info(f"Found {len(gif_files)} GIF files to process")

        if not gif_files:
            logger.info("No GIF files found to process")
            log_gif_progress("No GIF files found to process", "skipped")
            return {}

        # Use default target size if not specified
        if target_size_mb is None:
            target_size_mb = GIF_COMPRESSION.get('min_size_mb', 10)

        # Process GIFs in parallel
        logger.info(
            f"Processing {len(gif_files)} GIFs with target size {target_size_mb}MB")

        quality_message = " (prioritizing quality)" if prioritize_quality else ""
        log_gif_progress(
            f"Processing {len(gif_files)} GIFs with target size {target_size_mb}MB{quality_message}", "processing")

        # Process files in batches for best performance and memory usage
        batch_size = 10
        results = {}

        for i in range(0, len(gif_files), batch_size):
            batch = gif_files[i:i+batch_size]
            logger.info(
                f"Processing batch {i//batch_size + 1}/{(len(gif_files) + batch_size - 1)//batch_size}")
            log_gif_progress(
                f"Processing batch {i//batch_size + 1}/{(len(gif_files) + batch_size - 1)//batch_size}", "processing")

            # Process this batch with quality-prioritized method if requested
            batch_results = optimizer.batch_optimize(
                batch, output_dir, target_size_mb, prioritize_quality=prioritize_quality)

            results.update(batch_results)

            # Force garbage collection between batches
            gc.collect()

        # Gather statistics
        total_files = len(gif_files)
        successful = sum(1 for success in results.values() if success[1])

        logger.info(
            f"GIF processing complete: {successful}/{total_files} successful")
        log_gif_progress(f"GIF processing complete: {successful}/{total_files} successful",
                         "success" if successful == total_files else "warning")

        # Log optimization statistics
        stats = optimizer.get_optimization_stats()
        logger.info(f"Optimization statistics: {stats}")
        if stats.get('bytes_saved', 0) > 0:
            mb_saved = stats['bytes_saved'] / (1024 * 1024)
            logger.info(f"Total size reduction: {mb_saved:.2f}MB")

        return results

    except Exception as e:
        logger.error(f"Error processing GIFs: {e}", exc_info=True)
        log_gif_progress(f"Error processing GIFs: {str(e)}", "error")
        return {}
    finally:
        # Clean up resources
        optimizer.cleanup()


def main() -> None:
    # Import GIFProcessor here to avoid circular import issues
    from pathlib import Path
    import shutil

    # Variable to hold processor reference for cleanup
    gif_processor = None

    # Initialize temp directory for file operations
    temp_dir = Path(TEMP_FILE_DIR)
    temp_dir.mkdir(parents=True, exist_ok=True)

    # Temp file tracking for final cleanup
    temp_files_to_clean = []

    # Parse command-line arguments
    args = parse_arguments()

    # Configure logging
    setup_application_logging(debug_mode=args.debug, verbose_mode=args.verbose)
    app_logger = logging.getLogger('app')

    # Verify dependencies are available
    if not verify_dependencies():
        app_logger.critical("Missing required dependencies")
        sys.exit(1)

    # Print startup message
    print(f"\n{BANNER}")
    print(f"[{ICONS['INFO']}] DisDrop started")
    print(f"[{ICONS['INFO']}] Input directory: {args.input_dir}")
    print(f"[{ICONS['INFO']}] Output directory: {args.output_dir}")

    if args.high_quality:
        print(
            f"[{ICONS['INFO']}] High quality mode enabled (target size will still be respected)")

    # Convert to Path objects
    input_dir = args.input_dir or INPUT_DIR
    output_dir = args.output_dir or OUTPUT_DIR

    for directory in [input_dir, output_dir, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Initialize FFmpeg log file - clear it at the start of each run
    clear_ffmpeg_log()
    setup_ffmpeg_logging()

    app_logger.info("FFmpeg logs will be written to ffmpeg.log")
    # Print to console for visibility
    print(f"[{ICONS['INFO']}] FFmpeg logs will be written to ffmpeg.log")

    try:
        # Initialize GPU acceleration once
        gpu_supported = False if args.no_gpu else setup_gpu_acceleration()
        app_logger.info(
            f"GPU acceleration {'enabled' if gpu_supported else 'disabled'}")
        # Print to console
        print(f"[{ICONS['SUCCESS' if gpu_supported else 'ERROR']}] GPU acceleration {'enabled' if gpu_supported else 'disabled'}")

        # Get optimal GPU settings to share with processors
        gpu_settings = get_optimal_settings() if gpu_supported else {}
        if gpu_supported:
            # Ensure no None values that might cause errors
            for key in list(gpu_settings.keys()):
                if gpu_settings[key] is None:
                    app_logger.warning(
                        f"Removing None value for GPU setting: {key}")
                    gpu_settings[key] = ""  # Replace None with empty string

            app_logger.info(f"Using GPU settings: {gpu_settings}")
            # Print to console
            print(f"[{ICONS['INFO']}] Using GPU settings: {gpu_settings}")

        # Initialize processor with GPU status from command line
        gif_processor = GIFProcessor(
            logger=app_logger, gpu_enabled=gpu_supported, gpu_settings=gpu_settings)

        # Setup signal handlers
        setup_signal_handlers(gif_processor)

        # Handle target size if specified
        target_size_mb = args.target_size

        # Set high quality mode by default unless specifically disabled
        high_quality_mode = True if not hasattr(
            args, 'low_quality') else not args.low_quality
        if not args.high_quality:  # If not explicitly set by user
            args.high_quality = high_quality_mode
            app_logger.info("High quality mode enabled by default")
            print(
                f"[{ICONS['INFO']}] High quality mode enabled by default (for better quality GIFs)")

        # Initialize stats and temp file tracking
        stats = {
            'start_time': time.time(),
            'videos_processed': 0,
            'gifs_processed': 0,
            'errors': 0,
            'files_skipped': 0
        }

        # Check if video processing is enabled
        if not args.no_video:
            try:
                video_start_time = time.time()
                app_logger.info("Starting video processing...")
                print(f"\n[{ICONS['STARTING']}] STARTING VIDEO PROCESSING")

                # Process videos with shared logging and GPU status
                process_videos(input_dir, output_dir, logger=app_logger,
                               gpu_enabled=gpu_supported, gpu_settings=gpu_settings)

                # Update statistics
                stats['videos_processed'] = 1
                stats['video_time'] = time.time() - video_start_time

                # Log completion
                app_logger.info(f"Video processing completed")
                print(f"[{ICONS['SUCCESS']}] Video processing completed")

            except Exception as e:
                app_logger.exception(f"Error in video processing: {e}")
                print(f"[{ICONS['ERROR']}] Error in video processing: {str(e)}")
                stats['errors'] += 1

        # Check if GIF processing is enabled
        if not args.no_gif:
            try:
                gif_start_time = time.time()
                app_logger.info("Starting GIF processing...")
                print(f"\n[{ICONS['STARTING']}] STARTING GIF PROCESSING")

                # Process all GIFs in input directory
                gif_results = process_gifs(
                    input_dir, output_dir, app_logger, target_size_mb, args.high_quality)

                # Update statistics
                stats['gifs_processed'] = len(gif_results)
                stats['gif_time'] = time.time() - gif_start_time

                # Log completion
                app_logger.info(f"GIF processing completed")
                print(f"[{ICONS['SUCCESS']}] GIF processing completed")

            except Exception as e:
                app_logger.exception(f"Error in GIF processing: {e}")
                print(f"[{ICONS['ERROR']}] Error in GIF processing: {str(e)}")
                stats['errors'] += 1

        # Process MP4 files in output directory
        app_logger.info("Processing MP4 files in output directory...")
        print(f"\n[{ICONS['STARTING']}] STARTING MP4 TO GIF CONVERSION")

        # Find MP4 files in output directory
        mp4_files = list(output_dir.glob('*.mp4'))
        if mp4_files:
            app_logger.info(
                f"Found {len(mp4_files)} MP4 files to convert to GIFs")
            print(
                f"[{ICONS['SUCCESS']}] Found {len(mp4_files)} MP4 files to convert")

            # Create enhanced optimizer for batch usage
            enhanced_optimizer = EnhancedGIFOptimizer(max_workers=4)
            print(
                f"[{ICONS['INFO']}] Preparing GIF optimizer with {4} worker threads. Starting conversion process...")
            print(
                f"[{ICONS['INFO']}] The conversion process is CPU-intensive and may appear to pause between updates.")

            # Create a thread-safe progress logger for overall conversion
            progress_logger = ThreadSafeProgressLogger(
                total=len(mp4_files),
                desc="MP4 to GIF Conversion",
                unit="files"
            )

            # Add progress counter variables
            conversion_start_time = time.time()
            files_completed = 0
            total_mp4_files = len(mp4_files)
            print(
                f"[{ICONS['INFO']}] Starting conversion of {total_mp4_files} files. This may take a while...")

            # Initial progress bar display
            display_progress_update(
                0, total_mp4_files,
                description="MP4 to GIF Conversion",
                start_time=conversion_start_time,
                status="STARTING",
                is_new_line=True
            )

            # Process MP4 files
            for i, mp4_file in enumerate(mp4_files, 1):
                file_start_time = time.time()

                # Show detailed progress for current file
                display_progress_update(
                    files_completed, total_mp4_files,
                    description="MP4 to GIF Conversion",
                    file_name=mp4_file.name,
                    status="PROCESSING",
                    start_time=conversion_start_time
                )

                try:
                    progress_message = f"Processing GIF {i}/{len(mp4_files)}: {mp4_file.name}"
                    app_logger.info(progress_message)
                    # Use log_gif_progress for better visual indicators
                    log_gif_progress(
                        f"Processing file {i}/{len(mp4_files)}: {mp4_file.name}", "processing")

                    # Generate temporary and final paths
                    temp_gif = temp_dir / f"{mp4_file.stem}_temp.gif"
                    final_gif = output_dir / f"{mp4_file.stem}.gif"

                    # Track temp files for cleanup
                    temp_files_to_clean.append(temp_gif)

                    # Skip if final already exists and is good size
                    if final_gif.exists() and final_gif.stat().st_size > 0:
                        file_size_mb = final_gif.stat().st_size / (1024 * 1024)
                        target_size_mb = GIF_COMPRESSION.get('min_size_mb', 10)
                        if file_size_mb <= target_size_mb:
                            app_logger.info(
                                f"Skipping {mp4_file.name} - Already converted ({file_size_mb:.2f}MB)")
                            log_gif_progress(
                                f"Skipping {mp4_file.name} - Already converted ({file_size_mb:.2f}MB)", "skipped")

                            # Update progress display with skip status
                            display_progress_update(
                                files_completed + 1, total_mp4_files,
                                description="MP4 to GIF Conversion",
                                file_name=mp4_file.name,
                                status="SKIPPED",
                                start_time=conversion_start_time
                            )

                            files_completed += 1
                            progress_logger.update(
                                1, f"Skipped: {mp4_file.name}")
                            continue

                    # Step 1: Convert MP4 to temporary GIF
                    app_logger.info(f"Converting {mp4_file.name} to GIF...")
                    log_gif_progress(
                        f"Converting {mp4_file.name} to GIF", "processing")

                    # Update progress with detail about current operation
                    display_progress_update(
                        files_completed, total_mp4_files,
                        description="MP4 to GIF Conversion - Creating palette",
                        file_name=mp4_file.name,
                        status="PROCESSING",
                        start_time=conversion_start_time
                    )

                    # Get video dimensions
                    probe_cmd = [
                        'ffprobe', '-v', 'error',
                        '-select_streams', 'v:0',
                        '-show_entries', 'stream=width,height',
                        '-of', 'json', str(mp4_file)
                    ]
                    probe_result = run_ffprobe_command(probe_cmd)

                    import json
                    probe_data = json.loads(
                        probe_result) if probe_result else {}

                    # Default dimensions if probe fails
                    width = 640
                    height = 360

                    if 'streams' in probe_data and probe_data['streams']:
                        width = probe_data['streams'][0].get('width', 640)
                        height = probe_data['streams'][0].get('height', 360)

                    # Revised scaling strategy: prioritize scaling over lossy compression
                    # Use more gradual scaling that preserves sharpness
                    scale_factor = 1.0
                    max_dimension = max(width, height)

                    # More aggressive scaling for very large videos
                    if max_dimension > 1920:
                        scale_factor = 0.65  # More aggressive scaling for very large videos
                        log_gif_progress(
                            f"Scaling down large video (65%): {width}x{height}", "processing")

                        # Update progress display with scaling info
                        display_progress_update(
                            files_completed, total_mp4_files,
                            description=f"MP4 to GIF - Scaling down large video (65%)",
                            file_name=mp4_file.name,
                            status="PROCESSING",
                            start_time=conversion_start_time
                        )
                    elif max_dimension > 1440:
                        scale_factor = 0.70
                        log_gif_progress(
                            f"Scaling down video (70%): {width}x{height}", "processing")
                        display_progress_update(
                            files_completed, total_mp4_files,
                            description=f"MP4 to GIF - Scaling down video (70%)",
                            file_name=mp4_file.name,
                            status="PROCESSING",
                            start_time=conversion_start_time
                        )
                    elif max_dimension > 1080:
                        scale_factor = 0.80
                        log_gif_progress(
                            f"Scaling down video (80%): {width}x{height}", "processing")
                        display_progress_update(
                            files_completed, total_mp4_files,
                            description=f"MP4 to GIF - Scaling down video (80%)",
                            file_name=mp4_file.name,
                            status="PROCESSING",
                            start_time=conversion_start_time
                        )
                    elif max_dimension > 720:
                        scale_factor = 0.90  # Less aggressive for smaller videos
                        log_gif_progress(
                            f"Scaling down video (90%): {width}x{height}", "processing")
                        display_progress_update(
                            files_completed, total_mp4_files,
                            description=f"MP4 to GIF - Scaling down video (90%)",
                            file_name=mp4_file.name,
                            status="PROCESSING",
                            start_time=conversion_start_time
                        )
                    else:
                        # For smaller videos, keep high resolution to maintain sharpness
                        scale_factor = 1.0
                        log_gif_progress(
                            f"Maintaining original size for sharpness: {width}x{height}", "processing")
                        display_progress_update(
                            files_completed, total_mp4_files,
                            description=f"MP4 to GIF - Maintaining original size for sharpness: {width}x{height}",
                            file_name=mp4_file.name,
                            status="PROCESSING",
                            start_time=conversion_start_time
                        )

                    new_width = int(width * scale_factor // 2 * 2)
                    new_height = int(height * scale_factor // 2 * 2)

                    # Create palette
                    palette_file = temp_dir / f"palette_{mp4_file.stem}.png"

                    # Track temp files for cleanup
                    temp_files_to_clean.append(palette_file)

                    # Create palette with more colors for higher quality (improved)
                    log_gif_progress(
                        "Creating enhanced color palette...", "processing")
                    palette_cmd = [
                        'ffmpeg', '-i', str(mp4_file),
                        # Use full quality input frame sampling for palette
                        '-vf', f'fps=24,scale={new_width}:{new_height}:flags=lanczos,palettegen=max_colors=256:stats_mode=diff:reserve_transparent=0',
                        '-y', str(palette_file)
                    ]

                    # Log the palette generation settings
                    app_logger.info(
                        f"GIF Palette settings: fps=24, scale={new_width}x{new_height}, max_colors=256, stats_mode=diff:reserve_transparent=0")
                    log_gif_progress("Creating palette", "processing", {
                        "width": new_width,
                        "height": new_height,
                        "colors": 256,
                        "fps": 24
                    })

                    if not run_ffmpeg_command(palette_cmd):
                        app_logger.error("Failed to create palette")
                        log_gif_progress(
                            "Failed to create color palette", "error")
                        continue  # Skip this file instead of raising exception

                    # Create GIF with quality settings optimized for 10MB limit
                    log_gif_progress(
                        "Converting video frames to GIF with quality optimizations...", "processing")

                    # Use a two-pass approach for better quality within size constraints

                    # First create a high-quality temporary GIF
                    temp_hq_gif = temp_dir / f"{mp4_file.stem}_hq_temp.gif"
                    temp_files_to_clean.append(temp_hq_gif)

                    # High quality settings for initial conversion (improved)
                    hq_gif_cmd = [
                        'ffmpeg', '-i', str(mp4_file),
                        '-i', str(palette_file),
                        # Use better scaling algorithm and improved dithering
                        '-lavfi', f'fps=20,scale={new_width}:{new_height}:flags=lanczos+accurate_rnd[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
                        '-y', str(temp_hq_gif)
                    ]

                    # Log the high quality GIF conversion settings
                    app_logger.info(
                        f"High Quality GIF Conversion: fps=20, scale={new_width}x{new_height}, dither=bayer:bayer_scale=5")
                    log_gif_progress("Creating high quality base GIF", "processing", {
                        "width": new_width,
                        "height": new_height,
                        "fps": 20,
                        "dither": "bayer:bayer_scale=5",
                        "diff_mode": "rectangle"
                    })

                    if not run_ffmpeg_command(hq_gif_cmd):
                        app_logger.error("Failed to create high quality GIF")
                        log_gif_progress(
                            "Failed to create high quality GIF", "error")
                        # Fall back to standard approach (improved)
                        gif_cmd = [
                            'ffmpeg', '-i', str(mp4_file),
                            '-i', str(palette_file),
                            # Use a less aggressive dithering approach
                            '-lavfi', f'fps=15,scale={new_width}:{new_height}:flags=lanczos+accurate_rnd[x];[x][1:v]paletteuse=dither=floyd_steinberg:diff_mode=rectangle',
                            '-y', str(temp_gif)
                        ]
                        if not run_ffmpeg_command(gif_cmd):
                            app_logger.error(
                                "Failed to create GIF with fallback method")
                            log_gif_progress("Failed to create GIF", "error")
                            continue
                    else:
                        # Check size of high-quality GIF
                        hq_size_mb = temp_hq_gif.stat().st_size / (1024 * 1024)
                        if hq_size_mb <= GIF_COMPRESSION.get('min_size_mb', 10):
                            # If it's already under size limit, use it directly
                            shutil.copy2(temp_hq_gif, temp_gif)
                            app_logger.info(
                                f"High quality GIF is already under size limit: {hq_size_mb:.2f}MB")
                            log_gif_progress(
                                f"Using high quality GIF directly: {hq_size_mb:.2f}MB", "processing")
                        else:
                            # Otherwise, use the high-quality GIF as input to gifsicle for optimization
                            app_logger.info(
                                f"Optimizing high quality GIF: {hq_size_mb:.2f}MB")
                            log_gif_progress(
                                f"Optimizing high quality GIF: {hq_size_mb:.2f}MB", "processing")

                            # Use gifsicle to optimize with minimal quality loss
                            # Prioritize higher color count over lossy compression
                            gifsicle_cmd = [
                                'gifsicle', '--optimize=3',
                                # Always try to keep maximum colors first
                                '--colors', str(256),
                                # Use mild lossy compression
                                '--lossy=' + str(30),
                                '-o', str(temp_gif),
                                str(temp_hq_gif)
                            ]

                            # Use subprocess directly for gifsicle
                            try:
                                subprocess.run(
                                    gifsicle_cmd, check=True, capture_output=True)
                                app_logger.info(
                                    "Gifsicle optimization successful")
                                log_gif_progress(
                                    "Gifsicle optimization successful", "processing")
                            except subprocess.CalledProcessError as e:
                                app_logger.error(
                                    f"Gifsicle optimization failed: {e}")
                                log_gif_progress(
                                    "Gifsicle optimization failed", "error")
                                # Fall back to copying the high-quality GIF
                                shutil.copy2(temp_hq_gif, temp_gif)

                    # Optimize the temporary GIF
                    if temp_gif.exists():
                        # Use config value for target size (strictly 10MB)
                        target_size_mb = GIF_COMPRESSION.get('min_size_mb', 10)
                        initial_size_mb = temp_gif.stat().st_size / (1024 * 1024)

                        app_logger.info(
                            f"Original GIF size: {initial_size_mb:.2f}MB, optimizing...")
                        log_gif_progress(
                            f"Optimizing GIF", "optimizing", {
                                "original_size": f"{initial_size_mb:.2f}MB",
                                "target_size": f"{target_size_mb:.2f}MB",
                                "fps_range": f"{GIF_COMPRESSION.get('fps_range')}",
                                "colors": GIF_COMPRESSION.get('colors'),
                                "lossy": GIF_COMPRESSION.get('lossy_value')
                            })

                        # Update progress display for optimization stage
                        display_progress_update(
                            files_completed, total_mp4_files,
                            description=f"MP4 to GIF - Optimizing GIF ({initial_size_mb:.2f}MB → {target_size_mb:.2f}MB)",
                            file_name=mp4_file.name,
                            status="OPTIMIZING",
                            start_time=conversion_start_time
                        )

                        # Log optimization settings from configuration
                        app_logger.info(f"GIF Optimization settings: target_size={target_size_mb:.2f}MB, " +
                                        f"fps_range={GIF_COMPRESSION.get('fps_range')}, " +
                                        f"colors={GIF_COMPRESSION.get('colors')}, " +
                                        f"lossy_value={GIF_COMPRESSION.get('lossy_value')}")

                        # Optimize GIF (now with quality prioritization if enabled)
                        if args.high_quality:
                            size, success = enhanced_optimizer.optimize_gif_prioritize_quality(
                                temp_gif, final_gif, target_size_mb)
                        else:
                            size, success = enhanced_optimizer.optimize_gif(
                                temp_gif, final_gif, target_size_mb)

                        # Log results
                        if success:
                            reduction = ((initial_size_mb - size) /
                                         initial_size_mb) * 100 if initial_size_mb > 0 else 0
                            app_logger.info(
                                f"GIF optimization successful: {initial_size_mb:.2f}MB → {size:.2f}MB ({reduction:.1f}% reduction)")
                            log_gif_progress(
                                "Optimization successful", "success", {
                                    "original_size": f"{initial_size_mb:.2f}MB",
                                    "final_size": f"{size:.2f}MB",
                                    "reduction": f"{reduction:.1f}%",
                                    "ratio": f"{initial_size_mb/size:.2f}x"
                                })

                            # Update progress display for success
                            display_progress_update(
                                files_completed + 1, total_mp4_files,
                                description=f"MP4 to GIF - Success: {size:.2f}MB ({reduction:.1f}% reduction)",
                                file_name=mp4_file.name,
                                status="SUCCESS",
                                start_time=conversion_start_time
                            )

                            # Log detailed optimization metrics
                            app_logger.info(f"GIF optimization metrics: " +
                                            f"Size reduced by {(initial_size_mb - size):.2f}MB, " +
                                            f"Compression ratio: {initial_size_mb/size:.2f}x, " +
                                            f"Final file: {final_gif}")
                        else:
                            app_logger.warning(
                                f"GIF optimization failed, using original conversion")
                            log_gif_progress(
                                "Optimization failed", "warning", {
                                    "original_size": f"{initial_size_mb:.2f}MB",
                                    "target_size": f"{target_size_mb:.2f}MB",
                                    "reason": "Could not reach target size"
                                })

                            # Update progress display for failure
                            display_progress_update(
                                files_completed + 1, total_mp4_files,
                                description=f"MP4 to GIF - Warning: Using unoptimized GIF",
                                file_name=mp4_file.name,
                                status="WARNING",
                                start_time=conversion_start_time
                            )

                            # Log the reason for the failure if possible
                            app_logger.warning(f"GIF optimization failed: Could not reach target size of {target_size_mb:.2f}MB " +
                                               f"from original size {initial_size_mb:.2f}MB")

                            # Copy the temporary GIF as fallback
                            if temp_gif.exists():
                                shutil.copy2(temp_gif, final_gif)
                    else:
                        app_logger.error(
                            f"Failed to create temporary GIF: {temp_gif}")
                        log_gif_progress(
                            f"Failed to create temporary GIF", "error")

                        # Update progress display for error
                        display_progress_update(
                            files_completed, total_mp4_files,
                            description=f"MP4 to GIF - Error: Failed to create temporary GIF",
                            file_name=mp4_file.name,
                            status="ERROR",
                            start_time=conversion_start_time
                        )
                except Exception as e:
                    app_logger.error(f"Error processing {mp4_file.name}: {e}")
                    log_gif_progress(
                        f"Error processing {mp4_file.name}: {str(e)}", "error")

                    # Update progress display for exception
                    display_progress_update(
                        files_completed, total_mp4_files,
                        description=f"MP4 to GIF - Error: {str(e)[:50]}...",
                        file_name=mp4_file.name,
                        status="ERROR",
                        start_time=conversion_start_time
                    )

                    # Add error to progress logger
                    progress_logger.add_error(
                        f"Error with {mp4_file.name}: {str(e)}")

                # Clean up temp files
                try:
                    for file in [temp_gif, palette_file]:
                        if file.exists():
                            file.unlink(missing_ok=True)
                except Exception as e:
                    app_logger.warning(
                        f"Failed to clean up temporary files: {e}")

                # Success counter for progress tracking
                files_completed += 1

                # Update thread-safe progress logger
                file_time = time.time() - file_start_time
                file_time_str = f"{int(file_time // 60)}m {int(file_time % 60)}s"
                progress_logger.update(
                    1, f"Completed: {mp4_file.name} in {file_time_str}")

                # Show completion status for each file
                file_percent = (files_completed / total_mp4_files) * 100
                print(
                    f"[{ICONS['SUCCESS']}] Completed {files_completed}/{total_mp4_files} ({file_percent:.1f}%): {mp4_file.name}")

                # Force memory cleanup and pause between files
                gc.collect()
                time.sleep(1)

            # Clean up enhanced optimizer resources
            enhanced_optimizer.cleanup()

            # Close the progress logger
            progress_logger.close()

            # Show final progress bar at 100%
            display_progress_update(
                total_mp4_files, total_mp4_files,
                description="MP4 to GIF Conversion - Complete",
                status="SUCCESS",
                start_time=conversion_start_time
            )

            # Log summary metrics for the entire MP4 to GIF conversion process
            total_original_size = 0
            total_final_size = 0
            successful_conversions = 0

            # Calculate total conversion time
            total_conversion_time = time.time() - conversion_start_time
            total_minutes = int(total_conversion_time // 60)
            total_seconds = int(total_conversion_time % 60)
            avg_time_per_file = total_conversion_time / max(1, files_completed)
            avg_time_str = f"{int(avg_time_per_file // 60)}m {int(avg_time_per_file % 60)}s"

            # Print comprehensive conversion summary
            print(f"\n[{ICONS['INFO']}] MP4 TO GIF CONVERSION COMPLETED")
            print(
                f"[{ICONS['TIME']}] Total conversion time: {total_minutes}m {total_seconds}s")
            print(
                f"[{ICONS['INFO']}] Files processed: {files_completed}/{len(mp4_files)}")
            print(f"[{ICONS['INFO']}] Average time per file: {avg_time_str}")

            # Gather statistics for MP4 files that were converted to GIFs
            for mp4_file in mp4_files:
                final_gif = output_dir / f"{mp4_file.stem}.gif"
                if final_gif.exists():
                    successful_conversions += 1
                    mp4_size = mp4_file.stat().st_size / (1024 * 1024)  # MB
                    gif_size = final_gif.stat().st_size / (1024 * 1024)  # MB
                    total_original_size += mp4_size
                    total_final_size += gif_size

            if successful_conversions > 0:
                avg_reduction = ((total_original_size - total_final_size) /
                                 total_original_size) * 100 if total_original_size > 0 else 0
                app_logger.info(
                    f"MP4 to GIF conversion summary: {successful_conversions}/{len(mp4_files)} files completed")
                app_logger.info(
                    f"Total size: Original MP4s: {total_original_size:.2f}MB, Final GIFs: {total_final_size:.2f}MB")
                app_logger.info(
                    f"Average reduction: {avg_reduction:.1f}%, Average compression ratio: {total_original_size/total_final_size:.2f}x")
                log_gif_progress("MP4 to GIF conversion complete", "success", {
                    "files": f"{successful_conversions}/{len(mp4_files)}",
                    "original_size": f"{total_original_size:.2f}MB",
                    "final_size": f"{total_final_size:.2f}MB",
                    "reduction": f"{avg_reduction:.1f}%",
                    "ratio": f"{total_original_size/total_final_size:.2f}x"
                })
        else:
            app_logger.info("No MP4 files found for conversion")
            print(f"[{ICONS['WARNING']}] No MP4 files found for conversion")

        # Clean up resources
        app_logger.info("Cleaning up resources...")
        if gif_processor:
            gif_processor.cleanup_resources()

        app_logger.info("All processing completed successfully")
        print(f"[{ICONS['SUCCESS']}] All processing completed successfully")

    except KeyboardInterrupt:
        app_logger.warning("\nProcess interrupted by user")
        log_gif_progress("Process interrupted by user", "warning")
        # Clean temp files before exiting
        cleanup_temp_files(temp_files_to_clean, temp_dir, app_logger)
        if gif_processor:  # Only call if gif_processor was successfully created
            gif_processor._immediate_shutdown_handler(signal.SIGINT, None)
        sys.exit(0)  # Ensure we exit here
    except Exception as e:
        app_logger.critical(f"Fatal error: {e}", exc_info=True)
        # Clean temp files before exiting
        cleanup_temp_files(temp_files_to_clean, temp_dir, app_logger)
        if gif_processor:  # Only call if gif_processor was successfully created
            gif_processor._immediate_shutdown_handler(signal.SIGTERM, None)
        sys.exit(1)  # Exit with error code
    finally:
        # One last attempt at cleanup
        if gif_processor:
            try:
                gif_processor._enhanced_temp_directory_cleanup()
            except Exception as e:
                app_logger.error(f"Error in final cleanup: {e}")


def cleanup_temp_files(tracked_files, temp_dir, logger):
    """Clean up temporary files and perform general temp directory cleanup."""
    try:
        # First, clean tracked files
        for temp_file in tracked_files:
            try:
                if temp_file.exists():
                    temp_file.unlink(missing_ok=True)
            except Exception as e:
                logger.error(f"Error removing temp file {temp_file}: {e}")

        # Then clean up by pattern
        patterns = ["*.gif", "*.png", "palette_*",
                    "lossless_*", "opt*_*.gif", "temp_*"]
        for pattern in patterns:
            try:
                for file_path in temp_dir.glob(pattern):
                    if file_path.is_file():
                        try:
                            file_path.unlink(missing_ok=True)
                        except Exception as e:
                            logger.error(f"Error cleaning up {file_path}: {e}")
            except Exception as e:
                logger.error(f"Error processing pattern {pattern}: {e}")

        # Force garbage collection to release file handles
        gc.collect()
    except Exception as e:
        logger.error(f"Error in cleanup_temp_files: {e}")


if __name__ == "__main__":
    main()
