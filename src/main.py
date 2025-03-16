import argparse
import logging
import signal
import sys
from pathlib import Path
import time
import traceback
import gc

# Use relative imports since we're in the src package
from .default_config import (GIF_PASS_OVERS, INPUT_DIR, LOG_DIR, OUTPUT_DIR,
                             TEMP_FILE_DIR, GIF_COMPRESSION, VIDEO_COMPRESSION)
# Use package-level import
from .gif_operations import GIFProcessor, DynamicGIFOptimizer
from .gpu_acceleration import setup_gpu_acceleration, get_optimal_settings
from .logging_system import setup_application_logging, clear_ffmpeg_log, setup_ffmpeg_logging
from .temp_file_manager import TempFileManager
from .video_optimization import VideoProcessor
from .utils.error_handler import VideoProcessingError


def setup_signal_handlers(processor):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        logger = logging.getLogger('app')
        logger.warning(
            "\nReceived interrupt signal, initiating immediate shutdown...")
        processor._immediate_shutdown_handler(signum, frame)

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
    return parser.parse_args()


def verify_dependencies() -> bool:
    logger = logging.getLogger('app')
    import shutil
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
                last_progress_time = current_time

        # Add progress callback to processor
        processor.set_progress_callback(log_progress)

        # Process videos without timeout parameter
        failed_files = processor.process_videos(
            input_dir,
            output_dir,
            target_size_mb=15.0
        )

        # Report results
        if failed_files:
            logger.warning(f"Failed to process {len(failed_files)} files:")
            for file in failed_files:
                logger.warning(f"  - {file}")
        else:
            logger.info("All videos processed successfully")

    except Exception as e:
        logger.error(f"Video processing error: {str(e)}", exc_info=True)
        sys.exit(1)


def main() -> None:
    # Parse arguments first
    args = parse_arguments()

    # Initialize directories before any logging or processing
    input_dir = args.input_dir or INPUT_DIR
    output_dir = args.output_dir or OUTPUT_DIR

    for directory in [input_dir, output_dir, LOG_DIR, TEMP_FILE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Setup logging system - get the app logger
    app_logger = setup_application_logging(
        debug_mode=args.debug, verbose_mode=args.verbose)

    # Initialize FFmpeg log file - clear it at the start of each run
    clear_ffmpeg_log()
    setup_ffmpeg_logging()

    app_logger.info("FFmpeg logs will be written to ffmpeg.log")

    try:
        if not verify_dependencies():
            sys.exit(1)

        # Initialize GPU acceleration once
        gpu_supported = False if args.no_gpu else setup_gpu_acceleration()
        app_logger.info(
            f"GPU acceleration {'enabled' if gpu_supported else 'disabled'}")

        # Get optimal GPU settings to share with processors
        gpu_settings = get_optimal_settings() if gpu_supported else {}
        if gpu_supported:
            app_logger.info(f"Using GPU settings: {gpu_settings}")

        # Initialize processor with GPU status from command line
        gif_processor = GIFProcessor(
            logger=app_logger, gpu_enabled=gpu_supported, gpu_settings=gpu_settings)

        # Setup signal handlers
        setup_signal_handlers(gif_processor)

        # Process videos with shared logging and GPU status
        app_logger.info("Processing videos...")

        # Pass GPU status to video processor
        process_videos(input_dir, output_dir, logger=app_logger,
                       gpu_enabled=gpu_supported, gpu_settings=gpu_settings)

        app_logger.info("Processing GIFs...")

        # SIMPLIFIED APPROACH: Process MP4 files one by one, directly
        # Get directories from imported constants
        input_dir = Path(INPUT_DIR)
        output_dir = Path(OUTPUT_DIR)
        temp_dir = Path(TEMP_FILE_DIR)

        # Clear the temp directory before starting
        app_logger.info("Clearing temporary directory...")
        try:
            for temp_file in temp_dir.glob('*'):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                except Exception as e:
                    app_logger.warning(
                        f"Failed to remove temp file {temp_file}: {e}")
            # Force memory cleanup
            gc.collect()
            time.sleep(1)
        except Exception as e:
            app_logger.error(f"Error clearing temp directory: {e}")

        # First find MP4 files in output directory to convert to GIFs
        app_logger.info("Finding files to process...")
        mp4_files = list(output_dir.glob('*.mp4'))
        gif_files = list(input_dir.glob('*.gif'))

        app_logger.info(
            f"Found {len(mp4_files)} MP4 files to convert and {len(gif_files)} GIFs to optimize")

        # Process MP4 files one by one
        if mp4_files:
            app_logger.info("Processing MP4 files one by one...")
            # Process just one file initially
            for i, mp4_file in enumerate(mp4_files, 1):
                try:
                    app_logger.info(
                        f"Processing file {i}/{len(mp4_files)}: {mp4_file.name}")

                    # Generate temporary and final paths
                    temp_gif = temp_dir / f"{mp4_file.stem}.gif"
                    final_gif = output_dir / f"{mp4_file.stem}.gif"

                    # Skip if final already exists and is good size
                    if final_gif.exists() and final_gif.stat().st_size > 0:
                        file_size_mb = final_gif.stat().st_size / (1024 * 1024)
                        if file_size_mb <= 15.0:  # Assuming 15MB is the target size
                            app_logger.info(
                                f"Skipping {mp4_file.name} - Already converted ({file_size_mb:.2f}MB)")
                            continue

                    # Step 1: Convert MP4 to temporary GIF
                    app_logger.info(f"Converting {mp4_file.name} to GIF...")

                    # Basic conversion with direct ffmpeg command
                    try:
                        import subprocess
                        # Get dimensions first
                        probe_cmd = [
                            'ffprobe', '-v', 'error',
                            '-select_streams', 'v:0',
                            '-show_entries', 'stream=width,height',
                            '-of', 'json', str(mp4_file)
                        ]
                        probe_result = subprocess.check_output(
                            probe_cmd, text=True)
                        import json
                        probe_data = json.loads(probe_result)

                        if 'streams' in probe_data and probe_data['streams']:
                            width = probe_data['streams'][0].get('width', 640)
                            height = probe_data['streams'][0].get(
                                'height', 360)

                            # Scale down if too large
                            scale_factor = 1.0
                            max_dimension = max(width, height)
                            if max_dimension > 1280:
                                scale_factor = 0.5
                            elif max_dimension > 720:
                                scale_factor = 0.75

                            new_width = int(width * scale_factor // 2 * 2)
                            new_height = int(height * scale_factor // 2 * 2)

                            # Create palette
                            palette_file = temp_dir / \
                                f"palette_{mp4_file.stem}.png"

                            palette_cmd = [
                                'ffmpeg', '-i', str(mp4_file),
                                '-vf', f'fps=15,scale={new_width}:{new_height}:flags=lanczos,palettegen=max_colors=128',
                                '-y', str(palette_file)
                            ]
                            subprocess.run(palette_cmd, check=True)

                            # Create GIF
                            gif_cmd = [
                                'ffmpeg', '-i', str(mp4_file),
                                '-i', str(palette_file),
                                '-lavfi', f'fps=15,scale={new_width}:{new_height}:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=3',
                                '-y', str(temp_gif)
                            ]
                            subprocess.run(gif_cmd, check=True)

                            # Optimize with gifsicle
                            if temp_gif.exists():
                                app_logger.info(
                                    f"Optimizing GIF: {temp_gif.name}")
                                opt_cmd = [
                                    'gifsicle', '--optimize=3',
                                    '--colors', '128',
                                    '--lossy=30',
                                    str(temp_gif),
                                    '-o', str(final_gif)
                                ]
                                subprocess.run(opt_cmd, check=True)

                                # Verify final file
                                if final_gif.exists():
                                    final_size_mb = final_gif.stat().st_size / (1024 * 1024)
                                    app_logger.success(
                                        f"Successfully created {final_gif.name} ({final_size_mb:.2f}MB)")
                                else:
                                    app_logger.error(
                                        f"Failed to create final GIF: {final_gif}")
                            else:
                                app_logger.error(
                                    f"Failed to create temporary GIF: {temp_gif}")
                        else:
                            app_logger.error(
                                f"Failed to get dimensions for {mp4_file}")
                    except Exception as e:
                        app_logger.error(
                            f"Error processing {mp4_file.name}: {e}")

                    # Clean up temp files
                    try:
                        if temp_gif.exists():
                            temp_gif.unlink(missing_ok=True)
                        if palette_file.exists():
                            palette_file.unlink(missing_ok=True)
                    except Exception as e:
                        app_logger.warning(
                            f"Failed to clean up temporary files: {e}")

                    # Force memory cleanup and pause between files
                    gc.collect()
                    time.sleep(1)

                except Exception as e:
                    app_logger.error(
                        f"Error processing {mp4_file.name}: {str(e)}")

        # Report results
        app_logger.success("GIF processing complete")

    except KeyboardInterrupt:
        app_logger.warning("\nProcess interrupted by user")
        gif_processor._immediate_shutdown_handler(signal.SIGINT, None)
    except Exception as e:
        app_logger.critical(f"Fatal error: {e}", exc_info=True)
        gif_processor._immediate_shutdown_handler(signal.SIGTERM, None)
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
