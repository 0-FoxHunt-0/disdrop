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


def setup_signal_handlers(processor=None):
    """
    Setup signal handlers for graceful shutdown.

    Args:
        processor: The processor instance that has an _immediate_shutdown_handler method
    """
    logger = logging.getLogger('app')

    def signal_handler(signum, frame):
        signal_name = "SIGINT" if signum == signal.SIGINT else "SIGTERM" if signum == signal.SIGTERM else f"Signal {signum}"
        logger.warning(
            f"\nReceived {signal_name}, initiating immediate shutdown...")

        if processor is not None and hasattr(processor, '_immediate_shutdown_handler'):
            try:
                processor._immediate_shutdown_handler(signum, frame)
            except Exception as e:
                logger.error(f"Error during shutdown handler: {e}")
                traceback.print_exc()
        else:
            logger.warning(
                "No processor with _immediate_shutdown_handler method available")

        # Always exit after handling the signal
        logger.info("Exiting application...")
        sys.exit(1)

    # Register for both SIGINT (Ctrl+C) and SIGTERM
    try:
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        logger.debug("Signal handlers registered successfully")
    except Exception as e:
        logger.error(f"Failed to register signal handlers: {e}")


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
    parser.add_argument('--max-size', type=float, default=15.0,
                        help='Maximum file size in MB')
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


def process_failed_items(failed_files: list[Path], pass_over_index: int, max_size_mb: float) -> list[Path]:
    logger = logging.getLogger('app')
    logger.info(f"Starting optimization pass {pass_over_index + 1}")

    # Update compression settings for this pass
    pass_over = GIF_PASS_OVERS[pass_over_index]
    GIF_COMPRESSION.update(pass_over)

    # Create processor with input/output directories and max size
    processor = GIFProcessor(
        input_dir=INPUT_DIR,
        output_dir=OUTPUT_DIR,
        max_size_mb=max_size_mb,
        logger=logger
    )

    remaining_failed = []

    for file_path in failed_files:
        try:
            source_file = Path(file_path)
            output_path = OUTPUT_DIR / f"{source_file.stem}.gif"
            is_video = source_file.suffix.lower() in ['.mp4', '.mkv', '.avi']

            # Process the file using the new processor
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
            target_size_mb=10.0
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


def process_gifs(input_dir: Path, output_dir: Path, temp_dir: Path, max_size_mb: float, logger=None, gpu_enabled=False, gpu_settings=None) -> None:
    """Process GIFs using the new GIFProcessor."""
    app_logger = logging.getLogger('app')
    logger = logger or app_logger

    try:
        # Create GIF processor
        gif_processor = GIFProcessor(
            input_dir=input_dir,
            output_dir=output_dir,
            max_size_mb=max_size_mb,
            logger=logger,
            gpu_enabled=gpu_enabled,
            gpu_settings=gpu_settings
        )

        # Clear the temp directory before starting
        logger.info("Clearing temporary directory...")
        try:
            for temp_file in temp_dir.glob('*'):
                try:
                    if temp_file.is_file():
                        temp_file.unlink()
                except Exception as e:
                    logger.warning(
                        f"Failed to remove temp file {temp_file}: {e}")
            # Force memory cleanup
            gc.collect()
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error clearing temp directory: {e}")

        # Find files to process
        logger.info("Finding files to process...")

        # Process all files using the GIFProcessor
        failed_items = gif_processor.process_all()

        # Report results
        if failed_items:
            logger.warning(f"Failed to process {len(failed_items)} files:")
            for item in failed_items:
                logger.warning(f"  - {item}")

            # Try to process failed items with more aggressive settings
            if len(failed_items) > 0:
                logger.info(
                    "Attempting to process failed items with more aggressive settings...")
                for pass_index in range(len(GIF_PASS_OVERS)):
                    if not failed_items:
                        break
                    failed_items = process_failed_items(
                        failed_items, pass_index, max_size_mb)
                    if not failed_items:
                        logger.success(
                            "All previously failed items processed successfully!")
                        break

                if failed_items:
                    logger.warning(
                        f"Still failed to process {len(failed_items)} files after all attempts")
        else:
            logger.success("All GIFs processed successfully")

    except Exception as e:
        logger.error(f"GIF processing error: {str(e)}", exc_info=True)
        return


def main() -> None:
    # Parse arguments first
    args = parse_arguments()

    # Initialize directories before any logging or processing
    input_dir = args.input_dir or INPUT_DIR
    output_dir = args.output_dir or OUTPUT_DIR
    max_size_mb = args.max_size

    for directory in [input_dir, output_dir, LOG_DIR, TEMP_FILE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Setup logging system - get the app logger
    app_logger = setup_application_logging(
        debug_mode=args.debug, verbose_mode=args.verbose)

    # Initialize FFmpeg log file - clear it at the start of each run
    clear_ffmpeg_log()
    setup_ffmpeg_logging()

    app_logger.info("FFmpeg logs will be written to ffmpeg.log")

    # Define gif_processor at module level so it's accessible in exception handlers
    gif_processor = None

    try:
        if not verify_dependencies():
            sys.exit(1)

        # Initialize GPU acceleration once
        gpu_supported = False if args.no_gpu else setup_gpu_acceleration()

        # Get optimal GPU settings to share with processors
        gpu_settings = get_optimal_settings() if gpu_supported else {}
        if gpu_supported:
            app_logger.info(f"Using GPU settings: {gpu_settings}")
        else:
            app_logger.info("GPU acceleration disabled")

        # Initialize processor with GPU status from command line
        gif_processor = GIFProcessor(
            input_dir=input_dir,
            output_dir=output_dir,
            max_size_mb=max_size_mb,
            logger=app_logger,
            gpu_enabled=gpu_supported,
            gpu_settings=gpu_settings
        )

        # Setup signal handlers with the processor instance
        setup_signal_handlers(gif_processor)

        # Process videos with shared logging and GPU status
        app_logger.info("Processing videos...")
        process_videos(input_dir, output_dir, logger=app_logger,
                       gpu_enabled=gpu_supported, gpu_settings=gpu_settings)

        # Process GIFs
        app_logger.info("Processing GIFs...")
        process_gifs(input_dir, output_dir, Path(TEMP_FILE_DIR), max_size_mb,
                     logger=app_logger, gpu_enabled=gpu_supported,
                     gpu_settings=gpu_settings)

        # Report success
        app_logger.success("Processing complete")

    except KeyboardInterrupt:
        app_logger.warning("\nProcess interrupted by user")
        if gif_processor is not None and hasattr(gif_processor, '_immediate_shutdown_handler'):
            try:
                gif_processor._immediate_shutdown_handler(signal.SIGINT, None)
            except Exception as e:
                app_logger.error(f"Error during shutdown: {e}")
                traceback.print_exc()
    except Exception as e:
        app_logger.critical(f"Fatal error: {e}", exc_info=True)
        if gif_processor is not None and hasattr(gif_processor, '_immediate_shutdown_handler'):
            try:
                gif_processor._immediate_shutdown_handler(signal.SIGTERM, None)
            except Exception as shutdown_error:
                app_logger.error(f"Error during shutdown: {shutdown_error}")
                traceback.print_exc()
    finally:
        app_logger.info("Exiting application")
        sys.exit(0)


if __name__ == "__main__":
    main()
