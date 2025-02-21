import argparse
import logging
import signal
import sys
from pathlib import Path
import time

# Use relative imports since we're in the src package
from .default_config import (GIF_PASS_OVERS, INPUT_DIR, LOG_DIR, OUTPUT_DIR,
                             TEMP_FILE_DIR, GIF_COMPRESSION, VIDEO_COMPRESSION)
# Use package-level import
from src.gif_operations import GIFProcessor, DynamicGIFOptimizer
from .gpu_acceleration import setup_gpu_acceleration
from .logging_system import setup_logger
from .temp_file_manager import TempFileManager
from .video_optimization import VideoProcessor
from .utils.error_handler import VideoProcessingError


def setup_signal_handlers(processor):
    """Setup signal handlers for graceful shutdown."""
    def signal_handler(signum, frame):
        print("\nReceived interrupt signal, initiating immediate shutdown...")
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
    return parser.parse_args()


def verify_dependencies() -> bool:
    import shutil
    required_commands = ['ffmpeg', 'ffprobe', 'gifsicle']
    missing_commands = [
        cmd for cmd in required_commands if not shutil.which(cmd)]

    if missing_commands:
        logging.error(f"Missing required dependencies: {
                      ', '.join(missing_commands)}")
        return False
    return True


def process_failed_items(failed_files: list[Path], pass_over_index: int) -> list[Path]:
    logging.info(f"Starting optimization pass {pass_over_index + 1}")

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
            logging.error(f"Failed to process {file_path}: {e}")
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


def setup_logging():
    """Remove standalone video_processing.log setup."""
    # Remove this entire function as we're using the logging_system module
    pass


def process_videos(input_dir: Path, output_dir: Path) -> None:
    """Process videos with enhanced error handling and progress tracking."""
    processor = VideoProcessor(use_gpu=True)

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
                logging.info(f"Progress: {progress:.1f}% ({current}/{total})")
                logging.info(f"Currently processing: {file_name}")
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
            logging.warning(f"Failed to process {len(failed_files)} files:")
            for file in failed_files:
                logging.warning(f"  - {file}")
        else:
            logging.info("All videos processed successfully")

    except Exception as e:
        logging.error(f"Video processing error: {str(e)}")
        sys.exit(1)


def main() -> None:
    args = parse_arguments()
    logger = setup_logger(debug_mode=args.debug, verbose_mode=args.verbose)

    # Initialize processor first
    gif_processor = GIFProcessor(verbose=args.verbose)

    # Setup signal handlers
    setup_signal_handlers(gif_processor)

    # Initialize directories
    input_dir = args.input_dir or INPUT_DIR
    output_dir = args.output_dir or OUTPUT_DIR

    for directory in [input_dir, output_dir, LOG_DIR, TEMP_FILE_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    try:
        if not verify_dependencies():
            sys.exit(1)

        gpu_supported = False if args.no_gpu else setup_gpu_acceleration()

        # Process videos with shared logging
        logger.info("Processing videos...")
        process_videos(input_dir, output_dir)

        logger.info("Processing GIFs...")
        # Pass verbose flag to GIF processor
        failed_gifs = gif_processor.process_all()

        if failed_gifs:
            logger.warning(f"Failed to process {len(failed_gifs)} gifs:")
            for gif in failed_gifs:
                logger.warning(f"  - {gif}")
        else:
            logger.success("All gifs processed successfully")

        failed_files = failed_gifs

        # Try multiple passes with different compression settings
        for i, _ in enumerate(GIF_PASS_OVERS):
            if not failed_files:
                logger.success("All files processed successfully")
                break
            failed_files = process_failed_items(failed_files, i)

        if failed_files:
            logger.warning("Failed to process the following files:")
            for file in failed_files:
                logger.warning(f"  - {file}")

    except KeyboardInterrupt:
        logger.warning("\nProcess interrupted by user")
        gif_processor._immediate_shutdown_handler(signal.SIGINT, None)
    except Exception as e:
        logger.critical(f"Fatal error: {e}", exc_info=True)
        gif_processor._immediate_shutdown_handler(signal.SIGTERM, None)
    finally:
        sys.exit(0)


if __name__ == "__main__":
    main()
