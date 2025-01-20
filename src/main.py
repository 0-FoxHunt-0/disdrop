import argparse
import logging
import signal
import sys
from pathlib import Path

from default_config import (GIF_PASS_OVERS, INPUT_DIR, LOG_DIR, OUTPUT_DIR,
                            TEMP_FILE_DIR, GIF_COMPRESSION)
from gif_optimization import GIFProcessor
from gpu_acceleration import setup_gpu_acceleration
from logging_system import setup_logger
from temp_file_manager import TempFileManager
from video_optimization import BatchVideoProcessor


def signal_handler(signum, frame):
    logging.warning("\nGracefully shutting down...")
    TempFileManager.cleanup()
    TempFileManager.cleanup_dir(TEMP_FILE_DIR)
    logging.info("Cleanup complete")
    sys.exit(0)


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
    remaining_failed = []

    for file_path in failed_files:
        try:
            source_file = Path(file_path)
            output_path = OUTPUT_DIR / f"{source_file.stem}.gif"
            is_video = source_file.suffix.lower() in ['.mp4', '.mkv', '.avi']

            processor.process_file(source_file, output_path, is_video)

            if not output_path.exists():
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


def main() -> None:
    try:
        args = parse_arguments()

        if args.input_dir:
            global INPUT_DIR
            INPUT_DIR = args.input_dir
        if args.output_dir:
            global OUTPUT_DIR
            OUTPUT_DIR = args.output_dir

        for directory in [INPUT_DIR, OUTPUT_DIR, LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

        setup_logger(args.debug)

        if not verify_dependencies():
            sys.exit(1)

        gpu_supported = False if args.no_gpu else setup_gpu_acceleration()

        logging.info("Processing videos...")
        video_processor = BatchVideoProcessor(use_gpu=gpu_supported)
        failed_videos = video_processor.process_all_videos()

        logging.info("Processing GIFs...")
        gif_processor = GIFProcessor()
        failed_gifs = gif_processor.process_all()

        failed_files = failed_videos + failed_gifs

        # Try multiple passes with different compression settings
        for i, _ in enumerate(GIF_PASS_OVERS):
            if not failed_files:
                logging.info("All files processed successfully")
                break
            failed_files = process_failed_items(failed_files, i)

        if failed_files:
            logging.warning("Failed to process the following files:")
            for file in failed_files:
                logging.warning(f"  - {file}")

    except KeyboardInterrupt:
        logging.warning("Process interrupted by user")
    except Exception as e:
        logging.critical(f"Fatal error: {e}", exc_info=True)
    finally:
        TempFileManager.cleanup()
        TempFileManager.cleanup_dir(TEMP_FILE_DIR)
        logging.info("Cleanup complete")


if __name__ == "__main__":
    main()
