import argparse
import logging
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path
from typing import Optional, List

from default_config import (GIF_PASS_OVERS, INPUT_DIR, LOG_DIR, OUTPUT_DIR,
                            TEMP_FILE_DIR, GIF_COMPRESSION)
from gif_optimization import GIFProcessor, process_gifs
from gpu_acceleration import setup_gpu_acceleration
from logging_system import setup_logger
from temp_file_manager import TempFileManager
from video_optimization import process_videos


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
    required_commands = ['ffmpeg', 'ffprobe', 'gifsicle']
    missing_commands = []

    for cmd in required_commands:
        if not shutil.which(cmd):
            missing_commands.append(cmd)

    if missing_commands:
        logging.error(f"Missing required dependencies: {
                      ', '.join(missing_commands)}")
        return False
    return True


def process_failed_items(failed_files: List[Path], pass_over_index: int) -> List[Path]:
    logging.info(f"Starting optimization pass {pass_over_index + 1}")
    pass_over = GIF_PASS_OVERS[pass_over_index]
    GIF_COMPRESSION.update(pass_over)

    remaining_failed = []
    processor = GIFProcessor()

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

    # Reset compression settings
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
        failed_videos = process_videos(gpu_supported)

        logging.info("Processing GIFs...")
        failed_gifs = process_gifs()

        failed_files = failed_videos + failed_gifs

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
