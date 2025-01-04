# main.py

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional, Union

from default_config import (GIF_COMPRESSION, GIF_PASS_OVERS, INPUT_DIR,
                            LOG_DIR, OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS,
                            TEMP_FILE_DIR, VIDEO_COMPRESSION)
from gif_optimization import GIFProcessor, process_gifs
from gpu_acceleration import setup_gpu_acceleration
from logging_system import setup_logging, logging
from temp_file_manager import TempFileManager
from utils import get_video_dimensions
from video_optimization import VideoProcessor, process_videos


def signal_handler(signum, frame):
    logging.warning("\nGracefully shutting down...")
    TempFileManager.cleanup()
    TempFileManager.cleanup_dir(TEMP_FILE_DIR)
    logging.info("Cleanup complete")
    sys.exit(0)


signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# TODO: Add a flag to process only videos and not output gifs


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
    parser.add_argument('--videos-only', action='store_true',
                        help='Process only videos, skip GIF generation')
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


def validate_config():
    """Validate configuration settings."""
    errors = []

    if not isinstance(VIDEO_COMPRESSION['scale_factor'], (int, float)) or \
       not 0 < VIDEO_COMPRESSION['scale_factor'] <= 1:
        errors.append(
            "VIDEO_COMPRESSION['scale_factor'] must be between 0 and 1")

    if not isinstance(VIDEO_COMPRESSION['crf'], int) or \
       not 0 <= VIDEO_COMPRESSION['crf'] <= 51:
        errors.append("VIDEO_COMPRESSION['crf'] must be between 0 and 51")

    if not isinstance(GIF_COMPRESSION['colors'], int) or \
       not 2 <= GIF_COMPRESSION['colors'] <= 256:
        errors.append("GIF_COMPRESSION['colors'] must be between 2 and 256")

    if errors:
        raise ValueError(
            "Configuration validation failed:\n" + "\n".join(errors))


def safe_process_file(processor: Union[VideoProcessor, GIFProcessor],
                      file_path: Path,
                      output_path: Path,
                      max_retries: int = 3) -> bool:
    """Process a file with error recovery."""
    for attempt in range(max_retries):
        try:
            return processor.process_file(file_path, output_path,
                                          is_video=file_path.suffix.lower() in SUPPORTED_VIDEO_FORMATS)
        except Exception as e:
            if attempt < max_retries - 1:
                logging.warning(f"Attempt {attempt + 1} failed, retrying: {e}")
                time.sleep(1)  # Add delay between retries
            else:
                logging.error(f"All attempts failed for {file_path}: {e}")
                return False


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


def verify_directories():
    """Verify all required directories exist and are accessible."""
    required_dirs = {
        'input': INPUT_DIR,
        'output': OUTPUT_DIR,
        'temp': TEMP_FILE_DIR,
        'logs': LOG_DIR
    }

    for name, path in required_dirs.items():
        try:
            path.mkdir(parents=True, exist_ok=True)
            # Test write permissions
            test_file = path / '.write_test'
            test_file.touch()
            test_file.unlink()
        except Exception as e:
            logging.error(f"Directory {name} ({path}) error: {e}")
            return False
    return True


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

        setup_logging(args.debug)
        validate_config()  # Add config validation

        if not verify_dependencies():
            sys.exit(1)

        if not verify_directories():
            logging.error("Directory verification failed")
            sys.exit(1)

        gpu_supported = False if args.no_gpu else setup_gpu_acceleration()

        failed_videos = []
        failed_gifs = []

        # Update video processing
        video_processor = VideoProcessor(gpu_supported)
        for video in Path(INPUT_DIR).glob('*.mp4'):
            output_path = OUTPUT_DIR / video.name
            if not output_path.exists():
                if not safe_process_file(video_processor, video, output_path):
                    failed_videos.append(video)

        # Update GIF processing
        gif_processor = GIFProcessor()
        for gif in Path(INPUT_DIR).glob('*.gif'):
            output_path = OUTPUT_DIR / gif.name
            if not output_path.exists():
                if not safe_process_file(gif_processor, gif, output_path):
                    failed_gifs.append(gif)

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


if __name__ == "__main__":
    main()
