#!/usr/bin/env python3
import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Import custom modules
from src.terminal_gui import TerminalGUI


def setup_logging(config):
    """
    Set up logging based on the configuration.
    """
    log_level = getattr(logging, config.get(
        'logging', {}).get('level', 'INFO'))
    log_file = config.get('logging', {}).get('file', './logs/processing.log')

    # Ensure log directory exists
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

    return logging.getLogger('main')


def ensure_directories(config):
    """
    Ensure all required directories exist.
    """
    dirs = config.get('directories', {})

    for dir_name, dir_path in dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Ensured directory exists: {dir_path}")


def scan_input_directory(input_dir):
    """
    Scan the input directory for video files and GIFs.
    """
    video_extensions = ['.mp4', '.avi', '.mkv',
                        '.mov', '.wmv', '.flv', '.webm']
    gif_extension = '.gif'

    video_files = []
    gif_files = []

    for file_path in Path(input_dir).glob('**/*'):
        if file_path.is_file():
            ext = file_path.suffix.lower()

            if ext in video_extensions:
                video_files.append(file_path)
            elif ext == gif_extension:
                gif_files.append(file_path)

    return video_files, gif_files


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(
        description='Video and GIF compression tool')
    parser.add_argument(
        '-c', '--config', help='Path to a specific config file')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Enable verbose output')

    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()

    # Initialize terminal GUI
    terminal_gui = TerminalGUI()

    # Get configuration
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            print(f"Error: Config file not found: {args.config}")
            return 1

        import yaml
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Use terminal GUI to select a config file
        config = terminal_gui.select_config_file()

    if not config:
        print("No configuration loaded. Exiting.")
        return 1

    # Set up logging
    logger = setup_logging(config)

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    logger.info("Starting video and GIF compression tool")

    # Ensure required directories exist
    try:
        ensure_directories(config)
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        terminal_gui.display_error(f"Failed to create directories: {e}")
        return 1

    # Scan input directory for video files and GIFs
    input_dir = config.get('directories', {}).get('input', './input')

    try:
        video_files, gif_files = scan_input_directory(input_dir)

        logger.info(
            f"Found {len(video_files)} video files and {len(gif_files)} GIF files")
        terminal_gui.display_info(
            f"Found {len(video_files)} video files and {len(gif_files)} GIF files")

        # Initialize progress display
        terminal_gui.initialize_progress_display(video_files, gif_files)

        # TODO: Implement video conversion and compression
        # This will be implemented in separate modules

        # TODO: Implement GIF optimization
        # This will be implemented in separate modules

        # Display summary
        terminal_gui.display_summary()

    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        terminal_gui.display_error(f"An error occurred: {e}")
        return 1

    logger.info("Processing completed successfully")
    return 0


if __name__ == "__main__":
    sys.exit(main())
