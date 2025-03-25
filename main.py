#!/usr/bin/env python3
import os
import sys
import logging
import argparse
from pathlib import Path
from datetime import datetime

# Import custom modules
from src.terminal_gui import TerminalGUI
from src.gpu_detector import GPUDetector, AccelerationType
from src.processors.video_processor import VideoProcessor
from src.processors.gif_processor import GIFProcessor
from src.resource_manager import ResourceManager
from src.logging_system import LoggingSystem


def ensure_directories(config):
    """
    Ensure all required directories exist.
    """
    dirs = config.get('directories', {})

    for dir_name, dir_path in dirs.items():
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Ensured directory exists: {dir_path}")


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
    parser.add_argument('--gpu-info', action='store_true',
                        help='Display GPU acceleration information and exit')
    parser.add_argument('--videos-only', action='store_true',
                        help='Process only video files')
    parser.add_argument('--gifs-only', action='store_true',
                        help='Process only GIF files')
    parser.add_argument('--no-gui', action='store_true',
                        help='Run with default config without showing GUI')

    return parser.parse_args()


def display_gpu_info():
    """
    Display information about GPU acceleration capabilities.
    """
    detector = GPUDetector()
    gpu_types, accel_types = detector.detect()
    preferred = detector.get_preferred_acceleration()

    print("\n=== GPU Acceleration Information ===")
    print(f"Detected GPU types: {[gpu.name for gpu in gpu_types]}")
    print(
        f"Available acceleration frameworks: {[accel.name for accel in accel_types]}")
    print(f"Preferred acceleration: {preferred.name}")

    # Get detailed info
    device_info = detector.get_device_info()
    if device_info["detailed_info"]:
        print("\nDetailed GPU information:")
        for gpu, info in device_info["detailed_info"].items():
            print(f"  {gpu.upper()}:")
            for line in info.strip().split('\n'):
                print(f"    {line}")

    print("\nThis information will be used to optimize video and GIF processing.")
    return 0


def main():
    # Parse command line arguments
    args = parse_args()

    # If GPU info display is requested, show it and exit
    if args.gpu_info:
        return display_gpu_info()

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
    elif args.no_gui:
        # Use default config without showing GUI
        default_config_path = Path("config/default.yaml")
        if not default_config_path.exists():
            print("Error: Default config file not found. Please create config/default.yaml or specify a config file.")
            return 1

        import yaml
        with open(default_config_path, 'r') as f:
            config = yaml.safe_load(f)
    else:
        # Use terminal GUI to select a config file
        config = terminal_gui.select_config_file()

    if not config:
        print("No configuration loaded. Exiting.")
        return 1

    # Ensure logging config section exists
    if 'logging' not in config:
        config['logging'] = {}

    # Setup logging with verbose mode if requested
    if args.verbose:
        config['logging']['level'] = 'DEBUG'
        config['logging']['console_level'] = 'DEBUG'

    # Initialize the logging system
    logging_system = LoggingSystem(config)
    logger = logging_system.get_logger('main')

    # Log application startup
    logger.info("Starting video and GIF compression tool")
    logging_system.start_new_log_section("Initialization")

    # Log system information
    logging_system.log_system_info()

    # Initialize resource manager for handling interruption signals and cleanup
    resource_manager = ResourceManager(config)

    # Ensure required directories exist
    try:
        ensure_directories(config)
    except Exception as e:
        logger.error(f"Failed to create directories: {e}")
        terminal_gui.display_error(f"Failed to create directories: {e}")
        return 1

    # Register temporary directories with resource manager
    temp_dir = Path(config.get('directories', {}).get('temp', './temp'))
    resource_manager.register_temp_dir(temp_dir)

    # Process video files if not limited to GIFs only
    if not args.gifs_only:
        try:
            # Start a new log section for video processing
            logging_system.start_new_log_section("Video Processing")

            # Initialize video processor
            video_processor = VideoProcessor(config)

            # Register shutdown flag check with video processor
            video_processor.shutdown_requested = False

            def check_shutdown():
                video_processor.shutdown_requested = resource_manager.get_shutdown_flag()
            resource_manager.register_cleanup_hook(check_shutdown)

            # Display acceleration info
            terminal_gui.display_info(
                f"Using {video_processor.preferred_acceleration.name} acceleration for video processing")

            # Find video files
            mp4_files, other_video_files = video_processor.find_video_files()
            total_videos = len(mp4_files) + len(other_video_files)

            if total_videos > 0:
                logger.info(f"Found {total_videos} video files to process")
                terminal_gui.display_info(
                    f"Found {total_videos} video files to process")

                # Initialize progress display
                terminal_gui.initialize_progress_display(
                    video_files=mp4_files + other_video_files)

                # Process videos with enhanced workflow
                logger.info("Starting video processing workflow")
                results = video_processor.process_videos()

                # Update progress display
                for input_file, output_file in results.items():
                    if output_file:
                        terminal_gui.update_progress(
                            'videos', message=f"Processed: {input_file.name}")
                        logger.success(
                            f"Successfully processed: {input_file.name}")
                    else:
                        terminal_gui.update_progress(
                            'videos', message=f"Failed: {input_file.name}")
                        logger.error(f"Failed to process: {input_file.name}")

                logger.success("Video processing completed")
            else:
                logger.info("No video files found to process")
                terminal_gui.display_info("No video files found to process")

        except Exception as e:
            logger.error(
                f"An error occurred during video processing: {e}", exc_info=True)
            terminal_gui.display_error(
                f"An error occurred during video processing: {e}")
            return 1

    # Process GIF files if not limited to videos only
    if not args.videos_only:
        try:
            # Start a new log section for GIF processing
            logging_system.start_new_log_section("GIF Processing")

            # Initialize GIF processor
            gif_processor = GIFProcessor(config)

            # Register shutdown flag check with GIF processor
            gif_processor.shutdown_requested = False

            def check_gif_shutdown():
                gif_processor.shutdown_requested = resource_manager.get_shutdown_flag()
            resource_manager.register_cleanup_hook(check_gif_shutdown)

            # Scan for GIF files
            gif_files = gif_processor.find_gif_files()

            if gif_files:
                logger.info(f"Found {len(gif_files)} GIF files to process")
                terminal_gui.display_info(
                    f"Found {len(gif_files)} GIF files to process")

                # Initialize progress display if needed
                if args.gifs_only or not terminal_gui.progress_bars:
                    terminal_gui.initialize_progress_display(
                        gif_files=gif_files)

                # Process GIF files
                logger.info("Starting GIF processing")
                results = gif_processor.process_gifs()

                # Update progress display
                for input_file, output_file in results.items():
                    if output_file:
                        terminal_gui.update_progress(
                            'gifs', message=f"Processed: {input_file.name}")
                        logger.success(
                            f"Successfully processed: {input_file.name}")
                    else:
                        terminal_gui.update_progress(
                            'gifs', message=f"Failed: {input_file.name}")
                        logger.error(f"Failed to process: {input_file.name}")

                logger.success("GIF processing completed")
            else:
                logger.info("No GIF files found to process")
                terminal_gui.display_info("No GIF files found to process")

        except Exception as e:
            logger.error(
                f"An error occurred during GIF processing: {e}", exc_info=True)
            terminal_gui.display_error(
                f"An error occurred during GIF processing: {e}")
            if not args.videos_only:  # Only return error if we were exclusively processing GIFs
                return 1

    # Log completion
    logging_system.start_new_log_section("Summary")
    logger.success("Processing completed successfully")

    # Display summary
    terminal_gui.display_summary()
    return 0


if __name__ == "__main__":
    sys.exit(main())
