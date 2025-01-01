# main.py
# Handles the orchestration of the script and the main loop.

import logging
import os
import subprocess
import sys
from pathlib import Path

from default_config import (GIF_PASS_OVERS, INPUT_DIR, LOG_DIR, OUTPUT_DIR,
                            TEMP_FILE_DIR, GIF_COMPRESSION)
from gif_optimization import process_gifs
from gpu_acceleration import setup_gpu_acceleration
from logging_system import setup_logger, setup_error_termination
from video_optimization import process_videos
from gif_optimization import process_file
from temp_file_manager import TempFileManager  # Added import


def create_and_activate_venv(venv_dir, requirements_file):
    """Create a virtual environment and install the required packages."""
    try:
        if not venv_dir.exists():
            logging.info(
                "Virtual environment not found. Creating a new one...")
            # Create virtual environment
            subprocess.check_call(
                [sys.executable, "-m", "venv", str(venv_dir)])

        # Activate the virtual environment and install the required packages
        activate_script = venv_dir / "Scripts" / \
            "activate" if os.name == "nt" else venv_dir / "bin" / "activate"

        logging.info(
            "Activating virtual environment and installing requirements...")

        # Install packages
        subprocess.check_call([str(venv_dir / 'bin' / 'pip'), "install",
                              "-r", str(requirements_file)], stdout=subprocess.DEVNULL)

        logging.info("Virtual environment setup complete.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error setting up the virtual environment: {e}")
        sys.exit(1)


def check_requirements(requirements_file):
    """Check if all required packages in requirements.txt are installed."""
    try:
        with open(requirements_file, 'r') as f:
            required_packages = f.readlines()

        for package in required_packages:
            package = package.strip()
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'show', package],
                                      stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                logging.info(f"Package {package} is installed.")
            except subprocess.CalledProcessError:
                logging.error(
                    f"Package {package} is not installed. Please install the required packages.")
                sys.exit(1)

    except FileNotFoundError:
        logging.error(f"Requirements file {requirements_file} not found.")
        sys.exit(1)


def process_failed_gifs(failed_files, pass_over_index):
    logging.info(f"Starting additional pass for failed GIFs: Pass {
                 pass_over_index + 1}")
    pass_over = GIF_PASS_OVERS[pass_over_index]
    GIF_COMPRESSION.update(pass_over)

    new_failed_files = []
    for file_path in failed_files:
        try:
            source_file = Path(file_path)
            output_gif = OUTPUT_DIR / f"{source_file.stem}.gif"
            TempFileManager.register(output_gif)

            if source_file.suffix == '.gif':
                process_file(source_file, output_gif, is_video=False)
            elif source_file.suffix == '.mp4':
                process_file(source_file, output_gif, is_video=True)
            else:
                logging.warning(
                    f"File {file_path} not in expected formats, skipping.")

            if not output_gif.exists():
                logging.error(f"Failed to process file: {file_path}")
                new_failed_files.append(file_path)
            else:
                TempFileManager.unregister(output_gif)
        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            new_failed_files.append(file_path)

    GIF_COMPRESSION.update({
        'fps_range': (15, 10),
        'colors': 256,
        'lossy_value': 55,
        'min_size_mb': 10,
        'min_width': 120,
        'min_height': 120
    })

    return new_failed_files


def main():
    try:
        for directory in [INPUT_DIR, OUTPUT_DIR, LOG_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

        setup_logger()
        setup_error_termination()

        logging.info("Script starting...")
        gpu_supported = setup_gpu_acceleration()

        logging.info("Processing videos...")
        failed_videos = process_videos(gpu_supported)

        logging.info("Processing GIFs...")
        failed_gifs = process_gifs()

        all_failed_files = failed_videos + failed_gifs

        for i in range(len(GIF_PASS_OVERS)):
            if not all_failed_files:
                logging.info(f"All files processed successfully by pass {i}.")
                break
            all_failed_files = process_failed_gifs(all_failed_files, i)

        if all_failed_files:
            logging.warning(
                "Some files could not be processed after all passes:")
            for file in all_failed_files:
                logging.warning(file)

        logging.info("Script completed successfully.")
    finally:
        logging.info("Performing cleanup...")
        TempFileManager.cleanup()
        TempFileManager.cleanup_dir(TEMP_FILE_DIR)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Interrupted by user. Cleaning up...")
        TempFileManager.cleanup()
        TempFileManager.cleanup_dir(TEMP_FILE_DIR)
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        TempFileManager.cleanup()
        TempFileManager.cleanup_dir(TEMP_FILE_DIR)
        sys.exit(1)
