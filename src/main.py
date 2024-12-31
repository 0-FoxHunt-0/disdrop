# main.py
# Handles the orchestration of the script and the main loop.

import logging
import os
import subprocess
import sys
from pathlib import Path

from default_config import INPUT_DIR, LOG_DIR, LOG_FILE, OUTPUT_DIR, TEMP_FILE_DIR
from gif_optimization import process_gifs
from gpu_acceleration import setup_gpu_acceleration
from logging_system import setup_logger
from video_optimization import process_videos


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


def main():
    # Ensure directories exist
    for directory in [INPUT_DIR, OUTPUT_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)

    # Set up logging
    setup_logger(LOG_FILE)

    logging.info("Starting the script")

    # Check for GPU setup
    gpu_supported = setup_gpu_acceleration()

    # First process videos
    logging.info("Starting video processing...")
    failed_videos = process_videos(gpu_supported)

    # Then process GIFs (including those created from videos)
    logging.info("Starting GIF processing...")
    failed_gifs = process_gifs()

    # Combine all failed files
    all_failed_files = failed_videos + failed_gifs

    if all_failed_files:
        logging.error("Failed files:")
        for file in set(all_failed_files):  # Using set to remove duplicates
            logging.error(file)
    else:
        logging.info("All files processed successfully")

    logging.info("Script finished successfully")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logging.warning("Script interrupted by user.")
        for file in TEMP_FILE_DIR.glob('*'):
            file.unlink()
        sys.exit(0)
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        sys.exit(1)
