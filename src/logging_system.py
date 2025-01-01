# logging_system.py
# Handles logging setup and TeeLogger implementation.

import atexit
import logging
import subprocess
import sys
from pathlib import Path
from default_config import LOG_FILE, FFPMEG_LOG_FILE


class TeeLogger:
    """Custom logger that writes to both stdout and a file."""

    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log_file = open(filename, 'w', encoding='utf-8')
        atexit.register(self.close)

    def write(self, message):
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log_file.flush()

    def close(self):
        if not self.log_file.closed:
            self.log_file.close()

    def fileno(self):
        return self.terminal.fileno()


class FFmpegFilter(logging.Filter):
    """Filter to identify FFmpeg-related log messages."""

    def filter(self, record):
        return not record.getMessage().startswith('frame=')


def setup_logger():
    """Set up logging configuration and clear existing log files."""
    # Clear existing log files
    log_files = [Path(LOG_FILE), Path(FFPMEG_LOG_FILE).parent / 'ffmpeg.log']
    for log_file in log_files:
        try:
            if log_file.exists():
                log_file.unlink()  # Delete the file if it exists
        except PermissionError:
            logging.warning(f"Could not clear {log_file}: Permission denied")
        except Exception as e:
            logging.error(f"Error clearing {log_file}: {e}")

    # Create a file handler for the log file
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)

    # Create a console handler for stdout
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)

    # Add FFmpeg filter to console handler only
    console_handler.addFilter(FFmpegFilter())

    # Configure the root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logging.info("Logger initialized")


def log_exception(exc_type, exc_value, exc_traceback):
    """Log uncaught exceptions."""
    if issubclass(exc_type, KeyboardInterrupt):
        logging.warning("Script interrupted by user.")
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return
    logging.critical("Uncaught exception", exc_info=(
        exc_type, exc_value, exc_traceback))


def log_function_call(func):
    """Decorator to log function calls and their results."""
    def wrapper(*args, **kwargs):
        logging.info(f"Calling function: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logging.debug(f"Function {func.__name__} returned: {result}")
            return result
        except Exception as e:
            logging.error(f"Error in function {
                          func.__name__}: {e}", exc_info=True)
            raise
    return wrapper


def run_ffmpeg_command(command):
    """Run an FFmpeg command and redirect output to log file."""
    log_path = Path(FFPMEG_LOG_FILE).parent / 'ffmpeg.log'
    with open(log_path, 'a') as ffmpeg_log:
        try:
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=True
            )
            # Log stdout and stderr to the FFmpeg log file
            if process.stdout:
                ffmpeg_log.write(process.stdout)
            if process.stderr:
                ffmpeg_log.write(process.stderr)
            return True
        except subprocess.CalledProcessError as e:
            logging.error(f"FFmpeg command failed: {e}")
            if e.stdout:
                ffmpeg_log.write(e.stdout)
            if e.stderr:
                ffmpeg_log.write(e.stderr)
            return False


# Override the default exception hook
sys.excepthook = log_exception
