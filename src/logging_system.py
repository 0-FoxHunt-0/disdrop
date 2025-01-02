# logging_system.py
# Handles logging setup and TeeLogger implementation.

import atexit
import logging
import subprocess
import sys
from functools import partial
from pathlib import Path
from datetime import datetime


from default_config import FFPMEG_LOG_FILE, LOG_FILE, LOG_DIR


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
    """Set up logging configuration and create log files."""
    # Create log directory if it doesn't exist
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    # Clear existing log files
    log_files = [Path(LOG_FILE), Path(FFPMEG_LOG_FILE)]
    for log_file in log_files:
        try:
            # Ensure parent directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)
            if log_file.exists():
                log_file.unlink()  # Delete the file if it exists
            # Create empty log file with proper permissions
            log_file.touch(mode=0o666)  # Read/write for all users
            logging.info(f"Created log file: {log_file}")
        except PermissionError:
            logging.warning(f"Could not clear {log_file}: Permission denied")
        except Exception as e:
            logging.error(f"Error handling {log_file}: {e}")

    # Create a file handler specifically for FFmpeg logs
    ffmpeg_handler = logging.FileHandler(FFPMEG_LOG_FILE)
    ffmpeg_handler.setLevel(logging.DEBUG)
    ffmpeg_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    ffmpeg_handler.setFormatter(ffmpeg_formatter)

    # Configure the FFmpeg logger
    ffmpeg_logger = logging.getLogger('ffmpeg')
    ffmpeg_logger.setLevel(logging.DEBUG)
    ffmpeg_logger.addHandler(ffmpeg_handler)

    # Create a file handler for the main log file
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


def error_handler(logger, *args, **kwargs):
    """Custom error handler to terminate the program on logging.error."""
    message = args[0]
    logger._log(logging.ERROR, message, args, **kwargs)
    logging.warning("A critical error occurred. Terminating execution.")
    # Clean up resources or perform any necessary shutdown tasks here
    # For example:
    for file in Path(TEMP_FILE_DIR).glob('*'):
        try:
            file.unlink()
        except Exception as e:
            logging.warning(f"Failed to remove temporary file {file}: {e}")
    sys.exit(1)  # Exit with error status


def setup_error_termination():
    """Set up error handler to terminate on logging.error."""
    logger = logging.getLogger()
    error_handler_func = partial(error_handler, logger)
    logger.error = error_handler_func


def log_function_call(func):
    """Decorator to log function calls and their results."""
    def wrapper(*args, **kwargs):
        logging.debug(f"Calling function: {func.__name__}")
        try:
            result = func(*args, **kwargs)
            # logging.debug(f"Function {func.__name__} returned: {result}")
            return result
        except Exception as e:
            logging.error(f"Error in function {
                          func.__name__}: {e}", exc_info=True)
            raise
    return wrapper


def run_ffmpeg_command(command):
    """Run an FFmpeg command and redirect FFmpeg's output to log file."""
    ffmpeg_logger = logging.getLogger('ffmpeg')

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        # Process both stdout and stderr streams simultaneously
        while True:
            # Read from both streams
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()

            # Break if both streams are empty and process has finished
            if not stdout_line and not stderr_line and process.poll() is not None:
                break

            # Log FFmpeg's direct output
            if stdout_line:
                # Only log actual FFmpeg output, not empty lines
                line = stdout_line.strip()
                if line:
                    ffmpeg_logger.debug(line)

            if stderr_line:
                # FFmpeg outputs most of its progress information to stderr
                line = stderr_line.strip()
                if line:
                    ffmpeg_logger.debug(line)

        # Get return code
        return_code = process.poll()

        # Only return success/failure
        return return_code == 0

    except Exception as e:
        ffmpeg_logger.error(f"FFmpeg process failed: {str(e)}")
        return False


# Override the default exception hook
sys.excepthook = log_exception
