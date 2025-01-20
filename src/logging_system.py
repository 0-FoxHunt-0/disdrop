import atexit
import functools
import logging
import os
import subprocess
import sys
from datetime import datetime
from functools import partial, wraps
from pathlib import Path
from typing import Optional, TextIO, Union

from default_config import FFPMEG_LOG_FILE, LOG_DIR, LOG_FILE, TEMP_FILE_DIR

# Add custom log levels
SUCCESS_LEVEL = 25  # Between INFO and WARNING
SKIP_LEVEL = 15    # Between DEBUG and INFO

# Register new log levels
logging.addLevelName(SUCCESS_LEVEL, 'SUCCESS')
logging.addLevelName(SKIP_LEVEL, 'SKIP')

# Add methods to logger


def success(self, message, *args, **kwargs):
    self.log(SUCCESS_LEVEL, message, *args, **kwargs)


def skip(self, message, *args, **kwargs):
    self.log(SKIP_LEVEL, message, *args, **kwargs)


logging.Logger.success = success
logging.Logger.skip = skip


def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        func_name = func.__name__
        logging.debug(f"Entering {func_name}")
        try:
            result = func(*args, **kwargs)
            logging.debug(f"Exiting {func_name}")
            return result
        except Exception as e:
            logging.error(f"Error in {func_name}: {str(e)}")
            raise
    return wrapper


class ColorFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[0;36m',    # Cyan
        'INFO': '\033[0;37m',     # White
        'WARNING': '\033[0;33m',  # Yellow
        'ERROR': '\033[0;31m',    # Red
        'CRITICAL': '\033[0;35m',  # Purple
        'SUCCESS': '\033[0;32m',  # Green
        'SKIP': '\033[0;34m'      # Blue
    }
    RESET = '\033[0m'

    def __init__(self, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.default_color = self.RESET

    def format(self, record):
        # Save original message
        original_msg = record.msg

        # Apply color
        color = self.COLORS.get(record.levelname, self.default_color)
        record.msg = f"{color}{record.msg}{self.RESET}"

        # Format the message
        formatted_msg = super().format(record)

        # Restore original message
        record.msg = original_msg

        return formatted_msg


class RotatingFileHandler(logging.FileHandler):
    def __init__(self, filename: Union[str, Path], max_bytes: int = 10485760, backup_count: int = 5):
        super().__init__(filename)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self._should_rotate()

    def _should_rotate(self) -> None:
        if not Path(self.baseFilename).exists():
            return

        if Path(self.baseFilename).stat().st_size > self.max_bytes:
            # Remove old backups if exceeding backup_count
            backup_files = sorted(
                Path(self.baseFilename).parent.glob(f"{Path(self.baseFilename).name}.*"))
            while len(backup_files) >= self.backup_count:
                backup_files[0].unlink()
                backup_files = backup_files[1:]

            # Create new backup
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup = Path(f"{self.baseFilename}.{timestamp}")
            Path(self.baseFilename).rename(backup)

    def emit(self, record):
        self._should_rotate()
        super().emit(record)


class TeeLogger:
    def __init__(self, filename: Union[str, Path], encoding: str = 'utf-8', buffer_size: int = 1):
        self.terminal: TextIO = sys.stdout
        try:
            self.log_file: TextIO = open(
                filename, 'w', encoding=encoding, buffering=buffer_size)
            self.encoding = encoding
            atexit.register(self.close)
        except Exception as e:
            logger.error(f"Failed to open log file: {e}")
            self.lo

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, message: str) -> None:
        try:
            self.terminal.write(message)
            self.log_file.write(message)
            self.flush()
        except (IOError, ValueError) as e:
            sys.stderr.write(f"TeeLogger write error: {str(e)}\n")

    def flush(self) -> None:
        try:
            self.terminal.flush()
            self.log_file.flush()
        except (IOError, ValueError) as e:
            sys.stderr.write(f"TeeLogger flush error: {str(e)}\n")

    def close(self) -> None:
        if hasattr(self, 'log_file') and not self.log_file.closed:
            try:
                self.log_file.close()
            except Exception as e:
                sys.stderr.write(f"TeeLogger close error: {str(e)}\n")

    def fileno(self) -> int:
        return self.terminal.fileno()


def setup_logger(debug_mode: bool = False, log_rotation_size: int = 10485760,
                 backup_count: int = 5) -> None:
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter('%(levelname)s: %(message)s'))
    console_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Main log file handler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        max_bytes=log_rotation_size,
        backup_count=backup_count
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'))
    file_handler.setLevel(logging.DEBUG)

    # FFmpeg log file handler
    ffmpeg_handler = RotatingFileHandler(
        FFPMEG_LOG_FILE,
        max_bytes=log_rotation_size,
        backup_count=backup_count
    )
    ffmpeg_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'))

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    ffmpeg_logger = logging.getLogger('ffmpeg')
    ffmpeg_logger.handlers.clear()
    ffmpeg_logger.addHandler(ffmpeg_handler)

    sys.excepthook = log_exception
    logging.success("Logging system initialized successfully")


def log_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.critical("Uncaught exception",
                     exc_info=(exc_type, exc_value, exc_traceback))


def run_ffmpeg_command(command: list, timeout: Optional[int] = 300) -> bool:
    ffmpeg_logger = logging.getLogger('ffmpeg')
    current_dir = os.getcwd()

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        with open(FFPMEG_LOG_FILE, 'a', encoding='utf-8') as log_file:
            while True:
                try:
                    output = process.stdout.readline()
                    error = process.stderr.readline()

                    if output == error == '' and process.poll() is not None:
                        break

                    if output:
                        log_file.write(output)
                        ffmpeg_logger.debug(output.strip())
                    if error:
                        log_file.write(error)
                        ffmpeg_logger.debug(error.strip())

                except Exception as e:
                    ffmpeg_logger.error(f"Error processing FFmpeg output: {e}")
                    break

            return_code = process.wait(timeout=timeout)
            return return_code == 0

    except subprocess.TimeoutExpired:
        process.kill()
        ffmpeg_logger.error("FFmpeg command timed out")
        return False
    except Exception as e:
        ffmpeg_logger.error(f"FFmpeg command failed: {e}")
        return False
    finally:
        os.chdir(current_dir)
