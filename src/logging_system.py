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
        'DEBUG': '\033[0;36m',  # Cyan
        'INFO': '\033[0;37m',   # White
        'WARNING': '\033[0;33m',  # Yellow
        'ERROR': '\033[0;31m',   # Red
        'CRITICAL': '\033[0;35m',  # Purple
        'SUCCESS': '\033[0;32m'   # Green
    }
    RESET = '\033[0m'

    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)


class RotatingFileHandler(logging.FileHandler):
    def __init__(self, filename: Union[str, Path], max_bytes: int = 10485760):
        super().__init__(filename)
        self.max_bytes = max_bytes
        self._should_rotate()

    def _should_rotate(self) -> None:
        if Path(self.baseFilename).exists():
            if Path(self.baseFilename).stat().st_size > self.max_bytes:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup = Path(f"{self.baseFilename}.{timestamp}")
                Path(self.baseFilename).rename(backup)


class TeeLogger:
    def __init__(self, filename: Union[str, Path], encoding: str = 'utf-8'):
        self.terminal: TextIO = sys.stdout
        self.log_file: TextIO = open(filename, 'w', encoding=encoding)
        atexit.register(self.close)

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log_file.write(message)
        self.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log_file.flush()

    def close(self) -> None:
        if not self.log_file.closed:
            self.log_file.close()

    def fileno(self) -> int:
        return self.terminal.fileno()

# TODO: Add a success level to logging


def setup_logger(debug_mode: bool = False) -> None:
    Path(LOG_DIR).mkdir(parents=True, exist_ok=True)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(ColorFormatter('%(levelname)s: %(message)s'))
    console_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    file_handler = RotatingFileHandler(LOG_FILE)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'))
    file_handler.setLevel(logging.DEBUG)

    ffmpeg_handler = RotatingFileHandler(FFPMEG_LOG_FILE)
    ffmpeg_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'))

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    ffmpeg_logger = logging.getLogger('ffmpeg')
    ffmpeg_logger.addHandler(ffmpeg_handler)

    sys.excepthook = log_exception
    logging.info("Logging system initialized")


def log_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.critical("Uncaught exception",
                     exc_info=(exc_type, exc_value, exc_traceback))


def run_ffmpeg_command(command: list) -> bool:
    ffmpeg_logger = logging.getLogger('ffmpeg')
    current_dir = os.getcwd()

    try:
        # Create temporary directory for FFmpeg pass logs
        # temp_pass_dir = Path(TEMP_FILE_DIR) / "ffmpeg_pass"
        # temp_pass_dir.mkdir(parents=True, exist_ok=True)
        # os.chdir(str(temp_pass_dir))

        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        with open(FFPMEG_LOG_FILE, 'w', encoding='utf-8') as log_file:
            while True:
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

        return process.returncode == 0

    except Exception as e:
        ffmpeg_logger.error(f"FFmpeg command failed: {e}")
        return False
    finally:
        os.chdir(current_dir)
