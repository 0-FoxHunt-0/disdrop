# logging_system.py

import atexit
import functools
import logging
import os
import queue
import subprocess
import sys
from datetime import datetime
from functools import partial, wraps
from pathlib import Path
import threading
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
        'DEBUG': '\033[0;36m',    # Cyan
        'INFO': '\033[0;37m',     # White
        'SUCCESS': '\033[0;32m',  # Green
        'WARNING': '\033[0;33m',  # Yellow
        'ERROR': '\033[0;31m',    # Red
        'CRITICAL': '\033[0;35m'  # Purple
    }
    RESET = '\033[0m'

    def format(self, record):
        # Mark successful operation messages in green
        if record.levelname == 'INFO' and any(success_term in record.msg.lower()
           for success_term in ['successful', 'succeeded', 'completed', 'finished']):
            color = self.COLORS['SUCCESS']
        else:
            color = self.COLORS.get(record.levelname, self.RESET)

        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)


class AsyncRotatingFileHandler(logging.Handler):
    def __init__(self, filename: Path, max_bytes: int = 10485760,
                 backup_count: int = 5):
        super().__init__()
        self.filename = Path(filename)
        self.max_bytes = max_bytes
        self.backup_count = backup_count
        self.queue = queue.Queue(maxsize=10000)
        self.writer_thread = threading.Thread(target=self._writer_thread,
                                              daemon=True)
        self.writer_thread.start()
        self.lock = threading.Lock()
        atexit.register(self.close)

    def _writer_thread(self):
        while True:
            try:
                record = self.queue.get()
                if record is None:
                    break
                self._do_write(record)
            except Exception as e:
                print(f"Error in writer thread: {e}")

    def _do_write(self, record):
        msg = self.format(record)
        with self.lock:
            if self.should_rotate():
                self.do_rotation()
            with open(self.filename, 'a', encoding='utf-8') as f:
                f.write(msg + '\n')

    def should_rotate(self) -> bool:
        try:
            return self.filename.stat().st_size > self.max_bytes
        except FileNotFoundError:
            return False

    def do_rotation(self):
        if not self.filename.exists():
            return

        for i in range(self.backup_count - 1, 0, -1):
            sfn = f"{self.filename}.{i}"
            dfn = f"{self.filename}.{i + 1}"
            if Path(sfn).exists():
                Path(sfn).rename(dfn)

        dfn = f"{self.filename}.1"
        Path(self.filename).rename(dfn)

    def emit(self, record):
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            print("Warning: Logging queue full, dropping message")

    def close(self):
        self.queue.put(None)
        self.writer_thread.join(timeout=1.0)
        super().close()


class MetricsLogger:
    def __init__(self):
        self.metrics = {}
        self.lock = threading.Lock()

    def track(self, metric: str, value: float):
        with self.lock:
            if metric not in self.metrics:
                self.metrics[metric] = []
            self.metrics[metric].append(value)

    def get_stats(self, metric: str) -> dict:
        with self.lock:
            values = self.metrics.get(metric, [])
            if not values:
                return {}
            return {
                'count': len(values),
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values)
            }


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


def setup_logging(debug_mode: bool = False):
    # Get root logger
    root = logging.getLogger()

    # Remove existing handlers
    if root.hasHandlers():
        root.handlers.clear()

    # Set level
    root.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Console handler with color formatting
    console = logging.StreamHandler()
    console.setFormatter(ColorFormatter('%(levelname)s: %(message)s'))
    root.addHandler(console)

    # Async file handler for main log
    main_handler = AsyncRotatingFileHandler(LOG_FILE)
    main_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'))
    root.addHandler(main_handler)

    # Async file handler for FFmpeg log
    ffmpeg_handler = AsyncRotatingFileHandler(FFPMEG_LOG_FILE)
    ffmpeg_handler.setFormatter(logging.Formatter(
        '%(asctime)s [FFmpeg] %(message)s'))
    ffmpeg_logger = logging.getLogger('ffmpeg')
    ffmpeg_logger.addHandler(ffmpeg_handler)

    # Set up metrics logger
    metrics = MetricsLogger()
    return metrics


def log_execution_time(logger: Optional[MetricsLogger] = None):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = datetime.now()
            try:
                result = func(*args, **kwargs)
                if logger:
                    duration = (datetime.now() - start).total_seconds()
                    logger.track(f'{func.__name__}_duration', duration)
                return result
            except Exception as e:
                logging.error(f"Error in {func.__name__}: {e}")
                raise
        return wrapper
    return decorator


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
