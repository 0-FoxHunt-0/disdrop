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

from .default_config import FFPMEG_LOG_FILE, LOG_DIR, LOG_FILE, TEMP_FILE_DIR

# Add custom log levels
SUCCESS_LEVEL = 25  # Between INFO and WARNING
SKIP_LEVEL = 15    # Between DEBUG and INFO

# Register new log levels
logging.addLevelName(SUCCESS_LEVEL, 'SUCCESS')
logging.addLevelName(SKIP_LEVEL, 'SKIP')

# Define the success and skip methods


def success(self, message, *args, **kwargs):
    if self.isEnabledFor(SUCCESS_LEVEL):
        self._log(SUCCESS_LEVEL, message, args, **kwargs)


def skip(self, message, *args, **kwargs):
    if self.isEnabledFor(SKIP_LEVEL):
        self._log(SKIP_LEVEL, message, args, **kwargs)


# Add methods to Logger class
logging.Logger.success = success
logging.Logger.skip = skip


def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = logging.getLogger(func.__module__)
        logger.debug(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}")
            raise
    return wrapper


class ColorFormatter(logging.Formatter):
    """Enhanced color formatter with better visual separation."""
    COLORS = {
        'DEBUG': '\033[0;36m',     # Cyan
        'INFO': '\033[0;37m',      # White
        'WARNING': '\033[0;33m',   # Yellow
        'ERROR': '\033[1;31m',     # Bright Red
        'CRITICAL': '\033[1;35m',  # Bright Purple
        'SUCCESS': '\033[1;32m',   # Bright Green
        'SKIP': '\033[0;34m'       # Blue
    }
    RESET = '\033[0m'

    def __init__(self, fmt=None, datefmt=None, style='%', use_color=True):
        super().__init__(fmt, datefmt, style)
        self.use_color = use_color

    def format(self, record):
        # Replace Unicode characters first
        if isinstance(record.msg, str):
            record.msg = record.msg.replace('✓', '>').replace('⚠', '!').replace(
                '→', '->').replace('⟶', '->').replace('─', '-').replace('│', '|')

        if record.levelname in ['ERROR', 'CRITICAL', 'SUCCESS']:
            record.msg = f"\n{'-' * 50}\n{record.msg}"

        if self.use_color:
            color = self.COLORS.get(record.levelname, self.RESET)
            if record.levelname in ['ERROR', 'CRITICAL']:
                record.msg = f"{color}! {record.msg}{self.RESET}"
            elif record.levelname == 'SUCCESS':
                record.msg = f"{color}> {record.msg}{self.RESET}"
            else:
                record.msg = f"{color}{record.msg}{self.RESET}"

        if isinstance(record.msg, str):
            record.msg = record.msg.replace('\\', '/')

        if record.levelname in ['ERROR', 'CRITICAL', 'SUCCESS']:
            record.msg = f"{record.msg}\n{'-' * 50}"

        return super().format(record)


class RotatingFileHandler(logging.FileHandler):
    def __init__(self, filename: Union[str, Path], max_bytes: int = 10485760, backup_count: int = 5):
        # Ensure parent directory exists
        Path(filename).parent.mkdir(parents=True, exist_ok=True)

        super().__init__(filename)
        self.max_bytes = max_bytes
        self.backup_count = backup_count

        # Ensure file has proper permissions
        try:
            Path(filename).chmod(0o644)
        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Failed to set log file permissions: {e}")

        self._should_rotate()

    def _should_rotate(self) -> None:
        if not Path(self.baseFilename).exists():
            return

        if Path(self.baseFilename).stat().st_size > self.max_bytes:
            backup_files = sorted(
                Path(self.baseFilename).parent.glob(f"{Path(self.baseFilename).name}.*"))
            while len(backup_files) >= self.backup_count:
                backup_files[0].unlink()
                backup_files = backup_files[1:]

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
            logging.getLogger(__name__).error(f"Failed to open log file: {e}")
            self.log_file = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def write(self, message: str) -> None:
        try:
            self.terminal.write(message)
            if self.log_file:
                self.log_file.write(message)
            self.flush()
        except (IOError, ValueError) as e:
            sys.stderr.write(f"TeeLogger write error: {str(e)}\n")

    def flush(self) -> None:
        try:
            self.terminal.flush()
            if self.log_file:
                self.log_file.flush()
        except (IOError, ValueError) as e:
            sys.stderr.write(f"TeeLogger flush error: {str(e)}\n")

    def close(self) -> None:
        if hasattr(self, 'log_file') and self.log_file and not self.log_file.closed:
            try:
                self.log_file.close()
            except Exception as e:
                sys.stderr.write(f"TeeLogger close error: {str(e)}\n")

    def fileno(self) -> int:
        return self.terminal.fileno()


def ensure_log_directories():
    """Ensure log directories exist and have proper permissions."""
    try:
        for directory in [LOG_DIR, Path(FFPMEG_LOG_FILE).parent]:
            directory.mkdir(parents=True, exist_ok=True)
            # Ensure write permissions
            directory.chmod(0o755)
    except Exception as e:
        print(f"Failed to set up log directories: {e}")
        raise


class WindowsConsoleHandler(logging.StreamHandler):
    """Custom handler for Windows console with robust Unicode support."""

    def __init__(self):
        super().__init__()
        if sys.platform == 'win32':
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleOutputCP(65001)
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            try:
                stream.write(msg + self.terminator)
                self.flush()
            except UnicodeEncodeError:
                stream.buffer.write(msg.encode('utf-8', errors='replace'))
                stream.buffer.write(self.terminator.encode('utf-8'))
                self.flush()
        except Exception:
            self.handleError(record)


def setup_logger(debug_mode: bool = False, verbose_mode: bool = False,
                 log_rotation_size: int = 10485760,
                 backup_count: int = 5) -> logging.Logger:
    """Setup enhanced logging system with improved visual formatting."""
    # Clear existing log files
    for log_file in [LOG_FILE, FFPMEG_LOG_FILE, LOG_DIR / 'debug.log']:
        if log_file.exists():
            log_file.unlink()

    root_logger = logging.getLogger()
    root_logger.handlers.clear()

    # Set base level based on debug flag
    base_level = logging.DEBUG if debug_mode else logging.INFO
    root_logger.setLevel(base_level)

    # Use a set to track handler names
    handler_names = set()

    def add_handler(handler, name):
        if name not in handler_names:
            root_logger.addHandler(handler)
            handler_names.add(name)

    # Console handler with color formatting
    console_handler = WindowsConsoleHandler()
    # Use improved formatting
    if verbose_mode:
        formatter = ColorFormatter(
            '\n%(asctime)s\n'
            '%(levelname)-8s | %(message)s\n'
        )
    else:
        formatter = ColorFormatter('%(levelname)-8s │ %(message)s')

    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    add_handler(console_handler, 'console')

    # Debug file handler for verbose logging
    if verbose_mode:
        debug_handler = RotatingFileHandler(
            LOG_DIR / 'debug.log',
            max_bytes=log_rotation_size,
            backup_count=backup_count
        )
        debug_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - '
            '%(funcName)s - %(message)s'
        ))
        debug_handler.setLevel(logging.DEBUG)
        add_handler(debug_handler, 'debug_file')

    # Regular file handler
    file_handler = RotatingFileHandler(
        LOG_FILE,
        max_bytes=log_rotation_size,
        backup_count=backup_count
    )
    file_handler.setFormatter(logging.Formatter(
        # Fixed levelname
        '%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
    ))
    file_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    add_handler(file_handler, 'file')

    # Set up FFmpeg logger with stream capture
    ffmpeg_logger = logging.getLogger('ffmpeg')
    ffmpeg_logger.handlers.clear()
    ffmpeg_logger.propagate = False

    ffmpeg_handler = RotatingFileHandler(
        FFPMEG_LOG_FILE,
        max_bytes=log_rotation_size,
        backup_count=backup_count
    )
    ffmpeg_handler.setFormatter(logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    ))
    ffmpeg_logger.addHandler(ffmpeg_handler)
    ffmpeg_logger.setLevel(logging.DEBUG if verbose_mode else logging.INFO)

    # Redirect FFmpeg output
    os.environ['FFMPEG_LOG_CAPTURE'] = '1'
    os.environ['FFMPEG_LOG_LEVEL'] = 'debug' if verbose_mode else 'info'
    os.environ['FFMPEG_LOG_FILE'] = str(FFPMEG_LOG_FILE)

    logger = logging.getLogger(__name__)
    logger.success("Logging system initialized successfully")

    if verbose_mode:
        logger.debug("Verbose debug logging enabled")
        # Log system info in verbose mode
        import platform
        import psutil
        logger.debug(f"System: {platform.system()} {platform.release()}")
        logger.debug(f"Python: {platform.python_version()}")
        logger.debug(f"CPU Cores: {psutil.cpu_count()}")
        logger.debug(
            f"Memory: {psutil.virtual_memory().total / (1024**3):.1f}GB")

    return logger


def log_exception(exc_type, exc_value, exc_traceback):
    if issubclass(exc_type, KeyboardInterrupt):
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    logging.critical("Uncaught exception",
                     exc_info=(exc_type, exc_value, exc_traceback))


def run_ffmpeg_command(command: list) -> bool:
    """Run FFmpeg command without timeout."""
    ffmpeg_logger = logging.getLogger('ffmpeg')
    current_dir = os.getcwd()

    try:
        process = subprocess.run(command, capture_output=True, text=True)

        if process.stderr:
            ffmpeg_logger.debug(process.stderr)

        return process.returncode == 0

    except Exception as e:
        ffmpeg_logger.error(f"Command failed: {e}")
        return False
    finally:
        os.chdir(current_dir)
