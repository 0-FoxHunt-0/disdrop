import atexit
from functools import wraps
import logging
import os
import subprocess
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, TextIO, Union

from .default_config import FFPMEG_LOG_FILE, LOG_DIR, LOG_FILE

# Custom log levels
SUCCESS_LEVEL = 25
SKIP_LEVEL = 15


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


# class LogLevel(Enum):
#     """Enhanced log levels with symbols and colors"""
#     DEBUG = ('◐', '\033[38;5;245m')    # Gray
#     INFO = ('ℹ', '\033[38;5;39m')      # Blue
#     WARNING = ('⚠', '\033[38;5;208m')   # Orange
#     ERROR = ('✖', '\033[38;5;196m')     # Red
#     CRITICAL = ('☠', '\033[38;5;200m')  # Magenta
#     SUCCESS = ('✔', '\033[38;5;40m')    # Green
#     SKIP = ('→', '\033[38;5;244m')      # Light gray


# class ColorFormatter(logging.Formatter):
#     """Enhanced color formatter with better visual separation."""

#     RESET = '\033[0m'
#     BOLD = '\033[1m'
#     DIM = '\033[2m'

#     def __init__(self, fmt=None, datefmt=None, style='%', use_color=True):
#         super().__init__(fmt, datefmt, style)
#         self.use_color = use_color and sys.platform != 'win32'
#         self._last_level = None
#         self._section_count = 0

#     def format(self, record):
#         if not self.use_color:
#             return super().format(record)

#         # Add visual breaks between different log levels
#         if self._last_level != record.levelname:
#             if self._section_count > 0:
#                 record.msg = f"\n{record.msg}"
#             self._last_level = record.levelname
#             self._section_count += 1

#         level_info = LogLevel[record.levelname] if record.levelname in LogLevel.__members__ else LogLevel.INFO
#         symbol, color = level_info.value

#         # Format the level name with fixed width and symbol
#         level_display = f"{color}{symbol} {record.levelname:<8}{self.RESET}"

#         # Add timestamp for debug messages
#         timestamp = ""
#         if record.levelname == 'DEBUG':
#             timestamp = f"{self.DIM}{self.format_time(record)}{self.RESET} "

#         # Format messages based on level
#         if record.levelname in ['ERROR', 'CRITICAL']:
#             msg = self._format_box(record.msg, color)
#         elif record.levelname == 'SUCCESS':
#             msg = self._format_box(record.msg, LogLevel.SUCCESS.value[1])
#         else:
#             msg = record.msg

#         return f"{timestamp}{level_display} │ {msg}"

#     def _format_box(self, msg: str, color: str) -> str:
#         """Format message in a colored box."""
#         lines = msg.split('\n')
#         width = max(len(line) for line in lines) + 2

#         box_top = f"{color}╭{'─' * width}╮{self.RESET}"
#         box_bottom = f"{color}╰{'─' * width}╯{self.RESET}"

#         formatted_lines = [box_top]
#         for line in lines:
#             padding = ' ' * (width - len(line))
#             formatted_lines.append(f"{color}│ {line}{padding} │{self.RESET}")
#         formatted_lines.append(box_bottom)

#         return '\n'.join(formatted_lines)


# class ConsoleHandler(logging.StreamHandler):
#     """Enhanced console handler with proper encoding and color support."""

#     def __init__(self):
#         super().__init__()
#         if sys.platform == 'win32':
#             # Enable ANSI colors on Windows
#             from ctypes import windll
#             k = windll.kernel32
#             k.SetConsoleMode(k.GetStdHandle(-11), 7)

class LogStyle:
    """Modern logging styles with rich formatting"""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'

    # Background colors
    BG_BLACK = '\033[40m'
    BG_RED = '\033[41m'
    BG_GREEN = '\033[42m'
    BG_YELLOW = '\033[43m'
    BG_BLUE = '\033[44m'
    BG_MAGENTA = '\033[45m'
    BG_CYAN = '\033[46m'

    # Modern color palette
    SLATE = '\033[38;5;246m'
    AZURE = '\033[38;5;39m'
    EMERALD = '\033[38;5;48m'
    AMBER = '\033[38;5;214m'
    ROSE = '\033[38;5;204m'
    VIOLET = '\033[38;5;141m'


class LogLevel(Enum):
    """Enhanced log levels with modern symbols and colors"""
    DEBUG = ('⚙', LogStyle.SLATE)
    INFO = ('○', LogStyle.AZURE)
    WARNING = ('△', LogStyle.AMBER)
    ERROR = ('✕', LogStyle.ROSE)
    CRITICAL = ('⬢', LogStyle.VIOLET)
    SUCCESS = ('●', LogStyle.EMERALD)
    SKIP = ('→', LogStyle.SLATE)


class ModernFormatter(logging.Formatter):
    """Modern formatter with improved visual hierarchy and readability"""

    def __init__(self, fmt=None, datefmt=None, style='%', use_color=True):
        super().__init__(fmt, datefmt, style)
        self.use_color = use_color and sys.platform != 'win32'
        self._last_timestamp = None

    def format(self, record):
        if not self.use_color:
            return super().format(record)

        # Get level styling
        level_info = LogLevel[record.levelname] if record.levelname in LogLevel.__members__ else LogLevel.INFO
        symbol, color = level_info.value

        # Format timestamp with visual grouping
        current_timestamp = datetime.fromtimestamp(record.created)
        timestamp_changed = (not self._last_timestamp or
                             current_timestamp.minute != self._last_timestamp.minute)

        if timestamp_changed:
            self._last_timestamp = current_timestamp
            timestamp_header = f"\n{LogStyle.DIM}╮ {current_timestamp.strftime('%H:%M')}{LogStyle.RESET}\n"
        else:
            timestamp_header = ""

        # Format the message
        if record.levelname in ['ERROR', 'CRITICAL']:
            msg = self._format_error(record.msg, color)
        elif record.levelname == 'SUCCESS':
            msg = self._format_success(record.msg)
        else:
            msg = f"{color}{record.msg}{LogStyle.RESET}"

        # Build the log line with modern indentation
        log_line = (
            f"{timestamp_header}"
            f"{LogStyle.DIM}│{LogStyle.RESET} "
            f"{color}{symbol}{LogStyle.RESET} "
            f"{msg}"
        )

        return log_line

    def _format_error(self, msg: str, color: str) -> str:
        """Format error messages with modern styling"""
        lines = msg.split('\n')
        formatted = [
            f"{color}{LogStyle.BG_BLACK} {lines[0]} {LogStyle.RESET}"
        ]
        if len(lines) > 1:
            formatted.extend(
                f"{LogStyle.DIM}│{LogStyle.RESET} {LogStyle.DIM}{line}{LogStyle.RESET}"
                for line in lines[1:]
            )
        return '\n'.join(formatted)

    def _format_success(self, msg: str) -> str:
        """Format success messages with subtle highlighting"""
        return f"{LogStyle.EMERALD}{LogStyle.ITALIC}{msg}{LogStyle.RESET}"


class ModernConsoleHandler(logging.StreamHandler):
    """Console handler with modern output handling"""

    def __init__(self):
        if sys.platform == 'win32':
            import codecs
            sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
            from ctypes import windll
            k = windll.kernel32
            k.SetConsoleMode(k.GetStdHandle(-11), 7)
        super().__init__()

    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            if sys.platform == 'win32':
                try:
                    stream.write(msg)
                except UnicodeEncodeError:
                    stream.buffer.write(msg.encode('utf-8'))
            else:
                stream.write(msg)
            stream.write(self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def setup_modern_logger(debug_mode: bool = False) -> logging.Logger:
    """Set up logger with modern styling"""
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    console_handler = ModernConsoleHandler()
    formatter = ModernFormatter()
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add custom log levels
    logging.addLevelName(25, 'SUCCESS')
    logging.addLevelName(15, 'SKIP')

    def success(self, message, *args, **kwargs):
        self._log(25, message, args, **kwargs)

    def skip(self, message, *args, **kwargs):
        self._log(15, message, args, **kwargs)

    logging.Logger.success = success
    logging.Logger.skip = skip

    logger = logging.getLogger(__name__)
    logger.success("System initialized")

    return logger


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
