import atexit
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from collections import defaultdict
from datetime import datetime
from enum import Enum, auto
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, TextIO, Tuple, Union

import colorlog
import psutil
from tqdm import tqdm

from .default_config import FFPMEG_LOG_FILE, LOG_DIR, LOG_FILE

# Custom log levels
SUCCESS_LEVEL = 25
SKIP_LEVEL = 15
PERFORMANCE_LEVEL = 26

# Global logger lock for thread safety
LOGGER_LOCK = threading.RLock()

# Global cache to track if a logger has been configured
CONFIGURED_LOGGERS = set()

# Setup FFmpeg logger once
FFMPEG_LOGGER_CONFIGURED = False


class ModernLogStyle(Enum):
    """Modern log styling and color codes for better readability."""
    DEFAULT = "white"
    AZURE = "#2563eb"  # A nice blue
    CYAN = "#06b6d4"
    SLATE = "#64748b"  # Subtle gray-blue
    AMBER = "#d97706"  # Warm amber
    EMERALD = "#10b981"  # Rich green
    ROSE = "#e11d48"  # Vibrant rose/red
    PURPLE = "#a855f7"
    PINK = "#ec4899"
    LIME = "#84cc16"
    INDIGO = "#6366f1"
    ORANGE = "#f97316"
    TEAL = "#14b8a6"
    RED = "#ef4444"
    YELLOW = "#eab308"
    GREEN = "#22c55e"
    BLUE = "#3b82f6"
    GRAY = "#6b7280"
    RESET = "\033[0m"
    BOLD = "\033[1m"

    def as_rgb(self) -> str:
        """Convert to standard terminal RGB format if hex value."""
        if self.value.startswith('#'):
            # Convert hex to rgb
            rgb = tuple(int(self.value.lstrip('#')[i:i+2], 16)
                        for i in (0, 2, 4))
            return f"\033[38;2;{rgb[0]};{rgb[1]};{rgb[2]}m"
        # Otherwise return as is
        return self.value

    def get_style(self) -> str:
        """Get the style string."""
        return self.value if not self.value.startswith('#') else self.as_rgb()


def get_color_for_severity(severity: str) -> str:
    """Return a color based on message severity."""
    return {
        'DEBUG': ModernLogStyle.SLATE.value,
        'INFO': ModernLogStyle.DEFAULT.value,
        'SUCCESS': ModernLogStyle.EMERALD.value,
        'WARNING': ModernLogStyle.AMBER.value,
        'ERROR': ModernLogStyle.ROSE.value,
        'CRITICAL': ModernLogStyle.RED.value,
        'SKIP': ModernLogStyle.GRAY.value,
        'PERFORMANCE': ModernLogStyle.INDIGO.value,
    }.get(severity, ModernLogStyle.DEFAULT.value)


class UnifiedLogFormatter(logging.Formatter):
    """Advanced formatter with time estimation and color support."""

    def __init__(self, log_colors=None):
        """Initialize formatter with color support."""
        super().__init__(fmt='%(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        self.log_colors = log_colors or {
            'DEBUG': ModernLogStyle.SLATE.value,
            'INFO': ModernLogStyle.DEFAULT.value,
            'SUCCESS': ModernLogStyle.EMERALD.value,
            'WARNING': ModernLogStyle.AMBER.value,
            'ERROR': ModernLogStyle.ROSE.value,
            'CRITICAL': ModernLogStyle.RED.value,
            'SKIP': ModernLogStyle.GRAY.value,
            'PERFORMANCE': ModernLogStyle.INDIGO.value,
        }
        self.supports_color = self._check_color_support()

    def _check_color_support(self):
        """Check if terminal supports colors."""
        # Check if output is a TTY
        is_tty = hasattr(sys.stdout, 'isatty') and sys.stdout.isatty()

        # Check for Windows terminal that supports ANSI
        if sys.platform == 'win32':
            # Windows 10 build 14931+ supports ANSI colors in cmd.exe
            # Check specific environments that support color
            if os.environ.get('TERM_PROGRAM') == 'vscode':
                return True  # VS Code integrated terminal supports colors

            if 'ANSICON' in os.environ:
                return True  # ANSICON is installed

            if 'WT_SESSION' in os.environ:
                return True  # Windows Terminal

            if 'ConEmuANSI' in os.environ:
                return True  # ConEmu

            # Check Windows version for native ANSI support (Windows 10+)
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32

                # Check if SetConsoleMode function exists and supports ENABLE_VIRTUAL_TERMINAL_PROCESSING
                if hasattr(kernel32, 'SetConsoleMode'):
                    # Try to enable ANSI escape sequence processing
                    ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
                    mode = ctypes.c_ulong()
                    if kernel32.GetConsoleMode(kernel32.GetStdHandle(-11), ctypes.byref(mode)):
                        # Successfully got console mode, try to set ANSI support
                        new_mode = mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
                        if kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), new_mode):
                            return True
            except:
                pass

            # Fallback check for TERM=xterm
            if os.environ.get('TERM') == 'xterm':
                return True

            # Default to no color for most Windows environments
            # unless we've detected a compatible terminal
            return False

        # For other platforms, just check if it's a TTY
        return is_tty

    def get_timestamp(self):
        """Get a nicely formatted timestamp."""
        return time.strftime("%H:%M:%S", time.localtime())

    def colorize(self, text, color=None):
        """Add ANSI color codes to text if supported."""
        if not color or not self.supports_color:
            return text

        # Convert hex color to ANSI RGB format
        if color.startswith('#'):
            try:
                r = int(color[1:3], 16)
                g = int(color[3:5], 16)
                b = int(color[5:7], 16)
                return f"\033[38;2;{r};{g};{b}m{text}\033[0m"
            except Exception:
                return text
        # Return as is for named colors or ANSI codes
        elif color.startswith('\033'):
            return f"{color}{text}{ModernLogStyle.RESET.value}"
        return text

    def format(self, record):
        """Format a record with enhanced styling and information."""
        if not hasattr(record, 'original_format_message'):
            record.original_format_message = record.getMessage
            record.getMessage = lambda: record.original_format_message()

        # Get log level color
        color = self.log_colors.get(record.levelname, None)

        # Get timestamp
        ts = self.get_timestamp()

        # Format basic message with level prefix for better visibility
        level_str = record.levelname

        # Format message based on log level and content
        if record.exc_info:
            # Handle exceptions with detailed formatting
            formatted_msg = super().format(record)
            if self.supports_color:
                # Add error trace formatting
                lines = formatted_msg.split('\n')
                if len(lines) > 1:
                    # First line is the message
                    formatted_message = self.colorize(lines[0], color)
                    # Format traceback with error color
                    error_color = self.log_colors.get(
                        'ERROR', ModernLogStyle.ROSE.value)
                    trace_lines = '\n'.join(
                        [self.colorize(line, error_color) for line in lines[1:]])
                    formatted_message = f"{formatted_message}\n{trace_lines}"
                else:
                    formatted_message = self.colorize(formatted_msg, color)
            else:
                formatted_message = formatted_msg
        else:
            # Basic message formatting with colors
            msg = record.getMessage()

            if self.supports_color:
                # Add colored level prefix for important messages
                if record.levelno >= logging.WARNING or record.levelname in ['SUCCESS', 'PERFORMANCE']:
                    level_display = f"[{level_str}] "
                    formatted_message = f"{self.colorize(level_display, color)}{self.colorize(msg, color)}"
                else:
                    formatted_message = self.colorize(msg, color)
            else:
                # Plain formatting for non-color terminals
                if record.levelno >= logging.WARNING or record.levelname in ['SUCCESS', 'PERFORMANCE']:
                    formatted_message = f"[{level_str}] {msg}"
                else:
                    formatted_message = msg

        # Add context details if available
        if hasattr(record, 'details') and record.details:
            details_str = ' | '.join(
                [f"{k}: {v}" for k, v in record.details.items()])
            if self.supports_color:
                details_str = self.colorize(
                    f"  {details_str}", ModernLogStyle.SLATE.value)
            formatted_message = f"{formatted_message}\n{details_str}"

        return formatted_message


class SafeUnicodeFormatter(logging.Formatter):
    """Formatter that handles Unicode encoding errors gracefully."""

    def __init__(self, fmt=None, datefmt=None, style='%', validate=True):
        super().__init__(fmt, datefmt, style, validate)

    def format(self, record):
        """Format the log record with Unicode safety."""
        try:
            # First, use the normal formatter
            message = super().format(record)

            # Check if we're on Windows and need special handling
            if sys.platform == 'win32':
                # Replace problematic Unicode characters
                message = self._make_windows_safe(message)

            return message
        except Exception as e:
            # Fallback to a safe version if formatting fails
            return f"[Log formatting error: {e}] {record.getMessage()}"

    def _make_windows_safe(self, message):
        """Replace problematic Unicode characters with ASCII equivalents."""
        # Map of Unicode characters to ASCII replacements
        replacements = {
            '\u2192': '->',  # Right arrow →
            '\u2190': '<-',  # Left arrow ←
            '\u2713': '+',   # Checkmark ✓
            '\u2717': 'x',   # Cross ✗
            '\u26a0': '!',   # Warning ⚠
            '\u2714': '+',   # Heavy checkmark ✔
            '\u2718': 'x',   # Heavy cross ✘
            '\u21bb': '*',   # Clockwise open circle arrow ↻
            '\u2699': '*',   # Gear ⚙
            '\u26a1': '*',   # Lightning ⚡
        }

        # Replace each character
        for char, replacement in replacements.items():
            message = message.replace(char, replacement)

        return message


class UnifiedLogger:
    """Modern unified logger with detailed output by default"""

    def __init__(self, logger_name='app'):
        self.logger = logging.getLogger(logger_name)
        self.logger_name = logger_name

        # Only configure if not already configured
        with LOGGER_LOCK:
            if not self._is_logger_configured(logger_name):
                self.logger.setLevel(logging.INFO)

                # Remove any existing handlers to prevent duplicates
                for handler in self.logger.handlers[:]:
                    self.logger.removeHandler(handler)

                # Set up console handler with unified formatter
                console_handler = logging.StreamHandler()
                console_handler.setFormatter(UnifiedLogFormatter())
                self.logger.addHandler(console_handler)

                # Mark as configured
                CONFIGURED_LOGGERS.add(logger_name)

        self._processing_start = None
        self._current_phase = None
        self._timers = {}
        self._timer_lock = threading.Lock()

    def _is_logger_configured(self, logger_name):
        """Check if logger is already configured properly."""
        if logger_name in CONFIGURED_LOGGERS:
            return True

        # Check if the logger has any handlers
        logger = logging.getLogger(logger_name)
        if logger.handlers:
            # If it has handlers, consider it configured and add to our tracking set
            CONFIGURED_LOGGERS.add(logger_name)
            return True

        return False

    def __getattr__(self, name):
        """Delegate any undefined attributes to the internal logger"""
        return getattr(self.logger, name)

    def start_phase(self, phase_name: str):
        """Start a new processing phase with timing"""
        self._current_phase = phase_name
        self._processing_start = time.time()
        extra = {'phase': phase_name, 'context': 'progress'}
        self.logger.info(f"Starting {phase_name}", extra=extra)

    def end_phase(self, phase_name: str = None):
        """End the current phase and log performance metrics"""
        if self._processing_start is None:
            return

        phase = phase_name or self._current_phase
        elapsed = time.time() - self._processing_start

        # Log with performance level
        extra = {'context': 'timing', 'details': {
            'Elapsed': f"{elapsed:.2f}s"}}
        self.logger.log(
            PERFORMANCE_LEVEL, f"Completed {phase}", extra=extra)

        # Reset timers
        self._processing_start = None

    def start_timer(self, name: str):
        """Start a named timer for performance tracking"""
        with self._timer_lock:
            self._timers[name] = time.time()

    def end_timer(self, name: str, log_level=PERFORMANCE_LEVEL, include_system_stats=False):
        """End a named timer and log performance metrics"""
        with self._timer_lock:
            if name not in self._timers:
                return

            elapsed = time.time() - self._timers[name]
            details = {'Elapsed time': f"{elapsed:.3f}s"}

            # Add system stats if requested
            if include_system_stats:
                try:
                    cpu_percent = psutil.cpu_percent()
                    mem_info = psutil.virtual_memory()
                    details.update({
                        'CPU': f"{cpu_percent:.1f}%",
                        'Memory': f"{mem_info.percent:.1f}% ({mem_info.used / (1024**3):.1f}GB)"
                    })
                except Exception:
                    pass  # Fail silently if psutil has issues

            # Log performance metrics
            extra = {'context': 'timing', 'details': details}
            self.logger.log(log_level, f"Performance [{name}]", extra=extra)

            # Remove the timer
            del self._timers[name]

    def log_progress(self, message: str, current: int, total: int, details: dict = None, error_count: int = 0):
        """Log progress with percentage, details and error count"""
        if total == 0:
            percentage = 100
        else:
            percentage = (current / total) * 100

        progress_bar = self._create_progress_bar(percentage, error_count)

        extra = {
            'highlight': True,
            'context': 'progress',
            'details': {
                'Progress': f"{current}/{total} ({percentage:.1f}%)",
                **(details or {})
            }
        }

        self.logger.info(f"{message}\n  {progress_bar}", extra=extra)

    def _create_progress_bar(self, percentage: float, error_count: int = 0, width: int = 30) -> str:
        """Create a vibrant progress bar with colored indicators.

        Args:
            percentage: Progress percentage (0-100)
            error_count: Number of errors encountered
            width: Width of the progress bar in characters

        Returns:
            Formatted progress bar string
        """
        s = ModernLogStyle
        filled = int(width * percentage / 100)

        # Select color based on percentage and errors
        if error_count > 3:
            # Critical errors
            bar_color = s.ROSE.as_rgb()
            empty_color = s.SLATE.as_rgb()
            fill_char = '█'  # Full block
            empty_char = '░'  # Light shade
        elif error_count > 0:
            # Some errors
            bar_color = s.AMBER.as_rgb()
            empty_color = s.SLATE.as_rgb()
            fill_char = '█'  # Full block
            empty_char = '░'  # Light shade
        elif percentage >= 100:
            # Completed
            bar_color = s.EMERALD.as_rgb()
            empty_color = s.SLATE.as_rgb()
            fill_char = '█'  # Full block
            empty_char = '░'  # Light shade
        else:
            # Normal progress
            bar_color = s.AZURE.as_rgb()
            empty_color = s.SLATE.as_rgb()
            fill_char = '█'  # Full block
            empty_char = '░'  # Light shade

        # Create the bar with appropriate color
        bar = f"{bar_color}{fill_char * filled}{empty_color}{empty_char * (width - filled)}{s.RESET.value}"

        # Add percentage display
        percentage_display = f" {s.BOLD.value}{percentage:.1f}%{s.RESET.value}"

        # Add error indicator if needed
        if error_count > 0:
            error_color = s.AMBER.as_rgb() if error_count < 3 else s.ROSE.as_rgb()
            bar += f"{percentage_display} {error_color}({error_count} error{'s' if error_count > 1 else ''}){s.RESET.value}"
        else:
            bar += percentage_display

        return bar


def setup_application_logging(debug_mode=False, verbose_mode=False):
    """Set up centralized application logging.

    This creates a root logger and an application logger that can be shared
    across all modules. It configures both console and file logging.

    Args:
        debug_mode: Enable debug logging
        verbose_mode: Enable verbose debug logging

    Returns:
        The configured application logger
    """
    with LOGGER_LOCK:
        # Clear all existing handlers from the root logger
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Create 'app' logger that will be used across all modules
        app_logger = logging.getLogger('app')

        # Remove any existing handlers to prevent duplicates
        for handler in app_logger.handlers[:]:
            app_logger.removeHandler(handler)

        # Create and configure console handler with unified formatter
        console_handler = logging.StreamHandler()
        formatter = UnifiedLogFormatter()
        console_handler.setFormatter(formatter)

        # Set appropriate log level
        level = logging.DEBUG if debug_mode or verbose_mode else logging.INFO
        root_logger.setLevel(level)
        app_logger.setLevel(level)

        # Add handlers to both loggers
        root_logger.addHandler(console_handler)

        # Configure file logging
        log_file = LOG_DIR / 'application.log'
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        ))
        file_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

        app_logger.addHandler(file_handler)

        # For debug mode, add an additional debug log file
        if debug_mode:
            debug_file = LOG_DIR / 'debug.log'
            debug_handler = logging.FileHandler(debug_file, mode='w')
            debug_handler.setLevel(logging.DEBUG)
            debug_handler.setFormatter(formatter)
            app_logger.addHandler(debug_handler)

        # Add custom log levels
        logging.addLevelName(SUCCESS_LEVEL, 'SUCCESS')
        logging.addLevelName(SKIP_LEVEL, 'SKIP')
        logging.addLevelName(PERFORMANCE_LEVEL, 'PERFORMANCE')

        # Add success method to logger
        def success(self, message, *args, **kwargs):
            self.log(SUCCESS_LEVEL, message, *args, **kwargs)

        def skip(self, message, *args, **kwargs):
            self.log(SKIP_LEVEL, message, *args, **kwargs)

        def performance(self, message, *args, **kwargs):
            self.log(PERFORMANCE_LEVEL, message, *args, **kwargs)

        # Add methods to Logger class if they don't already exist
        if not hasattr(logging.Logger, 'success'):
            logging.Logger.success = success

        if not hasattr(logging.Logger, 'skip'):
            logging.Logger.skip = skip

        if not hasattr(logging.Logger, 'performance'):
            logging.Logger.performance = performance

        # Mark loggers as configured
        CONFIGURED_LOGGERS.add('app')

        # Ensure propagation is disabled to prevent duplicate logs
        app_logger.propagate = False

        # Add shutdown cleanup
        atexit.register(shutdown_logging)

        return app_logger


def get_logger(name='app'):
    """Get a preconfigured logger.

    This ensures a logger is properly configured even if the main application
    logger hasn't been set up yet.

    Args:
        name: Logger name, defaults to 'app'

    Returns:
        A configured logger instance
    """
    with LOGGER_LOCK:
        logger = logging.getLogger(name)

        # If this logger or the app logger has not been configured, set up basic config
        if name not in CONFIGURED_LOGGERS and 'app' not in CONFIGURED_LOGGERS:
            # Clear any existing handlers to prevent duplicates
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Add basic console handler
            handler = logging.StreamHandler()
            formatter = UnifiedLogFormatter()
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)

            # Disable propagation to prevent duplicate logs
            logger.propagate = False

            # Mark as configured
            CONFIGURED_LOGGERS.add(name)

            # Add custom log levels for this handler
            if not hasattr(logging.Logger, 'success'):
                def success(self, message, *args, **kwargs):
                    self.log(SUCCESS_LEVEL, message, *args, **kwargs)
                logging.Logger.success = success

                def skip(self, message, *args, **kwargs):
                    self.log(SKIP_LEVEL, message, *args, **kwargs)
                logging.Logger.skip = skip

                def performance(self, message, *args, **kwargs):
                    self.log(PERFORMANCE_LEVEL, message, *args, **kwargs)
                logging.Logger.performance = performance

        return logger


def shutdown_logging():
    """Properly close all logs and handlers during shutdown."""
    loggers = [logging.getLogger(name) for name in CONFIGURED_LOGGERS]
    loggers.append(logging.getLogger())  # Add root logger

    for logger in loggers:
        for handler in logger.handlers[:]:
            try:
                handler.flush()
                handler.close()
                logger.removeHandler(handler)
            except:
                pass  # Ignore errors during shutdown


# Performance monitoring decorator with error handling
def performance_monitor(func: Callable) -> Callable:
    """Decorator to monitor and log function performance."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger('performance')
        func_name = func.__name__
        start_time = time.time()

        try:
            logger.debug(f"Starting {func_name}")
            result = func(*args, **kwargs)
            elapsed_time = time.time() - start_time

            # Log performance metrics
            system_info = {}
            try:
                system_info['CPU'] = f"{psutil.cpu_percent()}%"
                mem = psutil.virtual_memory()
                system_info['Memory'] = f"{mem.percent}% ({mem.used / (1024**3):.1f}GB)"
            except Exception:
                pass

            # Log detailed metrics
            logger.log(
                PERFORMANCE_LEVEL,
                f"Performance [{func_name}] completed in {elapsed_time:.3f}s",
                extra={'details': system_info, 'context': 'timing'}
            )

            return result
        except Exception as e:
            elapsed_time = time.time() - start_time
            logger.error(
                f"Error in {func_name} after {elapsed_time:.3f}s: {str(e)}", exc_info=True)
            raise

    return wrapper


# Helper decorator for logging function calls
def log_function_call(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger()
        logger.debug(f"Entering {func.__name__}")
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Exiting {func.__name__}")
            return result
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            raise
    return wrapper


# Thread-safe progress bar logger
class ThreadSafeProgressLogger:
    """Thread-safe progress logging for multi-threaded processing."""

    def __init__(self, total: int, desc: str, unit: str = "items"):
        self.progress_bar = tqdm(total=total, desc=desc, unit=unit)
        self.lock = threading.Lock()
        self.logger = get_logger()
        self.errors = []
        self.error_lock = threading.Lock()

    def update(self, n: int = 1, status: str = None):
        """Update progress safely from multiple threads."""
        with self.lock:
            self.progress_bar.update(n)
            if status:
                self.progress_bar.set_description(status)

    def add_error(self, error_msg: str):
        """Add error message thread-safely."""
        with self.error_lock:
            self.errors.append(error_msg)

    def get_error_summary(self) -> str:
        """Get summary of all errors."""
        with self.error_lock:
            if not self.errors:
                return "No errors occurred"

            if len(self.errors) > 5:
                # Summarize errors if there are too many
                return f"{len(self.errors)} errors occurred. First 5: " + ", ".join(self.errors[:5])
            else:
                return "Errors: " + ", ".join(self.errors)

    def close(self):
        """Close progress bar."""
        with self.lock:
            self.progress_bar.close()


# Function to setup the FFmpeg logger
def setup_ffmpeg_logging():
    """Configure a dedicated logger for FFmpeg output."""
    global FFMPEG_LOGGER_CONFIGURED

    with LOGGER_LOCK:
        if FFMPEG_LOGGER_CONFIGURED:
            return logging.getLogger('ffmpeg')

        # Create logs directory if it doesn't exist
        LOG_DIR.mkdir(parents=True, exist_ok=True)

        # Configure FFmpeg logger
        ffmpeg_logger = logging.getLogger('ffmpeg')
        ffmpeg_logger.setLevel(logging.DEBUG)
        ffmpeg_logger.propagate = False  # Don't propagate to root logger

        # Create file handler for FFmpeg logs
        file_handler = logging.FileHandler(FFPMEG_LOG_FILE, mode='a')
        file_handler.setLevel(logging.DEBUG)

        # Create simple formatter for FFmpeg logs
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        # Add handler to logger
        ffmpeg_logger.addHandler(file_handler)

        FFMPEG_LOGGER_CONFIGURED = True
        return ffmpeg_logger


def rotate_ffmpeg_log(max_size_mb=100, backup_count=3):
    """Rotate the FFmpeg log file if it exceeds the specified size."""
    try:
        if not FFPMEG_LOG_FILE.exists():
            return

        # Check file size
        file_size_mb = FFPMEG_LOG_FILE.stat().st_size / (1024 * 1024)

        if file_size_mb > max_size_mb:
            # Close all handlers to release the file
            logger = logging.getLogger('ffmpeg')
            for handler in logger.handlers[:]:
                handler.close()
                logger.removeHandler(handler)

            # Rotate logs
            for i in range(backup_count - 1, 0, -1):
                src = FFPMEG_LOG_FILE.with_suffix(f'.log.{i}')
                dst = FFPMEG_LOG_FILE.with_suffix(f'.log.{i+1}')
                if src.exists():
                    if dst.exists():
                        dst.unlink()
                    src.rename(dst)

            # Move current log to .1
            backup = FFPMEG_LOG_FILE.with_suffix('.log.1')
            if backup.exists():
                backup.unlink()
            FFPMEG_LOG_FILE.rename(backup)

            # Reconfigure the logger
            global FFMPEG_LOGGER_CONFIGURED
            FFMPEG_LOGGER_CONFIGURED = False
            setup_ffmpeg_logging()

            # Log the rotation
            ffmpeg_logger = logging.getLogger('ffmpeg')
            ffmpeg_logger.info(
                f"Log file rotated. Previous log saved as {backup}")

    except Exception as e:
        app_logger = logging.getLogger('app')
        app_logger.error(f"Error rotating FFmpeg log: {e}", exc_info=True)


def clear_ffmpeg_log():
    """Clear the FFmpeg log file."""
    try:
        # Close all handlers to release the file
        logger = logging.getLogger('ffmpeg')
        for handler in logger.handlers[:]:
            handler.close()
            logger.removeHandler(handler)

        # Truncate the file
        with open(FFPMEG_LOG_FILE, 'w') as f:
            f.write(
                f"FFmpeg log cleared at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Reconfigure the logger
        global FFMPEG_LOGGER_CONFIGURED
        FFMPEG_LOGGER_CONFIGURED = False
        setup_ffmpeg_logging()

    except Exception as e:
        app_logger = logging.getLogger('app')
        app_logger.error(f"Error clearing FFmpeg log: {e}", exc_info=True)


# Run FFmpeg command with unified logging
def run_ffmpeg_command(command: list, timeout=None, capture_output=True) -> bool:
    """Run FFmpeg command with unified logging.

    Args:
        command: Command list to run
        timeout: Optional timeout in seconds
        capture_output: Whether to capture and log output (True) or send to devnull (False)

    Returns:
        bool: Success status
    """
    ffmpeg_logger = setup_ffmpeg_logging()
    app_logger = get_logger('app')

    command_str = ' '.join(command)
    ffmpeg_logger.info(f"Running FFmpeg command: {command_str}")

    try:
        if capture_output:
            # Capture output for logging but don't display in terminal
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=timeout,
                check=False  # Don't raise exception on non-zero exit
            )

            # Log stdout and stderr to ffmpeg.log if not empty
            if process.stdout and not process.stdout.isspace():
                ffmpeg_logger.debug("STDOUT: " + process.stdout)

            if process.stderr and not process.stderr.isspace():
                ffmpeg_logger.debug("STDERR: " + process.stderr)
        else:
            # Completely suppress output for terminal
            with open(os.devnull, 'w') as devnull:
                process = subprocess.run(
                    command,
                    stdout=devnull,
                    stderr=devnull,
                    timeout=timeout,
                    check=False  # Don't raise exception on non-zero exit
                )

        # Log command result to application log
        if process.returncode == 0:
            app_logger.debug(f"FFmpeg command completed successfully")
        else:
            app_logger.error(
                f"FFmpeg command failed with code {process.returncode}")

        return process.returncode == 0
    except subprocess.TimeoutExpired:
        app_logger.error(f"FFmpeg command timed out after {timeout} seconds")
        return False
    except Exception as e:
        error_msg = f"FFmpeg command failed: {e}"
        ffmpeg_logger.error(error_msg, exc_info=True)
        app_logger.error(error_msg)
        return False


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)


# For backward compatibility
def setup_process_logging():
    """Create and return a configured ProcessLogger (replacement: UnifiedLogger)."""
    return UnifiedLogger('process')
