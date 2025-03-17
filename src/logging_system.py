import atexit
from functools import wraps
import logging
import os
import subprocess
import sys
import threading
import time
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, TextIO, Union, Dict, Any, Callable
from tqdm import tqdm
import colorlog
import psutil

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


class ModernLogStyle:
    """Modern and unified logging styles"""
    # Reset and basic styles
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    ITALIC = '\033[3m'

    # Modern color palette
    SLATE = '\033[38;5;246m'   # Subtle details
    AZURE = '\033[38;5;39m'    # Processing info
    EMERALD = '\033[38;5;48m'  # Success
    AMBER = '\033[38;5;214m'   # Warnings
    ROSE = '\033[38;5;204m'    # Errors
    VIOLET = '\033[38;5;141m'  # Critical info
    CYAN = '\033[38;5;51m'     # Statistics
    LIME = '\033[38;5;118m'    # Performance metrics

    # Progress indicators
    ARROW = '→'
    BULLET = '•'
    CHECK = '✓'
    CROSS = '✗'
    INFO = 'ℹ'
    WARN = '⚠'
    LIGHTNING = '⚡'
    GEARS = '⚙'
    CLOCK = '🕒'

    # Separator styles
    SEPARATOR_MAIN = f"{DIM}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}"
    SEPARATOR_SUB = f"{DIM}───────────────────────────────{RESET}"


class UnifiedLogFormatter(logging.Formatter):
    """Enhanced formatter combining verbose and normal modes"""

    def __init__(self):
        super().__init__()
        self._style = ModernLogStyle
        self._last_phase = None
        self._seen_messages = set()  # For deduplication
        self._message_count = {}  # Track message occurrence counts
        self._dedup_lock = threading.Lock()  # Thread-safe deduplication
        self._max_similar_messages = 3  # Show at most this many similar messages

    def format(self, record):
        # Extract phase information if available
        current_phase = getattr(record, 'phase', None)

        # Add visual separator between processing phases
        if current_phase and current_phase != self._last_phase:
            self._last_phase = current_phase
            header = self._format_phase_header(current_phase)
            record.msg = f"\n{header}\n{record.msg}"

        # Deduplicate messages properly - thread-safe
        message_key = f"{record.levelno}:{record.pathname}:{record.lineno}:{record.msg}"
        with self._dedup_lock:
            # Update count and check for excessive duplicates
            self._message_count[message_key] = self._message_count.get(
                message_key, 0) + 1
            count = self._message_count[message_key]

            # If we've seen too many similar messages, summarize or skip
            if count > self._max_similar_messages:
                if count == self._max_similar_messages + 1:
                    return f"{self._style.DIM}(Similar message repeated, further occurrences will be suppressed){self._style.RESET}"
                return ""  # Skip this record entirely by returning empty string

            # Normal message processing for non-duplicates or allowed duplicates
            if message_key in self._seen_messages and count <= self._max_similar_messages:
                # Add occurrence count for duplicates
                record.msg = f"{record.msg} {self._style.DIM}(repeat {count}){self._style.RESET}"

            # Add to seen messages
            self._seen_messages.add(message_key)

        # Format based on level
        if record.levelno >= logging.ERROR:
            return self._format_error(record)
        elif record.levelno >= logging.WARNING:
            return self._format_warning(record)
        elif record.levelno == SUCCESS_LEVEL:  # Custom success level
            return self._format_success(record)
        elif record.levelno == PERFORMANCE_LEVEL:  # Performance metrics
            return self._format_performance(record)
        else:
            return self._format_info(record)

    def _format_phase_header(self, phase):
        s = self._style
        return (
            f"{s.SEPARATOR_MAIN}\n"
            f"{s.BOLD}{s.AZURE}{phase}{s.RESET}\n"
            f"{s.SEPARATOR_SUB}"
        )

    def _format_error(self, record):
        s = self._style
        return (
            f"{s.ROSE}{s.CROSS} Error: {record.msg}{s.RESET}"
            f"{self._format_details(record)}"
        )

    def _format_warning(self, record):
        s = self._style
        return (
            f"{s.AMBER}{s.WARN} {record.msg}{s.RESET}"
            f"{self._format_details(record)}"
        )

    def _format_success(self, record):
        s = self._style
        return (
            f"{s.EMERALD}{s.CHECK} {record.msg}{s.RESET}"
            f"{self._format_details(record)}"
        )

    def _format_performance(self, record):
        s = self._style
        return (
            f"{s.LIME}{s.LIGHTNING} {record.msg}{s.RESET}"
            f"{self._format_details(record)}"
        )

    def _format_info(self, record):
        s = self._style

        # Use different icon based on context
        icon = s.BULLET
        if hasattr(record, 'context'):
            context = record.context
            if context == 'progress':
                icon = s.GEARS
            elif context == 'timing':
                icon = s.CLOCK

        prefix = f"{s.AZURE}{icon}{s.RESET}" if hasattr(
            record, 'highlight') else f"{s.DIM}{s.ARROW}{s.RESET}"

        return (
            f"{prefix} {record.msg}"
            f"{self._format_details(record)}"
        )

    def _format_details(self, record):
        if hasattr(record, 'details'):
            s = self._style
            details = record.details
            if isinstance(details, dict):
                detail_lines = [
                    f"\n  {s.DIM}├ {s.SLATE}{k}: {v}{s.RESET}"
                    for k, v in details.items()
                ]
                return ''.join(detail_lines)
            return f"\n  {s.DIM}└ {s.SLATE}{details}{s.RESET}"
        return ""


class UnifiedLogger:
    """Modern unified logger with detailed output by default"""

    def __init__(self, logger_name='app'):
        self.logger = logging.getLogger(logger_name)

        # Only configure if not already configured
        with LOGGER_LOCK:
            if logger_name not in CONFIGURED_LOGGERS:
                self.logger.setLevel(logging.INFO)

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
        extra = {'context': 'timing'}
        self.logger.log(
            PERFORMANCE_LEVEL, f"Completed {phase} in {elapsed:.2f} seconds", extra=extra)

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
        s = ModernLogStyle
        filled = int(width * percentage / 100)

        # Change color based on errors
        bar_color = s.AZURE
        if error_count > 0:
            bar_color = s.AMBER if error_count < 3 else s.ROSE

        # Create the bar with appropriate color
        bar = (
            f"{bar_color}{'━' * filled}{s.DIM}{'─' * (width - filled)}{s.RESET} "
            f"{s.BOLD}{percentage:.1f}%{s.RESET}"
        )

        # Add error indicator if needed
        if error_count > 0:
            bar += f" {s.AMBER}{error_count} errors{s.RESET}"

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
            # Add basic console handler if not already configured
            if not logger.handlers:
                handler = logging.StreamHandler()
                formatter = UnifiedLogFormatter()
                handler.setFormatter(formatter)
                logger.addHandler(handler)
                logger.setLevel(logging.INFO)

                # Mark as configured
                CONFIGURED_LOGGERS.add(name)

                # Add custom log levels for this handler
                if not hasattr(logging.Logger, 'success'):
                    def success(self, message, *args, **kwargs):
                        self.log(SUCCESS_LEVEL, message, *args, **kwargs)
                    logging.Logger.success = success

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
def run_ffmpeg_command(command: list) -> bool:
    """Run FFmpeg command with unified logging."""
    ffmpeg_logger = setup_ffmpeg_logging()
    app_logger = get_logger('app')

    command_str = ' '.join(command)
    ffmpeg_logger.info(f"Running FFmpeg command: {command_str}")

    try:
        # Use stdout and stderr redirection to prevent terminal output
        with open(os.devnull, 'w') as devnull:
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

        # Log stdout and stderr to ffmpeg.log
        if process.stdout and not process.stdout.isspace():
            ffmpeg_logger.debug("STDOUT: " + process.stdout)

        if process.stderr and not process.stderr.isspace():
            ffmpeg_logger.debug("STDERR: " + process.stderr)

        # Log command result to application log
        if process.returncode == 0:
            app_logger.debug(f"FFmpeg command completed successfully")
        else:
            app_logger.error(
                f"FFmpeg command failed with code {process.returncode}")

        return process.returncode == 0
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
