#!/usr/bin/env python3
import os
import sys
import logging
import platform
import subprocess
import datetime
import threading
import colorama
from pathlib import Path
from typing import Dict, Optional, Union, List, Set, Tuple, Any
from enum import Enum


# Initialize colorama for cross-platform colored terminal output
colorama.init()


class LogLevel(Enum):
    """Custom log levels extending the standard logging levels."""
    DEBUG = logging.DEBUG
    INFO = logging.INFO
    SUCCESS = 25  # Custom level between INFO and WARNING
    WARNING = logging.WARNING
    ERROR = logging.ERROR
    CRITICAL = logging.CRITICAL


class ColorScheme:
    """Color schemes for different log levels."""
    DEBUG = colorama.Fore.CYAN
    INFO = colorama.Fore.WHITE
    SUCCESS = colorama.Fore.GREEN
    WARNING = colorama.Fore.YELLOW
    ERROR = colorama.Fore.RED
    CRITICAL = colorama.Fore.MAGENTA + colorama.Style.BRIGHT
    RESET = colorama.Style.RESET_ALL


class ColoredFormatter(logging.Formatter):
    """Custom formatter to add colors to log messages in the console."""

    def __init__(self, fmt: str = None, datefmt: str = None):
        super().__init__(fmt, datefmt)
        self.colors = {
            logging.DEBUG: ColorScheme.DEBUG,
            logging.INFO: ColorScheme.INFO,
            LogLevel.SUCCESS.value: ColorScheme.SUCCESS,
            logging.WARNING: ColorScheme.WARNING,
            logging.ERROR: ColorScheme.ERROR,
            logging.CRITICAL: ColorScheme.CRITICAL,
        }

    def format(self, record):
        # Save original log level name
        original_levelname = record.levelname
        original_levelno = record.levelno

        # Add color based on log level
        color = self.colors.get(record.levelno, ColorScheme.RESET)

        # Check for custom SUCCESS level
        if record.levelno == LogLevel.SUCCESS.value:
            record.levelname = "SUCCESS"

        record.levelname = f"{color}{record.levelname}{ColorScheme.RESET}"
        result = super().format(record)

        # Restore original level name
        record.levelname = original_levelname
        return result


class LoggingSystem:
    """
    Advanced logging system with support for:
    - Custom log levels
    - Colored console output
    - Multiple log files with different levels
    - System diagnostics capture
    - Thread-safe logging

    This centralized logging system can be used across the entire application
    to maintain consistent logging patterns.
    """

    # Singleton instance
    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        """Implement singleton pattern."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(LoggingSystem, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self, config: Dict = None):
        """
        Initialize the logging system with configuration.

        Args:
            config: Configuration dictionary with logging settings
        """
        # Only initialize once (singleton pattern)
        if self._initialized:
            return

        self.config = config or {}
        self.logs_dir = Path(self.config.get(
            'logging', {}).get('directory', './logs'))
        self.loggers = {}

        # Register custom log levels
        self._register_custom_levels()

        # Ensure logs directory exists
        self._ensure_logs_directory()

        # Set up default loggers
        self._setup_root_logger()

        # Get dxdiag info if on Windows
        if platform.system() == 'Windows':
            self.capture_dxdiag()

        self._initialized = True

    def _register_custom_levels(self):
        """Register custom log levels with the logging system."""
        # Register SUCCESS level
        if not hasattr(logging, 'SUCCESS'):
            logging.SUCCESS = LogLevel.SUCCESS.value
            logging.addLevelName(logging.SUCCESS, 'SUCCESS')

        # Add success method to Logger class
        def success(self, message, *args, **kwargs):
            if self.isEnabledFor(logging.SUCCESS):
                self._log(logging.SUCCESS, message, args, **kwargs)

        # Only add the method if it doesn't exist yet
        if not hasattr(logging.Logger, 'success'):
            logging.Logger.success = success

    def _ensure_logs_directory(self):
        """Ensure the logs directory exists."""
        os.makedirs(self.logs_dir, exist_ok=True)

    def _setup_root_logger(self):
        """Set up the root logger with both file and console handlers."""
        root_logger = logging.getLogger()

        # Clear any existing handlers
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)

        # Set default level from config or use INFO
        log_level_name = self.config.get('logging', {}).get('level', 'INFO')
        log_level = getattr(logging, log_level_name, logging.INFO)
        root_logger.setLevel(log_level)

        # Create handlers
        console_handler = self._create_console_handler()
        file_handler = self._create_file_handler()

        # Add handlers to the root logger
        root_logger.addHandler(console_handler)
        root_logger.addHandler(file_handler)

        # Store in our loggers dict
        self.loggers['root'] = root_logger

    def _create_console_handler(self):
        """Create a console handler with colored output."""
        console_handler = logging.StreamHandler(sys.stdout)

        # Set level, default to INFO or from config
        console_level = self.config.get(
            'logging', {}).get('console_level', 'INFO')
        console_handler.setLevel(getattr(logging, console_level, logging.INFO))

        # Set formatter with colors
        console_format = self.config.get('logging', {}).get(
            'console_format',
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        console_formatter = ColoredFormatter(console_format)
        console_handler.setFormatter(console_formatter)

        return console_handler

    def _create_file_handler(self):
        """Create a file handler for the main log file."""
        # Get log file path from config or use default
        log_file = self.config.get('logging', {}).get(
            'file', str(self.logs_dir / 'processing.log'))

        # Clear the log file if clear_logs setting is True
        if self.config.get('logging', {}).get('clear_logs', True):
            self.clear_log_file(log_file)

        # Create file handler
        file_handler = logging.FileHandler(log_file)

        # Set level, default to DEBUG or from config
        file_level = self.config.get('logging', {}).get('file_level', 'DEBUG')
        file_handler.setLevel(getattr(logging, file_level, logging.DEBUG))

        # Set formatter (no colors in file)
        file_format = self.config.get('logging', {}).get(
            'file_format',
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)

        return file_handler

    def get_logger(self, name: str) -> logging.Logger:
        """
        Get a named logger configured with the current settings.

        Args:
            name: Name of the logger

        Returns:
            logging.Logger: Configured logger
        """
        if name in self.loggers:
            return self.loggers[name]

        logger = logging.getLogger(name)
        self.loggers[name] = logger
        return logger

    def capture_dxdiag(self) -> bool:
        """
        Capture DirectX diagnostic information on Windows systems.

        Returns:
            bool: True if capture succeeded, False otherwise
        """
        if platform.system() != 'Windows':
            return False

        try:
            # Path for dxdiag output
            dxdiag_path = self.logs_dir / 'dxdiag.txt'

            # Run dxdiag and capture output
            subprocess.run(
                ['dxdiag', '/t', str(dxdiag_path)],
                check=True,
                capture_output=True
            )

            # Log success
            root_logger = logging.getLogger()
            root_logger.info(
                f"DirectX diagnostic information saved to {dxdiag_path}")
            return True

        except (subprocess.SubprocessError, OSError) as e:
            # Log error
            root_logger = logging.getLogger()
            root_logger.error(
                f"Failed to capture DirectX diagnostic information: {e}")
            return False

    def clear_log_file(self, log_file: Union[str, Path]) -> None:
        """
        Clear a log file by truncating it.

        Args:
            log_file: Path to the log file to clear
        """
        log_path = Path(log_file)
        if log_path.exists():
            with open(log_path, 'w'):
                pass  # Simply open the file in write mode to truncate it

    def create_rotating_file_handler(
        self,
        name: str,
        log_file: Union[str, Path],
        level: Union[str, int] = 'INFO',
        max_bytes: int = 10485760,  # 10 MB
        backup_count: int = 5
    ) -> logging.Handler:
        """
        Create a rotating file handler for logging.

        Args:
            name: Name for the handler
            log_file: Path to the log file
            level: Log level (name or value)
            max_bytes: Maximum file size before rotation
            backup_count: Number of backup files to keep

        Returns:
            logging.Handler: Configured rotating file handler
        """
        from logging.handlers import RotatingFileHandler

        # Resolve the log level
        if isinstance(level, str):
            log_level = getattr(logging, level, logging.INFO)
        else:
            log_level = level

        # Ensure log directory exists
        log_path = Path(log_file)
        os.makedirs(log_path.parent, exist_ok=True)

        # Create handler
        handler = RotatingFileHandler(
            str(log_path),
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        handler.setLevel(log_level)

        # Set formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )
        handler.setFormatter(formatter)

        return handler

    def add_custom_handler(self, logger_name: str, handler: logging.Handler) -> None:
        """
        Add a custom handler to a logger.

        Args:
            logger_name: Name of the logger to add the handler to
            handler: The handler to add
        """
        logger = self.get_logger(logger_name)
        logger.addHandler(handler)

    def start_new_log_section(self, section_name: str) -> None:
        """
        Add a section divider to the log to help with readability.

        Args:
            section_name: Name of the new section
        """
        root_logger = logging.getLogger()
        divider = "=" * 40
        root_logger.info(f"\n{divider}\n{section_name}\n{divider}")

    def log_system_info(self) -> None:
        """Log basic system information to help with troubleshooting."""
        logger = logging.getLogger('system_info')

        logger.info(f"System: {platform.system()} {platform.release()}")
        logger.info(f"Python Version: {platform.python_version()}")
        logger.info(f"Platform: {platform.platform()}")
        logger.info(f"Processor: {platform.processor()}")

        if platform.system() == 'Windows':
            try:
                import psutil
                memory = psutil.virtual_memory()
                logger.info(
                    f"Memory: Total={memory.total / (1024**3):.2f} GB, Available={memory.available / (1024**3):.2f} GB")

                for disk in psutil.disk_partitions():
                    usage = psutil.disk_usage(disk.mountpoint)
                    logger.info(
                        f"Disk {disk.device}: Total={usage.total / (1024**3):.2f} GB, Free={usage.free / (1024**3):.2f} GB")
            except ImportError:
                logger.info(
                    "psutil not available for detailed memory and disk info")
