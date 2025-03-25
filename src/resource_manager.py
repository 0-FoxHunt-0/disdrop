#!/usr/bin/env python3
import os
import signal
import logging
import atexit
import threading
from pathlib import Path
from typing import List, Dict, Set, Callable, Optional, Any


class ResourceManager:
    """
    Manages system resources and handles interruption signals (CTRL+C).
    Provides functionality to register temporary resources that need to be cleaned up
    when the program exits normally or is interrupted.

    Features:
    - Signal handling for graceful termination (CTRL+C)
    - Registration system for temporary files and directories
    - Cleanup hooks for custom resource cleanup
    - Thread-safe operation
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the ResourceManager.

        Args:
            config: Configuration dictionary for the resource manager
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Temp directories to monitor
        self.temp_dirs: List[Path] = []

        # Set of temporary files to clean up
        self.temp_files: Set[Path] = set()

        # Custom cleanup hooks
        self.cleanup_hooks: List[Callable] = []

        # Flag to track if shutdown is in progress
        self.shutdown_in_progress = False

        # Thread lock for synchronization
        self.lock = threading.RLock()

        # Setup signal handlers
        self._setup_signal_handlers()

        # Register atexit handler for normal program termination
        atexit.register(self.cleanup_resources)

        self.logger.info("ResourceManager initialized")

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for interruption signals."""
        signal.signal(signal.SIGINT, self._signal_handler)  # CTRL+C
        # Termination signal
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Handle Windows specific signals if on Windows
        try:
            # CTRL+BREAK on Windows
            signal.signal(signal.SIGBREAK, self._signal_handler)
        except AttributeError:
            # SIGBREAK not available on non-Windows platforms
            pass

        self.logger.debug("Signal handlers have been set up")

    def _signal_handler(self, sig, frame) -> None:
        """
        Handle interruption signals.

        Args:
            sig: Signal number
            frame: Current stack frame
        """
        signal_names = {
            signal.SIGINT: "SIGINT (CTRL+C)",
            signal.SIGTERM: "SIGTERM"
        }

        # Add Windows-specific signal name if available
        try:
            signal_names[signal.SIGBREAK] = "SIGBREAK (CTRL+BREAK)"
        except AttributeError:
            pass

        signal_name = signal_names.get(sig, f"Signal {sig}")
        self.logger.info(
            f"Received {signal_name}. Initiating graceful shutdown...")

        # Perform cleanup
        self.cleanup_resources()

        # Re-raise the signal with default behavior if needed
        if self.config.get('raise_after_cleanup', True):
            # Reset signal handler to default
            signal.signal(sig, signal.SIG_DFL)
            # Re-raise the signal
            os.kill(os.getpid(), sig)

    def register_temp_dir(self, dir_path: Path) -> None:
        """
        Register a temporary directory to monitor.

        Args:
            dir_path: Path to the temporary directory
        """
        with self.lock:
            if dir_path not in self.temp_dirs:
                self.temp_dirs.append(dir_path)
                self.logger.debug(f"Registered temp directory: {dir_path}")

    def register_temp_file(self, file_path: Path) -> None:
        """
        Register a specific temporary file for cleanup.

        Args:
            file_path: Path to the temporary file
        """
        with self.lock:
            self.temp_files.add(file_path)
            self.logger.debug(f"Registered temp file: {file_path}")

    def register_cleanup_hook(self, hook: Callable) -> None:
        """
        Register a custom cleanup hook function.

        Args:
            hook: Callable function to execute during cleanup
        """
        with self.lock:
            if hook not in self.cleanup_hooks:
                self.cleanup_hooks.append(hook)
                self.logger.debug(f"Registered cleanup hook: {hook.__name__}")

    def cleanup_resources(self) -> None:
        """Clean up all registered resources."""
        with self.lock:
            # Avoid multiple cleanup attempts
            if self.shutdown_in_progress:
                return

            self.shutdown_in_progress = True
            self.logger.info("Initiating resource cleanup...")

            # Execute custom cleanup hooks first
            for hook in self.cleanup_hooks:
                try:
                    hook()
                except Exception as e:
                    self.logger.error(
                        f"Error in cleanup hook {hook.__name__}: {e}")

            # Clean up registered temp files
            for file_path in self.temp_files:
                self._remove_file(file_path)

            # Clean up all files in registered temp directories
            for dir_path in self.temp_dirs:
                self._clean_temp_dir(dir_path)

            self.logger.info("Resource cleanup completed")

    def _clean_temp_dir(self, dir_path: Path) -> None:
        """
        Clean all files in a temporary directory.

        Args:
            dir_path: Path to the temporary directory
        """
        try:
            if not dir_path.exists() or not dir_path.is_dir():
                return

            # Delete all files in directory
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    self._remove_file(file_path)

            self.logger.debug(f"Cleaned temp directory: {dir_path}")
        except Exception as e:
            self.logger.error(f"Error cleaning temp directory {dir_path}: {e}")

    def _remove_file(self, file_path: Path) -> None:
        """
        Remove a specific file.

        Args:
            file_path: Path to the file to remove
        """
        try:
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                self.logger.debug(f"Removed temp file: {file_path}")
        except Exception as e:
            self.logger.error(f"Error removing temp file {file_path}: {e}")

    def get_shutdown_flag(self) -> bool:
        """
        Get the current shutdown flag status.

        Returns:
            bool: True if shutdown is in progress
        """
        with self.lock:
            return self.shutdown_in_progress
