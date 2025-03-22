import logging
import threading
import time
from pathlib import Path
from typing import List, Optional
from src.logging_system import get_logger


class BaseProcessor:
    """Base class for processors with unified logging."""

    def __init__(self, logger=None):
        """Initialize with shared logger or create a new one.

        Args:
            logger: Optional logger to use. If None, a new one is created.
        """
        # Use the provided logger or get a configured one with the class name
        self.logger = logger or get_logger(self.__class__.__name__.lower())

        # Thread management
        self._shutdown_initiated = False
        self._shutdown_event = threading.Event()
        self._immediate_termination = threading.Event()
        self._processing_cancelled = threading.Event()

        # Locks and tracking
        self._threads_lock = threading.Lock()
        self._file_locks_lock = threading.Lock()
        self._active_threads = set()
        self._file_locks = {}
        self.logging_lock = threading.Lock()
        self.processed_files = set()
        self._cleanup_handlers = []

        # Log initialization
        self.logger.debug(f"Initialized {self.__class__.__name__}")

    def process_files(self, input_files: List[Path], output_dir: Path) -> List[Path]:
        """Process multiple files with logging."""
        failed_files = []
        total = len(input_files)

        self.logger.info(f"Starting to process {total} files")

        for idx, file_path in enumerate(input_files, 1):
            try:
                self.logger.info(
                    f"Processing {idx}/{total}: {file_path.name} "
                    f"({self.get_file_size(file_path):.2f}MB)"
                )
                # ...existing processing code...
                pass
            except Exception as e:
                self.logger.error(
                    f"Failed to process {file_path.name}: {str(e)}",
                    exc_info=True
                )
                failed_files.append(file_path)

        self.logger.info(
            f"Processing complete. Failed files: {len(failed_files)}")
        return failed_files

    @staticmethod
    def get_file_size(file_path: Path, force_refresh: bool = False) -> float:
        """Get file size in MB."""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except OSError as e:
            # Use local logger for static method
            logger = get_logger('baseprocessor')
            logger.error(
                f"Failed to get file size for {file_path}: {e}", exc_info=True)
            return 0.0

    @staticmethod
    def wait_for_file_completion(file_path: Path, timeout: int = 10) -> bool:
        """Wait for a file to be completely written."""
        try:
            start_time = time.time()
            last_size = -1
            while time.time() - start_time < timeout:
                current_size = file_path.stat().st_size
                if current_size == last_size:
                    return True
                last_size = current_size
                time.sleep(0.5)
            return False
        except OSError:
            return False

    def _should_exit(self) -> bool:
        """Check if processing should be stopped."""
        return self._shutdown_event.is_set() or self._immediate_termination.is_set()

    def _immediate_shutdown_handler(self, signum, frame):
        """Handle immediate shutdown, e.g., from SIGINT or SIGTERM."""
        if self._shutdown_initiated:
            self.logger.warning("Forced shutdown initiated")
            self._immediate_termination.set()
            # Make sure cleanup is called
            self.cleanup_resources()
        else:
            self._shutdown_initiated = True
            self.logger.warning("Graceful shutdown initiated")
            self._shutdown_event.set()
            # Start cleanup in a separate thread that won't be blocked
            threading.Thread(target=self.cleanup_resources,
                             daemon=True).start()

    def cleanup_resources(self):
        """Clean up resources during shutdown."""
        self.logger.debug("Cleaning up resources")
        for handler in self._cleanup_handlers:
            try:
                handler()
            except Exception as e:
                self.logger.error(f"Error during cleanup: {e}", exc_info=True)
