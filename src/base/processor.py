import logging
import threading
import time
from pathlib import Path


class BaseProcessor:
    """Base class providing common logging functionality and utilities."""

    def __init__(self):
        # Loggers
        self.dev_logger = logging.getLogger('developer')
        self.dev_logger.setLevel(logging.DEBUG)
        self.user_logger = logging.getLogger('user')

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

    @staticmethod
    def get_file_size(file_path: Path, force_refresh: bool = False) -> float:
        """Get file size in MB."""
        try:
            return file_path.stat().st_size / (1024 * 1024)
        except OSError as e:
            logging.error(f"Failed to get file size for {file_path}: {e}")
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
