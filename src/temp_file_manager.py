import atexit
import logging
import os
import signal
import threading
import time
from pathlib import Path
from typing import Set, Union


class TempFileManager:
    """Thread-safe temporary file manager with enhanced cleanup."""
    _temp_files: Set[Path] = set()
    _lock = threading.Lock()
    _initialized = False
    MAX_TEMP_FILES = 1000
    MAX_AGE_HOURS = 24
    TEMP_FILE_DIR = "/tmp"

    def __enter__(self):
        self._lock.acquire()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._lock.release()

    @classmethod
    def initialize(cls) -> None:
        """Initialize cleanup handlers once."""
        if not cls._initialized:
            with cls._lock:
                if not cls._initialized:
                    atexit.register(cls.cleanup)
                    signal.signal(signal.SIGINT, cls._signal_handler)
                    signal.signal(signal.SIGTERM, cls._signal_handler)
                    cls._initialized = True

    @classmethod
    def register(cls, file_path: Path) -> None:
        """Thread-safely register a temporary file."""
        with cls._lock:
            # Ensure the file is directly in TEMP_FILE_DIR
            if file_path.parent != Path(cls.TEMP_FILE_DIR):
                new_path = Path(cls.TEMP_FILE_DIR) / file_path.name
                if file_path.exists():
                    file_path.replace(new_path)
                file_path = new_path
            cls._temp_files.add(file_path)
            logging.debug(f"Registered temp file: {file_path}")

    @classmethod
    def unregister(cls, file_path: Path) -> None:
        """Thread-safely unregister a temporary file."""
        with cls._lock:
            cls._temp_files.add(Path(file_path))
            if len(cls._temp_files) > cls.MAX_TEMP_FILES:
                cls._cleanup_oldest()

    @classmethod
    def cleanup(cls) -> None:
        """Clean up all registered temporary files."""
        with cls._lock:
            for file_path in cls._temp_files.copy():
                try:
                    if file_path.exists():
                        file_path.unlink()
                        logging.debug(f"Cleaned up temp file: {file_path}")
                except Exception as e:
                    logging.error(f"Failed to clean up {file_path}: {e}")
            cls._temp_files.clear()

    @classmethod
    def cleanup_dir(cls, directory: Union[str, Path]) -> None:
        """Clean up all files in a directory."""
        try:
            directory = Path(directory)
            if directory.exists():
                for file in directory.iterdir():
                    if file.is_file():
                        try:
                            file.unlink()
                            logging.debug(f"Cleaned up file: {file}")
                        except Exception as e:
                            logging.error(f"Failed to clean up {file}: {e}")
                directory.rmdir()
                logging.debug(f"Removed directory: {directory}")
        except Exception as e:
            logging.error(f"Failed to clean up directory {directory}: {e}")

    @classmethod
    def _signal_handler(cls, signum: int, frame) -> None:
        """Handle interruption signals."""
        cls.cleanup()
        signal.default_int_handler(signum, frame)

    @classmethod
    def get_temp_count(cls) -> int:
        """Get count of registered temporary files."""
        with cls._lock:
            return len(cls._temp_files)

    @classmethod
    def _cleanup_oldest(cls):
        with cls._lock:
            now = time.time()
            cls._temp_files = {
                f for f in cls._temp_files
                if f.exists() and
                (now - f.stat().st_mtime) / 3600 < cls.MAX_AGE_HOURS
            }


TempFileManager.initialize()
