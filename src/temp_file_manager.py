import atexit
import logging
import signal
import threading
from pathlib import Path
from typing import Set


class TempFileManager:
    """Thread-safe temporary file manager with enhanced cleanup."""
    _temp_files: Set[Path] = set()
    _lock = threading.Lock()
    _initialized = False

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
            cls._temp_files.add(Path(file_path))
            logging.debug(f"Registered temp file: {file_path}")

    @classmethod
    def unregister(cls, file_path: Path) -> None:
        """Thread-safely unregister a temporary file."""
        with cls._lock:
            cls._temp_files.discard(Path(file_path))
            logging.debug(f"Unregistered temp file: {file_path}")

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
    def _signal_handler(cls, signum: int, frame) -> None:
        """Handle interruption signals."""
        cls.cleanup()
        signal.default_int_handler(signum, frame)

    @classmethod
    def get_temp_count(cls) -> int:
        """Get count of registered temporary files."""
        with cls._lock:
            return len(cls._temp_files)


TempFileManager.initialize()
