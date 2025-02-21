import os
import time
import logging
from pathlib import Path
from typing import Union
from cachetools import TTLCache

logger = logging.getLogger(__name__)


class FileProcessor:
    """Base class for file processing operations."""

    def __init__(self):
        self.file_size_cache = TTLCache(maxsize=1000, ttl=3600)

    @staticmethod
    def wait_for_file_completion(file_path: Union[str, Path], timeout: int = 30) -> bool:
        """Wait for file to be completely written and accessible."""
        file_path = Path(file_path)
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                # Try to open the file in read mode
                with open(file_path, 'rb') as f:
                    # Try to seek to end to ensure complete file access
                    f.seek(0, 2)
                # Force sync to filesystem
                if hasattr(os, 'sync'):
                    os.sync()
                # Get initial size
                initial_size = file_path.stat().st_size
                # Wait a small interval
                time.sleep(0.1)
                # Check if size is stable
                if initial_size == file_path.stat().st_size:
                    return True
            except (IOError, OSError):
                time.sleep(0.1)
                continue
        return False

    def get_file_size(self, file_path: Union[str, Path], force_refresh: bool = True) -> float:
        """Get file size in MB with improved reliability."""
        try:
            file_path = Path(file_path)

            if force_refresh:
                # Clear any cached size
                self.file_size_cache.pop(str(file_path), None)

                # Wait for file to be completely written
                if not self.wait_for_file_completion(file_path):
                    logging.warning(
                        f"File may not be completely written: {file_path}")

            # Get fresh size
            size = file_path.stat().st_size / (1024 * 1024)
            # Update cache
            self.file_size_cache[str(file_path)] = size
            return size
        except FileNotFoundError:
            logging.error(f"File not found: {file_path}")
            return float('inf')
        except Exception as e:
            logging.error(f"Error getting file size: {e}")
            return float('inf')

    @staticmethod
    def ensure_directory(directory: Path) -> None:
        """Ensure directory exists."""
        directory.mkdir(parents=True, exist_ok=True)
