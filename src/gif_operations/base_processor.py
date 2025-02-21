import logging
from pathlib import Path


class BaseProcessor:
    """Base class for all processors with logging and verbose settings."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.dev_logger = logging.getLogger('dev')
        self.user_logger = logging.getLogger('user')

    def get_file_size(self, path: Path) -> float:
        """Get file size in MB."""
        try:
            return path.stat().st_size / (1024 * 1024)
        except Exception as e:
            self.dev_logger.error(f"Failed to get file size: {e}")
            return 0.0
