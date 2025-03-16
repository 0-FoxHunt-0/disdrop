import logging
from pathlib import Path


class BaseProcessor:
    """Base class for all processors with logging and verbose settings."""

    def __init__(self, logger=None, verbose: bool = False):
        self.verbose = verbose
        self.logger = logger or logging.getLogger('app')

    def get_file_size(self, path: Path) -> float:
        """Get file size in MB."""
        try:
            return path.stat().st_size / (1024 * 1024)
        except Exception as e:
            self.dev_logger.error(f"Failed to get file size: {e}")
            return 0.0
