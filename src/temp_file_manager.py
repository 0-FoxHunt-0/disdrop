# temp_file_manager.py
import logging
from pathlib import Path


class TempFileManager:
    """Manages temporary files and ensures cleanup."""
    _temp_files = set()

    @classmethod
    def register(cls, file_path):
        """Register a temporary file for cleanup."""
        cls._temp_files.add(Path(file_path))

    @classmethod
    def unregister(cls, file_path):
        """Unregister a temporary file (if it was moved or already cleaned)."""
        try:
            cls._temp_files.remove(Path(file_path))
        except KeyError:
            pass

    @classmethod
    def cleanup(cls):
        """Clean up all registered temporary files."""
        for file_path in cls._temp_files.copy():
            try:
                if file_path.exists():
                    file_path.unlink()
                    logging.debug(f"Cleaned up temporary file: {file_path}")
                cls._temp_files.remove(file_path)
            except Exception as e:
                logging.error(f"Failed to clean up temporary file {
                              file_path}: {e}")

    @classmethod
    def cleanup_dir(cls, directory):
        """Clean up all files in a directory."""
        try:
            for file_path in Path(directory).glob('*'):
                try:
                    file_path.unlink()
                    logging.debug(f"Cleaned up file in directory: {file_path}")
                except Exception as e:
                    logging.error(f"Failed to clean up file {file_path}: {e}")
        except Exception as e:
            logging.error(f"Failed to clean up directory {directory}: {e}")

    @classmethod
    def get_temp_count(cls):
        """Get count of registered temporary files."""
        return len(cls._temp_files)

    @classmethod
    def list_temp_files(cls):
        """List all registered temporary files."""
        return list(cls._temp_files)
