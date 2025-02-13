import logging
import os
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


class FFmpegHandler:
    """Handles FFmpeg operations with proper process management."""

    _instance = None
    _lock = threading.Lock()
    _current_processes = set()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self._process_lock = threading.Lock()
            self._initialized = True

    def run_command(self, command: List[str], timeout: Optional[int] = None) -> bool:
        """Run FFmpeg command with proper handling and logging."""
        current_dir = os.getcwd()
        process = None

        try:
            with self._process_lock:
                process = subprocess.Popen(
                    command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                self._current_processes.add(process)

            stdout, stderr = process.communicate(timeout=timeout)

            if stdout:
                logger.debug(f"FFmpeg output: {stdout}")
            if stderr:
                logger.debug(f"FFmpeg error: {stderr}")

            success = process.returncode == 0
            if not success:
                logger.error(
                    f"FFmpeg command failed with code {process.returncode}")

            return success

        except subprocess.TimeoutExpired:
            logger.error("FFmpeg command timed out")
            self._kill_process(process)
            return False

        except Exception as e:
            logger.error(f"FFmpeg error: {str(e)}")
            if process:
                self._kill_process(process)
            return False

        finally:
            if process:
                with self._process_lock:
                    self._current_processes.discard(process)
            os.chdir(current_dir)

    def _kill_process(self, process: subprocess.Popen) -> None:
        """Kill a specific FFmpeg process."""
        if not process:
            return

        try:
            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception as e:
            logger.error(f"Error killing process {process.pid}: {e}")

    def kill_all_processes(self) -> None:
        """Kill all active FFmpeg processes."""
        with self._process_lock:
            for process in self._current_processes.copy():
                self._kill_process(process)
            self._current_processes.clear()


# Global instance
ffmpeg_handler = FFmpegHandler()


def run_ffmpeg_command(command: List[str], timeout: Optional[int] = None) -> bool:
    """Global function to run FFmpeg commands."""
    return ffmpeg_handler.run_command(command, timeout)
