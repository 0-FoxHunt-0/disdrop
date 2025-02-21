import asyncio
import logging
import os
import re
import signal
import subprocess
import sys
import threading
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import json

logger = logging.getLogger(__name__)


class FFmpegHandler:
    """Singleton handler for FFmpeg operations."""

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

    def create_optimized_gif(self, file_path: Path, output_path: Path,
                             fps: int, dimensions: Tuple[int, int], settings: Dict) -> bool:
        """Create optimized GIF in a single pass."""
        try:
            cmd = [
                'ffmpeg',
                '-i', str(file_path),
                '-vf',
                f'fps={fps},'
                f'scale={dimensions[0]}:{dimensions[1]}:flags=lanczos,'
                'split[s0][s1];'
                '[s0]palettegen=max_colors={colors}:stats_mode=diff[p];'
                '[s1][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle'.format(
                    colors=settings.get('colors', 256)
                ),
                '-y',
                str(output_path)
            ]

            return self.run_command(cmd)

        except Exception as e:
            logging.error(f"GIF creation failed: {str(e)}")
            return False

    def get_dimensions(self, file_path: Path) -> Tuple[Optional[int], Optional[int]]:
        """Get video dimensions using multiple methods."""
        methods = [
            self._get_dimensions_ffprobe,
            self._get_dimensions_ffmpeg
        ]

        last_error = None
        for method in methods:
            try:
                dimensions = method(file_path)
                if dimensions and all(isinstance(d, int) and d > 0 for d in dimensions):
                    return dimensions
            except Exception as e:
                last_error = str(e)
                continue

        logger.error(
            f"All dimension detection methods failed. Last error: {last_error}")
        return None, None

    def _get_dimensions_ffprobe(self, file_path: Path) -> Tuple[int, int]:
        """Get dimensions using ffprobe."""
        try:
            cmd = [
                'ffprobe',
                '-v', 'error',
                '-select_streams', 'v:0',
                '-show_entries', 'stream=width,height,rotation,sample_aspect_ratio',
                '-of', 'json',
                str(file_path)
            ]

            output = subprocess.check_output(
                cmd, stderr=subprocess.PIPE, text=True)
            data = json.loads(output)

            if not data.get('streams'):
                raise ValueError("No video streams found")

            stream = data['streams'][0]
            width = int(stream.get('width', 0))
            height = int(stream.get('height', 0))

            if width <= 0 or height <= 0:
                raise ValueError("Invalid dimensions in stream")

            # Handle rotation
            rotation = int(stream.get('rotation', '0') or '0')
            if rotation in [90, 270]:
                width, height = height, width

            return width, height

        except Exception as e:
            raise ValueError(f"FFprobe dimension detection failed: {e}")

    def _get_dimensions_ffmpeg(self, file_path: Path) -> Tuple[int, int]:
        """Get dimensions using FFmpeg as fallback."""
        try:
            cmd = ['ffmpeg', '-i', str(file_path)]
            output = subprocess.run(cmd, capture_output=True, text=True).stderr

            patterns = [
                r'Stream.*Video.* (\d+)x(\d+)',
                r'Video: .* (\d+)x(\d+)',
                r', (\d+)x(\d+)[,\s]'
            ]

            for pattern in patterns:
                matches = re.findall(pattern, output)
                if matches:
                    width, height = map(int, matches[0])
                    if width > 0 and height > 0:
                        return (width, height)

            raise ValueError("No valid dimensions found in FFmpeg output")
        except Exception as e:
            raise ValueError(f"FFmpeg dimension detection failed: {e}")

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

    async def run_async(self, command: List[str]) -> bool:
        """Run FFmpeg command asynchronously."""
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )

            stdout, stderr = await process.communicate()

            if stderr:
                logger.debug(f"FFmpeg output: {stderr.decode()}")

            return process.returncode == 0

        except Exception as e:
            logger.error(f"Async FFmpeg command failed: {e}")
            return False

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
