import asyncio
import logging
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import json

from src.logging_system import get_logger, performance_monitor, setup_ffmpeg_logging

# Use module-specific logger
logger = get_logger('ffmpeg')


class FFmpegHandler:
    """Handles FFmpeg operations with proper resource management."""

    def __init__(self):
        """Initialize FFmpeg handler with proper logging."""
        self._process_lock = threading.Lock()
        self._current_processes = set()
        self.gpu_support = {
            'nvidia': False,
            'intel_qsv': False,
            'amd_amf': False,
            'vaapi': False,
            'nvenc_available': False,
            'qsv_available': False,
            'amf_available': False,
            'vaapi_available': False
        }
        logger.debug("FFmpegHandler initialized")

    def run_command(self, command: List[str], timeout: int = 300) -> bool:
        """Run FFmpeg command with proper error handling and timeout."""
        if not command:
            logger.error("Empty command passed to run_command")
            return False

        # Get dedicated FFmpeg logger
        ffmpeg_logger = setup_ffmpeg_logging()

        # Log the command to both the application log and FFmpeg log
        command_str = ' '.join(command)
        logger.debug(f"Running FFmpeg command")
        ffmpeg_logger.info(f"Running FFmpeg command: {command_str}")

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            with self._process_lock:
                self._current_processes.add(process)

            try:
                stdout, stderr = process.communicate(timeout=timeout)

                # Log all output to the FFmpeg log file
                if stdout:
                    ffmpeg_logger.debug("STDOUT: " + stdout)

                if stderr:
                    ffmpeg_logger.debug("STDERR: " + stderr)
                    # Keep a brief message in the application log
                    logger.debug(
                        "FFmpeg produced output (see ffmpeg.log for details)")

                # Return success based on return code
                success = process.returncode == 0
                if success:
                    logger.debug("FFmpeg command completed successfully")
                    ffmpeg_logger.info("Command completed successfully")
                else:
                    logger.error(
                        f"FFmpeg command failed with exit code {process.returncode}")
                    ffmpeg_logger.error(
                        f"Command failed with exit code {process.returncode}")

                return success

            except subprocess.TimeoutExpired:
                error_msg = f"FFmpeg command timed out after {timeout}s"
                logger.warning(error_msg)
                ffmpeg_logger.error(error_msg)
                self._kill_process(process)
                return False

            finally:
                with self._process_lock:
                    if process in self._current_processes:
                        self._current_processes.remove(process)

        except Exception as e:
            error_msg = f"Error running FFmpeg command: {e}"
            logger.error(error_msg, exc_info=True)
            ffmpeg_logger.error(error_msg, exc_info=True)
            return False

    async def run_async(self, command: List[str]) -> bool:
        """Run FFmpeg command asynchronously."""
        if not command:
            logger.error("Empty command passed to run_async")
            return False

        # Get dedicated FFmpeg logger
        ffmpeg_logger = setup_ffmpeg_logging()
        
        # Log the command to both the application log and FFmpeg log
        command_str = ' '.join(command)
        logger.debug(f"Running async FFmpeg command")
        ffmpeg_logger.info(f"Running async FFmpeg command: {command_str}")

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                text=True
            )

            stdout, stderr = await process.communicate()

            # Log all output to the FFmpeg log file
            if stdout:
                ffmpeg_logger.debug("STDOUT: " + stdout)
            
            if stderr:
                ffmpeg_logger.debug("STDERR: " + stderr)
                # Keep a brief message in the application log
                logger.debug("FFmpeg produced output (see ffmpeg.log for details)")

            # Return success based on return code
            success = process.returncode == 0
            if success:
                logger.debug("Async FFmpeg command completed successfully")
                ffmpeg_logger.info("Async command completed successfully")
            else:
                logger.error(f"Async FFmpeg command failed with exit code {process.returncode}")
                ffmpeg_logger.error(f"Async command failed with exit code {process.returncode}")
            
            return success

        except Exception as e:
            error_msg = f"Error running async FFmpeg command: {e}"
            logger.error(error_msg, exc_info=True)
            ffmpeg_logger.error(error_msg, exc_info=True)
            return False

    def _kill_process(self, process: subprocess.Popen) -> None:
        """Kill a specific FFmpeg process."""
        if not process:
            return

        try:
            logger.debug(f"Killing FFmpeg process with PID {process.pid}")

            if sys.platform == 'win32':
                subprocess.run(['taskkill', '/F', '/T', '/PID', str(process.pid)],
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            else:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
        except Exception as e:
            logger.error(
                f"Error killing process {process.pid}: {e}", exc_info=True)

    def kill_all_processes(self) -> None:
        """Kill all active FFmpeg processes."""
        with self._process_lock:
            process_count = len(self._current_processes)
            if process_count > 0:
                logger.info(f"Killing {process_count} active FFmpeg processes")

            for process in self._current_processes.copy():
                self._kill_process(process)

            self._current_processes.clear()

    @performance_monitor
    def update_gpu_settings(self, gpu_settings: Dict[str, Any]) -> None:
        """Update GPU settings with externally detected capabilities.

        This allows sharing GPU detection across the application instead
        of each component detecting GPU capabilities separately.
        """
        if not gpu_settings:
            logger.debug("No GPU settings provided to update")
            return

        # Log the update
        logger.debug(f"Updating FFmpegHandler GPU settings: {gpu_settings}")

        # Update GPU support flags if provided
        if 'preferred_encoder' in gpu_settings:
            encoder = gpu_settings['preferred_encoder']
            if encoder == 'nvenc':
                self.gpu_support['nvidia'] = True
                self.gpu_support['intel_qsv'] = False
                self.gpu_support['amd_amf'] = False
                self.gpu_support['vaapi'] = False
                logger.info("Set NVIDIA as preferred encoder")
            elif encoder == 'qsv':
                self.gpu_support['nvidia'] = False
                self.gpu_support['intel_qsv'] = True
                self.gpu_support['amd_amf'] = False
                self.gpu_support['vaapi'] = False
                logger.info("Set Intel QSV as preferred encoder")
            elif encoder == 'amf':
                self.gpu_support['nvidia'] = False
                self.gpu_support['intel_qsv'] = False
                self.gpu_support['amd_amf'] = True
                self.gpu_support['vaapi'] = False
                logger.info("Set AMD AMF as preferred encoder")
            elif encoder == 'vaapi':
                self.gpu_support['nvidia'] = False
                self.gpu_support['intel_qsv'] = False
                self.gpu_support['amd_amf'] = False
                self.gpu_support['vaapi'] = True
                logger.info("Set VAAPI as preferred encoder")

        # Update specific encoder availability flags
        if 'encoders' in gpu_settings:
            encoders = gpu_settings['encoders']
            if isinstance(encoders, list):
                self.gpu_support['nvenc_available'] = 'NVENC' in encoders
                self.gpu_support['qsv_available'] = 'QSV' in encoders
                self.gpu_support['amf_available'] = 'AMF' in encoders
                self.gpu_support['vaapi_available'] = 'VAAPI' in encoders
                logger.debug(f"Updated encoder availability: {encoders}")

        # Log the updated settings
        logger.debug(f"Updated FFmpegHandler GPU settings: {self.gpu_support}")


# Global instance
ffmpeg_handler = FFmpegHandler()
