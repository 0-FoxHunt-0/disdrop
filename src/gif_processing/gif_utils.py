"""
GIF Processing Utilities
Provides temp file management, safe I/O operations, and validation functions
"""

import os
import time
import random
import threading
import shutil
import contextlib
import subprocess
from typing import Dict, Any, Tuple, Optional, Callable
import logging
from PIL import Image, ImageSequence

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def temp_file_context(prefix: str, suffix: str, temp_dir: str, cleanup: bool = True):
    """
    Context manager for temporary files that ensures cleanup even on exceptions.
    
    Args:
        prefix: Prefix for temp file name
        suffix: Suffix for temp file name
        temp_dir: Directory for temp file
        cleanup: Whether to cleanup on exit (default: True)
    
    Yields:
        Path to temporary file (file may not exist yet - caller creates it)
    """
    temp_path = None
    try:
        thread_id = threading.get_ident()
        random_suffix = random.randint(1000, 9999)
        timestamp = int(time.time())
        temp_path = os.path.join(temp_dir, f"{prefix}_{thread_id}_{timestamp}_{random_suffix}{suffix}")
        yield temp_path
    finally:
        # Only cleanup if requested and file exists
        if cleanup and temp_path:
            # Use retry logic for Windows file locking
            max_retries = 3
            for retry in range(max_retries):
                try:
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        break
                except (OSError, PermissionError) as e:
                    if retry < max_retries - 1:
                        time.sleep(0.1 * (retry + 1))  # Progressive backoff
                    else:
                        logger.debug(f"Failed to cleanup temp file {temp_path} after {max_retries} retries: {e}")


@contextlib.contextmanager
def temp_dir_context(prefix: str, temp_dir: str, cleanup: bool = True):
    """
    Context manager for temporary directories that ensures cleanup even on exceptions.
    
    Args:
        prefix: Prefix for temp dir name
        temp_dir: Parent directory for temp dir
        cleanup: Whether to cleanup on exit (default: True)
    
    Yields:
        Path to temporary directory
    """
    temp_path = None
    try:
        timestamp = int(time.time())
        temp_path = os.path.join(temp_dir, f"{prefix}_{timestamp}_{random.randint(1000, 9999)}")
        os.makedirs(temp_path, exist_ok=True)
        yield temp_path
    finally:
        if cleanup and temp_path and os.path.exists(temp_path):
            try:
                shutil.rmtree(temp_path, ignore_errors=True)
            except (OSError, PermissionError) as e:
                logger.debug(f"Failed to cleanup temp dir {temp_path}: {e}")


def safe_file_operation(operation: Callable, *args, max_retries: int = 3, **kwargs):
    """
    Perform file operation with retry logic for Windows file locking.
    
    Args:
        operation: The file operation function to execute
        *args: Positional arguments for the operation
        max_retries: Maximum number of retry attempts (default: 3)
        **kwargs: Keyword arguments for the operation
    
    Returns:
        Result of the operation
    
    Raises:
        OSError, PermissionError: If operation fails after all retries
    """
    for retry in range(max_retries):
        try:
            return operation(*args, **kwargs)
        except (PermissionError, OSError) as e:
            if retry < max_retries - 1:
                logger.debug(f"File operation retry {retry + 1}/{max_retries}: {e}")
                time.sleep(0.1 * (retry + 1))  # Progressive backoff
            else:
                logger.warning(f"File operation failed after {max_retries} retries: {e}")
                raise


def get_gif_info(gif_path: str) -> Dict[str, Any]:
    """
    Extract basic information from a GIF file.
    
    Args:
        gif_path: Path to the GIF file
    
    Returns:
        Dictionary with width, height, fps, frame_count, duration, and file_size_mb
    """
    info: Dict[str, Any] = {
        'width': 320,
        'height': 240,
        'fps': 12,
        'frame_count': 0,
        'duration': 0.0,
        'file_size_mb': 0.0
    }
    
    try:
        # Get file size
        if os.path.exists(gif_path):
            info['file_size_mb'] = os.path.getsize(gif_path) / (1024 * 1024)
        
        # Try ffprobe first for accurate info
        cmd = [
            'ffprobe', '-v', 'quiet', '-print_format', 'json',
            '-show_streams', '-show_format', gif_path
        ]
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            timeout=20
        )
        
        if result.returncode == 0 and result.stdout:
            import json as _json
            data = _json.loads(result.stdout)
            vs = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), None)
            if vs:
                info['width'] = int(vs.get('width', info['width']))
                info['height'] = int(vs.get('height', info['height']))
                fps_str = vs.get('r_frame_rate', '12/1')
                try:
                    if '/' in fps_str:
                        num, den = fps_str.split('/')
                        info['fps'] = max(1.0, float(num) / float(den or 1))
                    else:
                        info['fps'] = max(1.0, float(fps_str))
                except Exception:
                    pass
                try:
                    info['frame_count'] = int(vs.get('nb_frames', 0))
                except Exception:
                    pass
            fmt = data.get('format', {})
            try:
                info['duration'] = float(fmt.get('duration', 0.0))
            except Exception:
                pass
    except Exception:
        # Fallback to PIL if ffprobe fails
        try:
            with Image.open(gif_path) as img:
                info['width'], info['height'] = img.size
                if hasattr(img, 'is_animated') and img.is_animated:
                    try:
                        frame_count = 0
                        for frame in ImageSequence.Iterator(img):
                            frame_count += 1
                        info['frame_count'] = frame_count
                    except Exception:
                        pass
        except Exception as e:
            logger.debug(f"Failed to get GIF info: {e}")
    
    return info


def validate_gif(gif_path: str, max_size_mb: float) -> Tuple[bool, Optional[str]]:
    """
    Validate a GIF file for basic integrity and size constraints.
    
    Args:
        gif_path: Path to the GIF file
        max_size_mb: Maximum allowed file size in MB
    
    Returns:
        Tuple of (is_valid, error_message). error_message is None if valid.
    """
    try:
        if not os.path.exists(gif_path):
            return False, "File does not exist"
        
        size_mb = os.path.getsize(gif_path) / (1024 * 1024)
        if size_mb == 0:
            return False, "File is empty"
        
        if size_mb > max_size_mb:
            return False, f"File too large: {size_mb:.2f}MB > {max_size_mb:.2f}MB"
        
        # Quick animated check
        try:
            with Image.open(gif_path) as img:
                if not getattr(img, 'is_animated', False):
                    return False, "Not animated"
        except Exception as e:
            return False, f"Invalid GIF format: {str(e)}"
        
        return True, None
    except Exception as e:
        return False, str(e)


def create_unique_temp_filename(prefix: str, suffix: str, temp_dir: str) -> str:
    """
    Create a unique temporary filename to avoid Windows file locking conflicts.
    
    Args:
        prefix: Prefix for temp file name
        suffix: Suffix for temp file name
        temp_dir: Directory for temp file
    
    Returns:
        Full path to unique temporary file
    """
    thread_id = threading.get_ident()
    random_suffix = random.randint(1000, 9999)
    timestamp = int(time.time())
    return os.path.join(temp_dir, f"{prefix}_{thread_id}_{timestamp}_{random_suffix}{suffix}")






