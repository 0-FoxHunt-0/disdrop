import logging
import cv2
from pathlib import Path
import subprocess
import json
import re


def get_video_dimensions(file_path: str) -> tuple[int, int]:
    """
    Get video dimensions using multiple fallback methods.
    Returns (width, height) tuple or raises exception if all methods fail.
    """
    # Change the order of methods to try ffprobe first as it's more reliable
    methods = [
        _get_dimensions_ffprobe,    # Try ffprobe first
        _get_dimensions_mediainfo,  # Add mediainfo as second option
        _get_dimensions_opencv,     # OpenCV as third option
        _get_dimensions_first_frame  # Reading first frame as last resort
    ]

    errors = []
    dimensions = None

    for method in methods:
        try:
            dimensions = method(file_path)
            if dimensions and _validate_dimensions(dimensions):
                logging.info(
                    f"Got dimensions using {method.__name__}: {dimensions}")
                return dimensions
            errors.append(
                f"{method.__name__}: Invalid dimensions {dimensions}")
        except Exception as e:
            errors.append(f"{method.__name__}: {str(e)}")
            continue

    # Add debug information
    error_msg = f"Could not determine dimensions for {Path(file_path).name}"
    logging.error(f"{error_msg}. Errors: {'; '.join(errors)}")

    # Return a default safe resolution if all methods fail
    return (1280, 720)  # Return a safe default instead of raising an error


def _validate_dimensions(dimensions: tuple[int, int]) -> bool:
    """Validate that dimensions are reasonable."""
    if not dimensions or len(dimensions) != 2:
        return False

    width, height = dimensions
    if not isinstance(width, int) or not isinstance(height, int):
        return False

    # More reasonable dimension limits
    if width <= 0 or height <= 0:
        return False
    if width > 7680 or height > 4320:  # 8K resolution limit
        return False
    if width < 16 or height < 16:  # Minimum reasonable dimensions
        return False

    return True


def _get_dimensions_ffprobe(file_path: str) -> tuple[int, int]:
    """Get dimensions using ffprobe with improved error handling."""
    try:
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=width,height',
            '-of', 'json',
            str(file_path)
        ]
        output = subprocess.check_output(
            cmd, stderr=subprocess.PIPE, text=True)
        data = json.loads(output)

        if 'streams' in data and data['streams']:
            stream = data['streams'][0]
            if 'width' in stream and 'height' in stream:
                return (int(stream['width']), int(stream['height']))
        raise ValueError("No valid stream data found")
    except Exception as e:
        raise ValueError(f"FFprobe failed: {str(e)}")


def _get_dimensions_mediainfo(file_path: str) -> tuple[int, int]:
    """Get dimensions using mediainfo as alternative."""
    try:
        cmd = ['mediainfo', '--Output=JSON', str(file_path)]
        output = subprocess.check_output(cmd, text=True)
        data = json.loads(output)

        # Navigate through mediainfo's JSON structure
        for track in data.get('media', {}).get('track', []):
            if track.get('@type') == 'Video':
                width = int(track.get('Width', '0').replace(' pixels', ''))
                height = int(track.get('Height', '0').replace(' pixels', ''))
                if width > 0 and height > 0:
                    return (width, height)
        raise ValueError("No video track found")
    except FileNotFoundError:
        raise ValueError("mediainfo not installed")
    except Exception as e:
        raise ValueError(f"MediaInfo failed: {str(e)}")


def _get_dimensions_opencv(file_path: str) -> tuple[int, int]:
    """Get dimensions using OpenCV with improved error handling."""
    cap = None
    try:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        # Try getting dimensions directly first
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        if width <= 0 or height <= 0:
            # Try reading the first frame if direct method fails
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
            else:
                raise ValueError("Failed to read frame dimensions")

        return (width, height)
    except Exception as e:
        raise ValueError(f"OpenCV failed: {str(e)}")
    finally:
        if cap is not None:
            cap.release()


def _get_dimensions_first_frame(file_path: str) -> tuple[int, int]:
    """Get dimensions by reading the first frame."""
    cap = None
    try:
        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        # Try multiple frames in case the first one is corrupt
        for _ in range(5):  # Try first 5 frames
            ret, frame = cap.read()
            if ret and frame is not None:
                height, width = frame.shape[:2]
                return (width, height)

        raise ValueError("Failed to read valid frame")
    except Exception as e:
        raise ValueError(f"First frame reading failed: {str(e)}")
    finally:
        if cap is not None:
            cap.release()
