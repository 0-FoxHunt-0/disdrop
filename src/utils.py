# utils.py

import cv2
from pathlib import Path
from logging_system import logging


def get_video_dimensions(video_path: Path) -> tuple[int, int]:
    """Get dimensions of a video file using opencv."""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            logging.error(f"Failed to open video file: {video_path}")
            return None, None

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return width, height
    except Exception as e:
        logging.error(f"Failed to get video dimensions: {e}")
        return None, None
