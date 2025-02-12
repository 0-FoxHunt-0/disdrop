import logging
from pathlib import Path
import cv2
import os


def get_video_dimensions(file_path: str) -> tuple:
    """Get video dimensions using multiple methods"""
    try:
        # Try primary method with cv2
        cap = cv2.VideoCapture(file_path)
        if cap.isOpened():
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()

            if width > 0 and height > 0:
                return (width, height)

        # If primary method fails, try reading first frame
        cap = cv2.VideoCapture(file_path)
        ret, frame = cap.read()
        if ret and frame is not None:
            height, width = frame.shape[:2]
            cap.release()
            return (width, height)

        logging.error(f"Could not determine dimensions for {file_path}")
        return None
    except Exception as e:
        logging.error(f"Error getting dimensions for {file_path}: {str(e)}")
        return None


def check_file_integrity(file_path: str) -> bool:
    """Verify file integrity"""
    try:
        if not os.path.exists(file_path):
            return False

        # Check if file is empty
        if os.path.getsize(file_path) == 0:
            return False

        # Try to open and read the video
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            return False

        # Try to read some frames
        frame_count = 0
        while frame_count < 10:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1

        cap.release()
        return frame_count > 0

    except Exception as e:
        logging.error(
            f"Error checking file integrity for {file_path}: {str(e)}")
        return False
