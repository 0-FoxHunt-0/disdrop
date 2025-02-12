import logging
from pathlib import Path
import cv2


class VideoProcessingError(Exception):
    def __init__(self, message, file_path, details=None):
        self.file_path = file_path
        self.details = details
        super().__init__(f"{message} - File: {file_path} - Details: {details}")


def validate_video_file(file_path: str) -> bool:
    cap = None
    try:
        if not Path(file_path).exists():
            raise VideoProcessingError("File does not exist", file_path)

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise VideoProcessingError("Failed to open video file", file_path)

        # Check video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        if fps <= 0 or frame_count <= 0:
            raise VideoProcessingError("Invalid video properties", file_path,
                                       f"FPS: {fps}, Frames: {frame_count}")

        # Read sample frames
        frames_to_check = min(10, frame_count)
        for _ in range(frames_to_check):
            ret, frame = cap.read()
            if not ret or frame is None:
                raise VideoProcessingError("Failed to read frames", file_path)

        return True

    except Exception as e:
        logging.error(f"Validation error for {file_path}: {str(e)}")
        return False
    finally:
        if cap is not None:
            cap.release()


def retry_video_processing(func):
    def wrapper(*args, **kwargs):
        max_retries = 3
        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(
                        f"Failed after {max_retries} attempts: {str(e)}")
                    raise
                logging.warning(f"Attempt {attempt + 1} failed, retrying...")
        return None
    return wrapper
