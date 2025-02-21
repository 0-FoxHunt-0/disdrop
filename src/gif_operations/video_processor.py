from pathlib import Path
from typing import Callable, Optional, Tuple
import cv2
import logging


class VideoProcessor:
    def __init__(self):
        self._progress_callback = None
        self.logger = logging.getLogger(__name__)
        self._progress_callback = None

    def set_progress_callback(self, callback: Callable[[float], None]) -> None:
        """Set a callback function to report processing progress."""
        self._progress_callback = callback

    def report_progress(self, progress: float) -> None:
        """Report processing progress through callback."""
        if self._progress_callback:
            try:
                self._progress_callback(min(1.0, max(0.0, progress)))
            except Exception as e:
                self.logger.error(f"Progress callback error: {e}")

    def get_dimensions(self, file_path: Path) -> Tuple[Optional[int], Optional[int]]:
        """Get video dimensions."""
        cap = None
        try:
            cap = cv2.VideoCapture(str(file_path))
            if not cap.isOpened():
                return None, None

            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            if width <= 0 or height <= 0:
                ret, frame = cap.read()
                if ret and frame is not None:
                    height, width = frame.shape[:2]
                else:
                    return None, None

            return width, height
        except Exception as e:
            self.logger.error(f"Failed to get video dimensions: {e}")
            return None, None
        finally:
            if cap is not None:
                cap.release()

    def process_video(self, input_path: Path, output_path: Path) -> bool:
        """Process video with progress reporting."""
        try:
            self.report_progress(0.0)

            # Process video...

            self.report_progress(1.0)
            return True
        except Exception as e:
            self.logger.error(f"Video processing error: {e}")
            return False

    def process_videos(self, input_path: Path, output_path: Path, target_size_mb: float = 15.0) -> bool:
        """Process videos with progress reporting."""
        try:
            self.report_progress(0.0)
            result = self.process_video(input_path, output_path)
            self.report_progress(1.0)
            return result
        except Exception as e:
            self.logger.error(f"Video processing error: {e}")
            return False
