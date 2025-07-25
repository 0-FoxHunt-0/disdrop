"""
File Validation Module
Validates video and GIF files for corruption and size constraints
"""

import os
import subprocess
import logging
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import cv2
from PIL import Image

logger = logging.getLogger(__name__)

class FileValidator:
    """Validates video and GIF files for integrity and size constraints"""
    
    @staticmethod
    def is_valid_video(video_path: str) -> Tuple[bool, Optional[str]]:
        """
        Check if a video file is valid and not corrupted
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(video_path):
            return False, "File does not exist"
        
        if os.path.getsize(video_path) == 0:
            return False, "File is empty"
        
        try:
            # Method 1: Use ffprobe to check file integrity and get basic info
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                return False, f"FFprobe validation failed: {result.stderr.strip()}"
            
            # Parse JSON output
            import json
            try:
                data = json.loads(result.stdout)
            except json.JSONDecodeError:
                return False, "Invalid FFprobe output"
            
            # Check if there's at least one video stream
            video_streams = [s for s in data.get('streams', []) if s.get('codec_type') == 'video']
            if not video_streams:
                return False, "No video streams found"
            
            # Check if the video stream has frames
            video_stream = video_streams[0]
            nb_frames = video_stream.get('nb_frames')
            duration = float(data.get('format', {}).get('duration', 0))
            
            # If nb_frames is not available, check duration
            if nb_frames is None or nb_frames == 'N/A':
                if duration <= 0:
                    return False, "Video has no duration or frame count"
            else:
                try:
                    frame_count = int(nb_frames)
                    if frame_count <= 0:
                        return False, "Video has no frames"
                except (ValueError, TypeError):
                    # nb_frames might be 'N/A' or invalid, check duration instead
                    if duration <= 0:
                        return False, "Video has no valid duration or frame count"
            
            # Method 2: Try to open with OpenCV as additional validation
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                return False, "OpenCV cannot open video file"
            
            # Try to read first frame
            ret, frame = cap.read()
            cap.release()
            
            if not ret or frame is None:
                return False, "Cannot read video frames"
            
            return True, None
            
        except subprocess.TimeoutExpired:
            return False, "Validation timeout - file may be corrupted"
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def is_valid_gif(gif_path: str, max_size_mb: float = 10.0) -> Tuple[bool, Optional[str]]:
        """
        Check if a GIF file is valid and under size limit
        
        Args:
            gif_path: Path to the GIF file
            max_size_mb: Maximum file size in MB
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not os.path.exists(gif_path):
            return False, "File does not exist"
        
        file_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
        
        if file_size_mb == 0:
            return False, "File is empty"
        
        if file_size_mb > max_size_mb:
            return False, f"File too large: {file_size_mb:.2f}MB > {max_size_mb}MB"
        
        try:
            # Try to open and validate the GIF with PIL
            with Image.open(gif_path) as img:
                if not img.is_animated:
                    return False, "File is not an animated GIF"
                
                # Try to iterate through frames to check for corruption
                frame_count = 0
                try:
                    while True:
                        img.seek(frame_count)
                        frame_count += 1
                except EOFError:
                    # End of frames - this is expected
                    pass
                
                if frame_count == 0:
                    return False, "GIF has no frames"
                
                return True, None
                
        except Exception as e:
            return False, f"GIF validation error: {str(e)}"
    
    @staticmethod
    def is_video_under_size(video_path: str, max_size_mb: float) -> bool:
        """Check if video file is under the specified size limit"""
        if not os.path.exists(video_path):
            return False
        
        file_size_mb = os.path.getsize(video_path) / (1024 * 1024)
        return file_size_mb <= max_size_mb
    
    @staticmethod
    def get_file_size_mb(file_path: str) -> float:
        """Get file size in MB"""
        if not os.path.exists(file_path):
            return 0.0
        return os.path.getsize(file_path) / (1024 * 1024)
    
    @staticmethod
    def is_mp4_format(video_path: str) -> bool:
        """Check if video is in MP4 format"""
        if not os.path.exists(video_path):
            return False
        
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
                '-show_entries', 'stream=codec_name', '-of', 'csv=p=0',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
            
            if result.returncode == 0:
                codec = result.stdout.strip().lower()
                # Check for common MP4 video codecs
                return codec in ['h264', 'h.264', 'avc', 'hevc', 'h265', 'h.265']
            
        except Exception as e:
            logger.warning(f"Could not determine video format for {video_path}: {e}")
        
        # Fallback: check file extension
        return video_path.lower().endswith('.mp4')
    
    @staticmethod
    def get_supported_video_extensions() -> set:
        """Get set of supported video file extensions"""
        return {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp'}
    
    @staticmethod
    def is_supported_video_format(file_path: str) -> bool:
        """Check if file has a supported video extension"""
        ext = Path(file_path).suffix.lower()
        return ext in FileValidator.get_supported_video_extensions() 