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
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=30
            )
            
            if result.returncode != 0:
                return False, f"FFprobe validation failed: {result.stderr.strip()}"
            
            # Parse JSON output
            import json
            try:
                stdout_text = result.stdout or ""
                if not stdout_text.strip():
                    return False, "Invalid FFprobe output"
                data = json.loads(stdout_text)
            except (json.JSONDecodeError, TypeError):
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
        
        # Skip size validation if max_size_mb is None
        if max_size_mb is not None and file_size_mb > max_size_mb:
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
    def is_valid_gif_fast(gif_path: str, max_size_mb: float = 10.0) -> Tuple[bool, Optional[str]]:
        """
        Fast validation for GIFs avoiding full frame iteration.
        Checks: existence, size (if provided), basic GIF header/trailer, PIL open, animated flag,
        frame count via n_frames, and ability to seek a few key frames.
        """
        try:
            if not os.path.exists(gif_path):
                return False, "File does not exist"
            file_size_mb = os.path.getsize(gif_path) / (1024 * 1024)
            if file_size_mb == 0:
                return False, "File is empty"
            if max_size_mb is not None and file_size_mb > max_size_mb:
                return False, f"File too large: {file_size_mb:.2f}MB > {max_size_mb}MB"

            # Quick header/trailer check
            try:
                with open(gif_path, 'rb') as f:
                    header = f.read(6)
                    if header not in [b'GIF87a', b'GIF89a']:
                        return False, "Invalid GIF header"
                    f.seek(-1, 2)
                    if f.read(1) != b';':
                        return False, "Missing GIF trailer"
            except Exception as e:
                return False, f"Header/trailer check failed: {e}"

            # PIL-based light checks
            with Image.open(gif_path) as img:
                if img.format != 'GIF':
                    return False, "File is not a valid GIF format"
                if not getattr(img, 'is_animated', False):
                    return False, "File is not an animated GIF"

                frame_count = getattr(img, 'n_frames', None)
                if frame_count is None:
                    # Fallback: try seeking a couple frames without full iteration
                    try:
                        img.seek(0)
                    except Exception:
                        return False, "Cannot read first frame"
                    try:
                        img.seek(1)
                        frame_count = 2
                    except EOFError:
                        frame_count = 1
                if frame_count <= 0:
                    return False, "GIF has no frames"

                # Probe a few positions to ensure seeking/decoding works
                probes = {0, 1, max(0, (frame_count // 2)), max(0, frame_count - 1)}
                for pos in probes:
                    try:
                        img.seek(pos)
                        # Convert quickly to ensure decodability; avoid heavy getdata()
                        frame = img.convert('P') if img.mode != 'P' else img
                        w, h = frame.size
                        if w <= 0 or h <= 0:
                            return False, f"Invalid frame dimensions at {pos}"
                    except EOFError:
                        # If EOF on mid/last, still consider valid as long as first frames OK
                        if pos in (0, 1):
                            return False, f"Unexpected EOF at frame {pos}"
                    except Exception as e:
                        return False, f"Cannot decode frame {pos}: {e}"

            return True, None

        except Exception as e:
            return False, f"Fast GIF validation error: {str(e)}"
    
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
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=15
            )
            
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
    
    @staticmethod
    def is_valid_video_with_enhanced_checks(video_path: str, original_path: str = None, max_size_mb: float = None) -> Tuple[bool, Optional[str]]:
        """
        Enhanced video validation with sample reading and length comparison
        
        Args:
            video_path: Path to the video file to validate
            original_path: Path to the original file for length comparison
            max_size_mb: Maximum file size in MB
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic validation first
        is_valid, error_msg = FileValidator.is_valid_video(video_path)
        if not is_valid:
            return False, error_msg
        
        # File integrity check
        integrity_valid, integrity_error = FileValidator._validate_file_integrity(video_path)
        if not integrity_valid:
            return False, f"File integrity check failed: {integrity_error}"
        
        # Size check if specified
        if max_size_mb is not None:
            if not FileValidator.is_video_under_size(video_path, max_size_mb):
                file_size_mb = FileValidator.get_file_size_mb(video_path)
                return False, f"File too large: {file_size_mb:.2f}MB > {max_size_mb}MB"
        
        # Enhanced validation: Sample read test
        sample_read_valid, sample_error = FileValidator._validate_sample_read(video_path)
        if not sample_read_valid:
            return False, f"Sample read validation failed: {sample_error}"
        
        # Enhanced validation: Length comparison with original
        if original_path and os.path.exists(original_path):
            length_valid, length_error = FileValidator._validate_length_comparison(video_path, original_path)
            if not length_valid:
                return False, f"Length validation failed: {length_error}"
        
        return True, None
    
    @staticmethod
    def is_valid_gif_with_enhanced_checks(gif_path: str, original_path: str = None, max_size_mb: float = None) -> Tuple[bool, Optional[str]]:
        """
        Enhanced GIF validation with sample reading and length comparison
        
        Args:
            gif_path: Path to the GIF file to validate
            original_path: Path to the original file for length comparison
            max_size_mb: Maximum file size in MB
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        # Basic validation first
        is_valid, error_msg = FileValidator.is_valid_gif(gif_path, max_size_mb)
        if not is_valid:
            return False, error_msg
        
        # File integrity check
        integrity_valid, integrity_error = FileValidator._validate_file_integrity(gif_path)
        if not integrity_valid:
            return False, f"File integrity check failed: {integrity_error}"
        
        # Enhanced validation: Sample read test
        sample_read_valid, sample_error = FileValidator._validate_gif_sample_read(gif_path)
        if not sample_read_valid:
            return False, f"Sample read validation failed: {sample_error}"
        
        # Enhanced validation: Length comparison with original
        if original_path and os.path.exists(original_path):
            length_valid, length_error = FileValidator._validate_length_comparison(gif_path, original_path)
            if not length_valid:
                return False, f"Length validation failed: {length_error}"
        
        return True, None
    
    @staticmethod
    def _validate_sample_read(video_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that video file can be read and is not corrupted by sampling frames
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                cap.release()
                return False, "Cannot open video file"
            
            # Get video properties
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            duration = total_frames / fps if fps > 0 else 0
            
            if total_frames <= 0:
                cap.release()
                return False, "Video has no frames"
            
            # Sample read frames at different positions to check for corruption
            sample_positions = [0, total_frames // 4, total_frames // 2, total_frames // 4 * 3, total_frames - 1]
            sample_positions = [max(0, min(pos, total_frames - 1)) for pos in sample_positions]
            
            for frame_pos in sample_positions:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
                ret, frame = cap.read()
                
                if not ret or frame is None:
                    cap.release()
                    return False, f"Cannot read frame at position {frame_pos}/{total_frames}"
                
                # Basic frame validation
                if frame.size == 0 or frame.shape[0] == 0 or frame.shape[1] == 0:
                    cap.release()
                    return False, f"Invalid frame at position {frame_pos}"
            
            cap.release()
            return True, None
            
        except Exception as e:
            return False, f"Sample read error: {str(e)}"
    
    @staticmethod
    def _validate_gif_sample_read(gif_path: str) -> Tuple[bool, Optional[str]]:
        """
        Enhanced validation that GIF file can be read and is not corrupted by sampling frames
        
        Args:
            gif_path: Path to the GIF file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            with Image.open(gif_path) as img:
                if not img.is_animated:
                    return False, "File is not an animated GIF"
                
                # Get frame count
                frame_count = 0
                frame_durations = []
                try:
                    while True:
                        img.seek(frame_count)
                        # Check frame duration
                        duration = img.info.get('duration', 0)
                        frame_durations.append(duration)
                        frame_count += 1
                except EOFError:
                    pass
                
                if frame_count == 0:
                    return False, "GIF has no frames"
                
                # Check for reasonable frame durations (10ms to 5000ms)
                if frame_durations:
                    avg_duration = sum(frame_durations) / len(frame_durations)
                    if avg_duration < 10 or avg_duration > 5000:
                        return False, f"Suspicious frame duration: {avg_duration:.1f}ms (expected 10-5000ms)"
                
                # Sample read frames at different positions (more comprehensive)
                sample_positions = [0, frame_count // 8, frame_count // 4, frame_count // 2, 
                                  frame_count // 4 * 3, frame_count // 8 * 7, frame_count - 1]
                sample_positions = [max(0, min(pos, frame_count - 1)) for pos in sample_positions]
                
                for frame_pos in sample_positions:
                    try:
                        img.seek(frame_pos)
                        frame = img.convert('RGB')  # Convert to ensure it's readable
                        
                        # Basic frame validation
                        if frame.size[0] == 0 or frame.size[1] == 0:
                            return False, f"Invalid frame at position {frame_pos}/{frame_count}"
                        
                        # Check for reasonable frame dimensions
                        width, height = frame.size
                        if width < 10 or height < 10 or width > 2000 or height > 2000:
                            return False, f"Suspicious frame dimensions: {width}x{height}"
                        
                        # Sample pixel data to detect corruption
                        try:
                            pixel_data = list(frame.getdata())
                            if len(pixel_data) == 0:
                                return False, f"Empty pixel data at frame {frame_pos}"
                            
                            # Check for all-black or all-white frames (potential corruption)
                            if frame_pos > 0:  # Skip first frame check
                                unique_colors = set(pixel_data[:100])  # Sample first 100 pixels
                                if len(unique_colors) == 1:
                                    color = list(unique_colors)[0]
                                    if color == (0, 0, 0) or color == (255, 255, 255):
                                        # Allow some all-black/white frames but not too many
                                        pass  # Could add counter here if needed
                        except Exception as pixel_e:
                            return False, f"Cannot read pixel data at frame {frame_pos}: {str(pixel_e)}"
                        
                    except Exception as e:
                        return False, f"Cannot read frame at position {frame_pos}/{frame_count}: {str(e)}"
                
                # Additional validation: Check if GIF loops properly
                try:
                    img.seek(0)  # Go back to first frame
                    first_frame = img.convert('RGB')
                    if first_frame.size[0] == 0 or first_frame.size[1] == 0:
                        return False, "First frame is invalid after loop check"
                except Exception as e:
                    return False, f"GIF loop validation failed: {str(e)}"
                
                return True, None
                
        except Exception as e:
            return False, f"GIF sample read error: {str(e)}"
    
    @staticmethod
    def _validate_length_comparison(processed_path: str, original_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate that the processed file has reasonable length compared to original
        
        Args:
            processed_path: Path to the processed file
            original_path: Path to the original file
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Get durations using ffprobe
            original_duration = FileValidator._get_video_duration(original_path)
            processed_duration = FileValidator._get_video_duration(processed_path)
            
            if original_duration <= 0:
                return True, None  # Skip validation if original duration is invalid
            
            if processed_duration <= 0:
                return False, "Processed file has no valid duration"
            
            # Calculate duration ratio
            duration_ratio = processed_duration / original_duration
            
            # Allow some tolerance for processing (e.g., frame rate changes, trimming)
            # For GIFs, be more lenient due to frame-based timing vs continuous video timing
            # Check if the processed file is a GIF to apply more lenient validation
            is_gif = processed_path.lower().endswith('.gif')
            
            if is_gif:
                # For GIFs, be extremely lenient due to frame-based timing and compression
                # GIFs can be significantly shorter due to frame rate reduction, frame dropping, and segmentation
                # Since GIFs are inherently compressed and may represent segments of the original video,
                # we should be very permissive with duration validation
                min_ratio = 0.30  # Allow up to 70% shorter for GIFs (very permissive)
                max_ratio = 1.50  # Allow up to 50% longer for GIFs
            else:
                # For videos, use standard validation
                min_ratio = 0.80  # Allow up to 20% shorter for videos
                max_ratio = 1.25  # Allow up to 25% longer for videos
            
            if duration_ratio < min_ratio:
                logger.warning(f"Duration validation failed - too short: {processed_duration:.2f}s vs {original_duration:.2f}s ({duration_ratio:.1%})")
                return False, f"Processed file too short: {processed_duration:.2f}s vs {original_duration:.2f}s ({duration_ratio:.1%})"
            
            # Processed file should not be significantly longer
            if duration_ratio > max_ratio:
                logger.warning(f"Duration validation failed - too long: {processed_duration:.2f}s vs {original_duration:.2f}s ({duration_ratio:.1%})")
                return False, f"Processed file too long: {processed_duration:.2f}s vs {original_duration:.2f}s ({duration_ratio:.1%})"
            
            return True, None
            
        except Exception as e:
            return False, f"Length comparison error: {str(e)}"
    
    @staticmethod
    def _get_video_duration(video_path: str) -> float:
        """
        Get video duration using ffprobe
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Duration in seconds, or 0 if error
        """
        try:
            # For GIFs, use PIL to get accurate duration
            if video_path.lower().endswith('.gif'):
                try:
                    from PIL import Image
                    with Image.open(video_path) as img:
                        if hasattr(img, 'n_frames') and img.n_frames > 0:
                            # Calculate duration from frame count and frame duration
                            frame_duration_ms = img.info.get('duration', 100)  # Default 100ms
                            total_duration_ms = img.n_frames * frame_duration_ms
                            return total_duration_ms / 1000.0  # Convert to seconds
                        else:
                            # Fallback to ffprobe for non-animated GIFs
                            pass
                except Exception as e:
                    logger.debug(f"PIL duration calculation failed for {video_path}: {e}")
                    # Fall back to ffprobe
            
            # Use ffprobe for videos and as fallback for GIFs
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', video_path
            ]
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=15
            )
            
            if result.returncode == 0:
                import json
                stdout_text = result.stdout or ""
                if not stdout_text.strip():
                    return 0.0
                try:
                    data = json.loads(stdout_text)
                except Exception:
                    return 0.0
                duration = float(data.get('format', {}).get('duration', 0))
                return duration
            
        except Exception as e:
            logger.warning(f"Could not get duration for {video_path}: {e}")
        
        return 0.0 
    
    @staticmethod
    def _validate_file_integrity(file_path: str) -> Tuple[bool, Optional[str]]:
        """
        Validate file integrity by checking for truncation and corruption
        
        Args:
            file_path: Path to the file to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            if not os.path.exists(file_path):
                return False, "File does not exist"
            
            file_size = os.path.getsize(file_path)
            if file_size == 0:
                return False, "File is empty"
            
            # For video files, check if the file appears to be truncated
            # by looking at the end of the file for proper container structure
            if file_path.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
                return FileValidator._validate_video_file_integrity(file_path, file_size)
            
            # For GIF files, check if the file appears to be truncated
            elif file_path.lower().endswith('.gif'):
                return FileValidator._validate_gif_file_integrity(file_path, file_size)
            
            return True, None
            
        except Exception as e:
            return False, f"File integrity check error: {str(e)}"
    
    @staticmethod
    def _validate_video_file_integrity(file_path: str, file_size: int) -> Tuple[bool, Optional[str]]:
        """
        Validate video file integrity by checking for proper container structure
        
        Args:
            file_path: Path to the video file
            file_size: Size of the file in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Use ffprobe to check if the file can be properly parsed
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', file_path
            ]
            
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=15
            )
            
            if result.returncode != 0:
                return False, f"FFprobe cannot parse file: {result.stderr.strip()}"
            
            # Parse JSON output
            import json
            stdout_text = result.stdout or ""
            if not stdout_text.strip():
                return False, "Invalid FFprobe output"
            try:
                data = json.loads(stdout_text)
            except Exception:
                return False, "Invalid FFprobe output"
            
            # Check if format information is complete
            format_info = data.get('format', {})
            if not format_info:
                return False, "No format information found"
            
            # Check if file size matches what ffprobe reports
            reported_size = format_info.get('size')
            if reported_size:
                try:
                    reported_size = int(reported_size)
                    size_diff = abs(file_size - reported_size)
                    size_diff_percent = (size_diff / file_size) * 100
                    
                    # Allow 1% difference due to metadata variations
                    if size_diff_percent > 1.0:
                        return False, f"File size mismatch: {file_size} vs {reported_size} bytes ({size_diff_percent:.1f}% difference)"
                except (ValueError, TypeError):
                    pass  # Skip size comparison if reported size is invalid
            
            # Check if there are video streams
            streams = data.get('streams', [])
            video_streams = [s for s in streams if s.get('codec_type') == 'video']
            
            if not video_streams:
                return False, "No video streams found"
            
            return True, None
            
        except Exception as e:
            return False, f"Video integrity check error: {str(e)}"
    
    @staticmethod
    def _validate_gif_file_integrity(file_path: str, file_size: int) -> Tuple[bool, Optional[str]]:
        """
        Validate GIF file integrity by checking for proper GIF structure
        
        Args:
            file_path: Path to the GIF file
            file_size: Size of the file in bytes
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        try:
            # Check for proper GIF header and trailer
            with open(file_path, 'rb') as f:
                # Check GIF header (GIF87a or GIF89a)
                header = f.read(6)
                if header not in [b'GIF87a', b'GIF89a']:
                    return False, "Invalid GIF header"
                
                # Check for GIF trailer (0x3B) at the end
                f.seek(-1, 2)  # Go to last byte
                trailer = f.read(1)
                if trailer != b';':
                    return False, "Missing GIF trailer"
            
            # Additional validation with PIL
            with Image.open(file_path) as img:
                if not img.format == 'GIF':
                    return False, "File is not a valid GIF format"
                
                # Try to get frame count
                try:
                    frame_count = 0
                    while True:
                        img.seek(frame_count)
                        frame_count += 1
                except EOFError:
                    pass
                
                if frame_count == 0:
                    return False, "GIF has no frames"
            
            return True, None
            
        except Exception as e:
            return False, f"GIF integrity check error: {str(e)}" 