"""
GIF Generation Module
Converts videos to optimized GIFs for social media platforms
"""

import os
import subprocess
import tempfile
import shutil
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
from PIL import Image, ImageSequence
import cv2
from tqdm import tqdm
import time # Added for time.sleep

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

class GifGenerator:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.temp_dir = self.config.get_temp_dir()
        
        # Shutdown handling
        self.shutdown_requested = False
        self.current_ffmpeg_process = None
        
    def create_gif(self, input_video: str, output_path: str, platform: str = None,
                  max_size_mb: int = None, start_time: float = 0, 
                  duration: float = None) -> Dict[str, Any]:
        """
        Create optimized GIF from video with iterative quality optimization
        
        Args:
            input_video: Path to input video file
            output_path: Path for output GIF
            platform: Target platform (twitter, discord, slack, etc.)
            max_size_mb: Maximum file size in MB
            start_time: Start time in seconds (default: 0)
            duration: Duration in seconds (default: use platform/config limit)
            
        Returns:
            Dictionary with generation results and metadata
        """
        
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Input video file not found: {input_video}")
        
        logger.info(f"Starting GIF generation with quality optimization: {input_video} -> {output_path}")
        
        # Get platform configuration
        platform_config = {}
        if platform:
            platform_config = self.config.get_platform_config(platform, 'gif_settings')
            logger.info(f"Using platform configuration for: {platform}")
        
        # Get video information
        video_info = self._get_video_info(input_video)
        
        # Calculate initial parameters with aspect ratio preservation and content analysis
        gif_params = self._calculate_gif_params(platform_config, max_size_mb, duration, video_info, input_video, start_time)
        
        # Adjust duration if needed
        max_duration = gif_params['max_duration']
        if duration is not None:
            # If duration is explicitly provided, use it (don't limit by max_duration)
            actual_duration = min(duration, video_info['duration'] - start_time)
        else:
            # If no duration provided, use the config max_duration limit
            actual_duration = min(max_duration, video_info['duration'] - start_time)
        
        logger.info(f"Creating GIF: {gif_params['width']}x{gif_params['height']}, "
                   f"{actual_duration:.1f}s @ {gif_params['fps']}fps")
        
        # Check if video should be split into segments instead of creating one compressed GIF
        if self._should_split_video(actual_duration, video_info, gif_params['max_size_mb']):
            logger.info("Video will be split into high-quality segments instead of creating one compressed GIF")
            print("🎬 Video will be split into high-quality segments instead of creating one compressed GIF")
            results = self._split_video_into_segments(
                input_video, output_path, start_time, actual_duration, video_info
            )
            
            # If segmentation failed, clean up any temp files
            if not results.get('success', False):
                temp_files = results.get('temp_files_to_cleanup', [])
                temp_folder = results.get('temp_segments_folder')
                if temp_files:
                    self._cleanup_temp_files(temp_files)
                if temp_folder:
                    self._cleanup_temp_folder(temp_folder)
        else:
            # Use the standard quality optimization method for shorter/simpler videos
            results = self._create_gif_with_quality_optimization(
                input_video, output_path, gif_params, start_time, actual_duration, video_info
            )
        
        logger.info(f"GIF generation completed: {results.get('file_size_mb', 0):.2f}MB, {results.get('frame_count', 0)} frames")
        
        # Return results with consistent format
        if results.get('method') == 'Video Segmentation':
            # For segmented videos, provide detailed information
            logger.info(f"Video segmentation completed: {results.get('segments_created', 0)} segments created "
                       f"(Total: {results.get('total_size_mb', 0):.2f}MB)")
            print(f"🎬 Video segmentation completed: {results.get('segments_created', 0)} segments created "
                  f"(Total: {results.get('total_size_mb', 0):.2f}MB)")
            
            return {
                'success': results.get('success', False),
                'method': results.get('method', 'Video Segmentation'),
                'temp_segments_folder': results.get('temp_segments_folder'),  # Updated key name
                'base_name': results.get('base_name'),  # Add base_name
                'segments_created': results.get('segments_created', 0),
                'segments_failed': results.get('segments_failed', 0),
                'total_size_mb': results.get('total_size_mb', 0),
                'segments': results.get('segments', []),  # Keep segments list
                'output_file': results.get('temp_segments_folder', output_path),  # Use temp folder path
                'file_size_mb': results.get('total_size_mb', 0),
                'frame_count': sum(seg.get('frame_count', 0) for seg in results.get('segments', []))
            }
        else:
            # For single GIF results, return as before
            return results

    def request_shutdown(self):
        """Request graceful shutdown of the GIF generator"""
        logger.info("Shutdown requested for GIF generator")
        self.shutdown_requested = True
        if self.current_ffmpeg_process:
            logger.info("Terminating current GIF generation process...")
            self._terminate_ffmpeg_process()
    
    def _terminate_ffmpeg_process(self):
        """Terminate the current FFmpeg process gracefully"""
        if self.current_ffmpeg_process and self.current_ffmpeg_process.poll() is None:
            try:
                # Try graceful termination first
                self.current_ffmpeg_process.terminate()
                
                # Wait a bit for graceful termination
                try:
                    self.current_ffmpeg_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination fails
                    logger.warning("GIF generation process did not terminate gracefully, forcing kill...")
                    self.current_ffmpeg_process.kill()
                    self.current_ffmpeg_process.wait()
                
                logger.info("GIF generation process terminated successfully")
            except Exception as e:
                logger.error(f"Error terminating GIF generation process: {e}")
            finally:
                self.current_ffmpeg_process = None

    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video metadata using OpenCV"""
        try:
            cap = cv2.VideoCapture(video_path)
            
            if not cap.isOpened():
                raise ValueError(f"Could not open video file: {video_path}")
            
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0
            
            cap.release()
            
            return {
                'width': width,
                'height': height,
                'fps': fps,
                'frame_count': frame_count,
                'duration': duration,
                'size_bytes': os.path.getsize(video_path)
            }
            
        except Exception as e:
            logger.error(f"Failed to get video info: {e}")
            raise
    
    def _analyze_video_content(self, input_video: str, start_time: float, duration: float) -> Dict[str, Any]:
        """
        Analyze video content to determine optimal compression parameters
        
        Returns:
            Dictionary with content analysis results including complexity, motion level, etc.
        """
        try:
            # Use FFprobe to analyze video content
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_streams',
                '-ss', str(start_time), '-t', str(duration), input_video
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                probe_data = json.loads(result.stdout)
                video_stream = next((s for s in probe_data['streams'] if s['codec_type'] == 'video'), {})
                
                # Extract basic video properties
                width = int(video_stream.get('width', 0))
                height = int(video_stream.get('height', 0))
                fps = eval(video_stream.get('r_frame_rate', '30/1'))  # Handle fractional FPS
                
                # Calculate complexity based on resolution and bitrate
                bitrate = int(video_stream.get('bit_rate', 0)) if video_stream.get('bit_rate') else 0
                total_pixels = width * height
                
                # Estimate complexity score (1-10 scale)
                if total_pixels > 0 and bitrate > 0:
                    # Higher bitrate per pixel suggests more complex content
                    bitrate_per_pixel = bitrate / total_pixels
                    if bitrate_per_pixel > 0.1:
                        complexity_score = min(10, 6 + bitrate_per_pixel * 40)
                    elif bitrate_per_pixel > 0.05:
                        complexity_score = 4 + bitrate_per_pixel * 40
                    else:
                        complexity_score = max(1, bitrate_per_pixel * 80)
                else:
                    # Fallback based on resolution
                    if total_pixels >= 1920 * 1080:
                        complexity_score = 7
                    elif total_pixels >= 1280 * 720:
                        complexity_score = 5
                    else:
                        complexity_score = 3
                
                # Estimate motion level based on FPS and complexity
                if fps >= 50:
                    motion_level = 'high'
                elif fps >= 30:
                    motion_level = 'medium' if complexity_score < 7 else 'high'
                else:
                    motion_level = 'low'
                
                return {
                    'width': width,
                    'height': height,
                    'fps': fps,
                    'bitrate': bitrate,
                    'complexity_score': complexity_score,
                    'motion_level': motion_level,
                    'total_pixels': total_pixels
                }
            
        except Exception as e:
            logger.debug(f"Content analysis failed: {e}")
        
        # Fallback analysis - basic estimates
        return {
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'bitrate': 0,
            'complexity_score': 5.0,
            'motion_level': 'medium',
            'total_pixels': 1920 * 1080
        }

    def _apply_content_aware_adjustments(self, params: Dict[str, Any], 
                                       video_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply content-aware adjustments to GIF parameters based on video analysis
        
        Args:
            params: Base GIF parameters
            video_analysis: Video content analysis results
            
        Returns:
            Adjusted parameters optimized for the specific content
        """
        adjusted_params = params.copy()
        
        complexity = video_analysis.get('complexity_score', 5.0)
        motion_level = video_analysis.get('motion_level', 'medium')
        total_pixels = video_analysis.get('total_pixels', 1920 * 1080)
        
        # Adjust color palette based on complexity
        if complexity >= 8:
            # High complexity content benefits from more colors
            adjusted_params['colors'] = min(256, int(adjusted_params['colors'] * 1.2))
        elif complexity <= 3:
            # Simple content can use fewer colors
            adjusted_params['colors'] = max(64, int(adjusted_params['colors'] * 0.8))
        
        # Adjust FPS based on motion level
        if motion_level == 'high':
            # High motion content needs higher FPS to look smooth
            adjusted_params['fps'] = min(30, int(adjusted_params['fps'] * 1.1))
        elif motion_level == 'low':
            # Low motion content can use lower FPS
            adjusted_params['fps'] = max(5, int(adjusted_params['fps'] * 0.9))
        
        # Adjust dithering based on complexity
        if complexity >= 7:
            # Complex content benefits from better dithering
            adjusted_params['dither'] = 'floyd_steinberg'
        elif complexity <= 3:
            # Simple content can skip dithering for smaller size
            adjusted_params['dither'] = 'none'
        
        # Adjust lossy compression based on content characteristics
        if complexity >= 8 or motion_level == 'high':
            # Reduce lossy compression for complex/high-motion content
            adjusted_params['lossy'] = max(0, adjusted_params['lossy'] - 20)
        elif complexity <= 3 and motion_level == 'low':
            # Increase lossy compression for simple/low-motion content
            adjusted_params['lossy'] = min(200, adjusted_params['lossy'] + 30)
        
        # Adjust resolution based on source resolution efficiency
        source_pixels = video_analysis.get('width', 1920) * video_analysis.get('height', 1080)
        target_pixels = adjusted_params['width'] * adjusted_params['height']
        
        if source_pixels > 0:
            resolution_ratio = target_pixels / source_pixels
            
            # If we're scaling down significantly, we can be more aggressive with other parameters
            if resolution_ratio < 0.3:  # Scaling down to less than 30% of original
                adjusted_params['colors'] = max(32, int(adjusted_params['colors'] * 0.9))
                adjusted_params['lossy'] = min(200, adjusted_params['lossy'] + 10)
            elif resolution_ratio > 0.7:  # Keeping more than 70% of original resolution
                adjusted_params['colors'] = min(256, int(adjusted_params['colors'] * 1.1))
        
        logger.debug(f"Content-aware adjustments applied: complexity={complexity:.1f}, "
                    f"motion={motion_level}, colors={adjusted_params['colors']}, "
                    f"fps={adjusted_params['fps']}, lossy={adjusted_params['lossy']}")
        
        return adjusted_params

    def _calculate_dynamic_size_limit(self, duration: float, video_info: Dict[str, Any], 
                                     base_limit_mb: float = 10.0) -> float:
        """
        Calculate dynamic size limit based on video characteristics
        
        Args:
            duration: Video duration in seconds
            video_info: Video metadata including resolution, fps, complexity
            base_limit_mb: Base size limit from config
        
        Returns:
            Adjusted size limit in MB that's realistic for the video content
        """
        
        # Start with base limit
        dynamic_limit = base_limit_mb
        
        # Duration-based adjustments - Keep within platform constraints but optimize compression
        # For longer videos, we need more aggressive compression, NOT higher limits
        if duration <= 5:
            # Short videos can use slightly higher quality
            dynamic_limit *= 1.0  # Keep at base limit
        elif duration <= 15:
            # Standard length, use base limit
            dynamic_limit *= 1.0
        elif duration <= 30:
            # Medium length videos need same limit but will use more aggressive compression
            dynamic_limit *= 1.0
        elif duration <= 60:
            # Long videos keep same limit - compression stages will handle the challenge
            dynamic_limit *= 1.0
        else:
            # Very long videos (like 82.7s) keep strict limit - rely on advanced compression
            dynamic_limit *= 1.0
        
        # Resolution-based adjustments
        if video_info and video_info.get('width') and video_info.get('height'):
            total_pixels = video_info['width'] * video_info['height']
            
            # Adjust based on resolution complexity
            if total_pixels >= 1920 * 1080:  # 1080p+
                dynamic_limit *= 1.3
            elif total_pixels >= 1280 * 720:  # 720p
                dynamic_limit *= 1.1
            # Lower resolutions keep base multiplier
        
        # Complexity-based adjustments (if available)
        complexity = video_info.get('complexity_score', 5.0) if video_info else 5.0
        if complexity >= 8:
            # High complexity content needs more space
            dynamic_limit *= 1.2
        elif complexity <= 3:
            # Simple content can be compressed more
            dynamic_limit *= 0.9
        
        # FPS-based adjustments
        fps = video_info.get('fps', 30) if video_info else 30
        if fps >= 50:
            dynamic_limit *= 1.2
        elif fps >= 30:
            dynamic_limit *= 1.1
        # Lower fps keeps base multiplier
        
        # Set reasonable bounds - respect platform constraints
        min_limit = max(base_limit_mb * 0.8, 5.0)  # Don't go too far below base limit
        max_limit = base_limit_mb * 1.0  # NEVER exceed base limit for platform compatibility
        dynamic_limit = max(min_limit, min(dynamic_limit, max_limit))
        
        logger.info(f"Size limit maintained at {dynamic_limit:.1f}MB for platform compatibility "
                   f"(duration: {duration}s, complexity: {complexity:.1f}) - will use aggressive compression for long videos")
        
        return dynamic_limit

    def _calculate_gif_params(self, platform_config: Dict[str, Any], 
                            max_size_mb: Optional[int], 
                            duration: Optional[float],
                            video_info: Optional[Dict[str, Any]] = None,
                            input_video: Optional[str] = None,
                            start_time: float = 0) -> Dict[str, Any]:
        """Calculate optimal GIF parameters with aspect ratio preservation, dynamic sizing, and content analysis"""
        
        # Calculate dynamic size limit based on video characteristics
        base_size_mb = max_size_mb or self.config.get('gif_settings.max_file_size_mb', 10)
        actual_duration = duration or self.config.get('gif_settings.max_duration_seconds', 15)
        
        # Perform content analysis if input video is provided
        if input_video and os.path.exists(input_video):
            content_analysis = self._analyze_video_content(input_video, start_time, actual_duration)
            # Merge content analysis with existing video_info, but prioritize original video info
            video_info_merged = {**content_analysis, **(video_info or {})}
            
            # Ensure we don't override the original video dimensions with fallback values
            if video_info and video_info.get('width') and video_info.get('height'):
                video_info_merged['width'] = video_info['width']
                video_info_merged['height'] = video_info['height']
        else:
            video_info_merged = video_info or {}
        
        dynamic_size_mb = self._calculate_dynamic_size_limit(actual_duration, video_info_merged, base_size_mb)
        
        # Base parameters from config
        # Use original video FPS as starting point, with config as fallback
        original_fps = video_info_merged.get('fps', self.config.get('gif_settings.fps', 15))
        # Cap the maximum initial FPS to avoid extremely high values that would create large files
        base_fps = min(original_fps, 30)  # Cap at 30 FPS for reasonable file sizes
        
        logger.info(f"Using original video FPS: {original_fps:.1f}, base GIF FPS: {base_fps:.1f}")
        
        params = {
            'width': self.config.get('gif_settings.width', 480),
            'height': self.config.get('gif_settings.height', 480),
            'fps': base_fps,  # Use original video FPS as base
            'max_duration': self.config.get('gif_settings.max_duration_seconds', 15),
            'max_size_mb': dynamic_size_mb,  # Use dynamic size limit
            'colors': self.config.get('gif_settings.colors', 256),
            'dither': self.config.get('gif_settings.dither', 'bayer'),
            'lossy': self.config.get('gif_settings.lossy', 80)
        }
        
        # Apply platform-specific settings
        if platform_config:
            params.update({
                'width': platform_config.get('max_width', params['width']),
                'height': platform_config.get('max_height', params['height']),
                'max_duration': platform_config.get('max_duration', params['max_duration'])
            })
        
        # Override duration if specified
        if duration:
            params['max_duration'] = min(duration, params['max_duration'])
        
        # Apply content-aware adjustments to base parameters
        params = self._apply_content_aware_adjustments(params, video_info_merged)
        
        # Calculate aspect ratio preserving dimensions if video info is available
        if video_info_merged and video_info_merged.get('width') and video_info_merged.get('height'):
            original_width = video_info_merged['width']
            original_height = video_info_merged['height']
            original_aspect_ratio = original_width / original_height
            
            # Get maximum dimensions from config
            max_width = params['width']
            max_height = params['height']
            
            logger.info(f"Initial aspect ratio calculation: {original_width}x{original_height} (ratio: {original_aspect_ratio:.2f})")
            logger.info(f"Max dimensions from config: {max_width}x{max_height}")
            
            # Calculate new dimensions while preserving aspect ratio
            if original_aspect_ratio > 1:  # Landscape
                # Width is the limiting factor
                new_width = min(original_width, max_width)
                new_height = int(new_width / original_aspect_ratio)
                # Ensure height doesn't exceed max_height
                if new_height > max_height:
                    new_height = max_height
                    new_width = int(new_height * original_aspect_ratio)
            else:  # Portrait or square
                # Height is the limiting factor
                new_height = min(original_height, max_height)
                new_width = int(new_height * original_aspect_ratio)
                # Ensure width doesn't exceed max_width
                if new_width > max_width:
                    new_width = max_width
                    new_height = int(new_width / original_aspect_ratio)
            
            # Ensure dimensions are even numbers while preserving aspect ratio
            even_width = new_width - (new_width % 2)
            even_height = int(even_width / original_aspect_ratio)
            even_height = even_height - (even_height % 2)
            
            params['width'] = even_width
            params['height'] = even_height
            
            logger.info(f"Final aspect ratio calculation: {original_width}x{original_height} -> {new_width}x{new_height} (ratio: {original_aspect_ratio:.2f})")
            logger.info(f"Final params dimensions: {params['width']}x{params['height']}")
        
        logger.debug(f"GIF parameters: {params}")
        return params
    
    def _create_gif_ffmpeg_palette(self, input_video: str, output_gif: str, 
                                 params: Dict[str, Any], start_time: float, duration: float):
        """Create GIF using FFmpeg with optimized palette (highest quality)"""
        
        logger.info("Creating GIF using FFmpeg with palette optimization")
        logger.info(f"Final GIF creation parameters: {params['width']}x{params['height']}, {params['fps']}fps, {params['colors']} colors")
        
        # Check for shutdown before starting
        if self.shutdown_requested:
            raise RuntimeError("Shutdown requested before GIF generation")
        
        # Create temporary palette file
        palette_file = os.path.join(self.temp_dir, "palette.png")
        
        try:
            # Step 1: Generate optimized palette
            palette_cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', input_video,
                '-vf', f'fps={params["fps"]},scale={params["width"]}:{params["height"]}:flags=lanczos,palettegen=max_colors={params["colors"]}',
                palette_file
            ]
            
            logger.debug(f"Generating palette: {' '.join(palette_cmd)}")
            
            # Execute palette generation with shutdown handling
            process = subprocess.Popen(palette_cmd, capture_output=True, text=True)
            self.current_ffmpeg_process = process
            
            try:
                # Wait for palette generation with shutdown checking
                while process.poll() is None:
                    if self.shutdown_requested:
                        logger.info("Shutdown requested during palette generation, terminating...")
                        self._terminate_ffmpeg_process()
                        raise RuntimeError("Shutdown requested during palette generation")
                    time.sleep(0.1)  # Small delay to allow shutdown checking
                
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, palette_cmd, process.stderr.read())
                    
            finally:
                self.current_ffmpeg_process = None
            
            # Check for shutdown before step 2
            if self.shutdown_requested:
                raise RuntimeError("Shutdown requested between palette generation and GIF creation")
            
            # Step 2: Create GIF using the palette
            gif_cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', input_video,
                '-i', palette_file,
                '-lavfi', f'fps={params["fps"]},scale={params["width"]}:{params["height"]}:flags=lanczos[x];[x][1:v]paletteuse=dither={params["dither"]}',
                output_gif
            ]
            
            logger.debug(f"Creating GIF: {' '.join(gif_cmd)}")
            
            # Execute GIF creation with shutdown handling
            process = subprocess.Popen(gif_cmd, capture_output=True, text=True)
            self.current_ffmpeg_process = process
            
            try:
                # Wait for GIF creation with shutdown checking
                while process.poll() is None:
                    if self.shutdown_requested:
                        logger.info("Shutdown requested during GIF creation, terminating...")
                        self._terminate_ffmpeg_process()
                        raise RuntimeError("Shutdown requested during GIF creation")
                    time.sleep(0.1)  # Small delay to allow shutdown checking
                
                if process.returncode != 0:
                    raise subprocess.CalledProcessError(process.returncode, gif_cmd, process.stderr.read())
                    
            finally:
                self.current_ffmpeg_process = None
            
        finally:
            # Clean up palette file
            if os.path.exists(palette_file):
                os.remove(palette_file)
    
    def _create_gif_ffmpeg_direct(self, input_video: str, output_gif: str,
                                params: Dict[str, Any], start_time: float, duration: float):
        """Create GIF using direct FFmpeg conversion"""
        
        logger.info("Creating GIF using direct FFmpeg conversion")
        logger.info(f"Direct GIF creation parameters: {params['width']}x{params['height']}, {params['fps']}fps, {params['colors']} colors")
        
        # Check for shutdown before starting
        if self.shutdown_requested:
            raise RuntimeError("Shutdown requested before GIF generation")
        
        cmd = [
            'ffmpeg', '-y',
            '-ss', str(start_time),
            '-t', str(duration),
            '-i', input_video,
            '-vf', f'fps={params["fps"]},scale={params["width"]}:{params["height"]}:flags=lanczos',
            '-c:v', 'gif',
            output_gif
        ]
        
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        # Execute with shutdown handling
        process = subprocess.Popen(cmd, capture_output=True, text=True)
        self.current_ffmpeg_process = process
        
        try:
            # Wait for completion with shutdown checking
            while process.poll() is None:
                if self.shutdown_requested:
                    logger.info("Shutdown requested during direct GIF creation, terminating...")
                    self._terminate_ffmpeg_process()
                    raise RuntimeError("Shutdown requested during direct GIF creation")
                time.sleep(0.1)  # Small delay to allow shutdown checking
            
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd, process.stderr.read())
                
        finally:
            self.current_ffmpeg_process = None
    
    def _create_gif_opencv_pil(self, input_video: str, output_gif: str,
                             params: Dict[str, Any], start_time: float, duration: float):
        """Create GIF using OpenCV + PIL (fallback method)"""
        
        logger.info("Creating GIF using OpenCV + PIL (fallback method)")
        logger.info(f"OpenCV/PIL GIF creation parameters: {params['width']}x{params['height']}, {params['fps']}fps, {params['colors']} colors")
        
        cap = cv2.VideoCapture(input_video)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video}")
        
        try:
            # Calculate frame parameters
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = max(1, int(original_fps / params['fps']))  # Ensure at least 1
            start_frame = int(start_time * original_fps)
            end_frame = int((start_time + duration) * original_fps)
            
            # Apply intelligent frame skipping if specified
            frame_skip = params.get('frame_skip', 1)
            if frame_skip > 1:
                frame_interval *= frame_skip
                logger.debug(f"Applying frame skip factor {frame_skip}, new interval: {frame_interval}")
            
            # Calculate expected number of output frames for validation
            expected_frames = int(duration * params['fps'] / frame_skip)
            

            
            # Set starting frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames = []
            frame_count = 0
            extracted_frames = 0
            
            logger.debug(f"Extracting frames: original_fps={original_fps:.1f}, target_fps={params['fps']}, "
                        f"interval={frame_interval}, expected_frames={expected_frames}")
            
            with tqdm(total=expected_frames, desc="Extracting frames") as pbar:
                while cap.isOpened() and frame_count < end_frame - start_frame and extracted_frames < expected_frames:
                    ret, frame = cap.read()
                    
                    if not ret:
                        break
                    
                    # Skip frames to achieve target FPS
                    if frame_count % frame_interval == 0:
                        # Resize frame
                        frame_resized = cv2.resize(frame, (params['width'], params['height']))
                        
                        # Convert BGR to RGB
                        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
                        
                        # Convert to PIL Image
                        pil_frame = Image.fromarray(frame_rgb)
                        frames.append(pil_frame)
                        extracted_frames += 1
                        
                        pbar.update(1)
                    
                    frame_count += 1
            
            if not frames:
                raise ValueError("No frames extracted from video")
            
            # SIMPLE AND CORRECT: Calculate frame duration to maintain the original segment duration
            # Formula: Frame_delay_ms = (duration * 1000) / frames_extracted
            
            actual_frames_extracted = len(frames)
            
            # Calculate frame delay to maintain original duration
            if actual_frames_extracted > 0:
                frame_duration = int((duration * 1000) / actual_frames_extracted)
            else:
                frame_duration = int(1000 / params['fps'])  # Fallback
            
            # Ensure minimum frame duration of 10ms (100fps max) to prevent too-fast GIFs
            frame_duration = max(frame_duration, 10)
            
            # Debug logging for timing calculation
            actual_fps = 1000 / frame_duration if frame_duration > 0 else 0
            actual_duration = actual_frames_extracted * frame_duration / 1000.0
            logger.debug(f"Timing: {actual_frames_extracted} frames @ {frame_duration}ms/frame = {actual_fps:.1f}fps, {actual_duration:.2f}s duration")
            
            # Calculate actual effective FPS and total duration with the extracted frames
            effective_fps = 1000 / frame_duration if frame_duration > 0 else params['fps']
            actual_total_duration = len(frames) * frame_duration / 1000.0
            
            logger.debug(f"GIF timing: {len(frames)} frames @ {frame_duration}ms/frame = "
                        f"{actual_total_duration:.2f}s duration (effective fps: {effective_fps:.1f})")
            
            # Save as GIF with optimization
            frames[0].save(
                output_gif,
                save_all=True,
                append_images=frames[1:],
                duration=frame_duration,
                loop=0,  # Infinite loop
                optimize=True,
                quality=95
            )
            
        finally:
            cap.release()
    
    def _adjust_gif_params(self, params: Dict[str, Any], current_size_mb: float, video_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Adjust GIF parameters to reduce file size while preserving original video aspect ratio"""
        
        size_ratio = params['max_size_mb'] / current_size_mb
        
        # Reduce colors first
        if params['colors'] > 128:
            params['colors'] = max(int(params['colors'] * 0.7), 64)
        
        # Reduce FPS more conservatively
        if params['fps'] > 10:
            params['fps'] = max(int(params['fps'] * 0.85), 8)  # Less aggressive FPS reduction
        
        # Reduce resolution if still too large, preserving ORIGINAL aspect ratio
        if size_ratio < 0.7:
            # Get original aspect ratio from video info if available
            if video_info and video_info.get('width') and video_info.get('height'):
                original_aspect_ratio = video_info['width'] / video_info['height']
            else:
                # Fallback to current aspect ratio if video info not available
                original_aspect_ratio = params['width'] / params['height']
            
            # Reduce dimensions while maintaining original aspect ratio
            reduction_factor = 0.9
            new_width = int(params['width'] * reduction_factor)
            new_height = int(new_width / original_aspect_ratio)  # Calculate height from width using original ratio
            
            # Ensure dimensions are even numbers
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            # Ensure minimum dimensions while preserving aspect ratio
            min_width = 160
            min_height = int(min_width / original_aspect_ratio)
            
            new_width = max(new_width, min_width)
            new_height = max(new_height, min_height)
            
            params['width'] = new_width
            params['height'] = new_height
            
            logger.info(f"Reduced resolution while preserving original aspect ratio ({original_aspect_ratio:.2f}): {new_width}x{new_height}")
            logger.info(f"Reduced params dimensions: {params['width']}x{params['height']}")
        
        # Increase lossy compression
        params['lossy'] = min(params['lossy'] + 20, 150)
        
        logger.debug(f"Adjusted GIF parameters: {params}")
        return params
    
    def optimize_existing_gif(self, input_gif: str, output_gif: str, 
                            max_size_mb: float) -> Dict[str, Any]:
        """Optimize an existing GIF file"""
        
        logger.info(f"Optimizing existing GIF: {input_gif}")
        
        # Get current GIF info
        original_size_mb = os.path.getsize(input_gif) / (1024 * 1024)
        
        if original_size_mb <= max_size_mb:
            # Already small enough, just copy
            shutil.copy2(input_gif, output_gif)
            logger.info("GIF already meets size requirements")
            return self._get_gif_optimization_results(input_gif, output_gif, "copy")
        
        try:
            # Load GIF with PIL
            with Image.open(input_gif) as img:
                frames = []
                durations = []
                
                # Extract all frames and their durations
                for frame in ImageSequence.Iterator(img):
                    frames.append(frame.copy())
                    durations.append(frame.info.get('duration', 100))
                
                # Calculate reduction factor needed
                reduction_factor = max_size_mb / original_size_mb
                
                # Reduce frame count if needed
                if reduction_factor < 0.8:
                    # Keep every nth frame
                    skip_factor = max(2, int(1 / reduction_factor))
                    frames = frames[::skip_factor]
                    durations = durations[::skip_factor]
                
                # Reduce colors
                colors = min(256, max(64, int(256 * reduction_factor)))
                
                # Optimize and save
                if frames:
                    # Quantize frames to reduce colors
                    optimized_frames = []
                    for frame in frames:
                        # Convert to P mode with optimized palette
                        frame_quantized = frame.quantize(colors=colors, method=Image.Dither.FLOYDSTEINBERG)
                        optimized_frames.append(frame_quantized)
                    
                    # Save optimized GIF
                    optimized_frames[0].save(
                        output_gif,
                        save_all=True,
                        append_images=optimized_frames[1:],
                        duration=durations,
                        loop=0,
                        optimize=True
                    )
                    
                    logger.info("GIF optimization completed using PIL")
                    return self._get_gif_optimization_results(input_gif, output_gif, "optimized")
                
        except Exception as e:
            logger.error(f"PIL optimization failed: {e}")
            
            # Fallback to FFmpeg optimization
            try:
                cmd = [
                    'ffmpeg', '-y',
                    '-i', input_gif,
                    '-vf', f'palettegen=max_colors={min(256, int(256 * reduction_factor))}',
                    output_gif
                ]
                
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0:
                    logger.info("GIF optimization completed using FFmpeg")
                    return self._get_gif_optimization_results(input_gif, output_gif, "ffmpeg_optimized")
                
            except Exception as ffmpeg_error:
                logger.error(f"FFmpeg optimization also failed: {ffmpeg_error}")
        
        raise RuntimeError("Failed to optimize GIF with all available methods")
    
    def _estimate_gif_size(self, params: Dict[str, Any], duration: float, video_info: Dict[str, Any]) -> float:
        """Enhanced GIF file size estimation based on video content characteristics"""
        
        # Calculate frame count
        frame_count = int(duration * params['fps'])
        
        # Estimate bytes per frame based on resolution and colors
        pixels_per_frame = params['width'] * params['height']
        
        # Color depth affects compression
        if params['colors'] <= 64:
            bits_per_pixel = 6  # 6 bits for 64 colors
        elif params['colors'] <= 128:
            bits_per_pixel = 7  # 7 bits for 128 colors
        else:
            bits_per_pixel = 8  # 8 bits for 256 colors
        
        # Enhanced compression factor considering video characteristics
        base_compression = 1.0 - (params['lossy'] / 200.0)  # 0.5 to 1.0
        
        # Adjust for video complexity
        complexity = video_info.get('complexity_score', 5.0)
        motion_level = video_info.get('motion_level', 'medium')
        
        # Complex content compresses less effectively
        if complexity >= 8:
            complexity_factor = 1.3  # 30% larger for high complexity
        elif complexity >= 6:
            complexity_factor = 1.15  # 15% larger for medium-high complexity
        elif complexity <= 3:
            complexity_factor = 0.8  # 20% smaller for low complexity
        else:
            complexity_factor = 1.0
        
        # Motion affects compression
        if motion_level == 'high':
            motion_factor = 1.25  # High motion = less compression
        elif motion_level == 'low':
            motion_factor = 0.85  # Low motion = better compression
        else:
            motion_factor = 1.0
        
        # Dithering affects compression
        dither_factor = 0.8 if params['dither'] == 'floyd_steinberg' else 0.9 if params['dither'] == 'bayer' else 1.0
        
        # Calculate estimated size with all factors
        estimated_bytes = (
            pixels_per_frame * 
            bits_per_pixel / 8 * 
            frame_count * 
            base_compression * 
            complexity_factor * 
            motion_factor * 
            dither_factor
        )
        
        # Add overhead for GIF headers and color tables
        estimated_bytes *= 1.1
        
        return estimated_bytes / (1024 * 1024)  # Convert to MB

    def _increase_params_for_quality(self, params: Dict[str, Any], current_size_mb: float, target_size_mb: float, video_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Increase parameters to use more of the available size for better quality"""
        size_ratio = current_size_mb / target_size_mb
        
        # Only increase if we have significant room to spare
        if size_ratio < 0.7:
            # Increase colors if possible
            if params['colors'] < 256:
                params['colors'] = min(params['colors'] + 32, 256)
            
            # Increase FPS if possible
            if params['fps'] < 30:
                params['fps'] = min(params['fps'] + 2, 30)
            
            # Increase resolution if possible while preserving aspect ratio
            if params['width'] < 800 and params['height'] < 800:
                # Get original aspect ratio from video info if available
                if video_info and video_info.get('width') and video_info.get('height'):
                    original_aspect_ratio = video_info['width'] / video_info['height']
                else:
                    # Fallback to current aspect ratio
                    original_aspect_ratio = params['width'] / params['height']
                
                # Increase width and calculate height to preserve aspect ratio
                new_width = min(params['width'] + 20, 800)
                new_height = int(new_width / original_aspect_ratio)
                
                # Ensure height doesn't exceed maximum
                if new_height > 800:
                    new_height = 800
                    new_width = int(new_height * original_aspect_ratio)
                
                # Ensure even dimensions
                params['width'] = new_width - (new_width % 2)
                params['height'] = new_height - (new_height % 2)
            
            # Reduce lossy compression if possible
            if params['lossy'] > 0:
                params['lossy'] = max(params['lossy'] - 10, 0)
        
        return params

    def _create_gif_with_quality_optimization(self, input_video: str, output_path: str,
                                            initial_params: Dict[str, Any], start_time: float,
                                            duration: float, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create GIF with proper optimization workflow:
        1. Optimize until parameters reach shrinking threshold
        2. Try to increase parameters while staying under limit
        3. Analyze candidates under 10MB and select best quality
        4. Move final result to output and cleanup temp files
        """
        
        max_size_mb = initial_params['max_size_mb']
        logger.info(f"Starting systematic GIF optimization (target: {max_size_mb}MB)")
        
        # Get quality optimization settings from config
        quality_config = self.config.get('gif_settings.quality_optimization', {})
        enabled = quality_config.get('enabled', True)
        
        if not enabled:
            logger.info("Quality optimization disabled, using standard generation")
            return self._create_gif_standard(input_video, output_path, initial_params, start_time, duration, video_info)
        
        temp_files_to_cleanup = []
        valid_candidates = []  # Store all candidates under the size limit
        
        try:
            # Phase 1: Find a baseline that fits within size constraints
            logger.info("Phase 1: Finding baseline parameters that fit size constraints")
            baseline_result = self._find_baseline_parameters(
                input_video, initial_params, start_time, duration, max_size_mb, temp_files_to_cleanup
            )
            
            if baseline_result:
                valid_candidates.append(baseline_result)
                logger.info(f"Baseline found: {baseline_result['size_mb']:.2f}MB, Quality: {baseline_result['quality_score']:.2f}")
            
            # Phase 2: Incrementally improve quality while staying under limit
            if baseline_result:
                logger.info("Phase 2: Incrementally improving quality while staying under size limit")
                improved_candidates = self._improve_quality_incrementally(
                    input_video, baseline_result['params'], start_time, duration, max_size_mb, temp_files_to_cleanup
                )
                valid_candidates.extend(improved_candidates)
            
            # Phase 3: Try progressive compression stages if no candidates found
            if not valid_candidates and duration > 30:
                logger.info("Phase 3: Trying progressive compression for long video")
                progressive_candidates = self._try_progressive_compression(
                    input_video, initial_params, start_time, duration, max_size_mb, video_info, temp_files_to_cleanup
                )
                valid_candidates.extend(progressive_candidates)
            
            # Phase 4: Ultimate fallback strategies if still no candidates
            if not valid_candidates:
                logger.info("Phase 4: Trying ultimate fallback strategies")
                fallback_candidates = self._try_ultimate_fallback_strategies(
                    input_video, initial_params, start_time, duration, max_size_mb, video_info, temp_files_to_cleanup
                )
                valid_candidates.extend(fallback_candidates)
            
            # Phase 5: Final analysis and selection
            if not valid_candidates:
                raise RuntimeError(f"Failed to generate GIF under {max_size_mb}MB with any optimization strategy, "
                                 f"including all fallback methods. Consider increasing size limit or reducing duration.")
            
            logger.info(f"Phase 5: Analyzing {len(valid_candidates)} candidates to select best quality")
            final_result = self._select_best_candidate(valid_candidates, max_size_mb)
            
            # Phase 6: Move final result to output location and cleanup
            logger.info(f"Final selection: {final_result['method']} - {final_result['size_mb']:.2f}MB, Quality: {final_result['quality_score']:.2f}")
            
            # Move the selected result to final output path
            if os.path.exists(final_result['temp_file']):
                shutil.move(final_result['temp_file'], output_path)
                # Remove from cleanup list since it's now the final output
                if final_result['temp_file'] in temp_files_to_cleanup:
                    temp_files_to_cleanup.remove(final_result['temp_file'])
            
            # Log detailed file specifications
            try:
                from .ffmpeg_utils import FFmpegUtils
                specs = FFmpegUtils.get_detailed_file_specifications(output_path)
                specs_log = FFmpegUtils.format_file_specifications_for_logging(specs)
                logger.info(f"GIF optimization completed successfully - {specs_log}")
            except Exception as e:
                logger.warning(f"Could not log detailed GIF specifications: {e}")
            
            # Prepare final result data
            result_data = {
                'success': True,
                'output_file': output_path,
                'file_size_mb': final_result['size_mb'],
                'frame_count': self._count_gif_frames(output_path),
                'quality_score': final_result['quality_score'],
                'optimization_method': final_result['method'],
                'params': final_result['params']
            }
            
            return result_data
        
        except Exception as e:
            logger.error(f"GIF optimization failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'output_file': output_path,
                'file_size_mb': 0,
                'frame_count': 0
            }
        
        finally:
            # Clean up all temporary files
            logger.info(f"Cleaning up {len(temp_files_to_cleanup)} temporary files")
            for temp_file in temp_files_to_cleanup:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                        logger.debug(f"Cleaned up: {temp_file}")
                    except Exception as e:
                        logger.debug(f"Failed to clean up {temp_file}: {e}")
    
    def _find_baseline_parameters(self, input_video: str, initial_params: Dict[str, Any], 
                                start_time: float, duration: float, max_size_mb: float,
                                temp_files_to_cleanup: List[str]) -> Optional[Dict[str, Any]]:
        """
        Phase 1: Find baseline parameters that produce a GIF under the size limit
        """
        # Start with aggressive compression to ensure we get something under the limit
        current_params = initial_params.copy()
        
        # Reduce parameters significantly to get under size limit quickly while preserving aspect ratio
        original_aspect_ratio = current_params['width'] / current_params['height']
        new_width = int(current_params['width'] * 0.7)
        new_height = int(new_width / original_aspect_ratio)
        
        logger.info(f"Baseline reduction: {current_params['width']}x{current_params['height']} -> {new_width}x{new_height} (ratio: {original_aspect_ratio:.2f})")
        
        current_params.update({
            'colors': 64,
            'fps': max(6, current_params['fps'] * 0.5),
            'width': new_width,
            'height': new_height,
            'lossy': 100,
            'dither': 'bayer'
        })
        
        # Ensure even dimensions while preserving aspect ratio
        even_width = current_params['width'] - (current_params['width'] % 2)
        even_height = int(even_width / original_aspect_ratio)
        even_height = even_height - (even_height % 2)
        
        current_params['width'] = even_width
        current_params['height'] = even_height
        
        logger.info(f"Baseline after even dimensions: {current_params['width']}x{current_params['height']} (ratio: {original_aspect_ratio:.2f})")
        
        temp_gif = os.path.join(self.temp_dir, "baseline_candidate.gif")
        temp_files_to_cleanup.append(temp_gif)
        
        try:
            # Try OpenCV + PIL method for reliability
            self._create_gif_opencv_pil(input_video, temp_gif, current_params, start_time, duration)
            
            if os.path.exists(temp_gif):
                file_size_mb = os.path.getsize(temp_gif) / (1024 * 1024)
                
                if file_size_mb <= max_size_mb:
                    quality_score = self._calculate_quality_score(current_params, file_size_mb, max_size_mb)
                    
                    return {
                        'temp_file': temp_gif,
                        'params': current_params,
                        'size_mb': file_size_mb,
                        'quality_score': quality_score,
                        'method': 'Baseline'
                    }
        
        except Exception as e:
            logger.debug(f"Baseline generation failed: {e}")
        
        return None
    
    def _improve_quality_incrementally(self, input_video: str, baseline_params: Dict[str, Any],
                                     start_time: float, duration: float, max_size_mb: float,
                                     temp_files_to_cleanup: List[str]) -> List[Dict[str, Any]]:
        """
        Phase 2: Incrementally improve quality while staying under size limit
        """
        candidates = []
        current_params = baseline_params.copy()
        
        # Define improvement steps
        improvements = [
            {'colors': 96, 'fps_mult': 1.2, 'res_mult': 1.1, 'lossy': 80},
            {'colors': 128, 'fps_mult': 1.4, 'res_mult': 1.2, 'lossy': 60},
            {'colors': 192, 'fps_mult': 1.6, 'res_mult': 1.3, 'lossy': 40},
            {'colors': 256, 'fps_mult': 1.8, 'res_mult': 1.4, 'lossy': 20},
        ]
        
        for i, improvement in enumerate(improvements):
            test_params = baseline_params.copy()
            
            # Preserve aspect ratio when improving resolution
            # Use the previous improvement's result as the base, or baseline if this is the first improvement
            base_width = baseline_params['width']
            base_height = baseline_params['height']
            
            original_aspect_ratio = base_width / base_height
            new_width = int(base_width * improvement['res_mult'])
            new_height = int(new_width / original_aspect_ratio)
            
            logger.info(f"Quality improvement {i+1}: {base_width}x{base_height} -> {new_width}x{new_height} (ratio: {original_aspect_ratio:.2f})")
            
            test_params.update({
                'colors': improvement['colors'],
                'fps': max(6, int(baseline_params['fps'] * improvement['fps_mult'])),
                'width': new_width,
                'height': new_height,
                'lossy': improvement['lossy']
            })
            
            # Ensure even dimensions while preserving aspect ratio
            even_width = test_params['width'] - (test_params['width'] % 2)
            even_height = int(even_width / original_aspect_ratio)
            even_height = even_height - (even_height % 2)
            
            test_params['width'] = even_width
            test_params['height'] = even_height
            
            logger.info(f"Quality improvement {i+1} after even dimensions: {test_params['width']}x{test_params['height']} (ratio: {original_aspect_ratio:.2f})")
            
            temp_gif = os.path.join(self.temp_dir, f"improved_candidate_{i}.gif")
            temp_files_to_cleanup.append(temp_gif)
            
            try:
                self._create_gif_opencv_pil(input_video, temp_gif, test_params, start_time, duration)
                
                if os.path.exists(temp_gif):
                    file_size_mb = os.path.getsize(temp_gif) / (1024 * 1024)
                    
                    if file_size_mb <= max_size_mb:
                        quality_score = self._calculate_quality_score(test_params, file_size_mb, max_size_mb)
                        
                        candidates.append({
                            'temp_file': temp_gif,
                            'params': test_params,
                            'size_mb': file_size_mb,
                            'quality_score': quality_score,
                            'method': f'Improved Quality Level {i+1}'
                        })
                        
                        logger.info(f"Improved candidate {i+1}: {file_size_mb:.2f}MB, Quality: {quality_score:.2f}")
                    else:
                        logger.debug(f"Improvement {i+1} too large: {file_size_mb:.2f}MB")
                        break  # Stop trying higher quality levels
            
            except Exception as e:
                logger.debug(f"Improvement {i+1} failed: {e}")
                break
        
        return candidates
    
    def _try_progressive_compression(self, input_video: str, initial_params: Dict[str, Any],
                                    start_time: float, duration: float, max_size_mb: float,
                                    video_info: Dict[str, Any], temp_files_to_cleanup: List[str]) -> List[Dict[str, Any]]:
        """
        Phase 3: Try progressive compression stages adapted to video characteristics
        """
        candidates = []
        progressive_stages = self._get_progressive_compression_stages(initial_params.copy(), duration, video_info)
        
        for i, stage in enumerate(progressive_stages):
            temp_gif = os.path.join(self.temp_dir, f"progressive_candidate_{i}.gif")
            temp_files_to_cleanup.append(temp_gif)
            
            try:
                # Handle frame skipping if specified
                stage_params = stage['params'].copy()
                if 'frame_skip' in stage_params:
                    # Implement frame skipping in the GIF creation process
                    # This will be handled by the GIF creation method
                    pass
                
                self._create_gif_opencv_pil(input_video, temp_gif, stage_params, start_time, duration)
                
                if os.path.exists(temp_gif):
                    file_size_mb = os.path.getsize(temp_gif) / (1024 * 1024)
                    
                    if file_size_mb <= max_size_mb:
                        quality_score = self._calculate_quality_score(stage_params, file_size_mb, max_size_mb)
                        
                        candidates.append({
                            'temp_file': temp_gif,
                            'params': stage_params,
                            'size_mb': file_size_mb,
                            'quality_score': quality_score,
                            'method': stage['name'],
                            'reduction_factor': stage['reduction_factor']
                        })
                        
                        logger.info(f"Progressive candidate {stage['name']}: {file_size_mb:.2f}MB, "
                                   f"Quality: {quality_score:.2f}, Reduction: {stage['reduction_factor']:.2f}")
                    else:
                        logger.debug(f"Progressive stage {stage['name']} too large: {file_size_mb:.2f}MB > {max_size_mb}MB")
            
            except Exception as e:
                logger.debug(f"Progressive stage {stage['name']} failed: {e}")
        
        return candidates
    
    def _try_ultimate_fallback_strategies(self, input_video: str, initial_params: Dict[str, Any],
                                          start_time: float, duration: float, max_size_mb: float,
                                          video_info: Dict[str, Any], temp_files_to_cleanup: List[str]) -> List[Dict[str, Any]]:
        """
        Ultimate fallback strategies when all progressive compression attempts fail
        
        These are last-resort methods for extremely challenging videos
        """
        candidates = []
        
        logger.info("Trying ultimate fallback strategies for extremely challenging video")
        
        # Strategy 1: Segment-based processing for very long videos
        if duration > 60:
            logger.info("Fallback 1: Trying segment-based processing")
            segment_candidates = self._try_segment_based_processing(
                input_video, initial_params, start_time, duration, max_size_mb, video_info, temp_files_to_cleanup
            )
            candidates.extend(segment_candidates)
        
        # Strategy 2: Ultra-aggressive single-pass compression
        logger.info("Fallback 2: Trying ultra-aggressive compression")
        ultra_aggressive_candidate = self._try_ultra_aggressive_compression(
            input_video, initial_params, start_time, duration, max_size_mb, temp_files_to_cleanup
        )
        if ultra_aggressive_candidate:
            candidates.append(ultra_aggressive_candidate)
        
        # Strategy 3: Keyframe-only extraction for motion-heavy content
        if video_info.get('motion_level') == 'high':
            logger.info("Fallback 3: Trying keyframe-only extraction")
            keyframe_candidate = self._try_keyframe_only_extraction(
                input_video, initial_params, start_time, duration, max_size_mb, temp_files_to_cleanup
            )
            if keyframe_candidate:
                candidates.append(keyframe_candidate)
        
        # Strategy 4: Time-lapse style compression (for very long videos)
        if duration > 45:
            logger.info("Fallback 4: Trying time-lapse compression")
            timelapse_candidate = self._try_timelapse_compression(
                input_video, initial_params, start_time, duration, max_size_mb, temp_files_to_cleanup
            )
            if timelapse_candidate:
                candidates.append(timelapse_candidate)
        
        return candidates
    
    def _try_segment_based_processing(self, input_video: str, initial_params: Dict[str, Any],
                                     start_time: float, duration: float, max_size_mb: float,
                                     video_info: Dict[str, Any], temp_files_to_cleanup: List[str]) -> List[Dict[str, Any]]:
        """
        Process very long videos by selecting key segments instead of the entire duration
        """
        candidates = []
        
        # For videos longer than 60 seconds, create multiple short segments
        segment_duration = min(15, duration / 4)  # Aim for ~4 segments or 15s max each
        segments_to_try = min(4, int(duration / segment_duration))
        
        for i in range(segments_to_try):
            segment_start = start_time + (i * duration / segments_to_try)
            
            temp_gif = os.path.join(self.temp_dir, f"segment_candidate_{i}.gif")
            temp_files_to_cleanup.append(temp_gif)
            
            try:
                # Use compressed parameters for segments - balanced for 10MB limit with better FPS
                segment_params = initial_params.copy()
                
                # Preserve aspect ratio when reducing dimensions for segments
                original_aspect_ratio = segment_params['width'] / segment_params['height']
                new_width = int(segment_params['width'] * 0.6)
                new_height = int(new_width / original_aspect_ratio)
                
                segment_params.update({
                    'colors': 64,  # More reasonable color reduction (was 32)
                    'fps': max(10, int(segment_params['fps'] * 0.6)),  # Less aggressive FPS reduction (was 0.4)
                    'width': new_width,
                    'height': new_height,
                    'lossy': 140,  # Slightly reduced lossy compression (was 160)
                    'dither': 'bayer',  # Use some dithering for better quality
                    'frame_skip': 2  # Keep frame skipping for size control
                })
                
                # Ensure even dimensions and lower minimums for extreme compression while preserving aspect ratio
                even_width = segment_params['width'] - (segment_params['width'] % 2)
                even_height = int(even_width / original_aspect_ratio)
                even_height = even_height - (even_height % 2)
                
                segment_params['width'] = max(even_width, 120)
                segment_params['height'] = max(even_height, 90)
                
                self._create_gif_opencv_pil(input_video, temp_gif, segment_params, segment_start, segment_duration)
                
                if os.path.exists(temp_gif):
                    file_size_mb = os.path.getsize(temp_gif) / (1024 * 1024)
                    
                    if file_size_mb <= max_size_mb:
                        quality_score = self._calculate_quality_score(segment_params, file_size_mb, max_size_mb, duration_bonus=0.5)
                        
                        candidates.append({
                            'temp_file': temp_gif,
                            'params': segment_params,
                            'size_mb': file_size_mb,
                            'quality_score': quality_score,
                            'method': f"Segment {i+1} ({segment_start:.1f}s-{segment_start+segment_duration:.1f}s)"
                        })
                        
                        logger.info(f"Segment candidate {i+1}: {file_size_mb:.2f}MB, "
                                   f"Quality: {quality_score:.2f}")
            
            except Exception as e:
                logger.debug(f"Segment {i+1} processing failed: {e}")
        
        return candidates
    
    def _try_ultra_aggressive_compression(self, input_video: str, initial_params: Dict[str, Any],
                                         start_time: float, duration: float, max_size_mb: float,
                                         temp_files_to_cleanup: List[str]) -> Optional[Dict[str, Any]]:
        """
        Ultra-aggressive compression as last resort - Improved for better motion quality
        """
        temp_gif = os.path.join(self.temp_dir, "ultra_aggressive_candidate.gif")
        temp_files_to_cleanup.append(temp_gif)
        
        try:
            # Aggressive but more reasonable parameters for 10MB constraint
            ultra_params = initial_params.copy()
            
            # Preserve aspect ratio when reducing dimensions
            original_aspect_ratio = initial_params['width'] / initial_params['height']
            new_width = max(120, int(initial_params['width'] * 0.4))
            new_height = int(new_width / original_aspect_ratio)
            
            # Ensure height doesn't go below minimum
            if new_height < 90:
                new_height = 90
                new_width = int(new_height * original_aspect_ratio)
            
            ultra_params.update({
                'colors': 32,  # Increased from 8 for better quality
                'fps': max(8, int(initial_params['fps'] * 0.4)),  # Less aggressive FPS reduction (was 0.1)
                'width': new_width,
                'height': new_height,
                'lossy': 150,  # Reduced from 200 for better quality
                'dither': 'bayer',  # Use some dithering for better quality
                'frame_skip': max(3, int(duration / 10))  # Less aggressive frame skipping (was duration/5)
            })
            
            # Ensure even dimensions while preserving aspect ratio
            even_width = ultra_params['width'] - (ultra_params['width'] % 2)
            even_height = int(even_width / original_aspect_ratio)
            even_height = even_height - (even_height % 2)
            
            ultra_params['width'] = even_width
            ultra_params['height'] = even_height
            
            self._create_gif_opencv_pil(input_video, temp_gif, ultra_params, start_time, duration)
            
            if os.path.exists(temp_gif):
                file_size_mb = os.path.getsize(temp_gif) / (1024 * 1024)
                
                if file_size_mb <= max_size_mb:
                    quality_score = self._calculate_quality_score(ultra_params, file_size_mb, max_size_mb, duration_bonus=0.2)
                    
                    return {
                        'temp_file': temp_gif,
                        'params': ultra_params,
                        'size_mb': file_size_mb,
                        'quality_score': quality_score,
                        'method': "Ultra Aggressive Compression"
                    }
        
        except Exception as e:
            logger.debug(f"Ultra aggressive compression failed: {e}")
        
        return None
    
    def _try_keyframe_only_extraction(self, input_video: str, initial_params: Dict[str, Any],
                                     start_time: float, duration: float, max_size_mb: float,
                                     temp_files_to_cleanup: List[str]) -> Optional[Dict[str, Any]]:
        """
        Extract only keyframes for high-motion content
        """
        temp_gif = os.path.join(self.temp_dir, "keyframe_candidate.gif")
        temp_files_to_cleanup.append(temp_gif)
        
        try:
            # Parameters optimized for keyframe extraction
            keyframe_params = initial_params.copy()
            
            # Preserve aspect ratio when reducing dimensions
            original_aspect_ratio = initial_params['width'] / initial_params['height']
            new_width = int(initial_params['width'] * 0.7)
            new_height = int(new_width / original_aspect_ratio)
            
            keyframe_params.update({
                'colors': 128,
                'fps': max(2, int(initial_params['fps'] * 0.1)),  # Very low FPS for keyframes
                'width': new_width,
                'height': new_height,
                'lossy': 100,
                'dither': 'bayer',
                'frame_skip': 8  # Extract every 8th frame
            })
            
            # Ensure even dimensions while preserving aspect ratio
            even_width = keyframe_params['width'] - (keyframe_params['width'] % 2)
            even_height = int(even_width / original_aspect_ratio)
            even_height = even_height - (even_height % 2)
            
            keyframe_params['width'] = even_width
            keyframe_params['height'] = even_height
            
            self._create_gif_opencv_pil(input_video, temp_gif, keyframe_params, start_time, duration)
            
            if os.path.exists(temp_gif):
                file_size_mb = os.path.getsize(temp_gif) / (1024 * 1024)
                
                if file_size_mb <= max_size_mb:
                    quality_score = self._calculate_quality_score(keyframe_params, file_size_mb, max_size_mb, duration_bonus=0.3)
                    
                    return {
                        'temp_file': temp_gif,
                        'params': keyframe_params,
                        'size_mb': file_size_mb,
                        'quality_score': quality_score,
                        'method': "Keyframe Only Extraction"
                    }
        
        except Exception as e:
            logger.debug(f"Keyframe extraction failed: {e}")
        
        return None
    
    def _try_timelapse_compression(self, input_video: str, initial_params: Dict[str, Any],
                                  start_time: float, duration: float, max_size_mb: float,
                                  temp_files_to_cleanup: List[str]) -> Optional[Dict[str, Any]]:
        """
        Create a time-lapse style GIF for very long videos - Improved for better motion quality
        """
        temp_gif = os.path.join(self.temp_dir, "timelapse_candidate.gif")
        temp_files_to_cleanup.append(temp_gif)
        
        try:
            # Time-lapse parameters - Better balance of skip rate and quality
            timelapse_params = initial_params.copy()
            
            # Preserve aspect ratio when reducing dimensions
            original_aspect_ratio = initial_params['width'] / initial_params['height']
            new_width = int(initial_params['width'] * 0.85)
            new_height = int(new_width / original_aspect_ratio)
            
            timelapse_params.update({
                'colors': 128,  # Increased from 96 for better quality
                'fps': max(12, int(initial_params['fps'] * 0.6)),  # Less aggressive FPS reduction
                'width': new_width,
                'height': new_height,
                'lossy': 70,  # Reduced from 80 for better quality
                'dither': 'floyd_steinberg',  # Better dithering for quality
                'frame_skip': max(3, int(duration / 15))  # Less aggressive skipping for smoother motion
            })
            
            # Ensure even dimensions while preserving aspect ratio
            even_width = timelapse_params['width'] - (timelapse_params['width'] % 2)
            even_height = int(even_width / original_aspect_ratio)
            even_height = even_height - (even_height % 2)
            
            timelapse_params['width'] = even_width
            timelapse_params['height'] = even_height
            
            self._create_gif_opencv_pil(input_video, temp_gif, timelapse_params, start_time, duration)
            
            if os.path.exists(temp_gif):
                file_size_mb = os.path.getsize(temp_gif) / (1024 * 1024)
                
                if file_size_mb <= max_size_mb:
                    quality_score = self._calculate_quality_score(timelapse_params, file_size_mb, max_size_mb, duration_bonus=0.4)
                    
                    return {
                        'temp_file': temp_gif,
                        'params': timelapse_params,
                        'size_mb': file_size_mb,
                        'quality_score': quality_score,
                        'method': "Time-lapse Compression"
                    }
        
        except Exception as e:
            logger.debug(f"Time-lapse compression failed: {e}")
        
        return None

    def _select_best_candidate(self, candidates: List[Dict[str, Any]], max_size_mb: float) -> Dict[str, Any]:
        """
        Phase 4: Analyze all candidates and select the best quality one
        """
        if not candidates:
            raise RuntimeError("No valid candidates to select from")
        
        # Sort candidates by quality score (descending) then by size efficiency
        def candidate_score(candidate):
            size_efficiency = candidate['size_mb'] / max_size_mb
            quality_weight = 0.7
            size_weight = 0.3
            return (candidate['quality_score'] * quality_weight) + (size_efficiency * size_weight)
        
        candidates.sort(key=candidate_score, reverse=True)
        
        best_candidate = candidates[0]
        logger.info(f"Selected best candidate: {best_candidate['method']} - "
                   f"{best_candidate['size_mb']:.2f}MB ({best_candidate['size_mb']/max_size_mb:.1%} of limit), "
                   f"Quality: {best_candidate['quality_score']:.2f}")
        
        return best_candidate
    
    def _count_gif_frames(self, gif_path: str) -> int:
        """Count the number of frames in a GIF file"""
        try:
            with Image.open(gif_path) as img:
                return img.n_frames
        except Exception:
            return 0

    def _create_gif_standard(self, input_video: str, output_path: str,
                           params: Dict[str, Any], start_time: float,
                           duration: float, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create GIF using standard method (fallback when quality optimization is disabled)"""
        
        logger.info("Using standard GIF generation method")
        
        temp_gif = os.path.join(self.temp_dir, "temp_gif_standard.gif")
        
        try:
            # Try FFmpeg palette method first
            try:
                self._create_gif_ffmpeg_palette(input_video, temp_gif, params, start_time, duration)
            except Exception as e:
                logger.debug(f"FFmpeg palette failed: {e}")
                # Fallback to direct method
                self._create_gif_ffmpeg_direct(input_video, temp_gif, params, start_time, duration)
            
            # Move to final output
            shutil.move(temp_gif, output_path)
            
            return self._get_gif_results(input_video, output_path, video_info, params, duration)
            
        except Exception as e:
            if os.path.exists(temp_gif):
                os.remove(temp_gif)
            raise

    def _build_quality_stage_params(self, base_params: Dict[str, Any], stage_config: Dict[str, Any]) -> Dict[str, Any]:
        """Build quality stage parameters from configuration"""
        params = base_params.copy()
        
        # Apply stage-specific settings
        if 'colors' in stage_config:
            params['colors'] = stage_config['colors']
        
        if 'fps_multiplier' in stage_config:
            params['fps'] = min(30, int(params['fps'] * stage_config['fps_multiplier']))
        
        if 'resolution_multiplier' in stage_config:
            # Preserve aspect ratio when adjusting resolution
            original_aspect_ratio = params['width'] / params['height']
            new_width = min(800, int(params['width'] * stage_config['resolution_multiplier']))
            new_height = int(new_width / original_aspect_ratio)
            
            # Ensure height doesn't exceed maximum
            if new_height > 800:
                new_height = 800
                new_width = int(new_height * original_aspect_ratio)
            
            # Ensure even dimensions
            params['width'] = new_width - (new_width % 2)
            params['height'] = new_height - (new_height % 2)
        
        if 'lossy' in stage_config:
            params['lossy'] = stage_config['lossy']
        
        if 'dither' in stage_config:
            params['dither'] = stage_config['dither']
        
        return params

    def _get_maximum_quality_params(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters for maximum quality GIF generation"""
        params = base_params.copy()
        
        # Maximum quality settings
        params['colors'] = 256
        params['fps'] = min(30, params['fps'] * 2)  # Increase FPS but cap at 30
        params['lossy'] = 0  # No lossy compression
        params['dither'] = 'floyd_steinberg'  # Best dithering
        
        # Try to increase resolution if possible while preserving aspect ratio
        original_aspect_ratio = params['width'] / params['height']
        new_width = min(params['width'] * 1.5, 800)  # Increase width but cap at 800
        new_height = int(new_width / original_aspect_ratio)
        
        # Ensure height doesn't exceed maximum
        if new_height > 800:
            new_height = 800
            new_width = int(new_height * original_aspect_ratio)
        
        # Ensure even dimensions
        params['width'] = int(new_width) - (int(new_width) % 2)
        params['height'] = int(new_height) - (int(new_height) % 2)
        
        return params

    def _get_high_quality_params(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters for high quality GIF generation"""
        params = base_params.copy()
        
        # High quality settings
        params['colors'] = 256
        params['fps'] = min(25, params['fps'] * 1.5)  # Good FPS
        params['lossy'] = 20  # Minimal lossy compression
        params['dither'] = 'floyd_steinberg'
        
        # Slightly increased resolution while preserving aspect ratio
        original_aspect_ratio = params['width'] / params['height']
        new_width = min(params['width'] * 1.2, 600)
        new_height = int(new_width / original_aspect_ratio)
        
        # Ensure height doesn't exceed maximum
        if new_height > 600:
            new_height = 600
            new_width = int(new_height * original_aspect_ratio)
        
        # Ensure even dimensions
        params['width'] = int(new_width) - (int(new_width) % 2)
        params['height'] = int(new_height) - (int(new_height) % 2)
        
        return params

    def _get_progressive_compression_stages(self, base_params: Dict[str, Any], 
                                           duration: float, video_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate progressive compression stages adapted to individual video characteristics
        
        Creates multiple compression stages that progressively reduce quality while maintaining
        visual coherence. The stages adapt based on video duration, resolution, and complexity.
        """
        stages = []
        
        # Determine the number and aggressiveness of stages based on duration
        if duration <= 15:
            # Short videos: fewer, gentler stages
            stage_configs = [
                {'quality_level': 'high', 'reduction_factor': 0.85},
                {'quality_level': 'medium', 'reduction_factor': 0.70},
                {'quality_level': 'low', 'reduction_factor': 0.55}
            ]
        elif duration <= 30:
            # Medium videos: balanced approach
            stage_configs = [
                {'quality_level': 'high', 'reduction_factor': 0.80},
                {'quality_level': 'medium_high', 'reduction_factor': 0.65},
                {'quality_level': 'medium', 'reduction_factor': 0.50},
                {'quality_level': 'low', 'reduction_factor': 0.40}
            ]
        elif duration <= 60:
            # Long videos: more aggressive but gradual
            stage_configs = [
                {'quality_level': 'medium_high', 'reduction_factor': 0.75},
                {'quality_level': 'medium', 'reduction_factor': 0.60},
                {'quality_level': 'medium_low', 'reduction_factor': 0.45},
                {'quality_level': 'low', 'reduction_factor': 0.35},
                {'quality_level': 'minimal', 'reduction_factor': 0.25}
            ]
        else:
            # Very long videos (like 82.7s): extremely aggressive compression within 10MB limit
            stage_configs = [
                {'quality_level': 'medium', 'reduction_factor': 0.50},
                {'quality_level': 'medium_low', 'reduction_factor': 0.35},
                {'quality_level': 'low', 'reduction_factor': 0.25},
                {'quality_level': 'minimal', 'reduction_factor': 0.18},
                {'quality_level': 'ultra_minimal', 'reduction_factor': 0.12},
                {'quality_level': 'extreme', 'reduction_factor': 0.08},
                {'quality_level': 'ultra_extreme', 'reduction_factor': 0.05}
            ]
        
        # Get video characteristics for adaptive compression
        complexity = video_info.get('complexity_score', 5.0)
        motion_level = video_info.get('motion_level', 'medium')
        
        for i, config in enumerate(stage_configs):
            stage_params = base_params.copy()
            reduction_factor = config['reduction_factor']
            
            # Adaptive frame rate reduction based on content
            if motion_level == 'high':
                # High motion content needs to preserve more frames
                fps_reduction = min(0.8, reduction_factor + 0.15)
            elif motion_level == 'low':
                # Low motion content can afford more aggressive frame reduction
                fps_reduction = max(0.3, reduction_factor - 0.1)
            else:
                fps_reduction = reduction_factor
            
            # Adaptive color reduction based on complexity
            if complexity >= 8:
                # Complex content needs more colors
                color_reduction = min(0.8, reduction_factor + 0.1)
            elif complexity <= 3:
                # Simple content can use fewer colors
                color_reduction = max(0.2, reduction_factor - 0.1)
            else:
                color_reduction = reduction_factor
            
            # Calculate progressive parameters while preserving aspect ratio
            # Get original aspect ratio from video info if available
            if video_info and video_info.get('width') and video_info.get('height'):
                original_aspect_ratio = video_info['width'] / video_info['height']
            else:
                # Fallback to current aspect ratio
                original_aspect_ratio = stage_params['width'] / stage_params['height']
            
            # Calculate new dimensions while preserving aspect ratio
            reduction_factor_with_offset = reduction_factor + 0.2
            new_width = max(120, int(stage_params['width'] * reduction_factor_with_offset))
            new_height = int(new_width / original_aspect_ratio)  # Calculate height from width using original ratio
            
            # Ensure dimensions are even numbers
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            # Ensure minimum dimensions while preserving aspect ratio
            min_width = 120
            min_height = int(min_width / original_aspect_ratio)
            
            new_width = max(new_width, min_width)
            new_height = max(new_height, min_height)
            
            stage_params.update({
                'colors': max(16, int(256 * color_reduction)),
                'fps': max(3, int(stage_params['fps'] * fps_reduction)),
                'width': new_width,
                'height': new_height,
                'lossy': min(200, int(20 + (1 - reduction_factor) * 180)),  # Progressive lossy compression
            })
            
            # Adaptive dithering based on stage
            if reduction_factor >= 0.6:
                stage_params['dither'] = 'floyd_steinberg'  # Better quality for higher stages
            elif reduction_factor >= 0.4:
                stage_params['dither'] = 'bayer'
            else:
                stage_params['dither'] = 'none'  # No dithering for highly compressed stages
            
            logger.info(f"Progressive stage {i+1}: {new_width}x{new_height} (aspect ratio: {original_aspect_ratio:.2f})")
            logger.info(f"Stage {i+1} params dimensions: {stage_params['width']}x{stage_params['height']}")
            
            stages.append({
                'name': f"Progressive {config['quality_level'].replace('_', ' ').title()}",
                'params': stage_params,
                'reduction_factor': reduction_factor
            })
        
        # Add intelligent frame sampling for very long videos - Less aggressive skipping
        if duration > 45:
            # Add frame sampling stages that reduce total frames by skipping - REDUCED AGGRESSIVENESS
            for skip_factor in [2, 3, 4]:  # Much less aggressive skipping (was [3, 5, 8, 12])
                if len(stages) < 12:  # Allow more stages for challenging content
                    sample_params = stages[-1]['params'].copy()  # Use last stage as base
                    sample_params['frame_skip'] = skip_factor
                    # Less aggressive frame sampling to maintain better motion quality
                    sample_params['colors'] = max(32, int(sample_params['colors'] * 0.9))  # Less color reduction
                    sample_params['fps'] = max(8, int(sample_params['fps'] * 0.9))  # Less FPS reduction
                    stages.append({
                        'name': f"Frame Sampled (skip {skip_factor})",
                        'params': sample_params,
                        'reduction_factor': stages[-1]['reduction_factor'] / skip_factor
                    })
        
        logger.info(f"Generated {len(stages)} progressive compression stages for {duration}s video "
                   f"(complexity: {complexity:.1f}, motion: {motion_level})")
        
        return stages

    def _get_medium_quality_params(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters for medium quality GIF generation"""
        params = base_params.copy()
        
        # Medium quality settings
        params['colors'] = 192
        params['fps'] = min(20, params['fps'] * 1.2)
        params['lossy'] = 40
        params['dither'] = 'bayer'
        
        return params

    def _reduce_params_for_size(self, params: Dict[str, Any], current_size_mb: float, target_size_mb: float, video_info: Dict[str, Any] = None) -> Dict[str, Any]:
        """Reduce parameters to achieve target size while preserving aspect ratio and reasonable FPS"""
        size_ratio = target_size_mb / current_size_mb
        
        # Get original aspect ratio for proper scaling
        if video_info and video_info.get('width') and video_info.get('height'):
            original_aspect_ratio = video_info['width'] / video_info['height']
        else:
            original_aspect_ratio = params['width'] / params['height']
        
        # More conservative reduction based on how far over the limit we are
        if size_ratio < 0.5:
            # Way too large - aggressive but reasonable reduction
            reduction_factor = 0.7  # Less aggressive than before
            params['colors'] = max(int(params['colors'] * 0.6), 64)
            params['fps'] = max(int(params['fps'] * 0.75), 7)  # Keep FPS more reasonable
        elif size_ratio < 0.7:
            # Quite large - moderate reduction
            reduction_factor = 0.8
            params['colors'] = max(int(params['colors'] * 0.75), 64)
            params['fps'] = max(int(params['fps'] * 0.85), 8)  # Less aggressive FPS reduction
        else:
            # Slightly large - gentle reduction
            reduction_factor = 0.9
            params['colors'] = max(int(params['colors'] * 0.85), 64)
            params['fps'] = max(int(params['fps'] * 0.9), 9)  # Keep higher minimum FPS
        
        # Reduce resolution if still too large, preserving aspect ratio
        if size_ratio < 0.8:
            new_width = int(params['width'] * reduction_factor)
            new_height = int(new_width / original_aspect_ratio)  # Calculate height from width using original ratio
            
            # Ensure even dimensions
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            # Ensure minimum dimensions while preserving aspect ratio
            min_width = 160
            min_height = int(min_width / original_aspect_ratio)
            
            params['width'] = max(new_width, min_width)
            params['height'] = max(new_height, min_height)
        
        # Increase lossy compression more gradually
        if size_ratio < 0.6:
            params['lossy'] = min(params['lossy'] + 25, 140)  # Less aggressive lossy compression
        else:
            params['lossy'] = min(params['lossy'] + 15, 100)
        
        return params

    def _calculate_quality_score(self, params: Dict[str, Any], actual_size_mb: float, max_size_mb: float, duration_bonus: float = 1.0) -> float:
        """Calculate a quality score based on parameters and size efficiency - optimized for maximum quality"""
        
        # Base quality factors
        color_score = params['colors'] / 256.0  # 0-1
        fps_score = min(params['fps'] / 30.0, 1.0)  # 0-1, cap at 30fps
        resolution_score = min((params['width'] * params['height']) / (800 * 800), 1.0)  # 0-1, cap at 800x800
        
        # Size efficiency - heavily reward using more of the available size
        size_ratio = actual_size_mb / max_size_mb
        if size_ratio > 0.95:
            # Maximum reward for using 95%+ of the limit
            size_efficiency = 1.0
        elif size_ratio > 0.8:
            # High reward for using 80%+ of the limit
            size_efficiency = 0.9 + (size_ratio - 0.8) * 0.5  # 0.9 to 1.0
        elif size_ratio > 0.6:
            # Moderate reward for using 60%+ of the limit
            size_efficiency = 0.7 + (size_ratio - 0.6) * 1.0  # 0.7 to 0.9
        else:
            # Penalize using less than 60% of the limit (wasted space)
            size_efficiency = size_ratio * 0.7  # 0 to 0.42
        
        # Lossy compression penalty
        lossy_penalty = 1.0 - (params['lossy'] / 200.0)  # 0-1, higher lossy = lower score
        
        # Dithering quality
        dither_score = 1.0 if params['dither'] == 'floyd_steinberg' else 0.7 if params['dither'] == 'bayer' else 0.5
        
        # Duration bonus
        duration_score_bonus = 0.0
        if duration_bonus > 0:
            # Reward for achieving a good size with a longer duration
            if actual_size_mb < max_size_mb * 0.8: # If size is very good for duration
                duration_score_bonus = 0.5
            elif actual_size_mb < max_size_mb * 0.9: # If size is good for duration
                duration_score_bonus = 0.3
            elif actual_size_mb < max_size_mb * 0.95: # If size is slightly above average for duration
                duration_score_bonus = 0.1
        
        # Calculate weighted quality score - prioritize size efficiency more
        quality_score = (
            color_score * 0.20 +
            fps_score * 0.20 +
            resolution_score * 0.20 +
            size_efficiency * 0.30 +  # Increased weight for size efficiency
            lossy_penalty * 0.05 +
            dither_score * 0.05 +
            duration_score_bonus # Add duration bonus
        )
        
        return quality_score
    
    def _get_gif_results(self, input_video: str, output_gif: str, video_info: Dict[str, Any],
                        gif_params: Dict[str, Any], duration: float) -> Dict[str, Any]:
        """Generate GIF creation results"""
        
        gif_size = os.path.getsize(output_gif)
        
        # Estimate frame count
        estimated_frames = int(duration * gif_params['fps'])
        
        return {
            'success': True,  # Add success flag
            'input_video': input_video,
            'output_gif': output_gif,
            'file_size_mb': gif_size / (1024 * 1024),
            'duration_seconds': duration,
            'frame_count': estimated_frames,
            'fps': gif_params['fps'],
            'width': gif_params['width'],
            'height': gif_params['height'],
            'colors': gif_params['colors'],
            'compression_ratio': ((video_info['size_bytes'] - gif_size) / video_info['size_bytes']) * 100,
            'video_info': video_info
        }
    
    def _get_gif_optimization_results(self, input_gif: str, output_gif: str, method: str) -> Dict[str, Any]:
        """Generate GIF optimization results"""
        
        original_size = os.path.getsize(input_gif)
        optimized_size = os.path.getsize(output_gif)
        
        return {
            'success': True,  # Add success flag
            'input_gif': input_gif,
            'output_gif': output_gif,
            'method': method,
            'original_size_mb': original_size / (1024 * 1024),
            'optimized_size_mb': optimized_size / (1024 * 1024),
            'compression_ratio': ((original_size - optimized_size) / original_size) * 100,
            'space_saved_mb': (original_size - optimized_size) / (1024 * 1024)
        } 

    def _should_split_video(self, duration: float, video_info: Dict[str, Any], max_size_mb: float) -> bool:
        """
        Determine if a video should be split into segments based on estimated GIF file size
        
        Args:
            duration: Video duration in seconds
            video_info: Video metadata including complexity, resolution, etc.
            max_size_mb: Target size limit
            
        Returns:
            True if video should be split into segments
        """
        
        # Create optimal GIF parameters for size estimation using config
        estimation_config = self.config.get('gif_settings.segmentation.estimation', {})
        estimation_params = {
            'width': min(480, video_info.get('width', 480)),
            'height': min(480, video_info.get('height', 480)),
            'fps': estimation_config.get('default_fps', 15),
            'colors': estimation_config.get('default_colors', 256),
            'dither': 'floyd_steinberg',
            'lossy': estimation_config.get('default_lossy', 60)
        }
        
        # Preserve aspect ratio for estimation
        if video_info.get('width') and video_info.get('height'):
            original_width = video_info['width']
            original_height = video_info['height']
            original_aspect_ratio = original_width / original_height
            
            max_width = estimation_params['width']
            max_height = estimation_params['height']
            
            if original_aspect_ratio > 1:  # Landscape
                new_width = min(original_width, max_width)
                new_height = int(new_width / original_aspect_ratio)
                if new_height > max_height:
                    new_height = max_height
                    new_width = int(new_height * original_aspect_ratio)
            else:  # Portrait or square
                new_height = min(original_height, max_height)
                new_width = int(new_height * original_aspect_ratio)
                if new_width > max_width:
                    new_width = max_width
                    new_height = int(new_width / original_aspect_ratio)
            
            # Ensure even dimensions
            estimation_params['width'] = new_width - (new_width % 2)
            estimation_params['height'] = new_height - (new_height % 2)
        
        # Estimate file size with optimal compression
        estimated_size_mb = self._estimate_gif_size(estimation_params, duration, video_info)
        
        # Primary decision: split if estimated size exceeds limit significantly
        size_threshold_multiplier = self.config.get('gif_settings.segmentation.size_threshold_multiplier', 2.5)
        if estimated_size_mb > max_size_mb * size_threshold_multiplier:
            logger.info(f"Video splitting recommended: estimated size {estimated_size_mb:.1f}MB > "
                       f"{max_size_mb * size_threshold_multiplier:.1f}MB threshold")
            return True
        
        # Secondary decision: split if estimated size is close to limit but video has challenging characteristics
        if estimated_size_mb > max_size_mb * 1.5:  # Within 1.5x of limit
            complexity = video_info.get('complexity_score', 5.0)
            motion_level = video_info.get('motion_level', 'medium')
            
            # Split if high complexity or motion makes compression unpredictable
            if complexity >= 7 or motion_level == 'high':
                logger.info(f"Video splitting recommended: estimated size {estimated_size_mb:.1f}MB with "
                           f"challenging characteristics (complexity: {complexity:.1f}, motion: {motion_level})")
                return True
        
        # Fallback: still split extremely long videos regardless of estimated size
        fallback_duration_limit = self.config.get('gif_settings.segmentation.fallback_duration_limit', 120)
        if duration > fallback_duration_limit:
            logger.info(f"Video splitting recommended: duration {duration}s exceeds fallback limit {fallback_duration_limit}s")
            return True
        
        # Try with more aggressive compression to see if single GIF is feasible
        aggressive_params = estimation_params.copy()
        aggressive_params.update({
            'fps': estimation_config.get('aggressive_fps', 12),
            'colors': estimation_config.get('aggressive_colors', 192),
            'lossy': estimation_config.get('aggressive_lossy', 100)
        })
        
        aggressive_size_mb = self._estimate_gif_size(aggressive_params, duration, video_info)
        
        if aggressive_size_mb > max_size_mb * 1.2:  # Even aggressive compression won't work well
            logger.info(f"Video splitting recommended: even aggressive compression estimates {aggressive_size_mb:.1f}MB > "
                       f"{max_size_mb * 1.2:.1f}MB")
            return True
        
        logger.info(f"Single GIF recommended: estimated size {estimated_size_mb:.1f}MB (aggressive: {aggressive_size_mb:.1f}MB) "
                   f"within acceptable range for {max_size_mb}MB target")
        return False
    
    def _split_video_into_segments(self, input_video: str, output_base_path: str, 
                                  start_time: float, duration: float, 
                                  video_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Split a long/complex video into multiple high-quality GIF segments
        
        Args:
            input_video: Path to input video
            output_base_path: Base path for output (will create folder)
            start_time: Start time in video
            duration: Duration to process
            video_info: Video metadata
            
        Returns:
            Dictionary with results including segment paths and metadata
        """
        
        # Create temp folder for segments first
        base_name = os.path.splitext(os.path.basename(output_base_path))[0]
        temp_segments_folder = os.path.join(self.config.get_temp_dir(), f"{base_name}_segments_temp")
        
        os.makedirs(temp_segments_folder, exist_ok=True)
        logger.info(f"Created temp segments folder: {temp_segments_folder}")
        
        # Track all temporary files for cleanup
        temp_files_to_cleanup = []
        segments_created = []
        failed_segments = []
        
        try:
            # Calculate optimal segment duration
            segment_duration = self._calculate_optimal_segment_duration(duration, video_info)
            num_segments = max(1, int(duration / segment_duration))
            
            # Validate segmentation parameters
            if segment_duration < 5.0:
                logger.warning(f"Very short segment duration calculated: {segment_duration:.1f}s - adjusting to minimum 15s")
                segment_duration = 15.0
                num_segments = max(1, int(duration / segment_duration))
            
            if num_segments > 10:
                logger.warning(f"Too many segments calculated: {num_segments} - limiting to 10 segments")
                num_segments = 10
                segment_duration = duration / num_segments
            
            logger.info(f"Splitting {duration:.1f}s video into {num_segments} segments of {segment_duration:.1f}s each")
            
            for i in range(num_segments):
                # Check for shutdown before each segment
                if self.shutdown_requested:
                    logger.info("Shutdown requested during video segmentation")
                    break
                
                segment_start = start_time + (i * segment_duration)
                # Calculate remaining duration from the current segment start
                remaining_duration = duration - (i * segment_duration)
                actual_segment_duration = min(segment_duration, remaining_duration)
                
                if actual_segment_duration < 1.0:  # Skip very short segments
                    continue
                
                segment_name = f"001_{base_name}_part{i+1:02d}.gif"
                temp_segment_path = os.path.join(temp_segments_folder, segment_name)
                temp_files_to_cleanup.append(temp_segment_path)  # Track for cleanup
                
                try:
                    logger.info(f"Creating segment {i+1}/{num_segments}: {segment_start:.1f}s-{segment_start + actual_segment_duration:.1f}s")
                    
                    # Use optimized high-quality segment creation
                    segment_result = self._create_high_quality_segment(
                        input_video, temp_segment_path, segment_start, actual_segment_duration, video_info
                    )
                    
                    if segment_result['success']:
                        # Set proper metadata and thumbnail for Windows
                        segment_metadata = {
                            'name': segment_name,
                            'duration': actual_segment_duration,
                            'size_mb': segment_result.get('size_mb', segment_result.get('file_size_mb', 0))
                        }
                        self._set_gif_metadata_and_thumbnail(temp_segment_path, video_info, segment_metadata)
                        
                        segments_created.append({
                            'temp_path': temp_segment_path,  # Store temp path for later movement
                            'name': segment_name,
                            'start_time': segment_start,
                            'duration': actual_segment_duration,
                            'size_mb': segment_result.get('size_mb', segment_result.get('file_size_mb', 0)),  # Handle both possible keys
                            'frame_count': segment_result.get('frame_count', 0),
                            'method': segment_result.get('method', 'Segment Optimization'),
                            'quality_score': segment_result.get('quality_score', 0),
                            'optimization_type': segment_result.get('optimization_type', 'full_pipeline_segment')
                        })
                        size_mb = segment_result.get('size_mb', segment_result.get('file_size_mb', 0))
                        method = segment_result.get('method', 'optimization')
                        logger.info(f"SUCCESS: Segment {i+1} created: {size_mb:.2f}MB using {method}")
                    else:
                        failed_segments.append(i+1)
                        logger.warning(f"Segment {i+1} failed: {segment_result.get('error', 'Unknown error')}")
                
                except Exception as e:
                    logger.error(f"Segment {i+1} failed with exception: {e}")
                    failed_segments.append(i+1)
            
            # Create summary file in temp folder
            if segments_created:  # Only create summary if we have segments
                self._create_segments_summary(temp_segments_folder, segments_created, base_name, duration)
            
            total_size_mb = sum(seg['size_mb'] for seg in segments_created)
            
            # If we have successful segments, return them (cleanup will happen later during move)
            if segments_created:
                return {
                    'success': True,
                    'temp_segments_folder': temp_segments_folder,  # Track temp location
                    'segments_created': len(segments_created),
                    'segments_failed': len(failed_segments),
                    'total_size_mb': total_size_mb,
                    'segments': segments_created,
                    'method': 'Video Segmentation',
                    'base_name': base_name,  # Store base name for output folder creation
                    'original_output_path': output_base_path,  # Store intended output path
                    'temp_files_to_cleanup': temp_files_to_cleanup  # Track files for cleanup
                }
            else:
                # No segments created - this is a failure case
                return {
                    'success': False,
                    'error': f'No segments were successfully created. {len(failed_segments)} segments failed.',
                    'segments_failed': len(failed_segments),
                    'temp_segments_folder': temp_segments_folder,
                    'temp_files_to_cleanup': temp_files_to_cleanup
                }
                
        except Exception as e:
            logger.error(f"Critical error during video segmentation: {e}")
            return {
                'success': False,
                'error': f'Video segmentation failed: {str(e)}',
                'temp_segments_folder': temp_segments_folder,
                'temp_files_to_cleanup': temp_files_to_cleanup
            }
        
        finally:
            # If we're here due to failure or shutdown, clean up temp files immediately
            if not segments_created or self.shutdown_requested:
                logger.info(f"Cleaning up temporary files due to failure or shutdown")
                self._cleanup_temp_files(temp_files_to_cleanup)
                # Also try to remove the temp folder if it's empty
                try:
                    if os.path.exists(temp_segments_folder) and not os.listdir(temp_segments_folder):
                        os.rmdir(temp_segments_folder)
                        logger.info(f"Removed empty temp segments folder: {temp_segments_folder}")
                except Exception as cleanup_e:
                    logger.warning(f"Could not remove temp segments folder: {cleanup_e}")
    
    def _cleanup_temp_files(self, temp_files: List[str]):
        """Clean up a list of temporary files"""
        if not temp_files:
            return
            
        logger.info(f"Cleaning up {len(temp_files)} temporary files")
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up {temp_file}: {e}")
    
    def _cleanup_temp_folder(self, temp_folder: str):
        """Clean up a temporary folder and all its contents"""
        if not temp_folder or not os.path.exists(temp_folder):
            return
            
        try:
            logger.info(f"Cleaning up temp folder: {temp_folder}")
            shutil.rmtree(temp_folder)
            logger.info(f"Successfully removed temp folder: {temp_folder}")
        except Exception as e:
            logger.warning(f"Failed to remove temp folder {temp_folder}: {e}")
            # Try to clean individual files if folder removal fails
            try:
                for file_path in os.listdir(temp_folder):
                    full_path = os.path.join(temp_folder, file_path)
                    if os.path.isfile(full_path):
                        os.remove(full_path)
                        logger.debug(f"Removed file: {full_path}")
                os.rmdir(temp_folder)
                logger.info(f"Successfully cleaned temp folder after file-by-file removal")
            except Exception as cleanup_e:
                logger.error(f"Failed to clean temp folder even with file-by-file approach: {cleanup_e}")
    
    def _calculate_optimal_segment_duration(self, total_duration: float, video_info: Dict[str, Any]) -> float:
        """Calculate optimal duration for each segment based on content characteristics and config"""
        
        complexity = video_info.get('complexity_score', 5.0)
        motion_level = video_info.get('motion_level', 'medium')
        
        # Get segment duration settings from config
        config_base = self.config.get('gif_settings.segmentation.base_durations', {})
        short_video_max = config_base.get('short_video_max', 40)
        medium_video_max = config_base.get('medium_video_max', 80)
        long_video_max = config_base.get('long_video_max', 120)
        
        short_segment_duration = config_base.get('short_segment_duration', 20)
        medium_segment_duration = config_base.get('medium_segment_duration', 25)
        long_segment_duration = config_base.get('long_segment_duration', 30)
        very_long_segment_duration = config_base.get('very_long_segment_duration', 35)
        
        # Ultra-aggressive segmentation for much better quality
        # Prioritize quality over number of segments - create many small segments
        if total_duration <= 20:  # Short videos
            base_duration = 8   # Very short segments
        elif total_duration <= 40:  # Medium videos  
            base_duration = 10  # Short segments
        elif total_duration <= 80:  # Long videos
            base_duration = 12  # Still short
        else:  # Very long videos
            base_duration = 15  # Maximum segment length
        
        # Ultra-aggressive segmentation adjustments - prioritize quality over everything else
        if complexity >= 7 or motion_level == 'high':
            # Ultra-aggressive reduction for complex content
            base_duration *= 0.6  # Very aggressive
        elif complexity >= 5:
            # Very aggressive reduction for medium-high complexity
            base_duration *= 0.7
        elif complexity >= 3:
            # Significant reduction for moderate complexity
            base_duration *= 0.8
        # All content gets shorter segments to ensure quality
        
        # Get bounds from config - ultra-permissive for quality
        min_duration = self.config.get('gif_settings.segmentation.min_segment_duration', 5)  # Allow ultra-short segments
        max_duration = self.config.get('gif_settings.segmentation.max_segment_duration', 18)  # Much lower max for better quality
        
        # Apply bounds from config
        final_duration = max(min_duration, min(max_duration, base_duration))
        
        logger.debug(f"Segment duration calculation: base={base_duration:.1f}s, final={final_duration:.1f}s (bounds: {min_duration}-{max_duration}s)")
        
        return final_duration
    
    def _create_high_quality_segment(self, input_video: str, output_path: str, 
                                   start_time: float, duration: float, 
                                   video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create a high-quality GIF from a video segment using optimized parameters for segments"""
        
        # Validate duration parameter to catch potential bugs
        if duration <= 0 or duration > 300:  # Sanity check: 0-300 seconds
            logger.error(f"Invalid segment duration: {duration}s - this should be between 0 and 300 seconds")
            return {'success': False, 'error': f'Invalid duration: {duration}s'}
        
        if duration < 5.0:
            logger.warning(f"Short segment duration: {duration:.2f}s - adjusting parameters for short content")
        
        try:
            # Create initial parameters optimized for longer segments using config - IMPROVED FPS
            quality_scaling_enabled = self.config.get('gif_settings.segmentation.quality_scaling.enabled', True)
            # Use original video FPS as base, with config as fallback
            original_fps = video_info.get('fps', self.config.get('gif_settings.fps', 20))
            base_fps = min(original_fps, 30)  # Cap at 30 FPS for reasonable file sizes
            
            # Start with more reasonable parameters and let the quality levels handle optimization
            # The size estimation is rough and we shouldn't make drastic cuts upfront
            width = min(480, video_info.get('width', 480))
            height = min(480, video_info.get('height', 480))
            
            # Be more conservative with initial FPS - start closer to original and let optimization reduce it gradually
            if duration > 25:  # Very long segments - moderate reduction
                initial_fps = max(15, int(base_fps * 0.6))  # More conservative: 30*0.6 = 18fps
                initial_colors = 128  # Start with reasonable colors
                initial_lossy = 60  # Moderate lossy compression
                logger.info(f"Using moderate compression for long segment ({duration:.1f}s)")
            elif duration > 15:  # Long segments - slight reduction
                initial_fps = max(18, int(base_fps * 0.7))  # Conservative: 30*0.7 = 21fps  
                initial_colors = 160  # Good color count
                initial_lossy = 40  # Light lossy compression
                logger.info(f"Using light compression for medium segment ({duration:.1f}s)")
            else:  # Short segments - minimal reduction
                initial_fps = max(20, int(base_fps * 0.8))  # Very conservative: 30*0.8 = 24fps
                initial_colors = 200  # High color count
                initial_lossy = 30  # Very light lossy compression
                logger.info(f"Using minimal compression for short segment ({duration:.1f}s)")
            
            segment_params = {
                'width': width,  # Use calculated width (may be reduced for large content)
                'height': height,  # Use calculated height (may be reduced for large content)
                'fps': initial_fps,
                'colors': initial_colors,
                'dither': 'floyd_steinberg',
                'lossy': initial_lossy,
                'max_size_mb': 10,  # Enforce 10MB limit per segment [[memory:4419373]]
                'max_duration': duration  # Use the segment duration
            }
            
            # Apply content-aware adjustments
            segment_params = self._apply_content_aware_adjustments(segment_params, video_info)
            
            # Preserve aspect ratio
            if video_info.get('width') and video_info.get('height'):
                original_width = video_info['width']
                original_height = video_info['height']
                original_aspect_ratio = original_width / original_height
                
                # Calculate new dimensions while preserving aspect ratio
                max_width = segment_params['width']
                max_height = segment_params['height']
                
                if original_aspect_ratio > 1:  # Landscape
                    new_width = min(original_width, max_width)
                    new_height = int(new_width / original_aspect_ratio)
                    if new_height > max_height:
                        new_height = max_height
                        new_width = int(new_height * original_aspect_ratio)
                else:  # Portrait or square
                    new_height = min(original_height, max_height)
                    new_width = int(new_height * original_aspect_ratio)
                    if new_width > max_width:
                        new_width = max_width
                        new_height = int(new_width / original_aspect_ratio)
                
                # Ensure even dimensions
                segment_params['width'] = new_width - (new_width % 2)
                segment_params['height'] = new_height - (new_height % 2)
            
            logger.info(f"Creating segment with optimization: {segment_params['width']}x{segment_params['height']}, "
                       f"{duration:.1f}s @ {segment_params['fps']}fps")
            
            # Use direct GIF creation for segments to avoid ultra-aggressive fallbacks
            result = self._create_segment_gif_direct(
                input_video, output_path, segment_params, start_time, duration, video_info
            )
            
            # Add segment-specific metadata
            if result.get('success'):
                result['segment_duration'] = duration
                result['segment_start_time'] = start_time
                result['optimization_type'] = 'full_pipeline_segment'
                
                # Ensure consistent key names for segment processing
                if 'file_size_mb' in result and 'size_mb' not in result:
                    result['size_mb'] = result['file_size_mb']
                if 'optimization_method' in result and 'method' not in result:
                    result['method'] = result['optimization_method']
                
                logger.info(f"Segment optimization successful: {result.get('size_mb', 0):.2f}MB using {result.get('method', 'unknown')} method")
            
            return result
            
        except Exception as e:
            logger.error(f"Error creating high-quality segment: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _create_segment_gif_direct(self, input_video: str, output_path: str, 
                                 segment_params: Dict[str, Any], start_time: float, 
                                 duration: float, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Create GIF segment with direct optimization, avoiding ultra-aggressive fallbacks"""
        
        max_size_mb = segment_params['max_size_mb']
        temp_files_to_cleanup = []
        
        try:
            # Try progressive quality levels for segments - more aggressive for size compliance
            quality_levels = [
                # Level 1: High quality - minimal reduction
                {
                    'colors': max(160, int(segment_params['colors'] * 0.85)),
                    'fps': max(18, int(segment_params['fps'] * 0.95)),  # Very conservative
                    'lossy': min(60, segment_params['lossy'] + 10),
                    'frame_skip': 1,
                    'method': 'High Quality Segment'
                },
                # Level 2: Good quality - light reduction
                {
                    'colors': max(128, int(segment_params['colors'] * 0.75)),
                    'fps': max(15, int(segment_params['fps'] * 0.85)),  # Still conservative
                    'lossy': min(80, segment_params['lossy'] + 20),
                    'frame_skip': 1,
                    'method': 'Good Quality Segment'
                },
                # Level 3: Balanced quality - moderate reduction
                {
                    'colors': max(96, int(segment_params['colors'] * 0.6)),
                    'fps': max(12, int(segment_params['fps'] * 0.7)),  # More reasonable
                    'lossy': min(100, segment_params['lossy'] + 30),
                    'frame_skip': 2,
                    'method': 'Balanced Quality Segment'
                },
                # Level 4: Compact quality - stronger reduction
                {
                    'colors': max(64, int(segment_params['colors'] * 0.45)),
                    'fps': max(10, int(segment_params['fps'] * 0.55)),  # Still reasonable
                    'lossy': min(120, segment_params['lossy'] + 40),
                    'frame_skip': 2,
                    'method': 'Compact Quality Segment'
                },
                # Level 5: Compressed quality - aggressive but not extreme
                {
                    'colors': max(48, int(segment_params['colors'] * 0.35)),
                    'fps': max(8, int(segment_params['fps'] * 0.4)),  # More aggressive
                    'lossy': min(140, segment_params['lossy'] + 50),
                    'frame_skip': 3,
                    'method': 'Compressed Quality Segment'
                },
                # Level 6: Ultra-compressed - very aggressive (fallback)
                {
                    'colors': max(32, int(segment_params['colors'] * 0.25)),
                    'fps': max(6, int(segment_params['fps'] * 0.3)),
                    'lossy': min(160, segment_params['lossy'] + 60),
                    'frame_skip': 4,
                    'method': 'Ultra-Compressed Segment'
                }
            ]
            
            for level_idx, quality_level in enumerate(quality_levels):
                temp_gif = os.path.join(self.temp_dir, f"segment_level_{level_idx}.gif")
                temp_files_to_cleanup.append(temp_gif)
                
                try:
                    # Create parameters for this quality level
                    level_params = segment_params.copy()
                    level_params.update(quality_level)
                    
                    # Create GIF with this quality level
                    self._create_gif_opencv_pil(input_video, temp_gif, level_params, start_time, duration)
                    
                    if os.path.exists(temp_gif):
                        file_size_mb = os.path.getsize(temp_gif) / (1024 * 1024)
                        
                        logger.info(f"Quality level {level_idx + 1} created GIF: {file_size_mb:.3f}MB "
                                   f"(target: <={max_size_mb}MB, minimum: >0.01MB)")
                        
                        if file_size_mb <= max_size_mb and file_size_mb > 0.01:  # At least 10KB
                            # Success! Move to final location
                            shutil.move(temp_gif, output_path)
                            
                            # Calculate actual FPS from the GIF file
                            try:
                                from PIL import Image
                                with Image.open(output_path) as gif:
                                    actual_frame_count = gif.n_frames
                                    frame_delay_ms = gif.info.get('duration', 100)  # Default 100ms if not found
                                    actual_fps = 1000 / frame_delay_ms if frame_delay_ms > 0 else 0
                            except:
                                # Fallback calculation
                                actual_frame_count = int(duration * level_params['fps'])
                                actual_fps = level_params['fps']
                            
                            # Clean up remaining temp files
                            self._cleanup_temp_files(temp_files_to_cleanup)
                            
                            logger.info(f"Segment created successfully with {quality_level['method']}: "
                                       f"{file_size_mb:.2f}MB, {actual_frame_count} frames @ {actual_fps:.1f}fps")
                            
                            return {
                                'success': True,
                                'size_mb': file_size_mb,
                                'file_size_mb': file_size_mb,  # Compatibility
                                'frame_count': actual_frame_count,
                                'fps': actual_fps,
                                'method': quality_level['method'],
                                'quality_score': self._calculate_quality_score(level_params, file_size_mb, max_size_mb),
                                'optimization_type': 'segment_direct'
                            }
                        else:
                            if file_size_mb > max_size_mb:
                                logger.warning(f"Quality level {level_idx + 1} too large: {file_size_mb:.3f}MB > {max_size_mb}MB")
                            else:
                                logger.warning(f"Quality level {level_idx + 1} too small: {file_size_mb:.3f}MB ≤ 0.01MB")
                
                except Exception as e:
                    logger.warning(f"Quality level {level_idx + 1} failed with error: {e}")
                    continue
            
            # If all quality levels failed, return error
            logger.error(f"All segment quality levels failed for {duration:.1f}s segment")
            return {
                'success': False,
                'error': 'All segment quality levels failed'
            }
            
        except Exception as e:
            logger.error(f"Segment creation failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
        finally:
            # Clean up any remaining temp files
            self._cleanup_temp_files(temp_files_to_cleanup)
    
    def _create_segments_summary(self, segments_folder: str, segments: List[Dict[str, Any]], 
                               base_name: str, total_duration: float):
        """Create a summary file with information about all segments"""
        
        summary_path = os.path.join(segments_folder, f"{base_name}_segments_info.txt")
        
        try:
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(f"GIF Segments Summary\n")
                f.write(f"==================\n\n")
                f.write(f"Original video duration: {total_duration:.1f} seconds\n")
                f.write(f"Number of segments: {len(segments)}\n")
                f.write(f"Total size: {sum(seg['size_mb'] for seg in segments):.2f} MB\n")
                f.write(f"All segments optimized to stay under 10MB for platform compatibility\n\n")
                
                f.write("Segments:\n")
                f.write("---------\n")
                
                for i, segment in enumerate(segments, 1):
                    f.write(f"{i:2d}. {segment['name']}\n")
                    f.write(f"    Time: {segment['start_time']:.1f}s - {segment['start_time'] + segment['duration']:.1f}s "
                           f"(duration: {segment['duration']:.1f}s)\n")
                    f.write(f"    Size: {segment['size_mb']:.2f} MB\n")
                    f.write(f"    Frames: {segment['frame_count']}\n")
                    f.write(f"    Optimization attempts: {segment.get('optimization_attempts', 1)}\n\n")
                
                f.write("Usage:\n")
                f.write("------\n")
                f.write("• Upload segments individually to Discord/social media (each under 10MB)\n")
                f.write("• Use in sequence to show the complete video content\n")
                f.write("• Each segment is optimized for quality while respecting size limits\n")
                
        except Exception as e:
            logger.error(f"Failed to create segments summary: {e}")

    def _set_gif_metadata_and_thumbnail(self, gif_path: str, video_info: Dict[str, Any], 
                                       segment_info: Dict[str, Any] = None) -> None:
        """Set proper metadata and ensure Windows thumbnail generation for GIF files"""
        
        try:
            # Force Windows to recognize the file and generate thumbnail
            import subprocess
            import os
            
            # Set file attributes to ensure Windows generates thumbnails
            # Remove any existing attributes that might interfere
            subprocess.run(['attrib', '-H', '-S', '-R', gif_path], capture_output=True, check=False)
            
            # Set normal file attribute to trigger thumbnail generation
            subprocess.run(['attrib', '+A', gif_path], capture_output=True, check=False)
            
            # Access the file to trigger Windows thumbnail cache
            try:
                stat_result = os.stat(gif_path)
                # Touch the file modification time to force thumbnail regeneration
                os.utime(gif_path, None)
                logger.debug(f"Set Windows thumbnail attributes for: {os.path.basename(gif_path)}")
            except Exception as stat_e:
                logger.debug(f"Could not access file for thumbnail: {stat_e}")
                
        except Exception as e:
            logger.debug(f"Could not set Windows thumbnail attributes: {e}")
            # Not critical, continue without thumbnail optimization