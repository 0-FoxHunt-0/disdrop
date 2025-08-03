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
import json
import re

from .config_manager import ConfigManager
from .ffmpeg_utils import FFmpegUtils
from .gif_optimizer_advanced import AdvancedGifOptimizer

logger = logging.getLogger(__name__)

class GifGenerator:
    """Advanced GIF generator with optimization and platform-specific settings"""
    
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.ffmpeg_utils = FFmpegUtils()
        self.optimizer = AdvancedGifOptimizer(config_manager)
        
        # Platform-specific settings
        self.platform_settings = {
            'discord': {
                'max_size_mb': 8.0,
                'max_duration': 10.0,
                'fps': 15,
                'scale': 480
            },
            'twitter': {
                'max_size_mb': 5.0,
                'max_duration': 2.5,
                'fps': 12,
                'scale': 400
            },
            'slack': {
                'max_size_mb': 10.0,
                'max_duration': 15.0,
                'fps': 15,
                'scale': 480
            },
            'telegram': {
                'max_size_mb': 50.0,  # Telegram allows larger files
                'max_duration': 30.0,
                'fps': 15,
                'scale': 480
            },
            'reddit': {
                'max_size_mb': 100.0,  # Reddit allows very large files
                'max_duration': 60.0,
                'fps': 15,
                'scale': 480
            }
        }
    
    def create_gif(self, input_video: str, output_path: str, platform: str = None,
                  max_size_mb: int = None, start_time: float = 0, 
                  duration: float = None, disable_segmentation: bool = False) -> Dict[str, Any]:
        """
        Create optimized GIF from video with iterative quality optimization
        
        Args:
            input_video: Path to input video file
            output_path: Path for output GIF
            platform: Target platform (twitter, discord, slack, etc.)
            max_size_mb: Maximum file size in MB
            start_time: Start time in seconds (default: 0)
            duration: Duration in seconds (default: use platform/config limit)
            disable_segmentation: If True, prevents the GIF generator from segmenting the video
            
        Returns:
            Dict with success status and metadata
        """
        try:
            # Validate input
            if not os.path.exists(input_video):
                return {'success': False, 'error': f'Input video not found: {input_video}'}
            
            # Get platform settings
            settings = self._get_platform_settings(platform, max_size_mb)
            
            # Get video info
            video_info = self.ffmpeg_utils.get_video_info(input_video)
            if not video_info:
                return {'success': False, 'error': 'Could not get video information'}
            
            # Determine duration
            if duration is None:
                duration = min(settings['max_duration'], video_info.get('duration', settings['max_duration']))
            
            # Check if video should be split (unless segmentation is disabled)
            if not disable_segmentation and self._should_split_video(input_video, duration, settings):
                logger.info(f"Video duration ({duration:.1f}s) exceeds platform limit ({settings['max_duration']:.1f}s), splitting into segments")
                return self._create_segmented_gifs(input_video, output_path, settings, start_time, duration)
            
            # Create single GIF
            logger.info(f"Creating single GIF: {input_video} -> {output_path}")
            return self._create_single_gif(input_video, output_path, settings, start_time, duration)
            
        except Exception as e:
            logger.error(f"Error creating GIF: {e}")
            return {'success': False, 'error': str(e)}
    
    def _get_platform_settings(self, platform: str, max_size_mb: float) -> Dict[str, Any]:
        """Get platform-specific settings"""
        if platform and platform.lower() in self.platform_settings:
            settings = self.platform_settings[platform.lower()].copy()
        else:
            # Default settings
            settings = {
                'max_size_mb': 10.0,
                'max_duration': 15.0,
                'fps': 15,
                'scale': 480
            }
        
        # Override with provided max_size_mb if specified
        if max_size_mb is not None:
            settings['max_size_mb'] = max_size_mb
        
        return settings
    
    def _should_split_video(self, input_video: str, duration: float, settings: Dict[str, Any]) -> bool:
        """Determine if video should be split into segments"""
        # Check if this is already a segmented video (from video segmentation)
        # If so, disable GIF segmentation entirely
        if self._is_segmented_video(input_video):
            logger.info(f"Video appears to be from segmentation, disabling GIF segmentation: {input_video}")
            return False
        
        # Split if duration exceeds platform limit
        if duration > settings['max_duration']:
            return True
        
        # Split if estimated file size would be too large
        estimated_size = self._estimate_gif_size(input_video, duration, settings)
        if estimated_size > settings['max_size_mb'] * 0.8:  # 80% threshold
            return True
        
        return False
    
    def _is_segmented_video(self, input_video: str) -> bool:
        """Check if the video is from a segmented video folder"""
        try:
            # Check if the video is in a segments folder
            video_path = Path(input_video)
            parent_dir = video_path.parent
            
            # Check if parent directory name contains '_segments'
            if '_segments' in parent_dir.name:
                return True
            
            # Check if the video filename contains segment indicators
            video_name = video_path.stem.lower()
            if any(indicator in video_name for indicator in ['segment', 'part', 'split']):
                return True
            
            # Check if the video is in a segments folder (case insensitive)
            if 'segments' in parent_dir.name.lower():
                return True
            
            return False
        except Exception as e:
            logger.warning(f"Error checking if video is segmented: {e}")
            return False
    
    def _estimate_gif_size(self, input_video: str, duration: float, settings: Dict[str, Any]) -> float:
        """Estimate GIF file size based on video characteristics"""
        try:
            # Get video resolution
            resolution = self.ffmpeg_utils.get_video_resolution(input_video)
            if not resolution:
                return settings['max_size_mb']  # Conservative estimate
            
            width, height = resolution
            
            # Scale down if needed
            if width > settings['scale']:
                scale_factor = settings['scale'] / width
                width = int(width * scale_factor)
                height = int(height * scale_factor)
            
            # Rough size estimation (very approximate)
            pixels = width * height
            fps = settings['fps']
            frames = duration * fps
            
            # Rough size calculation (this is very approximate)
            estimated_size_mb = (pixels * frames * 0.1) / (1024 * 1024)  # Rough compression factor
            
            return estimated_size_mb
            
        except Exception as e:
            logger.warning(f"Could not estimate GIF size: {e}")
            return settings['max_size_mb']  # Conservative estimate
    
    def _create_single_gif(self, input_video: str, output_path: str, settings: Dict[str, Any], 
                          start_time: float, duration: float) -> Dict[str, Any]:
        """Create a single optimized GIF"""
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_path)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Generate palette first
            palette_path = self._generate_palette(input_video, settings, start_time, duration)
            if not palette_path:
                return {'success': False, 'error': 'Failed to generate palette'}
            
            # Create GIF with palette
            success = self._create_gif_with_palette(input_video, output_path, palette_path, settings, start_time, duration)
            
            # Clean up palette
            if os.path.exists(palette_path):
                os.unlink(palette_path)
            
            if success:
                # Get final file size
                if os.path.exists(output_path):
                    size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    
                    # Optimize if needed
                    if size_mb > settings['max_size_mb']:
                        logger.info(f"GIF size ({size_mb:.2f}MB) exceeds limit ({settings['max_size_mb']:.2f}MB), optimizing...")
                        optimized = self.optimizer.optimize_gif(output_path, settings['max_size_mb'])
                        if optimized:
                            size_mb = os.path.getsize(output_path) / (1024 * 1024)
                    
                    return {
                        'success': True,
                        'size_mb': size_mb,
                        'duration': duration,
                        'fps': settings['fps'],
                        'scale': settings['scale']
                    }
                else:
                    return {'success': False, 'error': 'GIF file not created'}
            else:
                return {'success': False, 'error': 'Failed to create GIF'}
                
        except Exception as e:
            logger.error(f"Error creating single GIF: {e}")
            return {'success': False, 'error': str(e)}
    
    def _generate_palette(self, input_video: str, settings: Dict[str, Any], 
                         start_time: float, duration: float) -> Optional[str]:
        """Generate optimized palette for GIF creation"""
        try:
            # Create temporary palette file with unique name
            palette_path = tempfile.mktemp(suffix='.png')
            
            # Build FFmpeg command for palette generation
            # Use -frames:v 1 to ensure only one frame is output
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', input_video,
                '-vf', f'fps={settings["fps"]},scale={settings["scale"]}:-1:flags=lanczos,palettegen=max_colors=256:stats_mode=single',
                '-frames:v', '1',
                '-f', 'image2',
                palette_path
            ]
            
            logger.debug(f"Generating palette: {' '.join(cmd)}")
            
            # Run FFmpeg
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(palette_path):
                return palette_path
            else:
                logger.error(f"Palette generation failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error generating palette: {e}")
            return None
    
    def _create_gif_with_palette(self, input_video: str, output_path: str, palette_path: str,
                                settings: Dict[str, Any], start_time: float, duration: float) -> bool:
        """Create GIF using generated palette"""
        try:
            # Build FFmpeg command for GIF creation
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', input_video,
                '-i', palette_path,
                '-lavfi', f'fps={settings["fps"]},scale={settings["scale"]}:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=5:diff_mode=rectangle',
                '-f', 'gif',
                output_path
            ]
            
            logger.debug(f"Creating GIF: {' '.join(cmd)}")
            
            # Run FFmpeg
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            
            if result.returncode == 0 and os.path.exists(output_path):
                return True
            else:
                logger.error(f"GIF creation failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error creating GIF with palette: {e}")
            return False
    
    def _create_segmented_gifs(self, input_video: str, output_path: str, settings: Dict[str, Any],
                              start_time: float, duration: float) -> Dict[str, Any]:
        """Create multiple GIF segments from long video"""
        try:
            # Calculate segment duration
            segment_duration = min(settings['max_duration'], duration / 2)
            num_segments = max(1, int(duration / segment_duration))
            
            # Create output directory for segments
            output_dir = os.path.dirname(output_path)
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            segments_dir = os.path.join(output_dir, f"{base_name}_segments")
            
            if not os.path.exists(segments_dir):
                os.makedirs(segments_dir)
            
            successful_segments = 0
            total_size = 0
            
            for i in range(num_segments):
                segment_start = start_time + (i * segment_duration)
                segment_end = min(start_time + duration, segment_start + segment_duration)
                segment_duration_actual = segment_end - segment_start
                
                if segment_duration_actual <= 0:
                    break
                
                # Create segment GIF
                segment_name = f"{base_name}_segment_{i+1:02d}.gif"
                segment_path = os.path.join(segments_dir, segment_name)
                
                result = self._create_single_gif(input_video, segment_path, settings, segment_start, segment_duration_actual)
                
                if result.get('success', False):
                    successful_segments += 1
                    total_size += result.get('size_mb', 0)
                    logger.info(f"Created segment {i+1}/{num_segments}: {segment_name} ({result.get('size_mb', 0):.2f}MB)")
                else:
                    logger.warning(f"Failed to create segment {i+1}/{num_segments}: {result.get('error', 'Unknown error')}")
            
            if successful_segments > 0:
                return {
                    'success': True,
                    'segments_created': successful_segments,
                    'total_size_mb': total_size,
                    'segments_directory': segments_dir,
                    'method': 'segmentation'
                }
            else:
                return {'success': False, 'error': 'Failed to create any segments'}
                
        except Exception as e:
            logger.error(f"Error creating segmented GIFs: {e}")
            return {'success': False, 'error': str(e)}
    
    def optimize_existing_gif(self, gif_path: str, max_size_mb: float) -> bool:
        """Optimize an existing GIF file to meet size requirements"""
        try:
            if not os.path.exists(gif_path):
                logger.error(f"GIF file not found: {gif_path}")
                return False
            
            current_size = os.path.getsize(gif_path) / (1024 * 1024)
            if current_size <= max_size_mb:
                logger.info(f"GIF already within size limit: {current_size:.2f}MB <= {max_size_mb:.2f}MB")
                return True
            
            logger.info(f"Optimizing GIF: {current_size:.2f}MB -> target {max_size_mb:.2f}MB")
            return self.optimizer.optimize_gif(gif_path, max_size_mb)
            
        except Exception as e:
            logger.error(f"Error optimizing GIF: {e}")
            return False
    
    def get_gif_info(self, gif_path: str) -> Dict[str, Any]:
        """Get information about a GIF file"""
        try:
            if not os.path.exists(gif_path):
                return {'error': 'File not found'}
            
            # Get basic file info
            file_size = os.path.getsize(gif_path)
            file_size_mb = file_size / (1024 * 1024)
            
            # Try to get GIF-specific info using FFmpeg
            cmd = ['ffmpeg', '-i', gif_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            
            duration = None
            fps = None
            resolution = None
            
            if result.stderr:
                # Parse FFmpeg output for GIF info
                lines = result.stderr.split('\n')
                for line in lines:
                    if 'Duration:' in line:
                        # Extract duration
                        duration_match = re.search(r'Duration: (\d{2}):(\d{2}):(\d{2})\.(\d{2})', line)
                        if duration_match:
                            h, m, s, cs = map(int, duration_match.groups())
                            duration = h * 3600 + m * 60 + s + cs / 100
                    
                    elif 'Video:' in line:
                        # Extract resolution and fps
                        res_match = re.search(r'(\d+)x(\d+)', line)
                        if res_match:
                            width, height = map(int, res_match.groups())
                            resolution = (width, height)
                        
                        fps_match = re.search(r'(\d+(?:\.\d+)?) fps', line)
                        if fps_match:
                            fps = float(fps_match.group(1))
            
            return {
                'file_size_bytes': file_size,
                'file_size_mb': file_size_mb,
                'duration': duration,
                'fps': fps,
                'resolution': resolution
            }
            
        except Exception as e:
            logger.error(f"Error getting GIF info: {e}")
            return {'error': str(e)}
    
    def _set_windows_thumbnail_attributes(self, gif_path: str) -> bool:
        """Set Windows thumbnail attributes for GIF file"""
        try:
            if os.name == 'nt':  # Windows
                # Remove read-only, system, and hidden attributes
                subprocess.run(['attrib', '-H', '-S', '-R', gif_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                
                # Set archive attribute
                subprocess.run(['attrib', '+A', gif_path], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
                
                logger.debug(f"Set Windows thumbnail attributes for: {gif_path}")
                return True
        except Exception as e:
            logger.debug(f"Could not set Windows thumbnail attributes: {e}")
            # Not critical, continue without thumbnail optimization
            return False