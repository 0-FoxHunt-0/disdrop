"""
GIF Generation Module
Converts videos to optimized GIFs for social media platforms
"""

import os
import subprocess
import tempfile
import shutil
from typing import Dict, Any, Optional, Tuple
import logging
from pathlib import Path
from PIL import Image, ImageSequence
import cv2
from tqdm import tqdm

from .config_manager import ConfigManager

logger = logging.getLogger(__name__)

class GifGenerator:
    def __init__(self, config_manager: ConfigManager):
        self.config = config_manager
        self.temp_dir = self.config.get_temp_dir()
        
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
        
        # Calculate initial parameters with aspect ratio preservation
        gif_params = self._calculate_gif_params(platform_config, max_size_mb, duration, video_info)
        
        # Adjust duration if needed
        max_duration = gif_params['max_duration']
        actual_duration = min(duration or max_duration, video_info['duration'] - start_time, max_duration)
        
        logger.info(f"Creating GIF: {gif_params['width']}x{gif_params['height']}, "
                   f"{actual_duration:.1f}s @ {gif_params['fps']}fps")
        
        # Use the new quality optimization method
        results = self._create_gif_with_quality_optimization(
            input_video, output_path, gif_params, start_time, actual_duration, video_info
        )
        
        logger.info(f"GIF generation completed: {results['file_size_mb']:.2f}MB, {results['frame_count']} frames")
        
        return results

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
    
    def _calculate_gif_params(self, platform_config: Dict[str, Any], 
                            max_size_mb: Optional[int], 
                            duration: Optional[float],
                            video_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Calculate optimal GIF parameters with aspect ratio preservation"""
        
        # Base parameters from config
        params = {
            'width': self.config.get('gif_settings.width', 480),
            'height': self.config.get('gif_settings.height', 480),
            'fps': self.config.get('gif_settings.fps', 15),
            'max_duration': self.config.get('gif_settings.max_duration_seconds', 15),
            'max_size_mb': max_size_mb or self.config.get('gif_settings.max_file_size_mb', 10),
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
        
        # Calculate aspect ratio preserving dimensions if video info is available
        if video_info and video_info.get('width') and video_info.get('height'):
            original_width = video_info['width']
            original_height = video_info['height']
            original_aspect_ratio = original_width / original_height
            
            # Get maximum dimensions from config
            max_width = params['width']
            max_height = params['height']
            
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
            
            # Ensure dimensions are even numbers (required for some codecs)
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            params['width'] = new_width
            params['height'] = new_height
            
            logger.info(f"Preserving aspect ratio: {original_width}x{original_height} -> {new_width}x{new_height} (ratio: {original_aspect_ratio:.2f})")
        
        logger.debug(f"GIF parameters: {params}")
        return params
    
    def _create_gif_ffmpeg_palette(self, input_video: str, output_gif: str, 
                                 params: Dict[str, Any], start_time: float, duration: float):
        """Create GIF using FFmpeg with optimized palette (highest quality)"""
        
        logger.info("Creating GIF using FFmpeg with palette optimization")
        
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
            result = subprocess.run(palette_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, palette_cmd, result.stderr)
            
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
            result = subprocess.run(gif_cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, gif_cmd, result.stderr)
            
        finally:
            # Clean up palette file
            if os.path.exists(palette_file):
                os.remove(palette_file)
    
    def _create_gif_ffmpeg_direct(self, input_video: str, output_gif: str,
                                params: Dict[str, Any], start_time: float, duration: float):
        """Create GIF using direct FFmpeg conversion"""
        
        logger.info("Creating GIF using direct FFmpeg conversion")
        
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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
    
    def _create_gif_opencv_pil(self, input_video: str, output_gif: str,
                             params: Dict[str, Any], start_time: float, duration: float):
        """Create GIF using OpenCV + PIL (fallback method)"""
        
        logger.info("Creating GIF using OpenCV + PIL (fallback method)")
        
        cap = cv2.VideoCapture(input_video)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {input_video}")
        
        try:
            # Calculate frame parameters
            original_fps = cap.get(cv2.CAP_PROP_FPS)
            frame_interval = int(original_fps / params['fps'])
            start_frame = int(start_time * original_fps)
            end_frame = int((start_time + duration) * original_fps)
            
            # Set starting frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames = []
            frame_count = 0
            
            with tqdm(total=int(duration * params['fps']), desc="Extracting frames") as pbar:
                while cap.isOpened() and frame_count < end_frame - start_frame:
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
                        
                        pbar.update(1)
                    
                    frame_count += 1
            
            if not frames:
                raise ValueError("No frames extracted from video")
            
            # Calculate frame duration in milliseconds
            frame_duration = int(1000 / params['fps'])
            
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
    
    def _adjust_gif_params(self, params: Dict[str, Any], current_size_mb: float) -> Dict[str, Any]:
        """Adjust GIF parameters to reduce file size while preserving aspect ratio"""
        
        size_ratio = params['max_size_mb'] / current_size_mb
        
        # Reduce colors first
        if params['colors'] > 128:
            params['colors'] = max(int(params['colors'] * 0.7), 64)
        
        # Reduce FPS
        if params['fps'] > 8:
            params['fps'] = max(int(params['fps'] * 0.8), 6)
        
        # Reduce resolution if still too large, preserving aspect ratio
        if size_ratio < 0.7:
            # Calculate current aspect ratio
            current_aspect_ratio = params['width'] / params['height']
            
            # Reduce dimensions while maintaining aspect ratio
            reduction_factor = 0.9
            new_width = int(params['width'] * reduction_factor)
            new_height = int(params['height'] * reduction_factor)
            
            # Ensure dimensions are even numbers
            new_width = new_width - (new_width % 2)
            new_height = new_height - (new_height % 2)
            
            # Ensure minimum dimensions
            new_width = max(new_width, 160)
            new_height = max(new_height, 160)
            
            params['width'] = new_width
            params['height'] = new_height
            
            logger.debug(f"Reduced resolution while preserving aspect ratio: {new_width}x{new_height}")
        
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
        """Estimate GIF file size based on parameters - helps avoid unnecessary generation attempts"""
        
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
        
        # Compression factor based on lossy setting
        compression_factor = 1.0 - (params['lossy'] / 200.0)  # 0.5 to 1.0
        
        # Dithering affects compression
        dither_factor = 0.8 if params['dither'] == 'floyd_steinberg' else 0.9 if params['dither'] == 'bayer' else 1.0
        
        # Calculate estimated size
        estimated_bytes = (
            pixels_per_frame * 
            bits_per_pixel / 8 * 
            frame_count * 
            compression_factor * 
            dither_factor
        )
        
        # Add some overhead for GIF headers and color tables
        estimated_bytes *= 1.1
        
        return estimated_bytes / (1024 * 1024)  # Convert to MB

    def _increase_params_for_quality(self, params: Dict[str, Any], current_size_mb: float, target_size_mb: float) -> Dict[str, Any]:
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
            
            # Increase resolution if possible
            if params['width'] < 800 and params['height'] < 800:
                params['width'] = min(params['width'] + 20, 800)
                params['height'] = min(params['height'] + 20, 800)
                # Ensure even dimensions
                params['width'] = params['width'] - (params['width'] % 2)
                params['height'] = params['height'] - (params['height'] % 2)
            
            # Reduce lossy compression if possible
            if params['lossy'] > 0:
                params['lossy'] = max(params['lossy'] - 10, 0)
        
        return params

    def _create_gif_with_quality_optimization(self, input_video: str, output_path: str,
                                            initial_params: Dict[str, Any], start_time: float,
                                            duration: float, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create GIF with iterative quality optimization to achieve the best quality within size limits
        """
        
        max_size_mb = initial_params['max_size_mb']
        logger.info(f"Starting quality optimization process (target size: {max_size_mb}MB)")
        
        # Get quality optimization settings from config
        quality_config = self.config.get('gif_settings.quality_optimization', {})
        enabled = quality_config.get('enabled', True)
        max_attempts_per_stage = quality_config.get('max_attempts_per_stage', 3)
        target_size_efficiency = quality_config.get('target_size_efficiency', 0.9)
        quality_stages_config = quality_config.get('quality_stages', [])
        
        if not enabled:
            logger.info("Quality optimization disabled, using standard generation")
            return self._create_gif_standard(input_video, output_path, initial_params, start_time, duration, video_info)
        
        # Build quality stages from config
        optimization_stages = []
        for stage_config in quality_stages_config:
            stage_params = self._build_quality_stage_params(initial_params.copy(), stage_config)
            optimization_stages.append({
                'name': stage_config.get('name', 'Unknown'),
                'params': stage_params
            })
        
        # If no stages configured, use default stages
        if not optimization_stages:
            optimization_stages = [
                {'name': 'Maximum Quality', 'params': self._get_maximum_quality_params(initial_params.copy())},
                {'name': 'High Quality', 'params': self._get_high_quality_params(initial_params.copy())},
                {'name': 'Medium Quality', 'params': self._get_medium_quality_params(initial_params.copy())},
                {'name': 'Standard Quality', 'params': initial_params.copy()},
            ]
        
        best_result = None
        best_quality_score = 0
        temp_files_to_cleanup = []  # Track temp files for cleanup
        
        try:
            for stage_idx, stage in enumerate(optimization_stages):
                logger.info(f"Quality optimization stage {stage_idx + 1}/{len(optimization_stages)}: {stage['name']}")
                
                # Try multiple attempts with this quality level
                stage_params = stage['params']
                stage_success = False
                
                for attempt in range(max_attempts_per_stage):
                    logger.info(f"  Attempt {attempt + 1}/{max_attempts_per_stage}")
                    
                    # Estimate file size before generation
                    estimated_size = self._estimate_gif_size(stage_params, duration, video_info)
                    if estimated_size > max_size_mb * 1.5:  # If estimated size is way too large
                        logger.debug(f"    Skipping - estimated size {estimated_size:.2f}MB is too large")
                        # Reduce parameters more aggressively
                        stage_params = self._reduce_params_for_size(stage_params, estimated_size, max_size_mb)
                        continue
                    
                    temp_gif = os.path.join(self.temp_dir, f"temp_gif_stage_{stage_idx}_attempt_{attempt}.gif")
                    
                    try:
                        # Try different generation methods
                        success = False
                        
                        # Method 1: FFmpeg with palette (highest quality)
                        if not success:
                            try:
                                self._create_gif_ffmpeg_palette(
                                    input_video, temp_gif, stage_params, start_time, duration
                                )
                                success = True
                            except Exception as e:
                                logger.debug(f"FFmpeg palette method failed: {e}")
                        
                        # Method 2: Direct FFmpeg conversion
                        if not success:
                            try:
                                self._create_gif_ffmpeg_direct(
                                    input_video, temp_gif, stage_params, start_time, duration
                                )
                                success = True
                            except Exception as e:
                                logger.debug(f"FFmpeg direct method failed: {e}")
                        
                        # Method 3: OpenCV + PIL (fallback)
                        if not success:
                            try:
                                self._create_gif_opencv_pil(
                                    input_video, temp_gif, stage_params, start_time, duration
                                )
                                success = True
                            except Exception as e:
                                logger.debug(f"OpenCV+PIL method failed: {e}")
                        
                        if success and os.path.exists(temp_gif):
                            gif_size_mb = os.path.getsize(temp_gif) / (1024 * 1024)
                            
                            if gif_size_mb <= max_size_mb:
                                # Calculate quality score
                                quality_score = self._calculate_quality_score(stage_params, gif_size_mb, max_size_mb)
                                
                                logger.info(f"    Success! Size: {gif_size_mb:.2f}MB, Quality Score: {quality_score:.2f}")
                                
                                # Check if this is the best result so far
                                if quality_score > best_quality_score:
                                    # Clean up previous best result if it exists
                                    if best_result and os.path.exists(best_result['temp_file']):
                                        try:
                                            os.remove(best_result['temp_file'])
                                        except Exception as e:
                                            logger.debug(f"Failed to clean up previous best result: {e}")
                                    
                                    best_quality_score = quality_score
                                    best_result = {
                                        'temp_file': temp_gif,
                                        'params': stage_params.copy(),
                                        'size_mb': gif_size_mb,
                                        'quality_score': quality_score
                                    }
                                    logger.info(f"    New best result! Quality score: {quality_score:.2f}")
                                    
                                    # Don't add this file to cleanup list since it's our best result
                                    stage_success = True
                                else:
                                    # This file is not the best result, add to cleanup
                                    temp_files_to_cleanup.append(temp_gif)
                                
                                # If we have room to spare, try increasing quality for next attempt
                                if gif_size_mb < max_size_mb * 0.8:
                                    logger.debug(f"    Room to spare ({gif_size_mb:.2f}MB < {max_size_mb * 0.8:.2f}MB), trying higher quality")
                                    stage_params = self._increase_params_for_quality(stage_params, gif_size_mb, max_size_mb)
                                
                                # Early termination: if we're at maximum quality and under size limit, we can stop
                                if stage_idx == 0 and gif_size_mb <= max_size_mb * target_size_efficiency:
                                    logger.info(f"Maximum quality achieved within size limit ({target_size_efficiency*100:.0f}%) - stopping optimization")
                                    break
                            
                            else:
                                logger.debug(f"    Too large: {gif_size_mb:.2f}MB > {max_size_mb}MB")
                                # Add to cleanup since it's too large
                                temp_files_to_cleanup.append(temp_gif)
                                # Try to reduce parameters for next attempt
                                stage_params = self._reduce_params_for_size(stage_params, gif_size_mb, max_size_mb)
                        
                        else:
                            # Generation failed, no file to clean up
                            pass
                            
                    except Exception as e:
                        logger.debug(f"    Attempt failed: {e}")
                        # Add to cleanup if file was created
                        if os.path.exists(temp_gif):
                            temp_files_to_cleanup.append(temp_gif)
                
                # Only stop early if we've achieved maximum quality and are very close to the size limit
                # This ensures we push for maximum quality up to the limit
                if (stage_success and best_result and 
                    best_result['size_mb'] >= max_size_mb * 0.95 and 
                    best_result['quality_score'] >= 0.9):
                    logger.info(f"Maximum quality achieved at {best_result['size_mb']:.2f}MB ({best_result['size_mb']/max_size_mb:.1%} of limit) - stopping optimization")
                    break
        
        finally:
            # Clean up all temporary files except the best result
            for temp_file in temp_files_to_cleanup:
                if os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        logger.debug(f"Failed to clean up temp file {temp_file}: {e}")
        
        if not best_result:
            raise RuntimeError(f"Failed to generate GIF under {max_size_mb}MB with any quality level")
        
        # Verify the best result file still exists
        if not os.path.exists(best_result['temp_file']):
            raise RuntimeError(f"Best result file was lost: {best_result['temp_file']}")
        
        # Move the best result to final output
        try:
            shutil.move(best_result['temp_file'], output_path)
            logger.info(f"Final result: {best_result['size_mb']:.2f}MB with quality score {best_result['quality_score']:.2f}")
        except Exception as e:
            logger.error(f"Failed to move best result to output: {e}")
            # Try to copy instead
            try:
                shutil.copy2(best_result['temp_file'], output_path)
                logger.info(f"Copied best result to output: {best_result['size_mb']:.2f}MB")
            except Exception as copy_error:
                raise RuntimeError(f"Failed to copy best result to output: {copy_error}")
        
        # Generate final results
        return self._get_gif_results(input_video, output_path, video_info, best_result['params'], duration)

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
            params['width'] = min(800, int(params['width'] * stage_config['resolution_multiplier']))
            params['height'] = min(800, int(params['height'] * stage_config['resolution_multiplier']))
            
            # Ensure even dimensions
            params['width'] = params['width'] - (params['width'] % 2)
            params['height'] = params['height'] - (params['height'] % 2)
        
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
        
        # Try to increase resolution if possible
        params['width'] = min(params['width'] * 1.5, 800)  # Increase width but cap at 800
        params['height'] = min(params['height'] * 1.5, 800)  # Increase height but cap at 800
        
        # Ensure even dimensions
        params['width'] = int(params['width']) - (int(params['width']) % 2)
        params['height'] = int(params['height']) - (int(params['height']) % 2)
        
        return params

    def _get_high_quality_params(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters for high quality GIF generation"""
        params = base_params.copy()
        
        # High quality settings
        params['colors'] = 256
        params['fps'] = min(25, params['fps'] * 1.5)  # Good FPS
        params['lossy'] = 20  # Minimal lossy compression
        params['dither'] = 'floyd_steinberg'
        
        # Slightly increased resolution
        params['width'] = min(params['width'] * 1.2, 600)
        params['height'] = min(params['height'] * 1.2, 600)
        
        # Ensure even dimensions
        params['width'] = int(params['width']) - (int(params['width']) % 2)
        params['height'] = int(params['height']) - (int(params['height']) % 2)
        
        return params

    def _get_medium_quality_params(self, base_params: Dict[str, Any]) -> Dict[str, Any]:
        """Get parameters for medium quality GIF generation"""
        params = base_params.copy()
        
        # Medium quality settings
        params['colors'] = 192
        params['fps'] = min(20, params['fps'] * 1.2)
        params['lossy'] = 40
        params['dither'] = 'bayer'
        
        return params

    def _reduce_params_for_size(self, params: Dict[str, Any], current_size_mb: float, target_size_mb: float) -> Dict[str, Any]:
        """Reduce parameters to achieve target size - optimized for performance"""
        size_ratio = target_size_mb / current_size_mb
        
        # More aggressive reduction based on how far over the limit we are
        if size_ratio < 0.5:
            # Way too large - aggressive reduction
            reduction_factor = 0.6
            params['colors'] = max(int(params['colors'] * 0.5), 64)
            params['fps'] = max(int(params['fps'] * 0.7), 6)
        elif size_ratio < 0.7:
            # Quite large - moderate reduction
            reduction_factor = 0.8
            params['colors'] = max(int(params['colors'] * 0.7), 64)
            params['fps'] = max(int(params['fps'] * 0.8), 8)
        else:
            # Slightly large - gentle reduction
            reduction_factor = 0.9
            params['colors'] = max(int(params['colors'] * 0.8), 64)
            params['fps'] = max(int(params['fps'] * 0.9), 8)
        
        # Reduce resolution if still too large
        if size_ratio < 0.8:
            params['width'] = int(params['width'] * reduction_factor)
            params['height'] = int(params['height'] * reduction_factor)
            
            # Ensure even dimensions
            params['width'] = params['width'] - (params['width'] % 2)
            params['height'] = params['height'] - (params['height'] % 2)
            
            # Ensure minimum dimensions
            params['width'] = max(params['width'], 160)
            params['height'] = max(params['height'], 160)
        
        # Increase lossy compression more aggressively
        if size_ratio < 0.6:
            params['lossy'] = min(params['lossy'] + 30, 150)
        else:
            params['lossy'] = min(params['lossy'] + 15, 100)
        
        return params

    def _calculate_quality_score(self, params: Dict[str, Any], actual_size_mb: float, max_size_mb: float) -> float:
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
        
        # Calculate weighted quality score - prioritize size efficiency more
        quality_score = (
            color_score * 0.20 +
            fps_score * 0.20 +
            resolution_score * 0.20 +
            size_efficiency * 0.30 +  # Increased weight for size efficiency
            lossy_penalty * 0.05 +
            dither_score * 0.05
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