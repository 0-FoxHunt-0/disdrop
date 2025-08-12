"""
Video Segmentation Module
Handles segmentation of large video files into smaller, manageable segments
Similar to GIF segmentation but optimized for video files
"""

import os
import subprocess
import tempfile
import shutil
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
from tqdm import tqdm
import json
import math
import time

from .config_manager import ConfigManager
from .hardware_detector import HardwareDetector
from .ffmpeg_utils import FFmpegUtils

logger = logging.getLogger(__name__)

class VideoSegmenter:
    def __init__(self, config_manager: ConfigManager, hardware_detector: HardwareDetector):
        self.config = config_manager
        self.hardware = hardware_detector
        self.temp_dir = self.config.get_temp_dir()
        
        # Shutdown handling
        self.shutdown_requested = False
        self.current_ffmpeg_process = None
        
    def should_segment_video(self, duration: float, video_info: Dict[str, Any], max_size_mb: float) -> bool:
        """
        Determine if a video should be split into segments based on estimated file size
        Video segmentation is now a LAST RESORT - only when single file compression cannot achieve decent quality
        
        Args:
            duration: Video duration in seconds
            video_info: Video metadata including complexity, resolution, etc.
            max_size_mb: Target size limit
            
        Returns:
            True if video should be split into segments (last resort only)
        """
        
        # Get segmentation settings from config
        segmentation_config = self.config.get('video_compression.segmentation', {})
        
        # Direct size check: if original video is already significantly larger than target, segment immediately
        original_size_mb = video_info.get('size_bytes', 0) / (1024 * 1024)
        logger.info(f"Video segmentation check: original size {original_size_mb:.1f}MB, target {max_size_mb}MB, "
                   f"duration {duration:.1f}s, resolution {video_info.get('width', 0)}x{video_info.get('height', 0)}")
        
        # Only segment if original is extremely large (5x target size)
        if original_size_mb > max_size_mb * 5.0:  # Increased from 3x to 5x
            logger.info(f"Video segmentation recommended: original size {original_size_mb:.1f}MB > "
                       f"{max_size_mb * 5.0:.1f}MB (5x target size)")
            return True
        
        # Create optimal video parameters for size estimation
        estimation_params = {
            'width': min(1280, video_info.get('width', 1280)),
            'height': min(720, video_info.get('height', 720)),
            'fps': min(30, video_info.get('fps', 30)),
            'bitrate': '1000k',  # Conservative bitrate for estimation
            'crf': 28  # Conservative quality for estimation
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
        
        logger.info(f"Estimation parameters: {estimation_params['width']}x{estimation_params['height']}, "
                   f"{estimation_params['fps']}fps, {estimation_params['bitrate']}, CRF {estimation_params['crf']}")
        
        # Estimate file size with optimal compression
        estimated_size_mb = self._estimate_video_size(estimation_params, duration, video_info)
        
        # Try with more aggressive compression first to see if single video is feasible
        aggressive_params = estimation_params.copy()
        aggressive_params.update({
            'fps': 24,
            'bitrate': '800k',
            'crf': 32
        })
        
        aggressive_size_mb = self._estimate_video_size(aggressive_params, duration, video_info)
        logger.info(f"Aggressive estimation: {aggressive_size_mb:.1f}MB (target: {max_size_mb * 1.5:.1f}MB)")
        
        # Try with very aggressive compression as last resort
        very_aggressive_params = estimation_params.copy()
        very_aggressive_params.update({
            'fps': 20,
            'bitrate': '600k',
            'crf': 35
        })
        
        very_aggressive_size_mb = self._estimate_video_size(very_aggressive_params, duration, video_info)
        logger.info(f"Very aggressive estimation: {very_aggressive_size_mb:.1f}MB (target: {max_size_mb * 1.8:.1f}MB)")
        
        # Only segment if even very aggressive compression cannot achieve target
        if very_aggressive_size_mb > max_size_mb * 1.8:
            logger.info(f"Video segmentation recommended: even very aggressive compression estimates {very_aggressive_size_mb:.1f}MB > "
                       f"{max_size_mb * 1.8:.1f}MB")
            return True
        
        # Primary decision: split if estimated size exceeds limit significantly (increased threshold)
        size_threshold_multiplier = segmentation_config.get('size_threshold_multiplier', 3.0)  # Increased from 2.5
        threshold_size = max_size_mb * size_threshold_multiplier
        logger.info(f"Estimated size: {estimated_size_mb:.1f}MB, threshold: {threshold_size:.1f}MB "
                   f"(multiplier: {size_threshold_multiplier})")
        
        if estimated_size_mb > threshold_size:
            logger.info(f"Video segmentation recommended: estimated size {estimated_size_mb:.1f}MB > "
                       f"{threshold_size:.1f}MB threshold")
            return True
        
        # Secondary decision: split if estimated size is close to limit but video has challenging characteristics
        if estimated_size_mb > max_size_mb * 2.2:  # Increased from 1.8
            complexity = video_info.get('complexity_score', 5.0)
            motion_level = video_info.get('motion_level', 'medium')
            
            # Split only if very high complexity or motion makes compression unpredictable
            if complexity >= 9.0 or motion_level == 'very_high':  # Increased complexity threshold
                logger.info(f"Video segmentation recommended: estimated size {estimated_size_mb:.1f}MB with "
                           f"challenging characteristics (complexity: {complexity:.1f}, motion: {motion_level})")
                return True
        
        # Fallback: still split extremely long videos regardless of estimated size (increased limit)
        fallback_duration_limit = segmentation_config.get('fallback_duration_limit', 600)  # Increased from 300 to 10 minutes
        if duration > fallback_duration_limit:
            logger.info(f"Video segmentation recommended: duration {duration}s exceeds fallback limit {fallback_duration_limit}s")
            return True
        
        logger.info(f"Single video recommended: estimated size {estimated_size_mb:.1f}MB (aggressive: {aggressive_size_mb:.1f}MB, "
                   f"very aggressive: {very_aggressive_size_mb:.1f}MB) within acceptable range for {max_size_mb}MB target")
        return False
    
    def segment_video(self, input_video: str, output_base_path: str, platform: str = None,
                     max_size_mb: float = None, start_time: float = 0, 
                     duration: float = None) -> Dict[str, Any]:
        """
        Split a large video into multiple high-quality segments
        
        Args:
            input_video: Path to input video
            output_base_path: Base path for output (will create folder)
            platform: Target platform for compression settings
            max_size_mb: Maximum file size per segment
            start_time: Start time in video
            duration: Duration to process
            
        Returns:
            Dictionary with results including segment paths and metadata
        """
        
        if not os.path.exists(input_video):
            raise FileNotFoundError(f"Input video file not found: {input_video}")
        
        logger.info(f"Starting video segmentation: {input_video} -> {output_base_path}")
        
        # Get platform configuration
        platform_config = {}
        if platform:
            platform_config = self.config.get_platform_config(platform, 'video_compression')
            logger.info(f"Using platform configuration for: {platform}")
        
        # Get video information
        video_info = self._get_video_info(input_video)
        
        # Determine target file size
        target_size_mb = max_size_mb or platform_config.get('max_file_size_mb') or self.config.get('video_compression.max_file_size_mb', 10)
        
        # Adjust duration if needed
        if duration is not None:
            actual_duration = min(duration, video_info['duration'] - start_time)
        else:
            actual_duration = video_info['duration'] - start_time
        
        # Check if video should be segmented
        if self.should_segment_video(actual_duration, video_info, target_size_mb):
            logger.info("Video will be split into high-quality segments")
            print("🎬 Video will be split into high-quality segments")
            results = self._split_video_into_segments(
                input_video, output_base_path, start_time, actual_duration, 
                video_info, platform_config, target_size_mb
            )
            
            # If segmentation failed, clean up any temp files
            if not results.get('success', False):
                self._cleanup_temp_files(results.get('temp_files', []))
                return results
            
            # Move segments to final location
            final_results = self._finalize_segments(results, output_base_path)
            return final_results
        else:
            logger.info("Video does not need segmentation, processing as single file")
            return self._process_single_video(input_video, output_base_path, start_time, 
                                           actual_duration, video_info, platform_config, target_size_mb)
    
    def _split_video_into_segments(self, input_video: str, output_base_path: str, 
                                  start_time: float, duration: float, 
                                  video_info: Dict[str, Any], platform_config: Dict[str, Any],
                                  target_size_mb: float) -> Dict[str, Any]:
        """
        Split a long/complex video into multiple high-quality video segments
        
        Args:
            input_video: Path to input video
            output_base_path: Base path for output (will create folder)
            start_time: Start time in video
            duration: Duration to process
            video_info: Video metadata
            platform_config: Platform-specific configuration
            target_size_mb: Target size per segment
            
        Returns:
            Dictionary with results including segment paths and metadata
        """
        
        # Create temp folder for segments first
        base_name = os.path.splitext(os.path.basename(output_base_path))[0]
        temp_segments_folder = os.path.join(self.temp_dir, f"{base_name}_segments_temp")
        
        os.makedirs(temp_segments_folder, exist_ok=True)
        logger.info(f"Created temp segments folder: {temp_segments_folder}")
        
        # Calculate optimal segment duration
        segment_duration = self._calculate_optimal_segment_duration(duration, video_info, target_size_mb)
        
        # Calculate number of segments
        num_segments = math.ceil(duration / segment_duration)
        
        logger.info(f"Splitting {duration:.1f}s video into {num_segments} segments of ~{segment_duration:.1f}s each")
        
        segments = []
        temp_files = []
        
        try:
            # Process each segment
            for i in range(num_segments):
                segment_start = start_time + (i * segment_duration)
                segment_end = min(start_time + ((i + 1) * segment_duration), start_time + duration)
                segment_duration_actual = segment_end - segment_start
                
                if segment_duration_actual <= 0:
                    continue
                
                # Create segment filename
                segment_filename = f"{base_name}_segment_{i+1:03d}.mp4"
                segment_path = os.path.join(temp_segments_folder, segment_filename)
                
                logger.info(f"Processing segment {i+1}/{num_segments}: {segment_duration_actual:.1f}s "
                           f"({segment_start:.1f}s - {segment_end:.1f}s)")
                
                # Create high-quality segment
                segment_result = self._create_high_quality_segment(
                    input_video, segment_path, segment_start, segment_duration_actual,
                    video_info, platform_config, target_size_mb
                )
                
                if segment_result.get('success', False):
                    segment_size_mb = segment_result.get('size_mb', 0)
                    
                    # Check if segment is too large and needs optimization
                    if segment_size_mb > target_size_mb:
                        logger.warning(f"Segment {i+1} is {segment_size_mb:.2f}MB (exceeds target {target_size_mb}MB), applying optimization...")
                        
                        # Apply post-segment optimization
                        optimized_result = self._optimize_oversized_segment(
                            segment_path, segment_duration_actual, video_info, 
                            platform_config, target_size_mb
                        )
                        
                        if optimized_result.get('success', False):
                            segment_size_mb = optimized_result.get('size_mb', segment_size_mb)
                            segment_result['method'] = f"{segment_result.get('method', 'unknown')}_optimized"
                            segment_result['size_mb'] = segment_size_mb
                            segment_result['optimization_applied'] = True
                            logger.info(f"Segment {i+1} optimized to {segment_size_mb:.2f}MB")
                        else:
                            logger.warning(f"Failed to optimize segment {i+1}: {optimized_result.get('error', 'Unknown error')}")
                            segment_result['optimization_applied'] = False
                    else:
                        segment_result['optimization_applied'] = False
                    
                    segments.append({
                        'index': i + 1,
                        'path': segment_path,
                        'start_time': segment_start,
                        'duration': segment_duration_actual,
                        'size_mb': segment_size_mb,
                        'method': segment_result.get('method', 'unknown'),
                        'quality_score': segment_result.get('quality_score', 0),
                        'optimization_applied': segment_result.get('optimization_applied', False)
                    })
                    temp_files.append(segment_path)
                    
                    logger.info(f"Segment {i+1} completed: {segment_size_mb:.2f}MB "
                               f"using {segment_result.get('method', 'unknown')} method")
                else:
                    logger.error(f"Failed to create segment {i+1}: {segment_result.get('error', 'Unknown error')}")
                    return {
                        'success': False,
                        'error': f"Failed to create segment {i+1}: {segment_result.get('error', 'Unknown error')}",
                        'temp_files': temp_files
                    }
            
            # Create segments summary
            self._create_segments_summary(temp_segments_folder, segments, base_name, duration)
            
            return {
                'success': True,
                'segments': segments,
                'num_segments': len(segments),
                'total_duration': duration,
                'segment_duration': segment_duration,
                'temp_folder': temp_segments_folder,
                'temp_files': temp_files,
                'base_name': base_name
            }
            
        except Exception as e:
            logger.error(f"Error during video segmentation: {e}")
            return {
                'success': False,
                'error': str(e),
                'temp_files': temp_files
            }
    
    def _calculate_optimal_segment_duration(self, total_duration: float, video_info: Dict[str, Any], target_size_mb: float) -> float:
        """Calculate optimal duration for each segment based on content characteristics and target size"""
        
        complexity = video_info.get('complexity_score', 5.0)
        motion_level = video_info.get('motion_level', 'medium')
        
        # Calculate base duration to target 8-9MB segments (closer to the 10MB ceiling)
        # Estimate duration needed to reach target size based on video characteristics
        bitrate = video_info.get('bitrate', 0)
        if bitrate > 0:
            # Use actual bitrate to estimate duration for target size
            # Account for compression efficiency (segments often compress better)
            target_bits = (target_size_mb * 0.85) * 8 * 1024 * 1024  # 85% for video, 15% for audio
            estimated_duration = target_bits / bitrate
            # Double the duration to get closer to target size (as you requested)
            base_duration = min(estimated_duration * 2, total_duration / 3)  # At least 3 segments max
        else:
            # Fallback: aim for fewer, larger segments
            base_duration = total_duration / max(3, int(total_duration / 90))  # Max 90s per segment
        
        # Adjust for content complexity
        if complexity >= 8 or motion_level == 'very_high':
            # Reduce duration for very complex content to maintain quality
            base_duration *= 0.85
        elif complexity >= 6:
            # Slight reduction for high complexity
            base_duration *= 0.95
        elif complexity < 4:
            # Can afford longer segments for simple content
            base_duration *= 1.1
        
        # Ensure reasonable bounds: 45s minimum, 120s maximum
        final_duration = max(45, min(120, base_duration))
        
        # Calculate expected number of segments
        expected_segments = math.ceil(total_duration / final_duration)
        
        logger.info(f"Optimized segment duration to: {final_duration:.1f}s "
                    f"(complexity: {complexity:.1f}, motion: {motion_level}, "
                    f"target: {target_size_mb}MB, expected segments: {expected_segments})")
        
        return final_duration
    
    def _create_high_quality_segment(self, input_video: str, output_path: str, 
                                   start_time: float, duration: float, 
                                   video_info: Dict[str, Any], platform_config: Dict[str, Any],
                                   target_size_mb: float) -> Dict[str, Any]:
        """Create a high-quality video segment using optimized parameters"""
        
        # Validate duration parameter
        if duration <= 0 or duration > 600:  # Sanity check: 0-10 minutes
            logger.error(f"Invalid segment duration: {duration}s - this should be between 0 and 600 seconds")
            return {'success': False, 'error': f'Invalid duration: {duration}s'}
        
        try:
            # Calculate segment-specific compression parameters
            segment_params = self._calculate_segment_compression_params(
                video_info, platform_config, target_size_mb, duration
            )
            
            # Apply content-aware adjustments
            segment_params = self._apply_content_aware_adjustments(segment_params, video_info)
            
            logger.info(f"Creating segment with optimization: {segment_params['width']}x{segment_params['height']}, "
                       f"{duration:.1f}s @ {segment_params['fps']}fps")
            
            # Build FFmpeg command for segment creation (always use software encoding for reliability)
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', input_video,
                '-c:v', 'libx264',  # Force software encoding for segments
                '-crf', str(segment_params.get('crf', 23)),  # Use CRF instead of bitrate for better quality
                '-preset', segment_params.get('preset', 'medium'),
                '-s', f"{segment_params['width']}x{segment_params['height']}",
                '-r', str(segment_params['fps']),
                '-c:a', 'aac',
                '-b:a', '128k',  # Increased audio bitrate
                '-ac', '2',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            
            # Execute FFmpeg command
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
                
                if result.returncode == 0:
                    # Get segment info
                    segment_size = os.path.getsize(output_path) / (1024 * 1024)
                    result = {
                        'success': True,
                        'segment_duration': duration,
                        'segment_start_time': start_time,
                        'optimization_type': 'segment_compression',
                        'size_mb': segment_size,
                        'method': 'direct_ffmpeg',
                        'output_file': output_path
                    }
                    
                    logger.info(f"Segment optimization successful: {segment_size:.2f}MB using direct FFmpeg method")
                    return result
                else:
                    logger.error(f"FFmpeg failed: {result.stderr}")
                    return {
                        'success': False,
                        'error': f"FFmpeg failed: {result.stderr}"
                    }
                    
            except subprocess.TimeoutExpired:
                logger.error("Segment creation timed out")
                return {
                    'success': False,
                    'error': 'Segment creation timed out'
                }
            except Exception as e:
                logger.error(f"Error creating segment: {e}")
                return {
                    'success': False,
                    'error': str(e)
                }
            
        except Exception as e:
            logger.error(f"Error creating high-quality segment: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _calculate_segment_compression_params(self, video_info: Dict[str, Any], 
                                            platform_config: Dict[str, Any], 
                                            target_size_mb: float, duration: float) -> Dict[str, Any]:
        """Calculate compression parameters optimized for video segments"""
        
        # Get best encoder
        encoder, accel_type = self.hardware.get_best_encoder("h264")
        
        # Base parameters
        params = {
            'encoder': encoder,
            'acceleration_type': accel_type,
            'width': video_info['width'],
            'height': video_info['height'],
            'fps': min(video_info['fps'], 30),
        }
        
        # Content-aware quality adjustment
        complexity = video_info.get('complexity_score', 5.0)
        motion_level = video_info.get('motion_level', 'medium')
        
        # Adaptive CRF based on content complexity
        if accel_type == 'software':
            base_crf = 23  # Higher quality starting point
            if complexity > 7:
                params['crf'] = base_crf + 3  # Higher CRF for complex content
            elif complexity < 3:
                params['crf'] = base_crf - 2  # Lower CRF for simple content
            else:
                params['crf'] = base_crf
        
        # Motion-aware preset selection
        if motion_level == 'high':
            params['preset'] = 'faster'  # Faster preset for high motion
        elif motion_level == 'low':
            params['preset'] = 'slower'  # Slower preset for low motion
        else:
            params['preset'] = 'medium'
        
        # Calculate bitrate for segment
        params['bitrate'] = self._calculate_segment_bitrate(video_info, target_size_mb, duration)
        
        # Platform-specific adjustments
        if platform_config:
            params.update(self._apply_platform_constraints(params, platform_config))
        
        return params
    
    def _calculate_segment_bitrate(self, video_info: Dict[str, Any], target_size_mb: float, duration: float) -> int:
        """Calculate bitrate for video segment"""
        
        # Base calculation accounting for audio overhead
        target_bits = target_size_mb * 8 * 1024 * 1024 * 0.85  # 85% for video, 15% for audio
        base_bitrate = int(target_bits / duration / 1000)  # Convert to kbps
        
        # Adjust based on content complexity
        complexity = video_info.get('complexity_score', 5.0)
        motion_level = video_info.get('motion_level', 'medium')
        
        # Complexity adjustment factor
        if complexity > 7:
            bitrate_multiplier = 1.2  # Need more bitrate for complex content
        elif complexity < 3:
            bitrate_multiplier = 0.8  # Can use less bitrate for simple content
        else:
            bitrate_multiplier = 1.0
        
        # Motion adjustment
        motion_multipliers = {'low': 0.9, 'medium': 1.0, 'high': 1.1}
        bitrate_multiplier *= motion_multipliers.get(motion_level, 1.0)
        
        # Resolution efficiency factor
        pixel_count = video_info['width'] * video_info['height']
        if pixel_count > 1920 * 1080:  # 4K+ content
            bitrate_multiplier *= 1.3
        elif pixel_count < 1280 * 720:  # Lower resolution
            bitrate_multiplier *= 0.8
        
        final_bitrate = int(base_bitrate * bitrate_multiplier)
        
        # Ensure minimum viable bitrate
        min_bitrate = 200
        final_bitrate = max(final_bitrate, min_bitrate)
        
        # Return as string with 'k' suffix for FFmpeg
        return f"{final_bitrate}k"
    
    def _optimize_oversized_segment(self, segment_path: str, duration: float, 
                                  video_info: Dict[str, Any], platform_config: Dict[str, Any], 
                                  target_size_mb: float) -> Dict[str, Any]:
        """Optimize an oversized segment using direct compression (bypassing segmentation)"""
        
        try:
            # Import the video compressor to use its optimization capabilities
            from .video_compressor import DynamicVideoCompressor
            
            # Create a temporary compressor instance with our existing config and hardware
            temp_compressor = DynamicVideoCompressor(self.config, self.hardware)
            
            # Create temp output path for optimized segment
            temp_optimized_path = segment_path.replace('.mp4', '_optimized_temp.mp4')
            
            logger.info(f"Applying direct compression optimization to oversized segment: {segment_path}")
            logger.info(f"Target size for optimization: {target_size_mb}MB")
            
            # Use 90% of target to provide buffer for potential size variations
            optimization_target = target_size_mb * 0.9
            
            # Get video info for the segment (it may be different from the original video)
            segment_video_info = temp_compressor._analyze_video_content(segment_path)
            
            # Try different compression strategies to optimize the segment
            optimization_result = None
            
            # Strategy 1: Advanced optimization (best quality but may fail)
            try:
                logger.info("Attempting advanced optimization strategy")
                optimization_result = temp_compressor._compress_with_advanced_optimization(
                    input_path=segment_path,
                    output_path=temp_optimized_path,
                    target_size_mb=optimization_target,
                    platform_config=platform_config or {},
                    video_info=segment_video_info
                )
            except Exception as e:
                logger.warning(f"Advanced optimization failed, trying standard: {e}")
                optimization_result = None
            
            # Strategy 2: Standard optimization (fallback)
            if not optimization_result or not optimization_result.get('success', False):
                logger.info("Attempting standard optimization strategy")
                try:
                    optimization_result = temp_compressor._compress_with_standard_optimization(
                        input_path=segment_path,
                        output_path=temp_optimized_path,
                        target_size_mb=optimization_target,
                        platform_config=platform_config or {},
                        video_info=segment_video_info
                    )
                except Exception as e:
                    logger.warning(f"Standard optimization failed, trying adaptive: {e}")
                    optimization_result = None
            
            # Strategy 3: Adaptive quality (last resort)
            if not optimization_result or not optimization_result.get('success', False):
                logger.info("Attempting adaptive quality optimization strategy")
                try:
                    optimization_result = temp_compressor._compress_with_adaptive_quality(
                        input_path=segment_path,
                        output_path=temp_optimized_path,
                        target_size_mb=optimization_target,
                        platform_config=platform_config or {},
                        video_info=segment_video_info
                    )
                except Exception as e:
                    logger.error(f"All optimization strategies failed: {e}")
                    optimization_result = None
            
            # Check results of optimization attempts
            if optimization_result and optimization_result.get('success', False):
                optimized_size_mb = optimization_result.get('size_mb', 0)
                
                # Check if optimization was successful (made it smaller)
                if optimized_size_mb < target_size_mb:
                    # Replace original segment with optimized version
                    if os.path.exists(temp_optimized_path):
                        shutil.move(temp_optimized_path, segment_path)
                        
                        logger.info(f"Segment optimization successful: {optimized_size_mb:.2f}MB "
                                   f"(was oversized, now within {target_size_mb}MB target)")
                        
                        return {
                            'success': True,
                            'size_mb': optimized_size_mb,
                            'method': optimization_result.get('method', 'compression_optimization'),
                            'original_method': 'segment_creation',
                            'compression_applied': True
                        }
                    else:
                        logger.error("Optimized segment file not found after processing")
                        return {
                            'success': False,
                            'error': 'Optimized segment file not found'
                        }
                else:
                    # Optimization didn't help enough, keep original
                    if os.path.exists(temp_optimized_path):
                        os.remove(temp_optimized_path)
                    
                    logger.warning(f"Optimization did not sufficiently reduce size: {optimized_size_mb:.2f}MB "
                                  f"still exceeds target {target_size_mb}MB")
                    return {
                        'success': False,
                        'error': f'Optimization insufficient: {optimized_size_mb:.2f}MB still too large'
                    }
            else:
                # All optimization strategies failed
                if os.path.exists(temp_optimized_path):
                    os.remove(temp_optimized_path)
                    
                error_msg = optimization_result.get('error', 'All optimization strategies failed') if optimization_result else 'All optimization strategies failed'
                logger.error(f"Segment optimization failed: {error_msg}")
                return {
                    'success': False,
                    'error': f'Optimization failed: {error_msg}'
                }
            
        except Exception as e:
            logger.error(f"Error during segment optimization: {e}")
            # Clean up any temp files
            temp_optimized_path = segment_path.replace('.mp4', '_optimized_temp.mp4')
            if os.path.exists(temp_optimized_path):
                try:
                    os.remove(temp_optimized_path)
                except:
                    pass
            
            return {
                'success': False,
                'error': f'Optimization error: {str(e)}'
            }
    
    def _apply_content_aware_adjustments(self, params: Dict[str, Any], video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Apply content-aware adjustments to compression parameters"""
        
        complexity = video_info.get('complexity_score', 5.0)
        motion_level = video_info.get('motion_level', 'medium')
        
        # Adjust FPS based on motion level
        if motion_level == 'high':
            params['fps'] = min(params['fps'], 30)  # Keep high FPS for high motion
        elif motion_level == 'low':
            params['fps'] = min(params['fps'], 24)  # Lower FPS for low motion
        
        # Adjust quality based on complexity
        if 'crf' in params:
            if complexity > 7:
                params['crf'] = min(params['crf'] + 2, 35)  # Higher CRF for complex content
            elif complexity < 3:
                params['crf'] = max(params['crf'] - 2, 18)  # Lower CRF for simple content
        
        return params
    
    def _apply_platform_constraints(self, params: Dict[str, Any], platform_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply platform-specific constraints to parameters"""
        constraints = {}
        
        # Resolution constraints
        if 'max_width' in platform_config and 'max_height' in platform_config:
            max_width = platform_config['max_width']
            max_height = platform_config['max_height']
            
            scale_factor = min(max_width / params['width'], max_height / params['height'], 1.0)
            constraints['width'] = int(params['width'] * scale_factor)
            constraints['height'] = int(params['height'] * scale_factor)
            
            # Ensure even dimensions
            constraints['width'] = constraints['width'] if constraints['width'] % 2 == 0 else constraints['width'] - 1
            constraints['height'] = constraints['height'] if constraints['height'] % 2 == 0 else constraints['height'] - 1
        
        # FPS constraints
        if 'fps' in platform_config:
            constraints['fps'] = min(platform_config['fps'], params['fps'])
        
        return constraints
    
    def _estimate_video_size(self, params: Dict[str, Any], duration: float, video_info: Dict[str, Any]) -> float:
        """Estimate video file size based on parameters and duration"""
        
        # Calculate bitrate in kbps
        if 'bitrate' in params:
            if isinstance(params['bitrate'], str):
                # Parse bitrate string (e.g., "1000k")
                bitrate_str = params['bitrate'].lower()
                if bitrate_str.endswith('k'):
                    bitrate = int(bitrate_str[:-1])
                elif bitrate_str.endswith('m'):
                    bitrate = int(bitrate_str[:-1]) * 1000
                else:
                    bitrate = int(bitrate_str)
            else:
                bitrate = params['bitrate']
        else:
            # Estimate bitrate based on resolution and quality
            pixel_count = params['width'] * params['height']
            fps = params.get('fps', 30)
            crf = params.get('crf', 28)
            
            # Rough bitrate estimation
            if pixel_count > 1920 * 1080:  # 4K+
                bitrate = 4000
            elif pixel_count > 1280 * 720:  # 1080p
                bitrate = 2000
            elif pixel_count > 854 * 480:  # 480p
                bitrate = 1000
            else:
                bitrate = 500
            
            # Adjust for quality
            if crf < 20:
                bitrate *= 1.5
            elif crf > 35:
                bitrate *= 0.7
        
        # Calculate size in MB
        bits_per_second = bitrate * 1000
        total_bits = bits_per_second * duration
        size_mb = total_bits / (8 * 1024 * 1024)
        
        # Add audio overhead (typically 10-15%)
        size_mb *= 1.15
        
        return size_mb
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get comprehensive video information using FFprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', '-show_streams', '-show_entries', 
                'frame=pkt_pts_time,pict_type,pkt_size', '-select_streams', 'v:0',
                video_path
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=60)
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
            
            probe_data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = next(
                (stream for stream in probe_data['streams'] if stream['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise ValueError("No video stream found in file")
            
            # Basic video properties
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            duration = float(probe_data['format'].get('duration', 0))
            bitrate = int(probe_data['format'].get('bit_rate', 0))
            from .ffmpeg_utils import FFmpegUtils
            fps = FFmpegUtils.parse_fps(video_stream.get('r_frame_rate', '30/1'))
            
            # Content complexity analysis
            complexity_score = self._calculate_content_complexity(probe_data, width, height, duration)
            
            # Motion analysis
            motion_level = self._estimate_motion_level(video_stream, duration)
            
            # Scene change detection
            scene_changes = self._estimate_scene_changes(duration, fps)
            
            return {
                'width': width,
                'height': height,
                'duration': duration,
                'bitrate': bitrate,
                'fps': fps,
                'codec': video_stream.get('codec_name', 'unknown'),
                'size_bytes': int(probe_data['format'].get('size', 0)),
                'complexity_score': complexity_score,
                'motion_level': motion_level,
                'scene_changes': scene_changes,
                'aspect_ratio': width / height if height > 0 else 1.0,
                'pixel_count': width * height,
                'bitrate_per_pixel': bitrate / (width * height) if (width * height) > 0 else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to analyze video content: {e}")
            # Fallback to basic analysis
            return self._get_basic_video_info(video_path)
    
    def _calculate_content_complexity(self, probe_data: Dict, width: int, height: int, duration: float) -> float:
        """Calculate content complexity score (0-10, higher = more complex)"""
        try:
            # Base complexity from resolution and duration
            pixel_complexity = math.log10(width * height) / 6.0 * 10  # Normalize to 0-10
            
            # Estimate complexity from bitrate if available
            format_info = probe_data.get('format', {})
            bitrate = int(format_info.get('bit_rate', 0))
            
            if bitrate > 0:
                # Expected bitrate for resolution
                expected_bitrate = (width * height * 30 * 0.1)  # Rough estimate
                bitrate_complexity = min(bitrate / expected_bitrate * 3, 10)
            else:
                bitrate_complexity = 5  # Default medium complexity
            
            # Combine factors
            complexity = (pixel_complexity * 0.4 + bitrate_complexity * 0.6)
            return max(0, min(10, complexity))
            
        except Exception:
            return 5.0  # Default medium complexity
    
    def _estimate_motion_level(self, video_stream: Dict, duration: float) -> str:
        """Estimate motion level in video"""
        from .ffmpeg_utils import FFmpegUtils
        fps = FFmpegUtils.parse_fps(video_stream.get('r_frame_rate', '30/1'))
        
        if fps >= 50:
            return "high"
        elif fps >= 30:
            return "medium"
        else:
            return "low"
    
    def _estimate_scene_changes(self, duration: float, fps: float) -> int:
        """Estimate number of scene changes"""
        # Rough estimate: average video has a scene change every 3-5 seconds
        return max(1, int(duration / 4))
    
    def _get_basic_video_info(self, video_path: str) -> Dict[str, Any]:
        """Fallback method for basic video info"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
                video_path
            ]
            
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
            
            probe_data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = next(
                (stream for stream in probe_data['streams'] if stream['codec_type'] == 'video'),
                None
            )
            
            if not video_stream:
                raise ValueError("No video stream found in file")
            
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            duration = float(probe_data['format'].get('duration', 0))
            
            return {
                'width': width,
                'height': height,
                'duration': duration,
                'bitrate': int(probe_data['format'].get('bit_rate', 0)),
                'fps': FFmpegUtils.parse_fps(video_stream.get('r_frame_rate', '30/1')),
                'codec': video_stream.get('codec_name', 'unknown'),
                'size_bytes': int(probe_data['format'].get('size', 0)),
                'complexity_score': 5.0,  # Default medium complexity
                'motion_level': 'medium',
                'scene_changes': max(1, int(duration / 4)),
                'aspect_ratio': width / height if height > 0 else 1.0,
                'pixel_count': width * height,
                'bitrate_per_pixel': 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get basic video info: {e}")
            raise
    
    def _create_segments_summary(self, segments_folder: str, segments: List[Dict[str, Any]], 
                               base_name: str, total_duration: float):
        """Create a summary file with segment information"""
        try:
            summary_file = os.path.join(segments_folder, f"{base_name}_segments_summary.txt")
            
            with open(summary_file, 'w') as f:
                f.write(f"Video Segments Summary\n")
                f.write(f"=====================\n\n")
                f.write(f"Base Name: {base_name}\n")
                f.write(f"Total Duration: {total_duration:.2f} seconds\n")
                f.write(f"Number of Segments: {len(segments)}\n")
                f.write(f"Created: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                f.write(f"Segment Details:\n")
                f.write(f"---------------\n")
                
                total_size = 0
                optimized_count = 0
                for segment in segments:
                    f.write(f"Segment {segment['index']:03d}:\n")
                    f.write(f"  Duration: {segment['duration']:.2f}s\n")
                    f.write(f"  Start Time: {segment['start_time']:.2f}s\n")
                    f.write(f"  Size: {segment['size_mb']:.2f}MB\n")
                    f.write(f"  Method: {segment['method']}\n")
                    f.write(f"  Quality Score: {segment['quality_score']:.1f}\n")
                    if segment.get('optimization_applied', False):
                        f.write(f"  Optimization Applied: Yes\n")
                        optimized_count += 1
                    else:
                        f.write(f"  Optimization Applied: No\n")
                    f.write(f"  File: {os.path.basename(segment['path'])}\n\n")
                    
                    total_size += segment['size_mb']
                
                f.write(f"Total Size: {total_size:.2f}MB\n")
                f.write(f"Average Segment Size: {total_size/len(segments):.2f}MB\n")
                f.write(f"Segments Optimized: {optimized_count}/{len(segments)}\n")
            
            logger.info(f"Created segments summary: {summary_file}")
            
        except Exception as e:
            logger.warning(f"Failed to create segments summary: {e}")
    
    def _finalize_segments(self, results: Dict[str, Any], output_base_path: str) -> Dict[str, Any]:
        """Move segments from temp folder to final location"""
        try:
            temp_folder = results['temp_folder']
            base_name = results['base_name']
            
            # Create final segments folder inside output directory
            output_dir = os.path.dirname(output_base_path)
            final_segments_folder = os.path.join(output_dir, f"{base_name}_segments")
            if not os.path.exists(final_segments_folder):
                os.makedirs(final_segments_folder, exist_ok=True)
            
            # Move segments to final location
            final_segments = []
            for segment in results['segments']:
                temp_path = segment['path']
                final_filename = f"{base_name}_segment_{segment['index']:03d}.mp4"
                final_path = os.path.join(final_segments_folder, final_filename)
                
                shutil.move(temp_path, final_path)
                
                # Update segment info
                segment['path'] = final_path
                segment['filename'] = final_filename
                final_segments.append(segment)
            
            # Move summary file if it exists
            summary_file = os.path.join(temp_folder, f"{base_name}_segments_summary.txt")
            if os.path.exists(summary_file):
                final_summary_path = os.path.join(final_segments_folder, f"{base_name}_segments_summary.txt")
                shutil.move(summary_file, final_summary_path)
            
            # Clean up temp folder
            try:
                shutil.rmtree(temp_folder)
            except Exception as e:
                logger.warning(f"Failed to clean up temp folder {temp_folder}: {e}")
            
            return {
                'success': True,
                'segments': final_segments,
                'num_segments': len(final_segments),
                'total_duration': results['total_duration'],
                'segment_duration': results['segment_duration'],
                'output_folder': final_segments_folder,
                'base_name': base_name
            }
            
        except Exception as e:
            logger.error(f"Error finalizing segments: {e}")
            return {
                'success': False,
                'error': str(e),
                'temp_files': results.get('temp_files', [])
            }
    
    def _process_single_video(self, input_video: str, output_path: str, start_time: float,
                             duration: float, video_info: Dict[str, Any], 
                             platform_config: Dict[str, Any], target_size_mb: float) -> Dict[str, Any]:
        """Process video as single file without segmentation"""
        try:
            # Calculate compression parameters
            params = self._calculate_segment_compression_params(video_info, platform_config, target_size_mb, duration)
            
            # Build FFmpeg command
            cmd = [
                'ffmpeg', '-y',
                '-i', input_video,
                '-c:v', params['encoder'],
                '-s', f"{params['width']}x{params['height']}",
                '-r', str(params['fps']),
                '-b:v', params['bitrate'],
                '-c:a', 'aac',
                '-b:a', '96k',
                '-ac', '2',
                '-movflags', '+faststart',
                '-pix_fmt', 'yuv420p',
                output_path
            ]
            # Do not inject generic -hwaccel here; encoding choice already set by encoder
            
            # Execute FFmpeg command
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
            
            if result.returncode == 0:
                file_size = os.path.getsize(output_path) / (1024 * 1024)
                return {
                    'success': True,
                    'size_mb': file_size,
                    'method': 'direct_ffmpeg',
                    'output_file': output_path
                }
            else:
                logger.error(f"FFmpeg failed: {result.stderr}")
                return {
                    'success': False,
                    'error': f"FFmpeg failed: {result.stderr}"
                }
                
        except Exception as e:
            logger.error(f"Error processing single video: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _cleanup_temp_files(self, temp_files: List[str]):
        """Clean up temporary files"""
        for temp_file in temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    logger.debug(f"Cleaned up temp file: {temp_file}")
            except Exception as e:
                logger.warning(f"Failed to clean up temp file {temp_file}: {e}")
    
    def request_shutdown(self):
        """Request graceful shutdown of the segmenter"""
        logger.info("Shutdown requested for video segmenter")
        self.shutdown_requested = True
        if self.current_ffmpeg_process:
            logger.info("Terminating current FFmpeg process...")
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
                    logger.warning("FFmpeg process did not terminate gracefully, forcing kill...")
                    self.current_ffmpeg_process.kill()
                    self.current_ffmpeg_process.wait()
                
                logger.info("FFmpeg process terminated successfully")
            except Exception as e:
                logger.error(f"Error terminating FFmpeg process: {e}")
            finally:
                self.current_ffmpeg_process = None 