"""
FFmpeg Utilities Module
Shared utilities for FFmpeg command building and video analysis
Consolidates duplicate logic across multiple modules
"""

import os
import subprocess
import logging
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)

class FFmpegUtils:
    """Shared utilities for FFmpeg operations"""
    
    @staticmethod
    def get_video_duration(video_path: str) -> float:
        """Get video duration using ffprobe"""
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
                   '-of', 'csv=p=0', video_path]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                return float(result.stdout.strip())
        except Exception as e:
            logger.warning(f"Failed to get video duration for {video_path}: {e}")
        
        return 30.0  # Default fallback
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """Get basic video information using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                import json
                data = json.loads(result.stdout)
                
                # Find video stream
                video_stream = None
                for stream in data.get('streams', []):
                    if stream.get('codec_type') == 'video':
                        video_stream = stream
                        break
                
                if video_stream:
                    return {
                        'width': int(video_stream.get('width', 1920)),
                        'height': int(video_stream.get('height', 1080)),
                        'fps': eval(video_stream.get('r_frame_rate', '30/1')),
                        'duration': float(data.get('format', {}).get('duration', 30.0)),
                        'bitrate': int(data.get('format', {}).get('bit_rate', 1000000)) // 1000,
                        'codec': video_stream.get('codec_name', 'unknown')
                    }
        except Exception as e:
            logger.warning(f"Failed to get video info for {video_path}: {e}")
        
        # Return default values
        return {
            'width': 1920,
            'height': 1080,
            'fps': 30.0,
            'duration': 30.0,
            'bitrate': 1000,
            'codec': 'unknown'
        }
    
    @staticmethod
    def build_base_ffmpeg_command(input_path: str, output_path: str, params: Dict[str, Any]) -> List[str]:
        """Build base FFmpeg command with common settings"""
        cmd = ['ffmpeg', '-y']
        
        # Input
        cmd.extend(['-i', input_path])
        
        # Hardware acceleration
        if params.get('acceleration_type') == 'nvidia':
            cmd.extend(['-hwaccel', 'cuda'])
        elif params.get('acceleration_type') == 'amd':
            cmd.extend(['-hwaccel', 'auto'])
        elif params.get('acceleration_type') == 'intel':
            cmd.extend(['-hwaccel', 'qsv'])
        
        # Video encoding
        if 'encoder' in params:
            cmd.extend(['-c:v', params['encoder']])
            
            # Add AMD-specific encoder options
            if params['encoder'] in ['h264_amf', 'hevc_amf', 'av1_amf']:
                from .hardware_detector import HardwareDetector
                hardware = HardwareDetector()
                amd_options = hardware.get_amd_encoder_options(params['encoder'])
                cmd.extend(amd_options)
        
        return cmd
    
    @staticmethod
    def add_video_settings(cmd: List[str], params: Dict[str, Any]) -> List[str]:
        """Add video encoding settings to FFmpeg command"""
        # Resolution
        if 'width' in params and 'height' in params:
            cmd.extend(['-s', f"{params['width']}x{params['height']}"])
        
        # Frame rate
        if 'fps' in params:
            cmd.extend(['-r', str(params['fps'])])
        
        # Quality settings based on acceleration type
        acceleration_type = params.get('acceleration_type', 'software')
        encoder = params.get('encoder', '')
        
        if acceleration_type == 'software':
            # Software encoding quality settings
            if 'crf' in params:
                cmd.extend(['-crf', str(params['crf'])])
            if 'preset' in params:
                cmd.extend(['-preset', params['preset']])
        
        elif acceleration_type == 'amd' and 'amf' in encoder:
            # AMD AMF-specific quality settings - simplified for maximum compatibility
            if 'crf' in params:
                # AMF uses qp instead of crf, but keep it simple
                qp_value = max(18, min(51, int(params['crf'])))
                cmd.extend(['-qp', str(qp_value)])
            
            # AMD-specific optimizations - simplified for compatibility
            # Use baseline profile for maximum compatibility
            cmd.extend(['-profile:v', 'baseline'])
            
            # Remove any problematic parameters that might cause invalid argument errors
        
        elif acceleration_type == 'nvidia' and 'nvenc' in encoder:
            # NVIDIA NVENC-specific settings
            if 'crf' in params:
                cmd.extend(['-cq', str(params['crf'])])
            cmd.extend(['-preset', 'p4'])  # Good balance of quality and speed
            
        elif acceleration_type == 'intel' and 'qsv' in encoder:
            # Intel QuickSync-specific settings
            if 'crf' in params:
                cmd.extend(['-global_quality', str(params['crf'])])
        
        return cmd
    
    @staticmethod
    def add_bitrate_control(cmd: List[str], params: Dict[str, Any], buffer_multiplier: float = 2.0) -> List[str]:
        """Add bitrate control settings to FFmpeg command"""
        if 'bitrate' in params:
            bitrate = params['bitrate']
            
            # AMD AMF bitrate limiting - prevent extremely high bitrates that cause failures
            if params.get('acceleration_type') == 'amd' and 'amf' in params.get('encoder', ''):
                # Limit AMD AMF bitrate to prevent failures (max ~4 Mbps for better compatibility)
                max_amd_bitrate = 4000  # 4 Mbps - very conservative
                if bitrate > max_amd_bitrate:
                    logger.warning(f"Limiting AMD AMF bitrate from {bitrate}k to {max_amd_bitrate}k for compatibility")
                    bitrate = max_amd_bitrate
            
            cmd.extend(['-b:v', f"{bitrate}k"])
            
            # Maxrate with optional multiplier
            maxrate_multiplier = params.get('maxrate_multiplier', 1.2)
            maxrate = int(bitrate * maxrate_multiplier)
            
            # For AMD AMF, use very conservative maxrate
            if params.get('acceleration_type') == 'amd' and 'amf' in params.get('encoder', ''):
                maxrate_multiplier = 1.05  # Very conservative for AMD
                maxrate = int(bitrate * maxrate_multiplier)
            
            cmd.extend(['-maxrate', f"{maxrate}k"])
            
            # Buffer size - very conservative for AMD AMF
            if params.get('acceleration_type') == 'amd' and 'amf' in params.get('encoder', ''):
                buffer_multiplier = 1.2  # Very conservative buffer
            
            cmd.extend(['-bufsize', f"{int(bitrate * buffer_multiplier)}k"])
        
        return cmd
    
    @staticmethod
    def add_audio_settings(cmd: List[str], params: Dict[str, Any] = None) -> List[str]:
        """Add audio encoding settings to FFmpeg command"""
        if params and params.get('audio_bitrate'):
            cmd.extend(['-c:a', 'aac', '-b:a', f"{params['audio_bitrate']}k"])
        else:
            cmd.extend(['-c:a', 'aac', '-b:a', '96k'])
        
        # Audio channels
        if params and params.get('audio_channels'):
            cmd.extend(['-ac', str(params['audio_channels'])])
        else:
            cmd.extend(['-ac', '2'])  # Default stereo
        
        return cmd
    
    @staticmethod
    def add_output_optimizations(cmd: List[str], output_path: str) -> List[str]:
        """Add output optimization settings to FFmpeg command"""
        cmd.extend(['-movflags', '+faststart', '-pix_fmt', 'yuv420p'])
        cmd.append(output_path)
        return cmd
    
    @staticmethod
    def build_standard_ffmpeg_command(input_path: str, output_path: str, params: Dict[str, Any]) -> List[str]:
        """Build a standard FFmpeg command with common settings"""
        cmd = FFmpegUtils.build_base_ffmpeg_command(input_path, output_path, params)
        cmd = FFmpegUtils.add_video_settings(cmd, params)
        cmd = FFmpegUtils.add_bitrate_control(cmd, params)
        cmd = FFmpegUtils.add_audio_settings(cmd, params)
        cmd = FFmpegUtils.add_output_optimizations(cmd, output_path)
        
        logger.debug(f"Standard FFmpeg command: {' '.join(cmd)}")
        return cmd
    
    @staticmethod
    def build_two_pass_command(input_path: str, output_path: str, params: Dict[str, Any], 
                              pass_num: int, log_file: str) -> List[str]:
        """Build two-pass FFmpeg command"""
        cmd = FFmpegUtils.build_base_ffmpeg_command(input_path, output_path, params)
        cmd = FFmpegUtils.add_video_settings(cmd, params)
        
        # Two-pass specific settings
        cmd.extend(['-pass', str(pass_num)])
        cmd.extend(['-passlogfile', log_file])
        
        cmd = FFmpegUtils.add_bitrate_control(cmd, params, buffer_multiplier=2.0)
        
        if pass_num == 1:
            # First pass: analysis only
            cmd.extend(['-an', '-f', 'null'])
            if os.name == 'nt':  # Windows
                cmd.append('NUL')
            else:  # Unix/Linux/Mac
                cmd.append('/dev/null')
        else:
            # Second pass: encoding with audio
            cmd = FFmpegUtils.add_audio_settings(cmd, params)
            cmd = FFmpegUtils.add_output_optimizations(cmd, output_path)
        
        logger.debug(f"Two-pass FFmpeg command (pass {pass_num}): {' '.join(cmd)}")
        return cmd
    
    @staticmethod
    def execute_ffmpeg_with_progress(cmd: List[str], duration: float = None, 
                                   description: str = "Processing") -> bool:
        """Execute FFmpeg command with progress tracking"""
        try:
            if duration and duration > 5:  # Only show progress for longer operations
                from tqdm import tqdm
                
                process = subprocess.Popen(
                    cmd, 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE, 
                    universal_newlines=True
                )
                
                progress_bar = tqdm(total=100, desc=description, unit="%")
                
                # Parse progress from stderr
                for line in process.stderr:
                    if 'time=' in line:
                        try:
                            time_str = line.split('time=')[1].split()[0]
                            # Parse time format (HH:MM:SS.ms)
                            time_parts = time_str.split(':')
                            if len(time_parts) == 3:
                                hours = float(time_parts[0])
                                minutes = float(time_parts[1])
                                seconds = float(time_parts[2])
                                current_time = hours * 3600 + minutes * 60 + seconds
                                
                                progress = min(int((current_time / duration) * 100), 100)
                                progress_bar.n = progress
                                progress_bar.refresh()
                        except:
                            continue
                
                process.wait()
                progress_bar.close()
                
                return process.returncode == 0
            else:
                # Simple execution without progress
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
                return result.returncode == 0
                
        except subprocess.TimeoutExpired:
            logger.error("FFmpeg command timed out")
            return False
        except Exception as e:
            logger.error(f"FFmpeg execution failed: {e}")
            return False
    
    @staticmethod
    def extract_video_segment(input_path: str, output_path: str, 
                            start_time: float, duration: float) -> bool:
        """Extract a segment from video"""
        cmd = [
            'ffmpeg', '-ss', str(start_time), '-i', input_path,
            '-t', str(duration), '-c', 'copy', '-y', output_path
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=120)
            return result.returncode == 0
        except Exception as e:
            logger.error(f"Failed to extract video segment: {e}")
            return False 