"""
FFmpeg Utilities Module
Shared utilities for FFmpeg command building and video analysis
Consolidates duplicate logic across multiple modules
"""

import os
import subprocess
import logging
from typing import Dict, Any, List, Optional, Tuple
import shlex

logger = logging.getLogger(__name__)

class FFmpegUtils:
    """Shared utilities for FFmpeg operations"""
    
    @staticmethod
    def _safe_file_path(file_path: str) -> str:
        """Safely handle file paths with special characters"""
        try:
            # Convert to absolute path and normalize
            abs_path = os.path.abspath(file_path)
            # On Windows, ensure proper path handling
            if os.name == 'nt':
                # Remove any double quotes that might cause issues
                abs_path = abs_path.replace('"', '')
            return abs_path
        except Exception as e:
            logger.warning(f"Error normalizing file path {file_path}: {e}")
            return file_path
    
    @staticmethod
    def get_video_duration(video_path: str) -> float:
        """Get video duration using ffprobe"""
        try:
            # Safely handle the file path
            safe_path = FFmpegUtils._safe_file_path(video_path)
            
            # Check if file exists before running ffprobe
            if not os.path.exists(safe_path):
                logger.error(f"Video file not found: {safe_path}")
                return 30.0  # Default fallback
            
            # Check if file is accessible
            if not os.access(safe_path, os.R_OK):
                logger.error(f"Video file not accessible: {safe_path}")
                return 30.0  # Default fallback
            
            cmd = ['ffprobe', '-v', 'quiet', '-show_entries', 'format=duration', 
                   '-of', 'csv=p=0', safe_path]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            
            if result.returncode == 0:
                duration_str = result.stdout.strip()
                if duration_str and duration_str != 'N/A':
                    return float(duration_str)
                else:
                    logger.warning(f"Invalid duration returned for {video_path}: {duration_str}")
                    return 30.0  # Default fallback
            else:
                logger.warning(f"FFprobe failed for {video_path}: {result.stderr}")
                return 30.0  # Default fallback
                
        except FileNotFoundError as e:
            logger.error(f"File not found error for {safe_path}: {e}")
            return 30.0  # Default fallback
        except Exception as e:
            logger.warning(f"Failed to get video duration for {safe_path}: {e}")
            return 30.0  # Default fallback
    
    @staticmethod
    def get_video_info(video_path: str) -> Dict[str, Any]:
        """Get basic video information using ffprobe"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=30)
            
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
    def get_default_thread_counts() -> Tuple[int, int]:
        """Return sensible defaults for FFmpeg thread and filter thread counts.

        Caps threads to avoid contention on machines with many cores.
        """
        try:
            cpu_count = os.cpu_count() or 4
            threads = min(8, max(2, cpu_count))
            filter_threads = min(8, max(2, cpu_count // 2 or 1))
            return threads, filter_threads
        except Exception:
            return 4, 2

    @staticmethod
    def add_ffmpeg_perf_flags(cmd: List[str], threads: Optional[int] = None, filter_threads: Optional[int] = None,
                              add_probe: bool = True) -> List[str]:
        """Insert common performance and noise-reduction flags into an FFmpeg command list.

        - Adds: -hide_banner -loglevel error -probesize 2M -analyzeduration 10M
        - Adds: -threads N -filter_threads N

        This function mutates and returns the same list for convenience.
        """
        try:
            if not cmd:
                return cmd
            # Determine insertion point (after program name 'ffmpeg')
            insert_index = 1 if cmd[0].lower() == 'ffmpeg' else 0

            # Threads
            t, ft = FFmpegUtils.get_default_thread_counts()
            if threads is None:
                threads = t
            if filter_threads is None:
                filter_threads = ft

            perf_flags: List[str] = []
            perf_flags.extend(['-hide_banner', '-loglevel', 'error'])
            if add_probe:
                perf_flags.extend(['-probesize', '2M', '-analyzeduration', '10M'])
            # Only add if not already present
            def _missing(flag: str) -> bool:
                try:
                    return flag not in cmd
                except Exception:
                    return True

            if _missing('-threads'):
                perf_flags.extend(['-threads', str(threads)])
            if _missing('-filter_threads'):
                perf_flags.extend(['-filter_threads', str(filter_threads)])

            # Insert after program name
            for i, flag in enumerate(perf_flags):
                cmd.insert(insert_index + i, flag)
        except Exception:
            # Best-effort; ignore if anything goes wrong
            pass
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
            # AMD AMF-specific quality settings - ultra-conservative for maximum compatibility
            # Avoid advanced features that might cause "Invalid argument" errors
            
            if 'crf' in params:
                # AMF uses qp instead of crf - use conservative values
                # Map CRF to QP more conservatively to avoid errors
                crf_value = params['crf']
                if crf_value < 18:
                    qp_value = 18  # Minimum safe QP
                elif crf_value > 40:
                    qp_value = 40  # Maximum safe QP for compatibility
                else:
                    qp_value = int(crf_value)
                cmd.extend(['-qp', str(qp_value)])
            
            # Use most conservative profile for maximum compatibility
            cmd.extend(['-profile:v', 'baseline'])
            
            # Conservative level setting
            cmd.extend(['-level', '3.1'])
            
            # Disable advanced features that might cause issues
            cmd.extend(['-refs', '1'])  # Single reference frame
            
            # Remove any problematic parameters that might cause invalid argument errors
            # Don't add any experimental or advanced AMD-specific flags
        
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
                # Ultra-conservative bitrate limiting for AMD AMF to prevent failures
                max_amd_bitrate = 3000  # 3 Mbps - extremely conservative
                if bitrate > max_amd_bitrate:
                    logger.warning(f"Limiting AMD AMF bitrate from {bitrate}k to {max_amd_bitrate}k for compatibility")
                    bitrate = max_amd_bitrate
                
                # Also ensure minimum bitrate to avoid edge cases
                min_amd_bitrate = 200  # 200 kbps minimum
                if bitrate < min_amd_bitrate:
                    logger.info(f"Increasing AMD AMF bitrate from {bitrate}k to {min_amd_bitrate}k for stability")
                    bitrate = min_amd_bitrate
            
            cmd.extend(['-b:v', f"{bitrate}k"])
            
            # Maxrate with optional multiplier
            maxrate_multiplier = params.get('maxrate_multiplier', 1.2)
            maxrate = int(bitrate * maxrate_multiplier)
            
            # For AMD AMF, use very conservative maxrate
            if params.get('acceleration_type') == 'amd' and 'amf' in params.get('encoder', ''):
                maxrate_multiplier = 1.02  # Extremely conservative for AMD (2% above target)
                maxrate = int(bitrate * maxrate_multiplier)
                # Ensure maxrate is at least slightly above bitrate
                if maxrate <= bitrate:
                    maxrate = bitrate + 50  # Add 50kbps minimum headroom
            
            cmd.extend(['-maxrate', f"{maxrate}k"])
            
            # Buffer size - very conservative for AMD AMF
            if params.get('acceleration_type') == 'amd' and 'amf' in params.get('encoder', ''):
                buffer_multiplier = 1.1  # Extremely conservative buffer (10% above bitrate)
                # Ensure minimum buffer size
                buffer_size = max(int(bitrate * buffer_multiplier), bitrate + 100)
                cmd.extend(['-bufsize', f"{buffer_size}k"])
            else:
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

    @staticmethod
    def get_detailed_file_specifications(file_path: str) -> Dict[str, Any]:
        """
        Get detailed file specifications for logging purposes
        
        Args:
            file_path: Path to the video or GIF file
            
        Returns:
            Dictionary containing detailed file specifications
        """
        try:
            # Check if it's a GIF file
            if file_path.lower().endswith('.gif'):
                return FFmpegUtils._get_gif_specifications(file_path)
            else:
                return FFmpegUtils._get_video_specifications(file_path)
                
        except Exception as e:
            logger.warning(f"Failed to get detailed specifications for {file_path}: {e}")
            return {
                'file_type': 'unknown',
                'error': str(e)
            }
    
    @staticmethod
    def _get_video_specifications(video_path: str) -> Dict[str, Any]:
        """Get detailed video file specifications"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
            
            import json
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            audio_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                elif stream.get('codec_type') == 'audio':
                    audio_stream = stream
            
            format_info = data.get('format', {})
            
            # Basic file info
            file_size = int(format_info.get('size', 0))
            duration = float(format_info.get('duration', 0))
            bitrate = int(format_info.get('bit_rate', 0))
            
            specs = {
                'file_type': 'video',
                'file_size_mb': file_size / (1024 * 1024),
                'duration_seconds': duration,
                'bitrate_kbps': bitrate // 1000,
                'container_format': format_info.get('format_name', 'unknown')
            }
            
            # Video stream specifications
            if video_stream:
                width = int(video_stream.get('width', 0))
                height = int(video_stream.get('height', 0))
                fps_str = video_stream.get('r_frame_rate', '30/1')
                
                # Parse FPS (handle fractions like "30000/1001")
                try:
                    if '/' in fps_str:
                        num, den = map(int, fps_str.split('/'))
                        fps = num / den
                    else:
                        fps = float(fps_str)
                except:
                    fps = 30.0
                
                specs.update({
                    'width': width,
                    'height': height,
                    'resolution': f"{width}x{height}",
                    'fps': fps,
                    'video_codec': video_stream.get('codec_name', 'unknown'),
                    'video_profile': video_stream.get('profile', 'unknown'),
                    'pixel_format': video_stream.get('pix_fmt', 'unknown'),
                    'aspect_ratio': video_stream.get('display_aspect_ratio', 'unknown'),
                    'frame_count': int(video_stream.get('nb_frames', 0)),
                    'video_bitrate_kbps': int(video_stream.get('bit_rate', 0)) // 1000 if video_stream.get('bit_rate') else 0
                })
            
            # Audio stream specifications
            if audio_stream:
                specs.update({
                    'audio_codec': audio_stream.get('codec_name', 'unknown'),
                    'audio_channels': int(audio_stream.get('channels', 0)),
                    'audio_sample_rate': int(audio_stream.get('sample_rate', 0)),
                    'audio_bitrate_kbps': int(audio_stream.get('bit_rate', 0)) // 1000 if audio_stream.get('bit_rate') else 0
                })
            
            return specs
            
        except Exception as e:
            logger.warning(f"Failed to get video specifications for {video_path}: {e}")
            return {
                'file_type': 'video',
                'error': str(e)
            }
    
    @staticmethod
    def _get_gif_specifications(gif_path: str) -> Dict[str, Any]:
        """Get detailed GIF file specifications"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', gif_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
            
            import json
            data = json.loads(result.stdout)
            
            format_info = data.get('format', {})
            video_stream = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), None)
            
            # Basic file info
            file_size = int(format_info.get('size', 0))
            duration = float(format_info.get('duration', 0))
            
            specs = {
                'file_type': 'gif',
                'file_size_mb': file_size / (1024 * 1024),
                'duration_seconds': duration,
                'container_format': 'gif'
            }
            
            # GIF-specific specifications
            if video_stream:
                width = int(video_stream.get('width', 0))
                height = int(video_stream.get('height', 0))
                fps_str = video_stream.get('r_frame_rate', '10/1')
                
                # Parse FPS for GIF
                try:
                    if '/' in fps_str:
                        num, den = map(int, fps_str.split('/'))
                        fps = num / den
                    else:
                        fps = float(fps_str)
                except:
                    fps = 10.0
                
                frame_count = int(video_stream.get('nb_frames', 0))
                
                specs.update({
                    'width': width,
                    'height': height,
                    'resolution': f"{width}x{height}",
                    'fps': fps,
                    'frame_count': frame_count,
                    'video_codec': 'gif',
                    'pixel_format': video_stream.get('pix_fmt', 'unknown'),
                    'colors': 256,  # GIFs typically use 256 colors
                    'loop_count': format_info.get('tags', {}).get('loop', '0') if format_info.get('tags') else '0'
                })
            
            return specs
            
        except Exception as e:
            logger.warning(f"Failed to get GIF specifications for {gif_path}: {e}")
            return {
                'file_type': 'gif',
                'error': str(e)
            } 

    @staticmethod
    def format_file_specifications_for_logging(specs: Dict[str, Any]) -> str:
        """
        Format file specifications into a readable log message
        
        Args:
            specs: Dictionary containing file specifications from get_detailed_file_specifications
            
        Returns:
            Formatted string for logging
        """
        if 'error' in specs:
            return f"File specifications unavailable: {specs['error']}"
        
        file_type = specs.get('file_type', 'unknown')
        
        if file_type == 'video':
            return FFmpegUtils._format_video_specifications_for_logging(specs)
        elif file_type == 'gif':
            return FFmpegUtils._format_gif_specifications_for_logging(specs)
        else:
            return f"Unknown file type: {file_type}"
    
    @staticmethod
    def _format_video_specifications_for_logging(specs: Dict[str, Any]) -> str:
        """Format video specifications for logging"""
        parts = []
        
        # Basic info
        parts.append(f"Size: {specs.get('file_size_mb', 0):.2f}MB")
        parts.append(f"Duration: {specs.get('duration_seconds', 0):.2f}s")
        
        # Video specs
        if 'resolution' in specs:
            parts.append(f"Resolution: {specs['resolution']}")
        if 'fps' in specs:
            parts.append(f"FPS: {specs['fps']:.1f}")
        if 'video_codec' in specs:
            parts.append(f"Codec: {specs['video_codec']}")
        if 'bitrate_kbps' in specs and specs['bitrate_kbps'] > 0:
            parts.append(f"Bitrate: {specs['bitrate_kbps']}kbps")
        if 'frame_count' in specs and specs['frame_count'] > 0:
            parts.append(f"Frames: {specs['frame_count']}")
        
        # Audio specs (if available)
        if 'audio_codec' in specs and specs['audio_codec'] != 'unknown':
            parts.append(f"Audio: {specs['audio_codec']}")
            if 'audio_channels' in specs:
                parts.append(f"Channels: {specs['audio_channels']}")
            if 'audio_sample_rate' in specs:
                parts.append(f"Sample Rate: {specs['audio_sample_rate']}Hz")
        
        return f"Video specs: {' | '.join(parts)}"
    
    @staticmethod
    def _format_gif_specifications_for_logging(specs: Dict[str, Any]) -> str:
        """Format GIF specifications for logging"""
        parts = []
        
        # Basic info
        parts.append(f"Size: {specs.get('file_size_mb', 0):.2f}MB")
        parts.append(f"Duration: {specs.get('duration_seconds', 0):.2f}s")
        
        # GIF specs
        if 'resolution' in specs:
            parts.append(f"Resolution: {specs['resolution']}")
        if 'fps' in specs:
            parts.append(f"FPS: {specs['fps']:.1f}")
        if 'frame_count' in specs and specs['frame_count'] > 0:
            parts.append(f"Frames: {specs['frame_count']}")
        if 'colors' in specs:
            parts.append(f"Colors: {specs['colors']}")
        if 'loop_count' in specs:
            parts.append(f"Loop: {specs['loop_count']}")
        
        return f"GIF specs: {' | '.join(parts)}" 