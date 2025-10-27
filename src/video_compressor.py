"""
Video Compression Module
Handles video compression with hardware acceleration for social media platforms
Enhanced with dynamic optimization for maximum quality within size constraints
"""

import os
import subprocess
import shutil
import math
import threading
import time
from typing import Dict, Any, Optional, Tuple, List
import logging
from pathlib import Path
from tqdm import tqdm
import json
from functools import lru_cache

from .hardware_detector import HardwareDetector
from .config_manager import ConfigManager
from .advanced_optimizer import AdvancedVideoOptimizer
from .performance_enhancer import PerformanceEnhancer
from .ffmpeg_utils import FFmpegUtils
from .video_segmenter import VideoSegmenter
from .quality_scorer import QualityScorer

logger = logging.getLogger(__name__)

class GracefulCancellation(Exception):
    """Raised to indicate a user-requested shutdown/cancellation."""
    pass


class DynamicVideoCompressor:
    def __init__(self, config_manager: ConfigManager, hardware_detector: HardwareDetector):
        self.config = config_manager
        self.hardware = hardware_detector
        self.temp_dir = self.config.get_temp_dir()
        
        # Performance enhancement
        self.performance_enhancer = PerformanceEnhancer(config_manager)
        
        # Shutdown handling
        self.shutdown_requested = False
        self.current_ffmpeg_process = None
        self._ffmpeg_processes = []  # Track all running FFmpeg processes
        self._shutdown_lock = threading.Lock()  # Protect shutdown state
        
        # Statistics
        self.stats = {
            'total_compressions': 0,
            'successful_compressions': 0,
            'failed_compressions': 0,
            'hardware_fallbacks': 0,
            'average_compression_time': 0,
            'total_processing_time': 0
        }
        
        # Initialize advanced components
        self.advanced_optimizer = AdvancedVideoOptimizer(config_manager, hardware_detector)
        self.video_segmenter = VideoSegmenter(config_manager, hardware_detector)
        self.quality_scorer = QualityScorer(config_manager)
        
        # Initialize CAE components
        from .quality_gates import QualityGates
        from .artifact_detection import ArtifactDetector
        self.quality_gates = QualityGates(config_manager)
        self.artifact_detector = ArtifactDetector(config_manager)
        
        # Optimize system resources
        self.system_optimizations = self.performance_enhancer.optimize_system_resources()
        
        # Cache for expensive calculations
        self._resolution_cache = {}

    # ===== New helpers for refined final-pass encoding =====
    def _calculate_safe_bitrate_cap(self, width: int, height: int, fps: float, duration: float) -> int:
        """Calculate reasonable bitrate cap based on video characteristics"""
        pixels_per_sec = width * height * fps
        # Cap at 0.15 bits-per-pixel (high quality ceiling)
        max_bpp = 0.15
        cap = int(pixels_per_sec * max_bpp / 1000)
        # Absolute bounds: 500k-8000k
        return max(500, min(cap, 8000))
    
    def _cleanup_two_pass_logs(self, log_base: str):
        """Clean up all two-pass log files reliably"""
        for ext in ['-0.log', '-0.log.mbtree', '-0.log.temp', '-0.log.mbtree.temp']:
            try:
                log_path = f"{log_base}{ext}"
                if os.path.exists(log_path):
                    os.remove(log_path)
            except Exception as e:
                logger.debug(f"Could not remove {log_path}: {e}")
    
    def _binary_search_bitrate(self, input_path: str, output_path: str, params: Dict[str, Any],
                               video_info: Dict[str, Any], target_mb: float,
                               low_kbps: int, high_kbps: int, log_base: str,
                               min_utilization: float = 0.95, max_utilization: float = 0.98,
                               max_iterations: int = 3) -> Optional[Tuple[int, float]]:
        """Binary search for optimal bitrate within target utilization range.
        Returns (best_bitrate, best_size_mb) or None if failed."""
        
        best_bitrate = low_kbps
        best_size = 0.0
        
        for iteration in range(max_iterations):
            # Calculate midpoint bitrate
            mid_kbps = (low_kbps + high_kbps) // 2
            
            # Test this bitrate
            params_test = params.copy()
            params_test['bitrate'] = mid_kbps
            
            logger.info(f"Binary search iteration {iteration+1}: testing bitrate {mid_kbps}k (range: {low_kbps}k-{high_kbps}k)")
            
            # Build and execute command
            cmd = self._build_two_pass_command(input_path, output_path, params_test, pass_num=2, log_file=log_base)
            success = self._execute_ffmpeg_with_progress(cmd, video_info['duration'])
            
            if not success or not os.path.exists(output_path):
                logger.warning(f"Binary search iteration {iteration+1} failed")
                high_kbps = mid_kbps - 1
                continue
            
            test_size = os.path.getsize(output_path) / (1024 * 1024)
            utilization = test_size / target_mb
            
            logger.info(f"Binary search iteration {iteration+1}: size={test_size:.2f}MB, utilization={utilization*100:.1f}%")
            
            # Check if within target range
            if min_utilization <= utilization <= max_utilization:
                logger.info(f"Binary search converged: bitrate={mid_kbps}k, size={test_size:.2f}MB")
                return mid_kbps, test_size
            elif utilization > max_utilization:
                # Too large, reduce bitrate
                high_kbps = mid_kbps - 1
            else:
                # Too small, increase bitrate
                low_kbps = mid_kbps + 1
                if test_size > best_size and test_size <= target_mb:
                    best_bitrate = mid_kbps
                    best_size = test_size
            
            # Stop if range collapsed
            if low_kbps >= high_kbps:
                break
        
        # Return best found if we have one
        if best_size > 0:
            logger.info(f"Binary search ended: best bitrate={best_bitrate}k, size={best_size:.2f}MB")
            return best_bitrate, best_size
        
        return None
    
    def _calculate_target_video_kbps(self, target_size_mb: float, duration_s: float, audio_kbps: int) -> int:
        total_bits = target_size_mb * 8 * 1024 * 1024
        kbps = int(max((total_bits / max(duration_s, 1.0) / 1000) - audio_kbps, 64))
        return kbps
    
    def _calculate_optimal_fps(self, video_info: Dict[str, Any], target_size_mb: float, strategy: str = 'balanced') -> float:
        """Determine best FPS based on motion, size constraints, and strategy"""
        original_fps = video_info.get('fps', 30.0)
        motion_level = video_info.get('motion_level', 'medium')
        duration = video_info.get('duration', 0)
        original_size_mb = video_info.get('size_bytes', 0) / (1024 * 1024)
        
        # Calculate compression ratio to gauge difficulty
        compression_ratio = original_size_mb / max(target_size_mb, 0.1) if original_size_mb > 0 else 1.0
        
        # Base FPS selection based on motion level
        if motion_level == 'high':
            base_fps = min(original_fps, 60)  # Keep high FPS for motion
        elif motion_level == 'low':
            base_fps = 24  # Low motion can use 24fps
        else:
            base_fps = 30  # Medium motion uses 30fps
        
        # Adjust based on strategy
        if strategy == 'aggressive':
            # Reduce FPS more aggressively
            if compression_ratio > 10:
                base_fps = min(base_fps * 0.5, 24)
            elif compression_ratio > 5:
                base_fps = min(base_fps * 0.7, 30)
        elif strategy == 'quality':
            # Try to preserve FPS
            base_fps = min(original_fps, 60)
        
        # Further reduce if extreme compression is needed
        if compression_ratio > 15:
            base_fps = max(base_fps * 0.6, 15)
        
        # Always cap at 60 and floor at 10
        return max(10, min(base_fps, 60))
    
    def _calculate_optimal_audio_bitrate(self, input_path: str, video_info: Dict[str, Any], 
                                        target_size_mb: float) -> int:
        """Choose audio bitrate based on input quality and size budget"""
        duration = video_info.get('duration', 0)
        
        # Try to probe input audio bitrate
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-select_streams', 'a:0',
                '-show_entries', 'stream=bit_rate', '-of', 'json', input_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                probe_data = json.loads(result.stdout)
                if probe_data.get('streams') and len(probe_data['streams']) > 0:
                    input_audio_bitrate = int(probe_data['streams'][0].get('bit_rate', 0)) // 1000
                else:
                    input_audio_bitrate = 128  # Default if no audio stream
            else:
                input_audio_bitrate = 128
        except Exception:
            input_audio_bitrate = 128  # Default on error
        
        # Calculate what we can afford (reserve 10-15% of budget for audio)
        total_bitrate = (target_size_mb * 8 * 1024) / max(duration, 1)  # kbps
        audio_budget = total_bitrate * 0.12  # 12% for audio
        
        # Scale down from input quality, never exceed it
        target_audio = min(input_audio_bitrate, audio_budget)
        
        # Quantize to standard bitrates
        if target_audio >= 160:
            return min(192, input_audio_bitrate)
        elif target_audio >= 112:
            return min(128, input_audio_bitrate)
        elif target_audio >= 80:
            return min(96, input_audio_bitrate)
        elif target_audio >= 56:
            return 64
        elif target_audio >= 40:
            return 48
        else:
            return 32  # Minimum acceptable quality

    def _utilization_ratio(self, actual_bytes: int, target_mb: float) -> float:
        return float(actual_bytes) / max(target_mb * 1024 * 1024, 1.0)

    def _ensure_min_bpp(self, width: int, height: int, fps: float, kbps: int, min_bpp: float) -> bool:
        try:
            pixels_per_sec = max(1, int(width) * int(height)) * max(1.0, float(fps))
            current_bpp = (int(kbps) * 1000.0) / pixels_per_sec
            return current_bpp >= float(min_bpp)
        except Exception:
            return True
    
    def _validate_encoding_parameters(self, width: int, height: int, fps: float, bitrate_kbps: int) -> bool:
        """Ensure encoding parameters are feasible and will produce acceptable quality
        
        Returns True if parameters are valid, False if BPP is too low for acceptable quality
        """
        try:
            pixels_per_sec = width * height * fps
            if pixels_per_sec <= 0 or bitrate_kbps <= 0:
                return False
            
            actual_bpp = (bitrate_kbps * 1000) / pixels_per_sec
            
            # Absolute minimum for any acceptable quality
            # Below 0.012, quality will be severely degraded (heavy pixelation)
            min_acceptable_bpp = 0.012
            
            if actual_bpp < min_acceptable_bpp:
                logger.warning(
                    f"QUALITY WARNING: BPP too low for acceptable quality: {actual_bpp:.4f} < {min_acceptable_bpp:.4f} "
                    f"at {width}x{height}@{fps:.1f}fps with {bitrate_kbps}kbps. "
                    f"This will result in severe pixelation. Consider reducing resolution or increasing target size."
                )
                return False
            
            # Warn if below recommended minimum but still technically feasible
            recommended_min_bpp = 0.018
            if actual_bpp < recommended_min_bpp:
                logger.info(
                    f"Quality notice: BPP below recommended: {actual_bpp:.4f} < {recommended_min_bpp:.4f} "
                    f"at {width}x{height}@{fps:.1f}fps with {bitrate_kbps}kbps. "
                    f"Quality may be lower than optimal."
                )
            
            return True
        except Exception as e:
            logger.debug(f"Parameter validation error: {e}")
            return True  # Allow on error to avoid blocking valid encodes

    def _calculate_codec_efficiency(self, codec_key: str, complexity: float) -> float:
        """Get realistic bits-per-pixel efficiency for codec
        Returns efficiency factor (higher = more efficient = fewer bits needed)
        """
        # Base efficiency values (bits per pixel at medium complexity)
        # These are empirically derived from real-world encoding
        base_efficiency = {
            'hevc_10bit': 0.017,  # Most efficient
            'hevc_8bit': 0.019,
            'h264_10bit': 0.021,
            'h264_8bit': 0.025,   # Least efficient, needs most bits
        }
        
        # Get base for this codec
        base_bpp = base_efficiency.get(codec_key, 0.025)
        
        # Adjust for complexity (higher complexity = needs more bits = lower efficiency)
        # Complexity ranges 0-10, normalize to 0-1
        complexity_factor = 1.0 + (complexity / 10.0) * 0.4  # Up to 40% more bits for complex content
        
        adjusted_bpp = base_bpp * complexity_factor
        
        # Return as efficiency factor (inverted, so higher = better)
        return 1.0 / adjusted_bpp
    
    def _derive_codec_priority(self) -> List[str]:
        """Derive ordered codec preference keys like ['hevc_10bit', 'hevc_8bit', 'h264_10bit', 'h264_8bit']
        based on config and runtime capability.
        """
        # Read config
        allow_hevc = bool(self.config.get('video_compression.codec.allow_hevc', True))
        allow_h264_high10 = bool(self.config.get('video_compression.codec.allow_h264_high10', True))
        configured_priority = self.config.get('video_compression.codec.priority',
                                              ['hevc_10bit', 'hevc_8bit', 'h264_10bit', 'h264_8bit'])

        # Detect runtime capabilities
        try:
            caps = self.hardware.detect_codec_capabilities()
        except Exception:
            caps = {
                'hevc_10bit': True,
                'hevc_8bit': True,
                'h264_10bit': False,
                'h264_8bit': True,
            }

        ordered: List[str] = []
        for key in configured_priority:
            if key.startswith('hevc') and not allow_hevc:
                continue
            if key == 'h264_10bit' and not allow_h264_high10:
                continue
            if caps.get(key, False):
                ordered.append(key)
            else:
                # If capability is unknown for 8-bit paths, allow graceful attempt
                if key.endswith('8bit'):
                    ordered.append(key)
        # Ensure at least one fallback
        if not ordered:
            ordered = ['h264_8bit']
        return ordered

    def _build_final_vf_chain(self, width: int, height: int, out_fps: float) -> Tuple[str, float]:
        """Construct the final video filter chain with optional prefilters."""
        filters: List[str] = [f"scale={width}:{height}:flags=lanczos", "setsar=1"]
        if bool(self.config.get('video_compression.prefilters.denoise', True)):
            filters.append("hqdn3d=1.5:1.5:6:6")
        if bool(self.config.get('video_compression.prefilters.deband', True)):
            filters.append("gradfun")
        return ",".join(filters), float(out_fps)

    def _codec_to_encoder(self, codec_key: str) -> Tuple[str, Optional[str]]:
        """Map codec key to encoder name and pixel format.
        Returns (encoder, pix_fmt) where pix_fmt may be None for 8-bit.
        """
        if codec_key.startswith('hevc'):
            encoder = 'libx265'
            pix_fmt = 'yuv420p10le' if codec_key.endswith('10bit') else 'yuv420p'
        else:
            encoder = 'libx264'
            pix_fmt = 'yuv420p10le' if codec_key.endswith('10bit') else 'yuv420p'
        return encoder, pix_fmt

    def _run_two_pass_encode_once(self,
                                  codec_key: str,
                                  input_path: str,
                                  output_path: str,
                                  v_kbps: int,
                                  a_kbps: int,
                                  width: int,
                                  height: int,
                                  out_fps: float) -> Tuple[bool, float, str]:
        """Run a two-pass encode for the given codec key.
        Returns (success, size_mb, log_base) where log_base is the passlogfile base path.
        """
        vf, out_fps = self._build_final_vf_chain(width, height, out_fps)
        encoder, pix_fmt = self._codec_to_encoder(codec_key)

        # Software-only for final pass for better quality/size
        params = {
            'encoder': encoder,
            'vf': vf,
            'fps': out_fps,
            'preset': self.config.get('video_compression.codec.final_preset', 'slow'),
            'tune': self.config.get('video_compression.codec.tune', 'animation'),
            'bitrate': int(v_kbps),
            'audio_bitrate': int(a_kbps),
        }
        if pix_fmt:
            params['pix_fmt'] = pix_fmt

        # Use unique passlogfile for final refinement to avoid conflicts
        final_id = f"final_refinement_{int(time.time())}"
        log_base = os.path.join(self.temp_dir, f'ffmpeg2pass_{final_id}')
        # Pass 1
        cmd1 = FFmpegUtils.build_two_pass_with_filters(input_path, output_path, params, pass_num=1, log_file=log_base)
        ok1 = FFmpegUtils.execute_ffmpeg_with_progress(cmd1, description="Analyzing", duration=None)
        if not ok1:
            return False, 0.0, log_base
        # Pass 2
        cmd2 = FFmpegUtils.build_two_pass_with_filters(input_path, output_path, params, pass_num=2, log_file=log_base)
        ok2 = FFmpegUtils.execute_ffmpeg_with_progress(cmd2, description="Encoding", duration=None)
        if not ok2 or not os.path.exists(output_path):
            return False, 0.0, log_base
        size_mb = os.path.getsize(output_path) / (1024 * 1024)
        # Track encoder used
        try:
            self._last_encoder_used = f"{encoder} ({'10-bit' if pix_fmt == 'yuv420p10le' else '8-bit'})"
        except Exception:
            pass
        return True, size_mb, log_base

    def _rerun_pass2_with_bitrate(self,
                                  input_path: str,
                                  output_path: str,
                                  params: Dict[str, Any],
                                  log_base: str,
                                  new_v_kbps: int,
                                  a_kbps: int) -> bool:
        try:
            params2 = dict(params)
            params2['bitrate'] = int(new_v_kbps)
            params2['audio_bitrate'] = int(a_kbps)
            cmd2 = FFmpegUtils.build_two_pass_with_filters(input_path, output_path, params2, pass_num=2, log_file=log_base)
            return FFmpegUtils.execute_ffmpeg_with_progress(cmd2, description="Refining", duration=None)
        except Exception:
            return False

    def _final_single_file_refinement(self, input_path: str, output_path: str, target_size_mb: float,
                                      video_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Run a refined, size-targeted software two-pass with codec priority and utilization tuning.
        Returns result dict if applied and better, else None.
        """
        try:
            # Config reads
            min_util = float(self.config.get('video_compression.quality_controls.min_target_utilization', 0.92))
            max_iters = int(self.config.get('video_compression.quality_controls.refine_iterations', 2) or 0)
            prefer_24fps = bool(self.config.get('video_compression.fps_policy.prefer_24fps_for_low_motion', True))
            low_motion_threshold = float(self.config.get('video_compression.fps_policy.low_motion_threshold', 0.35))

            # Determine fps target
            out_fps = min(video_info.get('fps', 30.0), 30.0)
            if prefer_24fps and (video_info.get('motion_level') == 'low' or out_fps < 28.0):
                out_fps = 24.0

            # Use 98% of target as safety margin to account for encoding variance
            safe_target_mb = target_size_mb * 0.98
            
            # Compute conservative output resolution for target size
            width, height = self._calculate_optimal_resolution(video_info, safe_target_mb, {})

            # Calculate initial bitrates
            a_kbps = 96
            v_kbps = self._calculate_target_video_kbps(safe_target_mb, video_info['duration'], a_kbps)
            # Safety check: prevent extremely high bitrates that might cause FFmpeg to crash
            # Use dynamic cap based on video characteristics
            safe_cap = self._calculate_safe_bitrate_cap(width, height, out_fps, video_info['duration'])
            v_kbps = min(v_kbps, safe_cap)

            # Select codec order
            codec_order = self._derive_codec_priority()

            # Min BPP floors
            min_bpp_map = {
                'h264_8bit': float(self.config.get('video_compression.codec.min_bpp.h264_8bit', 0.023)),
                'h264_10bit': float(self.config.get('video_compression.codec.min_bpp.h264_10bit', 0.020)),
                'hevc_8bit': float(self.config.get('video_compression.codec.min_bpp.hevc_8bit', 0.018)),
                'hevc_10bit': float(self.config.get('video_compression.codec.min_bpp.hevc_10bit', 0.015)),
            }

            best_local_file = None
            best_local_size_mb = 0.0
            best_local_codec = None

            for codec_key in codec_order:
                # Ensure BPP floor by adjusting fps or bitrate if under
                needed_kbps = v_kbps
                min_bpp = min_bpp_map.get(codec_key, 0.018)
                if not self._ensure_min_bpp(width, height, out_fps, v_kbps, min_bpp):
                    if prefer_24fps and out_fps > 24.0:
                        out_fps = 24.0
                    # Recompute needed kbps to meet min_bpp given selected fps/res
                    pixels_per_sec = max(1, width * height) * max(1.0, out_fps)
                    needed_kbps = int(max(v_kbps, math.ceil(min_bpp * pixels_per_sec / 1000.0)))
                    # Cap by theoretical target video kbps
                    needed_kbps = min(needed_kbps, self._calculate_target_video_kbps(target_size_mb, video_info['duration'], a_kbps))

                temp_out = os.path.join(self.temp_dir, 'refined_final_temp.mp4')
                try:
                    if os.path.exists(temp_out):
                        os.remove(temp_out)
                except Exception:
                    pass

                ok, size_mb, log_base = self._run_two_pass_encode_once(
                    codec_key, input_path, temp_out, needed_kbps, a_kbps, width, height, out_fps
                )
                if not ok:
                    logger.info(f"Final-pass encode failed for {codec_key}, trying next fallback")
                    continue

                util = self._utilization_ratio(int(size_mb * 1024 * 1024), target_size_mb)
                logger.info(f"Final-pass initial result {codec_key}: {size_mb:.2f}MB, utilization={util*100:.1f}%")

                # Refinement iterations to increase size utilization if too low
                params_snapshot = {
                    'encoder': self._codec_to_encoder(codec_key)[0],
                    'vf': self._build_final_vf_chain(width, height, out_fps)[0],
                    'fps': out_fps,
                    'preset': self.config.get('video_compression.codec.final_preset', 'slow'),
                    'tune': self.config.get('video_compression.codec.tune', 'animation'),
                    'pix_fmt': self._codec_to_encoder(codec_key)[1],
                }

                iter_count = 0
                cur_v_kbps = needed_kbps
                while util < min_util and iter_count < max_iters:
                    cur_v_kbps = int(cur_v_kbps * 1.2)
                    ok2 = self._rerun_pass2_with_bitrate(input_path, temp_out, params_snapshot, log_base, cur_v_kbps, a_kbps)
                    if not ok2 or not os.path.exists(temp_out):
                        logger.info("Refinement pass failed, stopping refinement loop")
                        break
                    size_mb = os.path.getsize(temp_out) / (1024 * 1024)
                    util = self._utilization_ratio(int(size_mb * 1024 * 1024), target_size_mb)
                    logger.info(f"Refinement iteration {iter_count+1}: size={size_mb:.2f}MB, utilization={util*100:.1f}%")
                    iter_count += 1

                # Optional quality sampling and enforcement with limited uplift
                quality_enforce = bool(self.config.get('video_compression.quality_controls.hybrid_quality_check.enabled', True))
                single_file_floor = float(self.config.get('video_compression.quality_controls.hybrid_quality_check.single_file_floor', 65))
                uplift_enabled = bool(self.config.get('video_compression.quality_controls.uplift.enabled', True))
                uplift_max = int(self.config.get('video_compression.quality_controls.uplift.max_passes', 2) or 0)
                uplift_step = float(self.config.get('video_compression.quality_controls.uplift.bitrate_step', 1.08))
                uplift_cap = float(self.config.get('video_compression.quality_controls.uplift.max_multiplier', 1.2))
                
                # Calculate quality score for this codec result
                if quality_enforce:
                    quality_result = self.quality_scorer.calculate_quality_score(
                        input_path, temp_out, target_size_mb, is_segment=False
                    )
                    quality_score = quality_result['overall_score']
                    logger.info(f"Quality sample for {codec_key}: {quality_score:.1f}/100 (floor: {single_file_floor:.1f})")
                    
                    # If enforcing and below floor while under size, try bounded bitrate uplift
                    if quality_score < single_file_floor and size_mb <= target_size_mb and uplift_enabled:
                        uplift_count = 0
                        base_kbps = cur_v_kbps
                        while uplift_count < uplift_max and quality_score < single_file_floor:
                            # Increase bitrate conservatively within cap
                            next_kbps = int(min(base_kbps * (uplift_step ** (uplift_count + 1)), needed_kbps * uplift_cap))
                            if next_kbps <= cur_v_kbps:
                                break
                            ok2 = self._rerun_pass2_with_bitrate(input_path, temp_out, params_snapshot, log_base, next_kbps, a_kbps)
                            if not ok2 or not os.path.exists(temp_out):
                                break
                            cur_v_kbps = next_kbps
                            size_mb = os.path.getsize(temp_out) / (1024 * 1024)
                            if size_mb > target_size_mb:
                                logger.info(f"Uplift pass {uplift_count+1} exceeded target size ({size_mb:.2f}MB), stopping uplift")
                                break
                            quality_result_new = self.quality_scorer.calculate_quality_score(
                                input_path, temp_out, target_size_mb, is_segment=False
                            )
                            if quality_result_new is None:
                                break
                            quality_score = quality_result_new['overall_score']
                            logger.info(f"Uplift pass {uplift_count+1}: bitrate={cur_v_kbps} kbps, size={size_mb:.2f}MB, quality={quality_score:.1f}/100")
                            uplift_count += 1
                        # If still below floor after uplifts, mark as not acceptable
                        if quality_score < single_file_floor:
                            logger.info(f"Quality remains below floor ({quality_score:.1f} < {single_file_floor:.1f}) after {uplift_count} uplifts; will prefer other strategies")

                # Track best so far (prefer closer to target; quality considered implicitly via enforcement)
                if size_mb > best_local_size_mb:
                    best_local_file = temp_out
                    best_local_size_mb = size_mb
                    best_local_codec = codec_key
                else:
                    try:
                        os.remove(temp_out)
                    except Exception:
                        pass

                # Good enough if utilization is close to 100%
                if util >= min(0.98, max(min_util, 0.92)):
                    break

            if best_local_file and os.path.exists(best_local_file):
                # Validate that refined result doesn't exceed target
                if best_local_size_mb > target_size_mb:
                    logger.warning(f"Final refinement exceeded target ({best_local_size_mb:.2f}MB > {target_size_mb}MB), keeping previous result")
                    if os.path.exists(best_local_file):
                        os.remove(best_local_file)
                    return None  # Keep existing output_path as-is
                
                # Replace output with validated refinement
                try:
                    if os.path.exists(output_path):
                        os.remove(output_path)
                except Exception:
                    pass
                shutil.move(best_local_file, output_path)
                logger.info(f"Refined final-pass applied using {best_local_codec}: {best_local_size_mb:.2f}MB")
                return self._get_compression_results(input_path, output_path, video_info, "refined_final")
        except Exception as e:
            logger.warning(f"Final single-file refinement failed: {e}")
        return None
    
    # ===== Content-Adaptive Encoding (CAE) for Discord 10MB =====
    
    def _compress_with_cae_discord_10mb(self, input_path: str, output_path: str,
                                       video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Content-Adaptive Encoding pipeline optimized for Discord 10MB limit.
        
        Uses 2-pass x264, BPP-driven resolution/FPS selection, VMAF/SSIM quality gates,
        artifact detection, and adaptive refine loop.
        """
        logger.info("=== Starting Discord 10MB CAE Pipeline ===")
        
        # Load profile config
        profile_cfg = self.config.get('video_compression.profiles.discord_10mb', {})
        size_limit_mb = profile_cfg.get('size_limit_mb', 10.0)
        
        # Step 1: Analyze content complexity
        motion_level = video_info.get('motion_level', 'medium')  # from _analyze_video_content
        logger.info(f"Content analysis: motion={motion_level}, complexity={video_info.get('complexity_score', 5.0):.1f}")
        
        # Step 2: Budget calculation (reserve audio)
        audio_range = profile_cfg.get('audio_kbps_range', [64, 96])
        audio_bitrate_kbps = audio_range[1] if motion_level == 'low' else audio_range[0]  # More audio for low-motion
        duration = video_info['duration']
        audio_bits = audio_bitrate_kbps * 1000 * duration
        total_bits_budget = size_limit_mb * 8 * 1024 * 1024
        video_bits_budget = int(total_bits_budget - audio_bits)
        target_video_bitrate_kbps = int(video_bits_budget / duration / 1000)
        
        logger.info(f"Budget: {size_limit_mb}MB = {target_video_bitrate_kbps}k video + {audio_bitrate_kbps}k audio over {duration:.1f}s")
        
        # Step 3: BPP-driven resolution/FPS selection
        bpp_floors = profile_cfg.get('bpp_floor', {})
        bpp_min = bpp_floors.get(motion_level, bpp_floors.get('normal', 0.035))
        
        initial_params = self._select_resolution_fps_by_bpp(
            video_info, target_video_bitrate_kbps, bpp_min, profile_cfg
        )
        initial_params['bitrate'] = target_video_bitrate_kbps
        initial_params['audio_bitrate'] = audio_bitrate_kbps
        
        logger.info(f"Initial target: {initial_params['width']}x{initial_params['height']}@{initial_params['fps']}fps, "
                   f"BPP={bpp_min:.3f}, bitrate={target_video_bitrate_kbps}k")
        
        # Step 4: 2-pass encode with refine loop
        x264_cfg = profile_cfg.get('x264', {})
        refine_cfg = profile_cfg.get('refine', {})
        max_passes = refine_cfg.get('max_passes', 3)
        bitrate_step = refine_cfg.get('bitrate_step', 1.08)
        
        vmaf_threshold = profile_cfg.get('vmaf_threshold', 80.0)
        vmaf_threshold_low_res = profile_cfg.get('vmaf_threshold_low_res', 78.0)
        ssim_threshold = profile_cfg.get('ssim_threshold', 0.94)
        
        for refine_pass in range(max_passes):
            logger.info(f"--- CAE Refine Pass {refine_pass + 1}/{max_passes} ---")
            
            # Encode with current params
            encode_params = self._build_cae_encode_params(initial_params, x264_cfg, profile_cfg)
            success = self._execute_two_pass_x264(input_path, output_path, encode_params)
            
            if not success or not os.path.exists(output_path):
                logger.error(f"Encode failed on pass {refine_pass + 1}")
                continue
            
            # Check size
            output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Encoded size: {output_size_mb:.2f}MB (target: {size_limit_mb}MB)")
            
            if output_size_mb > size_limit_mb:
                logger.warning(f"Output exceeds size limit: {output_size_mb:.2f} > {size_limit_mb}MB")
                # Try reducing bitrate slightly
                initial_params['bitrate'] = int(initial_params['bitrate'] * 0.92)
                try:
                    os.remove(output_path)
                except:
                    pass
                continue
            
            # Step 5: Quality gates
            eval_height = min(initial_params['height'], 540)  # Eval at 540p max
            vmaf_thresh = vmaf_threshold_low_res if initial_params['height'] <= 480 else vmaf_threshold
            
            quality_result = self.quality_gates.evaluate_quality(
                input_path, output_path, vmaf_thresh, ssim_threshold, eval_height
            )
            
            # Step 6: Artifact detection
            artifact_result = self.artifact_detector.detect_artifacts(
                output_path, blockiness_threshold=0.12, banding_threshold=0.10
            )
            
            logger.info(f"Quality: VMAF={quality_result.get('vmaf_score', 'N/A')}, "
                       f"SSIM={quality_result.get('ssim_score', 'N/A')}, passes={quality_result.get('passes')}")
            logger.info(f"Artifacts: blockiness={artifact_result.get('blockiness_score', 'N/A'):.4f}, "
                       f"banding={artifact_result.get('banding_score', 'N/A'):.4f}, passes={artifact_result.get('passes')}")
            
            # Check if all gates pass
            quality_pass = quality_result.get('passes', False)
            artifact_pass = artifact_result.get('passes', True)  # Default pass if detection fails
            
            if quality_pass and artifact_pass:
                logger.info(f"✓ All quality gates passed on refine pass {refine_pass + 1}")
                
                # Log structured metrics
                self._log_cae_metrics({
                    'input_path': input_path,
                    'output_path': output_path,
                    'params': encode_params,
                    'size_mb': output_size_mb,
                    'quality': quality_result,
                    'artifacts': artifact_result,
                    'refine_pass': refine_pass + 1
                })
                
                return self._get_compression_results(input_path, output_path, video_info, "cae_discord_10mb")
            
            # Step 7: Refine strategy if gates failed
            logger.warning(f"Quality gates failed on pass {refine_pass + 1}, attempting refinement")
            
            if refine_pass + 1 < max_passes:
                # Try refinements in order: +bitrate, drop FPS, lower resolution
                if not quality_pass and output_size_mb < size_limit_mb * 0.95:
                    # Room to increase bitrate
                    new_bitrate = int(initial_params['bitrate'] * bitrate_step)
                    headroom_bitrate = int((size_limit_mb * 0.98 - output_size_mb) * 8 * 1024 / duration)
                    initial_params['bitrate'] = min(new_bitrate, initial_params['bitrate'] + headroom_bitrate)
                    logger.info(f"Refine: increasing bitrate to {initial_params['bitrate']}k")
                    
                elif initial_params['fps'] > 24:
                    # Drop FPS to 24
                    initial_params['fps'] = 24
                    logger.info(f"Refine: dropping FPS to 24")
                    
                else:
                    # Scale down resolution
                    new_width, new_height = self._scale_down_one_step(initial_params['width'], initial_params['height'])
                    if new_height >= 360:
                        initial_params['width'] = new_width
                        initial_params['height'] = new_height
                        logger.info(f"Refine: scaling down to {new_width}x{new_height}")
                    else:
                        logger.warning("Cannot refine further (already at minimum resolution)")
                        break
                
                # Clean up failed attempt
                try:
                    os.remove(output_path)
                except:
                    pass
        
        # If we exhausted refine passes, return best attempt or fail to segmentation
        if os.path.exists(output_path):
            output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            if output_size_mb <= size_limit_mb:
                logger.warning("CAE completed with compromised quality (some gates failed)")
                return self._get_compression_results(input_path, output_path, video_info, "cae_discord_10mb_compromised")
        
        logger.error("CAE pipeline failed after all refine passes")
        return {'success': False, 'error': 'CAE quality gates failed', 'method': 'cae_failed'}
    
    def _select_resolution_fps_by_bpp(self, video_info: Dict[str, Any], 
                                     target_bitrate_kbps: int, bpp_min: float,
                                     profile_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Select highest resolution and FPS that satisfy BPP floor constraint."""
        orig_w = video_info['width']
        orig_h = video_info['height']
        orig_fps = video_info['fps']
        
        # Candidate resolutions (maintain aspect ratio)
        aspect_ratio = orig_w / orig_h
        candidate_heights = [1080, 720, 540, 480, 360]
        candidates = []
        
        for h in candidate_heights:
            if h > orig_h:
                continue
            w = int(h * aspect_ratio)
            # Ensure even dimensions
            w = w if w % 2 == 0 else w - 1
            candidates.append((w, h))
        
        # Add original resolution if not in list
        if orig_h not in candidate_heights:
            candidates.insert(0, (orig_w, orig_h))
        
        # Candidate FPS values
        fps_candidates = [30, 24] if orig_fps >= 30 else [24, int(orig_fps)]
        fps_prefer_24 = profile_cfg.get('fps_policy', {}).get('prefer_24_for_low_motion', True)
        motion_level = video_info.get('motion_level', 'medium')
        
        if fps_prefer_24 and motion_level == 'low':
            fps_candidates = [24, 30]  # Prefer 24 for low motion
        
        # Find best combo that satisfies BPP
        target_bits_per_sec = target_bitrate_kbps * 1000
        
        for w, h in candidates:
            for fps in fps_candidates:
                pixels_per_sec = w * h * fps
                actual_bpp = target_bits_per_sec / pixels_per_sec
                
                if actual_bpp >= bpp_min:
                    logger.info(f"Selected {w}x{h}@{fps}fps: BPP={actual_bpp:.4f} >= {bpp_min:.4f}")
                    return {'width': w, 'height': h, 'fps': fps}
        
        # Fallback: use smallest resolution at lowest FPS
        w, h = candidates[-1]
        fps = fps_candidates[-1]
        logger.warning(f"BPP floor not met; using minimum {w}x{h}@{fps}fps")
        return {'width': w, 'height': h, 'fps': fps}
    
    def _build_cae_encode_params(self, base_params: Dict[str, Any], 
                                 x264_cfg: Dict[str, Any],
                                 profile_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Build full encoding parameters from base + x264 config."""
        params = base_params.copy()
        
        # Apply x264 tuning from config
        params['preset'] = x264_cfg.get('preset', 'slow')
        params['tune'] = x264_cfg.get('tune', 'film')
        params['gop'] = x264_cfg.get('gop', 240)
        params['keyint_min'] = x264_cfg.get('keyint_min', 23)
        params['sc_threshold'] = x264_cfg.get('sc_threshold', 40)
        params['qcomp'] = x264_cfg.get('qcomp', 0.65)
        params['aq_mode'] = x264_cfg.get('aq_mode', 2)
        params['aq_strength'] = x264_cfg.get('aq_strength', 1.1)
        params['rc_lookahead'] = x264_cfg.get('rc_lookahead', 40)
        params['maxrate_multiplier'] = x264_cfg.get('maxrate_multiplier', 1.10)
        params['bufsize_multiplier'] = x264_cfg.get('bufsize_multiplier', 2.0)
        
        return params
    
    def _execute_two_pass_x264(self, input_path: str, output_path: str, 
                               params: Dict[str, Any]) -> bool:
        """Execute 2-pass x264 encode with given parameters."""
        from .ffmpeg_utils import FFmpegUtils
        
        log_base = os.path.join(self.temp_dir, f"cae_pass_{int(time.time())}")
        
        try:
            # Pass 1
            logger.info("Running 2-pass encode: pass 1/2")
            cmd_pass1 = FFmpegUtils.build_x264_two_pass_cae(
                input_path, output_path, params, pass_num=1, log_file=log_base
            )
            result1 = subprocess.run(
                cmd_pass1,
                capture_output=True,
                text=True,
                timeout=600,
                encoding='utf-8',
                errors='replace'
            )
            
            if result1.returncode != 0:
                logger.error(f"Pass 1 failed: {result1.stderr}")
                return False
            
            # Pass 2
            logger.info("Running 2-pass encode: pass 2/2")
            cmd_pass2 = FFmpegUtils.build_x264_two_pass_cae(
                input_path, output_path, params, pass_num=2, log_file=log_base
            )
            result2 = subprocess.run(
                cmd_pass2,
                capture_output=True,
                text=True,
                timeout=600,
                encoding='utf-8',
                errors='replace'
            )
            
            if result2.returncode != 0:
                logger.error(f"Pass 2 failed: {result2.stderr}")
                return False
            
            logger.info("2-pass encode completed successfully")
            return True
            
        except subprocess.TimeoutExpired:
            logger.error("2-pass encode timed out")
            return False
        except Exception as e:
            logger.error(f"2-pass encode error: {e}")
            return False
        finally:
            # Clean up pass logs
            self._cleanup_two_pass_logs(log_base)
    
    def _scale_down_one_step(self, width: int, height: int) -> Tuple[int, int]:
        """Scale resolution down one step (e.g., 720p -> 540p)."""
        common_heights = [1080, 720, 540, 480, 360, 270]
        
        for i, h in enumerate(common_heights):
            if height >= h and i + 1 < len(common_heights):
                new_h = common_heights[i + 1]
                aspect_ratio = width / height
                new_w = int(new_h * aspect_ratio)
                new_w = new_w if new_w % 2 == 0 else new_w - 1
                return (new_w, new_h)
        
        # Fallback: scale by 0.75
        new_h = int(height * 0.75)
        new_h = new_h if new_h % 2 == 0 else new_h - 1
        new_w = int(width * 0.75)
        new_w = new_w if new_w % 2 == 0 else new_w - 1
        return (new_w, new_h)
    
    def _log_cae_metrics(self, metrics: Dict[str, Any]):
        """Log structured JSON metrics for CAE encoding."""
        try:
            import json
            metrics_json = {
                'timestamp': time.time(),
                'input': os.path.basename(metrics['input_path']),
                'output': os.path.basename(metrics['output_path']),
                'size_mb': metrics['size_mb'],
                'params': {
                    'width': metrics['params']['width'],
                    'height': metrics['params']['height'],
                    'fps': metrics['params']['fps'],
                    'bitrate_kbps': metrics['params']['bitrate'],
                    'audio_bitrate_kbps': metrics['params']['audio_bitrate']
                },
                'quality': {
                    'vmaf': metrics['quality'].get('vmaf_score'),
                    'ssim': metrics['quality'].get('ssim_score'),
                    'method': metrics['quality'].get('method')
                },
                'artifacts': {
                    'blockiness': metrics['artifacts'].get('blockiness_score'),
                    'banding': metrics['artifacts'].get('banding_score')
                },
                'refine_pass': metrics['refine_pass']
            }
            logger.info(f"CAE_METRICS: {json.dumps(metrics_json)}")
        except Exception as e:
            logger.warning(f"Failed to log CAE metrics: {e}")
        
    def compress_video(self, input_path: str, output_path: str, platform: str = None,
                      max_size_mb: int = None, use_advanced_optimization: bool = True,
                      force_single_file: bool = False) -> Dict[str, Any]:
        """
        Compress video with dynamic optimization for maximum quality within size constraints
        Enhanced with advanced optimization techniques and performance improvements
        """
        
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video file not found: {input_path}")
        
        logger.info(f"Starting enhanced video compression: {input_path} -> {output_path}")
        
        # Get platform configuration
        platform_config = {}
        if platform:
            platform_config = self.config.get_platform_config(platform, 'video_compression')
            logger.info(f"Using platform configuration for: {platform}")
        
        # Determine target file size
        target_size_mb = max_size_mb or platform_config.get('max_file_size_mb') or self.config.get('video_compression.max_file_size_mb', 10)
        
        # Check cache for similar operations
        @self.performance_enhancer.cached_operation(ttl_hours=6)
        def cached_analysis(file_path, file_size, target_size):
            return self._analyze_video_content(file_path)
        
        # Early-cancel before analysis if shutdown requested
        if self.shutdown_requested:
            return {'success': False, 'cancelled': True}

        # Get comprehensive video analysis (cached), but handle graceful cancellation
        file_size = os.path.getsize(input_path)
        try:
            video_info = cached_analysis(input_path, file_size, target_size_mb)
        except GracefulCancellation:
            return {'success': False, 'cancelled': True}
        original_size_mb = file_size / (1024 * 1024)
        
        logger.info(f"Original video: {video_info['width']}x{video_info['height']}, "
                   f"{video_info['duration']:.2f}s, {original_size_mb:.2f}MB, "
                   f"complexity: {video_info['complexity_score']:.2f}")
        
        # Route to Discord 10MB CAE pipeline if target is exactly 10MB (or close)
        # and no specific platform is set or platform is discord
        use_cae = False
        if 9.5 <= target_size_mb <= 10.5:
            if platform is None or platform == 'discord':
                use_cae = True
                logger.info("Routing to Discord 10MB CAE pipeline")
        
        if use_cae:
            try:
                cae_result = self._compress_with_cae_discord_10mb(input_path, output_path, video_info)
                if cae_result.get('success'):
                    return cae_result
                logger.warning("CAE pipeline failed, falling back to standard compression")
            except Exception as e:
                logger.error(f"CAE pipeline error: {e}, falling back to standard compression")
        
        # Check if segmentation should be considered now or deferred to last resort
        logger.info(f"Checking video segmentation: original size {original_size_mb:.1f}MB, target {target_size_mb}MB")
        logger.info(f"Video info keys: {list(video_info.keys())}")
        logger.info(f"Video info size_bytes: {video_info.get('size_bytes', 'NOT FOUND')}")

        # If forced single-file, skip segmentation entirely
        if force_single_file:
            logger.info("Segmentation disabled via --no-segmentation flag, forcing single-file processing")
            return self._compress_with_aggressive_single_file(
                input_path, output_path, target_size_mb, platform_config, video_info
            )

        seg_last_resort = bool(self.config.get('video_compression.segmentation.only_if_single_file_unacceptable', False))
        if not seg_last_resort:
            if self.video_segmenter.should_segment_video(video_info['duration'], video_info, target_size_mb):
                logger.info("Video will be segmented instead of compressed as single file")
                return self._compress_with_segmentation(
                    input_path, output_path, target_size_mb, platform_config, video_info, platform
                )
        
        # If already under target size and no platform specified, just copy
        if original_size_mb <= target_size_mb and not platform:
            shutil.copy2(input_path, output_path)
            logger.info("Video already meets size requirements, copied without compression")
            return self._get_compression_results(input_path, output_path, video_info, "copy")
        
        # Choose optimization strategy based on file size and system resources
        if use_advanced_optimization and (original_size_mb > 50 or target_size_mb < 5):
            logger.info("Using advanced optimization for challenging compression requirements")
            try:
                adv_result = self._compress_with_advanced_optimization(
                    input_path, output_path, target_size_mb, platform_config, video_info
                )
            except Exception:
                adv_result = None
            if adv_result and adv_result.get('success'):
                return adv_result
            # Fallback to standard pipeline or segmentation as last resort
            try:
                std_result = self._compress_with_standard_optimization(
                    input_path, output_path, target_size_mb, platform_config, video_info
                )
                # Attempt refinement
                try:
                    refined_result = self._final_single_file_refinement(input_path, output_path, target_size_mb, video_info)
                    if refined_result and refined_result.get('success'):
                        return refined_result
                except Exception:
                    pass
                return std_result
            except Exception:
                if seg_last_resort:
                    logger.info("Standard/advanced pipelines failed; falling back to segmentation as last resort")
                    return self._compress_with_segmentation(
                        input_path, output_path, target_size_mb, platform_config, video_info, platform
                    )
                raise
        
        
        # Use adaptive quality processing for medium complexity files
        elif original_size_mb > 20:
            logger.info("Using adaptive quality processing for medium complexity file")
            return self._compress_with_adaptive_quality(
                input_path, output_path, target_size_mb, platform_config, video_info
            )

        # Use standard dynamic optimization for simpler cases
        else:
            try:
                std_result = self._compress_with_standard_optimization(
                    input_path, output_path, target_size_mb, platform_config, video_info
                )
                # Attempt refinement
                try:
                    refined_result = self._final_single_file_refinement(input_path, output_path, target_size_mb, video_info)
                    if refined_result and refined_result.get('success'):
                        return refined_result
                except Exception:
                    pass
                return std_result
            except Exception:
                if seg_last_resort:
                    logger.info("Standard pipeline failed; falling back to segmentation as last resort")
                    return self._compress_with_segmentation(
                        input_path, output_path, target_size_mb, platform_config, video_info, platform
                    )
                raise
    
    def _compress_with_advanced_optimization(self, input_path: str, output_path: str,
                                           target_size_mb: float, platform_config: Dict[str, Any],
                                           video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Use advanced optimization techniques for challenging compression"""
        
        try:
            # Early bailout: Skip advanced optimization if source/target ratio is too high
            # Advanced optimization is expensive and unlikely to help with extreme compression ratios
            source_size_mb = os.path.getsize(input_path) / (1024 * 1024)
            size_ratio = source_size_mb / target_size_mb
            
            if size_ratio > 6.0:
                logger.info(f"Skipping advanced optimization: size ratio too high ({size_ratio:.1f}x > 6x)")
                raise RuntimeError(f"Size ratio {size_ratio:.1f}x exceeds threshold for advanced optimization")
            
            # Use the advanced optimizer
            result = self.advanced_optimizer.optimize_with_advanced_techniques(
                input_path, output_path, target_size_mb, platform_config
            )
            
            # Convert result format to match expected output
            compressed_size_mb = result['size_mb']
            return {
                'success': True,  # Add success flag
                'input_file': input_path,
                'output_file': output_path,
                'method': 'advanced_optimized',
                'original_size_mb': os.path.getsize(input_path) / (1024 * 1024),
                'compressed_size_mb': compressed_size_mb,
                'size_mb': compressed_size_mb,  # Add alias for automated workflow
                'compression_ratio': ((os.path.getsize(input_path) - os.path.getsize(output_path)) / os.path.getsize(input_path)) * 100,
                'space_saved_mb': (os.path.getsize(input_path) - os.path.getsize(output_path)) / (1024 * 1024),
                'video_info': video_info,
                'optimization_strategy': result.get('candidate_name', 'advanced'),
                'quality_score': result.get('quality_score', 8.0),
                'attempts_made': 1,
                'encoder_used': result.get('candidate_data', {}).get('encoder', 'unknown')
            }
            
        except Exception as e:
            logger.warning(f"Advanced optimization failed, falling back to standard: {e}")
            return self._compress_with_standard_optimization(
                input_path, output_path, target_size_mb, platform_config, video_info
            )
    
    def _compress_with_adaptive_quality(self, input_path: str, output_path: str,
                                      target_size_mb: float, platform_config: Dict[str, Any],
                                      video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Use adaptive quality processing for optimal results"""
        
        try:
            # Use performance enhancer's adaptive processing
            quality_levels = ['fast', 'balanced', 'high_quality']
            
            # Create a processing function that uses our compression logic
            def process_quality_level(task):
                quality_level = task['quality_level']
                temp_output = task['temp_output']
                
                # Map quality levels to compression parameters
                quality_params = {
                    'fast': {'crf': 28, 'preset': 'veryfast', 'priority': 'speed'},
                    'balanced': {'crf': 23, 'preset': 'medium', 'priority': 'balanced'},
                    'high_quality': {'crf': 18, 'preset': 'slow', 'priority': 'quality'}
                }
                
                params = quality_params[quality_level]
                
                # Use our compression logic with these parameters
                compression_params = self._calculate_compression_params_with_quality(
                    video_info, platform_config, target_size_mb, params
                )
                
                # Build and execute FFmpeg command
                ffmpeg_cmd = self._build_ffmpeg_command(input_path, temp_output, compression_params, 1)
                self._execute_ffmpeg_with_progress(ffmpeg_cmd, video_info['duration'])
                
                # Calculate result quality
                if os.path.exists(temp_output):
                    file_size_mb = os.path.getsize(temp_output) / (1024 * 1024)
                    quality_score = self._calculate_quality_score(compression_params, video_info)
                    
                    return {
                        'quality_level': quality_level,
                        'output_path': temp_output,
                        'size_mb': file_size_mb,
                        'quality_score': quality_score,
                        'settings': params,
                        'processing_time': 1.0,  # Placeholder
                        'success': file_size_mb <= target_size_mb
                    }
                
                return None
            
            # Use adaptive quality processing
            best_result = self.performance_enhancer.adaptive_quality_processing(
                input_path, target_size_mb, quality_levels
            )
            
            # Move result to final output
            if best_result and os.path.exists(best_result.get('output_path', '')):
                shutil.move(best_result['output_path'], output_path)
                
                return {
                    'input_file': input_path,
                    'output_file': output_path,
                    'method': 'adaptive_quality',
                    'original_size_mb': os.path.getsize(input_path) / (1024 * 1024),
                    'compressed_size_mb': best_result.get('size_mb', 0),
                    'compression_ratio': ((os.path.getsize(input_path) - os.path.getsize(output_path)) / os.path.getsize(input_path)) * 100,
                    'space_saved_mb': (os.path.getsize(input_path) - os.path.getsize(output_path)) / (1024 * 1024),
                    'video_info': video_info,
                    'optimization_strategy': f"adaptive_{best_result.get('quality_level', 'balanced')}",
                    'quality_score': best_result.get('quality_score', 7.0),
                    'attempts_made': len(quality_levels),
                    'encoder_used': self.hardware.get_best_encoder("h264")[0]
                }
            
        except Exception as e:
            logger.warning(f"Adaptive quality processing failed, falling back to standard: {e}")
        
        # Fallback to standard optimization
        return self._compress_with_standard_optimization(
            input_path, output_path, target_size_mb, platform_config, video_info
        )
    
    def _refine_strategy_result(self, result: Dict[str, Any], input_path: str, 
                                video_info: Dict[str, Any], target_size_mb: float,
                                max_iterations: int = 2) -> Dict[str, Any]:
        """
        Refine a strategy result to better utilize the target size budget.
        Only refines if result is between 60-95% of target utilization.
        """
        if not result or result['size_mb'] > target_size_mb:
            return result
        
        utilization = result['size_mb'] / target_size_mb
        
        # Only refine if under-utilizing (60-93% range) - more conservative to avoid overshooting
        if utilization < 0.60 or utilization >= 0.93:
            return result
        
        logger.info(f"Result under-utilizing target ({utilization*100:.1f}%), attempting refinement...")
        
        try:
            refined_result = result
            iteration = 0
            
            # Target 93% max to leave safety margin, stop if we get close
            while utilization < 0.93 and iteration < max_iterations:
                # More conservative multiplier to avoid overshooting (0.90 instead of 0.95)
                bitrate_multiplier = min(target_size_mb / result['size_mb'] * 0.90, 1.2)
                
                # Get current params and adjust bitrate
                params = result.get('params', {}).copy()
                if 'bitrate' in params:
                    params['bitrate'] = int(params['bitrate'] * bitrate_multiplier)
                    
                    # Re-encode with higher bitrate
                    temp_refined = os.path.join(self.temp_dir, f"refined_{iteration}_{result['strategy']}.mp4")
                    
                    if result['strategy'] == 'two_pass':
                        # Use unique passlogfile for each refinement attempt
                        refine_id = f"refine_{iteration}_{result['strategy']}_{int(time.time())}"
                        refine_log_file = os.path.join(self.temp_dir, f"ffmpeg2pass_{refine_id}")
                        
                        # Pass 1: Analysis
                        ffmpeg_cmd_pass1 = self._build_two_pass_command(input_path, temp_refined, params, pass_num=1,
                                                                         log_file=refine_log_file)
                        success = self._execute_ffmpeg_with_progress(ffmpeg_cmd_pass1, video_info['duration'])
                        if not success:
                            break
                        
                        # Pass 2: Encoding
                        ffmpeg_cmd = self._build_two_pass_command(input_path, temp_refined, params, pass_num=2,
                                                                  log_file=refine_log_file)
                    else:
                        ffmpeg_cmd = self._build_intelligent_ffmpeg_command(input_path, temp_refined, params)
                    
                    success = self._execute_ffmpeg_with_progress(ffmpeg_cmd, video_info['duration'])
                    
                    # Clean up two-pass log files if applicable
                    if result['strategy'] == 'two_pass':
                        self._cleanup_two_pass_logs(refine_log_file)
                    
                    if success and os.path.exists(temp_refined):
                        new_size_mb = os.path.getsize(temp_refined) / (1024 * 1024)
                        new_utilization = new_size_mb / target_size_mb
                        
                        logger.info(f"Refinement iteration {iteration+1}: {new_size_mb:.2f}MB ({new_utilization*100:.1f}% utilization)")
                        
                        if new_size_mb <= target_size_mb and new_size_mb > refined_result['size_mb']:
                            # Better result - update
                            if os.path.exists(refined_result['temp_file']):
                                os.remove(refined_result['temp_file'])
                            refined_result = {
                                'temp_file': temp_refined,
                                'size_mb': new_size_mb,
                                'strategy': result['strategy'] + '_refined',
                                'quality_score': result.get('quality_score', 0) + 0.5,
                                'params': params
                            }
                            utilization = new_utilization
                        else:
                            # Worse result or exceeded target - stop
                            if os.path.exists(temp_refined):
                                os.remove(temp_refined)
                            break
                    else:
                        break
                else:
                    # Can't refine without bitrate param
                    break
                
                iteration += 1
            
            if refined_result != result:
                logger.info(f"Refinement successful: {refined_result['size_mb']:.2f}MB ({utilization*100:.1f}% utilization)")
            
            return refined_result
            
        except Exception as e:
            logger.warning(f"Refinement failed: {e}")
            return result
    
    def _validate_and_uplift_result(self, result: Dict[str, Any], input_path: str, 
                                    video_info: Dict[str, Any], target_size_mb: float,
                                    quality_attempt: int = 0) -> Optional[Dict[str, Any]]:
        """
        Validate result meets hybrid quality floor; if not, attempt bounded bitrate uplifts.
        Uses perceptual quality scoring instead of SSIM for better content-aware validation.
        
        Args:
            result: Compression result to validate
            input_path: Original input video path
            video_info: Video metadata
            target_size_mb: Target size limit
            quality_attempt: For segments, tracks progressive fallback attempt (0-3)
            
        Returns:
            Validated result or None if quality floor cannot be met
        """
        if not result or result.get('size_mb', float('inf')) > target_size_mb:
            return None
        
        # Read config
        quality_check_enabled = bool(self.config.get('video_compression.quality_controls.hybrid_quality_check.enabled', True))
        if not quality_check_enabled:
            return result  # No enforcement, accept as-is
        
        uplift_enabled = bool(self.config.get('video_compression.quality_controls.uplift.enabled', True))
        uplift_max = int(self.config.get('video_compression.quality_controls.uplift.max_passes', 3) or 0)
        uplift_step = float(self.config.get('video_compression.quality_controls.uplift.bitrate_step', 1.12))
        uplift_cap = float(self.config.get('video_compression.quality_controls.uplift.max_multiplier', 1.35))
        
        temp_file = result.get('temp_file')
        if not temp_file or not os.path.exists(temp_file):
            return result  # Can't validate without file
        
        # Determine if this is a segment (progressive quality floor)
        is_segment = '_segment_' in temp_file.lower() or '_segments' in input_path.lower()
        
        # Get quality floor (with progressive fallback for segments)
        quality_floor_override = None
        if is_segment:
            quality_floor_override = self.quality_scorer.get_segment_quality_floor(quality_attempt)
            logger.info(f"Segment quality validation (attempt {quality_attempt + 1}): floor = {quality_floor_override:.1f}/100")
        
        # Calculate hybrid quality score
        quality_result = self.quality_scorer.calculate_quality_score(
            input_path, temp_file, target_size_mb, 
            is_segment=is_segment,
            quality_floor_override=quality_floor_override
        )
        
        quality_score = quality_result['overall_score']
        quality_passes = quality_result['passes']
        quality_floor = quality_result['floor_used']
        
        logger.info(
            f"Strategy '{result.get('strategy')}' quality: {quality_score:.1f}/100 (floor: {quality_floor:.1f})"
        )
        
        if quality_passes:
            result['quality_score'] = quality_score
            result['quality_details'] = quality_result['components']
            return result  # Passes floor
        
        if not uplift_enabled or uplift_max <= 0:
            logger.warning(
                f"Quality {quality_score:.1f}/100 below floor {quality_floor:.1f}; "
                f"uplift disabled, rejecting result"
            )
            return None
        
        # Attempt uplift
        params = result.get('params', {}).copy()
        base_bitrate = params.get('bitrate')
        if not base_bitrate:
            logger.warning("Cannot uplift: no bitrate in params")
            return None
        
        logger.info(f"Attempting quality uplift: {quality_score:.1f}/100 < {quality_floor:.1f}/100")
        
        best_file = temp_file
        best_size = result['size_mb']
        best_quality = quality_score
        
        for pass_num in range(uplift_max):
            new_bitrate = int(base_bitrate * (uplift_step ** (pass_num + 1)))
            if new_bitrate > base_bitrate * uplift_cap:
                break

            # Safety check: prevent extremely high bitrates using dynamic cap
            safe_cap = self._calculate_safe_bitrate_cap(
                params.get('width', video_info['width']),
                params.get('height', video_info['height']),
                params.get('fps', video_info['fps']),
                video_info['duration']
            )
            new_bitrate = min(new_bitrate, safe_cap)
            
            temp_uplift = os.path.join(self.temp_dir, f"uplift_{result.get('strategy')}_{pass_num}.mp4")
            params['bitrate'] = new_bitrate
            
            logger.info(f"Uplift pass {pass_num+1}: increasing bitrate from {base_bitrate}k to {new_bitrate}k")
            
            # Re-encode with higher bitrate using ORIGINAL strategy
            strategy = result.get('strategy', '')
            
            # Preserve original encoding strategy for better quality
            if 'two_pass' in strategy:
                uplift_id = f"uplift_{result.get('strategy', 'unknown')}_{pass_num}_{int(time.time())}"
                log_file = os.path.join(self.temp_dir, f"ffmpeg2pass_{uplift_id}")
                
                # Pass 1: Analysis
                cmd_pass1 = self._build_two_pass_command(input_path, temp_uplift, params, pass_num=1, log_file=log_file)
                success = self._execute_ffmpeg_with_progress(cmd_pass1, video_info['duration'])
                if not success:
                    break
                
                # Pass 2: Encoding
                cmd = self._build_two_pass_command(input_path, temp_uplift, params, pass_num=2, log_file=log_file)
            else:
                # Use single-pass for non-two-pass strategies
                cmd = self._build_intelligent_ffmpeg_command(input_path, temp_uplift, params)
            
            success = self._execute_ffmpeg_with_progress(cmd, video_info['duration'])
            if not success or not os.path.exists(temp_uplift):
                logger.warning(f"Uplift pass {pass_num+1} failed to produce output")
                break
            
            # Clean up two-pass log files if applicable
            if 'two_pass' in strategy:
                self._cleanup_two_pass_logs(log_file)
            
            new_size = os.path.getsize(temp_uplift) / (1024 * 1024)
            if new_size > target_size_mb:
                logger.info(f"Uplift pass {pass_num+1} exceeded size ({new_size:.2f}MB > {target_size_mb}MB), stopping")
                os.remove(temp_uplift)
                break
            
            # Calculate new quality score
            new_quality_result = self.quality_scorer.calculate_quality_score(
                input_path, temp_uplift, target_size_mb,
                is_segment=is_segment,
                quality_floor_override=quality_floor_override
            )
            new_quality = new_quality_result['overall_score']
            
            # Log detailed uplift progress
            quality_delta = new_quality - best_quality
            logger.info(
                f"Uplift pass {pass_num+1}: bitrate={new_bitrate}k, size={new_size:.2f}MB, "
                f"quality={new_quality:.1f}/100 (Δ{quality_delta:+.1f}, floor={quality_floor:.1f})"
            )
            
            # Replace best if improved
            if os.path.exists(best_file) and best_file != temp_file:
                os.remove(best_file)
            best_file = temp_uplift
            best_size = new_size
            best_quality = new_quality
            
            if new_quality >= quality_floor:
                logger.info(f"Quality floor met after {pass_num+1} uplift passes")
                result['temp_file'] = best_file
                result['size_mb'] = best_size
                result['quality_score'] = best_quality
                result['quality_details'] = new_quality_result['components']
                result['params'] = params
                return result
        
        # Clean up temp uplift files
        if best_file != temp_file and os.path.exists(best_file):
            os.remove(best_file)
        
        # Downscale recovery before rejection/segmentation: try modest resolution reduction
        try:
            enable_recovery = True
            if enable_recovery and not is_segment:  # Only for non-segments
                logger.info("Attempting quality downscale recovery before segmentation")
                # Compute a slightly smaller effective target to encourage downscale
                recovery_target = max(0.85 * target_size_mb, min(target_size_mb, result.get('size_mb', target_size_mb)))
                # Use current video_info but ensure duration is set
                rec_info = dict(video_info)
                rec_info['duration'] = video_info.get('duration', rec_info.get('duration', 0.0))
                rec = self._compress_with_adaptive_resolution(input_path, rec_info, platform_config={}, target_size_mb=recovery_target)
                if rec and rec.get('temp_file') and os.path.exists(rec['temp_file']):
                    new_size = os.path.getsize(rec['temp_file']) / (1024 * 1024)
                    if new_size <= target_size_mb:
                        recovery_quality_result = self.quality_scorer.calculate_quality_score(
                            input_path, rec['temp_file'], target_size_mb,
                            is_segment=False,
                            quality_floor_override=quality_floor_override
                        )
                        recovery_quality = recovery_quality_result['overall_score']
                        if recovery_quality >= quality_floor:
                            logger.info(f"Quality downscale recovery succeeded: size={new_size:.2f}MB, quality={recovery_quality:.1f}/100")
                            result['temp_file'] = rec['temp_file']
                            result['size_mb'] = new_size
                            result['quality_score'] = recovery_quality
                            result['quality_details'] = recovery_quality_result['components']
                            result['strategy'] = 'adaptive_resolution_recovery'
                            return result
                    # Cleanup temp if not used
                    try:
                        os.remove(rec['temp_file'])
                    except Exception:
                        pass
        except Exception as e:
            logger.warning(f"Quality downscale recovery failed: {e}")

        # Log detailed rejection with improvement achieved
        quality_improvement = best_quality - quality_score
        logger.warning(
            f"Quality {best_quality:.1f}/100 remains below floor {quality_floor:.1f}/100 after {uplift_max} uplifts "
            f"(improved {quality_improvement:+.1f} from {quality_score:.1f}/100); rejecting result"
        )
        return None
    
    def _execute_strategies_in_parallel(self, strategies: List[Dict[str, Any]], 
                                       input_path: str, video_info: Dict[str, Any],
                                       platform_config: Dict[str, Any], target_size_mb: float) -> List[Dict[str, Any]]:
        """
        Execute multiple compression strategies in parallel.
        Returns list of successful results.
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        results = []
        # Dynamic worker calculation based on system capabilities
        cpu_count = getattr(self.hardware, 'cpu_count', 4)  # Default to 4 if not available
        max_workers = min(len(strategies), max(2, cpu_count // 2))
        
        logger.info(f"Executing {len(strategies)} strategies in parallel with {max_workers} workers (CPU count: {cpu_count})")
        
        def execute_strategy(strategy):
            try:
                strategy_name = strategy['name']
                strategy_func = strategy['func']
                strategy_args = strategy.get('args', [])
                
                logger.info(f"Starting parallel strategy: {strategy_name}")
                result = strategy_func(input_path, video_info, platform_config, target_size_mb, *strategy_args)
                
                if result:
                    # Refine if under-utilizing
                    result = self._refine_strategy_result(result, input_path, video_info, target_size_mb)
                    logger.info(f"Parallel strategy {strategy_name} completed: {result['size_mb']:.2f}MB")
                    return result
                return None
            except Exception as e:
                logger.warning(f"Parallel strategy {strategy.get('name', 'unknown')} failed: {e}")
                return None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_strategy = {executor.submit(execute_strategy, s): s for s in strategies}
            
            for future in as_completed(future_to_strategy):
                result = future.result()
                if result and result['size_mb'] <= target_size_mb:
                    results.append(result)
        
        logger.info(f"Parallel execution completed: {len(results)} successful results")
        return results
    
    def _compress_with_standard_optimization(self, input_path: str, output_path: str,
                                           target_size_mb: float, platform_config: Dict[str, Any],
                                           video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Use standard dynamic optimization with optional parallel strategy execution"""
        
        best_result = None
        compression_attempts = []
        
        try:
            # Determine if we should use parallel execution
            # Use parallel for medium-high complexity videos
            use_parallel = video_info.get('complexity_score', 5.0) >= 5.0 and self.hardware.has_hardware_acceleration()
            
            if use_parallel:
                logger.info("Using parallel strategy execution for better performance")
                
                # Define strategies to run in parallel
                parallel_strategies = [
                    {
                        'name': 'content_aware',
                        'func': self._compress_with_content_awareness,
                        'args': []
                    },
                    {
                        'name': 'two_pass',
                        'func': self._compress_with_two_pass,
                        'args': [None]  # previous_result = None
                    }
                ]
                
                # Execute in parallel
                results = self._execute_strategies_in_parallel(
                    parallel_strategies, input_path, video_info, platform_config, target_size_mb
                )
                
                # Select best result from parallel execution with SSIM validation
                for result in results:
                    # Validate quality and attempt uplift if needed
                    result = self._validate_and_uplift_result(result, input_path, video_info, target_size_mb)
                    if result:  # Only process validated results
                        compression_attempts.append(result)
                        if result['size_mb'] <= target_size_mb:
                            if not best_result or result['quality_score'] > best_result.get('quality_score', 0):
                                best_result = result
                                logger.info(f"Parallel strategy '{result['strategy']}' selected: {result['size_mb']:.2f}MB")
            else:
                # Sequential execution (original behavior)
                # Strategy 1: Content-aware optimal compression
                logger.info("Attempting Strategy 1: Content-aware optimal compression")
                strategy1_result = self._compress_with_content_awareness(
                    input_path, video_info, platform_config, target_size_mb
                )
                if strategy1_result:
                    # Refine if under-utilizing target
                    strategy1_result = self._refine_strategy_result(strategy1_result, input_path, video_info, target_size_mb)
                    # Validate quality and attempt uplift if needed
                    strategy1_result = self._validate_and_uplift_result(strategy1_result, input_path, video_info, target_size_mb)
                    if strategy1_result:  # Only process validated results
                        compression_attempts.append(strategy1_result)
                        if strategy1_result['size_mb'] <= target_size_mb:
                            best_result = strategy1_result
                            logger.info(f"Strategy 1 successful: {strategy1_result['size_mb']:.2f}MB")
                
                # Strategy 2: Two-pass encoding if Strategy 1 didn't work perfectly
                if not best_result or (strategy1_result and strategy1_result.get('size_mb', 0) > target_size_mb * 0.95):
                    logger.info("Attempting Strategy 2: Two-pass precision encoding")
                    strategy2_result = self._compress_with_two_pass(
                        input_path, video_info, platform_config, target_size_mb, best_result
                    )
                    if strategy2_result:
                        # Refine if under-utilizing target
                        strategy2_result = self._refine_strategy_result(strategy2_result, input_path, video_info, target_size_mb)
                        # Validate quality and attempt uplift if needed
                        strategy2_result = self._validate_and_uplift_result(strategy2_result, input_path, video_info, target_size_mb)
                        if strategy2_result:  # Only process validated results
                            compression_attempts.append(strategy2_result)
                            if strategy2_result['size_mb'] <= target_size_mb:
                                if not best_result or strategy2_result['quality_score'] > best_result.get('quality_score', 0):
                                    best_result = strategy2_result
                                    logger.info(f"Strategy 2 improved result: {strategy2_result['size_mb']:.2f}MB")
            
            # Strategy 3: Adaptive resolution and quality optimization
            if not best_result:
                logger.info("Attempting Strategy 3: Adaptive resolution optimization")
                strategy3_result = self._compress_with_adaptive_resolution(
                    input_path, video_info, platform_config, target_size_mb
                )
                if strategy3_result:
                    # Refine if under-utilizing target
                    strategy3_result = self._refine_strategy_result(strategy3_result, input_path, video_info, target_size_mb)
                    # Validate quality and attempt uplift if needed
                    strategy3_result = self._validate_and_uplift_result(strategy3_result, input_path, video_info, target_size_mb)
                    if strategy3_result:  # Only process validated results
                        compression_attempts.append(strategy3_result)
                        if strategy3_result['size_mb'] <= target_size_mb:
                            best_result = strategy3_result
                            logger.info(f"Strategy 3 successful: {strategy3_result['size_mb']:.2f}MB")
            
            # Strategy 4: Aggressive optimization as last resort
            if not best_result:
                logger.info("Attempting Strategy 4: Aggressive optimization (last resort)")
                strategy4_result = self._compress_with_aggressive_optimization(
                    input_path, video_info, platform_config, target_size_mb
                )
                if strategy4_result:
                    # Validate quality and attempt uplift if needed
                    strategy4_result = self._validate_and_uplift_result(strategy4_result, input_path, video_info, target_size_mb)
                    if strategy4_result:  # Only process validated results
                        compression_attempts.append(strategy4_result)
                        if strategy4_result['size_mb'] <= target_size_mb:
                            best_result = strategy4_result
                            logger.info(f"Strategy 4 successful: {strategy4_result['size_mb']:.2f}MB")
            
            if not best_result:
                # Check if segmentation fallback is enabled
                on_fail_segment = bool(self.config.get('video_compression.quality_controls.on_fail_segment', False))
                if on_fail_segment:
                    logger.warning("All strategies failed to meet quality floor; falling back to segmentation")
                    return self._compress_with_segmentation(
                        input_path, output_path, target_size_mb, platform_config, video_info, platform=None
                    )
                raise RuntimeError(f"Failed to compress video under {target_size_mb}MB with acceptable quality")
            
            # Move best result to final output
            shutil.move(best_result['temp_file'], output_path)
            
            # Defensive: final size must not exceed target
            try:
                final_mb = os.path.getsize(output_path) / (1024 * 1024)
                if final_mb > target_size_mb:
                    logger.error(f"Final output exceeds size limit: {final_mb:.2f}MB > {target_size_mb}MB")
                    try:
                        os.remove(output_path)
                    except Exception:
                        pass
                    raise RuntimeError(f"Final output exceeds size limit: {final_mb:.2f}MB > {target_size_mb}MB")
            except Exception as e:
                if isinstance(e, RuntimeError):
                    raise
                logger.warning(f"Final size validation error: {e}")

            # Clean up other attempts
            for attempt in compression_attempts:
                if attempt != best_result and os.path.exists(attempt['temp_file']):
                    os.remove(attempt['temp_file'])
            
            # Generate final results
            results = self._get_compression_results(input_path, output_path, video_info, "dynamic_optimized")
            results['optimization_strategy'] = best_result['strategy']
            results['quality_score'] = best_result.get('quality_score', 0)
            results['attempts_made'] = len(compression_attempts)
            
            logger.info(f"Dynamic compression completed: {results.get('compression_ratio', 0):.1f}% reduction, "
                       f"strategy: {best_result.get('strategy', 'unknown')}")

            # Note: Do not create folder.jpg for single MP4 outputs; reserved for segmented outputs only
            
            return results
            
        except Exception as e:
            # Clean up any temporary files
            for attempt in compression_attempts:
                if os.path.exists(attempt['temp_file']):
                    os.remove(attempt['temp_file'])
            raise
    
    def _analyze_video_content(self, video_path: str) -> Dict[str, Any]:
        """Analyze video content for intelligent compression decisions"""
        try:
            # Respect shutdown requests: abort analysis without noisy errors
            if self.shutdown_requested:
                raise GracefulCancellation()
            # Get basic video info
            # Lightweight ffprobe: avoid frame-level enumeration which is slow on long videos
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams',
                video_path
            ]
            
            # Increase timeout for large files to prevent premature timeouts
            try:
                file_size_mb_for_timeout = os.path.getsize(video_path) / (1024 * 1024)
            except Exception:
                file_size_mb_for_timeout = 0
            probe_timeout = 60
            if file_size_mb_for_timeout > 1000:
                probe_timeout = 240
            elif file_size_mb_for_timeout > 300:
                probe_timeout = 120

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=probe_timeout
            )
            
            if result.returncode != 0:
                # If shutdown occurred during probe, treat as graceful cancel
                if self.shutdown_requested:
                    raise GracefulCancellation()
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
            
            stdout_text = result.stdout or ""
            probe_data = json.loads(stdout_text) if stdout_text.strip() else {}
            
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
            
        except GracefulCancellation:
            # Propagate cancellation to caller to avoid further work/log noise
            raise
        except subprocess.TimeoutExpired as e:
            # Gracefully degrade on timeout for massive videos
            logger.warning(f"ffprobe analysis timed out after {getattr(e, 'timeout', 'unknown')}s; using basic video info")
            return self._get_basic_video_info(video_path)
        except Exception as e:
            if self.shutdown_requested:
                # Avoid alarming error logs on shutdown; return minimal info to allow fast exit
                logger.info("Video analysis aborted due to shutdown request")
                raise GracefulCancellation()
            else:
                logger.warning(f"Failed to analyze video content quickly; using basic info: {e}")
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
        # This is a simplified estimation - in reality you'd analyze actual frames
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
    
    def _compress_with_content_awareness(self, input_path: str, video_info: Dict[str, Any], 
                                       platform_config: Dict[str, Any], target_size_mb: float) -> Optional[Dict[str, Any]]:
        """Content-aware compression strategy"""
        try:
            temp_output = os.path.join(self.temp_dir, "content_aware_output.mp4")
            
            # Calculate optimal parameters based on content analysis
            params = self._calculate_intelligent_params(video_info, platform_config, target_size_mb)
            
            # Validate parameters before encoding
            if not self._validate_encoding_parameters(
                params.get('width', video_info['width']),
                params.get('height', video_info['height']),
                params.get('fps', video_info['fps']),
                params.get('bitrate', 0)
            ):
                logger.warning("Content-aware strategy rejected: insufficient BPP for quality")
                return None
            
            # Build and execute FFmpeg command
            ffmpeg_cmd = self._build_intelligent_ffmpeg_command(input_path, temp_output, params)
            self._execute_ffmpeg_with_progress(ffmpeg_cmd, video_info['duration'])
            
            if os.path.exists(temp_output):
                size_mb = os.path.getsize(temp_output) / (1024 * 1024)
                quality_score = self._calculate_quality_score(params, video_info)
                
                return {
                    'temp_file': temp_output,
                    'size_mb': size_mb,
                    'strategy': 'content_aware',
                    'quality_score': quality_score,
                    'params': params
                }
            
        except Exception as e:
            logger.warning(f"Content-aware compression failed: {e}")
            return None
    
    def _compress_with_two_pass(self, input_path: str, video_info: Dict[str, Any], 
                              platform_config: Dict[str, Any], target_size_mb: float,
                              previous_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Two-pass precision encoding for optimal quality/size ratio"""
        try:
            temp_output = os.path.join(self.temp_dir, "two_pass_output.mp4")
            # Use unique passlogfile for two-pass strategy to avoid conflicts
            strategy_id = f"two_pass_{int(time.time())}"
            temp_log = os.path.join(self.temp_dir, f"ffmpeg2pass_{strategy_id}")

            # Calculate precise bitrate for target size
            target_bitrate = self._calculate_precise_bitrate(video_info, target_size_mb)
            # Safety check: prevent extremely high bitrates that might cause FFmpeg to crash
            # Use dynamic cap based on video characteristics
            safe_cap = self._calculate_safe_bitrate_cap(
                video_info['width'], video_info['height'], 
                min(video_info['fps'], 30), video_info['duration']
            )
            target_bitrate = min(target_bitrate, safe_cap)

            # Adjust parameters based on previous attempt if available
            params = self._calculate_two_pass_params(video_info, platform_config, target_bitrate, previous_result)

            # Pass 1: Analysis
            ffmpeg_cmd_pass1 = self._build_two_pass_command(input_path, temp_output, params, pass_num=1, log_file=temp_log)
            
            # Store the command for error logging
            pass1_cmd_str = ' '.join(ffmpeg_cmd_pass1)
            
            pass1_success = self._execute_ffmpeg_with_progress(ffmpeg_cmd_pass1, video_info['duration'])
            
            if not pass1_success:
                # Log the actual command and check for common issues
                logger.warning("Two-pass compression: Pass 1 failed")
                logger.debug(f"Failed Pass 1 command: {pass1_cmd_str}")
                
                # Check if log file was partially created
                if os.path.exists(f"{temp_log}-0.log"):
                    try:
                        log_size = os.path.getsize(f"{temp_log}-0.log")
                        logger.debug(f"Pass 1 log file exists but incomplete: {log_size} bytes")
                    except Exception:
                        pass
                else:
                    logger.debug("Pass 1 log file was not created - FFmpeg may have failed immediately")
                
                # Check common failure reasons
                if target_bitrate < 500:
                    logger.warning(f"Pass 1 may have failed due to very low bitrate: {target_bitrate}k")
                
                return None

            # Pass 2: Encoding
            ffmpeg_cmd_pass2 = self._build_two_pass_command(input_path, temp_output, params, pass_num=2, log_file=temp_log)
            pass2_success = self._execute_ffmpeg_with_progress(ffmpeg_cmd_pass2, video_info['duration'])
            
            if not pass2_success:
                logger.warning("Two-pass compression: Pass 2 failed")
                logger.debug(f"Failed Pass 2 command: {' '.join(ffmpeg_cmd_pass2)}")
                return None
            
            # Clean up log files
            self._cleanup_two_pass_logs(temp_log)
            
            if os.path.exists(temp_output):
                size_mb = os.path.getsize(temp_output) / (1024 * 1024)
                quality_score = self._calculate_quality_score(params, video_info) + 1  # Bonus for two-pass
                
                return {
                    'temp_file': temp_output,
                    'size_mb': size_mb,
                    'strategy': 'two_pass',
                    'quality_score': quality_score,
                    'params': params
                }
            
        except Exception as e:
            logger.warning(f"Two-pass compression failed: {e}")
            return None
    
    def _compress_with_adaptive_resolution(self, input_path: str, video_info: Dict[str, Any], 
                                         platform_config: Dict[str, Any], target_size_mb: float) -> Optional[Dict[str, Any]]:
        """Adaptive resolution scaling based on content and size constraints"""
        try:
            temp_output = os.path.join(self.temp_dir, "adaptive_resolution_output.mp4")
            
            # Calculate optimal resolution for target size
            optimal_resolution = self._calculate_optimal_resolution(video_info, target_size_mb, platform_config)
            
            params = self._calculate_adaptive_params(video_info, platform_config, target_size_mb, optimal_resolution)
            
            # Validate parameters before encoding
            if not self._validate_encoding_parameters(
                params.get('width', video_info['width']),
                params.get('height', video_info['height']),
                params.get('fps', video_info['fps']),
                params.get('bitrate', 0)
            ):
                logger.warning("Adaptive resolution strategy rejected: insufficient BPP for quality")
                return None
            
            ffmpeg_cmd = self._build_adaptive_ffmpeg_command(input_path, temp_output, params)
            self._execute_ffmpeg_with_progress(ffmpeg_cmd, video_info['duration'])
            
            if os.path.exists(temp_output):
                size_mb = os.path.getsize(temp_output) / (1024 * 1024)
                quality_score = self._calculate_quality_score(params, video_info)
                
                return {
                    'temp_file': temp_output,
                    'size_mb': size_mb,
                    'strategy': 'adaptive_resolution',
                    'quality_score': quality_score,
                    'params': params
                }
            
        except Exception as e:
            logger.warning(f"Adaptive resolution compression failed: {e}")
            return None
    
    def _compress_with_segmentation(self, input_path: str, output_path: str, target_size_mb: float,
                                  platform_config: Dict[str, Any], video_info: Dict[str, Any], 
                                  platform: str = None) -> Dict[str, Any]:
        """Compress video using segmentation approach"""
        try:
            logger.info("Starting video segmentation compression")
            
            # Use video segmenter to handle the segmentation
            result = self.video_segmenter.segment_video(
                input_video=input_path,
                output_base_path=output_path,
                platform=platform,
                max_size_mb=target_size_mb
            )
            
            if result.get('success', False):
                # If segmentation was successful, return the results
                if 'segments' in result:
                    # Multiple segments were created
                    total_size = sum(segment.get('size_mb', 0) for segment in result['segments'])
                    return {
                        'success': True,
                        'input_file': input_path,
                        'output_file': output_path,
                        'method': 'segmentation',
                        'original_size_mb': os.path.getsize(input_path) / (1024 * 1024),
                        'compressed_size_mb': total_size,
                        'size_mb': total_size,
                        'compression_ratio': ((os.path.getsize(input_path) - (total_size * 1024 * 1024)) / os.path.getsize(input_path)) * 100,
                        'space_saved_mb': (os.path.getsize(input_path) - (total_size * 1024 * 1024)) / (1024 * 1024),
                        'video_info': video_info,
                        'optimization_strategy': 'segmentation',
                        'quality_score': 8.0,  # High quality for segments
                        'attempts_made': 1,
                        'encoder_used': 'segmentation',
                        'segments': result['segments'],
                        'num_segments': result['num_segments'],
                        'output_folder': result.get('output_folder', ''),
                        'segment_duration': result.get('segment_duration', 0)
                    }
                else:
                    # Single file was processed
                    return result
            else:
                # Segmentation failed, fall back to standard compression
                logger.warning("Video segmentation failed, falling back to standard compression")
                return self._compress_with_standard_optimization(
                    input_path, output_path, target_size_mb, platform_config, video_info
                )
                
        except Exception as e:
            logger.error(f"Error in video segmentation compression: {e}")
            # Fall back to standard compression
            return self._compress_with_standard_optimization(
                input_path, output_path, target_size_mb, platform_config, video_info
            )
    

    def _compress_with_aggressive_optimization(self, input_path: str, video_info: Dict[str, Any],
                                             platform_config: Dict[str, Any], target_size_mb: float) -> Optional[Dict[str, Any]]:
        """Aggressive optimization as last resort"""
        try:
            temp_output = os.path.join(self.temp_dir, "aggressive_output.mp4")
            
            # Very aggressive parameters to ensure size compliance
            params = self._calculate_aggressive_params(video_info, platform_config, target_size_mb)
            
            ffmpeg_cmd = self._build_aggressive_ffmpeg_command(input_path, temp_output, params)
            self._execute_ffmpeg_with_progress(ffmpeg_cmd, video_info['duration'])
            
            if os.path.exists(temp_output):
                size_mb = os.path.getsize(temp_output) / (1024 * 1024)
                quality_score = self._calculate_quality_score(params, video_info) - 1  # Penalty for aggressive
                
                return {
                    'temp_file': temp_output,
                    'size_mb': size_mb,
                    'strategy': 'aggressive',
                    'quality_score': quality_score,
                    'params': params
                }
            
        except Exception as e:
            logger.warning(f"Aggressive compression failed: {e}")
            return None

    def _compress_with_aggressive_settings(self, input_path: str, output_path: str,
                                          target_size_mb: float, platform_config: Dict[str, Any],
                                          video_info: Dict[str, Any], aggressiveness: int = 1) -> Optional[Dict[str, Any]]:
        """Unified aggressive compression with severity levels 1-4
        
        Args:
            aggressiveness: 1 = Conservative aggressive (90% res, moderate CRF)
                          2 = Standard aggressive (75% res, higher CRF)
                          3 = Very aggressive (60% res, low FPS)
                          4 = Extreme (minimal quality, last resort)
        """
        # Aggressiveness level configurations
        configs = {
            1: {  # Conservative aggressive (matches old "standard" behavior)
                'res_scale': 0.90,  # 90% of original (old high-res behavior)
                'fps_scale': 0.9,
                'crf': 28,  # Moderate CRF
                'min_res': (720, 540),
                'size_budget': 0.82,
                'audio_bitrate': 80,
                'name': 'conservative_aggressive'
            },
            2: {  # Standard aggressive (matches old aggressive behavior exactly)
                'res_scale': 0.80,  # 80% of original (old standard-res behavior)
                'fps_scale': 0.8,
                'crf': 35,  # Matches old CRF
                'min_res': (640, 480),
                'size_budget': 0.75,  # Matches old 75% budget
                'audio_bitrate': 64,
                'name': 'aggressive'
            },
            3: {  # Very aggressive
                'res_scale': 0.65,
                'fps_scale': 0.6,
                'crf': 38,
                'min_res': (540, 360),
                'size_budget': 0.70,
                'audio_bitrate': 48,
                'name': 'very_aggressive'
            },
            4: {  # Extreme (last resort)
                'res_scale': 0.50,
                'fps_scale': 0.5,
                'crf': 42,
                'min_res': (480, 320),
                'size_budget': 0.65,
                'audio_bitrate': 32,
                'name': 'extreme_aggressive'
            }
        }
        
        config = configs.get(aggressiveness, configs[2])
        logger.info(f"Using aggressive level {aggressiveness}: {config['name']}")
        
        temp_output = os.path.join(self.temp_dir, f"aggressive_{aggressiveness}.mp4")
        
        try:
            encoder, accel_type = self.hardware.get_best_encoder("h264")
            
            # Calculate aggressive resolution
            aggressive_width = max(config['min_res'][0], int(video_info['width'] * config['res_scale']))
            aggressive_height = max(config['min_res'][1], int(video_info['height'] * config['res_scale']))
            
            # Ensure even dimensions
            aggressive_width = aggressive_width if aggressive_width % 2 == 0 else aggressive_width - 1
            aggressive_height = aggressive_height if aggressive_height % 2 == 0 else aggressive_height - 1
            
            # Calculate FPS
            aggressive_fps = max(12, int(video_info['fps'] * config['fps_scale']))
            
            # Calculate bitrate
            duration = video_info['duration']
            target_bits = target_size_mb * 8 * 1024 * 1024 * config['size_budget']
            video_bitrate = max(int(target_bits / duration / 1000), 100)
            
            params = {
                'encoder': encoder,
                'acceleration_type': accel_type,
                'width': aggressive_width,
                'height': aggressive_height,
                'fps': aggressive_fps,
                'crf': config['crf'] if accel_type == 'software' else None,
                'preset': 'veryfast',
                'bitrate': video_bitrate,
                'audio_bitrate': config['audio_bitrate']
            }
            
            if platform_config:
                params.update(self._apply_platform_constraints(params, platform_config))
            
            ffmpeg_cmd = self._build_aggressive_ffmpeg_command(input_path, temp_output, params)
            self._execute_ffmpeg_with_progress(ffmpeg_cmd, duration)
            
            if os.path.exists(temp_output):
                compressed_size_mb = os.path.getsize(temp_output) / (1024 * 1024)
                
                # Quality score decreases with aggressiveness
                base_quality = 5.0 - (aggressiveness * 1.0)
                
                return {
                    'success': True,
                    'input_file': input_path,
                    'output_file': temp_output,
                    'method': config['name'],
                    'original_size_mb': os.path.getsize(input_path) / (1024 * 1024),
                    'compressed_size_mb': compressed_size_mb,
                    'compression_ratio': ((os.path.getsize(input_path) - os.path.getsize(temp_output)) / os.path.getsize(input_path)) * 100,
                    'space_saved_mb': (os.path.getsize(input_path) - os.path.getsize(temp_output)) / (1024 * 1024),
                    'video_info': video_info,
                    'optimization_strategy': config['name'],
                    'quality_score': base_quality,
                    'attempts_made': 1,
                    'encoder_used': encoder
                }
                
        except Exception as e:
            logger.warning(f"Aggressive level {aggressiveness} compression failed: {e}")
            return None
    
    def _compress_with_aggressive_single_file(self, input_path: str, output_path: str,
                                           target_size_mb: float, platform_config: Dict[str, Any],
                                           video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Aggressive single-file optimization when segmentation is disabled"""
        logger.info("Starting aggressive single-file optimization with progressive severity")

        # Try each aggressiveness level in order
        for level in range(1, 5):
            logger.info(f"Attempting aggressive level {level}/4")
            
            result = self._compress_with_aggressive_settings(
                input_path, output_path, target_size_mb, platform_config, video_info, 
                aggressiveness=level
            )
            
            if result and result.get('compressed_size_mb', float('inf')) <= target_size_mb:
                # Move temp to final output
                if os.path.exists(result['output_file']):
                    try:
                        if os.path.exists(output_path):
                            os.remove(output_path)
                    except Exception:
                        pass
                    shutil.move(result['output_file'], output_path)
                    result['output_file'] = output_path
                return result

        # If all levels failed, return error
        return {
            'success': False,
            'error': 'All aggressive single-file optimization levels (1-4) failed to meet target size',
            'method': 'failed_aggressive_single_file'
        }

    def _calculate_intelligent_params(self, video_info: Dict[str, Any], 
                                    platform_config: Dict[str, Any], target_size_mb: float) -> Dict[str, Any]:
        """Calculate intelligent compression parameters based on content analysis"""
        
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
                params['crf'] = base_crf + 2  # Higher CRF for complex content
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
        
        # Intelligent bitrate calculation
        params['bitrate'] = self._calculate_content_aware_bitrate(video_info, target_size_mb)
        
        # Platform-specific adjustments
        if platform_config:
            params.update(self._apply_platform_constraints(params, platform_config))
        
        return params
    
    def _calculate_content_aware_bitrate(self, video_info: Dict[str, Any], target_size_mb: float) -> int:
        """Calculate bitrate based on content complexity and target size"""
        
        # Base calculation accounting for audio overhead
        duration = video_info['duration']
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
        
        # Resolution-aware minimum bitrates
        fps = video_info.get('fps', 30)
        
        # Calculate bits-per-pixel minimum (0.023 for h264_8bit)
        min_bpp = 0.023
        pixels_per_sec = pixel_count * fps
        min_bitrate_bpp = int(math.ceil(min_bpp * pixels_per_sec / 1000.0))
        
        # Resolution-based absolute minimums (fallback)
        if pixel_count >= 1920 * 1080:  # 1080p+
            min_bitrate_res = 1500
        elif pixel_count >= 1280 * 720:  # 720p+
            min_bitrate_res = 800
        elif pixel_count >= 854 * 480:   # 480p+
            min_bitrate_res = 400
        else:
            min_bitrate_res = 200
        
        # Use highest of calculated, BPP-based, or resolution-based minimum
        min_bitrate = max(min_bitrate_bpp, min_bitrate_res)
        final_bitrate = max(final_bitrate, min_bitrate)
        
        if final_bitrate < min_bitrate * 1.1:
            logger.warning(
                f"Bitrate {final_bitrate}k is at/near minimum {min_bitrate}k for "
                f"{video_info['width']}x{video_info['height']}@{fps}fps. "
                f"Quality may be severely degraded. Consider reducing resolution or increasing target size."
            )
        
        # Check if compression ratio is extreme (>10x)
        original_size_mb = video_info.get('size_bytes', 0) / (1024 * 1024)
        if original_size_mb > 0:
            compression_ratio = original_size_mb / target_size_mb
            if compression_ratio > 10:
                logger.warning(
                    f"Extreme compression ratio detected: {compression_ratio:.1f}x "
                    f"({original_size_mb:.1f}MB -> {target_size_mb:.1f}MB). "
                    f"Consider using segmentation or resolution reduction for better results."
                )
        
        return final_bitrate
    
    def _calculate_precise_bitrate(self, video_info: Dict[str, Any], target_size_mb: float) -> int:
        """Calculate precise bitrate for two-pass encoding"""
        # Use same resolution-aware calculation as content-aware
        return self._calculate_content_aware_bitrate(video_info, target_size_mb)
    
    def _calculate_optimal_resolution(self, video_info: Dict[str, Any], target_size_mb: float, 
                                    platform_config: Dict[str, Any]) -> Tuple[int, int]:
        """Calculate optimal resolution for target file size"""
        
        # Check cache first
        cache_key = (video_info['width'], video_info['height'], 
                    video_info.get('duration', 0), target_size_mb,
                    video_info.get('complexity_score', 5.0))
        if cache_key in self._resolution_cache:
            logger.debug(f"Using cached resolution calculation for {cache_key[0]}x{cache_key[1]}")
            return self._resolution_cache[cache_key]
        
        original_width, original_height = video_info['width'], video_info['height']
        original_pixels = original_width * original_height
        
        logger.info(f"Calculating optimal resolution: original {original_width}x{original_height} "
                   f"({original_pixels:,} pixels), target size {target_size_mb}MB")
        
        # Safety check: if the target size is much larger than what we need, 
        # don't aggressively downscale. This prevents the 1.19MB issue.
        original_size_mb = video_info.get('size_bytes', 0) / (1024 * 1024)
        if original_size_mb > 0 and target_size_mb > original_size_mb * 2:
            logger.info(f"Target size {target_size_mb}MB is much larger than original {original_size_mb:.1f}MB, "
                       f"keeping original resolution to maintain quality")
            return original_width, original_height
        
        # For high-resolution videos, be much more conservative about downscaling
        # The goal is to maintain quality while meeting size constraints
        if original_pixels > 1000000:  # Over 1M pixels (like your 1080x1440)
            logger.info(f"High-resolution video detected ({original_pixels:,} pixels), using conservative scaling")
            
            # Start with a much more conservative approach
            # Instead of aggressive downscaling, try to maintain resolution and adjust other parameters
            if original_pixels <= 2000000:  # 1M-2M pixels (like 1080p, 1440p)
                # For videos in this range, try to keep at least 80% of original resolution
                min_scale_factor = 0.8
            else:  # Over 2M pixels (4K+)
                # For very high resolution, be more aggressive but still reasonable
                min_scale_factor = 0.6
            
            # Calculate what we can afford based on target size
            duration = video_info['duration']
            complexity = video_info.get('complexity_score', 5.0)
            
            # Conservative bitrate calculation (reserve for audio)
            target_bits = target_size_mb * 8 * 1024 * 1024 * 0.88  # 88% for video, 12% for audio
            bits_per_second = target_bits / duration
            
            # Use codec-based efficiency (assume h264_8bit as conservative default)
            codec_efficiency = self._calculate_codec_efficiency('h264_8bit', complexity)
            # Convert efficiency to BPP requirement
            required_bpp = 1.0 / codec_efficiency
            
            # Calculate affordable pixels at target FPS (assume 24fps for calculations)
            target_fps = 24
            
            # CRITICAL FIX: Apply conservative multiplier to match old behavior
            # Old formula gave: bps * 0.0146 pixels
            # New formula without multiplier gives: bps / 0.6 = bps * 1.667
            # That's 114x more! Need to scale down to match quality expectations
            conservative_multiplier = 0.35  # Empirically tuned to match old behavior
            affordable_pixels = int((bits_per_second / (required_bpp * target_fps)) * conservative_multiplier)
            
            logger.info(f"High-res calculation: duration={duration:.1f}s, complexity={complexity:.1f}, "
                       f"required_bpp={required_bpp:.4f}, conservative_mult={conservative_multiplier:.2f}, "
                       f"affordable_pixels={affordable_pixels:,}")
            
            if affordable_pixels >= original_pixels:
                # Can keep original resolution
                optimal_width, optimal_height = original_width, original_height
                logger.info(f"Keeping original resolution: {optimal_width}x{optimal_height}")
            else:
                # Need to scale down, but be conservative
                scale_factor = math.sqrt(affordable_pixels / original_pixels)
                
                # Apply minimum scale factor to prevent extreme downscaling
                scale_factor = max(scale_factor, min_scale_factor)
                
                optimal_width = int(original_width * scale_factor)
                optimal_height = int(original_height * scale_factor)
                
                logger.info(f"Conservative scaling: factor={scale_factor:.3f}, new size={optimal_width}x{optimal_height}")
        else:
            # For lower resolution videos, use codec-based model
            duration = video_info['duration']
            complexity = video_info.get('complexity_score', 5.0)
            
            # Conservative bitrate calculation (reserve for audio)
            target_bits = target_size_mb * 8 * 1024 * 1024 * 0.88  # 88% for video
            bits_per_second = target_bits / duration
            
            # Use codec-based efficiency (h264_8bit as default)
            codec_efficiency = self._calculate_codec_efficiency('h264_8bit', complexity)
            required_bpp = 1.0 / codec_efficiency
            
            # Calculate affordable pixels at 24fps
            target_fps = 24
            
            # CRITICAL FIX: Apply conservative multiplier to match old behavior
            # This prevents trying to encode at too high resolution with insufficient bitrate
            conservative_multiplier = 0.35  # Empirically tuned to match old behavior
            affordable_pixels = int((bits_per_second / (required_bpp * target_fps)) * conservative_multiplier)
            
            logger.info(f"Standard calculation: duration={duration:.1f}s, complexity={complexity:.1f}, "
                       f"required_bpp={required_bpp:.4f}, conservative_mult={conservative_multiplier:.2f}, "
                       f"affordable_pixels={affordable_pixels:,}")
            
            if affordable_pixels >= original_pixels:
                optimal_width, optimal_height = original_width, original_height
                logger.info(f"Keeping original resolution: {optimal_width}x{optimal_height}")
            else:
                scale_factor = math.sqrt(affordable_pixels / original_pixels)
                
                # Apply minimum scale factor
                min_scale_factor = 0.6  # Never go below 60% of original
                scale_factor = max(scale_factor, min_scale_factor)
                
                optimal_width = int(original_width * scale_factor)
                optimal_height = int(original_height * scale_factor)
                
                logger.info(f"Standard scaling: factor={scale_factor:.3f}, new size={optimal_width}x{optimal_height}")
        
        # Apply platform constraints if available
        if platform_config:
            max_width = platform_config.get('max_width', optimal_width)
            max_height = platform_config.get('max_height', optimal_height)
            
            # Calculate how much we need to scale to fit within platform constraints
            # while preserving aspect ratio
            width_scale = max_width / optimal_width if optimal_width > max_width else 1.0
            height_scale = max_height / optimal_height if optimal_height > max_height else 1.0
            
            # Use the more restrictive scale factor to ensure we fit within both constraints
            scale_factor = min(width_scale, height_scale)
            
            # Apply the scale factor to maintain aspect ratio
            optimal_width = int(optimal_width * scale_factor)
            optimal_height = int(optimal_height * scale_factor)
            
            logger.info(f"Applied platform constraints: max {max_width}x{max_height}, "
                       f"scale factor: {scale_factor:.3f}, final size: {optimal_width}x{optimal_height}")
        else:
            # When no platform constraints, apply intelligent defaults based on video characteristics
            if original_height > original_width:  # Vertical video
                # For vertical videos, maintain reasonable minimums (slightly less strict)
                min_width = max(480, int(original_width * 0.65))   # 65% of original width
                min_height = max(640, int(original_height * 0.65)) # 65% of original height
            else:  # Horizontal video
                # Reduce the minimum to allow smaller outputs when bitrate is constrained
                min_width = max(640, int(original_width * 0.6))   # 60% of original width
                min_height = max(480, int(original_height * 0.6)) # 60% of original height

            optimal_width = max(optimal_width, min_width)
            optimal_height = max(optimal_height, min_height)

            logger.info(f"No platform constraints, applied intelligent defaults: min {min_width}x{min_height}, "
                       f"adjusted size={optimal_width}x{optimal_height}")
        
        # Ensure even dimensions for H.264 compatibility
        optimal_width = optimal_width if optimal_width % 2 == 0 else optimal_width - 1
        optimal_height = optimal_height if optimal_height % 2 == 0 else optimal_height - 1
        
        # Final safety check - ensure reasonable minimum resolution
        # These are much higher than the previous 128x96 to prevent extreme downscaling
        if original_pixels > 1000000:  # High resolution videos
            final_min_width = 720   # Much higher minimum for high-res videos
            final_min_height = 540  # Much higher minimum for high-res videos
        else:  # Lower resolution videos
            final_min_width = 480   # Reasonable minimum for standard videos
            final_min_height = 360  # Reasonable minimum for standard videos
        
        final_width = max(optimal_width, final_min_width)
        final_height = max(optimal_height, final_min_height)

        # Never upscale beyond original resolution
        try:
            if final_width > original_width:
                final_width = original_width if original_width % 2 == 0 else original_width - 1
            if final_height > original_height:
                final_height = original_height if original_height % 2 == 0 else original_height - 1
        except Exception:
            pass
        
        logger.info(f"Final resolution: {final_width}x{final_height} "
                   f"(min allowed: {final_min_width}x{final_min_height})")
        
        # Store in cache before returning
        self._resolution_cache[cache_key] = (final_width, final_height)
        
        return final_width, final_height
    
    def _calculate_quality_score(self, params: Dict[str, Any], video_info: Dict[str, Any]) -> float:
        """Calculate estimated quality score for comparison (heuristic-based for strategy selection)
        
        Note: This is a fast heuristic for comparing strategies during compression.
        For actual quality validation, use QualityScorer.calculate_quality_score() instead.
        """
        score = 10.0  # Start with perfect score
        
        # Resolution penalty
        original_pixels = video_info['width'] * video_info['height']
        current_pixels = params.get('width', original_pixels) * params.get('height', video_info['height'])
        resolution_factor = current_pixels / max(original_pixels, 1)
        score *= resolution_factor
        
        # CRF penalty (if using software encoding)
        if params.get('crf'):
            crf = params['crf']
            # CRF 18 = excellent, 23 = good, 28 = acceptable, 35+ = poor
            crf_score = max(0, (40 - crf) / 22)  # Normalize to 0-1
            score *= crf_score
        
        # Bitrate consideration
        if params.get('bitrate'):
            # Higher bitrate generally means better quality
            # Normalize against resolution for fair comparison
            bpp = params['bitrate'] * 1000 / (current_pixels * params.get('fps', 30))
            # Target BPP around 0.02-0.05 for good quality
            bpp_score = min(bpp / 0.03, 1.2)  # Cap bonus at 1.2x
            score *= bpp_score
        
        # FPS penalty for reduced frame rates
        original_fps = video_info.get('fps', 30)
        current_fps = params.get('fps', original_fps)
        fps_factor = current_fps / max(original_fps, 1)
        score *= max(fps_factor, 0.7)  # At least 70% of score even at low FPS
        
        return max(0, min(10, score))
    
    # Include all the helper methods from the original implementation
    def _get_basic_video_info(self, video_path: str) -> Dict[str, Any]:
        """Fallback method for basic video info"""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams',
                video_path
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
                raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
            
            stdout_text = result.stdout or ""
            probe_data = json.loads(stdout_text) if stdout_text.strip() else {}
            
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
    
    def _calculate_two_pass_params(self, video_info: Dict[str, Any], platform_config: Dict[str, Any], 
                                 target_bitrate: int, previous_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate parameters for two-pass encoding"""
        # Force software encoding for two-pass; hardware encoders generally don't support it reliably
        encoder, accel_type = self.hardware.get_best_encoder("h264")
        if accel_type != 'software':
            encoder = 'libx264'
            accel_type = 'software'
        
        params = {
            'encoder': encoder,
            'acceleration_type': accel_type,
            'width': video_info['width'],
            'height': video_info['height'],
            'fps': min(video_info['fps'], 30),
            'bitrate': target_bitrate,
            'preset': 'slow',  # Use slower preset for two-pass
        }
        
        # Adjust based on previous attempt
        if previous_result and previous_result.get('size_mb', 0) > 0:
            previous_params = previous_result.get('params', {})
            # Fine-tune bitrate based on previous result
            if previous_result['size_mb'] > target_bitrate * video_info['duration'] / (8 * 1024):
                params['bitrate'] = int(target_bitrate * 0.9)  # Reduce bitrate
        
        # Apply platform constraints
        if platform_config:
            params.update(self._apply_platform_constraints(params, platform_config))
        
        return params
    
    def _calculate_adaptive_params(self, video_info: Dict[str, Any], platform_config: Dict[str, Any], 
                                 target_size_mb: float, optimal_resolution: Tuple[int, int]) -> Dict[str, Any]:
        """Calculate parameters for adaptive resolution encoding"""
        
        encoder, accel_type = self.hardware.get_best_encoder("h264")
        
        params = {
            'encoder': encoder,
            'acceleration_type': accel_type,
            'width': optimal_resolution[0],
            'height': optimal_resolution[1],
            'fps': min(video_info['fps'], 30),
            'preset': 'medium',
        }
        
        # Calculate bitrate for new resolution and enforce min_bpp floor to avoid starving bits
        pixel_ratio = (optimal_resolution[0] * optimal_resolution[1]) / (video_info['width'] * video_info['height'])
        base_bitrate = self._calculate_content_aware_bitrate(video_info, target_size_mb)
        computed_bitrate = int(base_bitrate * pixel_ratio * 1.6)
        # Enforce BPP floor based on codec family (assume h264 8-bit floor if unknown)
        min_bpp_map = {
            'h264_8bit': float(self.config.get('video_compression.codec.min_bpp.h264_8bit', 0.023)),
            'h264_10bit': float(self.config.get('video_compression.codec.min_bpp.h264_10bit', 0.020)),
            'hevc_8bit': float(self.config.get('video_compression.codec.min_bpp.hevc_8bit', 0.018)),
            'hevc_10bit': float(self.config.get('video_compression.codec.min_bpp.hevc_10bit', 0.015)),
        }
        # Infer a floor using selected encoder family
        enc = (encoder or '').lower()
        if 'hevc' in enc or '265' in enc:
            min_bpp = min_bpp_map['hevc_8bit']
        else:
            min_bpp = min_bpp_map['h264_8bit']
        pixels_per_sec = max(1, params['width'] * params['height']) * max(1.0, params['fps'])
        min_kbps_for_bpp = int(math.ceil(min_bpp * pixels_per_sec / 1000.0))
        computed_bitrate = max(computed_bitrate, min_kbps_for_bpp)
        if params['width'] >= 1280:
            computed_bitrate = max(computed_bitrate, 600)  # slightly higher floor for ~720p+ outputs
        params['bitrate'] = computed_bitrate
        
        # CRF for software encoding
        if accel_type == 'software':
            params['crf'] = 25  # Balanced quality for adaptive resolution
        
        # Prefer software encoding at very low bitrates for better quality-per-bit
        try:
            # Prefer software at low bitrates for higher quality-per-bit
            if params['bitrate'] < 600:  # kbps
                params['encoder'] = 'libx264'
                params['acceleration_type'] = 'software'
                params['preset'] = 'slow'
                # CRF optional; keep bitrate-targeted encode for size compliance
                if 'crf' in params:
                    params.pop('crf', None)
        except Exception:
            pass
        
        return params
    
    def _calculate_aggressive_params(self, video_info: Dict[str, Any], platform_config: Dict[str, Any], 
                                   target_size_mb: float) -> Dict[str, Any]:
        """Calculate aggressive compression parameters as last resort"""
        
        encoder, accel_type = self.hardware.get_best_encoder("h264")
        
        original_pixels = video_info['width'] * video_info['height']
        logger.info(f"Aggressive compression: original {video_info['width']}x{video_info['height']} "
                   f"({original_pixels:,} pixels), target size {target_size_mb}MB")
        
        # Much less aggressive resolution reduction for high-resolution videos
        if original_pixels > 1000000:  # High resolution (like your 1080x1440)
            # For high-res videos, be very conservative about downscaling
            scale_factor = 0.9  # Only reduce to 90% of original (was 0.8)
            logger.info(f"High-resolution video detected, using conservative scale factor: {scale_factor}")
        else:
            # For lower resolution videos, use standard aggressive scaling
            scale_factor = 0.8  # Standard aggressive scaling
            logger.info(f"Standard resolution video, using aggressive scale factor: {scale_factor}")
        
        aggressive_width = int(video_info['width'] * scale_factor)
        aggressive_height = int(video_info['height'] * scale_factor)
        
        logger.info(f"Initial aggressive scaling: factor={scale_factor}, size={aggressive_width}x{aggressive_height}")
        
        # Apply much better minimum resolution constraints to prevent extreme downscaling
        if video_info['height'] > video_info['width']:  # Vertical video
            if original_pixels > 1000000:  # High resolution
                min_width = max(720, int(video_info['width'] * 0.8))   # At least 720px wide or 80% of original
                min_height = max(960, int(video_info['height'] * 0.8)) # At least 960px tall or 80% of original
            else:  # Standard resolution
                min_width = max(640, int(video_info['width'] * 0.7))   # At least 640px wide or 70% of original
                min_height = max(480, int(video_info['height'] * 0.7)) # At least 480px tall or 70% of original
        else:  # Horizontal video
            if original_pixels > 1000000:  # High resolution
                min_width = max(960, int(video_info['width'] * 0.8))   # At least 960px wide or 80% of original
                min_height = max(720, int(video_info['height'] * 0.8)) # At least 720px tall or 80% of original
            else:  # Standard resolution
                min_width = max(640, int(video_info['width'] * 0.7))   # At least 640px wide or 70% of original
                min_height = max(480, int(video_info['height'] * 0.7)) # At least 480px tall or 70% of original
        
        aggressive_width = max(aggressive_width, min_width)
        aggressive_height = max(aggressive_height, min_height)
        
        logger.info(f"Applied minimum constraints: min {min_width}x{min_height}, "
                   f"adjusted size={aggressive_width}x{aggressive_height}")
        
        # Ensure even dimensions
        aggressive_width = aggressive_width if aggressive_width % 2 == 0 else aggressive_width - 1
        aggressive_height = aggressive_height if aggressive_height % 2 == 0 else aggressive_height - 1
        
        params = {
            'encoder': encoder,
            'acceleration_type': accel_type,
            'width': aggressive_width,
            'height': aggressive_height,
            'fps': min(video_info['fps'], 24),  # Reduce FPS
            'preset': 'veryfast',  # Fast preset for aggressive compression
        }
        
        # Very conservative bitrate
        duration = video_info['duration']
        target_bits = target_size_mb * 8 * 1024 * 1024 * 0.75  # Use only 75% of target
        params['bitrate'] = max(int(target_bits / duration / 1000), 100)
        
        # High CRF for software encoding
        if accel_type == 'software':
            params['crf'] = 35  # Very high CRF for maximum compression
        
        logger.info(f"Final aggressive params: {aggressive_width}x{aggressive_height}, "
                   f"bitrate={params['bitrate']}k, crf={params.get('crf', 'N/A')}")
        
        return params
    
    def _build_intelligent_ffmpeg_command(self, input_path: str, output_path: str, 
                                        params: Dict[str, Any]) -> List[str]:
        """Build FFmpeg command with intelligent parameters"""
        params['maxrate_multiplier'] = 1.2  # Set specific multiplier for intelligent command
        try:
            logger.info(f"Encoder selected: {params.get('encoder', 'unknown')} ({params.get('acceleration_type', 'software')})")
        except Exception:
            pass
        return FFmpegUtils.build_standard_ffmpeg_command(input_path, output_path, params)
    
    def _build_two_pass_command(self, input_path: str, output_path: str, params: Dict[str, Any], 
                              pass_num: int, log_file: str) -> List[str]:
        """Build two-pass FFmpeg command"""
        params['maxrate_multiplier'] = 1.1  # Set specific multiplier for two-pass
        try:
            logger.info(f"Encoder selected: {params.get('encoder', 'unknown')} ({params.get('acceleration_type', 'software')})")
        except Exception:
            pass
        return FFmpegUtils.build_two_pass_command(input_path, output_path, params, pass_num, log_file)
    
    def _build_adaptive_ffmpeg_command(self, input_path: str, output_path: str, 
                                     params: Dict[str, Any]) -> List[str]:
        """Build adaptive FFmpeg command with smart filtering"""
        
        # Use shared utilities for base command
        params['maxrate_multiplier'] = 1.15
        try:
            logger.info(f"Encoder selected: {params.get('encoder', 'unknown')} ({params.get('acceleration_type', 'software')})")
        except Exception:
            pass
        cmd = FFmpegUtils.build_base_ffmpeg_command(input_path, output_path, params)
        
        # Smart scaling with high-quality filter and normalized SAR
        scale_filter = f"scale={params['width']}:{params['height']}:flags=lanczos,setsar=1"
        cmd.extend(['-vf', scale_filter])
        
        # Add frame rate
        if 'fps' in params:
            cmd.extend(['-r', str(params['fps'])])
        
        # Quality settings for software encoding
        if params.get('acceleration_type') == 'software':
            if 'crf' in params:
                cmd.extend(['-crf', str(params['crf'])])
            cmd.extend(['-preset', params.get('preset', 'medium')])
        
        # Use shared utilities for remaining settings
        cmd = FFmpegUtils.add_bitrate_control(cmd, params, buffer_multiplier=1.5)
        cmd = FFmpegUtils.add_audio_settings(cmd, params)
        cmd = FFmpegUtils.add_output_optimizations(cmd, output_path)
        
        logger.debug(f"Adaptive FFmpeg command: {' '.join(cmd)}")
        return cmd
    
    def _build_aggressive_ffmpeg_command(self, input_path: str, output_path: str,
                                       params: Dict[str, Any]) -> List[str]:
        """Build aggressive compression FFmpeg command"""

        # Set aggressive parameters
        params['maxrate_multiplier'] = 1.0  # No tolerance
        params['audio_bitrate'] = 64  # Lower audio quality
        params['audio_channels'] = 1  # Mono audio

        # Use shared utilities for base command
        try:
            logger.info(f"Encoder selected: {params.get('encoder', 'unknown')} ({params.get('acceleration_type', 'software')})")
        except Exception:
            pass
        cmd = FFmpegUtils.build_base_ffmpeg_command(input_path, output_path, params)
        cmd = FFmpegUtils.add_video_settings(cmd, params)  # Already adds -crf and -preset

        # Strict bitrate control with small buffer
        cmd = FFmpegUtils.add_bitrate_control(cmd, params, buffer_multiplier=0.5)
        cmd = FFmpegUtils.add_audio_settings(cmd, params)
        cmd = FFmpegUtils.add_output_optimizations(cmd, output_path)

        logger.debug(f"Aggressive FFmpeg command: {' '.join(cmd)}")
        return cmd

    def _build_ultra_aggressive_ffmpeg_command(self, input_path: str, output_path: str, 
                                             params: Dict[str, Any]) -> List[str]:
        """Build ultra-aggressive compression FFmpeg command"""

        # Set ultra-aggressive parameters
        params['maxrate_multiplier'] = 0.8  # Very tight tolerance
        params['audio_bitrate'] = 32  # Very low audio quality
        params['audio_channels'] = 1  # Mono audio

        # Use shared utilities for base command
        try:
            logger.info(f"Encoder selected: {params.get('encoder', 'unknown')} ({params.get('acceleration_type', 'software')})")
        except Exception:
            pass
        cmd = FFmpegUtils.build_base_ffmpeg_command(input_path, output_path, params)
        cmd = FFmpegUtils.add_video_settings(cmd, params)

        # Ultra-aggressive quality settings
        if params.get('acceleration_type') == 'software':
            cmd.extend(['-crf', str(params.get('crf', 40))])
            cmd.extend(['-preset', 'ultrafast'])

        # Very strict bitrate control
        cmd = FFmpegUtils.add_bitrate_control(cmd, params, buffer_multiplier=0.3)
        cmd = FFmpegUtils.add_audio_settings(cmd, params)
        cmd = FFmpegUtils.add_output_optimizations(cmd, output_path)

        logger.debug(f"Ultra-aggressive FFmpeg command: {' '.join(cmd)}")
        return cmd

    def _build_extreme_aggressive_ffmpeg_command(self, input_path: str, output_path: str, 
                                              params: Dict[str, Any]) -> List[str]:
        """Build extreme aggressive compression FFmpeg command"""

        # Set extreme parameters
        params['maxrate_multiplier'] = 0.6  # Extremely tight tolerance
        params['audio_bitrate'] = 16  # Minimal audio quality
        params['audio_channels'] = 1  # Mono audio

        # Use shared utilities for base command
        try:
            logger.info(f"Encoder selected: {params.get('encoder', 'unknown')} ({params.get('acceleration_type', 'software')})")
        except Exception:
            pass
        cmd = FFmpegUtils.build_base_ffmpeg_command(input_path, output_path, params)
        cmd = FFmpegUtils.add_video_settings(cmd, params)

        # Extreme quality settings
        if params.get('acceleration_type') == 'software':
            cmd.extend(['-crf', str(params.get('crf', 51))])
            cmd.extend(['-preset', 'ultrafast'])

        # Extremely strict bitrate control
        cmd = FFmpegUtils.add_bitrate_control(cmd, params, buffer_multiplier=0.2)
        cmd = FFmpegUtils.add_audio_settings(cmd, params)
        cmd = FFmpegUtils.add_output_optimizations(cmd, output_path)

        logger.debug(f"Extreme aggressive FFmpeg command: {' '.join(cmd)}")
        return cmd

    
    def _is_hardware_acceleration_error(self, error_output: str, return_code: int = None) -> tuple[bool, str]:
        """Check if error is hardware acceleration related and return encoder type"""
        error_lower = error_output.lower()
        
        # Enhanced hardware error detection patterns
        hardware_error_patterns = {
            'AMD AMF': ['amf', 'h264_amf', 'hevc_amf', 'av1_amf',
                       'failed to initialize amf', 'amf encoder init failed',
                       'cannot load amfrt64.dll', 'cannot load amfrt32.dll',
                       'no amf device found', 'amf session init failed'],
            'NVIDIA NVENC': ['nvenc', 'h264_nvenc', 'hevc_nvenc', 'av1_nvenc',
                            'cannot load nvcuda.dll', 'cuda driver not found',
                            'nvenc init failed', 'no nvidia device found'],
            'Intel QuickSync': ['qsv', 'h264_qsv', 'hevc_qsv', 'mfx session',
                               'qsv init failed', 'no qsv device found'],
            'General': ['hardware', 'hwaccel', 'gpu', 'device not found',
                       'encoder not found', 'codec not supported']
        }
        
        # Check patterns
        for encoder_type, patterns in hardware_error_patterns.items():
            if any(pattern in error_lower for pattern in patterns):
                return True, encoder_type
        
        # Check error codes
        if return_code is not None:
            hardware_error_codes = [4294967274, -22, -12, -2]
            if return_code in hardware_error_codes:
                return True, "Unknown"
        
        return False, ""
    
    def _execute_ffmpeg_with_progress(self, cmd: List[str], duration: float):
        """Execute FFmpeg command with progress bar and hardware fallback"""
        
        logger.debug("Starting FFmpeg compression...")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        # Try original command first
        try:
            self._execute_ffmpeg_command(cmd, duration)
            logger.info("FFmpeg compression completed")
        except subprocess.CalledProcessError as e:
            # Check if it's a hardware acceleration error
            error_output = str(e.stderr) if e.stderr else str(e)
            logger.error(f"FFmpeg command failed with error: {error_output[:500]}...")  # Limit error output length
            
            # Use consolidated hardware error detection
            return_code = e.returncode if hasattr(e, 'returncode') else None
            is_hardware_error, encoder_type = self._is_hardware_acceleration_error(error_output, return_code)
            
            if is_hardware_error:
                logger.warning(f"{encoder_type} hardware acceleration failed, trying software fallback...")
                
                # Replace hardware encoder with software encoder
                fallback_cmd = self._create_software_fallback_command(cmd)
                if fallback_cmd:
                    logger.debug(f"Fallback command: {' '.join(fallback_cmd)}")
                    try:
                        self._execute_ffmpeg_command(fallback_cmd, duration)
                        logger.info("FFmpeg compression completed with software fallback")
                        return
                    except subprocess.CalledProcessError as fallback_error:
                        logger.error(f"Software fallback also failed: {str(fallback_error)[:200]}...")
                        pass  # Fall through to original error
                else:
                    logger.error("Failed to create software fallback command")
            
            # Re-raise original error if fallback didn't work or wasn't attempted
            raise e
    
    def _execute_ffmpeg_command(self, cmd: List[str], duration: float):
        """Execute FFmpeg command with progress bar (internal method)"""
        # Check for shutdown before starting
        if self.shutdown_requested:
            raise subprocess.CalledProcessError(1, cmd, "Shutdown requested before FFmpeg execution")
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            bufsize=1,
            encoding='utf-8',
            errors='replace'
        )
        
        # Store reference to current process for shutdown handling
        self.current_ffmpeg_process = process
        self._ffmpeg_processes.append(process)
        
        # Capture stderr output for error analysis
        stderr_output = []
        
        try:
            # Create progress bar
            with tqdm(total=100, desc="Compressing", unit="%", bar_format="{l_bar}{bar}| {n:.1f}%") as pbar:
                
                while True:
                    # Check for shutdown request
                    if self.shutdown_requested:
                        logger.info("Shutdown requested during FFmpeg execution, terminating process...")
                        self._terminate_ffmpeg_process()
                        raise subprocess.CalledProcessError(1, cmd, "Shutdown requested during execution")
                    
                    # Check if process has finished
                    if process.poll() is not None:
                        break
                    
                    # Read stderr output with timeout to allow shutdown checking
                    try:
                        output = process.stderr.readline()
                        if output == '':
                            # No more output, check if process is done
                            if process.poll() is not None:
                                break
                            continue
                        
                        if output:
                            # Store stderr output for error analysis
                            stderr_output.append(output)
                            
                            # Parse FFmpeg progress output
                            if 'time=' in output:
                                try:
                                    time_str = output.split('time=')[1].split()[0]
                                    current_seconds = self._parse_time_to_seconds(time_str)
                                    progress = min((current_seconds / duration) * 100, 100)
                                    pbar.n = progress
                                    pbar.refresh()
                                except:
                                    pass  # Ignore parsing errors
                    except Exception as e:
                        logger.debug(f"Error reading FFmpeg output: {e}")
                        break
            
            # Wait for process to complete (with timeout for shutdown checking)
            return_code = process.wait()
            
        except Exception as e:
            # Ensure process is cleaned up
            if process.poll() is None:
                self._terminate_ffmpeg_process()
            raise e
        finally:
            # Clear current process reference and remove from tracking list
            if self.current_ffmpeg_process in self._ffmpeg_processes:
                self._ffmpeg_processes.remove(self.current_ffmpeg_process)
            self.current_ffmpeg_process = None
        
        if return_code != 0:
            # Join all stderr output for error analysis
            full_stderr_output = ''.join(stderr_output)
            
            # Use consolidated hardware error detection
            is_hardware_error, encoder_type = self._is_hardware_acceleration_error(full_stderr_output, return_code)
            
            if is_hardware_error:
                # Convert unsigned error code to signed for comparison (for logging)
                signed_return_code = return_code if return_code < 2147483648 else return_code - 4294967296
                
                logger.warning(f"{encoder_type} encoder failed with error code {return_code} (signed: {signed_return_code}), attempting software fallback...")
                logger.debug(f"{encoder_type} error details: {full_stderr_output[-500:]}")  # Log last 500 chars of stderr
                
                # Create software fallback command
                fallback_cmd = self._create_software_fallback_command(cmd)
                if fallback_cmd:
                    logger.info(f"Created software fallback command for {encoder_type}")
                    logger.info("Attempting software fallback...")
                    # Try the fallback command
                    try:
                        return self._execute_ffmpeg_command(fallback_cmd, duration)
                    except subprocess.CalledProcessError as fallback_error:
                        logger.error(f"Software fallback also failed: {fallback_error}")
                        raise fallback_error
                else:
                    logger.error("Failed to create software fallback command")
            
            raise subprocess.CalledProcessError(return_code, cmd, full_stderr_output)
    
    def _create_software_fallback_command(self, cmd: List[str]) -> Optional[List[str]]:
        """Create a software fallback version of an FFmpeg command"""
        try:
            fallback_cmd = cmd.copy()
            
            # Replace hardware encoders with software equivalents
            encoder_replacements = {
                'h264_qsv': 'libx264',
                'h264_nvenc': 'libx264', 
                'h264_amf': 'libx264',
                'hevc_qsv': 'libx265',
                'hevc_nvenc': 'libx265',
                'hevc_amf': 'libx265',
                'av1_amf': 'libsvtav1'  # AV1 software encoder
            }
            
            encoder_replaced = False
            for i, arg in enumerate(fallback_cmd):
                if arg in encoder_replacements:
                    old_encoder = arg
                    new_encoder = encoder_replacements[arg]
                    fallback_cmd[i] = new_encoder
                    encoder_replaced = True
                    logger.info(f"Replaced {old_encoder} with {new_encoder} for software fallback")
            
            if not encoder_replaced:
                logger.warning("No hardware encoder found to replace in fallback command")
            
            # Remove hardware-specific options that don't work with software encoders
            hardware_options_to_remove = [
                '-hwaccel', 'cuda', '-hwaccel', 'auto', '-hwaccel', 'qsv', '-hwaccel', 'dxva2',
                # AMD AMF specific options
                '-usage', 'transcoding', '-usage', 'lowlatency', '-usage', 'ultralowlatency',
                '-quality', 'speed', '-quality', 'balanced', '-quality', 'quality',
                '-rc', 'cbr', '-rc', 'vbr', '-rc', 'cqp',
                '-enforce_hrd', '-filler_data', '-frame_skipping', '-vbaq', '-preanalysis',
                # NVIDIA NVENC specific options
                '-preset', 'p1', '-preset', 'p2', '-preset', 'p3', '-preset', 'p4', 
                '-preset', 'p5', '-preset', 'p6', '-preset', 'p7',
                '-tune', 'hq', '-tune', 'll', '-tune', 'ull',
                # Intel QSV specific options
                '-global_quality', '-look_ahead', '-look_ahead_depth'
            ]
            
            # Remove hardware options and their values
            i = 0
            while i < len(fallback_cmd):
                if fallback_cmd[i] in hardware_options_to_remove:
                    option = fallback_cmd[i]
                    fallback_cmd.pop(i)  # Remove the option
                    # Check if the next argument is a value (not starting with -)
                    if i < len(fallback_cmd) and not fallback_cmd[i].startswith('-'):
                        value = fallback_cmd[i]
                        fallback_cmd.pop(i)  # Remove the value
                        logger.debug(f"Removed hardware option: {option} {value}")
                    else:
                        logger.debug(f"Removed hardware option: {option}")
                else:
                    i += 1
            
            # Convert hardware-specific quality settings to software equivalents
            for i, arg in enumerate(fallback_cmd):
                # Convert QP (quantization parameter) to CRF for software encoders
                if arg == '-qp' and i + 1 < len(fallback_cmd):
                    fallback_cmd[i] = '-crf'
                    # QP and CRF have similar ranges, so keep the value
                    logger.debug(f"Converted -qp to -crf for software encoding")
                
                # Convert NVENC CQ to CRF
                elif arg == '-cq' and i + 1 < len(fallback_cmd):
                    fallback_cmd[i] = '-crf'
                    logger.debug(f"Converted -cq to -crf for software encoding")
            
            # Add software encoder optimizations
            if 'libx264' in fallback_cmd:
                # Add reasonable preset if not present
                if '-preset' not in fallback_cmd:
                    fallback_cmd.extend(['-preset', 'medium'])
                    logger.debug("Added -preset medium for libx264")
            
            elif 'libx265' in fallback_cmd:
                # Add reasonable preset if not present
                if '-preset' not in fallback_cmd:
                    fallback_cmd.extend(['-preset', 'medium'])
                    logger.debug("Added -preset medium for libx265")
            
            return fallback_cmd
            
        except Exception as e:
            logger.warning(f"Failed to create software fallback command: {e}")
            return None
    
    def _parse_time_to_seconds(self, time_str: str) -> float:
        """Parse FFmpeg time format (HH:MM:SS.mmm) to seconds"""
        try:
            parts = time_str.split(':')
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
        except:
            return 0.0
    
    def _get_compression_results(self, input_path: str, output_path: str, 
                               video_info: Dict[str, Any], method: str) -> Dict[str, Any]:
        """Generate compression results summary"""
        
        original_size = os.path.getsize(input_path)
        compressed_size = os.path.getsize(output_path)
        
        compression_ratio = ((original_size - compressed_size) / original_size) * 100
        compressed_size_mb = compressed_size / (1024 * 1024)
        
        # Log detailed file specifications
        try:
            from .ffmpeg_utils import FFmpegUtils
            specs = FFmpegUtils.get_detailed_file_specifications(output_path)
            specs_log = FFmpegUtils.format_file_specifications_for_logging(specs)
            logger.info(f"Video compression completed successfully - {specs_log}")
        except Exception as e:
            logger.warning(f"Could not log detailed video specifications: {e}")
        
        return {
            'success': True,  # Add success flag for automated workflow
            'input_file': input_path,
            'output_file': output_path,
            'method': method,
            'original_size_mb': original_size / (1024 * 1024),
            'compressed_size_mb': compressed_size_mb,
            'size_mb': compressed_size_mb,  # Add alias for automated workflow
            'compression_ratio': compression_ratio,
            'space_saved_mb': (original_size - compressed_size) / (1024 * 1024),
            'video_info': video_info,
            'encoder_used': getattr(self, '_last_encoder_used', 'unknown')
        }
    
    def _get_error_results(self, input_path: str, output_path: str, error_message: str) -> Dict[str, Any]:
        """Generate error results for failed compression"""
        return {
            'success': False,
            'input_file': input_path,
            'output_file': output_path,
            'error': error_message,
            'method': 'failed',
            'original_size_mb': os.path.getsize(input_path) / (1024 * 1024) if os.path.exists(input_path) else 0,
            'compressed_size_mb': 0,
            'size_mb': 0,
            'compression_ratio': 0,
            'space_saved_mb': 0,
            'encoder_used': 'none'
        }

    def _calculate_compression_params_with_quality(self, video_info: Dict[str, Any], 
                                                 platform_config: Dict[str, Any], 
                                                 target_size_mb: float,
                                                 quality_params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate compression parameters with specific quality level"""
        
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
        
        # Apply quality-specific settings
        if accel_type == 'software':
            params['crf'] = quality_params.get('crf', 23)
            params['preset'] = quality_params.get('preset', 'medium')
        
        # Calculate bitrate based on quality priority
        if quality_params.get('priority') == 'speed':
            # Faster processing, less optimal bitrate
            params['bitrate'] = int(self._calculate_content_aware_bitrate(video_info, target_size_mb) * 1.1)
        elif quality_params.get('priority') == 'quality':
            # Quality priority, more conservative bitrate
            params['bitrate'] = int(self._calculate_content_aware_bitrate(video_info, target_size_mb) * 0.9)
        else:
            # Balanced
            params['bitrate'] = self._calculate_content_aware_bitrate(video_info, target_size_mb)
        
        # Platform-specific adjustments
        if platform_config:
            params.update(self._apply_platform_constraints(params, platform_config))
        
        return params
    
    def batch_compress_videos(self, video_list: List[str], output_dir: str, 
                            platform: str = None, max_size_mb: float = None) -> Dict[str, Any]:
        """
        Batch compress multiple videos with intelligent processing
        """
        
        logger.info(f"Starting batch compression of {len(video_list)} videos")
        
        # Create processing function for batch
        def process_single_video(file_info):
            input_path = file_info['input_path']
            filename = os.path.basename(input_path)
            name, ext = os.path.splitext(filename)
            output_path = os.path.join(output_dir, f"{name}_compressed{ext}")
            
            try:
                result = self.compress_video(
                    input_path=input_path,
                    output_path=output_path,
                    platform=platform,
                    max_size_mb=max_size_mb,
                    use_advanced_optimization=True
                )
                
                return {
                    'success': True,
                    'input_path': input_path,
                    'output_path': output_path,
                    'result': result
                }
                
            except Exception as e:
                logger.error(f"Failed to compress {input_path}: {e}")
                return {
                    'success': False,
                    'input_path': input_path,
                    'error': str(e)
                }
        
        # Use intelligent batch processing
        batch_results = self.performance_enhancer.intelligent_batch_processing(
            video_list, process_single_video
        )
        
        # Compile final results
        successful = [r for r in batch_results if r.get('success', False)]
        failed = [r for r in batch_results if not r.get('success', False)]
        
        return {
            'total_processed': len(video_list),
            'successful': len(successful),
            'failed': len(failed),
            'successful_results': successful,
            'failed_results': failed,
            'performance_stats': self.performance_enhancer.get_performance_stats()
        }
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics"""
        
        stats = {
            'performance_stats': self.performance_enhancer.get_performance_stats(),
            'system_optimizations': self.system_optimizations,
            'hardware_info': {
                'best_encoder': self.hardware.get_best_encoder("h264"),
                'has_hardware_acceleration': self.hardware.has_hardware_acceleration(),
                'gpu_info': [gpu for gpu in self.hardware.gpu_info if gpu],
                'system_report': self.hardware.get_system_report()
            }
        }
        
        return stats

    def request_shutdown(self):
        """Request graceful shutdown of the compressor"""
        with self._shutdown_lock:
            if not self.shutdown_requested:  # Only log and terminate if not already shutting down
                logger.info("Shutdown requested for video compressor")
                self.shutdown_requested = True
                logger.info("Terminating all FFmpeg processes...")
                self._terminate_ffmpeg_process()
    
    def _terminate_ffmpeg_process(self):
        """Terminate all tracked FFmpeg processes gracefully"""
        # Terminate current process
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

                logger.info("Current FFmpeg process terminated successfully")
            except Exception as e:
                logger.error(f"Error terminating current FFmpeg process: {e}")
            finally:
                if self.current_ffmpeg_process in self._ffmpeg_processes:
                    self._ffmpeg_processes.remove(self.current_ffmpeg_process)
                self.current_ffmpeg_process = None

        # Terminate any other tracked processes
        for process in list(self._ffmpeg_processes):
            if process and process.poll() is None:
                try:
                    logger.info("Terminating additional FFmpeg process...")
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        logger.warning("Additional FFmpeg process did not terminate gracefully, forcing kill...")
                        process.kill()
                        process.wait()
                    logger.info("Additional FFmpeg process terminated successfully")
                except Exception as e:
                    logger.error(f"Error terminating additional FFmpeg process: {e}")
                finally:
                    if process in self._ffmpeg_processes:
                        self._ffmpeg_processes.remove(process)

# Keep all the existing methods from the original VideoCompressor class
# by creating an alias for backward compatibility
VideoCompressor = DynamicVideoCompressor 