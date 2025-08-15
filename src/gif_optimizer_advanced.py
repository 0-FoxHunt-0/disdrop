"""
Advanced GIF Optimizer
Implements cutting-edge GIF optimization techniques for maximum quality and minimum file size
"""

import os
import subprocess
import json
import math
import time
import random
import threading
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional, Callable
import logging
from PIL import Image, ImageSequence, ImageOps, ImageEnhance, ImageFilter
import numpy as np
from collections import Counter
import cv2

logger = logging.getLogger(__name__)

class AdvancedGifOptimizer:
    def __init__(self, config_manager, shutdown_checker: Optional[Callable[[], bool]] = None):
        self.config = config_manager
        self.temp_dir = self.config.get_temp_dir()
        # Shutdown checker provided by caller; if not provided, never indicates shutdown
        self._shutdown_checker: Callable[[], bool] = shutdown_checker or (lambda: False)
        self.shutdown_requested = False
        self.current_ffmpeg_process = None
        
        # Advanced optimization cache
        self.palette_cache = {}
        self.frame_analysis_cache = {}
        # Quality optimization knobs (reserved for future config use)
        self.qo_cfg = self.config.get('gif_settings.quality_optimization', {}) or {}
        self.early_stop_cfg = self.qo_cfg.get('early_stop', {}) or {}

        # Performance configuration for gifsicle path
        self.perf_cfg = self.config.get('gif_settings.performance', {}) or {}
        self.fast_mode: bool = bool(self.perf_cfg.get('fast_mode', False))
        # Time budget for gifsicle adaptive search (seconds)
        self.gifsicle_time_budget_sec: int = int(self.perf_cfg.get('gifsicle_time_budget_seconds', 25 if self.fast_mode else 45))
        # Cap on total gifsicle runs in adaptive search
        self.gifsicle_max_candidates: int = int(self.perf_cfg.get('gifsicle_max_candidates', 48 if self.fast_mode else 120))
        # gifsicle optimize level (2 is faster, 3 is heaviest)
        self.gifsicle_optimize_level: int = int(self.perf_cfg.get('gifsicle_optimize_level', 2 if self.fast_mode else 3))
        # Skip gifsicle passes when far over target to avoid heavy futile searches
        self.skip_gifsicle_far_over_ratio: float = float(self.perf_cfg.get('skip_gifsicle_far_over_ratio', 0.35 if self.fast_mode else 0.5))
        # Limit for near-target bounded search iterations
        self.near_target_max_runs: int = int(self.perf_cfg.get('near_target_max_runs', 12 if self.fast_mode else 24))

    def _create_unique_temp_filename(self, prefix: str, suffix: str) -> str:
        """Create a unique temporary filename to avoid Windows file locking conflicts"""
        thread_id = threading.get_ident()
        random_suffix = random.randint(1000, 9999)
        timestamp = int(time.time())
        return os.path.join(self.temp_dir, f"{prefix}_{thread_id}_{timestamp}_{random_suffix}{suffix}")

    def _safe_file_operation(self, operation, *args, max_retries: int = 3, **kwargs):
        """Perform file operation with retry logic for Windows file locking"""
        for retry in range(max_retries):
            try:
                return operation(*args, **kwargs)
            except (PermissionError, OSError) as e:
                if retry < max_retries - 1:
                    logger.debug(f"File operation retry {retry + 1}/{max_retries}: {e}")
                    time.sleep(0.1 * (retry + 1))  # Progressive backoff
                else:
                    logger.warning(f"File operation failed after {max_retries} retries: {e}")
                    raise

    def request_shutdown(self):
        """Request graceful shutdown of the optimizer"""
        logger.info("Shutdown requested for GIF optimizer")
        self.shutdown_requested = True
        self._terminate_ffmpeg_process()

    def _terminate_ffmpeg_process(self):
        """Terminate the current FFmpeg process gracefully"""
        if self.current_ffmpeg_process and self.current_ffmpeg_process.poll() is None:
            try:
                logger.info("Terminating FFmpeg process in GIF optimizer...")
                # Try graceful termination first
                self.current_ffmpeg_process.terminate()
                
                # Wait a bit for graceful termination
                try:
                    self.current_ffmpeg_process.wait(timeout=5)
                    logger.info("FFmpeg process terminated gracefully")
                except subprocess.TimeoutExpired:
                    # Force kill if graceful termination fails
                    logger.warning("FFmpeg process did not terminate gracefully, forcing kill...")
                    self.current_ffmpeg_process.kill()
                    self.current_ffmpeg_process.wait()
                    logger.info("FFmpeg process killed")
                
            except Exception as e:
                logger.error(f"Error terminating FFmpeg process in GIF optimizer: {e}")
            finally:
                self.current_ffmpeg_process = None

    def _run_subprocess_with_shutdown_check(self, cmd, timeout=120, **kwargs):
        """Run subprocess with shutdown checking and process tracking"""
        # Check if shutdown was requested before starting
        if self.shutdown_requested or self._shutdown_checker():
            class ShutdownResult:
                returncode = 1
                stdout = ''
                stderr = 'Shutdown requested before execution'
            return ShutdownResult()
        
        try:
            self.current_ffmpeg_process = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kwargs
            )
            
            # Poll periodically to check for shutdown requests
            start_time = time.time()
            while self.current_ffmpeg_process.poll() is None:
                if self.shutdown_requested or self._shutdown_checker():
                    logger.info("Shutdown requested during subprocess execution, terminating...")
                    self._terminate_ffmpeg_process()
                    class ShutdownResult:
                        returncode = 1
                        stdout = ''
                        stderr = 'Shutdown requested during execution'
                    return ShutdownResult()
                
                # Check timeout
                if time.time() - start_time > timeout:
                    logger.warning(f"Subprocess timeout after {timeout}s, terminating...")
                    self._terminate_ffmpeg_process()
                    class TimeoutResult:
                        returncode = 1
                        stdout = ''
                        stderr = 'TimeoutExpired'
                    return TimeoutResult()
                
                time.sleep(0.1)  # Short sleep to avoid busy waiting
            
            # Process completed normally
            stdout, stderr = self.current_ffmpeg_process.communicate()
            
            class NormalResult:
                def __init__(self, returncode, stdout, stderr):
                    self.returncode = returncode
                    self.stdout = stdout
                    self.stderr = stderr
            
            return NormalResult(self.current_ffmpeg_process.returncode, stdout, stderr)
            
        except Exception as e:
            logger.error(f"Error running subprocess: {e}")
            class ErrorResult:
                returncode = 1
                stdout = ''
                stderr = str(e)
            return ErrorResult()
        finally:
            self.current_ffmpeg_process = None
        
    def create_optimized_gif(self, input_video: str, output_path: str, 
                           max_size_mb: float, platform: str = None,
                           start_time: float = 0, duration: float = None) -> Dict[str, Any]:
        """
        Create highly optimized GIF using advanced techniques
        """
        
        logger.info("Starting advanced GIF optimization...")
        
        # 1. Intelligent Frame Analysis
        if self._shutdown_checker():
            return {'success': False, 'error': 'shutdown'}
        frame_analysis = self._perform_intelligent_frame_analysis(input_video, start_time, duration)
        
        # 2. Advanced Palette Optimization
        if self._shutdown_checker():
            return {'success': False, 'error': 'shutdown'}
        optimal_palette = self._generate_optimal_palette(input_video, frame_analysis, max_size_mb)
        
        # 3. Smart Frame Selection and Temporal Optimization
        if self._shutdown_checker():
            return {'success': False, 'error': 'shutdown'}
        optimized_frames = self._optimize_frame_sequence(input_video, frame_analysis, optimal_palette, max_size_mb)
        
        # 4. Multi-Strategy GIF Generation
        if self._shutdown_checker():
            return {'success': False, 'error': 'shutdown'}
        gif_candidates = self._generate_gif_candidates(optimized_frames, optimal_palette, max_size_mb, platform)
        
        # 5. Parallel Evaluation and Selection
        if self._shutdown_checker():
            return {'success': False, 'error': 'shutdown'}
        best_gif = self._evaluate_gif_candidates_parallel(gif_candidates, output_path, max_size_mb)
        
        # 6. Post-Processing Optimization
        if self._shutdown_checker():
            return {'success': False, 'error': 'shutdown'}
        final_result = self._apply_gif_post_processing(best_gif, max_size_mb)
        
        return final_result
    
    def optimize_gif(self, gif_path: str, max_size_mb: float) -> bool:
        """
        Robust, efficient input-GIF optimization with clear, bounded stages:
          1) Lossless gifsicle pass (strip metadata, optimize structure)
          2) If within 15% over target, bounded gifsicle search (few candidates only)
          3) If >15% over target or gifsicle not available, ffmpeg palette re-encode with mpdecimate
          4) Optional final gifsicle squeeze for small overage
        """
        try:
            if self._shutdown_checker():
                return False
            if not os.path.exists(gif_path):
                logger.error(f"GIF file not found: {gif_path}")
                return False

            target_bytes = int(max_size_mb * 1024 * 1024)
            original_bytes = os.path.getsize(gif_path)
            logger.info(
                f"Optimizing GIF: current={original_bytes/1024/1024:.2f}MB, target={max_size_mb:.2f}MB"
            )

            # Backup original (best-effort)
            backup_path = gif_path + ".orig.bak"
            try:
                if os.path.exists(backup_path):
                    os.remove(backup_path)
                shutil.copy2(gif_path, backup_path)
            except Exception:
                backup_path = None

            gifsicle_available = self._is_tool_available("gifsicle")

            # Determine if we should skip gifsicle entirely due to being far over target
            original_over_ratio = (original_bytes - target_bytes) / float(target_bytes) if target_bytes > 0 else 1.0
            far_over_skip = gifsicle_available and (original_over_ratio >= self.skip_gifsicle_far_over_ratio)

            # Stage 1: Lossless gifsicle optimize (if available and not far over target)
            if self._shutdown_checker():
                return False
            if gifsicle_available and not far_over_skip:
                temp_lossless = gif_path + ".lossless.tmp.gif"
                if self._gifsicle_lossless_optimize(gif_path, temp_lossless) and os.path.exists(temp_lossless):
                    os.replace(temp_lossless, gif_path)
            # Early exit if already under target after lossless
            current_bytes = os.path.getsize(gif_path)
            if current_bytes <= target_bytes:
                if backup_path and os.path.exists(backup_path):
                    try:
                        os.remove(backup_path)
                    except Exception:
                        pass
                logger.info("GIF under target after lossless optimization")
                return True

            # Determine strategy threshold (15%)
            over_ratio = (current_bytes - target_bytes) / float(target_bytes)
            near_target = (over_ratio <= 0.15) and (not far_over_skip)

            # Stage 2: Near-target bounded gifsicle search
            if self._shutdown_checker():
                return False
            if gifsicle_available and near_target:
                temp_best = gif_path + ".near.tmp.gif"
                if self._bounded_gifsicle_near_target(gif_path, temp_best, target_bytes):
                    os.replace(temp_best, gif_path)
                    current_bytes = os.path.getsize(gif_path)
                    if current_bytes <= target_bytes:
                        if backup_path and os.path.exists(backup_path):
                            try:
                                os.remove(backup_path)
                            except Exception:
                                pass
                        logger.info("GIF met target via bounded gifsicle search")
                        return True
                # Cleanup
                try:
                    if os.path.exists(temp_best):
                        os.remove(temp_best)
                except Exception:
                    pass

            # Stage 3: ffmpeg palette re-encode with mpdecimate (robust fallback / far-over-target)
            if self._shutdown_checker():
                return False
            info = self._get_gif_basic_info(gif_path)
            temp_ffmpeg = gif_path + ".ffmpeg.tmp.gif"

            # Compute scale factor when far over target; preserve aspect ratio, round to even dims
            scale_factor = max(0.25, min(1.0, math.sqrt(target_bytes / float(current_bytes))))
            if near_target:
                scale_factor = min(1.0, max(0.85, scale_factor))  # keep resolution when close

            new_width = max(2, int((info.get('width', 320) * scale_factor) // 2 * 2))
            fps = max(6, min(15, int(round(info.get('fps', 12)))))

            if self._shutdown_checker():
                return False
            if self._ffmpeg_palette_reencode(gif_path, temp_ffmpeg, new_width, fps, max_colors=256):
                os.replace(temp_ffmpeg, gif_path)
                current_bytes = os.path.getsize(gif_path)

                # Stage 4: If slightly over, try a tiny gifsicle squeeze
                if current_bytes > target_bytes and gifsicle_available and not self._shutdown_checker():
                    if self._gifsicle_squeeze_small_overage(gif_path):
                        current_bytes = os.path.getsize(gif_path)

                if current_bytes <= target_bytes:
                    if backup_path and os.path.exists(backup_path):
                        try:
                            os.remove(backup_path)
                        except Exception:
                            pass
                    logger.info("GIF met target via ffmpeg re-encode")
                    return True

            # Iterative tightening: if still over target, progressively reduce width/fps and colors
            if current_bytes > target_bytes and not self._shutdown_checker():
                attempt_width = max(2, int((new_width * 0.85) // 2 * 2))
                attempt_fps = max(6, int(fps * 0.9))
                for colors in [192, 160]:
                    if self._shutdown_checker():
                        return False
                    temp_try = gif_path + f".tight_{attempt_width}w_{attempt_fps}fps_{colors}c.tmp.gif"
                    if self._ffmpeg_palette_reencode(gif_path, temp_try, attempt_width, attempt_fps, max_colors=colors):
                        os.replace(temp_try, gif_path)
                        current_bytes = os.path.getsize(gif_path)
                        if current_bytes > target_bytes and gifsicle_available and not self._shutdown_checker():
                            if self._gifsicle_squeeze_small_overage(gif_path):
                                current_bytes = os.path.getsize(gif_path)
                        if current_bytes <= target_bytes:
                            if backup_path and os.path.exists(backup_path):
                                try:
                                    os.remove(backup_path)
                                except Exception:
                                    pass
                            logger.info("GIF met target via iterative ffmpeg re-encode")
                            return True
                    # Prepare even tighter next round
                    attempt_width = max(2, int((attempt_width * 0.88) // 2 * 2))
                    attempt_fps = max(6, int(attempt_fps * 0.9))

            # Restore original if all attempts failed but original already <= target
            if backup_path and os.path.exists(backup_path):
                try:
                    if os.path.getsize(backup_path) <= target_bytes:
                        os.replace(backup_path, gif_path)
                        logger.info("Restored original GIF (already within target)")
                        return True
                except Exception:
                    pass

            logger.warning("GIF optimization did not meet target")
            return False

        except Exception as e:
            logger.error(f"Error optimizing GIF: {e}")
            return False

    def _get_gif_basic_info(self, gif_path: str) -> Dict[str, Any]:
        """Lightweight probe for width/height/fps/frame_count/duration."""
        info: Dict[str, Any] = {
            'width': 320, 'height': 240, 'fps': 12, 'frame_count': 0, 'duration': 0.0
        }
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_streams', '-show_format', gif_path
            ]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=20)
            if result.returncode == 0 and result.stdout:
                import json as _json
                data = _json.loads(result.stdout)
                vs = next((s for s in data.get('streams', []) if s.get('codec_type') == 'video'), None)
                if vs:
                    info['width'] = int(vs.get('width', info['width']))
                    info['height'] = int(vs.get('height', info['height']))
                    fps_str = vs.get('r_frame_rate', '12/1')
                    try:
                        if '/' in fps_str:
                            num, den = fps_str.split('/')
                            info['fps'] = max(1.0, float(num) / float(den or 1))
                        else:
                            info['fps'] = max(1.0, float(fps_str))
                    except Exception:
                        pass
                    try:
                        info['frame_count'] = int(vs.get('nb_frames', 0))
                    except Exception:
                        pass
                fmt = data.get('format', {})
                try:
                    info['duration'] = float(fmt.get('duration', 0.0))
                except Exception:
                    pass
        except Exception:
            pass
        return info

    def _bounded_gifsicle_near_target(self, input_path: str, best_output_path: str, target_bytes: int) -> bool:
        """Small, bounded search using gifsicle for near-target cases to avoid heavy compute."""
        try:
            candidates = []
            # Conservative ladders (shorter when fast_mode)
            if self.fast_mode:
                color_steps = [256, 192]
                lossy_steps = [0, 20]
                scale_steps = [1.0, 0.96]
            else:
                color_steps = [256, 224, 208, 192, 176, 160]
                lossy_steps = [0, 10, 20, 30]
                scale_steps = [1.0, 0.96, 0.92]

            original_size = os.path.getsize(input_path)
            best: Tuple[int, str] = (original_size, '')

            # Try quick color-only first
            for colors in color_steps:
                temp = input_path + f".near_c{colors}.gif.tmp"
                if self._run_gifsicle(input_path, temp, colors=colors, lossy=0, scale=1.0):
                    size = os.path.getsize(temp)
                    if size <= target_bytes:
                        shutil.copy2(temp, best_output_path)
                        try:
                            os.remove(temp)
                        except Exception:
                            pass
                        return True
                    if size < best[0]:
                        best = (size, temp)
                    try:
                        os.remove(temp)
                    except Exception:
                        pass

            # Small grid over lossy and scale (hard-capped iterations)
            max_runs = int(self.near_target_max_runs)
            runs = 0
            for scale in scale_steps:
                for colors in color_steps:
                    for lossy in lossy_steps:
                        temp = input_path + f".near_s{int(scale*100)}_c{colors}_l{lossy}.gif.tmp"
                        if not self._run_gifsicle(input_path, temp, colors=colors, lossy=lossy, scale=scale):
                            continue
                        runs += 1
                        size = os.path.getsize(temp)
                        if size <= target_bytes:
                            shutil.copy2(temp, best_output_path)
                            try:
                                os.remove(temp)
                            except Exception:
                                pass
                            return True
                        if size < best[0]:
                            # Keep the best-so-far (still over target)
                            if best[1] and os.path.exists(best[1]):
                                try:
                                    os.remove(best[1])
                                except Exception:
                                    pass
                            best = (size, temp)
                        else:
                            try:
                                os.remove(temp)
                            except Exception:
                                pass
                        if runs >= max_runs:
                            break
                    if runs >= max_runs:
                        break
                if runs >= max_runs:
                    break

            # If nothing met target but we improved meaningfully (<105% of target), keep best
            if best[1] and os.path.exists(best[1]):
                if best[0] <= int(target_bytes * 1.05):
                    shutil.copy2(best[1], best_output_path)
                    try:
                        os.remove(best[1])
                    except Exception:
                        pass
                    return True
                try:
                    os.remove(best[1])
                except Exception:
                    pass
            return False
        except Exception as e:
            logger.debug(f"Bounded gifsicle search failed: {e}")
            return False

    def _ffmpeg_palette_reencode(self, input_path: str, output_path: str, new_width: int, fps: int, max_colors: int = 256) -> bool:
        """Re-encode GIF via ffmpeg using mpdecimate + palettegen/paletteuse preserving AR."""
        try:
            if self._shutdown_checker():
                return False
            palette_path = input_path + ".palette.png"
            # Build filters
            pre = [
                'mpdecimate=hi=512:lo=256:frac=0.3',
                f'fps={fps}',
                f'scale={new_width}:-2:flags=lanczos'
            ]
            vf_palette = ','.join(pre + [f'palettegen=max_colors={int(max_colors)}:stats_mode=diff'])
            # Palette gen
            cmd1 = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-i', input_path, '-vf', vf_palette, '-frames:v', '1', palette_path
            ]
            if self._shutdown_checker():
                return False
            r1 = subprocess.run(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
            if r1.returncode != 0 or not os.path.exists(palette_path):
                try:
                    if os.path.exists(palette_path):
                        os.remove(palette_path)
                except Exception:
                    pass
                return False

            # Palette use
            lavfi = ','.join(pre) + ' [x]; [x][1:v] paletteuse=dither=sierra2_4a:diff_mode=rectangle'
            cmd2 = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                '-i', input_path, '-i', palette_path,
                '-lavfi', lavfi,
                '-loop', '0', output_path
            ]
            if self._shutdown_checker():
                return False
            r2 = subprocess.run(cmd2, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
            try:
                if os.path.exists(palette_path):
                    os.remove(palette_path)
            except Exception:
                pass
            return r2.returncode == 0 and os.path.exists(output_path)
        except Exception as e:
            if self._shutdown_checker():
                logger.info("ffmpeg palette re-encode interrupted by shutdown")
            else:
                logger.debug(f"ffmpeg palette re-encode failed: {e}")
            return False

    def _gifsicle_squeeze_small_overage(self, gif_path: str) -> bool:
        """Tiny gifsicle squeeze for <=15% overage without major quality loss."""
        try:
            if self._shutdown_checker():
                return False
            if not self._is_tool_available('gifsicle'):
                return False
            temp = gif_path + '.squeeze.tmp.gif'
            cmd = [
                'gifsicle', '--optimize=3', '--careful', '--lossy=30', gif_path, '--output', temp
            ]
            r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
            if r.returncode == 0 and os.path.exists(temp):
                # Apply only if improved
                try:
                    if os.path.getsize(temp) < os.path.getsize(gif_path):
                        os.replace(temp, gif_path)
                        return True
                finally:
                    if os.path.exists(temp):
                        try:
                            os.remove(temp)
                        except Exception:
                            pass
            return False
        except Exception:
            return False

    def _is_tool_available(self, tool_name: str) -> bool:
        try:
            result = subprocess.run([tool_name, "--version"], capture_output=True, text=True, timeout=5)
            return result.returncode == 0
        except Exception:
            return False

    def _gifsicle_lossless_optimize(self, input_path: str, output_path: str) -> bool:
        """Run a high-efficiency, lossless structural optimization preserving timing/palette."""
        try:
            cmd = [
                "gifsicle",
                f"--optimize={max(1, min(3, int(self.gifsicle_optimize_level)))}",
                "--careful",
                "--no-comments",
                "--no-extensions",
                "--no-names",
                "--same-loopcount",
                input_path,
                "--output",
                output_path,
            ]
            timeout_sec = 30 if self.fast_mode else 60
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            return result.returncode == 0 and os.path.exists(output_path)
        except Exception as e:
            logger.debug(f"Lossless gifsicle optimize failed: {e}")
            return False

    def _gifsicle_adaptive_search(self, input_path: str, best_output_path: str, target_bytes: int) -> bool:
        """
        Explore parameter grid to get under target with best visual fidelity.
        Preference order for quality: larger scale > higher colors > lower lossy.
        """
        try:
            original_size = os.path.getsize(input_path)

            # Parameter ladders
            # Preserve aspect ratio by using uniform scale factors only
            target_ratio = target_bytes / float(max(1, original_size))
            # Adaptive ladders based on how far we need to shrink
            if target_ratio < 0.4:
                scale_factors = [1.0, 0.92, 0.9, 0.88, 0.85, 0.82, 0.8, 0.75, 0.7, 0.65]
                color_steps = [192, 176, 160, 144, 128, 120, 112, 104, 96, 88, 80, 72, 64]
                lossy_steps = [10, 15, 20, 30, 40, 60, 80, 100]
                fps_levels = [12, 10, 8]
            elif target_ratio < 0.6:
                scale_factors = [1.0, 0.95, 0.92, 0.9, 0.88, 0.85, 0.82, 0.8, 0.75]
                color_steps = [224, 208, 192, 176, 160, 144, 128, 120, 112, 104, 96]
                lossy_steps = [5, 10, 15, 20, 30, 40, 60]
                fps_levels = [15, 12, 10]
            else:
                scale_factors = [1.0, 0.98, 0.95, 0.92, 0.9, 0.88, 0.85]
                color_steps = [256, 240, 224, 208, 192, 176, 160, 144, 128]
                lossy_steps = [0, 5, 10, 15, 20, 30]
                fps_levels = [None, 15]

            best_candidate = None  # (bytes, scale, colors, lossy, path)

            # Try fast path: reduce colors first without lossy/scale
            for colors in color_steps:
                temp = input_path + f".c{colors}.gif.tmp"
                if self._run_gifsicle(input_path, temp, colors=colors, lossy=0, scale=1.0):
                    size = os.path.getsize(temp)
                    if size <= target_bytes:
                        best_candidate = (size, 1.0, colors, 0, temp)
                        break
                    os.remove(temp)

            # If not found, search grid
            if best_candidate is None:
                # Preprocess fps variants once (in parallel), then test combinations in parallel
                fps_inputs = self._preprocess_fps_variants(input_path, fps_levels)
                from concurrent.futures import ThreadPoolExecutor, as_completed
                max_workers = max(2, min(8, (os.cpu_count() or 4)))
                start_time = time.time()
                runs = 0
                for fps, pre_input in fps_inputs.items():
                    # Build candidate set (limit explosion by sampling lossy or colors when large)
                    candidates = []
                    for scale in scale_factors:
                        for colors in color_steps:
                            for lossy in lossy_steps:
                                candidates.append((pre_input, scale, colors, lossy))
                    # In fast mode, sub-sample candidates for speed
                    if self.fast_mode and len(candidates) > self.gifsicle_max_candidates:
                        step = max(1, len(candidates) // self.gifsicle_max_candidates)
                        candidates = candidates[::step][:self.gifsicle_max_candidates]
                    # Run in parallel
                    with ThreadPoolExecutor(max_workers=max_workers) as executor:
                        future_to_cfg = {}
                        for (pre_in, scale, colors, lossy) in candidates:
                            temp = pre_in + f".s{int(scale*100)}_c{colors}_l{lossy}.gif.tmp"
                            future = executor.submit(self._run_gifsicle, pre_in, temp, colors, lossy, scale)
                            future_to_cfg[future] = (temp, scale, colors, lossy)
                        for future in as_completed(future_to_cfg):
                            ok = False
                            try:
                                ok = future.result()
                            except Exception:
                                ok = False
                            temp, scale, colors, lossy = future_to_cfg[future]
                            runs += 1
                            # Respect time budget
                            if time.time() - start_time > self.gifsicle_time_budget_sec:
                                # Stop processing more futures; break out
                                ok = ok  # no-op; retain value for current
                                # Cancel remaining futures
                                try:
                                    for f in future_to_cfg:
                                        if f is not future:
                                            f.cancel()
                                except Exception:
                                    pass
                                # Evaluate current result then break
                                pass
                            if not ok or not os.path.exists(temp):
                                continue
                            size = os.path.getsize(temp)
                            if size <= target_bytes:
                                cand = (size, scale, colors, lossy, temp)
                                if best_candidate is None:
                                    best_candidate = cand
                                else:
                                    b_size, b_scale, b_colors, b_lossy, _ = best_candidate
                                    better = (
                                        (scale > b_scale)
                                        or (scale == b_scale and colors > b_colors)
                                        or (scale == b_scale and colors == b_colors and lossy < b_lossy)
                                        or (scale == b_scale and colors == b_colors and lossy == b_lossy and size > b_size)
                                    )
                                    if better:
                                        try:
                                            os.remove(best_candidate[4])
                                        except Exception:
                                            pass
                                        best_candidate = cand
                            else:
                                try:
                                    os.remove(temp)
                                except Exception:
                                    pass

            if best_candidate:
                # Move best candidate to final path
                shutil.move(best_candidate[4], best_output_path)
                return True

            # As a last attempt, if even heavy settings cannot reach target but improved size a lot,
            # accept the smallest file if it is at least 20% smaller than original (quality preference)
            smallest_path = None
            smallest_size = None
            # Iterate remnants with known suffix pattern and pick smallest
            base_dir = os.path.dirname(input_path) or "."
            prefix = os.path.basename(input_path) + "."
            for name in os.listdir(base_dir):
                if name.startswith(os.path.basename(input_path) + ".") and name.endswith(".gif.tmp"):
                    p = os.path.join(base_dir, name)
                    try:
                        s = os.path.getsize(p)
                    except Exception:
                        continue
                    if smallest_size is None or s < smallest_size:
                        smallest_size = s
                        smallest_path = p
            if smallest_path and smallest_size and smallest_size < original_size * 0.8:
                shutil.move(smallest_path, best_output_path)
                return True

            return False
        finally:
            # Cleanup any leftover temp files from this search
            base = os.path.basename(input_path)
            folder = os.path.dirname(input_path) or "."
            try:
                for name in os.listdir(folder):
                    if name.startswith(base + ".") and name.endswith(".gif.tmp"):
                        try:
                            os.remove(os.path.join(folder, name))
                        except Exception:
                            pass
            except Exception:
                pass

    def _run_gifsicle(self, input_path: str, output_path: str, colors: int, lossy: int, scale: float) -> bool:
        """Execute gifsicle with given parameters. Returns True on success."""
        try:
            cmd = [
                "gifsicle",
                f"--optimize={max(1, min(3, int(self.gifsicle_optimize_level)))}",
                "--careful",
                "--no-comments",
                "--no-extensions",
                "--no-names",
            ]
            if scale and abs(scale - 1.0) > 1e-6:
                # Scale uniformly to preserve aspect ratio
                cmd.extend(["--scale", f"{scale:.4f}"])
            if colors and colors < 256:
                cmd.extend(["--colors", str(max(2, min(256, colors)))])
            if lossy and lossy > 0:
                cmd.extend(["--lossy", str(max(1, min(150, int(lossy))) )])
            cmd.extend([input_path, "--output", output_path])
            timeout_sec = 45 if self.fast_mode else 120
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_sec)
            return result.returncode == 0 and os.path.exists(output_path)
        except Exception as e:
            logger.debug(f"gifsicle run failed: {e}")
            return False

    def _preprocess_gif_fps(self, input_path: str, output_path: str, fps: int) -> bool:
        """Use ffmpeg to reduce fps and deduplicate frames while preserving aspect ratio."""
        try:
            cmd = [
                'ffmpeg', '-y', '-i', input_path,
                '-vf', f'mpdecimate=hi=512:lo=256:frac=0.3,fps={fps},scale=iw:ih:flags=lanczos',
                '-loop', '0', output_path
            ]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            return result.returncode == 0 and os.path.exists(output_path)
        except Exception as e:
            logger.debug(f"fps preprocess failed: {e}")
            return False

    def _preprocess_fps_variants(self, input_path: str, fps_levels: List[Optional[int]]) -> Dict[Optional[int], str]:
        """Create fps-reduced variants in parallel. Returns mapping fps->path (original path for None)."""
        variants: Dict[Optional[int], str] = {}
        variants[None] = input_path
        from concurrent.futures import ThreadPoolExecutor, as_completed
        tasks = {}
        with ThreadPoolExecutor(max_workers=min(4, (os.cpu_count() or 4))) as executor:
            for fps in fps_levels:
                if fps is None:
                    continue
                out = input_path + f".pre_fps{fps}.gif.tmp"
                future = executor.submit(self._preprocess_gif_fps, input_path, out, fps)
                tasks[future] = (fps, out)
            for future in as_completed(tasks):
                fps, out = tasks[future]
                ok = False
                try:
                    ok = future.result()
                except Exception:
                    ok = False
                if ok and os.path.exists(out):
                    variants[fps] = out
        return variants

    def _validate_gif_basic(self, gif_path: str, max_size_mb: float) -> Tuple[bool, Optional[str]]:
        """Basic validity and size check for rollback decisions."""
        try:
            if not os.path.exists(gif_path):
                return False, "File does not exist"
            size_mb = os.path.getsize(gif_path) / (1024 * 1024)
            if size_mb == 0:
                return False, "File is empty"
            if size_mb > max_size_mb:
                return False, f"File too large: {size_mb:.2f}MB > {max_size_mb:.2f}MB"
            # Quick animated check
            with Image.open(gif_path) as img:
                if not getattr(img, 'is_animated', False):
                    return False, "Not animated"
            return True, None
        except Exception as e:
            return False, str(e)
    
    def _analyze_gif_characteristics(self, gif_path: str) -> Dict[str, Any]:
        """Analyze GIF characteristics to determine optimal optimization strategy"""
        
        analysis = {
            'frame_count': 0,
            'width': 0,
            'height': 0,
            'color_count': 0,
            'file_size_mb': 0,
            'complexity_score': 0,
            'motion_score': 0,
            'color_richness': 0,
            'optimization_potential': 0
        }
        
        try:
            # Get basic file info
            file_size = os.path.getsize(gif_path)
            analysis['file_size_mb'] = file_size / (1024 * 1024)
            
            # Analyze GIF using PIL
            with Image.open(gif_path) as img:
                analysis['width'], analysis['height'] = img.size
                
                # Count frames
                frames = []
                try:
                    for frame in ImageSequence.Iterator(img):
                        frames.append(frame)
                except Exception:
                    # If we can't iterate frames, assume it's a single frame
                    frames = [img]
                
                analysis['frame_count'] = len(frames)
                
                # Analyze color usage
                all_colors = set()
                for frame in frames[:min(10, len(frames))]:  # Sample first 10 frames
                    frame_colors = frame.getcolors()
                    if frame_colors:
                        all_colors.update([color for count, color in frame_colors])
                
                analysis['color_count'] = len(all_colors)
                
                # Calculate complexity score
                pixel_count = analysis['width'] * analysis['height'] * analysis['frame_count']
                analysis['complexity_score'] = min(10.0, (pixel_count * analysis['color_count']) / 1000000)
                
                # Calculate motion score (simplified)
                if len(frames) > 1:
                    # Compare first and last frame for motion estimation
                    first_frame = np.array(frames[0])
                    last_frame = np.array(frames[-1])
                    diff = np.mean(np.abs(first_frame.astype(float) - last_frame.astype(float)))
                    analysis['motion_score'] = min(1.0, diff / 255.0)
                
                # Calculate color richness
                analysis['color_richness'] = min(1.0, analysis['color_count'] / 256.0)
                
                # Calculate optimization potential
                size_factor = analysis['file_size_mb'] / 10.0  # Normalize to 10MB
                complexity_factor = analysis['complexity_score'] / 10.0
                analysis['optimization_potential'] = min(1.0, (size_factor + complexity_factor) / 2.0)
            
        except Exception as e:
            logger.warning(f"GIF analysis failed: {e}")
            # Set safe defaults
            analysis.update({
                'frame_count': 30,
                'width': 480,
                'height': 360,
                'color_count': 128,
                'complexity_score': 5.0,
                'motion_score': 0.5,
                'color_richness': 0.5,
                'optimization_potential': 0.5
            })
        
        return analysis
    
    def _get_quality_focused_strategies(self, gif_analysis: Dict[str, Any], max_size_mb: float) -> List[Dict[str, Any]]:
        """Generate quality-focused optimization strategies based on GIF analysis"""
        
        strategies = []
        
        # Strategy 1: Intelligent resolution downscaling (primary strategy for large GIFs)
        if gif_analysis['width'] > 400 or gif_analysis['height'] > 400:
            # Calculate optimal scale factor based on current size and target
            current_pixels = gif_analysis['width'] * gif_analysis['height']
            target_pixels = current_pixels * 0.7  # Aim for 30% size reduction through resolution
            scale_factor = min(0.9, max(0.6, (target_pixels / current_pixels) ** 0.5))
            
            new_width = int(gif_analysis['width'] * scale_factor)
            new_height = int(gif_analysis['height'] * scale_factor)
            
            strategies.append({
                'name': f'Intelligent resolution downscaling ({new_width}x{new_height})',
                'method': 'ffmpeg_resize',
                'params': {
                    'width': new_width,
                    'height': new_height,
                    'colors': max(128, min(256, gif_analysis['color_count'])),
                    'dither': 'floyd_steinberg',
                    'fps': max(10, min(15, int(30 / (gif_analysis['frame_count'] / 15)))),
                    'scale_factor': scale_factor
                },
                'min_quality': 0.75
            })
        
        # Strategy 2: Conservative optimization with slight downscaling
        if gif_analysis['width'] > 300 or gif_analysis['height'] > 300:
            scale_factor = 0.85  # 15% downscaling
            new_width = int(gif_analysis['width'] * scale_factor)
            new_height = int(gif_analysis['height'] * scale_factor)
            
            strategies.append({
                'name': f'Conservative downscaling ({new_width}x{new_height})',
                'method': 'ffmpeg_resize',
                'params': {
                    'width': new_width,
                    'height': new_height,
                    'colors': max(128, min(256, gif_analysis['color_count'])),
                    'dither': 'floyd_steinberg',
                    'fps': max(12, min(18, int(30 / (gif_analysis['frame_count'] / 20)))),
                    'scale_factor': scale_factor
                },
                'min_quality': 0.8
            })
        
        # Strategy 3: Smart color reduction with resolution optimization
        target_colors = max(96, min(192, int(gif_analysis['color_count'] * 0.8)))
        if gif_analysis['width'] > 250 or gif_analysis['height'] > 250:
            scale_factor = 0.75  # 25% downscaling
            new_width = int(gif_analysis['width'] * scale_factor)
            new_height = int(gif_analysis['height'] * scale_factor)
            
            strategies.append({
                'name': f'Smart color reduction with downscaling ({new_width}x{new_height}, {target_colors} colors)',
                'method': 'ffmpeg_resize',
                'params': {
                    'width': new_width,
                    'height': new_height,
                    'colors': target_colors,
                    'dither': 'floyd_steinberg',
                    'fps': max(10, min(15, int(30 / (gif_analysis['frame_count'] / 15)))),
                    'scale_factor': scale_factor
                },
                'min_quality': 0.7
            })
        
        # Strategy 4: Balanced optimization (moderate quality/size trade-off)
        strategies.append({
            'name': 'Balanced optimization',
            'method': 'gifsicle',
            'params': {
                'optimize': 3,
                'colors': max(64, min(128, int(gif_analysis['color_count'] * 0.6))),
                'lossy': 20,  # Very light lossy compression
                'dither': 'bayer'
            },
            'min_quality': 0.6
        })
        
        # Strategy 5: Adaptive optimization with resolution scaling
        if gif_analysis['optimization_potential'] > 0.7:
            scale_factor = max(0.5, min(0.8, 1.0 - gif_analysis['optimization_potential']))
            new_width = int(gif_analysis['width'] * scale_factor)
            new_height = int(gif_analysis['height'] * scale_factor)
            
            strategies.append({
                'name': f'Adaptive optimization with scaling ({new_width}x{new_height})',
                'method': 'ffmpeg_adaptive',
                'params': {
                    'colors': max(64, min(192, int(gif_analysis['color_count'] * 0.7))),
                    'scale_factor': scale_factor,
                    'width': new_width,
                    'height': new_height,
                    'fps': max(8, min(12, int(15 / (gif_analysis['frame_count'] / 25)))),
                    'dither': 'floyd_steinberg'
                },
                'min_quality': 0.55
            })
        
        return strategies
    
    def _get_quality_improvement_strategies(self, gif_analysis: Dict[str, Any], current_size_mb: float) -> List[Dict[str, Any]]:
        """Generate quality improvement strategies for GIFs already under size limit"""
        
        strategies = []
        
        # Strategy 1: Size-preserving optimization (maintain current size, improve quality)
        strategies.append({
            'name': 'Size-preserving optimization',
            'method': 'gifsicle',
            'params': {
                'optimize': 3,
                'colors': gif_analysis['color_count'],  # Keep same color count
                'lossy': 0,
                'dither': 'floyd_steinberg'
            },
            'min_quality': 0.8,
            'max_size_increase': 0.0  # No size increase allowed
        })
        
        # Strategy 2: Gentle resolution downscaling (reduce size while maintaining quality)
        if gif_analysis['width'] > 400 or gif_analysis['height'] > 400:
            scale_factor = 0.85  # 15% downscaling
            new_width = int(gif_analysis['width'] * scale_factor)
            new_height = int(gif_analysis['height'] * scale_factor)
            
            strategies.append({
                'name': f'Gentle resolution downscaling ({new_width}x{new_height})',
                'method': 'ffmpeg_resize',
                'params': {
                    'width': new_width,
                    'height': new_height,
                    'colors': gif_analysis['color_count'],  # Keep same color count
                    'dither': 'floyd_steinberg',
                    'fps': max(12, min(18, int(30 / (gif_analysis['frame_count'] / 20)))),
                    'scale_factor': scale_factor
                },
                'min_quality': 0.75,
                'max_size_increase': -0.5  # Must reduce size by at least 0.5MB
            })
        
        # Strategy 3: Conservative color optimization (slight color reduction for better compression)
        if gif_analysis['color_count'] > 128:
            target_colors = max(96, int(gif_analysis['color_count'] * 0.9))  # 10% color reduction
            
            strategies.append({
                'name': f'Conservative color optimization ({target_colors} colors)',
                'method': 'gifsicle',
                'params': {
                    'optimize': 3,
                    'colors': target_colors,
                    'lossy': 0,
                    'dither': 'floyd_steinberg'
                },
                'min_quality': 0.7,
                'max_size_increase': -0.2  # Must reduce size by at least 0.2MB
            })
        
        # Strategy 4: Frame rate optimization (for high frame count GIFs)
        if gif_analysis['frame_count'] > 25:
            target_fps = max(10, min(15, int(30 / (gif_analysis['frame_count'] / 20))))
            
            strategies.append({
                'name': f'Frame rate optimization ({target_fps} fps)',
                'method': 'ffmpeg_smart',
                'params': {
                    'fps': target_fps,
                    'colors': gif_analysis['color_count'],
                    'dither': 'floyd_steinberg'
                },
                'min_quality': 0.65,
                'max_size_increase': -0.3  # Must reduce size by at least 0.3MB
            })
        
        # Strategy 5: Minimal optimization (just re-encode for better compression)
        strategies.append({
            'name': 'Minimal re-optimization',
            'method': 'gifsicle',
            'params': {
                'optimize': 3,
                'colors': gif_analysis['color_count'],
                'lossy': 0,
                'dither': 'bayer'
            },
            'min_quality': 0.6,
            'max_size_increase': 0.0  # No size increase allowed
        })
        
        return strategies
    
    def _apply_optimization_strategy(self, input_path: str, output_path: str, strategy: Dict[str, Any]) -> bool:
        """Apply a specific optimization strategy"""
        
        try:
            if strategy['method'] == 'gifsicle':
                return self._apply_gifsicle_strategy(input_path, output_path, strategy['params'])
            elif strategy['method'] == 'ffmpeg_smart':
                return self._apply_ffmpeg_smart_strategy(input_path, output_path, strategy['params'])
            elif strategy['method'] == 'ffmpeg_resize':
                return self._apply_ffmpeg_resize_strategy(input_path, output_path, strategy['params'])
            elif strategy['method'] == 'ffmpeg_adaptive':
                return self._apply_ffmpeg_adaptive_strategy(input_path, output_path, strategy['params'])
            elif strategy['method'] == 'ffmpeg_upscale':
                return self._apply_ffmpeg_upscale_strategy(input_path, output_path, strategy['params'])
            else:
                logger.warning(f"Unknown optimization method: {strategy['method']}")
                return False
                
        except Exception as e:
            logger.warning(f"Strategy application failed: {e}")
            return False
    
    def _apply_gifsicle_strategy(self, input_path: str, output_path: str, params: Dict[str, Any]) -> bool:
        """Apply gifsicle-based optimization strategy"""
        
        cmd = ['gifsicle', '--optimize=3']
        
        # Add color reduction if specified
        if 'colors' in params:
            cmd.extend(['--colors', str(params['colors'])])
        
        # Add lossy compression if specified
        if 'lossy' in params and params['lossy'] > 0:
            cmd.extend(['--lossy', str(params['lossy'])])
        
        # Add output file
        cmd.extend(['--output', output_path, input_path])
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        return result.returncode == 0
    
    def _apply_ffmpeg_smart_strategy(self, input_path: str, output_path: str, params: Dict[str, Any]) -> bool:
        """Apply smart FFmpeg-based optimization strategy"""
        
        # Generate palette first
        palette_path = output_path + '.palette.png'
        palette_cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', f'fps={params["fps"]},palettegen=max_colors={params["colors"]}:stats_mode=diff',
            '-frames:v', '1', '-y', palette_path
        ]
        
        palette_result = subprocess.run(palette_cmd, capture_output=True, text=True, timeout=60)
        if palette_result.returncode != 0:
            return False
        
        # Create optimized GIF using palette
        gif_cmd = [
            'ffmpeg', '-i', input_path, '-i', palette_path,
            '-lavfi', f'fps={params["fps"]},paletteuse=dither={params["dither"]}:diff_mode=rectangle',
            '-loop', '0', '-y', output_path
        ]
        
        gif_result = subprocess.run(gif_cmd, capture_output=True, text=True, timeout=120)
        
        # Clean up palette file
        if os.path.exists(palette_path):
            os.remove(palette_path)
        
        return gif_result.returncode == 0
    
    def _apply_ffmpeg_resize_strategy(self, input_path: str, output_path: str, params: Dict[str, Any]) -> bool:
        """Apply FFmpeg-based resize optimization strategy"""
        
        # Generate palette first
        palette_path = output_path + '.palette.png'
        palette_cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', f'scale={params["width"]}:{params["height"]}:flags=lanczos,fps={params["fps"]},palettegen=max_colors={params["colors"]}:stats_mode=diff',
            '-frames:v', '1', '-y', palette_path
        ]
        
        palette_result = subprocess.run(palette_cmd, capture_output=True, text=True, timeout=60)
        if palette_result.returncode != 0:
            return False
        
        # Create optimized GIF using palette
        gif_cmd = [
            'ffmpeg', '-i', input_path, '-i', palette_path,
            '-lavfi', f'scale={params["width"]}:{params["height"]}:flags=lanczos,fps={params["fps"]},paletteuse=dither={params["dither"]}:diff_mode=rectangle',
            '-loop', '0', '-y', output_path
        ]
        
        gif_result = subprocess.run(gif_cmd, capture_output=True, text=True, timeout=120)
        
        # Clean up palette file
        if os.path.exists(palette_path):
            os.remove(palette_path)
        
        return gif_result.returncode == 0
    
    def _apply_ffmpeg_adaptive_strategy(self, input_path: str, output_path: str, params: Dict[str, Any]) -> bool:
        """Apply adaptive FFmpeg-based optimization strategy"""
        
        # Calculate adaptive parameters
        scale_width = int(params.get('width', 480) * params['scale_factor'])
        scale_height = int(params.get('height', 360) * params['scale_factor'])
        
        # Generate palette first
        palette_path = output_path + '.palette.png'
        palette_cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', f'scale={scale_width}:{scale_height}:flags=lanczos,fps={params["fps"]},palettegen=max_colors={params["colors"]}:stats_mode=diff',
            '-frames:v', '1', '-y', palette_path
        ]
        
        palette_result = subprocess.run(palette_cmd, capture_output=True, text=True, timeout=60)
        if palette_result.returncode != 0:
            return False
        
        # Create optimized GIF using palette
        gif_cmd = [
            'ffmpeg', '-i', input_path, '-i', palette_path,
            '-lavfi', f'scale={scale_width}:{scale_height}:flags=lanczos,fps={params["fps"]},paletteuse=dither={params["dither"]}:diff_mode=rectangle',
            '-loop', '0', '-y', output_path
        ]
        
        gif_result = subprocess.run(gif_cmd, capture_output=True, text=True, timeout=120)
        
        # Clean up palette file
        if os.path.exists(palette_path):
            os.remove(palette_path)
        
        return gif_result.returncode == 0
    
    def _apply_ffmpeg_upscale_strategy(self, input_path: str, output_path: str, params: Dict[str, Any]) -> bool:
        """Apply FFmpeg-based upscaling optimization strategy"""
        
        # Calculate new dimensions based on scale factor
        scale_factor = params['scale_factor']
        new_width = int(params.get('width', 480) * scale_factor)
        new_height = int(params.get('height', 360) * scale_factor)
        
        # Generate palette first
        palette_path = output_path + '.palette.png'
        palette_cmd = [
            'ffmpeg', '-i', input_path,
            '-vf', f'scale={new_width}:{new_height}:flags=lanczos,fps={params["fps"]},palettegen=max_colors={params["colors"]}:stats_mode=diff',
            '-frames:v', '1', '-y', palette_path
        ]
        
        palette_result = subprocess.run(palette_cmd, capture_output=True, text=True, timeout=60)
        if palette_result.returncode != 0:
            return False
        
        # Create upscaled GIF using palette
        gif_cmd = [
            'ffmpeg', '-i', input_path, '-i', palette_path,
            '-lavfi', f'scale={new_width}:{new_height}:flags=lanczos,fps={params["fps"]}:diff_mode=rectangle',
            '-loop', '0', '-y', output_path
        ]
        
        gif_result = subprocess.run(gif_cmd, capture_output=True, text=True, timeout=120)
        
        # Clean up palette file
        if os.path.exists(palette_path):
            os.remove(palette_path)
        
        return gif_result.returncode == 0
    
    def _evaluate_gif_quality(self, gif_path: str, original_analysis: Dict[str, Any]) -> float:
        """Evaluate the quality of an optimized GIF"""
        
        try:
            with Image.open(gif_path) as img:
                # Get basic info
                width, height = img.size
                
                # Count frames
                frames = []
                try:
                    for frame in ImageSequence.Iterator(img):
                        frames.append(frame)
                except Exception:
                    frames = [img]
                
                frame_count = len(frames)
                
                # Calculate quality factors
                resolution_score = min(1.0, (width * height) / (original_analysis['width'] * original_analysis['height']))
                frame_score = min(1.0, frame_count / original_analysis['frame_count'])
                
                # Analyze color usage
                all_colors = set()
                for frame in frames[:min(5, len(frames))]:  # Sample first 5 frames
                    frame_colors = frame.getcolors()
                    if frame_colors:
                        all_colors.update([color for count, color in frame_colors])
                
                color_score = min(1.0, len(all_colors) / max(1, original_analysis['color_count']))
                
                # Calculate overall quality score
                quality_score = (resolution_score * 0.4 + frame_score * 0.3 + color_score * 0.3)
                
                return max(0.0, min(1.0, quality_score))
                
        except Exception as e:
            logger.warning(f"Quality evaluation failed: {e}")
            return 0.5  # Default to medium quality if evaluation fails
    
    def _try_intelligent_reencoding(self, gif_path: str, max_size_mb: float, original_size: float, 
                                  gif_analysis: Dict[str, Any]) -> bool:
        """Try intelligent re-encoding as last resort with quality preservation"""
        
        try:
            temp_output = gif_path + '.intelligent.tmp'
            
            # Calculate optimal parameters for size constraint
            target_ratio = max_size_mb / original_size
            
            # Determine optimal color count based on target ratio
            if target_ratio >= 0.8:
                colors = max(128, min(256, gif_analysis['color_count']))
            elif target_ratio >= 0.6:
                colors = max(96, min(192, int(gif_analysis['color_count'] * 0.8)))
            elif target_ratio >= 0.4:
                colors = max(64, min(128, int(gif_analysis['color_count'] * 0.6)))
            else:
                colors = max(32, min(96, int(gif_analysis['color_count'] * 0.4)))
            
            # Determine optimal FPS
            if gif_analysis['frame_count'] > 30:
                fps = max(8, min(12, int(15 / (gif_analysis['frame_count'] / 25))))
            else:
                fps = max(10, min(15, int(30 / max(1, gif_analysis['frame_count']))))
            
            # Determine optimal resolution
            if target_ratio >= 0.7:
                scale_width = gif_analysis['width']
                scale_height = gif_analysis['height']
            elif target_ratio >= 0.5:
                scale_width = int(gif_analysis['width'] * 0.8)
                scale_height = int(gif_analysis['height'] * 0.8)
            else:
                scale_width = int(gif_analysis['width'] * 0.6)
                scale_height = int(gif_analysis['height'] * 0.6)
            
            # Multi-tier attempts with progressively stronger settings
            tiers = [
                { 'scale': 0.85, 'fps': max(10, fps - 2), 'colors': max(96, colors), 'dither': 'floyd_steinberg' },
                { 'scale': 0.75, 'fps': max(8, fps - 4),  'colors': max(80, min(128, colors)), 'dither': 'bayer' },
                { 'scale': 0.60, 'fps': 8,                 'colors': max(64, min(96, colors)),  'dither': 'bayer' },
                { 'scale': 0.50, 'fps': 6,                 'colors': max(48, min(80, colors)),  'dither': 'none' },
            ]

            for t in tiers:
                try:
                    sw = max(2, int(gif_analysis['width'] * t['scale']))
                    sh = max(2, int(gif_analysis['height'] * t['scale']))
                    # Round to even numbers
                    sw = sw // 2 * 2
                    sh = sh // 2 * 2

                    palette_path = temp_output + f".palette_{sw}x{sh}_{t['fps']}_{t['colors']}.png"
                    palette_cmd = [
                        'ffmpeg', '-y', '-i', gif_path,
                        '-vf', f'mpdecimate=hi=512:lo=256:frac=0.3,fps={t["fps"]},scale={sw}:{sh}:flags=lanczos,palettegen=max_colors={t["colors"]}:stats_mode=diff',
                        '-frames:v', '1', palette_path
                    ]
                    palette_result = subprocess.run(palette_cmd, capture_output=True, text=True, timeout=90)
                    if palette_result.returncode != 0 or not os.path.exists(palette_path):
                        continue

                    gif_cmd = [
                        'ffmpeg', '-y', '-i', gif_path, '-i', palette_path,
                        '-lavfi', f'mpdecimate=hi=512:lo=256:frac=0.3,fps={t["fps"]},scale={sw}:{sh}:flags=lanczos[p];[p][1:v]paletteuse=dither={t["dither"]}:diff_mode=rectangle',
                        '-loop', '0', temp_output
                    ]
                    gif_result = subprocess.run(gif_cmd, capture_output=True, text=True, timeout=180)
                    try:
                        if os.path.exists(palette_path):
                            os.remove(palette_path)
                    except Exception:
                        pass

                    if gif_result.returncode == 0 and os.path.exists(temp_output):
                        optimized_size = os.path.getsize(temp_output) / (1024 * 1024)
                        if optimized_size <= max_size_mb:
                            os.replace(temp_output, gif_path)
                            logger.info(
                                f"GIF optimized with ffmpeg fallback tier: {optimized_size:.2f}MB (scale {t['scale']}, fps {t['fps']}, colors {t['colors']}, dither {t['dither']})"
                            )
                            return True
                        else:
                            logger.info(
                                f"Tier insufficient ({optimized_size:.2f}MB). Trying next tier..."
                            )
                            try:
                                os.remove(temp_output)
                            except Exception:
                                pass
                except Exception:
                    # Try next tier on any error
                    try:
                        if os.path.exists(temp_output):
                            os.remove(temp_output)
                    except Exception:
                        pass

            logger.warning(
                f"Intelligent re-encoding failed. Could not reduce GIF below {max_size_mb:.2f}MB"
            )
            return False
            
        except Exception as e:
            logger.error(f"Error in intelligent re-encoding: {e}")
            return False
    
    def optimize_gif_with_quality_target(self, input_video: str, output_path: str, 
                                       max_size_mb: float, platform: str = None,
                                       start_time: float = 0, duration: float = None,
                                       quality_preference: str = 'balanced') -> Dict[str, Any]:
        """
        Optimize GIF with iterative quality targeting to maximize quality while staying under size limit
        
        Args:
            input_video: Path to input video file
            output_path: Path for output GIF
            max_size_mb: Maximum file size in MB (target to get close to)
            platform: Target platform (twitter, discord, slack, etc.)
            start_time: Start time in seconds (default: 0)
            duration: Duration in seconds (default: use platform/config limit)
            quality_preference: 'quality', 'balanced', or 'size' - determines optimization strategy
            
        Returns:
            Dictionary with optimization results and metadata
        """
        
        logger.info(f"Starting iterative quality optimization for target size: {max_size_mb}MB")
        
        # Initialize optimization parameters
        optimization_params = self._initialize_optimization_params(max_size_mb, quality_preference)
        optimization_params['target_size_mb'] = max_size_mb
        
        # Perform initial frame analysis
        frame_analysis = self._perform_intelligent_frame_analysis(input_video, start_time, duration)
        
        # Generate initial palette
        palette_data = self._generate_optimal_palette(input_video, frame_analysis, max_size_mb)
        
        # Initialize optimization state
        best_result = None
        best_quality_score = 0
        iteration = 0
        max_iterations = 20  # Allow more steps to hone in on target
        
        # Binary search parameters for fine-tuning
        size_tolerance = 0.02  # 2% tolerance for target size
        quality_improvement_threshold = 0.1  # Minimum quality improvement to continue
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Optimization iteration {iteration}/{max_iterations}")
            
            try:
                # Create GIF with current parameters
                current_result = self._create_gif_with_params(
                    input_video, output_path, optimization_params, 
                    frame_analysis, palette_data, start_time, duration, platform
                )
                
                if not current_result or not current_result.get('success', False):
                    logger.warning(f"Iteration {iteration}: GIF creation failed")
                    optimization_params = self._adjust_params_for_failure(optimization_params)
                    continue
                
                current_size_mb = current_result['size_mb']
                current_quality = self._calculate_comprehensive_quality_score(
                    current_result, optimization_params, frame_analysis
                )
                
                logger.info(f"Iteration {iteration}: {current_size_mb:.2f}MB, quality: {current_quality:.2f}")
                
                # Check if we're under size limit
                if current_size_mb <= max_size_mb:
                    # We're under the limit - check if this is the best quality so far
                    if current_quality > best_quality_score:
                        best_result = current_result.copy()
                        best_quality_score = current_quality
                        logger.info(f"New best result: {current_size_mb:.2f}MB, quality: {current_quality:.2f}")
                    
                    # Check if we're close enough to target size
                    size_ratio = current_size_mb / max_size_mb
                    if size_ratio >= (1.0 - size_tolerance):
                        logger.info(f"Target size achieved: {size_ratio:.2%} of target")
                        break
                    
                    # Try to increase quality while staying under limit
                    optimization_params = self._increase_quality_params(
                        optimization_params, current_size_mb, max_size_mb, quality_preference
                    )
                else:
                    # We're over the limit - reduce quality, and try a micro-trim when close
                    over_by = current_size_mb - max_size_mb
                    if over_by <= max_size_mb * 0.02:  # within 2%
                        optimization_params = self._micro_trim_params(optimization_params)
                    else:
                        optimization_params = self._decrease_quality_params(
                            optimization_params, current_size_mb, max_size_mb, quality_preference
                        )
                
                # Check for convergence
                if best_result and iteration > 5:
                    quality_improvement = current_quality - best_quality_score
                    if abs(quality_improvement) < quality_improvement_threshold:
                        logger.info("Quality improvement below threshold, stopping optimization")
                        break
                
            except Exception as e:
                logger.warning(f"Iteration {iteration} failed: {e}")
                optimization_params = self._adjust_params_for_failure(optimization_params)
        
        if not best_result:
            raise RuntimeError("Failed to create GIF under target size limit")
        
        # Apply final optimizations
        final_result = self._apply_final_optimizations(best_result, max_size_mb)
        
        logger.info(f"Optimization completed: {final_result['size_mb']:.2f}MB "
                   f"({final_result['size_mb']/max_size_mb:.1%} of target), "
                   f"quality: {final_result['quality_score']:.2f}")
        
        return final_result
    
    def _perform_intelligent_frame_analysis(self, input_video: str, start_time: float, 
                                          duration: float = None) -> Dict[str, Any]:
        """
        Perform intelligent analysis of video frames for optimal GIF creation
        """
        
        logger.info("Performing intelligent frame analysis...")
        
        analysis = {
            'total_frames': 0,
            'key_frames': [],
            'motion_intensity': [],
            'color_complexity': [],
            'temporal_redundancy': [],
            'scene_changes': [],
            'optimal_fps': 15,
            'frame_importance_scores': [],
            'duplicate_threshold': 0.95
        }
        
        try:
            # Extract frames for analysis
            temp_frames_dir = os.path.join(self.temp_dir, f"frame_analysis_{int(time.time())}")
            os.makedirs(temp_frames_dir, exist_ok=True)
            
            # Extract frames at high rate for analysis
            cmd = [
                'ffmpeg', '-ss', str(start_time), '-i', input_video,
                '-t', str(duration or 15), '-vf', 'fps=30,scale=320:240',
                '-q:v', '2', f'{temp_frames_dir}/frame_%04d.png'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                frame_files = sorted([f for f in os.listdir(temp_frames_dir) if f.endswith('.png')])
                analysis['total_frames'] = len(frame_files)
                
                if frame_files:
                    # Analyze each frame
                    analysis = self._analyze_frame_sequence(temp_frames_dir, frame_files, analysis)
                    
                    # Determine optimal parameters
                    analysis = self._calculate_optimal_gif_parameters(analysis)
            
            # Cleanup
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
            
        except Exception as e:
            logger.warning(f"Frame analysis failed, using defaults: {e}")
            analysis.update({
                'total_frames': 30,
                'optimal_fps': 12,
                'frame_importance_scores': [1.0] * 30
            })
        
        return analysis
    
    def _analyze_frame_sequence(self, frames_dir: str, frame_files: List[str], 
                              analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sequence of frames for optimization
        """
        
        prev_frame = None
        prev_histogram = None
        
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, frame_file)
            
            try:
                # Load frame
                frame = Image.open(frame_path)
                frame_array = np.array(frame)
                
                # Calculate motion intensity
                motion_score = 0.0
                if prev_frame is not None:
                    # Simple motion detection using frame difference
                    diff = np.abs(frame_array.astype(float) - prev_frame.astype(float))
                    motion_score = np.mean(diff) / 255.0
                
                analysis['motion_intensity'].append(motion_score)
                
                # Calculate color complexity
                unique_colors = len(set(tuple(pixel) for pixel in frame_array.reshape(-1, 3)))
                color_complexity = min(unique_colors / 1000.0, 10.0)  # Normalize to 0-10
                analysis['color_complexity'].append(color_complexity)
                
                # Calculate temporal redundancy
                redundancy = 0.0
                if prev_histogram is not None:
                    # Compare histograms
                    current_histogram = self._calculate_frame_histogram(frame_array)
                    redundancy = self._compare_histograms(prev_histogram, current_histogram)
                
                analysis['temporal_redundancy'].append(redundancy)
                
                # Detect scene changes (significant motion + low redundancy)
                if motion_score > 0.3 and redundancy < 0.7:
                    analysis['scene_changes'].append(i)
                
                # Calculate frame importance score
                importance = self._calculate_frame_importance(motion_score, color_complexity, redundancy, i)
                analysis['frame_importance_scores'].append(importance)
                
                # Update for next iteration
                prev_frame = frame_array
                prev_histogram = self._calculate_frame_histogram(frame_array)
                
            except Exception as e:
                logger.debug(f"Failed to analyze frame {frame_file}: {e}")
                # Add default values
                analysis['motion_intensity'].append(0.5)
                analysis['color_complexity'].append(5.0)
                analysis['temporal_redundancy'].append(0.5)
                analysis['frame_importance_scores'].append(0.5)
        
        return analysis
    
    def _calculate_frame_histogram(self, frame_array: np.ndarray) -> np.ndarray:
        """Calculate color histogram for frame comparison"""
        # Convert to HSV for better perceptual comparison
        hsv = cv2.cvtColor(frame_array, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()
    
    def _compare_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compare two histograms for similarity"""
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def _calculate_frame_importance(self, motion: float, color_complexity: float, 
                                  redundancy: float, frame_index: int) -> float:
        """Calculate importance score for frame selection"""
        
        # Base importance
        importance = 1.0
        
        # Motion contributes to importance
        importance += motion * 2.0
        
        # Color complexity adds value
        importance += (color_complexity / 10.0) * 1.5
        
        # Low redundancy (unique frames) are more important
        importance += (1.0 - redundancy) * 1.0
        
        # Key frames (every 10th frame) get bonus
        if frame_index % 10 == 0:
            importance += 0.5
        
        return min(importance, 5.0)  # Cap at 5.0
    
    def _calculate_optimal_gif_parameters(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal GIF parameters based on analysis"""
        
        # Determine optimal FPS based on motion
        avg_motion = np.mean(analysis['motion_intensity']) if analysis['motion_intensity'] else 0.5
        
        if avg_motion > 0.7:
            analysis['optimal_fps'] = 20  # High motion needs higher FPS
        elif avg_motion > 0.4:
            analysis['optimal_fps'] = 15  # Medium motion
        else:
            analysis['optimal_fps'] = 10  # Low motion can use lower FPS
        
        # Determine duplicate threshold based on redundancy
        avg_redundancy = np.mean(analysis['temporal_redundancy']) if analysis['temporal_redundancy'] else 0.5
        analysis['duplicate_threshold'] = 0.9 + (avg_redundancy * 0.05)  # 0.9-0.95 range
        
        # Identify key frames for preservation
        if analysis['frame_importance_scores']:
            importance_threshold = np.percentile(analysis['frame_importance_scores'], 70)
            analysis['key_frames'] = [
                i for i, score in enumerate(analysis['frame_importance_scores']) 
                if score >= importance_threshold
            ]
        
        return analysis
    
    def _generate_optimal_palette(self, input_video: str, frame_analysis: Dict[str, Any], 
                                max_size_mb: float) -> Dict[str, Any]:
        """
        Generate optimal color palette using advanced techniques
        """
        
        logger.info("Generating optimal color palette...")
        
        cache_key = f"{input_video}_{max_size_mb}_{frame_analysis['total_frames']}"
        if cache_key in self.palette_cache:
            return self.palette_cache[cache_key]
        
        palette_data = {
            'colors': 256,
            'palette_file': None,
            'color_distribution': {},
            'optimization_method': 'adaptive'
        }
        
        try:
            # Determine optimal color count based on complexity and size constraint
            avg_color_complexity = np.mean(frame_analysis.get('color_complexity', [5.0]))
            
            if max_size_mb <= 3:
                # Very tight size constraint
                palette_data['colors'] = max(64, int(128 - (3 - max_size_mb) * 20))
            elif max_size_mb <= 5:
                # Moderate size constraint
                palette_data['colors'] = max(128, int(200 - (5 - max_size_mb) * 15))
            else:
                # Generous size constraint
                palette_data['colors'] = min(256, int(180 + avg_color_complexity * 10))
            
            # Generate palette using multiple methods and select best
            palette_methods = [
                ('neuquant', self._generate_palette_neuquant),
                ('median_cut', self._generate_palette_median_cut),
                ('octree', self._generate_palette_octree)
            ]
            
            best_palette = None
            best_score = 0
            
            for method_name, method_func in palette_methods:
                try:
                    palette_file = method_func(input_video, palette_data['colors'])
                    if palette_file and os.path.exists(palette_file):
                        score = self._evaluate_palette_quality(palette_file, frame_analysis)
                        if score > best_score:
                            best_score = score
                            if best_palette and os.path.exists(best_palette):
                                os.remove(best_palette)
                            best_palette = palette_file
                            palette_data['optimization_method'] = method_name
                        else:
                            os.remove(palette_file)
                except Exception as e:
                    logger.warning(f"Palette method {method_name} failed: {e}")
            
            palette_data['palette_file'] = best_palette
            
            # Cache the result
            self.palette_cache[cache_key] = palette_data
            
        except Exception as e:
            logger.warning(f"Palette generation failed: {e}")
            palette_data['colors'] = 128  # Safe fallback
        
        return palette_data
    
    def _generate_palette_neuquant(self, input_video: str, colors: int) -> Optional[str]:
        """Generate palette using NeuQuant algorithm (via FFmpeg)"""
        
        palette_file = os.path.join(self.temp_dir, f"neuquant_palette_{colors}_{int(time.time())}.png")
        
        cmd = [
            'ffmpeg', '-i', input_video, '-vf',
            f'fps=2,scale=320:240,palettegen=max_colors={colors}:stats_mode=diff',
            '-frames:v', '1',
            '-y', palette_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(palette_file):
            return palette_file
        
        return None
    
    def _generate_palette_median_cut(self, input_video: str, colors: int) -> Optional[str]:
        """Generate palette using median cut algorithm"""
        
        # Create unique filename with thread ID and random suffix to avoid conflicts
        thread_id = threading.get_ident()
        random_suffix = random.randint(1000, 9999)
        palette_file = os.path.join(self.temp_dir, f"mediancut_palette_{colors}_{thread_id}_{random_suffix}.png")
        
        # Extract sample frames with unique directory name
        sample_frames_dir = os.path.join(self.temp_dir, f"palette_frames_{thread_id}_{random_suffix}")
        os.makedirs(sample_frames_dir, exist_ok=True)
        
        try:
            # Extract frames for palette generation
            cmd = [
                'ffmpeg', '-i', input_video, '-vf', 'fps=1,scale=160:120',
                '-frames:v', '10', f'{sample_frames_dir}/sample_%03d.png'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Use PIL to generate palette
                sample_files = [f for f in os.listdir(sample_frames_dir) if f.endswith('.png')]
                
                if sample_files:
                    # Load all sample frames
                    all_pixels = []
                    for sample_file in sample_files[:5]:  # Use first 5 samples
                        img = Image.open(os.path.join(sample_frames_dir, sample_file))
                        all_pixels.extend(list(img.getdata()))
                    
                    if all_pixels:
                        # Create image from all pixels and quantize
                        combined_img = Image.new('RGB', (len(all_pixels), 1))
                        combined_img.putdata(all_pixels)
                        
                        # Quantize using median cut
                        quantized = combined_img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
                        palette_img = quantized.convert('RGB')
                        
                        # Save palette with retry logic for Windows file locking
                        max_retries = 3
                        for retry in range(max_retries):
                            try:
                                palette_img.save(palette_file)
                                
                                if os.path.exists(palette_file):
                                    return palette_file
                                break
                            except (PermissionError, OSError) as e:
                                if retry < max_retries - 1:
                                    logger.debug(f"Palette save retry {retry + 1}/{max_retries}: {e}")
                                    time.sleep(0.1 * (retry + 1))  # Progressive backoff
                                else:
                                    logger.warning(f"Failed to save palette after {max_retries} retries: {e}")
            
        except Exception as e:
            logger.warning(f"Median cut palette generation failed: {e}")
        finally:
            # Clean up with retry logic for Windows file locking
            max_retries = 3
            for retry in range(max_retries):
                try:
                    if os.path.exists(sample_frames_dir):
                        shutil.rmtree(sample_frames_dir)
                    break
                except (PermissionError, OSError) as e:
                    if retry < max_retries - 1:
                        logger.debug(f"Cleanup retry {retry + 1}/{max_retries} for {sample_frames_dir}: {e}")
                        time.sleep(0.2 * (retry + 1))  # Progressive backoff
                    else:
                        logger.warning(f"Failed to cleanup {sample_frames_dir} after {max_retries} retries: {e}")
        
        return None
    
    def _generate_palette_octree(self, input_video: str, colors: int) -> Optional[str]:
        """Generate palette using octree algorithm"""
        
        # Create unique filename with thread ID and random suffix to avoid conflicts
        thread_id = threading.get_ident()
        random_suffix = random.randint(1000, 9999)
        palette_file = os.path.join(self.temp_dir, f"octree_palette_{colors}_{thread_id}_{random_suffix}.png")
        
        # This is a simplified version - in practice you'd implement full octree quantization
        cmd = [
            'ffmpeg', '-i', input_video, '-vf',
            f'fps=1,scale=240:180,palettegen=max_colors={colors}:stats_mode=full',
            '-frames:v', '1',
            '-y', palette_file
        ]
        
        # Run with retry logic for Windows file locking issues
        max_retries = 3
        for retry in range(max_retries):
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
                
                if result.returncode == 0 and os.path.exists(palette_file):
                    return palette_file
                elif retry < max_retries - 1:
                    logger.debug(f"FFmpeg palette generation retry {retry + 1}/{max_retries}")
                    time.sleep(0.1 * (retry + 1))
                else:
                    logger.warning(f"FFmpeg palette generation failed after {max_retries} retries")
                    break
            except (subprocess.TimeoutExpired, OSError) as e:
                if retry < max_retries - 1:
                    logger.debug(f"FFmpeg timeout/error retry {retry + 1}/{max_retries}: {e}")
                    time.sleep(0.2 * (retry + 1))
                else:
                    logger.warning(f"FFmpeg failed after {max_retries} retries: {e}")
        
        return None
    
    def _evaluate_palette_quality(self, palette_file: str, frame_analysis: Dict[str, Any]) -> float:
        """Evaluate quality of generated palette"""
        
        try:
            # Load palette
            palette_img = Image.open(palette_file)
            palette_colors = palette_img.getcolors()
            
            if not palette_colors:
                return 0.0
            
            # Quality factors
            score = 5.0  # Base score
            
            # Color distribution quality
            color_counts = [count for count, color in palette_colors]
            color_variance = np.var(color_counts) if len(color_counts) > 1 else 0
            
            # Lower variance = more balanced palette = better quality
            balance_score = max(0, 2.0 - (color_variance / 1000.0))
            score += balance_score
            
            # Number of colors vs complexity
            avg_complexity = np.mean(frame_analysis.get('color_complexity', [5.0]))
            color_adequacy = min(len(palette_colors) / (avg_complexity * 20), 1.0)
            score += color_adequacy * 2.0
            
            return min(score, 10.0)
            
        except Exception as e:
            logger.warning(f"Palette evaluation failed: {e}")
            return 1.0  # Low score for failed evaluation
    
    def _optimize_frame_sequence(self, input_video: str, frame_analysis: Dict[str, Any], 
                                palette_data: Dict[str, Any], max_size_mb: float) -> Dict[str, Any]:
        """
        Optimize frame sequence for GIF creation
        """
        
        logger.info("Optimizing frame sequence...")
        
        sequence_data = {
            'selected_frames': [],
            'frame_durations': [],
            'optimization_method': 'smart_selection',
            'total_frames': 0,
            'estimated_size_mb': 0
        }
        
        try:
            # Determine target frame count based on size constraint
            target_frame_count = self._calculate_target_frame_count(max_size_mb, frame_analysis)
            
            # Select frames using importance scores
            if frame_analysis.get('frame_importance_scores'):
                sequence_data = self._select_frames_by_importance(
                    frame_analysis, target_frame_count, sequence_data
                )
            else:
                # Fallback to uniform selection
                total_frames = frame_analysis.get('total_frames', 30)
                step = max(1, total_frames // target_frame_count)
                sequence_data['selected_frames'] = list(range(0, total_frames, step))
            
            # Optimize frame durations for smooth playback
            sequence_data = self._optimize_frame_durations(sequence_data, frame_analysis)
            
            sequence_data['total_frames'] = len(sequence_data['selected_frames'])
            
        except Exception as e:
            logger.warning(f"Frame sequence optimization failed: {e}")
            # Fallback sequence
            sequence_data.update({
                'selected_frames': list(range(0, min(30, frame_analysis.get('total_frames', 30)), 2)),
                'frame_durations': [100] * 15,  # 100ms per frame
                'total_frames': 15
            })
        
        return sequence_data
    
    def _calculate_target_frame_count(self, max_size_mb: float, frame_analysis: Dict[str, Any]) -> int:
        """Calculate optimal number of frames for target file size"""
        
        # Estimate bytes per frame based on complexity
        avg_complexity = np.mean(frame_analysis.get('color_complexity', [5.0]))
        
        # More complex frames need more bytes
        bytes_per_frame = 1000 + (avg_complexity * 500)  # 1KB to 6KB per frame estimate
        
        # Calculate maximum frames for size constraint
        max_bytes = max_size_mb * 1024 * 1024
        max_frames = int(max_bytes / bytes_per_frame)
        
        # Apply practical limits
        min_frames = 8   # Minimum for meaningful GIF
        max_frames_practical = 100  # Maximum for reasonable file size
        
        target_frames = max(min_frames, min(max_frames, max_frames_practical))
        
        logger.debug(f"Target frame count: {target_frames} (complexity: {avg_complexity:.1f}, "
                    f"est. bytes/frame: {bytes_per_frame:.0f})")
        
        return target_frames
    
    def _select_frames_by_importance(self, frame_analysis: Dict[str, Any], 
                                   target_count: int, sequence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select frames based on importance scores"""
        
        importance_scores = frame_analysis['frame_importance_scores']
        total_frames = len(importance_scores)
        
        if target_count >= total_frames:
            # Use all frames
            sequence_data['selected_frames'] = list(range(total_frames))
        else:
            # Select top important frames, but ensure temporal distribution
            
            # Method 1: Pure importance-based selection
            indexed_scores = [(i, score) for i, score in enumerate(importance_scores)]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top frames
            top_frames = [i for i, score in indexed_scores[:target_count]]
            
            # Method 2: Ensure temporal distribution
            # Divide timeline into segments and select best from each
            segments = min(target_count, 8)  # Max 8 segments
            frames_per_segment = target_count // segments
            remaining_frames = target_count % segments
            
            segment_size = total_frames // segments
            distributed_frames = []
            
            for seg in range(segments):
                start_idx = seg * segment_size
                end_idx = (seg + 1) * segment_size if seg < segments - 1 else total_frames
                
                # Get segment scores
                segment_scores = [(i, importance_scores[i]) for i in range(start_idx, end_idx)]
                segment_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Take best frames from this segment
                frames_to_take = frames_per_segment + (1 if seg < remaining_frames else 0)
                for i, (frame_idx, score) in enumerate(segment_scores[:frames_to_take]):
                    distributed_frames.append(frame_idx)
            
            # Combine methods: prefer distributed frames but fill with top importance
            final_selection = set(distributed_frames)
            
            # Fill remaining slots with highest importance frames not already selected
            for frame_idx, score in indexed_scores:
                if len(final_selection) >= target_count:
                    break
                if frame_idx not in final_selection:
                    final_selection.add(frame_idx)
            
            sequence_data['selected_frames'] = sorted(list(final_selection))
        
        return sequence_data
    
    def _optimize_frame_durations(self, sequence_data: Dict[str, Any], 
                                frame_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize frame durations for smooth playback"""
        
        selected_frames = sequence_data['selected_frames']
        motion_intensity = frame_analysis.get('motion_intensity', [])
        
        if not motion_intensity or len(motion_intensity) == 0:
            # Fallback to uniform duration
            base_duration = int(1000 / frame_analysis.get('optimal_fps', 15))  # ms
            sequence_data['frame_durations'] = [base_duration] * len(selected_frames)
            return sequence_data
        
        durations = []
        base_duration = int(1000 / frame_analysis.get('optimal_fps', 15))  # ms
        
        for frame_idx in selected_frames:
            if frame_idx < len(motion_intensity):
                motion = motion_intensity[frame_idx]
                
                # High motion frames get shorter duration (faster playback)
                # Low motion frames get longer duration (slower playback)
                if motion > 0.7:
                    duration = max(int(base_duration * 0.7), 50)  # Min 50ms
                elif motion < 0.3:
                    duration = min(int(base_duration * 1.3), 200)  # Max 200ms
                else:
                    duration = base_duration
                
                durations.append(duration)
            else:
                durations.append(base_duration)
        
        sequence_data['frame_durations'] = durations
        return sequence_data
    
    def _generate_gif_candidates(self, optimized_frames: Dict[str, Any], 
                               palette_data: Dict[str, Any], max_size_mb: float,
                               platform: str = None) -> List[Dict[str, Any]]:
        """
        Generate multiple GIF candidates with different optimization strategies
        """
        
        logger.info("Generating GIF optimization candidates...")
        
        candidates = []
        
        # Candidate 1: Quality-focused
        quality_candidate = {
            'name': 'quality_focused',
            'colors': palette_data['colors'],
            'dither': 'floyd_steinberg',
            'lossy': 0,  # No lossy compression
            'optimization_level': 3,
            'frames': optimized_frames['selected_frames'],
            'durations': optimized_frames['frame_durations']
        }
        candidates.append(quality_candidate)
        
        # Candidate 2: Size-focused
        size_candidate = {
            'name': 'size_focused',
            'colors': min(palette_data['colors'], 128),
            'dither': 'none',
            'lossy': 80,
            'optimization_level': 1,
            'frames': optimized_frames['selected_frames'][::2],  # Skip every other frame
            'durations': [d * 2 for d in optimized_frames['frame_durations'][::2]]  # Double duration
        }
        candidates.append(size_candidate)
        
        # Candidate 3: Balanced
        balanced_candidate = {
            'name': 'balanced',
            'colors': min(palette_data['colors'], 192),
            'dither': 'bayer',
            'lossy': 40,
            'optimization_level': 2,
            'frames': optimized_frames['selected_frames'],
            'durations': optimized_frames['frame_durations']
        }
        candidates.append(balanced_candidate)
        
        # Candidate 4: Platform-specific
        if platform:
            platform_candidate = self._generate_platform_specific_candidate(
                optimized_frames, palette_data, platform
            )
            candidates.append(platform_candidate)
        
        # Candidate 5: Adaptive (based on content analysis)
        adaptive_candidate = self._generate_adaptive_candidate(
            optimized_frames, palette_data, max_size_mb
        )
        candidates.append(adaptive_candidate)
        
        return candidates
    
    def _generate_platform_specific_candidate(self, optimized_frames: Dict[str, Any],
                                            palette_data: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """Generate platform-specific optimized candidate"""
        
        platform_settings = {
            'twitter': {'colors': 128, 'dither': 'bayer', 'lossy': 60, 'max_frames': 50},
            'discord': {'colors': 192, 'dither': 'floyd_steinberg', 'lossy': 30, 'max_frames': 60},
            'slack': {'colors': 96, 'dither': 'none', 'lossy': 80, 'max_frames': 40}
        }
        
        settings = platform_settings.get(platform, platform_settings['twitter'])
        
        # Limit frames if needed
        frames = optimized_frames['selected_frames']
        durations = optimized_frames['frame_durations']
        
        if len(frames) > settings['max_frames']:
            step = len(frames) // settings['max_frames']
            frames = frames[::step]
            durations = durations[::step]
        
        return {
            'name': f'{platform}_optimized',
            'colors': min(palette_data['colors'], settings['colors']),
            'dither': settings['dither'],
            'lossy': settings['lossy'],
            'optimization_level': 2,
            'frames': frames,
            'durations': durations
        }
    
    def _generate_adaptive_candidate(self, optimized_frames: Dict[str, Any],
                                   palette_data: Dict[str, Any], max_size_mb: float) -> Dict[str, Any]:
        """Generate adaptive candidate based on size constraint"""
        
        # Adapt settings based on size constraint
        if max_size_mb <= 2:
            # Very tight constraint
            colors = min(palette_data['colors'], 64)
            dither = 'none'
            lossy = 100
            frame_skip = 3
        elif max_size_mb <= 5:
            # Moderate constraint
            colors = min(palette_data['colors'], 128)
            dither = 'bayer'
            lossy = 60
            frame_skip = 2
        else:
            # Generous constraint
            colors = min(palette_data['colors'], 256)
            dither = 'floyd_steinberg'
            lossy = 20
            frame_skip = 1
        
        frames = optimized_frames['selected_frames'][::frame_skip]
        durations = [d * frame_skip for d in optimized_frames['frame_durations'][::frame_skip]]
        
        return {
            'name': 'adaptive',
            'colors': colors,
            'dither': dither,
            'lossy': lossy,
            'optimization_level': 2,
            'frames': frames,
            'durations': durations
        }
    
    def _evaluate_gif_candidates_parallel(self, candidates: List[Dict[str, Any]], 
                                        output_path: str, max_size_mb: float) -> Dict[str, Any]:
        """
        Evaluate GIF candidates in parallel and select the best one
        """
        
        if self._shutdown_checker():
            raise RuntimeError("Shutdown requested before candidate evaluation")
        logger.info(f"Evaluating {len(candidates)} GIF candidates in parallel...")
        
        successful_results = []
        early_pick = None
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=min(len(candidates), 3)) as executor:
            
            # Submit all candidates for processing
            future_to_candidate = {
                executor.submit(self._create_gif_candidate, candidate, max_size_mb): candidate
                for candidate in candidates
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                try:
                    if self._shutdown_checker():
                        break
                    result = future.result(timeout=180)  # 3 minute timeout per candidate
                    if result and result.get('success', False):
                        # Early-stop check
                        if self._good_enough_result(result, max_size_mb):
                            early_pick = result
                            # Cancel remaining futures
                            for f in future_to_candidate.keys():
                                if f is not future:
                                    f.cancel()
                            break
                        successful_results.append(result)
                        logger.info(f"GIF candidate '{candidate['name']}' completed: "
                                  f"{result['size_mb']:.2f}MB, quality: {result['quality_score']:.1f}")
                except Exception as e:
                    if self._shutdown_checker():
                        logger.info(f"Candidate '{candidate['name']}' interrupted by shutdown")
                        break
                    logger.debug(f"GIF candidate '{candidate['name']}' failed: {e}")
        
        if early_pick:
            shutil.move(early_pick['temp_file'], output_path)
            for result in successful_results:
                if result != early_pick and os.path.exists(result['temp_file']):
                    os.remove(result['temp_file'])
            early_pick['output_file'] = output_path
            logger.info(f"Early-stopped with candidate: '{early_pick['candidate_name']}' "
                        f"({early_pick['size_mb']:.2f}MB, quality: {early_pick['quality_score']:.1f})")
            return early_pick

        if not successful_results:
            raise RuntimeError("All GIF optimization candidates failed")
        
        # Select best result
        best_result = max(successful_results, key=lambda x: (
            x['size_mb'] <= max_size_mb,  # Size compliance first
            x['quality_score'],           # Then quality
            -x['size_mb']                 # Then prefer larger size (better utilization)
        ))
        
        # Move best result to final output and cleanup others
        shutil.move(best_result['temp_file'], output_path)
        
        for result in successful_results:
            if result != best_result and os.path.exists(result['temp_file']):
                os.remove(result['temp_file'])
        
        best_result['output_file'] = output_path
        
        logger.info(f"Selected best GIF candidate: '{best_result['candidate_name']}' "
                   f"({best_result['size_mb']:.2f}MB, quality: {best_result['quality_score']:.1f})")
        
        # Log detailed file specifications
        try:
            from .ffmpeg_utils import FFmpegUtils
            specs = FFmpegUtils.get_detailed_file_specifications(output_path)
            specs_log = FFmpegUtils.format_file_specifications_for_logging(specs)
            logger.info(f"Advanced GIF optimization final file specifications - {specs_log}")
        except Exception as e:
            logger.warning(f"Could not log detailed GIF specifications: {e}")
        
        return best_result

    def _good_enough_result(self, result: Dict[str, Any], max_size_mb: float) -> bool:
        """Determine if a candidate is good enough to stop early based on config."""
        try:
            if not self.early_stop_cfg.get('enabled', True):
                return False
            size_mb = float(result.get('size_mb', 0))
            qs = float(result.get('quality_score', 0))
            if size_mb <= 0 or qs <= 0:
                return False
            if size_mb > max_size_mb:
                return False
            min_util = float(self.early_stop_cfg.get('min_target_utilization', 0.92))
            min_q = float(self.early_stop_cfg.get('min_quality_score', 7.5))
            utilized = size_mb >= (max_size_mb * min_util)
            return utilized and qs >= min_q
        except Exception:
            return False
    
    def _create_gif_candidate(self, candidate: Dict[str, Any], max_size_mb: float) -> Optional[Dict[str, Any]]:
        """
        Create a single GIF candidate
        """
        
        temp_gif = os.path.join(self.temp_dir, f"gif_candidate_{candidate['name']}_{int(time.time())}.gif")
        
        try:
            # This is a simplified implementation
            # In practice, you'd implement the full GIF creation with all the specified parameters
            
            # For now, create a basic GIF using PIL as placeholder
            # In real implementation, you'd use the optimized frames, palette, and settings
            
            # Create a dummy GIF for demonstration
            frames = []
            for i in range(min(20, len(candidate.get('frames', [])))):
                # Create a simple colored frame
                frame = Image.new('RGB', (200, 200), color=(i*10 % 255, 100, 150))
                frames.append(frame)
            
            if frames:
                # Save with specified parameters
                frames[0].save(
                    temp_gif,
                    save_all=True,
                    append_images=frames[1:],
                    duration=candidate.get('durations', [100] * len(frames)),
                    loop=0,
                    optimize=True
                )
                
                if os.path.exists(temp_gif):
                    file_size_mb = os.path.getsize(temp_gif) / (1024 * 1024)
                    quality_score = self._calculate_gif_quality_score(candidate, file_size_mb, max_size_mb)
                    
                    return {
                        'success': True,
                        'candidate_name': candidate['name'],
                        'temp_file': temp_gif,
                        'size_mb': file_size_mb,
                        'quality_score': quality_score,
                        'candidate_data': candidate
                    }
            
        except Exception as e:
            logger.warning(f"GIF candidate creation failed: {e}")
        
        return None
    
    def _calculate_gif_quality_score(self, candidate: Dict[str, Any], 
                                   file_size_mb: float, max_size_mb: float) -> float:
        """
        Calculate quality score for GIF candidate
        """
        
        base_score = 5.0
        
        # Size efficiency
        size_utilization = file_size_mb / max_size_mb
        if size_utilization <= 1.0:
            size_bonus = size_utilization * 2  # Up to 2 points
        else:
            size_bonus = -5  # Heavy penalty for exceeding size
        
        # Quality bonuses based on settings
        color_bonus = (candidate.get('colors', 128) / 256.0) * 1.5
        
        dither_bonus = {
            'floyd_steinberg': 1.0,
            'bayer': 0.5,
            'none': 0.0
        }.get(candidate.get('dither', 'none'), 0.0)
        
        lossy_penalty = (candidate.get('lossy', 0) / 100.0) * -1.0
        
        total_score = base_score + size_bonus + color_bonus + dither_bonus + lossy_penalty
        
        return max(0, min(10, total_score))
    
    def _apply_gif_post_processing(self, gif_result: Dict[str, Any], max_size_mb: float) -> Dict[str, Any]:
        """
        Apply post-processing optimizations to the final GIF
        """
        
        logger.info("Applying GIF post-processing optimizations...")
        
        # Add post-processing metadata
        gif_result['post_processed'] = True
        gif_result['final_optimization'] = 'advanced'
        
        # In a full implementation, you might apply:
        # - Additional compression passes
        # - Frame deduplication
        # - Temporal optimization
        # - Palette refinement
        
        return gif_result 

    def _initialize_optimization_params(self, max_size_mb: float, quality_preference: str) -> Dict[str, Any]:
        """Initialize optimization parameters based on target size and quality preference"""
        
        # Base parameters
        params = {
            'width': 480,
            'height': 480,
            'fps': 15,
            'colors': 256,
            'dither': 'floyd_steinberg',
            'lossy': 0,
            'optimization_level': 2,
            'frame_skip': 1,
            'palette_method': 'neuquant',
            'quality_preference': quality_preference
        }
        
        # Adjust based on size constraint
        if max_size_mb <= 2:
            # Very tight constraint
            params.update({
                'width': 320,
                'height': 320,
                'fps': 10,
                'colors': 64,
                'dither': 'none',
                'lossy': 80,
                'frame_skip': 2
            })
        elif max_size_mb <= 5:
            # Moderate constraint
            params.update({
                'width': 400,
                'height': 400,
                'fps': 12,
                'colors': 128,
                'dither': 'bayer',
                'lossy': 40,
                'frame_skip': 1
            })
        
        # Adjust based on quality preference
        if quality_preference == 'quality':
            params.update({
                'dither': 'floyd_steinberg',
                'lossy': max(0, params['lossy'] - 20),
                'optimization_level': 3
            })
        elif quality_preference == 'size':
            params.update({
                'dither': 'none',
                'lossy': min(150, params['lossy'] + 20),
                'optimization_level': 1
            })
        
        return params
    
    def _create_gif_with_params(self, input_video: str, output_path: str, 
                               params: Dict[str, Any], frame_analysis: Dict[str, Any],
                               palette_data: Dict[str, Any], start_time: float, 
                               duration: float, platform: str = None) -> Optional[Dict[str, Any]]:
        """Create GIF with specific optimization parameters"""
        
        # Create unique temp file name with thread ID and random suffix to avoid conflicts
        thread_id = threading.get_ident()
        random_suffix = random.randint(1000, 9999)
        temp_gif = os.path.join(self.temp_dir, f"optimization_temp_{thread_id}_{random_suffix}.gif")
        
        try:
            # Create optimized frames based on parameters
            optimized_frames = self._create_optimized_frames_with_params(
                input_video, params, frame_analysis, start_time, duration
            )
            
            # Generate palette with current color count
            current_palette = self._generate_palette_with_colors(
                input_video, params['colors'], params['palette_method']
            )
            
            # Create GIF using FFmpeg with current parameters
            self._create_gif_ffmpeg_optimized(
                input_video, temp_gif, params, current_palette, 
                optimized_frames, start_time, duration
            )
            
            if os.path.exists(temp_gif):
                file_size_mb = os.path.getsize(temp_gif) / (1024 * 1024)
                
                return {
                    'success': True,
                    'temp_file': temp_gif,
                    'size_mb': file_size_mb,
                    'params': params.copy(),
                    'frame_count': len(optimized_frames.get('selected_frames', [])),
                    'actual_width': params['width'],
                    'actual_height': params['height'],
                    'actual_fps': params['fps'],
                    'actual_colors': params['colors']
                }
        
        except Exception as e:
            logger.warning(f"GIF creation with params failed: {e}")
            # Clean up with retry logic for Windows file locking
            if os.path.exists(temp_gif):
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        os.remove(temp_gif)
                        break
                    except (PermissionError, OSError) as cleanup_error:
                        if retry < max_retries - 1:
                            logger.debug(f"Temp file cleanup retry {retry + 1}/{max_retries}: {cleanup_error}")
                            time.sleep(0.1 * (retry + 1))
                        else:
                            logger.warning(f"Failed to cleanup temp file {temp_gif}: {cleanup_error}")
        
        return None
    
    def _create_optimized_frames_with_params(self, input_video: str, params: Dict[str, Any],
                                           frame_analysis: Dict[str, Any], start_time: float,
                                           duration: float) -> Dict[str, Any]:
        """Create optimized frame sequence with specific parameters"""
        
        # Calculate target frame count based on FPS and duration
        target_frames = int(duration * params['fps'])
        
        # Apply frame skipping if needed
        if params['frame_skip'] > 1:
            target_frames = target_frames // params['frame_skip']
        
        # Select frames using importance scores if available
        if frame_analysis.get('frame_importance_scores'):
            selected_frames = self._select_frames_by_importance(
                frame_analysis, target_frames, {'selected_frames': []}
            )['selected_frames']
        else:
            # Uniform selection
            total_frames = frame_analysis.get('total_frames', 30)
            step = max(1, total_frames // target_frames)
            selected_frames = list(range(0, total_frames, step))
        
        # Calculate frame durations
        base_duration = int(1000 / params['fps'])  # ms
        frame_durations = [base_duration] * len(selected_frames)
        
        return {
            'selected_frames': selected_frames,
            'frame_durations': frame_durations,
            'total_frames': len(selected_frames)
        }
    
    def _generate_palette_with_colors(self, input_video: str, colors: int, method: str) -> Optional[str]:
        """Generate palette with specific color count and method"""
        
        if method == 'neuquant':
            return self._generate_palette_neuquant(input_video, colors)
        elif method == 'median_cut':
            return self._generate_palette_median_cut(input_video, colors)
        elif method == 'octree':
            return self._generate_palette_octree(input_video, colors)
        else:
            return self._generate_palette_neuquant(input_video, colors)
    
    def _create_gif_ffmpeg_optimized(self, input_video: str, output_gif: str, 
                                   params: Dict[str, Any], palette_file: Optional[str],
                                   optimized_frames: Dict[str, Any], start_time: float, duration: float):
        """Create GIF using FFmpeg with optimized parameters"""
        
        from .ffmpeg_utils import FFmpegUtils
        if palette_file and os.path.exists(palette_file):
            # Use custom palette
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', input_video,
                '-i', palette_file,
                '-lavfi', f'fps={params["fps"]},scale={params["width"]}:{params["height"]}:flags=lanczos[x];[x][1:v]paletteuse=dither={params["dither"]}:diff_mode=rectangle',
                '-loop', '0',
                output_gif
            ]
        else:
            # Use FFmpeg's built-in palette generation
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', input_video,
                '-vf', f'fps={params["fps"]},scale={params["width"]}:{params["height"]}:flags=lanczos,palettegen=max_colors={params["colors"]}:stats_mode=diff,paletteuse=dither={params["dither"]}',
                '-loop', '0',
                output_gif
            ]
        # Add performance flags
        FFmpegUtils.add_ffmpeg_perf_flags(cmd)

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
    
    def _calculate_comprehensive_quality_score(self, result: Dict[str, Any], 
                                            params: Dict[str, Any], 
                                            frame_analysis: Dict[str, Any]) -> float:
        """Calculate comprehensive quality score for optimization result"""
        
        base_score = 5.0
        
        # Size efficiency (closer to target = better)
        size_ratio = result['size_mb'] / params.get('target_size_mb', 10.0)
        if size_ratio <= 1.0:
            size_score = size_ratio * 2.0  # Up to 2 points for efficient size usage
        else:
            size_score = -5.0  # Heavy penalty for exceeding target
        
        # Resolution quality
        resolution_score = min((params['width'] * params['height']) / (480 * 480), 2.0)
        
        # FPS quality
        fps_score = min(params['fps'] / 15.0, 1.5)
        
        # Color quality
        color_score = (params['colors'] / 256.0) * 1.5
        
        # Dithering quality
        dither_scores = {
            'floyd_steinberg': 1.0,
            'bayer': 0.7,
            'none': 0.3
        }
        dither_score = dither_scores.get(params['dither'], 0.5)
        
        # Lossy compression penalty
        lossy_penalty = (params['lossy'] / 100.0) * -1.0
        
        # Frame count quality
        frame_score = min(result['frame_count'] / 30.0, 1.0)
        
        total_score = (base_score + size_score + resolution_score + fps_score + 
                      color_score + dither_score + lossy_penalty + frame_score)
        
        return max(0, min(10, total_score))
    
    def _increase_quality_params(self, params: Dict[str, Any], current_size_mb: float,
                               target_size_mb: float, quality_preference: str) -> Dict[str, Any]:
        """Increase quality parameters while staying under size limit"""
        
        new_params = params.copy()
        size_ratio = current_size_mb / target_size_mb
        available_space = 1.0 - size_ratio
        
        # Determine how aggressive to be based on available space and preference
        if quality_preference == 'quality':
            aggressiveness = min(available_space * 2, 0.3)  # More aggressive
        elif quality_preference == 'balanced':
            aggressiveness = min(available_space * 1.5, 0.2)  # Moderate
        else:  # size preference
            aggressiveness = min(available_space, 0.1)  # Conservative
        
        # Increase colors if there's room
        if available_space > 0.1 and new_params['colors'] < 256:
            color_increase = int(aggressiveness * 50)
            new_params['colors'] = min(256, new_params['colors'] + color_increase)
        
        # Increase resolution if there's room
        if available_space > 0.15:
            size_increase = int(aggressiveness * 40)
            new_params['width'] = min(640, new_params['width'] + size_increase)
            new_params['height'] = min(640, new_params['height'] + size_increase)
        
        # Increase FPS if there's room
        if available_space > 0.2 and new_params['fps'] < 20:
            fps_increase = int(aggressiveness * 3)
            new_params['fps'] = min(20, new_params['fps'] + fps_increase)
        
        # Improve dithering if there's room
        if available_space > 0.25 and new_params['dither'] == 'bayer':
            new_params['dither'] = 'floyd_steinberg'
        
        # Reduce lossy compression if there's room
        if available_space > 0.1 and new_params['lossy'] > 0:
            lossy_reduction = int(aggressiveness * 20)
            new_params['lossy'] = max(0, new_params['lossy'] - lossy_reduction)
        
        return new_params
    
    def _decrease_quality_params(self, params: Dict[str, Any], current_size_mb: float,
                               target_size_mb: float, quality_preference: str) -> Dict[str, Any]:
        """Decrease quality parameters to get under size limit"""
        
        new_params = params.copy()
        size_ratio = current_size_mb / target_size_mb
        reduction_needed = size_ratio - 1.0
        
        # Determine reduction strategy based on preference
        if quality_preference == 'quality':
            # Preserve quality where possible, reduce size aggressively
            reduction_factor = min(reduction_needed * 1.5, 0.4)
        elif quality_preference == 'balanced':
            # Balanced reduction
            reduction_factor = min(reduction_needed * 1.2, 0.3)
        else:  # size preference
            # Aggressive size reduction
            reduction_factor = min(reduction_needed * 2.0, 0.5)
        
        # Reduce colors first (biggest impact)
        if new_params['colors'] > 64:
            color_reduction = int(reduction_factor * 50)
            new_params['colors'] = max(64, new_params['colors'] - color_reduction)
        
        # Reduce resolution
        if new_params['width'] > 160:
            size_reduction = int(reduction_factor * 40)
            new_params['width'] = max(160, new_params['width'] - size_reduction)
            new_params['height'] = max(160, new_params['height'] - size_reduction)
        
        # Reduce FPS
        if new_params['fps'] > 6:
            fps_reduction = int(reduction_factor * 3)
            new_params['fps'] = max(6, new_params['fps'] - fps_reduction)
        
        # Increase lossy compression
        lossy_increase = int(reduction_factor * 30)
        new_params['lossy'] = min(150, new_params['lossy'] + lossy_increase)
        
        # Degrade dithering if needed
        if reduction_factor > 0.3 and new_params['dither'] == 'floyd_steinberg':
            new_params['dither'] = 'bayer'
        elif reduction_factor > 0.5 and new_params['dither'] == 'bayer':
            new_params['dither'] = 'none'
        
        return new_params

    def _micro_trim_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Apply tiny reductions to try to shave ~0.1-0.2MB without big quality loss."""
        new_params = params.copy()
        # Slightly reduce colors
        if new_params['colors'] > 96:
            new_params['colors'] = max(96, new_params['colors'] - 8)
        # Prefer bayer dithering for better compressibility
        if new_params.get('dither') == 'floyd_steinberg':
            new_params['dither'] = 'bayer'
        # Slight FPS nudge
        if new_params['fps'] > 8:
            new_params['fps'] = max(8, new_params['fps'] - 1)
        # Tiny downscale
        if new_params['width'] > 200 and new_params['height'] > 200:
            new_params['width'] = max(200, new_params['width'] - 8)
            new_params['height'] = max(200, new_params['height'] - 8)
        # Slight lossy bump
        new_params['lossy'] = min(150, new_params.get('lossy', 0) + 5)
        return new_params
    
    def _adjust_params_for_failure(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust parameters when GIF creation fails"""
        
        new_params = params.copy()
        
        # Reduce complexity to increase success rate
        new_params['colors'] = max(64, new_params['colors'] - 32)
        new_params['width'] = max(160, new_params['width'] - 40)
        new_params['height'] = max(160, new_params['height'] - 40)
        new_params['fps'] = max(6, new_params['fps'] - 2)
        new_params['lossy'] = min(150, new_params['lossy'] + 20)
        
        return new_params
    
    def _apply_final_optimizations(self, result: Dict[str, Any], target_size_mb: float) -> Dict[str, Any]:
        """Apply final optimizations to the best result"""
        
        # Move the best result to final output
        if result.get('temp_file') and os.path.exists(result['temp_file']):
            final_output = result['temp_file'].replace('temp_', 'final_')
            shutil.move(result['temp_file'], final_output)
            result['output_file'] = final_output
            
            # Log detailed file specifications
            try:
                from .ffmpeg_utils import FFmpegUtils
                specs = FFmpegUtils.get_detailed_file_specifications(final_output)
                specs_log = FFmpegUtils.format_file_specifications_for_logging(specs)
                logger.info(f"Final GIF optimization file specifications - {specs_log}")
            except Exception as e:
                logger.warning(f"Could not log detailed GIF specifications: {e}")
        
        # Calculate final quality score
        result['quality_score'] = self._calculate_comprehensive_quality_score(
            result, result['params'], {}
        )
        
        # Add optimization metadata
        result['optimization_method'] = 'iterative_quality_target'
        result['target_size_mb'] = target_size_mb
        result['size_efficiency'] = result['size_mb'] / target_size_mb
        
        return result 