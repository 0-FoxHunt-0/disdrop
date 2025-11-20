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

from ..hardware_detector import HardwareDetector
from ..config_manager import ConfigManager
from .advanced_optimizer import AdvancedVideoOptimizer
from .performance_enhancer import PerformanceEnhancer
from ..ffmpeg_utils import FFmpegUtils
from .video_segmenter import VideoSegmenter
from .quality_scorer import QualityScorer
from .bitrate_validator import BitrateValidator, BitrateValidationError
from .adaptive_parameter_adjuster import AdaptiveParameterAdjuster
from .compression_strategy import SmartCompressionStrategy, CompressionStrategy, CompressionConstraints
from .quality_improvement_engine import QualityImprovementEngine, FailureAnalysis, ImprovementPlan, CompressionParams as QICompressionParams
from ..utils.segments_summary import write_segments_summary
# FastQualityEstimator replaced by unified QualityPredictor
from .performance_monitor import PerformanceMonitor
from .performance_controls import PerformanceController

logger = logging.getLogger(__name__)

class GracefulCancellation(Exception):
    """Raised to indicate a user-requested shutdown/cancellation."""
    pass


class DynamicVideoCompressor:
    def __init__(self, config_manager: ConfigManager, hardware_detector: HardwareDetector):
        self.config = config_manager
        self.hardware = hardware_detector
        self.temp_dir = self.config.get_temp_dir()
        
        # Performance enhancement and monitoring
        self.performance_enhancer = PerformanceEnhancer(config_manager)
        self.performance_controller = PerformanceController(config_manager)
        self.performance_monitor = self.performance_controller.monitor
        
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
        
        # Initialize bitrate validation components
        self.bitrate_validator = BitrateValidator(config_manager)
        self.parameter_adjuster = AdaptiveParameterAdjuster(config_manager, self.bitrate_validator)
        
        # Initialize smart compression strategy
        self.compression_strategy = SmartCompressionStrategy(config_manager)
        
        # Optimize system resources
        self.system_optimizations = self.performance_enhancer.optimize_system_resources()
        
        # Cache for expensive calculations
        self._resolution_cache = {}
        
        # Enhanced debug logging configuration
        self.debug_logging_enabled = self._is_debug_logging_enabled()
        self.performance_logging_enabled = self._is_performance_logging_enabled()
        
        # Compression session tracking for detailed logging
        self._current_session = None
        self._session_counter = 0

    # ===== New helpers for refined final-pass encoding =====
    def _calculate_target_video_kbps(self, target_size_mb: float, duration_s: float, audio_kbps: int) -> int:
        """Calculate target video bitrate for given constraints"""
        total_bits = target_size_mb * 8 * 1024 * 1024
        video_kbps = int(max((total_bits / max(duration_s, 1.0) / 1000) - audio_kbps, 64))
        return video_kbps

    def _calculate_safe_bitrate_cap(self, width: int, height: int, fps: float, duration: float) -> int:
        """Calculate reasonable bitrate cap based on video characteristics"""
        pixels_per_sec = width * height * fps
        # Cap at 0.15 bits-per-pixel (high quality ceiling)
        max_bpp = 0.15
        cap = int(pixels_per_sec * max_bpp / 1000)
        # Absolute bounds: 500k-8000k
        return max(500, min(cap, 8000))
    
    def _is_debug_logging_enabled(self) -> bool:
        """Check if comprehensive debug logging is enabled."""
        if self.config:
            # Check if explicitly enabled in config or if logger is at DEBUG level
            config_enabled = self.config.get('video_compression.debug_logging.enabled', False)
            logger_debug = logger.isEnabledFor(logging.DEBUG)
            return config_enabled or logger_debug
        return logger.isEnabledFor(logging.DEBUG)
    
    def _is_performance_logging_enabled(self) -> bool:
        """Check if performance metrics logging is enabled."""
        if self.config:
            # Check both general performance metrics and specific performance logging settings
            general_enabled = self.config.get('video_compression.debug_logging.performance_metrics', False)
            specific_enabled = self.config.get('video_compression.debug_logging.performance_logging.operation_timing', False)
            return general_enabled or specific_enabled
        return False
    
    def _start_compression_session(self, input_path: str, target_size_mb: float, 
                                  platform: str = None) -> Dict[str, Any]:
        """Start a new compression session with detailed logging context."""
        self._session_counter += 1
        session_id = f"session_{self._session_counter}_{int(time.time())}"
        
        session_info = {
            'session_id': session_id,
            'input_path': input_path,
            'input_filename': os.path.basename(input_path),
            'target_size_mb': target_size_mb,
            'platform': platform,
            'start_time': time.time(),
            'parameters_history': [],
            'decisions_log': [],
            'performance_metrics': {},
            'quality_evaluations': [],
            'fallback_attempts': []
        }
        
        self._current_session = session_info
        
        if self.debug_logging_enabled:
            logger.info(f"=== COMPRESSION SESSION STARTED ===")
            logger.info(f"Session ID: {session_id}")
            logger.info(f"Input: {os.path.basename(input_path)}")
            logger.info(f"Target size: {target_size_mb:.2f}MB")
            logger.info(f"Platform: {platform or 'generic'}")
            logger.info(f"Debug logging: enabled")
        
        return session_info
    
    def _log_compression_decision(self, decision_type: str, decision: str, 
                                 context: Dict[str, Any] = None, 
                                 justification: str = None) -> None:
        """Log a compression parameter decision with detailed context and justification."""
        if not self.debug_logging_enabled or not self._current_session:
            return
        
        decision_entry = {
            'timestamp': time.time(),
            'decision_type': decision_type,
            'decision': decision,
            'context': context or {},
            'justification': justification
        }
        
        self._current_session['decisions_log'].append(decision_entry)
        
        # Log the decision
        logger.info(f"DECISION [{decision_type}]: {decision}")
        if justification:
            logger.info(f"  Justification: {justification}")
        
        if context and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"  Context: {context}")
    
    def _log_parameter_change(self, parameter_name: str, old_value: Any, 
                             new_value: Any, reason: str) -> None:
        """Log parameter changes with detailed reasoning."""
        if not self.debug_logging_enabled or not self._current_session:
            return
        
        change_entry = {
            'timestamp': time.time(),
            'parameter': parameter_name,
            'old_value': old_value,
            'new_value': new_value,
            'reason': reason
        }
        
        self._current_session['parameters_history'].append(change_entry)
        
        logger.info(f"PARAMETER CHANGE [{parameter_name}]: {old_value} → {new_value}")
        logger.info(f"  Reason: {reason}")
    
    def _log_fps_reduction_justification(self, original_fps: float, new_fps: float, 
                                        video_info: Dict[str, Any], 
                                        strategy_analysis: Dict[str, Any]) -> None:
        """Log detailed FPS reduction justification with comprehensive analysis."""
        if not self.debug_logging_enabled:
            return
        
        reduction_percent = ((original_fps - new_fps) / original_fps) * 100
        
        logger.info(f"=== FPS REDUCTION ANALYSIS ===")
        logger.info(f"FPS change: {original_fps:.1f} → {new_fps:.1f} fps ({reduction_percent:.1f}% reduction)")
        
        # Log video characteristics that influenced the decision
        motion_level = video_info.get('motion_level', 'unknown')
        duration = video_info.get('duration', 0)
        original_size_mb = video_info.get('size_bytes', 0) / (1024 * 1024) if video_info.get('size_bytes') else 0
        
        logger.info(f"Video characteristics:")
        logger.info(f"  Motion level: {motion_level}")
        logger.info(f"  Duration: {duration:.1f}s")
        logger.info(f"  Original size: {original_size_mb:.2f}MB")
        
        # Log strategy analysis
        if strategy_analysis:
            fps_impact = strategy_analysis.get('fps_impact', {})
            logger.info(f"Impact assessment:")
            logger.info(f"  Impact level: {fps_impact.get('impact_level', 'unknown')}")
            logger.info(f"  Impact score: {fps_impact.get('impact_score', 0)}")
            logger.info(f"  Recommendation: {fps_impact.get('recommendation', 'none')}")
            
            if strategy_analysis.get('prefer_bitrate_reduction'):
                logger.info(f"Alternative strategy analysis:")
                logger.info(f"  Bitrate headroom: {strategy_analysis.get('bitrate_headroom', 0)}kbps")
                logger.info(f"  Potential size reduction: {strategy_analysis.get('potential_size_reduction_mb', 0):.2f}MB")
                logger.info(f"  Strategy reason: {strategy_analysis.get('strategy_reason', 'none')}")
        
        # Log configuration constraints
        config_min_fps = self.config.get('video_compression.bitrate_validation.min_fps', 20)
        fps_reduction_steps = self.config.get('video_compression.bitrate_validation.fps_reduction_steps', [])
        
        logger.info(f"Configuration constraints:")
        logger.info(f"  Minimum FPS (config): {config_min_fps}")
        logger.info(f"  FPS reduction steps: {fps_reduction_steps}")
        logger.info(f"  Respects minimum: {new_fps >= config_min_fps}")
        
        # Log FPS reduction strategy and reasoning
        if reduction_percent > 0:
            if reduction_percent <= 20:
                impact_level = "minimal"
                quality_impact = "negligible quality impact expected"
            elif reduction_percent <= 40:
                impact_level = "moderate"
                quality_impact = "slight quality impact, should maintain smoothness"
            elif reduction_percent <= 60:
                impact_level = "significant"
                quality_impact = "noticeable quality impact, may affect smoothness"
            else:
                impact_level = "aggressive"
                quality_impact = "substantial quality impact, choppy playback likely"
            
            logger.info(f"FPS reduction strategy:")
            logger.info(f"  Impact level: {impact_level}")
            logger.info(f"  Quality impact: {quality_impact}")
            
            # Log alternative strategies considered
            if strategy_analysis and strategy_analysis.get('alternatives_considered'):
                alternatives = strategy_analysis['alternatives_considered']
                logger.info(f"Alternative strategies considered:")
                for alt in alternatives:
                    logger.info(f"  - {alt.get('strategy', 'unknown')}: {alt.get('reason', 'no reason given')}")
        
        logger.info(f"=== END FPS REDUCTION ANALYSIS ===")
    
    def _log_configuration_loading(self, config_section: str, loaded_values: Dict[str, Any]) -> None:
        """Log configuration loading and application steps."""
        if not self.debug_logging_enabled:
            return
        
        logger.debug(f"=== CONFIGURATION LOADING [{config_section}] ===")
        for key, value in loaded_values.items():
            logger.debug(f"  {key}: {value}")
        logger.debug(f"=== END CONFIGURATION LOADING ===")
    
    def _log_performance_metrics(self, operation: str, metrics: Dict[str, Any]) -> None:
        """Log performance metrics for compression operations."""
        if not self.performance_logging_enabled or not self._current_session:
            return
        
        # Add to session metrics
        if operation not in self._current_session['performance_metrics']:
            self._current_session['performance_metrics'][operation] = []
        
        metrics_entry = {
            'timestamp': time.time(),
            'metrics': metrics.copy()
        }
        self._current_session['performance_metrics'][operation].append(metrics_entry)
        
        # Log the metrics
        logger.info(f"PERFORMANCE [{operation}]:")
        for key, value in metrics.items():
            if isinstance(value, float):
                logger.info(f"  {key}: {value:.3f}")
            else:
                logger.info(f"  {key}: {value}")
    
    def _measure_operation_performance(self, operation_name: str):
        """Context manager to measure and log operation performance."""
        class PerformanceMeasurer:
            def __init__(self, compressor, operation_name):
                self.compressor = compressor
                self.operation_name = operation_name
                self.start_time = None
                self.start_memory = None
                
            def __enter__(self):
                self.start_time = time.time()
                try:
                    import psutil
                    process = psutil.Process()
                    self.start_memory = process.memory_info().rss / (1024 * 1024)  # MB
                except ImportError:
                    self.start_memory = None
                return self
                
            def __exit__(self, exc_type, exc_val, exc_tb):
                end_time = time.time()
                duration = end_time - self.start_time
                
                metrics = {
                    'duration': duration,
                    'start_time': self.start_time,
                    'end_time': end_time
                }
                
                if self.start_memory is not None:
                    try:
                        import psutil
                        process = psutil.Process()
                        end_memory = process.memory_info().rss / (1024 * 1024)  # MB
                        metrics['memory_start_mb'] = self.start_memory
                        metrics['memory_end_mb'] = end_memory
                        metrics['memory_delta_mb'] = end_memory - self.start_memory
                    except ImportError:
                        pass
                
                if exc_type is not None:
                    metrics['error'] = str(exc_val) if exc_val else 'Unknown error'
                    metrics['success'] = False
                else:
                    metrics['success'] = True
                
                self.compressor._log_performance_metrics(self.operation_name, metrics)
        
        return PerformanceMeasurer(self, operation_name)
    
    def _extract_encoder_info_from_cmd(self, cmd: List[str]) -> Dict[str, str]:
        """Extract encoder information from FFmpeg command for logging."""
        info = {}
        
        try:
            # Find encoder
            if '-c:v' in cmd:
                encoder_idx = cmd.index('-c:v') + 1
                if encoder_idx < len(cmd):
                    info['encoder'] = cmd[encoder_idx]
            
            # Find preset
            if '-preset' in cmd:
                preset_idx = cmd.index('-preset') + 1
                if preset_idx < len(cmd):
                    info['preset'] = cmd[preset_idx]
            
            # Determine acceleration type
            encoder = info.get('encoder', '')
            if 'nvenc' in encoder:
                info['acceleration'] = 'NVIDIA NVENC'
            elif 'amf' in encoder:
                info['acceleration'] = 'AMD AMF'
            elif 'qsv' in encoder:
                info['acceleration'] = 'Intel QSV'
            elif 'videotoolbox' in encoder:
                info['acceleration'] = 'Apple VideoToolbox'
            elif encoder in ['libx264', 'libx265']:
                info['acceleration'] = 'software'
            else:
                info['acceleration'] = 'unknown'
                
        except (ValueError, IndexError):
            pass
        
        return info
    
    def _end_compression_session(self, success: bool, output_path: str = None, 
                                final_size_mb: float = None, error: str = None) -> None:
        """End compression session and log comprehensive summary."""
        if not self._current_session:
            return
        
        session_duration = time.time() - self._current_session['start_time']
        self._current_session['end_time'] = time.time()
        self._current_session['duration'] = session_duration
        self._current_session['success'] = success
        self._current_session['final_size_mb'] = final_size_mb
        self._current_session['error'] = error
        
        if self.debug_logging_enabled:
            logger.info(f"=== COMPRESSION SESSION COMPLETED ===")
            logger.info(f"Session ID: {self._current_session['session_id']}")
            logger.info(f"Duration: {session_duration:.2f}s")
            logger.info(f"Success: {success}")
            
            if success and final_size_mb:
                target_size = self._current_session['target_size_mb']
                utilization = (final_size_mb / target_size) * 100
                logger.info(f"Final size: {final_size_mb:.2f}MB (target: {target_size:.2f}MB, {utilization:.1f}% utilization)")
            
            if error:
                logger.info(f"Error: {error}")
            
            # Log decision summary
            decisions_count = len(self._current_session['decisions_log'])
            parameter_changes = len(self._current_session['parameters_history'])
            logger.info(f"Decisions made: {decisions_count}")
            logger.info(f"Parameter changes: {parameter_changes}")
            
            # Log performance summary if enabled
            if self.performance_logging_enabled:
                perf_metrics = self._current_session['performance_metrics']
                logger.info(f"Performance operations logged: {len(perf_metrics)}")
                
                total_operation_time = 0
                for operation, metrics_list in perf_metrics.items():
                    if metrics_list:
                        durations = [m['metrics'].get('duration', 0) for m in metrics_list]
                        avg_duration = sum(durations) / len(durations)
                        max_duration = max(durations)
                        min_duration = min(durations)
                        total_duration = sum(durations)
                        total_operation_time += total_duration
                        
                        logger.info(f"  {operation}: {len(metrics_list)} operations")
                        logger.info(f"    Total time: {total_duration:.2f}s")
                        logger.info(f"    Average: {avg_duration:.2f}s")
                        logger.info(f"    Range: {min_duration:.2f}s - {max_duration:.2f}s")
                        
                        # Log memory usage if available
                        memory_deltas = [m['metrics'].get('memory_delta_mb', 0) for m in metrics_list if 'memory_delta_mb' in m['metrics']]
                        if memory_deltas:
                            avg_memory_delta = sum(memory_deltas) / len(memory_deltas)
                            max_memory_delta = max(memory_deltas)
                            logger.info(f"    Memory usage: avg {avg_memory_delta:+.1f}MB, peak {max_memory_delta:+.1f}MB")
                
                # Log overall performance metrics
                if total_operation_time > 0:
                    efficiency = (session_duration / total_operation_time) * 100 if total_operation_time > 0 else 0
                    logger.info(f"Performance summary:")
                    logger.info(f"  Total session time: {session_duration:.2f}s")
                    logger.info(f"  Total operation time: {total_operation_time:.2f}s")
                    logger.info(f"  Efficiency: {efficiency:.1f}% (operation time / session time)")
                    
                    if efficiency < 50:
                        logger.info(f"  Note: Low efficiency may indicate overhead or waiting time")
                    elif efficiency > 90:
                        logger.info(f"  Note: High efficiency indicates optimal resource utilization")
            
            logger.info(f"=== END SESSION SUMMARY ===")
        
        # Clear current session
        self._current_session = None
    
    def _cleanup_two_pass_logs(self, log_base: str):
        """Clean up all two-pass log files reliably with retry logic"""
        import time
        for ext in ['-0.log', '-0.log.mbtree', '-0.log.temp', '-0.log.mbtree.temp']:
            log_path = f"{log_base}{ext}"
            if not os.path.exists(log_path):
                continue
            
            # Retry logic with exponential backoff for locked files
            max_retries = 3
            retry_delay = 0.1  # Start with 100ms
            for attempt in range(max_retries):
                try:
                    os.remove(log_path)
                    logger.debug(f"Cleaned up temp file: {os.path.basename(log_path)}")
                    break
                except (OSError, PermissionError) as e:
                    if attempt < max_retries - 1:
                        # Wait before retrying with exponential backoff
                        time.sleep(retry_delay * (attempt + 1))
                    else:
                        # Last attempt failed, log but don't raise
                        logger.debug(f"Could not remove {log_path} after {max_retries} attempts: {e}")
                except Exception as e:
                    logger.debug(f"Could not remove {log_path}: {e}")
                    break
    
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
        """Calculate target video bitrate for given constraints with detailed logging"""
        total_bits = target_size_mb * 8 * 1024 * 1024
        total_kbps = total_bits / max(duration_s, 1.0) / 1000
        video_kbps = int(max(total_kbps - audio_kbps, 64))
        
        if self.debug_logging_enabled:
            logger.info(f"=== VIDEO BITRATE CALCULATION ===")
            logger.info(f"Input parameters:")
            logger.info(f"  Target size: {target_size_mb:.2f}MB")
            logger.info(f"  Duration: {duration_s:.1f}s")
            logger.info(f"  Audio bitrate: {audio_kbps}kbps")
            logger.info(f"Calculation:")
            logger.info(f"  Total bits: {total_bits:,.0f} bits")
            logger.info(f"  Total bitrate: {total_kbps:.1f}kbps")
            logger.info(f"  Video bitrate: {total_kbps:.1f} - {audio_kbps} = {video_kbps}kbps")
            
            # Log bitrate allocation
            audio_percentage = (audio_kbps / total_kbps) * 100 if total_kbps > 0 else 0
            video_percentage = (video_kbps / total_kbps) * 100 if total_kbps > 0 else 0
            logger.info(f"Bitrate allocation:")
            logger.info(f"  Audio: {audio_percentage:.1f}% ({audio_kbps}kbps)")
            logger.info(f"  Video: {video_percentage:.1f}% ({video_kbps}kbps)")
            
            # Log quality assessment
            if video_kbps < 200:
                quality_assessment = "very low quality expected, severe compression artifacts likely"
            elif video_kbps < 500:
                quality_assessment = "low quality expected, noticeable compression artifacts"
            elif video_kbps < 1000:
                quality_assessment = "medium quality expected, acceptable for most content"
            elif video_kbps < 2000:
                quality_assessment = "good quality expected, minimal compression artifacts"
            else:
                quality_assessment = "high quality expected, excellent visual fidelity"
            
            logger.info(f"Quality assessment: {quality_assessment}")
            logger.info(f"=== END VIDEO BITRATE CALCULATION ===")
        
        return video_kbps
    
    def _calculate_optimal_fps(self, video_info: Dict[str, Any], target_size_mb: float, strategy: str = 'balanced') -> float:
        """Determine best FPS based on motion, size constraints, and strategy"""
        original_fps = video_info.get('fps', 30.0)
        motion_level = video_info.get('motion_level', 'medium')
        duration = video_info.get('duration', 0)
        original_size_mb = video_info.get('size_bytes', 0) / (1024 * 1024)
        
        # Calculate compression ratio to gauge difficulty
        compression_ratio = original_size_mb / max(target_size_mb, 0.1) if original_size_mb > 0 else 1.0
        
        if self.debug_logging_enabled:
            logger.info(f"=== FPS CALCULATION ===")
            logger.info(f"Input parameters:")
            logger.info(f"  Original FPS: {original_fps:.1f}")
            logger.info(f"  Motion level: {motion_level}")
            logger.info(f"  Duration: {duration:.1f}s")
            logger.info(f"  Original size: {original_size_mb:.2f}MB")
            logger.info(f"  Target size: {target_size_mb:.2f}MB")
            logger.info(f"  Compression ratio: {compression_ratio:.1f}x")
            logger.info(f"  Strategy: {strategy}")
        
        # Base FPS selection based on motion level
        if motion_level == 'high':
            base_fps = min(original_fps, 60)  # Keep high FPS for motion
            fps_reasoning = "High motion detected, preserving higher FPS for smoothness"
        elif motion_level == 'low':
            base_fps = 24  # Low motion can use 24fps
            fps_reasoning = "Low motion detected, can reduce to cinematic 24fps"
        else:
            base_fps = 30  # Medium motion uses 30fps
            fps_reasoning = "Medium motion detected, using standard 30fps"
        
        if self.debug_logging_enabled:
            logger.info(f"Base FPS selection:")
            logger.info(f"  Base FPS: {base_fps:.1f}")
            logger.info(f"  Reasoning: {fps_reasoning}")
        
        # Adjust based on strategy
        strategy_adjustment = ""
        if strategy == 'aggressive':
            # Reduce FPS more aggressively
            if compression_ratio > 10:
                old_fps = base_fps
                base_fps = min(base_fps * 0.5, 24)
                strategy_adjustment = f"Aggressive strategy with extreme compression ratio ({compression_ratio:.1f}x): reduced FPS by 50% ({old_fps:.1f} → {base_fps:.1f})"
            elif compression_ratio > 5:
                old_fps = base_fps
                base_fps = min(base_fps * 0.7, 30)
                strategy_adjustment = f"Aggressive strategy with high compression ratio ({compression_ratio:.1f}x): reduced FPS by 30% ({old_fps:.1f} → {base_fps:.1f})"
            else:
                strategy_adjustment = "Aggressive strategy but compression ratio manageable, no additional FPS reduction"
        elif strategy == 'quality':
            # Try to preserve FPS
            base_fps = min(original_fps, 60)
            strategy_adjustment = f"Quality strategy: preserving original FPS ({base_fps:.1f})"
        else:
            strategy_adjustment = f"Balanced strategy: using motion-based FPS ({base_fps:.1f})"
        
        # Check against configuration constraints first
        config_min_fps = self.config.get('video_compression.bitrate_validation.min_fps', 20)
        
        # Further reduce if extreme compression is needed
        extreme_compression_adjustment = ""
        if compression_ratio > 15:
            old_fps = base_fps
            base_fps = max(base_fps * 0.6, config_min_fps)
            extreme_compression_adjustment = f"Extreme compression ratio ({compression_ratio:.1f}x): additional 40% FPS reduction ({old_fps:.1f} → {base_fps:.1f})"
        
        # Always cap at 60 and floor at config minimum (not hardcoded 10)
        final_fps = max(config_min_fps, min(base_fps, 60))
        if final_fps < config_min_fps:
            constraint_adjustment = f"FPS {final_fps:.1f} below configured minimum {config_min_fps}, clamping to minimum"
            final_fps = config_min_fps
        else:
            constraint_adjustment = f"FPS {final_fps:.1f} meets configured minimum {config_min_fps}"
        
        if self.debug_logging_enabled:
            logger.info(f"FPS adjustments:")
            if strategy_adjustment:
                logger.info(f"  Strategy: {strategy_adjustment}")
            if extreme_compression_adjustment:
                logger.info(f"  Extreme compression: {extreme_compression_adjustment}")
            logger.info(f"  Configuration constraint: {constraint_adjustment}")
            logger.info(f"Final FPS: {final_fps:.1f}")
            
            # Log FPS reduction impact assessment
            fps_reduction_percent = ((original_fps - final_fps) / original_fps) * 100 if original_fps > 0 else 0
            if fps_reduction_percent > 0:
                if fps_reduction_percent <= 20:
                    impact_assessment = "minimal impact on perceived quality"
                elif fps_reduction_percent <= 40:
                    impact_assessment = "moderate impact, should maintain acceptable smoothness"
                elif fps_reduction_percent <= 60:
                    impact_assessment = "significant impact, may affect smoothness"
                else:
                    impact_assessment = "severe impact, choppy playback likely"
                
                logger.info(f"FPS reduction impact: {fps_reduction_percent:.1f}% reduction, {impact_assessment}")
            
            logger.info(f"=== END FPS CALCULATION ===")
        
        return final_fps
    
    def _calculate_optimal_audio_bitrate(self, input_path: str, video_info: Dict[str, Any], 
                                        target_size_mb: float) -> int:
        """Choose audio bitrate based on input quality and size budget"""
        duration = video_info.get('duration', 0)
        
        if self.debug_logging_enabled:
            logger.info(f"=== AUDIO BITRATE CALCULATION ===")
            logger.info(f"Input parameters:")
            logger.info(f"  Duration: {duration:.1f}s")
            logger.info(f"  Target size: {target_size_mb:.2f}MB")
        
        # Try to probe input audio bitrate
        input_audio_bitrate = 128  # Default
        probe_success = False
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
                    probe_success = True
                    if self.debug_logging_enabled:
                        logger.info(f"Probed input audio bitrate: {input_audio_bitrate}kbps")
                else:
                    if self.debug_logging_enabled:
                        logger.info("No audio stream found in input, using default 128kbps")
            else:
                if self.debug_logging_enabled:
                    logger.info(f"FFprobe failed (return code {result.returncode}), using default 128kbps")
        except Exception as e:
            if self.debug_logging_enabled:
                logger.info(f"Audio probing failed ({e}), using default 128kbps")
        
        if not probe_success:
            input_audio_bitrate = 128
        
        # Calculate what we can afford (reserve 10-15% of budget for audio)
        total_bitrate = (target_size_mb * 8 * 1024) / max(duration, 1)  # kbps
        audio_budget = total_bitrate * 0.12  # 12% for audio
        
        if self.debug_logging_enabled:
            logger.info(f"Budget calculation:")
            logger.info(f"  Total bitrate budget: {total_bitrate:.1f}kbps")
            logger.info(f"  Audio budget (12%): {audio_budget:.1f}kbps")
            logger.info(f"  Input audio bitrate: {input_audio_bitrate}kbps")
        
        # Scale down from input quality, never exceed it
        target_audio = min(input_audio_bitrate, audio_budget)
        
        if self.debug_logging_enabled:
            logger.info(f"Target calculation:")
            logger.info(f"  Target audio (min of input/budget): {target_audio:.1f}kbps")
        
        # Quantize to standard bitrates
        final_bitrate = 32  # Default minimum
        quantization_reasoning = ""
        
        if target_audio >= 160:
            final_bitrate = min(192, input_audio_bitrate)
            quantization_reasoning = f"High quality target ({target_audio:.1f}kbps) → 192kbps (capped by input: {input_audio_bitrate}kbps)"
        elif target_audio >= 112:
            final_bitrate = min(128, input_audio_bitrate)
            quantization_reasoning = f"Good quality target ({target_audio:.1f}kbps) → 128kbps (capped by input: {input_audio_bitrate}kbps)"
        elif target_audio >= 80:
            final_bitrate = min(96, input_audio_bitrate)
            quantization_reasoning = f"Medium quality target ({target_audio:.1f}kbps) → 96kbps (capped by input: {input_audio_bitrate}kbps)"
        elif target_audio >= 56:
            final_bitrate = 64
            quantization_reasoning = f"Low-medium quality target ({target_audio:.1f}kbps) → 64kbps"
        elif target_audio >= 40:
            final_bitrate = 48
            quantization_reasoning = f"Low quality target ({target_audio:.1f}kbps) → 48kbps"
        else:
            final_bitrate = 32
            quantization_reasoning = f"Very low quality target ({target_audio:.1f}kbps) → 32kbps (minimum acceptable)"
        
        if self.debug_logging_enabled:
            logger.info(f"Quantization:")
            logger.info(f"  {quantization_reasoning}")
            logger.info(f"Final audio bitrate: {final_bitrate}kbps")
            
            # Log quality impact assessment
            quality_reduction = ((input_audio_bitrate - final_bitrate) / input_audio_bitrate) * 100 if input_audio_bitrate > 0 else 0
            if quality_reduction > 0:
                if quality_reduction <= 25:
                    quality_impact = "minimal audio quality impact"
                elif quality_reduction <= 50:
                    quality_impact = "moderate audio quality reduction"
                elif quality_reduction <= 75:
                    quality_impact = "significant audio quality reduction"
                else:
                    quality_impact = "severe audio quality reduction"
                
                logger.info(f"Quality impact: {quality_reduction:.1f}% reduction, {quality_impact}")
            else:
                logger.info("Quality impact: no reduction (using input bitrate or higher)")
            
            logger.info(f"=== END AUDIO BITRATE CALCULATION ===")
        
        return final_bitrate

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
        cmd1 = FFmpegUtils.build_two_pass_with_filters(input_path, output_path, params, pass_num=1, log_file=log_base, bitrate_validator=self.bitrate_validator)
        ok1 = FFmpegUtils.execute_ffmpeg_with_progress(cmd1, description="Analyzing", duration=None)
        if not ok1:
            return False, 0.0, log_base
        # Pass 2
        cmd2 = FFmpegUtils.build_two_pass_with_filters(input_path, output_path, params, pass_num=2, log_file=log_base, bitrate_validator=self.bitrate_validator)
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
            cmd2 = FFmpegUtils.build_two_pass_with_filters(input_path, output_path, params2, pass_num=2, log_file=log_base, bitrate_validator=self.bitrate_validator)
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
                        quality_aware = bool(self.config.get('video_compression.quality_controls.uplift.quality_aware', True))
                        min_quality_improvement = float(self.config.get('video_compression.quality_controls.uplift.min_quality_improvement', 2.0))
                        previous_quality_score = quality_score
                        
                        while uplift_count < uplift_max and quality_score < single_file_floor:
                            # Calculate adaptive bitrate step based on quality feedback
                            if quality_aware and uplift_count > 0:
                                quality_delta = quality_score - previous_quality_score
                                if quality_delta < min_quality_improvement:
                                    # Quality not improving enough, use smaller step
                                    adaptive_step = max(1.05, uplift_step * 0.9)
                                else:
                                    # Quality improving well, use normal step
                                    adaptive_step = uplift_step
                            else:
                                adaptive_step = uplift_step
                            
                            # Increase bitrate conservatively within cap
                            next_kbps = int(min(base_kbps * (adaptive_step ** (uplift_count + 1)), needed_kbps * uplift_cap))
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
                            
                            previous_quality_score = quality_score
                            quality_score = quality_result_new['overall_score']
                            quality_improvement = quality_score - previous_quality_score
                            
                            logger.info(f"Uplift pass {uplift_count+1}: bitrate={cur_v_kbps} kbps, size={size_mb:.2f}MB, quality={quality_score:.1f}/100 (Δ{quality_improvement:+.1f})")
                            
                            # Stop early if quality improvement is minimal and we're close to target
                            if quality_aware and uplift_count > 0:
                                if quality_improvement < min_quality_improvement and size_mb >= target_size_mb * 0.95:
                                    logger.info(f"Quality improvement minimal ({quality_improvement:.1f} < {min_quality_improvement:.1f}) and size close to target, stopping uplift")
                                    break
                            
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
                                       video_info: Dict[str, Any],
                                       platform_config: Optional[Dict[str, Any]] = None,
                                       platform: Optional[str] = None) -> Dict[str, Any]:
        """Content-Adaptive Encoding pipeline optimized for Discord 10MB limit.
        
        Uses 2-pass x264, BPP-driven resolution/FPS selection, VMAF/SSIM quality gates,
        artifact detection, and adaptive refine loop.
        """
        logger.info("=== Starting Discord 10MB CAE Pipeline ===")
        platform_config = platform_config or {}
        
        # Load profile config
        profile_cfg = self.config.get('video_compression.profiles.discord_10mb', {})
        size_limit_mb = profile_cfg.get('size_limit_mb', 10.0)
        refine_cfg = profile_cfg.get('refine', {})
        size_guard_mb = max(0.0, refine_cfg.get('size_guard_mb', 0.05))
        max_over_limit_mb = max(0.0, refine_cfg.get('max_over_limit_mb', 0.0))
        target_limit_mb = max(size_limit_mb - size_guard_mb, 0.1)
        acceptance_limit_mb = size_limit_mb + max_over_limit_mb
        logger.debug(
            "Discord 10MB size control: target %.3fMB, acceptance %.3fMB (guard %.3fMB, tolerance %.3fMB)",
            target_limit_mb,
            acceptance_limit_mb,
            size_guard_mb,
            max_over_limit_mb,
        )

        best_candidate = {
            'path': None,
            'size_mb': None,
            'pass_index': None,
            'reason': None,
            'quality': None,
            'artifacts': None,
        }

        def _store_best_candidate(size_mb: float, pass_index: int, reason: str,
                                  quality_result: Optional[Dict[str, Any]] = None,
                                  artifact_result: Optional[Dict[str, Any]] = None) -> None:
            """Preserve the latest under-limit encode so we can fall back if later passes fail."""
            if size_mb > acceptance_limit_mb:
                return
            if not os.path.exists(output_path):
                return
            if best_candidate['path'] is None:
                best_candidate['path'] = os.path.join(
                    self.temp_dir,
                    f"cae_best_{int(time.time())}_{threading.get_ident()}.mp4"
                )
            try:
                shutil.copy2(output_path, best_candidate['path'])
                best_candidate['size_mb'] = size_mb
                best_candidate['pass_index'] = pass_index
                best_candidate['reason'] = reason
                best_candidate['quality'] = quality_result
                best_candidate['artifacts'] = artifact_result
                logger.debug(f"Captured CAE best-effort candidate from pass {pass_index}: {size_mb:.4f}MB ({reason})")
            except Exception as e:
                logger.debug(f"Failed to preserve CAE best-effort candidate: {e}")

        def _cleanup_best_candidate() -> None:
            """Remove any temporary snapshot we created for best-effort fallback."""
            path = best_candidate.get('path')
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass
            best_candidate['path'] = None
        
        # Step 1: Analyze content complexity
        motion_level = video_info.get('motion_level', 'medium')  # from _analyze_video_content
        logger.info(f"Content analysis: motion={motion_level}, complexity={video_info.get('complexity_score', 5.0):.1f}")
        
        # Step 2: Budget calculation (reserve audio)
        audio_range = profile_cfg.get('audio_kbps_range', [64, 96])
        duration = video_info['duration']
        encoder = 'libx264'  # Default encoder for CAE pipeline
        min_encoder_bitrate = self.bitrate_validator.get_encoder_minimum(encoder)
        min_audio_bitrate = 64  # Minimum acceptable audio bitrate (prevents audio corruption)
        
        # Initial audio bitrate selection (motion-based)
        initial_audio_bitrate = audio_range[1] if motion_level == 'low' else audio_range[0]  # More audio for low-motion
        
        # Calculate resulting video bitrate with initial audio selection
        audio_bits = initial_audio_bitrate * 1000 * duration
        total_bits_budget = target_limit_mb * 8 * 1024 * 1024
        video_bits_budget = int(total_bits_budget - audio_bits)
        initial_video_bitrate = int(video_bits_budget / duration / 1000) if duration > 0 else 0
        
        # Duration-aware audio adjustment: reduce audio if video bitrate would be below minimum
        # For very long videos, high audio bitrate consumes too much of the budget
        audio_bitrate_kbps = initial_audio_bitrate
        target_video_bitrate_kbps = initial_video_bitrate
        
        if target_video_bitrate_kbps < min_encoder_bitrate and initial_audio_bitrate > min_audio_bitrate:
            # Try reducing audio bitrate to meet video bitrate minimum
            # Calculate maximum audio bitrate that still allows video bitrate >= minimum
            required_video_bitrate = min_encoder_bitrate
            required_video_bits = required_video_bitrate * 1000 * duration
            max_audio_bits = total_bits_budget - required_video_bits
            max_audio_bitrate = int(max_audio_bits / duration / 1000) if duration > 0 else min_audio_bitrate
            
            # Use the lower of: calculated max, or initial audio (but not below minimum)
            adjusted_audio_bitrate = max(min_audio_bitrate, min(max_audio_bitrate, initial_audio_bitrate))
            
            if adjusted_audio_bitrate < initial_audio_bitrate:
                # Recalculate with reduced audio
                audio_bitrate_kbps = adjusted_audio_bitrate
                audio_bits = audio_bitrate_kbps * 1000 * duration
                video_bits_budget = int(total_bits_budget - audio_bits)
                target_video_bitrate_kbps = int(video_bits_budget / duration / 1000) if duration > 0 else 0
                
                # Ensure we actually meet the minimum (handle rounding errors)
                if target_video_bitrate_kbps < min_encoder_bitrate:
                    # If still below minimum, reduce audio bitrate further
                    # This can happen due to integer truncation
                    remaining_bits = total_bits_budget - audio_bits
                    required_bits = min_encoder_bitrate * 1000 * duration
                    if remaining_bits < required_bits:
                        # Need to reduce audio more
                        additional_audio_reduction = int((required_bits - remaining_bits) / duration / 1000) + 1
                        audio_bitrate_kbps = max(min_audio_bitrate, audio_bitrate_kbps - additional_audio_reduction)
                        audio_bits = audio_bitrate_kbps * 1000 * duration
                        video_bits_budget = int(total_bits_budget - audio_bits)
                        target_video_bitrate_kbps = int(video_bits_budget / duration / 1000) if duration > 0 else 0
                
                logger.info(f"Duration-aware audio adjustment: reduced audio from {initial_audio_bitrate}k to {audio_bitrate_kbps}k "
                          f"to meet video bitrate minimum ({target_video_bitrate_kbps}k >= {min_encoder_bitrate}k)")
        
        logger.info(f"Budget: target {target_limit_mb:.2f}MB (limit {size_limit_mb:.2f}MB) = "
                    f"{target_video_bitrate_kbps}k video + {audio_bitrate_kbps}k audio over {duration:.1f}s")
        
        # Step 2.5: Immediate bitrate validation and emergency adjustment
        
        # Immediate validation with enhanced logging
        validation_result = self.bitrate_validator.validate_bitrate(target_video_bitrate_kbps, encoder)
        self.bitrate_validator.log_validation_result(
            validation_result, 
            context="CAE_immediate_validation", 
            video_info=video_info
        )
        
        if target_video_bitrate_kbps < min_encoder_bitrate:
            logger.warning(f"IMMEDIATE VALIDATION FAILURE: Calculated bitrate {target_video_bitrate_kbps}kbps < {encoder} minimum {min_encoder_bitrate}kbps")
            
            # Emergency resolution reduction to meet bitrate floor
            emergency_adjustment_needed = True
            original_target_bitrate = target_video_bitrate_kbps
            original_width = video_info['width']
            original_height = video_info['height']
            
            # Calculate deficit for segmentation estimation
            bitrate_deficit_ratio = min_encoder_bitrate / max(target_video_bitrate_kbps, 1)
            
            # Try emergency resolution reduction with enhanced logic
            emergency_resolutions = self.parameter_adjuster.get_fallback_resolutions((original_width, original_height))
            
            # Add ultra-low emergency resolutions if not already present
            ultra_low_resolutions = [(320, 180), (240, 135), (160, 90)]
            for ultra_res in ultra_low_resolutions:
                if ultra_res not in emergency_resolutions:
                    emergency_resolutions.append(ultra_res)
            
            for emergency_width, emergency_height in emergency_resolutions:
                logger.warning(f"EMERGENCY: Attempting resolution reduction to {emergency_width}x{emergency_height}")
                
                # Calculate new bitrate with emergency resolution
                # Key insight: same file size budget with smaller resolution = higher bitrate per pixel
                emergency_pixels = emergency_width * emergency_height
                original_pixels = original_width * original_height
                pixel_reduction_ratio = emergency_pixels / original_pixels
                
                # Estimate bitrate improvement from resolution reduction
                # Smaller resolution allows higher bitrate for same file size
                estimated_bitrate_improvement = 1.0 / math.sqrt(pixel_reduction_ratio)
                adjusted_bitrate = int(original_target_bitrate * estimated_bitrate_improvement)
                
                # Apply safety margin
                safe_adjusted_bitrate = int(adjusted_bitrate * 0.95)  # 5% safety margin
                
                logger.info(f"EMERGENCY: Resolution {emergency_width}x{emergency_height} would allow ~{safe_adjusted_bitrate}kbps (need {min_encoder_bitrate}kbps)")
                
                if safe_adjusted_bitrate >= min_encoder_bitrate:
                    # Success - emergency resolution reduction sufficient
                    target_video_bitrate_kbps = safe_adjusted_bitrate
                    
                    # Update video_info for downstream processing
                    video_info = video_info.copy()
                    video_info['emergency_width'] = emergency_width
                    video_info['emergency_height'] = emergency_height
                    video_info['emergency_adjustment'] = True
                    
                    logger.warning(f"EMERGENCY SUCCESS: Resolution reduced to {emergency_width}x{emergency_height}, bitrate adjusted to {safe_adjusted_bitrate}kbps")
                    
                    # Log the emergency fallback strategy
                    emergency_params = {
                        'width': emergency_width, 'height': emergency_height, 
                        'bitrate': safe_adjusted_bitrate
                    }
                    original_params = {
                        'width': original_width, 'height': original_height, 
                        'bitrate': original_target_bitrate
                    }
                    self.bitrate_validator.log_fallback_strategy_used(
                        'emergency_resolution_reduction', original_params, emergency_params,
                        context="CAE_immediate_validation",
                        reason=f"bitrate {original_target_bitrate}kbps < minimum {min_encoder_bitrate}kbps"
                    )
                    
                    emergency_adjustment_needed = False
                    break
                
                # If we're at the smallest resolution and still can't meet minimum, force minimum bitrate
                if emergency_width <= 320 and emergency_height <= 180:
                    logger.warning(f"EMERGENCY: Even ultra-low resolution insufficient. Forcing minimum bitrate {min_encoder_bitrate}kbps")
                    target_video_bitrate_kbps = min_encoder_bitrate
                    
                    # Update video_info for downstream processing
                    video_info = video_info.copy()
                    video_info['emergency_width'] = emergency_width
                    video_info['emergency_height'] = emergency_height
                    video_info['emergency_adjustment'] = True
                    video_info['forced_minimum_bitrate'] = True
                    
                    emergency_adjustment_needed = False
                    break
            
            if emergency_adjustment_needed:
                # Emergency resolution reduction completely insufficient - segmentation required
                logger.error(f"CRITICAL: Emergency resolution reduction insufficient. Original bitrate: {original_target_bitrate}kbps, Required: {min_encoder_bitrate}kbps")
                logger.error(f"CRITICAL: Bitrate deficit ratio: {bitrate_deficit_ratio:.1f}x below minimum")
                
                # Calculate segmentation parameters
                estimated_segments = math.ceil(bitrate_deficit_ratio)
                segment_duration = duration / estimated_segments
                
                logger.error(f"SEGMENTATION REQUIRED: Split into ~{estimated_segments} segments of ~{segment_duration:.1f}s each")
                logger.error("FALLBACK: Attempting segmentation after CAE pipeline failure")
                
                # Mark for segmentation fallback
                video_info = video_info.copy()
                video_info['requires_segmentation'] = True
                video_info['segmentation_reason'] = 'emergency_bitrate_failure'
                video_info['estimated_segments'] = estimated_segments
                video_info['segment_duration'] = segment_duration
                
                # Log segmentation fallback strategy
                self.bitrate_validator.log_fallback_strategy_used(
                    'segmentation', 
                    {'bitrate': original_target_bitrate}, 
                    {'estimated_segments': estimated_segments, 'segment_duration': segment_duration},
                    context="CAE_emergency_fallback",
                    reason=f"emergency resolution reduction insufficient for bitrate floor {min_encoder_bitrate}kbps"
                )
                
                # Continue with minimum bitrate for CAE attempt, but expect failure
                target_video_bitrate_kbps = min_encoder_bitrate
                logger.warning(f"PROCEEDING: Using minimum bitrate {min_encoder_bitrate}kbps (CAE likely to fail, segmentation fallback prepared)")
        
        # Step 3: BPP-driven resolution/FPS selection (with emergency resolution override)
        bpp_floors = profile_cfg.get('bpp_floor', {})
        bpp_min = bpp_floors.get(motion_level, bpp_floors.get('normal', 0.035))
        
        # Use emergency resolution if set during immediate validation
        if video_info.get('emergency_adjustment'):
            emergency_width = video_info.get('emergency_width')
            emergency_height = video_info.get('emergency_height')
            
            logger.info(f"Using emergency resolution override: {emergency_width}x{emergency_height}")
            
            # Calculate FPS for emergency resolution
            emergency_video_info = video_info.copy()
            emergency_video_info['width'] = emergency_width
            emergency_video_info['height'] = emergency_height
            
            initial_params = self._select_resolution_fps_by_bpp(
                emergency_video_info, target_video_bitrate_kbps, bpp_min, profile_cfg
            )
            
            # Override with emergency resolution (in case BPP selection chose different)
            initial_params['width'] = emergency_width
            initial_params['height'] = emergency_height
            
        else:
            # Normal BPP-driven selection
            initial_params = self._select_resolution_fps_by_bpp(
                video_info, target_video_bitrate_kbps, bpp_min, profile_cfg
            )
        
        initial_params['bitrate'] = target_video_bitrate_kbps
        initial_params['audio_bitrate'] = audio_bitrate_kbps
        initial_params['encoder'] = 'libx264'  # Default encoder for CAE pipeline

        guardrail_cfg = profile_cfg.get('guardrails', {}) or {}
        guardrail_details = {}
        min_short_side = int(guardrail_cfg.get('min_short_side_px', 0) or 0)
        short_side = min(initial_params['width'], initial_params['height'])
        if min_short_side and short_side < min_short_side:
            guardrail_details = {
                'requested_width': initial_params['width'],
                'requested_height': initial_params['height'],
                'short_side_px': short_side,
                'min_short_side_px': min_short_side,
                'video_duration': video_info.get('duration'),
                'video_size_mb': video_info.get('size_mb')
            }
            logger.warning(
                "CAE guardrail triggered before encode: predicted %dx%d falls below %dpx short-side floor",
                initial_params['width'],
                initial_params['height'],
                min_short_side
            )
            logger.info("Rerouting to segmentation to preserve watchable resolution before CAE encode")
            video_info_guard = video_info.copy()
            video_info_guard['requires_segmentation'] = True
            video_info_guard['segmentation_reason'] = 'cae_resolution_guardrail'
            video_info_guard['guardrail_details'] = guardrail_details
            segmentation_result = self._compress_with_segmentation(
                input_path,
                output_path,
                size_limit_mb,
                platform_config,
                video_info_guard,
                platform or 'discord'
            )
            if not segmentation_result:
                segmentation_result = {
                    'success': False,
                    'error': 'Segmentation guardrail failed to produce output',
                    'method': 'segmentation_guardrail_failed'
                }
            segmentation_result['guardrail_triggered'] = True
            segmentation_result['guardrail_reason'] = 'discord_resolution_guardrail'
            segmentation_result['guardrail_details'] = guardrail_details
            segmentation_result.setdefault('method', 'segmentation')
            segmentation_result.setdefault('is_segmented_output', True)
            segmentation_result.setdefault('video_info', video_info_guard)
            segmentation_result.setdefault('segmentation_trigger', 'cae_resolution_guardrail')
            return segmentation_result
        
        logger.info(
            f"Initial target: {initial_params['width']}x{initial_params['height']}@{initial_params['fps']}fps, "
            f"BPP={bpp_min:.3f}, bitrate={target_video_bitrate_kbps}k"
        )
        
        # Step 3.5: Pre-encoding bitrate validation and parameter adjustment
        if self.bitrate_validator.is_validation_enabled():
            validation_result = self.bitrate_validator.validate_bitrate(
                target_video_bitrate_kbps, initial_params['encoder']
            )
            self.bitrate_validator.log_validation_result(validation_result, "CAE Pre-encoding")
            
            if validation_result.adjustment_needed:
                # Enhanced logging with video context
                self.bitrate_validator.log_validation_result(
                    validation_result, 
                    context="CAE_pipeline", 
                    video_info=video_info
                )
                
                # Attempt automatic parameter adjustment
                adjustment_result = self.parameter_adjuster.adjust_for_bitrate_floor(
                    initial_params, validation_result.minimum_required, video_info, size_limit_mb
                )
                self.parameter_adjuster.log_adjustment_result(adjustment_result, "CAE Auto-adjustment")
                
                if adjustment_result.success:
                    # Use adjusted parameters and log the fallback strategy
                    original_params = initial_params.copy()
                    initial_params = adjustment_result.adjusted_params
                    
                    # Log the fallback strategy used with enhanced details
                    strategy = adjustment_result.adjustment_type
                    self.bitrate_validator.log_fallback_strategy_used(
                        strategy, original_params, initial_params, 
                        context="CAE_auto_adjustment",
                        reason="bitrate below encoder minimum"
                    )
                    
                    logger.info(f"Parameters automatically adjusted: {initial_params['width']}x{initial_params['height']}@{initial_params['fps']}fps, "
                               f"bitrate={initial_params['bitrate']}k")
                    
                    # Re-validate adjusted parameters
                    adjusted_validation = self.bitrate_validator.validate_bitrate(
                        initial_params['bitrate'], initial_params['encoder']
                    )
                    if not adjusted_validation.is_valid:
                        # Enhanced error logging with context
                        self.bitrate_validator.log_validation_result(
                            adjusted_validation,
                            context="CAE_post_adjustment",
                            video_info=video_info
                        )
                        
                        # Create detailed error for potential exception handling
                        validation_error = self.bitrate_validator.create_validation_error(
                            adjusted_validation,
                            encoder=initial_params['encoder'],
                            context="CAE_post_adjustment",
                            video_info=video_info
                        )
                        logger.error(f"Bitrate validation details:\n{validation_error.get_detailed_message()}")
                        
                        # Log suggestion for segmentation fallback
                        logger.error("Automatic parameter adjustment insufficient - segmentation fallback recommended")
                else:
                    logger.warning(f"Parameter adjustment failed: {adjustment_result.message}")
                    logger.warning("Consider enabling segmentation or using a different encoder")
                    
                    # Check if segmentation is recommended
                    if adjustment_result.adjustment_type == 'failed':
                        logger.info("Bitrate floor cannot be met with single file - segmentation may be required")
                        self.bitrate_validator.log_fallback_strategy_used(
                            'segmentation', initial_params, initial_params, 
                            context="CAE_fallback",
                            reason="parameter adjustment failed"
                        )
                        # Note: We continue with CAE pipeline first, segmentation fallback handled elsewhere
        
        # Step 4: 2-pass encode with refine loop
        x264_cfg = profile_cfg.get('x264', {})
        max_passes = refine_cfg.get('max_passes', 3)
        bitrate_step = refine_cfg.get('bitrate_step', 1.08)
        
        vmaf_threshold = profile_cfg.get('vmaf_threshold', 80.0)
        vmaf_threshold_low_res = profile_cfg.get('vmaf_threshold_low_res', 78.0)
        ssim_threshold = profile_cfg.get('ssim_threshold', 0.94)
        
        # Track quality history for early termination
        quality_history = []  # List of (pass, vmaf_score) tuples
        
        # Track previous passes for progress detection
        previous_sizes = []  # List of (pass, size_mb, bitrate_kbps) tuples
        encoder = initial_params.get('encoder', 'libx264')
        min_encoder_bitrate = self.bitrate_validator.get_encoder_minimum(encoder)
        same_size_count = 0  # Count consecutive passes with same size
        max_same_size_retries = 2  # Maximum retries when stuck at same size
        
        for refine_pass in range(max_passes):
            logger.info(f"--- CAE Refine Pass {refine_pass + 1}/{max_passes} ---")
            
            # Check if bitrate is already at minimum
            current_bitrate = initial_params.get('bitrate', target_video_bitrate_kbps)
            at_bitrate_minimum = current_bitrate <= min_encoder_bitrate
            
            if at_bitrate_minimum:
                logger.warning(f"Bitrate already at encoder minimum ({current_bitrate}kbps <= {min_encoder_bitrate}kbps). "
                             f"Will use resolution/FPS reduction instead of bitrate reduction.")
            
            # Encode with current params
            encode_params = self._build_cae_encode_params(initial_params, x264_cfg, profile_cfg)
            success = self._execute_two_pass_x264(input_path, output_path, encode_params)
            
            if not success or not os.path.exists(output_path):
                logger.error(f"Encode failed on pass {refine_pass + 1}")
                continue
            
            # Check size
            output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
            logger.info(f"Encoded size: {output_size_mb:.2f}MB (target: {target_limit_mb:.2f}MB, limit: {size_limit_mb:.2f}MB)")
            within_acceptance = output_size_mb <= acceptance_limit_mb
            
            # Track this pass for progress detection
            previous_sizes.append((refine_pass + 1, output_size_mb, current_bitrate))
            
            # Detect if stuck at same size (within 0.1MB tolerance)
            # But allow one more pass if close to target (within 0.1MB over limit)
            close_to_target_threshold = 0.1  # MB over target to consider "close enough" for one more pass
            is_close_to_target = (target_limit_mb < output_size_mb <= target_limit_mb + close_to_target_threshold)
            
            if len(previous_sizes) >= 2:
                prev_size = previous_sizes[-2][1]
                size_diff = abs(output_size_mb - prev_size)
                size_is_decreasing = output_size_mb < prev_size  # Moving toward target
                
                if size_diff < 0.1:  # Less than 0.1MB difference
                    # If we're close to target and size is decreasing, reset counter (one more pass might succeed)
                    if is_close_to_target and size_is_decreasing:
                        logger.info(f"Output size close to target ({output_size_mb:.2f}MB, target: {target_limit_mb:.2f}MB) "
                                   f"and decreasing (prev: {prev_size:.2f}MB). Allowing one more pass.")
                        same_size_count = 0  # Reset counter to allow one more pass
                    else:
                        same_size_count += 1
                        logger.warning(f"Output size unchanged: {output_size_mb:.2f}MB (prev: {prev_size:.2f}MB, diff: {size_diff:.3f}MB)")
                        # Only break if we're stuck AND not close to target
                        if same_size_count >= max_same_size_retries and not is_close_to_target:
                            if within_acceptance:
                                _store_best_candidate(output_size_mb, refine_pass + 1, "stuck_same_size")
                            logger.error(f"Stuck at same output size after {same_size_count} consecutive passes. "
                                       f"Bitrate: {current_bitrate}kbps (min: {min_encoder_bitrate}kbps). "
                                       f"Terminating refinement loop early.")
                            break
                        elif same_size_count >= max_same_size_retries and is_close_to_target:
                            logger.warning(f"Stuck at same size but close to target ({output_size_mb:.2f}MB). "
                                         f"Allowing one final pass with reduced parameters.")
                            same_size_count = max_same_size_retries - 1  # Reset to allow one more pass
                else:
                    same_size_count = 0  # Reset counter when size changes
            
            size_tolerance_mb = max_over_limit_mb
            effective_limit = target_limit_mb + size_tolerance_mb
            
            # Log actual byte size for debugging
            if os.path.exists(output_path):
                actual_bytes = os.path.getsize(output_path)
                actual_mb = actual_bytes / (1024 * 1024)
                limit_bytes = target_limit_mb * 1024 * 1024
                logger.debug(f"Size check (overage): {actual_mb:.6f}MB ({actual_bytes} bytes) vs "
                             f"target {target_limit_mb:.6f}MB ({limit_bytes:.0f} bytes), tolerance={size_tolerance_mb:.6f}MB")
            else:
                actual_mb = output_size_mb
            
            if actual_mb > effective_limit:
                logger.warning(f"Output exceeds encode target: {actual_mb:.6f}MB > {effective_limit:.6f}MB "
                               f"(target={target_limit_mb:.6f}MB + tolerance={size_tolerance_mb:.6f}MB, "
                               f"acceptance={acceptance_limit_mb:.6f}MB)")
                if within_acceptance:
                    _store_best_candidate(actual_mb, refine_pass + 1, "pre_quality_over_target")
                # Calculate how much we need to reduce relative to the encode target
                overage_ratio = actual_mb / max(target_limit_mb, 0.0001)
                
                # Check if bitrate is at minimum - if so, use resolution/FPS reduction instead
                if at_bitrate_minimum or current_bitrate <= min_encoder_bitrate:
                    logger.warning(f"Bitrate at minimum ({current_bitrate}kbps <= {min_encoder_bitrate}kbps). "
                                 f"Switching to resolution/FPS reduction instead of bitrate reduction.")
                    # Use resolution reduction
                    if initial_params['height'] > 480:
                        new_width, new_height = self._scale_down_one_step(initial_params['width'], initial_params['height'])
                        initial_params['width'] = new_width
                        initial_params['height'] = new_height
                        logger.info(f"Refine (overage, bitrate at min): scaling down to {new_width}x{new_height}")
                    else:
                        # Try FPS reduction as last resort
                        min_fps = self.config.get('video_compression.bitrate_validation.min_fps', 20)
                        current_fps = initial_params.get('fps', 30)
                        if current_fps > min_fps:
                            # Reduce FPS by one step
                            fps_reduction_steps = self.config.get('video_compression.bitrate_validation.fps_reduction_steps', [0.8, 0.6, 0.5])
                            if fps_reduction_steps:
                                new_fps = int(current_fps * fps_reduction_steps[-1])  # Use most aggressive step
                                new_fps = max(min_fps, new_fps)
                                initial_params['fps'] = new_fps
                                logger.info(f"Refine (overage, bitrate at min): reducing FPS to {new_fps}")
                            else:
                                logger.error("Cannot reduce further - bitrate at minimum, resolution at minimum, and FPS at minimum")
                                break
                        else:
                            logger.error("Cannot reduce further - all parameters at minimum")
                            break
                else:
                    # More aggressive reduction based on overage
                    if overage_ratio > 1.10:
                        # More than 10% over: reduce bitrate significantly AND consider scaling down
                        new_bitrate = int(current_bitrate * 0.85)
                        # Ensure we don't go below minimum
                        if new_bitrate < min_encoder_bitrate:
                            new_bitrate = min_encoder_bitrate
                            logger.warning(f"Bitrate reduction would go below minimum. Clamping to {min_encoder_bitrate}kbps")
                            # Also scale down resolution
                            if initial_params['height'] > 480:
                                new_width, new_height = self._scale_down_one_step(initial_params['width'], initial_params['height'])
                                initial_params['width'] = new_width
                                initial_params['height'] = new_height
                                logger.info(f"Refine (overage): scaling down to {new_width}x{new_height}")
                        initial_params['bitrate'] = new_bitrate
                    else:
                        # Less than 10% over: just reduce bitrate proportionally
                        reduction_factor = 0.98 / overage_ratio  # Target 98% of limit
                        new_bitrate = int(current_bitrate * reduction_factor)
                        # Ensure we don't go below minimum
                        if new_bitrate < min_encoder_bitrate:
                            new_bitrate = min_encoder_bitrate
                            logger.warning(f"Bitrate reduction would go below minimum. Clamping to {min_encoder_bitrate}kbps")
                            # Also try scaling down
                            if initial_params['height'] > 480:
                                new_width, new_height = self._scale_down_one_step(initial_params['width'], initial_params['height'])
                                initial_params['width'] = new_width
                                initial_params['height'] = new_height
                                logger.info(f"Refine (overage): scaling down to {new_width}x{new_height}")
                        initial_params['bitrate'] = new_bitrate
                
                logger.info(f"Refine (overage): reducing bitrate to {initial_params['bitrate']}k")
                
                # Check if we're stuck (bitrate at minimum and can't reduce further)
                if initial_params['bitrate'] <= min_encoder_bitrate and initial_params['height'] <= 480:
                    min_fps = self.config.get('video_compression.bitrate_validation.min_fps', 20)
                    if initial_params.get('fps', 30) <= min_fps:
                        logger.error("Cannot reduce further - all parameters at minimum. Terminating refinement.")
                        break
                
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
            
            vmaf_score = quality_result.get('vmaf_score')
            ssim_score = quality_result.get('ssim_score')
            vmaf_str = f"{vmaf_score:.2f}" if vmaf_score is not None else "N/A"
            ssim_str = f"{ssim_score:.4f}" if ssim_score is not None else "N/A"
            logger.info(f"Quality: VMAF={vmaf_str}, SSIM={ssim_str}, passes={quality_result.get('passes')}")
            
            blockiness_score = artifact_result.get('blockiness_score')
            banding_score = artifact_result.get('banding_score')
            blockiness_str = f"{blockiness_score:.4f}" if blockiness_score is not None else "N/A"
            banding_str = f"{banding_score:.4f}" if banding_score is not None else "N/A"
            logger.info(f"Artifacts: blockiness={blockiness_str}, banding={banding_str}, passes={artifact_result.get('passes')}")
            
            # Check if all gates pass
            quality_pass = quality_result.get('passes', False)
            artifact_pass = artifact_result.get('passes', True)  # Default pass if detection fails
            
            if quality_pass and artifact_pass:
                logger.info(f"✓ All quality gates passed on refine pass {refine_pass + 1}")
                
                size_tolerance_mb = max_over_limit_mb
                effective_limit = size_limit_mb + size_tolerance_mb
                
                # Log actual byte size for debugging
                if os.path.exists(output_path):
                    actual_bytes = os.path.getsize(output_path)
                    actual_mb = actual_bytes / (1024 * 1024)
                    limit_bytes = size_limit_mb * 1024 * 1024
                    logger.debug(f"Size check: {actual_mb:.6f}MB ({actual_bytes} bytes) vs limit {size_limit_mb:.6f}MB ({limit_bytes:.0f} bytes), tolerance={size_tolerance_mb:.6f}MB")
                else:
                    actual_mb = output_size_mb
                
                if actual_mb > acceptance_limit_mb:
                    logger.warning(f"Quality gates passed but output exceeds acceptance limit: "
                                   f"{actual_mb:.6f}MB > {acceptance_limit_mb:.6f}MB. Continuing refinement.")
                    # Continue to refinement instead of returning
                elif actual_mb > target_limit_mb:
                    if within_acceptance:
                        _store_best_candidate(actual_mb, refine_pass + 1, "headroom_continue", quality_result, artifact_result)
                    logger.info(f"Quality gates passed but size {actual_mb:.6f}MB remains above headroom target "
                                f"{target_limit_mb:.6f}MB; continuing refinement.")
                else:
                    # Log structured metrics
                    self._log_cae_metrics({
                        'input_path': input_path,
                        'output_path': output_path,
                        'params': encode_params,
                        'size_mb': actual_mb,
                        'quality': quality_result,
                        'artifacts': artifact_result,
                        'refine_pass': refine_pass + 1
                    })
                    
                    _cleanup_best_candidate()
                    return self._get_compression_results(input_path, output_path, video_info, "cae_discord_10mb")
            
            # Step 7: Refine strategy if gates failed
            logger.warning(f"Quality gates failed on pass {refine_pass + 1}, attempting refinement")
            if within_acceptance:
                _store_best_candidate(actual_mb, refine_pass + 1, "quality_retry", quality_result, artifact_result)
            
            # Track quality for early termination detection
            if vmaf_score is not None:
                quality_history.append((refine_pass + 1, vmaf_score))
            
            # Check for quality regression (early termination)
            if len(quality_history) >= 2:
                prev_vmaf = quality_history[-2][1] if len(quality_history) >= 2 else None
                curr_vmaf = quality_history[-1][1] if quality_history else None
                
                if prev_vmaf is not None and curr_vmaf is not None:
                    if curr_vmaf < prev_vmaf:
                        regression = prev_vmaf - curr_vmaf
                        logger.warning(f"Quality regression detected: VMAF decreased from {prev_vmaf:.2f} to {curr_vmaf:.2f} "
                                     f"(regression: {regression:.2f})")
                        if regression > 2.0:  # Significant regression (>2 points)
                            logger.error(f"Significant quality regression detected ({regression:.2f} points). "
                                       f"Consider stopping refinement to avoid further degradation.")
            
            if refine_pass + 1 < max_passes:
                # Use new refinement strategy function
                bpp_floors = profile_cfg.get('bpp_floor', {})
                motion_level = video_info.get('motion_level', 'medium')
                bpp_min = bpp_floors.get(motion_level, bpp_floors.get('normal', 0.035))
                
                refinement_result = self._calculate_refinement_strategy(
                    quality_pass=quality_pass,
                    output_size_mb=output_size_mb,
                    size_limit_mb=target_limit_mb,
                    initial_params=initial_params.copy(),
                    audio_bitrate_kbps=audio_bitrate_kbps,
                    video_info=video_info,
                    duration=duration,
                    bitrate_step=bitrate_step,
                    refine_pass=refine_pass,
                    quality_result=quality_result,
                    bpp_min=bpp_min
                )
                
                strategy = refinement_result['strategy']
                adjusted_params = refinement_result['adjusted_params']
                reasoning = refinement_result['reasoning']
                
                if strategy == 'exhausted':
                    logger.error("All refinement strategies exhausted. Cannot improve quality further.")
                    break
                
                # Apply refined parameters
                initial_params.update(adjusted_params)
                logger.info(f"Refinement applied: {reasoning}")
                
                # Clean up failed attempt
                try:
                    os.remove(output_path)
                except:
                    pass
            else:
                if within_acceptance:
                    _store_best_candidate(actual_mb, refine_pass + 1, "max_passes_reached", quality_result, artifact_result)
                logger.warning("Maximum refinement passes reached. Cannot refine further.")
        
        # If we exhausted refine passes, return best attempt or fail to segmentation
        if os.path.exists(output_path):
            output_size_bytes = os.path.getsize(output_path)
            output_size_mb = output_size_bytes / (1024 * 1024)
            size_tolerance_mb = max_over_limit_mb
            effective_limit = min(acceptance_limit_mb, size_limit_mb + size_tolerance_mb)
            limit_bytes = size_limit_mb * 1024 * 1024
            logger.debug(f"Final size check: {output_size_mb:.6f}MB ({output_size_bytes} bytes) vs acceptance "
                         f"{effective_limit:.6f}MB (limit {size_limit_mb:.6f}MB, tolerance {size_tolerance_mb:.6f}MB)")
            
            if output_size_mb <= effective_limit:
                logger.warning("CAE completed with compromised quality (some gates failed)")
                _cleanup_best_candidate()
                return self._get_compression_results(input_path, output_path, video_info, "cae_discord_10mb_compromised")
            else:
                logger.error(f"CAE output rejected for size: {output_size_mb:.6f}MB > acceptance {effective_limit:.6f}MB")
        
        logger.error("CAE pipeline failed after all refine passes")

        if best_candidate['path'] and os.path.exists(best_candidate['path']):
            try:
                shutil.copy2(best_candidate['path'], output_path)
                logger.warning(f"Returning best-effort CAE result from pass {best_candidate['pass_index']} "
                               f"(size {best_candidate['size_mb']:.2f}MB, reason={best_candidate['reason']})")
                result = self._get_compression_results(input_path, output_path, video_info, "cae_discord_10mb_best_effort")
                result['best_effort'] = True
                result['best_effort_pass'] = best_candidate['pass_index']
                result['best_effort_reason'] = best_candidate['reason']
                result['best_effort_size_mb'] = best_candidate['size_mb']
                _cleanup_best_candidate()
                return result
            except Exception as e:
                logger.error(f"Failed to restore best-effort CAE candidate: {e}")
        
        # Enhanced segmentation fallback handling
        encoder = initial_params.get('encoder', 'libx264')
        
        # Check if segmentation was already identified as necessary during immediate validation
        if video_info.get('requires_segmentation'):
            segmentation_reason = video_info.get('segmentation_reason', 'unknown')
            estimated_segments = video_info.get('estimated_segments', 2)
            
            logger.info(f"Segmentation fallback triggered: {segmentation_reason}")
            logger.info(f"Estimated segments needed: {estimated_segments}")
            
            _cleanup_best_candidate()
            return self._try_segmentation_fallback(
                input_path, output_path, size_limit_mb, initial_params, video_info, encoder,
                f"CAE pipeline failed - {segmentation_reason}"
            )
        
        # Check if segmentation should be used due to bitrate constraints
        should_segment, reason = self.video_segmenter.should_segment_for_bitrate_constraints(
            video_info, size_limit_mb, encoder
        )
        
        if should_segment:
            logger.info(f"CAE failed, trying segmentation fallback: {reason}")
            _cleanup_best_candidate()
            return self._try_segmentation_fallback(
                input_path, output_path, size_limit_mb, initial_params, video_info, encoder,
                f"CAE pipeline failed - {reason}"
            )
        
        # Final check for emergency cases that might need segmentation
        if video_info.get('emergency_adjustment') or video_info.get('forced_minimum_bitrate'):
            logger.warning("Emergency adjustments were made but CAE still failed - attempting segmentation as last resort")
            _cleanup_best_candidate()
            return self._try_segmentation_fallback(
                input_path, output_path, size_limit_mb, initial_params, video_info, encoder,
                "CAE pipeline failed after emergency adjustments - last resort segmentation"
            )
        
        _cleanup_best_candidate()
        return {'success': False, 'error': 'CAE quality gates failed', 'method': 'cae_failed'}
    
    def _select_resolution_fps_by_bpp(self, video_info: Dict[str, Any], 
                                     target_bitrate_kbps: int, bpp_min: float,
                                     profile_cfg: Dict[str, Any]) -> Dict[str, Any]:
        """Select highest resolution and FPS that satisfy BPP floor constraint and encoder minimums."""
        orig_w = video_info['width']
        orig_h = video_info['height']
        orig_fps = video_info['fps']
        
        # Get encoder minimum bitrate requirement
        encoder = 'libx264'  # Default encoder for CAE pipeline
        min_encoder_bitrate = self.bitrate_validator.get_encoder_minimum(encoder)
        
        # Validate initial bitrate against encoder minimum
        validation_result = self.bitrate_validator.validate_bitrate(target_bitrate_kbps, encoder)
        self.bitrate_validator.log_validation_result(validation_result, "BPP_calculation", video_info)
        
        # If bitrate is critically low, we need progressive resolution reduction
        if not validation_result.is_valid and validation_result.severity == 'critical':
            logger.warning(f"Target bitrate {target_bitrate_kbps}kbps critically below {encoder} minimum {min_encoder_bitrate}kbps")
            logger.warning("Implementing progressive resolution reduction to meet bitrate floor")
            return self._progressive_resolution_reduction(video_info, target_bitrate_kbps, bpp_min, profile_cfg, encoder)
        
        # Candidate resolutions (maintain aspect ratio) - enhanced with more aggressive options
        aspect_ratio = orig_w / orig_h
        candidate_heights = [1080, 720, 540, 480, 360, 270, 240, 180, 144, 120]  # Added ultra-low resolutions
        candidates = []
        
        for h in candidate_heights:
            if h > orig_h:
                continue
            w = int(h * aspect_ratio)
            # Ensure even dimensions
            w = w if w % 2 == 0 else w - 1
            # Ensure minimum width of 64 pixels
            if w < 64:
                w = 64
            candidates.append((w, h))
        
        # Add original resolution if not in list
        if orig_h not in candidate_heights:
            candidates.insert(0, (orig_w, orig_h))
        
        # Candidate FPS values - expanded for bitrate floor scenarios
        # Use config minimum FPS instead of hardcoded 10
        config_min_fps = self.config.get('video_compression.bitrate_validation.min_fps', 20)
        
        fps_candidates = [30, 24, 20, 15, 12, 8, 6] if orig_fps >= 30 else [24, 20, 15, 12, 8, 6, int(orig_fps)]
        fps_prefer_24 = profile_cfg.get('fps_policy', {}).get('prefer_24_for_low_motion', True)
        motion_level = video_info.get('motion_level', 'medium')
        
        if fps_prefer_24 and motion_level == 'low':
            fps_candidates = [24, 20, 15, 12, 8, 6, 30]  # Prefer 24 for low motion
        
        # Filter to only include FPS values >= config_min_fps
        fps_candidates = [fps for fps in fps_candidates if fps >= config_min_fps]
        
        # Find best combo that satisfies both BPP and bitrate floor
        target_bits_per_sec = target_bitrate_kbps * 1000
        
        # First pass: try to satisfy both BPP and bitrate floor
        for w, h in candidates:
            for fps in fps_candidates:
                pixels_per_sec = w * h * fps
                actual_bpp = target_bits_per_sec / pixels_per_sec
                
                # Check both BPP floor and encoder minimum bitrate
                if actual_bpp >= bpp_min and target_bitrate_kbps >= min_encoder_bitrate:
                    # Final validation before returning parameters
                    final_validation = self.bitrate_validator.validate_bitrate(target_bitrate_kbps, encoder)
                    if final_validation.is_valid:
                        logger.info(f"Selected {w}x{h}@{fps}fps: BPP={actual_bpp:.4f} >= {bpp_min:.4f}, bitrate={target_bitrate_kbps}kbps >= {min_encoder_bitrate}kbps")
                        return {'width': w, 'height': h, 'fps': fps}
        
        # Second pass: prioritize bitrate floor over BPP if necessary
        logger.warning(f"Cannot satisfy both BPP floor ({bpp_min:.4f}) and bitrate floor ({min_encoder_bitrate}kbps) simultaneously")
        logger.warning("Prioritizing encoder bitrate minimum over BPP floor")
        
        # Progressive resolution reduction until bitrate floor is met
        for w, h in candidates:
            for fps in fps_candidates:
                pixels_per_sec = w * h * fps
                actual_bpp = target_bits_per_sec / pixels_per_sec
                
                # Only check bitrate floor, ignore BPP floor if necessary
                if target_bitrate_kbps >= min_encoder_bitrate:
                    # Final validation before returning parameters
                    final_validation = self.bitrate_validator.validate_bitrate(target_bitrate_kbps, encoder)
                    if final_validation.is_valid:
                        logger.warning(f"Bitrate floor priority: {w}x{h}@{fps}fps: BPP={actual_bpp:.4f} (target: {bpp_min:.4f}), bitrate={target_bitrate_kbps}kbps >= {min_encoder_bitrate}kbps")
                        return {'width': w, 'height': h, 'fps': fps}
        
        # Third pass: Calculate minimum viable resolution for the given bitrate
        logger.error(f"Standard resolution candidates cannot meet bitrate floor {min_encoder_bitrate}kbps")
        return self._calculate_minimum_viable_resolution(video_info, target_bitrate_kbps, bpp_min, profile_cfg, encoder)
    
    def _progressive_resolution_reduction(self, video_info: Dict[str, Any], 
                                        target_bitrate_kbps: int, bpp_min: float,
                                        profile_cfg: Dict[str, Any], encoder: str) -> Dict[str, Any]:
        """Implement progressive resolution reduction until bitrate floor is met."""
        orig_w = video_info['width']
        orig_h = video_info['height']
        orig_fps = video_info['fps']
        min_encoder_bitrate = self.bitrate_validator.get_encoder_minimum(encoder)
        
        logger.info(f"Starting progressive resolution reduction from {orig_w}x{orig_h}")
        
        # Get fallback resolutions from bitrate validator
        fallback_resolutions = self.bitrate_validator.fallback_resolutions
        aspect_ratio = orig_w / orig_h
        
        # Generate progressive candidates based on aspect ratio
        progressive_candidates = []
        
        # Add fallback resolutions maintaining aspect ratio
        for fallback_w, fallback_h in fallback_resolutions:
            if fallback_w <= orig_w and fallback_h <= orig_h:
                # Adjust to maintain aspect ratio
                if abs(fallback_w / fallback_h - aspect_ratio) > 0.1:  # Significant aspect ratio difference
                    # Recalculate to maintain aspect ratio
                    adjusted_w = int(fallback_h * aspect_ratio)
                    adjusted_w = adjusted_w if adjusted_w % 2 == 0 else adjusted_w - 1
                    if adjusted_w >= 64:  # Minimum width
                        progressive_candidates.append((adjusted_w, fallback_h))
                else:
                    progressive_candidates.append((fallback_w, fallback_h))
        
        # Add ultra-low resolutions if needed
        ultra_low_heights = [120, 90, 72, 60]
        for h in ultra_low_heights:
            if h < orig_h:
                w = int(h * aspect_ratio)
                w = max(64, w if w % 2 == 0 else w - 1)  # Ensure minimum width and even dimensions
                progressive_candidates.append((w, h))
        
        # Remove duplicates and sort by resolution (largest first)
        progressive_candidates = list(set(progressive_candidates))
        progressive_candidates.sort(key=lambda x: x[0] * x[1], reverse=True)
        
        # Progressive FPS reduction candidates
        # Use config minimum FPS instead of hardcoded 10
        config_min_fps = self.config.get('video_compression.bitrate_validation.min_fps', 20)
        
        fps_candidates = [orig_fps, 24, 20, 15, 12, 8, 6, 5] if orig_fps > 6 else [orig_fps, 5, 4, 3, 2, 1]
        fps_candidates = [fps for fps in fps_candidates if fps <= orig_fps]
        # Filter to only include FPS values >= config_min_fps
        fps_candidates = [fps for fps in fps_candidates if fps >= config_min_fps]
        
        target_bits_per_sec = target_bitrate_kbps * 1000
        
        logger.info(f"Testing {len(progressive_candidates)} resolution candidates with {len(fps_candidates)} FPS options")
        
        # Try each resolution/FPS combination
        for i, (w, h) in enumerate(progressive_candidates):
            for fps in fps_candidates:
                pixels_per_sec = w * h * fps
                actual_bpp = target_bits_per_sec / pixels_per_sec
                
                # Check if this combination meets bitrate floor
                if target_bitrate_kbps >= min_encoder_bitrate:
                    # Validate with bitrate validator
                    validation_result = self.bitrate_validator.validate_bitrate(target_bitrate_kbps, encoder)
                    if validation_result.is_valid:
                        reduction_factor = (orig_w * orig_h) / (w * h)
                        logger.warning(f"Progressive reduction successful: {w}x{h}@{fps}fps")
                        logger.warning(f"Resolution reduction: {reduction_factor:.1f}x smaller ({orig_w}x{orig_h} → {w}x{h})")
                        logger.info(f"Final parameters: BPP={actual_bpp:.4f}, bitrate={target_bitrate_kbps}kbps >= {min_encoder_bitrate}kbps")
                        return {'width': w, 'height': h, 'fps': fps}
                
                # Log progress for very aggressive reductions
                if i % 3 == 0:  # Log every 3rd resolution attempt
                    logger.debug(f"Testing {w}x{h}@{fps}fps: BPP={actual_bpp:.4f}, bitrate={target_bitrate_kbps}kbps < {min_encoder_bitrate}kbps")
        
        # If we reach here, even progressive reduction failed
        logger.error("Progressive resolution reduction failed to meet bitrate floor")
        return self._handle_bitrate_floor_failure(video_info, target_bitrate_kbps, encoder)
    
    def _calculate_minimum_viable_resolution(self, video_info: Dict[str, Any], 
                                           target_bitrate_kbps: int, bpp_min: float,
                                           profile_cfg: Dict[str, Any], encoder: str) -> Dict[str, Any]:
        """Calculate the minimum viable resolution that can meet bitrate requirements."""
        min_encoder_bitrate = self.bitrate_validator.get_encoder_minimum(encoder)
        orig_w = video_info['width']
        orig_h = video_info['height']
        aspect_ratio = orig_w / orig_h
        
        logger.info("Calculating minimum viable resolution for bitrate floor compliance")
        
        # If target bitrate is below minimum, we need to work backwards
        if target_bitrate_kbps < min_encoder_bitrate:
            logger.error(f"Target bitrate {target_bitrate_kbps}kbps < minimum {min_encoder_bitrate}kbps")
            logger.error("Cannot calculate viable resolution - segmentation required")
            return self._handle_bitrate_floor_failure(video_info, target_bitrate_kbps, encoder)
        
        # Calculate minimum pixels per second needed for the given bitrate
        target_bits_per_sec = target_bitrate_kbps * 1000
        # Use config minimum FPS instead of hardcoded 5
        config_min_fps = self.config.get('video_compression.bitrate_validation.min_fps', 20)
        min_fps = config_min_fps  # Respect configured minimum FPS
        max_pixels_per_frame = target_bits_per_sec / (min_fps * bpp_min)
        
        # Calculate maximum resolution for this pixel budget
        max_height = int((max_pixels_per_frame / aspect_ratio) ** 0.5)
        max_width = int(max_height * aspect_ratio)
        
        # Ensure even dimensions and minimum size
        max_width = max(64, max_width if max_width % 2 == 0 else max_width - 1)
        max_height = max(48, max_height if max_height % 2 == 0 else max_height - 1)
        
        # Validate this calculated resolution
        pixels_per_sec = max_width * max_height * min_fps
        actual_bpp = target_bits_per_sec / pixels_per_sec
        
        if actual_bpp >= bpp_min and target_bitrate_kbps >= min_encoder_bitrate:
            logger.warning(f"Calculated minimum viable resolution: {max_width}x{max_height}@{min_fps}fps")
            logger.warning(f"Parameters: BPP={actual_bpp:.4f} >= {bpp_min:.4f}, bitrate={target_bitrate_kbps}kbps >= {min_encoder_bitrate}kbps")
            
            # Final validation
            validation_result = self.bitrate_validator.validate_bitrate(target_bitrate_kbps, encoder)
            if validation_result.is_valid:
                return {'width': max_width, 'height': max_height, 'fps': min_fps}
        
        # If calculated resolution still doesn't work, return failure
        logger.error(f"Calculated resolution {max_width}x{max_height}@{min_fps}fps still insufficient")
        return self._handle_bitrate_floor_failure(video_info, target_bitrate_kbps, encoder)
    
    def _handle_bitrate_floor_failure(self, video_info: Dict[str, Any], 
                                    target_bitrate_kbps: int, encoder: str) -> Dict[str, Any]:
        """Handle cases where bitrate floor cannot be met with any resolution/FPS combination."""
        min_encoder_bitrate = self.bitrate_validator.get_encoder_minimum(encoder)
        
        # Use absolute minimum resolution as final fallback
        min_width, min_height = 64, 48  # Absolute minimum viable resolution
        config_min_fps = self.config.get('video_compression.bitrate_validation.min_fps', 20)
        min_fps = config_min_fps  # Respect configured minimum even in emergency
        
        # Calculate final parameters
        pixels_per_sec = min_width * min_height * min_fps
        target_bits_per_sec = target_bitrate_kbps * 1000
        actual_bpp = target_bits_per_sec / pixels_per_sec if pixels_per_sec > 0 else 0
        
        # Log comprehensive failure information
        logger.error(f"CRITICAL: Cannot meet encoder minimum bitrate {min_encoder_bitrate}kbps with any resolution/FPS combination")
        logger.error(f"Final fallback: {min_width}x{min_height}@{min_fps}fps: BPP={actual_bpp:.4f}, bitrate={target_bitrate_kbps}kbps < {min_encoder_bitrate}kbps")
        
        # Calculate segmentation requirements
        if target_bitrate_kbps < min_encoder_bitrate:
            bitrate_deficit = min_encoder_bitrate - target_bitrate_kbps
            deficit_ratio = min_encoder_bitrate / max(target_bitrate_kbps, 1)
            logger.error(f"Bitrate deficit: {bitrate_deficit}kbps ({deficit_ratio:.1f}x below minimum)")
            
            # Suggest segmentation parameters
            duration = video_info.get('duration', 60)
            estimated_segments = math.ceil(deficit_ratio)
            segment_duration = duration / estimated_segments
            logger.error(f"SEGMENTATION REQUIRED: Estimated {estimated_segments} segments of ~{segment_duration:.1f}s each")
            
            # Log fallback strategy
            self.bitrate_validator.log_fallback_strategy_used(
                'segmentation',
                {'bitrate': target_bitrate_kbps, 'encoder': encoder},
                {'segment_duration': segment_duration, 'estimated_segments': estimated_segments},
                'bitrate_floor_failure',
                f'Bitrate {target_bitrate_kbps}kbps < minimum {min_encoder_bitrate}kbps'
            )
        
        logger.warning("Proceeding with absolute minimum resolution/FPS (encoding will likely fail)")
        logger.warning("Consider enabling segmentation or increasing target file size")
        
        return {'width': min_width, 'height': min_height, 'fps': min_fps}
    
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
        """Execute 2-pass x264 encode with given parameters and bitrate validation."""
        from ..ffmpeg_utils import FFmpegUtils
        
        log_base = os.path.join(self.temp_dir, f"cae_pass_{int(time.time())}")
        
        try:
            # Pass 1
            logger.info("Running 2-pass encode: pass 1/2")
            cmd_pass1 = FFmpegUtils.build_x264_two_pass_cae(
                input_path, output_path, params, pass_num=1, log_file=log_base,
                bitrate_validator=self.bitrate_validator
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
                input_path, output_path, params, pass_num=2, log_file=log_base,
                bitrate_validator=self.bitrate_validator
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
    
    def _calculate_refinement_strategy(
        self,
        quality_pass: bool,
        output_size_mb: float,
        size_limit_mb: float,
        initial_params: Dict[str, Any],
        audio_bitrate_kbps: int,
        video_info: Dict[str, Any],
        duration: float,
        bitrate_step: float,
        refine_pass: int,
        quality_result: Optional[Dict[str, Any]] = None,
        bpp_min: float = 0.035
    ) -> Dict[str, Any]:
        """
        Calculate refinement strategy using proper hierarchy: bitrate increase → FPS increase → resolution decrease.
        
        Returns:
            Dictionary with:
            - 'strategy': 'increase_bitrate', 'increase_fps', 'reduce_resolution', or 'exhausted'
            - 'adjusted_params': Updated parameters dict
            - 'reasoning': Log message explaining decision
        """
        # Get current parameters
        current_bitrate = initial_params['bitrate']
        current_fps = initial_params.get('fps', 30)
        current_width = initial_params['width']
        current_height = initial_params['height']
        encoder = initial_params.get('encoder', 'libx264')
        min_bitrate = self.bitrate_validator.get_encoder_minimum(encoder)
        config_min_fps = self.config.get('video_compression.bitrate_validation.min_fps', 20)
        
        # Get quality scores for logging
        vmaf_score = quality_result.get('vmaf_score') if quality_result else None
        ssim_score = quality_result.get('ssim_score') if quality_result else None
        vmaf_str = f"{vmaf_score:.2f}" if vmaf_score is not None else "N/A"
        ssim_str = f"{ssim_score:.4f}" if ssim_score is not None else "N/A"
        
        # Calculate current BPP
        pixels_per_sec = current_width * current_height * current_fps
        current_bpp = (current_bitrate * 1000.0) / pixels_per_sec if pixels_per_sec > 0 else 0
        
        logger.info(f"=== Refinement Strategy Decision (Pass {refine_pass + 1}) ===")
        logger.info(f"Current quality: VMAF={vmaf_str}, SSIM={ssim_str}")
        logger.info(f"Current params: {current_width}x{current_height}@{current_fps}fps, "
                   f"bitrate={current_bitrate}k, audio={audio_bitrate_kbps}k, BPP={current_bpp:.4f}")
        logger.info(f"Output size: {output_size_mb:.2f}MB / {size_limit_mb:.2f}MB limit")
        
        # Strategy 1: Try to increase bitrate (first priority)
        # Calculate available headroom
        headroom_mb = size_limit_mb - output_size_mb
        headroom_bits = headroom_mb * 8 * 1024 * 1024
        headroom_bitrate_kbps = int(headroom_bits / duration / 1000) if duration > 0 else 0
        
        # Try to reduce audio bitrate to free up space for video bitrate
        # Never reduce audio below 64kbps to prevent muffled/underwater sound
        min_audio_bitrate = 64  # Minimum acceptable audio bitrate (prevents audio corruption)
        current_audio = initial_params.get('audio_bitrate', audio_bitrate_kbps)
        audio_reduction_possible = current_audio > min_audio_bitrate
        
        if audio_reduction_possible:
            audio_savings_kbps = current_audio - min_audio_bitrate
            total_available_bitrate = headroom_bitrate_kbps + audio_savings_kbps
            logger.info(f"Audio reduction available: {current_audio}k → {min_audio_bitrate}k "
                       f"(saves {audio_savings_kbps}k for video)")
        else:
            total_available_bitrate = headroom_bitrate_kbps
        
        # Calculate proposed bitrate increase
        proposed_bitrate_step = int(current_bitrate * bitrate_step)
        max_safe_bitrate = current_bitrate + total_available_bitrate
        proposed_bitrate = min(proposed_bitrate_step, max_safe_bitrate)
        
        # Check if bitrate increase is viable
        if proposed_bitrate > current_bitrate and proposed_bitrate >= min_bitrate:
            # Calculate new BPP to ensure quality improvement
            new_bpp = (proposed_bitrate * 1000.0) / pixels_per_sec if pixels_per_sec > 0 else 0
            
            if new_bpp >= current_bpp:
                adjusted_params = initial_params.copy()
                adjusted_params['bitrate'] = proposed_bitrate
                
                if audio_reduction_possible:
                    adjusted_params['audio_bitrate'] = min_audio_bitrate
                    reasoning = (f"Increasing bitrate from {current_bitrate}k to {proposed_bitrate}k "
                               f"by reducing audio from {current_audio}k to {min_audio_bitrate}k "
                               f"(BPP: {current_bpp:.4f} → {new_bpp:.4f})")
                else:
                    reasoning = (f"Increasing bitrate from {current_bitrate}k to {proposed_bitrate}k "
                               f"(BPP: {current_bpp:.4f} → {new_bpp:.4f})")
                
                logger.info(f"Strategy selected: INCREASE_BITRATE - {reasoning}")
                return {
                    'strategy': 'increase_bitrate',
                    'adjusted_params': adjusted_params,
                    'reasoning': reasoning
                }
            else:
                logger.warning(f"Bitrate increase would decrease BPP ({current_bpp:.4f} → {new_bpp:.4f}), skipping")
        
        # Strategy 2: Try to increase FPS (second priority)
        # FPS increase candidates in order of preference (must be >= config minimum)
        fps_increase_candidates = []
        if current_fps < config_min_fps:
            # If below minimum, increase to minimum first
            fps_increase_candidates = [config_min_fps, 24, 30]
        elif current_fps < 24:
            fps_increase_candidates = [24, 30]
        elif current_fps < 30:
            fps_increase_candidates = [30]
        
        # Filter to only include FPS values >= config minimum and > current
        fps_increase_candidates = [fps for fps in fps_increase_candidates if fps >= config_min_fps and fps > current_fps]
        
        for new_fps in fps_increase_candidates:
            # Calculate required bitrate to maintain BPP with higher FPS
            new_pixels_per_sec = current_width * current_height * new_fps
            required_bitrate_for_bpp = int((current_bpp * new_pixels_per_sec) / 1000.0)
            
            # Check if we can afford the bitrate increase
            max_bitrate_with_fps = current_bitrate + total_available_bitrate
            viable_bitrate = min(required_bitrate_for_bpp, max_bitrate_with_fps)
            
            if viable_bitrate >= current_bitrate and viable_bitrate >= min_bitrate:
                new_bpp = (viable_bitrate * 1000.0) / new_pixels_per_sec if new_pixels_per_sec > 0 else 0
                
                if new_bpp >= current_bpp:
                    adjusted_params = initial_params.copy()
                    adjusted_params['fps'] = new_fps
                    adjusted_params['bitrate'] = viable_bitrate
                    
                    if audio_reduction_possible and viable_bitrate > current_bitrate:
                        adjusted_params['audio_bitrate'] = min_audio_bitrate
                        reasoning = (f"Increasing FPS from {current_fps}fps to {new_fps}fps "
                                   f"and bitrate from {current_bitrate}k to {viable_bitrate}k "
                                   f"by reducing audio from {current_audio}k to {min_audio_bitrate}k "
                                   f"(BPP: {current_bpp:.4f} → {new_bpp:.4f})")
                    else:
                        reasoning = (f"Increasing FPS from {current_fps}fps to {new_fps}fps "
                                   f"and bitrate from {current_bitrate}k to {viable_bitrate}k "
                                   f"(BPP: {current_bpp:.4f} → {new_bpp:.4f})")
                    
                    logger.info(f"Strategy selected: INCREASE_FPS - {reasoning}")
                    return {
                        'strategy': 'increase_fps',
                        'adjusted_params': adjusted_params,
                        'reasoning': reasoning
                    }
        
        # Strategy 3: Reduce resolution (last resort)
        # When reducing resolution, increase bitrate proportionally to maintain BPP
        new_width, new_height = self._scale_down_one_step(current_width, current_height)
        
        if new_height >= 360:  # Minimum resolution check
            # Calculate bitrate increase needed to maintain BPP
            new_pixels_per_sec = new_width * new_height * current_fps
            pixel_reduction_ratio = (new_width * new_height) / (current_width * current_height)
            
            # To maintain same BPP, we need proportionally less bitrate
            # But we want to increase bitrate to compensate for resolution loss
            # Target: maintain or improve BPP
            target_bpp = max(current_bpp, bpp_min)
            required_bitrate = int((target_bpp * new_pixels_per_sec) / 1000.0)
            
            # Calculate maximum bitrate we can afford
            max_bitrate = self._calculate_target_video_kbps(size_limit_mb, duration, 
                                                          min_audio_bitrate if audio_reduction_possible else current_audio)
            
            # Use the higher of required or max available
            new_bitrate = min(required_bitrate, max_bitrate)
            new_bitrate = max(new_bitrate, min_bitrate)  # Ensure minimum
            
            new_bpp = (new_bitrate * 1000.0) / new_pixels_per_sec if new_pixels_per_sec > 0 else 0
            
            if new_bpp >= current_bpp * 0.95:  # Allow slight BPP decrease (5%) if resolution improves efficiency
                adjusted_params = initial_params.copy()
                adjusted_params['width'] = new_width
                adjusted_params['height'] = new_height
                adjusted_params['bitrate'] = new_bitrate
                
                if audio_reduction_possible:
                    adjusted_params['audio_bitrate'] = min_audio_bitrate
                    reasoning = (f"Reducing resolution from {current_width}x{current_height} to {new_width}x{new_height} "
                               f"and increasing bitrate from {current_bitrate}k to {new_bitrate}k "
                               f"by reducing audio from {current_audio}k to {min_audio_bitrate}k "
                               f"(BPP: {current_bpp:.4f} → {new_bpp:.4f})")
                else:
                    reasoning = (f"Reducing resolution from {current_width}x{current_height} to {new_width}x{new_height} "
                               f"and increasing bitrate from {current_bitrate}k to {new_bitrate}k "
                               f"(BPP: {current_bpp:.4f} → {new_bpp:.4f})")
                
                logger.info(f"Strategy selected: REDUCE_RESOLUTION - {reasoning}")
                return {
                    'strategy': 'reduce_resolution',
                    'adjusted_params': adjusted_params,
                    'reasoning': reasoning
                }
            else:
                logger.warning(f"Resolution reduction would decrease BPP too much "
                             f"({current_bpp:.4f} → {new_bpp:.4f}), skipping")
        else:
            logger.warning(f"Cannot reduce resolution further (already at {current_width}x{current_height})")
        
        # No viable refinement strategy
        logger.warning("No viable refinement strategy found - all options exhausted")
        return {
            'strategy': 'exhausted',
            'adjusted_params': initial_params.copy(),
            'reasoning': 'All refinement strategies exhausted'
        }
    
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
        
        # Determine target file size early for session logging
        target_size_mb = max_size_mb or self.config.get('video_compression.max_file_size_mb', 10)
        
        # Start comprehensive compression session logging
        session_info = self._start_compression_session(input_path, target_size_mb, platform)
        
        try:
            # Check for configuration changes and reload if necessary
            if self.config.reload_config_if_changed():
                logger.info("Configuration reloaded due to file changes")
                self._log_compression_decision(
                    'config_reload',
                    'Configuration reloaded due to file changes',
                    {'config_file_changed': True},
                    'Configuration file was modified during execution'
                )
            
            # Log active configuration for debugging
            if self.debug_logging_enabled:
                logger.info("=== CONFIGURATION LOADING AND APPLICATION ===")
                
                # Log configuration source information
                config_dir = getattr(self.config, 'config_dir', 'unknown')
                logger.info(f"Configuration directory: {config_dir}")
                
                # Log configuration file timestamps if available
                if hasattr(self.config, '_config_file_timestamps') and self.config._config_file_timestamps:
                    logger.info("Configuration file timestamps:")
                    for config_file, timestamp in self.config._config_file_timestamps.items():
                        import datetime
                        dt = datetime.datetime.fromtimestamp(timestamp)
                        logger.info(f"  {config_file}: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
                
                self.config.log_active_configuration()
                
                # Log configuration loading details with enhanced context
                config_sections = {
                    'video_compression': {
                        'max_file_size_mb': self.config.get('video_compression.max_file_size_mb'),
                        'min_fps': self.config.get('video_compression.bitrate_validation.min_fps'),
                        'fps_reduction_steps': self.config.get('video_compression.bitrate_validation.fps_reduction_steps'),
                        'min_resolution': self.config.get('video_compression.bitrate_validation.min_resolution'),
                        'safety_margin': self.config.get('video_compression.bitrate_validation.safety_margin'),
                        'encoder_minimums': self.config.get('video_compression.bitrate_validation.encoder_minimums'),
                        'segmentation_threshold_mb': self.config.get('video_compression.bitrate_validation.segmentation_threshold_mb')
                    },
                    'hardware_acceleration': {
                        'nvidia': self.config.get('video_compression.hardware_acceleration.nvidia'),
                        'amd': self.config.get('video_compression.hardware_acceleration.amd'),
                        'fallback': self.config.get('video_compression.hardware_acceleration.fallback')
                    },
                    'quality_settings': {
                        'crf': self.config.get('video_compression.quality.crf'),
                        'preset': self.config.get('video_compression.quality.preset'),
                        'tune': self.config.get('video_compression.quality.tune')
                    },
                    'debug_logging': {
                        'enabled': self.config.get('video_compression.debug_logging.enabled'),
                        'performance_metrics': self.config.get('video_compression.debug_logging.performance_metrics'),
                        'compression_decisions': self.config.get('video_compression.debug_logging.compression_decisions'),
                        'fps_reduction_analysis': self.config.get('video_compression.debug_logging.fps_reduction_analysis'),
                        'configuration_loading': self.config.get('video_compression.debug_logging.configuration_loading')
                    },
                    'codec_settings': {
                        'allow_hevc': self.config.get('video_compression.codec.allow_hevc'),
                        'allow_h264_high10': self.config.get('video_compression.codec.allow_h264_high10'),
                        'priority': self.config.get('video_compression.codec.priority'),
                        'min_bpp': self.config.get('video_compression.codec.min_bpp')
                    }
                }
                
                for section_name, section_config in config_sections.items():
                    self._log_configuration_loading(section_name, section_config)
                
                # Log configuration validation status
                logger.info("Configuration validation:")
                config_issues = self.config.validate_configuration_values()
                if config_issues:
                    logger.info(f"  Found {len(config_issues)} validation issues")
                    for issue in config_issues:
                        logger.info(f"    - {issue}")
                else:
                    logger.info("  All configuration values valid")
                
                logger.info("=== END CONFIGURATION LOADING ===")
            
            # Validate configuration values
            config_issues = self.config.validate_configuration_values()
            if config_issues:
                logger.warning("Configuration validation issues found:")
                for issue in config_issues:
                    logger.warning(f"  - {issue}")
                
                self._log_compression_decision(
                    'config_validation_issues',
                    f'Found {len(config_issues)} configuration validation issues',
                    {'issues': config_issues},
                    'Configuration validation detected potential problems'
                )
            
            logger.info(f"Starting enhanced video compression: {input_path} -> {output_path}")
            
            # Get platform configuration
            platform_config = {}
            if platform:
                platform_config = self.config.get_platform_config(platform, 'video_compression')
                logger.info(f"Using platform configuration for: {platform}")
                
                self._log_compression_decision(
                    'platform_config',
                    f'Applied platform configuration for {platform}',
                    {'platform': platform, 'platform_config': platform_config},
                    f'Using platform-specific settings for {platform} optimization'
                )
            
            # Update target size with platform config if available
            target_size_mb = max_size_mb or platform_config.get('max_file_size_mb') or self.config.get('video_compression.max_file_size_mb', 10)
            
            if target_size_mb != session_info['target_size_mb']:
                self._log_parameter_change('target_size_mb', session_info['target_size_mb'], target_size_mb,
                                         'Updated target size based on platform configuration')
                session_info['target_size_mb'] = target_size_mb
        
            # Check cache for similar operations
            @self.performance_enhancer.cached_operation(ttl_hours=6)
            def cached_analysis(file_path, file_size, target_size):
                return self._analyze_video_content(file_path)
            
            # Early-cancel before analysis if shutdown requested
            if self.shutdown_requested:
                self._end_compression_session(False, output_path, error="Shutdown requested")
                return {'success': False, 'cancelled': True}

            # Get comprehensive video analysis (cached), but handle graceful cancellation
            file_size = os.path.getsize(input_path)
            try:
                with self._measure_operation_performance('video_analysis'):
                    video_info = cached_analysis(input_path, file_size, target_size_mb)
            except GracefulCancellation:
                self._end_compression_session(False, output_path, error="Graceful cancellation")
                return {'success': False, 'cancelled': True}
            original_size_mb = file_size / (1024 * 1024)
            
            # Log video analysis results
            self._log_compression_decision(
                'video_analysis_complete',
                f'Video analysis completed: {video_info["width"]}x{video_info["height"]}, {video_info["duration"]:.2f}s',
                {
                    'width': video_info.get('width'),
                    'height': video_info.get('height'),
                    'duration': video_info.get('duration'),
                    'fps': video_info.get('fps'),
                    'complexity_score': video_info.get('complexity_score'),
                    'motion_level': video_info.get('motion_level'),
                    'original_size_mb': original_size_mb
                },
                f'Analyzed video characteristics for compression planning'
            )
            
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
                    cae_result = self._compress_with_cae_discord_10mb(
                        input_path, output_path, video_info, platform_config, platform
                    )
                    if cae_result.get('success'):
                        final_size_mb = cae_result.get('compressed_size_mb') or cae_result.get('size_mb')
                        self._end_compression_session(True, output_path, final_size_mb)
                        return cae_result
                    logger.warning("CAE pipeline failed, falling back to standard compression")
                    
                    # If segmentation is last resort and CAE failed, try segmentation immediately
                    seg_last_resort = bool(self.config.get('video_compression.segmentation.only_if_single_file_unacceptable', False))
                    if seg_last_resort and not force_single_file:
                        logger.info("CAE pipeline failed; falling back to segmentation as last resort")
                        result = self._compress_with_segmentation(
                            input_path, output_path, target_size_mb, platform_config, video_info, platform
                        )
                        final_size_mb = result.get('compressed_size_mb') or result.get('size_mb')
                        self._end_compression_session(result.get('success', False), output_path, final_size_mb)
                        return result
                except Exception as e:
                    logger.error(f"CAE pipeline error: {e}, falling back to standard compression")
                    # If segmentation is last resort and CAE threw exception, try segmentation immediately
                    seg_last_resort = bool(self.config.get('video_compression.segmentation.only_if_single_file_unacceptable', False))
                    if seg_last_resort and not force_single_file:
                        logger.info("CAE pipeline error; falling back to segmentation as last resort")
                        result = self._compress_with_segmentation(
                            input_path, output_path, target_size_mb, platform_config, video_info, platform
                        )
                        final_size_mb = result.get('compressed_size_mb') or result.get('size_mb')
                        self._end_compression_session(result.get('success', False), output_path, final_size_mb)
                        return result
            
            # Check if segmentation should be considered now or deferred to last resort
            logger.info(f"Checking video segmentation: original size {original_size_mb:.1f}MB, target {target_size_mb}MB")
            logger.info(f"Video info keys: {list(video_info.keys())}")
            logger.info(f"Video info size_bytes: {video_info.get('size_bytes', 'NOT FOUND')}")

            # If forced single-file, skip segmentation entirely
            if force_single_file:
                logger.info("Segmentation disabled via --no-segmentation flag, forcing single-file processing")
                result = self._compress_with_aggressive_single_file(
                    input_path, output_path, target_size_mb, platform_config, video_info
                )
                final_size_mb = result.get('compressed_size_mb') or result.get('size_mb')
                self._end_compression_session(result.get('success', False), output_path, final_size_mb)
                return result

            seg_last_resort = bool(self.config.get('video_compression.segmentation.only_if_single_file_unacceptable', False))
            if not seg_last_resort:
                if self.video_segmenter.should_segment_video(video_info['duration'], video_info, target_size_mb):
                    logger.info("Video will be segmented instead of compressed as single file")
                    result = self._compress_with_segmentation(
                        input_path, output_path, target_size_mb, platform_config, video_info, platform
                    )
                    final_size_mb = result.get('compressed_size_mb') or result.get('size_mb')
                    self._end_compression_session(result.get('success', False), output_path, final_size_mb)
                    return result
            
            # If already under target size and no platform specified, just copy
            if original_size_mb <= target_size_mb and not platform:
                shutil.copy2(input_path, output_path)
                logger.info("Video already meets size requirements, copied without compression")
                result = self._get_compression_results(input_path, output_path, video_info, "copy")
                self._end_compression_session(True, output_path, original_size_mb)
                return result
            
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
                    final_size_mb = adv_result.get('compressed_size_mb') or adv_result.get('size_mb')
                    self._end_compression_session(True, output_path, final_size_mb)
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
                            final_size_mb = refined_result.get('compressed_size_mb') or refined_result.get('size_mb')
                            self._end_compression_session(True, output_path, final_size_mb)
                            return refined_result
                    except Exception:
                        pass
                    final_size_mb = std_result.get('compressed_size_mb') or std_result.get('size_mb')
                    self._end_compression_session(std_result.get('success', False), output_path, final_size_mb)
                    return std_result
                except Exception:
                    if seg_last_resort:
                        logger.info("Standard/advanced pipelines failed; falling back to segmentation as last resort")
                        result = self._compress_with_segmentation(
                            input_path, output_path, target_size_mb, platform_config, video_info, platform
                        )
                        final_size_mb = result.get('compressed_size_mb') or result.get('size_mb')
                        self._end_compression_session(result.get('success', False), output_path, final_size_mb)
                        return result
                    raise
            
            
            # Use adaptive quality processing for medium complexity files
            elif original_size_mb > 20:
                logger.info("Using adaptive quality processing for medium complexity file")
                result = self._compress_with_adaptive_quality(
                    input_path, output_path, target_size_mb, platform_config, video_info
                )
                final_size_mb = result.get('compressed_size_mb') or result.get('size_mb')
                self._end_compression_session(result.get('success', False), output_path, final_size_mb)
                return result

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
                            final_size_mb = refined_result.get('compressed_size_mb') or refined_result.get('size_mb')
                            self._end_compression_session(True, output_path, final_size_mb)
                            return refined_result
                    except Exception:
                        pass
                    final_size_mb = std_result.get('compressed_size_mb') or std_result.get('size_mb')
                    self._end_compression_session(std_result.get('success', False), output_path, final_size_mb)
                    return std_result
                except Exception:
                    if seg_last_resort:
                        logger.info("Standard pipeline failed; falling back to segmentation as last resort")
                        result = self._compress_with_segmentation(
                            input_path, output_path, target_size_mb, platform_config, video_info, platform
                        )
                        final_size_mb = result.get('compressed_size_mb') or result.get('size_mb')
                        self._end_compression_session(result.get('success', False), output_path, final_size_mb)
                        return result
                    raise
        
        except Exception as e:
            # Ensure session is always completed even on error
            self._end_compression_session(False, output_path, error=str(e))
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
            complexity_score = video_info.get('complexity_score', 5.0)
            has_hw_accel = self.hardware.has_hardware_acceleration()
            use_parallel = complexity_score >= 5.0 and has_hw_accel
            
            if self.debug_logging_enabled:
                logger.info(f"=== COMPRESSION STRATEGY SELECTION ===")
                logger.info(f"Strategy selection factors:")
                logger.info(f"  Complexity score: {complexity_score:.2f}")
                logger.info(f"  Hardware acceleration available: {has_hw_accel}")
                logger.info(f"  Parallel execution threshold: 5.0")
                logger.info(f"  Use parallel execution: {use_parallel}")
                
                if use_parallel:
                    logger.info(f"Parallel execution selected: high complexity video with hardware acceleration")
                else:
                    if complexity_score < 5.0:
                        logger.info(f"Sequential execution selected: complexity score {complexity_score:.2f} below threshold")
                    if not has_hw_accel:
                        logger.info(f"Sequential execution selected: no hardware acceleration available")
            
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
                
                if self.debug_logging_enabled:
                    self._log_compression_decision(
                        'strategy_selection',
                        'Strategy 1: Content-aware optimal compression',
                        {
                            'strategy_name': 'content_aware',
                            'reasoning': 'First attempt using content-aware analysis for optimal parameters',
                            'complexity_score': complexity_score,
                            'target_size_mb': target_size_mb
                        },
                        'Content-aware compression analyzes video characteristics to select optimal encoding parameters'
                    )
                
                with self._measure_operation_performance('strategy_1_content_aware'):
                    strategy1_result = self._compress_with_content_awareness(
                        input_path, video_info, platform_config, target_size_mb
                    )
                
                if strategy1_result:
                    if self.debug_logging_enabled:
                        logger.info(f"Strategy 1 initial result: {strategy1_result['size_mb']:.2f}MB")
                    
                    # Refine if under-utilizing target
                    strategy1_result = self._refine_strategy_result(strategy1_result, input_path, video_info, target_size_mb)
                    # Validate quality and attempt uplift if needed
                    strategy1_result = self._validate_and_uplift_result(strategy1_result, input_path, video_info, target_size_mb)
                    if strategy1_result:  # Only process validated results
                        compression_attempts.append(strategy1_result)
                        if strategy1_result['size_mb'] <= target_size_mb:
                            best_result = strategy1_result
                            logger.info(f"Strategy 1 successful: {strategy1_result['size_mb']:.2f}MB")
                            
                            if self.debug_logging_enabled:
                                self._log_compression_decision(
                                    'strategy_success',
                                    'Strategy 1 succeeded within target size',
                                    {
                                        'final_size_mb': strategy1_result['size_mb'],
                                        'target_size_mb': target_size_mb,
                                        'utilization': (strategy1_result['size_mb'] / target_size_mb) * 100,
                                        'quality_score': strategy1_result.get('quality_score', 'unknown')
                                    },
                                    'Content-aware compression achieved target size with acceptable quality'
                                )
                        else:
                            if self.debug_logging_enabled:
                                self._log_compression_decision(
                                    'strategy_partial_success',
                                    'Strategy 1 completed but exceeded target size',
                                    {
                                        'actual_size_mb': strategy1_result['size_mb'],
                                        'target_size_mb': target_size_mb,
                                        'overage_mb': strategy1_result['size_mb'] - target_size_mb,
                                        'overage_percent': ((strategy1_result['size_mb'] / target_size_mb) - 1) * 100
                                    },
                                    'Content-aware compression completed but requires additional optimization'
                                )
                else:
                    if self.debug_logging_enabled:
                        self._log_compression_decision(
                            'strategy_failure',
                            'Strategy 1 failed to produce valid result',
                            {'strategy_name': 'content_aware'},
                            'Content-aware compression failed, will try alternative strategies'
                        )
                
                # Strategy 2: Two-pass encoding if Strategy 1 didn't work perfectly
                strategy1_size = strategy1_result.get('size_mb', 0) if strategy1_result else 0
                target_threshold = target_size_mb * 0.95
                should_try_strategy2 = not best_result or strategy1_size > target_threshold
                
                if should_try_strategy2:
                    logger.info("Attempting Strategy 2: Two-pass precision encoding")
                    
                    if self.debug_logging_enabled:
                        strategy2_reasoning = []
                        if not best_result:
                            strategy2_reasoning.append("No successful result from Strategy 1")
                        if strategy1_size > target_threshold:
                            strategy2_reasoning.append(f"Strategy 1 result ({strategy1_size:.2f}MB) exceeds 95% threshold ({target_threshold:.2f}MB)")
                        
                        self._log_compression_decision(
                            'strategy_selection',
                            'Strategy 2: Two-pass precision encoding',
                            {
                                'strategy_name': 'two_pass',
                                'reasoning': '; '.join(strategy2_reasoning),
                                'strategy1_size_mb': strategy1_size,
                                'target_threshold_mb': target_threshold,
                                'has_previous_result': best_result is not None
                            },
                            'Two-pass encoding provides more precise bitrate control for better size targeting'
                        )
                    
                    with self._measure_operation_performance('strategy_2_two_pass'):
                        strategy2_result = self._compress_with_two_pass(
                            input_path, video_info, platform_config, target_size_mb, best_result
                        )
                    
                    if strategy2_result:
                        if self.debug_logging_enabled:
                            logger.info(f"Strategy 2 initial result: {strategy2_result['size_mb']:.2f}MB")
                        
                        # Refine if under-utilizing target
                        strategy2_result = self._refine_strategy_result(strategy2_result, input_path, video_info, target_size_mb)
                        # Validate quality and attempt uplift if needed
                        strategy2_result = self._validate_and_uplift_result(strategy2_result, input_path, video_info, target_size_mb)
                        if strategy2_result:  # Only process validated results
                            compression_attempts.append(strategy2_result)
                            if strategy2_result['size_mb'] <= target_size_mb:
                                strategy2_quality = strategy2_result.get('quality_score', 0)
                                best_quality = best_result.get('quality_score', 0) if best_result else 0
                                
                                if not best_result or strategy2_quality > best_quality:
                                    best_result = strategy2_result
                                    logger.info(f"Strategy 2 improved result: {strategy2_result['size_mb']:.2f}MB")
                                    
                                    if self.debug_logging_enabled:
                                        self._log_compression_decision(
                                            'strategy_improvement',
                                            'Strategy 2 improved upon previous result',
                                            {
                                                'new_size_mb': strategy2_result['size_mb'],
                                                'new_quality_score': strategy2_quality,
                                                'previous_quality_score': best_quality,
                                                'quality_improvement': strategy2_quality - best_quality
                                            },
                                            'Two-pass encoding achieved better quality while meeting size constraints'
                                        )
                                else:
                                    if self.debug_logging_enabled:
                                        self._log_compression_decision(
                                            'strategy_no_improvement',
                                            'Strategy 2 met size target but did not improve quality',
                                            {
                                                'strategy2_size_mb': strategy2_result['size_mb'],
                                                'strategy2_quality': strategy2_quality,
                                                'best_quality': best_quality
                                            },
                                            'Two-pass result acceptable but previous result had better quality'
                                        )
                            else:
                                if self.debug_logging_enabled:
                                    self._log_compression_decision(
                                        'strategy_failure',
                                        'Strategy 2 exceeded target size',
                                        {
                                            'actual_size_mb': strategy2_result['size_mb'],
                                            'target_size_mb': target_size_mb,
                                            'overage_mb': strategy2_result['size_mb'] - target_size_mb
                                        },
                                        'Two-pass encoding could not achieve target size'
                                    )
                    else:
                        if self.debug_logging_enabled:
                            self._log_compression_decision(
                                'strategy_failure',
                                'Strategy 2 failed to produce valid result',
                                {'strategy_name': 'two_pass'},
                                'Two-pass encoding failed, will try alternative strategies'
                            )
            
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
            
            # Strategy 5: Try alternative strategies if no good result yet
            if not best_result or best_result.get('size_mb', 0) > target_size_mb:
                logger.info("Attempting Strategy 5: Alternative compression strategies")
                
                if self.debug_logging_enabled:
                    self._log_compression_decision(
                        'strategy_selection',
                        'Strategy 5: Alternative compression strategies',
                        {
                            'strategy_name': 'alternative_strategies',
                            'reasoning': 'Previous strategies failed to meet target size, trying alternatives to FPS reduction',
                            'has_previous_result': best_result is not None,
                            'previous_size_mb': best_result.get('size_mb', 0) if best_result else 0
                        },
                        'Alternative strategies avoid aggressive FPS reduction by using resolution/quality adjustments'
                    )
                
                with self._measure_operation_performance('strategy_5_alternatives'):
                    alternative_result = self._try_alternative_strategies_when_fps_insufficient(
                        input_path, output_path, video_info, target_size_mb, best_result
                    )
                
                if alternative_result and alternative_result.get('success'):
                    if alternative_result.get('size_mb', 0) <= target_size_mb:
                        best_result = alternative_result
                        logger.info(f"Alternative strategy successful: {alternative_result['size_mb']:.2f}MB")
                        
                        if self.debug_logging_enabled:
                            self._log_compression_decision(
                                'strategy_success',
                                'Alternative strategy succeeded',
                                {
                                    'strategy_used': alternative_result.get('strategy', 'unknown'),
                                    'final_size_mb': alternative_result['size_mb'],
                                    'target_size_mb': target_size_mb,
                                    'alternative_strategy_used': True
                                },
                                'Alternative compression strategy achieved target size while preserving motion quality'
                            )
                    else:
                        logger.info(f"Alternative strategy completed but exceeded target: {alternative_result['size_mb']:.2f}MB")
            
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
        """Content-aware compression strategy using smart compression strategy"""
        try:
            temp_output = os.path.join(self.temp_dir, "content_aware_output.mp4")
            
            # Use smart compression strategy to calculate optimal parameters
            params = self._calculate_compression_params_with_smart_strategy(
                video_info, platform_config, target_size_mb, CompressionStrategy.BALANCED
            )
            
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
                    'strategy': 'content_aware_smart',
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
        """Compress video using segmentation approach with bitrate validation"""
        try:
            logger.info("Starting video segmentation compression")
            
            # Check if segmentation is needed due to bitrate constraints
            encoder = self.hardware.get_best_encoder("h264")[0]
            should_segment_bitrate, bitrate_reason = self.video_segmenter.should_segment_for_bitrate_constraints(
                video_info, target_size_mb, encoder
            )
            
            if should_segment_bitrate:
                logger.info(f"Using bitrate-constrained segmentation: {bitrate_reason}")
                # Use bitrate-specific segmentation
                result = self.video_segmenter.segment_video_for_bitrate_constraints(
                    input_video=input_path,
                    output_base_path=output_path,
                    encoder=encoder,
                    target_size_mb=target_size_mb,
                    video_info=video_info
                )
            else:
                logger.info(f"Using standard segmentation: {bitrate_reason}")
                # Use standard segmentation
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
                    
                    # Get output_folder from result, with fallback construction
                    output_folder = result.get('output_folder')
                    if not output_folder:
                        # Fallback: construct path from output_path
                        output_dir = os.path.dirname(output_path)
                        base_name = os.path.splitext(os.path.basename(output_path))[0]
                        output_folder = os.path.join(output_dir, f"{base_name}_segments")
                        logger.info(f"output_folder missing from result, constructed fallback: {output_folder}")
                    
                    # Convert to absolute path and validate
                    output_folder = os.path.abspath(output_folder)
                    if not os.path.exists(output_folder):
                        logger.error(f"Segments folder does not exist: {output_folder}")
                        # Try to create it if it doesn't exist
                        try:
                            os.makedirs(output_folder, exist_ok=True)
                            logger.info(f"Created missing segments folder: {output_folder}")
                        except Exception as e:
                            logger.error(f"Failed to create segments folder {output_folder}: {e}")
                    else:
                        logger.debug(f"Segments folder validated: {output_folder}")
                    
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
                        'output_folder': output_folder,
                        'segment_duration': result.get('segment_duration', 0),
                        'is_segmented_output': True  # Flag to indicate multiple output files
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
    
    def _try_segmentation_fallback(self, input_path: str, output_path: str, target_size_mb: float,
                                 platform_config: Dict[str, Any], video_info: Dict[str, Any],
                                 encoder: str, failure_reason: str = "") -> Dict[str, Any]:
        """
        Try segmentation as fallback when single-file compression fails
        
        Args:
            input_path: Input video path
            output_path: Output path (will be converted to segments folder)
            target_size_mb: Target size per segment
            platform_config: Platform configuration
            video_info: Video metadata
            encoder: Encoder that failed
            failure_reason: Reason for single-file failure
            
        Returns:
            Compression result dictionary with segment information
        """
        logger.info(f"Attempting segmentation fallback due to: {failure_reason}")
        
        # Check if segmentation is likely to help
        should_segment, reason = self.video_segmenter.should_segment_for_bitrate_constraints(
            video_info, target_size_mb, encoder
        )
        
        if not should_segment:
            logger.warning(f"Segmentation may not resolve the issue: {reason}")
            # Still try segmentation as last resort
        
        try:
            # Create segments folder path from output path
            output_dir = os.path.dirname(output_path)
            base_name = os.path.splitext(os.path.basename(output_path))[0]
            segments_folder = os.path.join(output_dir, f"{base_name}_segments")
            
            # Use bitrate-constrained segmentation since we know there's a bitrate issue
            result = self.video_segmenter.segment_video_for_bitrate_constraints(
                input_video=input_path,
                output_base_path=segments_folder,
                encoder=encoder,
                target_size_mb=target_size_mb,
                video_info=video_info
            )
            
            if result.get('success', False):
                # Check if all segments passed bitrate validation
                all_valid = result.get('all_segments_valid', True)
                segments = result.get('segments', [])
                
                if all_valid:
                    logger.info("Segmentation fallback successful - all segments meet bitrate requirements")
                else:
                    logger.warning("Segmentation fallback partially successful - some segments may have bitrate issues")
                
                # Organize segments with proper naming
                organized_result = self._organize_fallback_segments(
                    segments, segments_folder, base_name, target_size_mb
                )
                
                if organized_result.get('success', False):
                    total_size = sum(segment.get('size_mb', 0) for segment in organized_result['segments'])
                    
                    # Get segments_folder and ensure it's absolute and validated
                    # Save original path as fallback
                    original_segments_folder = segments_folder
                    segments_folder = organized_result.get('segments_folder')
                    if not segments_folder:
                        # Fallback: use the constructed path
                        segments_folder = original_segments_folder
                        logger.warning(f"segments_folder missing from organized_result, using fallback: {segments_folder}")
                    
                    # Convert to absolute path and validate
                    segments_folder = os.path.abspath(segments_folder)
                    if not os.path.exists(segments_folder):
                        logger.error(f"Segments folder does not exist: {segments_folder}")
                        # Try to create it if it doesn't exist
                        try:
                            os.makedirs(segments_folder, exist_ok=True)
                            logger.info(f"Created missing segments folder: {segments_folder}")
                        except Exception as e:
                            logger.error(f"Failed to create segments folder {segments_folder}: {e}")
                            # Return error if we can't create the folder
                            return {
                                'success': False,
                                'error': f"Segments folder does not exist and could not be created: {segments_folder}",
                                'input_file': input_path
                            }
                    else:
                        logger.debug(f"Segments folder validated: {segments_folder}")
                    
                    return {
                        'success': True,
                        'input_file': input_path,
                        'output_file': segments_folder,  # Return folder path for multiple outputs
                        'method': 'segmentation_fallback',
                        'original_size_mb': os.path.getsize(input_path) / (1024 * 1024),
                        'compressed_size_mb': total_size,
                        'size_mb': total_size,
                        'compression_ratio': ((os.path.getsize(input_path) - (total_size * 1024 * 1024)) / os.path.getsize(input_path)) * 100,
                        'space_saved_mb': (os.path.getsize(input_path) - (total_size * 1024 * 1024)) / (1024 * 1024),
                        'video_info': video_info,
                        'optimization_strategy': 'segmentation_fallback',
                        'quality_score': 8.0,  # High quality preserved in segments
                        'attempts_made': 1,
                        'encoder_used': encoder,
                        'segments': organized_result['segments'],
                        'segments_folder': segments_folder,
                        'num_segments': len(organized_result['segments']),
                        'all_segments_valid': all_valid,
                        'bitrate_validations': result.get('bitrate_validations', []),
                        'fallback_reason': failure_reason,
                        'is_segmented_output': True  # Flag to indicate multiple output files
                    }
                else:
                    logger.error(f"Failed to organize segmentation fallback outputs: {organized_result.get('error', 'Unknown error')}")
                    return {
                        'success': False,
                        'error': f"Segmentation organization failed: {organized_result.get('error', 'Unknown error')}",
                        'method': 'segmentation_fallback_organization_failed',
                        'fallback_reason': failure_reason
                    }
            else:
                logger.error(f"Segmentation fallback failed: {result.get('error', 'Unknown error')}")
                return {
                    'success': False,
                    'error': f"Segmentation fallback failed: {result.get('error', 'Unknown error')}",
                    'method': 'segmentation_fallback_failed',
                    'fallback_reason': failure_reason
                }
                
        except Exception as e:
            logger.error(f"Exception during segmentation fallback: {e}")
            return {
                'success': False,
                'error': f"Segmentation fallback exception: {str(e)}",
                'method': 'segmentation_fallback_exception',
                'fallback_reason': failure_reason
            }

    def _organize_fallback_segments(self, segments: List[Dict[str, Any]], segments_folder: str, 
                                  base_name: str, target_size_mb: float) -> Dict[str, Any]:
        """
        Organize segmentation fallback outputs with proper naming and validation
        
        Args:
            segments: List of segment information dictionaries
            segments_folder: Path to segments folder
            base_name: Base name for segment files
            target_size_mb: Target size per segment for validation
            
        Returns:
            Dictionary with organization results
        """
        try:
            # Ensure segments folder exists
            os.makedirs(segments_folder, exist_ok=True)
            
            organized_segments = []
            failed_segments = []
            
            for i, segment in enumerate(segments):
                try:
                    segment_path = segment.get('path', '')
                    if not segment_path or not os.path.exists(segment_path):
                        logger.warning(f"Segment {i+1} path not found: {segment_path}")
                        failed_segments.append(segment)
                        continue
                    
                    # Generate consistent segment naming for batch processing
                    segment_filename = f"{base_name}_segment_{i+1:03d}.mp4"
                    final_segment_path = os.path.join(segments_folder, segment_filename)
                    
                    # Move segment to final location if not already there
                    if segment_path != final_segment_path:
                        if os.path.exists(final_segment_path):
                            os.remove(final_segment_path)
                        shutil.move(segment_path, final_segment_path)
                    
                    # Validate segment size
                    segment_size_mb = os.path.getsize(final_segment_path) / (1024 * 1024)
                    
                    # Update segment information
                    organized_segment = {
                        'index': i + 1,
                        'path': final_segment_path,
                        'filename': segment_filename,
                        'size_mb': segment_size_mb,
                        'start_time': segment.get('start_time', 0),
                        'duration': segment.get('duration', 0),
                        'method': segment.get('method', 'segmentation_fallback'),
                        'quality_score': segment.get('quality_score', 8.0),
                        'bitrate_validated': segment.get('bitrate_validated', True)
                    }
                    
                    organized_segments.append(organized_segment)
                    logger.info(f"Organized segment {i+1}: {segment_filename} ({segment_size_mb:.2f}MB)")
                    
                except Exception as e:
                    logger.error(f"Failed to organize segment {i+1}: {e}")
                    failed_segments.append(segment)
            
            if not organized_segments:
                return {
                    'success': False,
                    'error': 'No segments could be organized successfully',
                    'failed_segments': failed_segments
                }
            
            # Create canonical segments summary for batch processing
            try:
                write_segments_summary(
                    segments_folder,
                    base_name,
                    logger=logger,
                )
            except Exception as e:
                logger.warning(f"Failed to create segments summary: {e}")
            
            logger.info(f"Successfully organized {len(organized_segments)} segments in {segments_folder}")
            
            return {
                'success': True,
                'segments': organized_segments,
                'segments_folder': segments_folder,
                'num_segments': len(organized_segments),
                'failed_segments': failed_segments,
                'total_size_mb': sum(s.get('size_mb', 0) for s in organized_segments)
            }
            
        except Exception as e:
            logger.error(f"Exception during segment organization: {e}")
            return {
                'success': False,
                'error': f"Segment organization exception: {str(e)}"
            }

    def _ensure_consistent_segment_naming(self, segments_folder: str, base_name: str) -> bool:
        """
        Ensure all segments in the folder follow consistent naming convention for batch processing
        
        Args:
            segments_folder: Path to segments folder
            base_name: Base name for segment files
            
        Returns:
            True if naming is consistent or was successfully updated
        """
        try:
            if not os.path.exists(segments_folder):
                return False
            
            # Find all video segments
            segment_files = []
            for file in os.listdir(segments_folder):
                if file.endswith('.mp4') and 'segment' in file.lower():
                    segment_files.append(file)
            
            if not segment_files:
                return True  # No segments to rename
            
            # Sort segments by their current names to maintain order
            segment_files.sort()
            
            renamed_count = 0
            for i, old_filename in enumerate(segment_files):
                expected_filename = f"{base_name}_segment_{i+1:03d}.mp4"
                
                if old_filename != expected_filename:
                    old_path = os.path.join(segments_folder, old_filename)
                    new_path = os.path.join(segments_folder, expected_filename)
                    
                    # Avoid conflicts by using temporary name if needed
                    if os.path.exists(new_path):
                        temp_path = os.path.join(segments_folder, f"temp_{expected_filename}")
                        shutil.move(old_path, temp_path)
                        if os.path.exists(new_path):
                            os.remove(new_path)
                        shutil.move(temp_path, new_path)
                    else:
                        shutil.move(old_path, new_path)
                    
                    renamed_count += 1
                    logger.info(f"Renamed segment: {old_filename} -> {expected_filename}")
            
            if renamed_count > 0:
                logger.info(f"Updated {renamed_count} segment names for consistent batch processing")
            
            return True
            
        except Exception as e:
            logger.error(f"Error ensuring consistent segment naming: {e}")
            return False

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
            
            # Calculate FPS - respect configured minimum
            config_min_fps = self.config.get('video_compression.bitrate_validation.min_fps', 20)
            aggressive_fps = max(config_min_fps, int(video_info['fps'] * config['fps_scale']))
            
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
        
        # Calculate initial bitrate estimate
        estimated_bitrate = self._calculate_content_aware_bitrate(video_info, target_size_mb)
        
        # Select optimal codec (considering HEVC when beneficial)
        encoder, accel_type = self._select_optimal_codec(video_info, target_size_mb, estimated_bitrate)
        
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
        
        # Motion-aware preset selection with size headroom consideration
        original_size_mb = video_info.get('size_bytes', 0) / (1024 * 1024)
        size_headroom = (target_size_mb / original_size_mb) if original_size_mb > 0 else 1.0
        params['preset'] = self._select_optimal_preset(video_info, size_headroom, motion_level)
        
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
        """Calculate parameters for two-pass encoding with enhanced psychovisual tuning and VBV optimization"""
        # Force software encoding for two-pass; hardware encoders generally don't support it reliably
        encoder, accel_type = self.hardware.get_best_encoder("h264")
        if accel_type != 'software':
            encoder = 'libx264'
            accel_type = 'software'
        
        # Get content characteristics for adaptive tuning
        complexity = video_info.get('complexity_score', 5.0)
        motion_level = video_info.get('motion_level', 'medium')
        duration = video_info.get('duration', 0)
        
        # Calculate optimal preset based on content and size headroom
        target_size_mb = (target_bitrate * duration / 1000) / 8  # Rough estimate
        original_size_mb = video_info.get('size_bytes', 0) / (1024 * 1024)
        # Size headroom: ratio of target to original (higher = more headroom, lower = tighter constraints)
        # If original_size_mb is 0 or very small, assume we have headroom
        size_headroom = (target_size_mb / original_size_mb) if original_size_mb > 0.1 else 1.0
        
        params = {
            'encoder': encoder,
            'acceleration_type': accel_type,
            'width': video_info['width'],
            'height': video_info['height'],
            'fps': min(video_info['fps'], 30),
            'bitrate': target_bitrate,
            'preset': self._select_optimal_preset(video_info, size_headroom, motion_level),
            'tune': self.config.get('video_compression.quality.tune', 'film'),
        }
        
        # Enhanced psychovisual tuning with AQ mode 3
        params.update(self._calculate_psychovisual_params(video_info, complexity, motion_level, target_bitrate, encoder))
        
        # VBV optimization for better rate control
        params.update(self._calculate_optimal_vbv_params(video_info, target_bitrate, complexity))
        
        # GOP structure optimization
        params.update(self._calculate_optimal_gop_structure(video_info, motion_level))
        
        # Adjust based on previous attempt
        if previous_result and previous_result.get('size_mb', 0) > 0:
            previous_params = previous_result.get('params', {})
            # Fine-tune bitrate based on previous result
            if previous_result['size_mb'] > target_bitrate * video_info['duration'] / (8 * 1024):
                params['bitrate'] = int(target_bitrate * 0.9)  # Reduce bitrate
                # Recalculate VBV params with new bitrate
                params.update(self._calculate_optimal_vbv_params(video_info, params['bitrate'], complexity))
        
        # Apply platform constraints
        if platform_config:
            params.update(self._apply_platform_constraints(params, platform_config))
        
        return params
    
    def _select_optimal_preset(self, video_info: Dict[str, Any], size_headroom: float, 
                              motion_level: str) -> str:
        """Select optimal preset based on content characteristics and size headroom"""
        # Base preset from config
        base_preset = self.config.get('video_compression.quality.preset', 'slower')
        
        # If we have good size headroom (>80% of target), use slower presets for better quality
        if size_headroom > 0.8:
            if motion_level == 'low':
                # Low motion can benefit from slower presets
                return 'slower' if base_preset in ['slow', 'slower', 'veryslow'] else 'slow'
            elif motion_level == 'high':
                # High motion needs faster presets to maintain smoothness
                return 'medium'
            else:
                return 'slow'
        elif size_headroom > 0.5:
            # Moderate headroom - balanced approach
            if motion_level == 'low':
                return 'slow'
            else:
                return 'medium'
        else:
            # Tight size constraints - use faster presets
            return 'medium' if base_preset in ['slow', 'slower', 'veryslow'] else base_preset
    
    def _calculate_psychovisual_params(self, video_info: Dict[str, Any], 
                                       complexity: float, motion_level: str,
                                       target_bitrate: int, encoder: str) -> Dict[str, Any]:
        """Calculate enhanced psychovisual tuning parameters with AQ mode 3"""
        params = {}
        
        # AQ mode 3 (advanced adaptive quantization) for better quality at lower bitrates
        # Enable AQ mode 3 when complexity is high or bitrate is constrained
        # Only for libx264 (software H.264 encoder)
        aq_mode = 3 if (encoder == 'libx264' and (complexity > 6.0 or target_bitrate < 1000)) else 2
        params['aq_mode'] = aq_mode
        
        # Adaptive AQ strength based on content complexity
        base_aq_strength = float(self.config.get('video_compression.profiles.discord_10mb.x264.aq_strength', 1.1))
        if complexity > 7.0:
            # High complexity needs stronger AQ
            aq_strength = base_aq_strength * 1.1
        elif complexity < 3.0:
            # Low complexity can use lower AQ strength
            aq_strength = base_aq_strength * 0.9
        else:
            aq_strength = base_aq_strength
        params['aq_strength'] = round(aq_strength, 2)
        
        # Qcomp (quantizer curve compression) - 0.6-0.7 range for better rate control
        base_qcomp = float(self.config.get('video_compression.profiles.discord_10mb.x264.qcomp', 0.65))
        if motion_level == 'high':
            # High motion benefits from slightly lower qcomp (more consistent quality)
            qcomp = max(0.6, base_qcomp - 0.05)
        elif motion_level == 'low':
            # Low motion can use higher qcomp (better compression)
            qcomp = min(0.7, base_qcomp + 0.05)
        else:
            qcomp = base_qcomp
        params['qcomp'] = round(qcomp, 2)
        
        # RC lookahead - adaptive based on motion and complexity
        base_rc_lookahead = int(self.config.get('video_compression.profiles.discord_10mb.x264.rc_lookahead', 40))
        if motion_level == 'high':
            # High motion needs more lookahead for better prediction
            rc_lookahead = min(60, base_rc_lookahead + 10)
        elif complexity > 7.0:
            # High complexity benefits from more lookahead
            rc_lookahead = min(60, base_rc_lookahead + 10)
        else:
            rc_lookahead = base_rc_lookahead
        params['rc_lookahead'] = rc_lookahead
        
        # Psy-rd and psy-trellis for perceptual quality (libx264 only)
        # These improve perceptual quality by preserving details that matter most
        if encoder == 'libx264':
            fps = video_info.get('fps', 30)
            if fps >= 24:
                # Psy-rd: perceptual rate-distortion optimization (0.0-4.0, typical 1.0-1.3)
                psy_rd = 1.2 if complexity > 6.0 else 1.0
                params['psy_rd'] = round(psy_rd, 2)
                
                # Psy-trellis: perceptual trellis quantization (0.0-0.15, typical 0.0-0.1)
                psy_trellis = 0.05 if complexity > 6.0 else 0.0
                params['psy_trellis'] = round(psy_trellis, 2)
        
        return params
    
    def _calculate_optimal_vbv_params(self, video_info: Dict[str, Any], 
                                     target_bitrate: int, complexity: float) -> Dict[str, Any]:
        """Calculate optimal VBV (Video Buffering Verifier) parameters for better rate control"""
        params = {}
        
        # Base multipliers from config
        base_maxrate_mult = float(self.config.get('video_compression.profiles.discord_10mb.x264.maxrate_multiplier', 1.10))
        base_bufsize_mult = float(self.config.get('video_compression.profiles.discord_10mb.x264.bufsize_multiplier', 2.0))
        
        # Adjust based on content complexity
        if complexity > 7.0:
            # High complexity needs larger buffer to handle peaks
            maxrate_mult = base_maxrate_mult * 1.1
            bufsize_mult = base_bufsize_mult * 1.2
        elif complexity < 3.0:
            # Low complexity can use tighter VBV
            maxrate_mult = base_maxrate_mult * 0.95
            bufsize_mult = base_bufsize_mult * 0.9
        else:
            maxrate_mult = base_maxrate_mult
            bufsize_mult = base_bufsize_mult
        
        # Adjust based on bitrate level
        if target_bitrate < 800:
            # Low bitrate needs tighter VBV to prevent underflows
            maxrate_mult *= 0.95
            bufsize_mult *= 0.9
        elif target_bitrate > 2000:
            # High bitrate can use more relaxed VBV
            maxrate_mult *= 1.05
            bufsize_mult *= 1.1
        
        params['maxrate_multiplier'] = round(maxrate_mult, 2)
        params['bufsize_multiplier'] = round(bufsize_mult, 2)
        
        # Calculate actual maxrate and bufsize
        params['maxrate'] = int(target_bitrate * maxrate_mult)
        params['bufsize'] = int(target_bitrate * bufsize_mult)
        
        # Ensure minimum buffer size for smooth playback (at least 1 second)
        min_bufsize = target_bitrate  # 1 second at target bitrate
        params['bufsize'] = max(params['bufsize'], min_bufsize)
        
        return params
    
    def _calculate_optimal_gop_structure(self, video_info: Dict[str, Any], 
                                         motion_level: str) -> Dict[str, Any]:
        """Calculate optimal GOP (Group of Pictures) structure based on content"""
        params = {}
        
        fps = video_info.get('fps', 30)
        duration = video_info.get('duration', 0)
        
        # Base GOP from config
        base_gop = int(self.config.get('video_compression.profiles.discord_10mb.x264.gop', 240))
        base_keyint_min = int(self.config.get('video_compression.profiles.discord_10mb.x264.keyint_min', 23))
        base_sc_threshold = int(self.config.get('video_compression.profiles.discord_10mb.x264.sc_threshold', 40))
        
        # GOP size: typically 2x FPS for good balance, but adapt based on motion
        if motion_level == 'high':
            # High motion benefits from more frequent keyframes (smaller GOP)
            gop = int(fps * 1.5)  # 1.5 seconds
            keyint_min = max(23, int(fps * 0.5))  # At least 0.5 seconds
        elif motion_level == 'low':
            # Low motion can use larger GOP for better compression
            gop = int(fps * 3)  # 3 seconds
            keyint_min = max(23, int(fps * 1.0))  # At least 1 second
        else:
            # Medium motion - balanced approach
            gop = int(fps * 2)  # 2 seconds
            keyint_min = max(23, int(fps * 0.75))  # At least 0.75 seconds
        
        # Clamp GOP to reasonable range
        gop = max(int(fps * 0.5), min(gop, int(fps * 4)))
        params['gop'] = gop
        
        # Keyint_min should be at least 23 (x264 minimum) and not exceed GOP
        params['keyint_min'] = max(23, min(keyint_min, gop - 1))
        
        # Scene change threshold - higher for high motion, lower for low motion
        if motion_level == 'high':
            # High motion needs more sensitive scene change detection
            sc_threshold = max(30, base_sc_threshold - 10)
        elif motion_level == 'low':
            # Low motion can use less sensitive detection
            sc_threshold = min(50, base_sc_threshold + 10)
        else:
            sc_threshold = base_sc_threshold
        params['sc_threshold'] = sc_threshold
        
        return params
    
    def _select_optimal_codec(self, video_info: Dict[str, Any], target_size_mb: float,
                            current_bitrate: int) -> Tuple[str, str]:
        """Select optimal codec considering HEVC benefits when size savings >15% and quality maintained"""
        # Get codec selection strategy from config
        prefer_hevc_config = self.config.get('video_compression.codec.prefer_hevc_when', {})
        size_savings_threshold = float(prefer_hevc_config.get('size_savings_threshold', 0.15))
        quality_maintained = bool(prefer_hevc_config.get('quality_maintained', True))
        min_bitrate = int(prefer_hevc_config.get('min_bitrate', 500))
        
        # Check if HEVC is allowed
        allow_hevc = bool(self.config.get('video_compression.codec.allow_hevc', True))
        
        # If bitrate is too low, prefer H264
        if current_bitrate < min_bitrate:
            return self.hardware.get_best_encoder("h264")
        
        # If HEVC is not allowed, use H264
        if not allow_hevc:
            return self.hardware.get_best_encoder("h264")
        
        # Estimate potential size savings with HEVC (typically 20-30% better compression)
        # HEVC is more efficient, so we can use lower bitrate for same quality
        hevc_bitrate_estimate = int(current_bitrate * 0.75)  # ~25% reduction
        hevc_size_estimate = (hevc_bitrate_estimate * video_info.get('duration', 0) / 1000) / 8
        current_size_estimate = (current_bitrate * video_info.get('duration', 0) / 1000) / 8
        
        if current_size_estimate > 0:
            size_savings = (current_size_estimate - hevc_size_estimate) / current_size_estimate
            
            # If size savings exceed threshold, prefer HEVC
            if size_savings >= size_savings_threshold:
                hevc_encoder, hevc_accel = self.hardware.get_best_encoder("hevc")
                # Ensure HEVC is actually available (check for hevc, x265, or nvenc/amf/qsv hevc variants)
                if hevc_encoder and ('265' in hevc_encoder.lower() or 'hevc' in hevc_encoder.lower()):
                    logger.info(f"Preferring HEVC ({hevc_encoder}) for estimated {size_savings*100:.1f}% size savings")
                    return hevc_encoder, hevc_accel
        
        # Default to H264
        return self.hardware.get_best_encoder("h264")
    
    def _calculate_perceptual_filter_strength(self, video_info: Dict[str, Any], 
                                             params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate perceptual preprocessing filter strengths based on content analysis"""
        filter_params = {}
        
        # Check if perceptual filters are enabled in config
        enable_denoise = self.config.get('video_compression.prefilters.denoise', True)
        enable_deband = self.config.get('video_compression.prefilters.deband', True)
        
        complexity = video_info.get('complexity_score', 5.0)
        motion_level = video_info.get('motion_level', 'medium')
        
        # Denoise filter - adaptive strength based on content
        if enable_denoise:
            # High complexity/static scenes benefit more from denoising
            if complexity > 7.0 and motion_level == 'low':
                # Strong denoising for high complexity static content
                filter_params['denoise_strength'] = 0.8
            elif complexity > 5.0:
                # Moderate denoising for medium-high complexity
                filter_params['denoise_strength'] = 0.5
            elif complexity > 3.0:
                # Light denoising for medium complexity
                filter_params['denoise_strength'] = 0.3
            else:
                # Minimal or no denoising for simple content
                filter_params['denoise_strength'] = 0.1
        
        # Deband filter - adaptive strength based on complexity
        if enable_deband:
            # High complexity content benefits more from debanding
            if complexity > 7.0:
                filter_params['deband_strength'] = 0.6
            elif complexity > 5.0:
                filter_params['deband_strength'] = 0.4
            else:
                filter_params['deband_strength'] = 0.2
        
        # Selective sharpening - only for content that benefits
        # Sharpen when complexity is medium-high and motion is low
        if 4.0 <= complexity <= 7.0 and motion_level in ['low', 'medium']:
            filter_params['sharpen_strength'] = 0.3
        elif complexity > 7.0 and motion_level == 'low':
            # Light sharpening for high complexity static content
            filter_params['sharpen_strength'] = 0.2
        
        return filter_params
    
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
        
        # Calculate FPS - respect configured minimum
        config_min_fps = self.config.get('video_compression.bitrate_validation.min_fps', 20)
        target_fps = min(video_info['fps'], 24)  # Reduce FPS but not below minimum
        target_fps = max(target_fps, config_min_fps)  # Ensure minimum is respected
        
        params = {
            'encoder': encoder,
            'acceleration_type': accel_type,
            'width': aggressive_width,
            'height': aggressive_height,
            'fps': target_fps,
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
        """Build FFmpeg command with intelligent parameters and perceptual preprocessing"""
        params['maxrate_multiplier'] = 1.2  # Set specific multiplier for intelligent command
        
        # Add perceptual preprocessing filters if enabled
        video_info = params.get('video_info', {})
        if video_info:
            filter_params = self._calculate_perceptual_filter_strength(video_info, params)
            if filter_params:
                params['perceptual_filters'] = filter_params
        
        try:
            logger.info(f"Encoder selected: {params.get('encoder', 'unknown')} ({params.get('acceleration_type', 'software')})")
        except Exception:
            pass
        return FFmpegUtils.build_standard_ffmpeg_command(input_path, output_path, params, self.bitrate_validator)
    
    def _build_two_pass_command(self, input_path: str, output_path: str, params: Dict[str, Any], 
                              pass_num: int, log_file: str) -> List[str]:
        """Build two-pass FFmpeg command with enhanced psychovisual tuning support"""
        params['maxrate_multiplier'] = 1.1  # Set specific multiplier for two-pass
        try:
            logger.info(f"Encoder selected: {params.get('encoder', 'unknown')} ({params.get('acceleration_type', 'software')})")
        except Exception:
            pass
        
        # Use x264-specific CAE builder if we have advanced psychovisual parameters (psy-rd, psy-trellis, etc.)
        encoder = params.get('encoder', 'libx264')
        if encoder == 'libx264' and ('psy_rd' in params or 'psy_trellis' in params or 
                                     params.get('aq_mode') == 3 or 'gop' in params):
            # Use x264 two-pass CAE builder for advanced psychovisual tuning
            return FFmpegUtils.build_x264_two_pass_cae(input_path, output_path, params, pass_num, log_file, self.bitrate_validator)
        else:
            # Use standard two-pass builder
            return FFmpegUtils.build_two_pass_command(input_path, output_path, params, pass_num, log_file, self.bitrate_validator)
    
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
        
        if self.debug_logging_enabled:
            logger.info("=== FFMPEG EXECUTION ===")
            logger.info(f"Command length: {len(cmd)} arguments")
            logger.info(f"Expected duration: {duration:.2f}s")
            
            # Log command with sensitive information masked
            masked_cmd = []
            for arg in cmd:
                if any(sensitive in arg.lower() for sensitive in ['password', 'key', 'token']):
                    masked_cmd.append('[MASKED]')
                else:
                    masked_cmd.append(arg)
            logger.info(f"FFmpeg command: {' '.join(masked_cmd)}")
            
            # Log encoder and acceleration info
            encoder_info = self._extract_encoder_info_from_cmd(cmd)
            if encoder_info:
                logger.info(f"Encoder: {encoder_info.get('encoder', 'unknown')}")
                logger.info(f"Acceleration: {encoder_info.get('acceleration', 'software')}")
                logger.info(f"Preset: {encoder_info.get('preset', 'default')}")
        
        execution_start_time = time.time()
        
        logger.debug("Starting FFmpeg compression...")
        logger.debug(f"FFmpeg command: {' '.join(cmd)}")
        
        # Try original command first
        try:
            with self._measure_operation_performance('ffmpeg_execution'):
                self._execute_ffmpeg_command(cmd, duration)
            
            execution_time = time.time() - execution_start_time
            logger.info("FFmpeg compression completed")
            
            if self.debug_logging_enabled:
                logger.info(f"Execution time: {execution_time:.2f}s")
                if duration > 0:
                    speed_ratio = duration / execution_time
                    logger.info(f"Processing speed: {speed_ratio:.2f}x realtime")
                    
                    if speed_ratio < 0.5:
                        logger.info("Note: Processing slower than 0.5x realtime, consider hardware acceleration")
                    elif speed_ratio > 2.0:
                        logger.info("Note: Excellent processing speed, hardware acceleration working well")
                
                logger.info("=== END FFMPEG EXECUTION ===")
        
        except subprocess.CalledProcessError as e:
            # Check if it's a hardware acceleration error
            error_output = str(e.stderr) if e.stderr else str(e)
            execution_time = time.time() - execution_start_time
            
            if self.debug_logging_enabled:
                logger.error("=== FFMPEG EXECUTION FAILED ===")
                logger.error(f"Execution time before failure: {execution_time:.2f}s")
                logger.error(f"Return code: {getattr(e, 'returncode', 'unknown')}")
                logger.error(f"Error output (first 500 chars): {error_output[:500]}")
                
                # Log error classification
                if "No such file or directory" in error_output:
                    logger.error("Error type: File not found")
                elif "Permission denied" in error_output:
                    logger.error("Error type: Permission denied")
                elif "Invalid data found" in error_output:
                    logger.error("Error type: Invalid input data")
                elif "Conversion failed" in error_output:
                    logger.error("Error type: Conversion failure")
                else:
                    logger.error("Error type: Unknown")
            
            logger.error(f"FFmpeg command failed with error: {error_output[:500]}...")  # Limit error output length
            
            # Use consolidated hardware error detection
            return_code = e.returncode if hasattr(e, 'returncode') else None
            is_hardware_error, encoder_type = self._is_hardware_acceleration_error(error_output, return_code)
            
            if is_hardware_error:
                logger.warning(f"{encoder_type} hardware acceleration failed, trying software fallback...")
                
                if self.debug_logging_enabled:
                    self._log_compression_decision(
                        'hardware_fallback',
                        f'{encoder_type} hardware acceleration failed',
                        {
                            'original_encoder': encoder_type,
                            'error_snippet': error_output[:200],
                            'return_code': return_code
                        },
                        'Hardware acceleration error detected, falling back to software encoding'
                    )
                
                # Replace hardware encoder with software encoder
                fallback_cmd = self._create_software_fallback_command(cmd)
                if fallback_cmd:
                    logger.debug(f"Fallback command: {' '.join(fallback_cmd)}")
                    try:
                        with self._measure_operation_performance('ffmpeg_fallback_execution'):
                            self._execute_ffmpeg_command(fallback_cmd, duration)
                        
                        fallback_time = time.time() - execution_start_time
                        logger.info("FFmpeg compression completed with software fallback")
                        
                        if self.debug_logging_enabled:
                            logger.info(f"Fallback execution time: {fallback_time:.2f}s")
                            self._log_compression_decision(
                                'hardware_fallback_success',
                                'Software fallback succeeded',
                                {
                                    'fallback_time': fallback_time,
                                    'total_time_with_retry': fallback_time
                                },
                                'Software encoding successfully completed after hardware failure'
                            )
                        return
                    except subprocess.CalledProcessError as fallback_error:
                        fallback_error_output = str(fallback_error.stderr) if fallback_error.stderr else str(fallback_error)
                        logger.error(f"Software fallback also failed: {fallback_error_output[:200]}...")
                        
                        if self.debug_logging_enabled:
                            self._log_compression_decision(
                                'hardware_fallback_failed',
                                'Software fallback also failed',
                                {
                                    'fallback_error': fallback_error_output[:200],
                                    'fallback_return_code': getattr(fallback_error, 'returncode', 'unknown')
                                },
                                'Both hardware and software encoding failed'
                            )
                        pass  # Fall through to original error
                else:
                    logger.error("Failed to create software fallback command")
                    
                    if self.debug_logging_enabled:
                        self._log_compression_decision(
                            'fallback_creation_failed',
                            'Could not create software fallback command',
                            {'original_cmd_length': len(cmd)},
                            'Unable to generate software fallback from hardware command'
                        )
            
            if self.debug_logging_enabled:
                logger.error("=== END FFMPEG EXECUTION FAILED ===")
            
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
            from ..ffmpeg_utils import FFmpegUtils
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
    
    def _calculate_compression_params_with_smart_strategy(self, video_info: Dict[str, Any],
                                                        platform_config: Dict[str, Any],
                                                        target_size_mb: float,
                                                        strategy: CompressionStrategy = CompressionStrategy.BALANCED) -> Dict[str, Any]:
        """
        Calculate compression parameters using smart strategy with FPS constraint validation
        and alternative strategies when needed.
        """
        self.logger.info("=== SMART COMPRESSION PARAMETER CALCULATION ===")
        
        # Use smart compression strategy to select parameters
        compression_params = self.compression_strategy.select_compression_parameters(
            video_info, target_size_mb, strategy
        )
        
        # Validate parameters against constraints
        constraints = self.compression_strategy.get_compression_constraints()
        validation_issues = self.compression_strategy.validate_compression_parameters(
            compression_params, constraints
        )
        
        if validation_issues:
            self.logger.warning("Compression parameter validation issues:")
            for issue in validation_issues:
                self.logger.warning(f"  - {issue}")
        
        # Assess FPS reduction impact
        impact = self.compression_strategy.assess_fps_reduction_impact(
            video_info.get('fps', 30.0), compression_params.target_fps, constraints
        )
        
        # Log FPS reduction justification with smart strategy analysis
        strategy_analysis = {
            'fps_impact': {
                'impact_level': impact.impact_level.value,
                'impact_score': impact.reduction_percent,
                'recommendation': impact.recommendation
            },
            'prefer_bitrate_reduction': constraints.prefer_quality_over_fps,
            'strategy_reason': compression_params.strategy_reason,
            'alternatives_considered': []
        }
        
        # If FPS reduction is significant, consider alternative strategies
        if impact.impact_level in [impact.impact_level.SIGNIFICANT, impact.impact_level.SEVERE]:
            alternatives = self.compression_strategy.get_alternative_strategies(
                video_info, target_size_mb, compression_params
            )
            
            strategy_analysis['alternatives_considered'] = [
                {
                    'strategy': alt.strategy_type,
                    'reason': alt.description
                }
                for alt in alternatives
            ]
            
            # Notify user about aggressive compression
            if alternatives:
                self.compression_strategy.notify_aggressive_compression_required(
                    video_info, target_size_mb, alternatives
                )
        
        # Log detailed FPS reduction justification
        self._log_fps_reduction_justification(
            video_info.get('fps', 30.0),
            compression_params.target_fps,
            video_info,
            strategy_analysis
        )
        
        # Get best encoder
        encoder, accel_type = self.hardware.get_best_encoder("h264")
        
        # Build final parameters
        params = {
            'encoder': encoder,
            'acceleration_type': accel_type,
            'width': compression_params.resolution[0],
            'height': compression_params.resolution[1],
            'fps': compression_params.target_fps,
            'bitrate': compression_params.target_bitrate,
            'quality_factor': compression_params.quality_factor,
            'fps_reduction_applied': compression_params.fps_reduction_applied,
            'alternative_strategy_used': compression_params.alternative_strategy_used,
            'strategy_reason': compression_params.strategy_reason
        }
        
        # Apply quality-specific settings based on quality factor
        if accel_type == 'software':
            # Adjust CRF based on quality factor (lower factor = higher CRF = lower quality)
            base_crf = 23
            crf_adjustment = int((1.0 - compression_params.quality_factor) * 10)
            params['crf'] = min(max(base_crf + crf_adjustment, 18), 35)
            
            # Adjust preset based on strategy
            if strategy == CompressionStrategy.QUALITY_FIRST:
                params['preset'] = 'slower'
            elif strategy == CompressionStrategy.SIZE_FIRST:
                params['preset'] = 'fast'
            else:
                params['preset'] = 'medium'
        
        # Platform-specific adjustments
        if platform_config:
            params.update(self._apply_platform_constraints(params, platform_config))
        
        # Log final parameters
        self.logger.info(f"Final compression parameters:")
        self.logger.info(f"  Resolution: {params['width']}x{params['height']}")
        self.logger.info(f"  FPS: {params['fps']:.1f}")
        self.logger.info(f"  Bitrate: {params['bitrate']}kbps")
        self.logger.info(f"  Quality factor: {params['quality_factor']:.2f}")
        if 'crf' in params:
            self.logger.info(f"  CRF: {params['crf']}")
        if 'preset' in params:
            self.logger.info(f"  Preset: {params['preset']}")
        self.logger.info(f"  Encoder: {params['encoder']} ({params['acceleration_type']})")
        
        return params
    
    def _apply_alternative_compression_strategy(self, input_path: str, output_path: str,
                                              video_info: Dict[str, Any], target_size_mb: float,
                                              alternative_strategy) -> Optional[Dict[str, Any]]:
        """
        Apply an alternative compression strategy when FPS reduction is insufficient
        """
        from .compression_strategy import AlternativeStrategy
        
        self.logger.info(f"=== APPLYING ALTERNATIVE STRATEGY: {alternative_strategy.strategy_type} ===")
        self.logger.info(f"Description: {alternative_strategy.description}")
        self.logger.info(f"Expected impact: {alternative_strategy.expected_impact}")
        
        try:
            if alternative_strategy.strategy_type == "resolution_reduction":
                return self._apply_resolution_reduction_strategy(
                    input_path, output_path, video_info, target_size_mb, alternative_strategy
                )
            elif alternative_strategy.strategy_type == "quality_adjustment":
                return self._apply_quality_adjustment_strategy(
                    input_path, output_path, video_info, target_size_mb, alternative_strategy
                )
            elif alternative_strategy.strategy_type == "bitrate_optimization":
                return self._apply_bitrate_optimization_strategy(
                    input_path, output_path, video_info, target_size_mb, alternative_strategy
                )
            elif alternative_strategy.strategy_type == "segmentation":
                return self._apply_segmentation_strategy(
                    input_path, output_path, video_info, target_size_mb, alternative_strategy
                )
            else:
                self.logger.warning(f"Unknown alternative strategy type: {alternative_strategy.strategy_type}")
                return None
                
        except Exception as e:
            self.logger.error(f"Alternative strategy {alternative_strategy.strategy_type} failed: {e}")
            return None
    
    def _apply_resolution_reduction_strategy(self, input_path: str, output_path: str,
                                           video_info: Dict[str, Any], target_size_mb: float,
                                           strategy) -> Optional[Dict[str, Any]]:
        """Apply resolution reduction as alternative to FPS reduction"""
        try:
            temp_output = os.path.join(self.temp_dir, "resolution_reduced_output.mp4")
            
            # Get parameters from strategy
            new_width = strategy.parameters['width']
            new_height = strategy.parameters['height']
            fps = strategy.parameters['fps']  # Restored original FPS
            
            # Calculate bitrate for new resolution
            duration = video_info.get('duration', 0)
            audio_bitrate = 64
            total_bits = target_size_mb * 8 * 1024 * 1024
            video_bits = total_bits - (audio_bitrate * 1000 * duration)
            video_bitrate = max(int(video_bits / duration / 1000), 64)
            
            # Get best encoder
            encoder, accel_type = self.hardware.get_best_encoder("h264")
            
            # Build parameters
            params = {
                'encoder': encoder,
                'acceleration_type': accel_type,
                'width': new_width,
                'height': new_height,
                'fps': fps,
                'bitrate': video_bitrate,
                'audio_bitrate': audio_bitrate
            }
            
            # Apply quality settings
            if accel_type == 'software':
                params['crf'] = 23  # Standard quality
                params['preset'] = 'medium'
            
            self.logger.info(f"Resolution reduction: {video_info['width']}x{video_info['height']} → {new_width}x{new_height}")
            self.logger.info(f"FPS preserved: {fps:.1f} (original: {video_info.get('fps', 30):.1f})")
            self.logger.info(f"Bitrate: {video_bitrate}kbps video + {audio_bitrate}kbps audio")
            
            # Build and execute FFmpeg command
            ffmpeg_cmd = self._build_intelligent_ffmpeg_command(input_path, temp_output, params)
            success = self._execute_ffmpeg_with_progress(ffmpeg_cmd, video_info['duration'])
            
            if success and os.path.exists(temp_output):
                # Move to final output
                if os.path.exists(output_path):
                    os.remove(output_path)
                shutil.move(temp_output, output_path)
                
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                
                self.logger.info(f"Resolution reduction successful: {size_mb:.2f}MB")
                
                return {
                    'success': True,
                    'size_mb': size_mb,
                    'strategy': 'resolution_reduction',
                    'alternative_strategy_used': True,
                    'params': params,
                    'quality_score': 7.5  # Estimate - resolution reduction typically maintains good quality
                }
            else:
                self.logger.warning("Resolution reduction strategy failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Resolution reduction strategy failed: {e}")
            return None
    
    def _apply_quality_adjustment_strategy(self, input_path: str, output_path: str,
                                         video_info: Dict[str, Any], target_size_mb: float,
                                         strategy) -> Optional[Dict[str, Any]]:
        """Apply quality parameter adjustment as alternative to FPS reduction"""
        try:
            temp_output = os.path.join(self.temp_dir, "quality_adjusted_output.mp4")
            
            # Get parameters from strategy
            crf = strategy.parameters['crf']
            preset = strategy.parameters['preset']
            fps = strategy.parameters['fps']  # Restored original FPS
            
            # Calculate bitrate
            duration = video_info.get('duration', 0)
            audio_bitrate = 64
            total_bits = target_size_mb * 8 * 1024 * 1024
            video_bits = total_bits - (audio_bitrate * 1000 * duration)
            video_bitrate = max(int(video_bits / duration / 1000), 64)
            
            # Get best encoder
            encoder, accel_type = self.hardware.get_best_encoder("h264")
            
            # Build parameters
            params = {
                'encoder': encoder,
                'acceleration_type': accel_type,
                'width': video_info['width'],
                'height': video_info['height'],
                'fps': fps,
                'bitrate': video_bitrate,
                'audio_bitrate': audio_bitrate
            }
            
            # Apply quality settings
            if accel_type == 'software':
                params['crf'] = crf
                params['preset'] = preset
            
            self.logger.info(f"Quality adjustment: CRF={crf}, preset={preset}")
            self.logger.info(f"FPS preserved: {fps:.1f} (original: {video_info.get('fps', 30):.1f})")
            self.logger.info(f"Bitrate: {video_bitrate}kbps video + {audio_bitrate}kbps audio")
            
            # Build and execute FFmpeg command
            ffmpeg_cmd = self._build_intelligent_ffmpeg_command(input_path, temp_output, params)
            success = self._execute_ffmpeg_with_progress(ffmpeg_cmd, video_info['duration'])
            
            if success and os.path.exists(temp_output):
                # Move to final output
                if os.path.exists(output_path):
                    os.remove(output_path)
                shutil.move(temp_output, output_path)
                
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                
                self.logger.info(f"Quality adjustment successful: {size_mb:.2f}MB")
                
                return {
                    'success': True,
                    'size_mb': size_mb,
                    'strategy': 'quality_adjustment',
                    'alternative_strategy_used': True,
                    'params': params,
                    'quality_score': 7.0  # Estimate - quality adjustment may reduce quality slightly
                }
            else:
                self.logger.warning("Quality adjustment strategy failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Quality adjustment strategy failed: {e}")
            return None
    
    def _apply_bitrate_optimization_strategy(self, input_path: str, output_path: str,
                                           video_info: Dict[str, Any], target_size_mb: float,
                                           strategy) -> Optional[Dict[str, Any]]:
        """Apply bitrate optimization as alternative to FPS reduction"""
        try:
            temp_output = os.path.join(self.temp_dir, "bitrate_optimized_output.mp4")
            
            # Get parameters from strategy
            video_bitrate = strategy.parameters['video_bitrate']
            audio_bitrate = strategy.parameters['audio_bitrate']
            fps = strategy.parameters['fps']  # Restored original FPS
            
            # Get best encoder
            encoder, accel_type = self.hardware.get_best_encoder("h264")
            
            # Build parameters
            params = {
                'encoder': encoder,
                'acceleration_type': accel_type,
                'width': video_info['width'],
                'height': video_info['height'],
                'fps': fps,
                'bitrate': video_bitrate,
                'audio_bitrate': audio_bitrate
            }
            
            # Apply quality settings
            if accel_type == 'software':
                params['crf'] = 25  # Slightly higher CRF for better compression
                params['preset'] = 'slower'  # Slower preset for better efficiency
            
            self.logger.info(f"Bitrate optimization: {video_bitrate}kbps video + {audio_bitrate}kbps audio")
            self.logger.info(f"FPS preserved: {fps:.1f} (original: {video_info.get('fps', 30):.1f})")
            
            # Build and execute FFmpeg command
            ffmpeg_cmd = self._build_intelligent_ffmpeg_command(input_path, temp_output, params)
            success = self._execute_ffmpeg_with_progress(ffmpeg_cmd, video_info['duration'])
            
            if success and os.path.exists(temp_output):
                # Move to final output
                if os.path.exists(output_path):
                    os.remove(output_path)
                shutil.move(temp_output, output_path)
                
                size_mb = os.path.getsize(output_path) / (1024 * 1024)
                
                self.logger.info(f"Bitrate optimization successful: {size_mb:.2f}MB")
                
                return {
                    'success': True,
                    'size_mb': size_mb,
                    'strategy': 'bitrate_optimization',
                    'alternative_strategy_used': True,
                    'params': params,
                    'quality_score': 7.2  # Estimate - bitrate optimization usually maintains good quality
                }
            else:
                self.logger.warning("Bitrate optimization strategy failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Bitrate optimization strategy failed: {e}")
            return None
    
    def _apply_segmentation_strategy(self, input_path: str, output_path: str,
                                   video_info: Dict[str, Any], target_size_mb: float,
                                   strategy) -> Optional[Dict[str, Any]]:
        """Apply video segmentation as alternative to aggressive compression"""
        try:
            # Get parameters from strategy
            num_segments = strategy.parameters['num_segments']
            segment_duration = strategy.parameters['segment_duration']
            fps = strategy.parameters['fps']  # Restored original FPS
            
            self.logger.info(f"Segmentation strategy: {num_segments} segments of ~{segment_duration}s each")
            self.logger.info(f"FPS preserved: {fps:.1f} (original: {video_info.get('fps', 30):.1f})")
            
            # Use the video segmenter
            platform_config = {}  # Use default platform config for segmentation
            result = self._compress_with_segmentation(
                input_path, output_path, target_size_mb, platform_config, video_info
            )
            
            if result and result.get('success'):
                result['alternative_strategy_used'] = True
                result['strategy'] = 'segmentation'
                self.logger.info(f"Segmentation strategy successful")
                return result
            else:
                self.logger.warning("Segmentation strategy failed")
                return None
                
        except Exception as e:
            self.logger.error(f"Segmentation strategy failed: {e}")
            return None
    
    def _try_alternative_strategies_when_fps_insufficient(self, input_path: str, output_path: str,
                                                        video_info: Dict[str, Any], target_size_mb: float,
                                                        current_result: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """
        Try alternative compression strategies when FPS reduction alone is insufficient
        """
        self.logger.info("=== TRYING ALTERNATIVE STRATEGIES ===")
        
        # Create compression parameters for analysis
        from .compression_strategy import CompressionParams
        
        if current_result:
            # Analyze current result to determine if alternatives are needed
            current_size = current_result.get('size_mb', 0)
            if current_size <= target_size_mb:
                self.logger.info("Current result meets target size, no alternatives needed")
                return current_result
        
        # Get alternative strategies from smart compression strategy
        # Create a mock compression params for analysis
        mock_params = CompressionParams(
            target_fps=video_info.get('fps', 30.0),
            target_bitrate=1000,  # Mock bitrate
            resolution=(video_info.get('width', 1920), video_info.get('height', 1080)),
            quality_factor=0.8,
            fps_reduction_applied=True  # Assume FPS reduction was attempted
        )
        
        alternatives = self.compression_strategy.get_alternative_strategies(
            video_info, target_size_mb, mock_params
        )
        
        if not alternatives:
            self.logger.info("No feasible alternative strategies available")
            return current_result
        
        # Notify user about aggressive compression requirement
        self.compression_strategy.notify_aggressive_compression_required(
            video_info, target_size_mb, alternatives
        )
        
        # Try each alternative strategy in order of preference
        best_result = current_result
        
        for alternative in alternatives:
            self.logger.info(f"Trying alternative strategy: {alternative.strategy_type}")
            
            try:
                alt_result = self._apply_alternative_compression_strategy(
                    input_path, output_path, video_info, target_size_mb, alternative
                )
                
                if alt_result and alt_result.get('success'):
                    alt_size = alt_result.get('size_mb', 0)
                    
                    # Check if this alternative is better
                    if alt_size <= target_size_mb:
                        if not best_result or alt_size > best_result.get('size_mb', 0):
                            best_result = alt_result
                            self.logger.info(f"Alternative strategy {alternative.strategy_type} successful: {alt_size:.2f}MB")
                            
                            # If we found a good solution, we can stop trying more alternatives
                            if alt_size >= target_size_mb * 0.9:  # Within 90% of target
                                break
                    else:
                        self.logger.info(f"Alternative strategy {alternative.strategy_type} exceeded target: {alt_size:.2f}MB > {target_size_mb:.2f}MB")
                else:
                    self.logger.warning(f"Alternative strategy {alternative.strategy_type} failed")
                    
            except Exception as e:
                self.logger.error(f"Alternative strategy {alternative.strategy_type} error: {e}")
                continue
        
        if best_result and best_result.get('alternative_strategy_used'):
            self.logger.info(f"Best alternative strategy: {best_result.get('strategy', 'unknown')}")
        else:
            self.logger.info("No alternative strategies improved the result")
        
        return best_result
    
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