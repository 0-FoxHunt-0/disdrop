"""
Performance Monitoring and Metrics System

This module provides comprehensive performance monitoring, metrics collection,
and analysis tools for video compression operations.
"""

import json
import os
import time
import threading
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging
import psutil
from contextlib import contextmanager

try:
    from .logger_setup import get_logger
    from .config_manager import ConfigManager
except ImportError:
    # Fallback for direct execution
    from logger_setup import get_logger
    from config_manager import ConfigManager

logger = get_logger(__name__)


@dataclass
class OperationMetrics:
    """Metrics for a single operation."""
    operation_name: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_message: Optional[str] = None
    memory_start_mb: Optional[float] = None
    memory_end_mb: Optional[float] = None
    memory_delta_mb: Optional[float] = None
    cpu_percent: Optional[float] = None
    additional_metrics: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'OperationMetrics':
        """Create from dictionary."""
        return cls(**data)


@dataclass
class CompressionSessionMetrics:
    """Comprehensive metrics for a compression session."""
    session_id: str
    input_file: str
    target_size_mb: float
    platform: Optional[str]
    start_time: float
    end_time: Optional[float] = None
    duration: Optional[float] = None
    success: Optional[bool] = None
    final_size_mb: Optional[float] = None
    compression_ratio: Optional[float] = None
    quality_score: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    operations: List[OperationMetrics] = None
    parameter_changes: List[Dict[str, Any]] = None
    quality_evaluations: List[Dict[str, Any]] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.operations is None:
            self.operations = []
        if self.parameter_changes is None:
            self.parameter_changes = []
        if self.quality_evaluations is None:
            self.quality_evaluations = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['operations'] = [op.to_dict() if hasattr(op, 'to_dict') else op for op in self.operations]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CompressionSessionMetrics':
        """Create from dictionary."""
        operations_data = data.pop('operations', [])
        operations = [OperationMetrics.from_dict(op) if isinstance(op, dict) else op for op in operations_data]
        return cls(operations=operations, **data)


@dataclass
class PerformanceProfile:
    """Performance profile configuration."""
    name: str
    max_quality_evaluation_time: float
    max_compression_iterations: int
    enable_fast_estimation: bool
    cache_enabled: bool
    parallel_processing: bool
    memory_optimization: bool
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PerformanceProfile':
        """Create from dictionary."""
        return cls(**data)


class PerformanceMonitor:
    """Comprehensive performance monitoring system."""
    
    def __init__(self, config_manager: ConfigManager, metrics_dir: Optional[str] = None):
        """
        Initialize performance monitor.
        
        Args:
            config_manager: Configuration manager instance
            metrics_dir: Directory for storing metrics data
        """
        self.config = config_manager
        self.metrics_dir = Path(metrics_dir) if metrics_dir else Path.home() / '.disdrop' / 'metrics'
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Current session tracking
        self._current_session: Optional[CompressionSessionMetrics] = None
        self._session_counter = 0
        
        # Performance statistics
        self._operation_stats = defaultdict(list)
        self._cache_stats = {
            'hits': 0,
            'misses': 0,
            'total_lookups': 0
        }
        
        # Recent metrics for trend analysis (keep last 100 operations)
        self._recent_operations = deque(maxlen=100)
        self._recent_sessions = deque(maxlen=50)
        
        # Performance profiles
        self._performance_profiles = self._load_performance_profiles()
        self._current_profile = self._performance_profiles.get('balanced')
        
        # System monitoring
        self._system_monitor_enabled = self.config.get('performance_monitoring.system_monitoring.enabled', True)
        self._memory_threshold_mb = self.config.get('performance_monitoring.system_monitoring.memory_threshold_mb', 1000)
        
        logger.info(f"Performance monitor initialized at {self.metrics_dir}")
    
    def _load_performance_profiles(self) -> Dict[str, PerformanceProfile]:
        """Load performance profiles from configuration."""
        profiles = {}
        
        # Default profiles
        profiles['fast'] = PerformanceProfile(
            name='fast',
            max_quality_evaluation_time=10.0,
            max_compression_iterations=2,
            enable_fast_estimation=True,
            cache_enabled=True,
            parallel_processing=True,
            memory_optimization=True
        )
        
        profiles['balanced'] = PerformanceProfile(
            name='balanced',
            max_quality_evaluation_time=30.0,
            max_compression_iterations=3,
            enable_fast_estimation=True,
            cache_enabled=True,
            parallel_processing=True,
            memory_optimization=False
        )
        
        profiles['quality'] = PerformanceProfile(
            name='quality',
            max_quality_evaluation_time=120.0,
            max_compression_iterations=5,
            enable_fast_estimation=False,
            cache_enabled=True,
            parallel_processing=False,
            memory_optimization=False
        )
        
        # Load custom profiles from config
        custom_profiles = self.config.get('performance_monitoring.profiles', {})
        for name, profile_data in custom_profiles.items():
            try:
                profiles[name] = PerformanceProfile.from_dict(profile_data)
            except Exception as e:
                logger.warning(f"Failed to load custom profile '{name}': {e}")
        
        return profiles
    
    def start_compression_session(self, input_file: str, target_size_mb: float, 
                                platform: Optional[str] = None) -> str:
        """
        Start a new compression session.
        
        Args:
            input_file: Path to input video file
            target_size_mb: Target output size in MB
            platform: Target platform (optional)
            
        Returns:
            Session ID
        """
        with self._lock:
            self._session_counter += 1
            session_id = f"session_{self._session_counter}_{int(time.time())}"
            
            self._current_session = CompressionSessionMetrics(
                session_id=session_id,
                input_file=input_file,
                target_size_mb=target_size_mb,
                platform=platform,
                start_time=time.time()
            )
            
            logger.info(f"Started compression session: {session_id}")
            logger.debug(f"Session details: input={os.path.basename(input_file)}, "
                        f"target={target_size_mb:.2f}MB, platform={platform}")
            
            return session_id
    
    def end_compression_session(self, success: bool, final_size_mb: Optional[float] = None,
                              quality_score: Optional[float] = None, 
                              error_message: Optional[str] = None) -> None:
        """
        End the current compression session.
        
        Args:
            success: Whether compression was successful
            final_size_mb: Final output file size in MB
            quality_score: Final quality score (VMAF, SSIM, etc.)
            error_message: Error message if failed
        """
        with self._lock:
            if not self._current_session:
                logger.warning("No active compression session to end")
                return
            
            end_time = time.time()
            self._current_session.end_time = end_time
            self._current_session.duration = end_time - self._current_session.start_time
            self._current_session.success = success
            self._current_session.final_size_mb = final_size_mb
            self._current_session.quality_score = quality_score
            self._current_session.error_message = error_message
            
            # Calculate compression ratio
            if final_size_mb and self._current_session.target_size_mb:
                self._current_session.compression_ratio = final_size_mb / self._current_session.target_size_mb
            
            # Calculate cache hit rate for this session
            session_cache_lookups = sum(1 for op in self._current_session.operations 
                                      if 'cache' in op.operation_name.lower())
            session_cache_hits = sum(1 for op in self._current_session.operations 
                                   if 'cache' in op.operation_name.lower() and op.success)
            
            if session_cache_lookups > 0:
                self._current_session.cache_hit_rate = session_cache_hits / session_cache_lookups
            
            # Store session metrics
            self._recent_sessions.append(self._current_session)
            self._save_session_metrics(self._current_session)
            
            logger.info(f"Ended compression session: {self._current_session.session_id}")
            logger.info(f"Session summary: success={success}, duration={self._current_session.duration:.2f}s")
            
            if success and final_size_mb:
                utilization = (final_size_mb / self._current_session.target_size_mb) * 100
                logger.info(f"Size utilization: {utilization:.1f}% ({final_size_mb:.2f}MB / {self._current_session.target_size_mb:.2f}MB)")
            
            self._current_session = None
    
    @contextmanager
    def measure_operation(self, operation_name: str, **additional_metrics):
        """
        Context manager to measure operation performance.
        
        Args:
            operation_name: Name of the operation being measured
            **additional_metrics: Additional metrics to record
        """
        start_time = time.time()
        start_memory = None
        cpu_percent = None
        
        # Capture system metrics if enabled
        if self._system_monitor_enabled:
            try:
                process = psutil.Process()
                start_memory = process.memory_info().rss / (1024 * 1024)  # MB
                cpu_percent = process.cpu_percent()
            except Exception as e:
                logger.debug(f"Failed to capture system metrics: {e}")
        
        success = True
        error_message = None
        
        try:
            yield
        except Exception as e:
            success = False
            error_message = str(e)
            raise
        finally:
            end_time = time.time()
            duration = end_time - start_time
            
            # Capture end memory
            end_memory = None
            memory_delta = None
            if start_memory is not None:
                try:
                    process = psutil.Process()
                    end_memory = process.memory_info().rss / (1024 * 1024)  # MB
                    memory_delta = end_memory - start_memory
                except Exception:
                    pass
            
            # Create operation metrics
            metrics = OperationMetrics(
                operation_name=operation_name,
                start_time=start_time,
                end_time=end_time,
                duration=duration,
                success=success,
                error_message=error_message,
                memory_start_mb=start_memory,
                memory_end_mb=end_memory,
                memory_delta_mb=memory_delta,
                cpu_percent=cpu_percent,
                additional_metrics=additional_metrics if additional_metrics else None
            )
            
            # Record metrics
            self._record_operation_metrics(metrics)
    
    def _record_operation_metrics(self, metrics: OperationMetrics) -> None:
        """Record operation metrics."""
        with self._lock:
            # Add to current session if active
            if self._current_session:
                self._current_session.operations.append(metrics)
            
            # Add to operation statistics
            self._operation_stats[metrics.operation_name].append(metrics)
            self._recent_operations.append(metrics)
            
            # Log performance metrics
            self._log_operation_performance(metrics)
    
    def _log_operation_performance(self, metrics: OperationMetrics) -> None:
        """Log operation performance metrics."""
        if not self.config.get('performance_monitoring.logging.operation_timing', True):
            return
        
        logger.info(f"PERFORMANCE [{metrics.operation_name}]: {metrics.duration:.3f}s")
        
        if metrics.memory_delta_mb is not None:
            logger.info(f"  Memory: {metrics.memory_delta_mb:+.1f}MB")
        
        if metrics.cpu_percent is not None:
            logger.info(f"  CPU: {metrics.cpu_percent:.1f}%")
        
        if not metrics.success and metrics.error_message:
            logger.warning(f"  Error: {metrics.error_message}")
        
        if metrics.additional_metrics:
            for key, value in metrics.additional_metrics.items():
                logger.debug(f"  {key}: {value}")
    
    def record_cache_hit(self) -> None:
        """Record a cache hit."""
        with self._lock:
            self._cache_stats['hits'] += 1
            self._cache_stats['total_lookups'] += 1
    
    def record_cache_miss(self) -> None:
        """Record a cache miss."""
        with self._lock:
            self._cache_stats['misses'] += 1
            self._cache_stats['total_lookups'] += 1
    
    def record_parameter_change(self, parameter_name: str, old_value: Any, 
                              new_value: Any, reason: str) -> None:
        """
        Record a parameter change during compression.
        
        Args:
            parameter_name: Name of the parameter that changed
            old_value: Previous value
            new_value: New value
            reason: Reason for the change
        """
        if not self._current_session:
            return
        
        change_record = {
            'timestamp': time.time(),
            'parameter': parameter_name,
            'old_value': old_value,
            'new_value': new_value,
            'reason': reason
        }
        
        self._current_session.parameter_changes.append(change_record)
        
        if self.config.get('performance_monitoring.logging.parameter_changes', True):
            logger.info(f"PARAMETER CHANGE [{parameter_name}]: {old_value} → {new_value}")
            logger.info(f"  Reason: {reason}")
    
    def record_quality_evaluation(self, evaluation_type: str, score: Optional[float],
                                computation_time: float, method: str = 'full') -> None:
        """
        Record a quality evaluation.
        
        Args:
            evaluation_type: Type of evaluation (VMAF, SSIM, etc.)
            score: Quality score (None if evaluation failed)
            computation_time: Time taken for evaluation
            method: Evaluation method (full, fast, predicted)
        """
        if not self._current_session:
            return
        
        evaluation_record = {
            'timestamp': time.time(),
            'evaluation_type': evaluation_type,
            'score': score,
            'computation_time': computation_time,
            'method': method,
            'success': score is not None
        }
        
        self._current_session.quality_evaluations.append(evaluation_record)
        
        if self.config.get('performance_monitoring.logging.quality_evaluations', True):
            if score is not None:
                logger.info(f"QUALITY EVALUATION [{evaluation_type}]: {score:.2f} ({method}, {computation_time:.2f}s)")
            else:
                logger.warning(f"QUALITY EVALUATION [{evaluation_type}]: FAILED ({method}, {computation_time:.2f}s)")
    
    def get_cache_hit_rate(self) -> float:
        """Get current cache hit rate as percentage."""
        with self._lock:
            if self._cache_stats['total_lookups'] == 0:
                return 0.0
            return (self._cache_stats['hits'] / self._cache_stats['total_lookups']) * 100
    
    def get_operation_statistics(self, operation_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics for operations.
        
        Args:
            operation_name: Specific operation to get stats for (None for all)
            
        Returns:
            Dictionary containing operation statistics
        """
        with self._lock:
            if operation_name:
                operations = self._operation_stats.get(operation_name, [])
                return self._calculate_operation_stats(operation_name, operations)
            
            # Return stats for all operations
            all_stats = {}
            for op_name, operations in self._operation_stats.items():
                all_stats[op_name] = self._calculate_operation_stats(op_name, operations)
            
            return all_stats
    
    def _calculate_operation_stats(self, operation_name: str, 
                                 operations: List[OperationMetrics]) -> Dict[str, Any]:
        """Calculate statistics for a list of operations."""
        if not operations:
            return {
                'operation_name': operation_name,
                'count': 0,
                'success_rate': 0.0,
                'avg_duration': 0.0,
                'min_duration': 0.0,
                'max_duration': 0.0,
                'total_duration': 0.0
            }
        
        durations = [op.duration for op in operations]
        successful_ops = [op for op in operations if op.success]
        
        stats = {
            'operation_name': operation_name,
            'count': len(operations),
            'success_rate': (len(successful_ops) / len(operations)) * 100,
            'avg_duration': sum(durations) / len(durations),
            'min_duration': min(durations),
            'max_duration': max(durations),
            'total_duration': sum(durations)
        }
        
        # Memory statistics if available
        memory_deltas = [op.memory_delta_mb for op in operations if op.memory_delta_mb is not None]
        if memory_deltas:
            stats['avg_memory_delta_mb'] = sum(memory_deltas) / len(memory_deltas)
            stats['max_memory_delta_mb'] = max(memory_deltas)
            stats['min_memory_delta_mb'] = min(memory_deltas)
        
        return stats
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary."""
        with self._lock:
            # Recent session statistics
            recent_sessions = list(self._recent_sessions)
            successful_sessions = [s for s in recent_sessions if s.success]
            
            session_stats = {
                'total_sessions': len(recent_sessions),
                'successful_sessions': len(successful_sessions),
                'success_rate': (len(successful_sessions) / len(recent_sessions) * 100) if recent_sessions else 0.0
            }
            
            if successful_sessions:
                durations = [s.duration for s in successful_sessions if s.duration]
                if durations:
                    session_stats.update({
                        'avg_session_duration': sum(durations) / len(durations),
                        'min_session_duration': min(durations),
                        'max_session_duration': max(durations)
                    })
                
                # Size utilization statistics
                utilizations = []
                for s in successful_sessions:
                    if s.final_size_mb and s.target_size_mb:
                        utilizations.append((s.final_size_mb / s.target_size_mb) * 100)
                
                if utilizations:
                    session_stats.update({
                        'avg_size_utilization': sum(utilizations) / len(utilizations),
                        'min_size_utilization': min(utilizations),
                        'max_size_utilization': max(utilizations)
                    })
            
            # Cache statistics
            cache_stats = {
                'hit_rate_percent': self.get_cache_hit_rate(),
                'total_hits': self._cache_stats['hits'],
                'total_misses': self._cache_stats['misses'],
                'total_lookups': self._cache_stats['total_lookups']
            }
            
            # System statistics
            system_stats = {}
            try:
                system_stats = {
                    'cpu_count': psutil.cpu_count(),
                    'memory_total_gb': psutil.virtual_memory().total / (1024**3),
                    'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                    'memory_percent': psutil.virtual_memory().percent
                }
            except Exception as e:
                logger.debug(f"Failed to get system stats: {e}")
            
            return {
                'timestamp': time.time(),
                'session_statistics': session_stats,
                'cache_statistics': cache_stats,
                'system_statistics': system_stats,
                'current_profile': self._current_profile.name if self._current_profile else None,
                'monitoring_enabled': self._system_monitor_enabled
            }
    
    def identify_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks based on collected metrics.
        
        Returns:
            List of identified bottlenecks with recommendations
        """
        bottlenecks = []
        
        with self._lock:
            # Analyze operation performance
            for op_name, operations in self._operation_stats.items():
                if len(operations) < 3:  # Need sufficient data
                    continue
                
                stats = self._calculate_operation_stats(op_name, operations)
                
                # Check for slow operations
                if stats['avg_duration'] > 30.0:  # Operations taking more than 30 seconds
                    bottlenecks.append({
                        'type': 'slow_operation',
                        'operation': op_name,
                        'avg_duration': stats['avg_duration'],
                        'recommendation': f"Consider optimizing {op_name} - average duration is {stats['avg_duration']:.1f}s",
                        'severity': 'high' if stats['avg_duration'] > 60.0 else 'medium'
                    })
                
                # Check for high failure rates
                if stats['success_rate'] < 80.0:
                    bottlenecks.append({
                        'type': 'high_failure_rate',
                        'operation': op_name,
                        'success_rate': stats['success_rate'],
                        'recommendation': f"Investigate {op_name} failures - success rate is {stats['success_rate']:.1f}%",
                        'severity': 'high' if stats['success_rate'] < 50.0 else 'medium'
                    })
            
            # Check cache performance
            cache_hit_rate = self.get_cache_hit_rate()
            if cache_hit_rate < 30.0 and self._cache_stats['total_lookups'] > 10:
                bottlenecks.append({
                    'type': 'low_cache_hit_rate',
                    'hit_rate': cache_hit_rate,
                    'recommendation': f"Cache hit rate is low ({cache_hit_rate:.1f}%) - consider cache optimization",
                    'severity': 'medium'
                })
            
            # Check memory usage patterns
            recent_ops = list(self._recent_operations)
            memory_intensive_ops = [op for op in recent_ops 
                                  if op.memory_delta_mb and op.memory_delta_mb > self._memory_threshold_mb]
            
            if len(memory_intensive_ops) > len(recent_ops) * 0.3:  # More than 30% of operations are memory intensive
                avg_memory_usage = sum(op.memory_delta_mb for op in memory_intensive_ops) / len(memory_intensive_ops)
                bottlenecks.append({
                    'type': 'high_memory_usage',
                    'avg_memory_delta_mb': avg_memory_usage,
                    'affected_operations': len(memory_intensive_ops),
                    'recommendation': f"High memory usage detected (avg {avg_memory_usage:.1f}MB) - consider memory optimization",
                    'severity': 'medium'
                })
        
        return bottlenecks
    
    def generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate optimization recommendations based on performance analysis.
        
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Get bottlenecks
        bottlenecks = self.identify_bottlenecks()
        
        # Convert bottlenecks to recommendations
        for bottleneck in bottlenecks:
            recommendations.append({
                'category': 'bottleneck_resolution',
                'priority': bottleneck['severity'],
                'description': bottleneck['recommendation'],
                'details': bottleneck
            })
        
        # Analyze performance trends
        with self._lock:
            recent_sessions = list(self._recent_sessions)
            
            if len(recent_sessions) >= 5:
                # Check for declining performance
                recent_durations = [s.duration for s in recent_sessions[-5:] if s.duration and s.success]
                if len(recent_durations) >= 3:
                    trend = self._calculate_trend(recent_durations)
                    if trend > 0.2:  # Increasing duration trend
                        recommendations.append({
                            'category': 'performance_degradation',
                            'priority': 'medium',
                            'description': 'Performance degradation detected - compression times are increasing',
                            'details': {
                                'trend_slope': trend,
                                'recent_avg_duration': sum(recent_durations) / len(recent_durations)
                            }
                        })
                
                # Check cache utilization
                cache_hit_rate = self.get_cache_hit_rate()
                if cache_hit_rate > 80.0:
                    recommendations.append({
                        'category': 'optimization_opportunity',
                        'priority': 'low',
                        'description': f'Excellent cache performance ({cache_hit_rate:.1f}% hit rate) - consider increasing cache size',
                        'details': {'cache_hit_rate': cache_hit_rate}
                    })
                elif cache_hit_rate < 20.0 and self._cache_stats['total_lookups'] > 20:
                    recommendations.append({
                        'category': 'cache_optimization',
                        'priority': 'high',
                        'description': f'Poor cache performance ({cache_hit_rate:.1f}% hit rate) - review cache strategy',
                        'details': {'cache_hit_rate': cache_hit_rate}
                    })
        
        return recommendations
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend slope for a list of values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        # Simple linear regression slope
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def set_performance_profile(self, profile_name: str) -> bool:
        """
        Set the current performance profile.
        
        Args:
            profile_name: Name of the profile to set
            
        Returns:
            True if profile was set successfully
        """
        if profile_name not in self._performance_profiles:
            logger.warning(f"Unknown performance profile: {profile_name}")
            return False
        
        self._current_profile = self._performance_profiles[profile_name]
        logger.info(f"Set performance profile to: {profile_name}")
        return True
    
    def get_performance_profile(self) -> Optional[PerformanceProfile]:
        """Get the current performance profile."""
        return self._current_profile
    
    def get_available_profiles(self) -> List[str]:
        """Get list of available performance profile names."""
        return list(self._performance_profiles.keys())
    
    def estimate_processing_time(self, input_file: str, target_size_mb: float) -> Dict[str, float]:
        """
        Estimate processing time based on historical data.
        
        Args:
            input_file: Path to input file
            target_size_mb: Target output size
            
        Returns:
            Dictionary with time estimates
        """
        try:
            # Get file size for estimation
            file_size_mb = os.path.getsize(input_file) / (1024 * 1024)
        except Exception:
            file_size_mb = 100  # Default assumption
        
        with self._lock:
            recent_sessions = [s for s in self._recent_sessions if s.success and s.duration]
            
            if len(recent_sessions) < 3:
                # Not enough data, use defaults based on file size
                base_time = max(30, file_size_mb * 0.5)  # 0.5 seconds per MB minimum
                return {
                    'estimated_seconds': base_time,
                    'confidence': 'low',
                    'min_estimate': base_time * 0.5,
                    'max_estimate': base_time * 2.0
                }
            
            # Find similar sessions (similar file sizes and target sizes)
            similar_sessions = []
            for session in recent_sessions:
                try:
                    session_file_size = os.path.getsize(session.input_file) / (1024 * 1024)
                    size_ratio = min(file_size_mb, session_file_size) / max(file_size_mb, session_file_size)
                    target_ratio = min(target_size_mb, session.target_size_mb) / max(target_size_mb, session.target_size_mb)
                    
                    # Consider similar if both ratios are > 0.5
                    if size_ratio > 0.5 and target_ratio > 0.5:
                        similar_sessions.append(session)
                except Exception:
                    continue
            
            if similar_sessions:
                durations = [s.duration for s in similar_sessions]
                avg_duration = sum(durations) / len(durations)
                min_duration = min(durations)
                max_duration = max(durations)
                
                confidence = 'high' if len(similar_sessions) >= 5 else 'medium'
                
                return {
                    'estimated_seconds': avg_duration,
                    'confidence': confidence,
                    'min_estimate': min_duration,
                    'max_estimate': max_duration,
                    'similar_sessions': len(similar_sessions)
                }
            else:
                # Use all sessions for rough estimate
                durations = [s.duration for s in recent_sessions]
                avg_duration = sum(durations) / len(durations)
                
                # Scale by file size ratio
                avg_file_sizes = []
                for session in recent_sessions:
                    try:
                        session_file_size = os.path.getsize(session.input_file) / (1024 * 1024)
                        avg_file_sizes.append(session_file_size)
                    except Exception:
                        continue
                
                if avg_file_sizes:
                    avg_file_size = sum(avg_file_sizes) / len(avg_file_sizes)
                    size_factor = file_size_mb / avg_file_size
                    estimated_duration = avg_duration * size_factor
                else:
                    estimated_duration = avg_duration
                
                return {
                    'estimated_seconds': estimated_duration,
                    'confidence': 'low',
                    'min_estimate': estimated_duration * 0.5,
                    'max_estimate': estimated_duration * 2.0,
                    'similar_sessions': 0
                }
    
    def _save_session_metrics(self, session: CompressionSessionMetrics) -> None:
        """Save session metrics to file."""
        try:
            # Create daily metrics file
            date_str = datetime.fromtimestamp(session.start_time).strftime('%Y-%m-%d')
            metrics_file = self.metrics_dir / f"sessions_{date_str}.jsonl"
            
            # Append session data as JSON line
            with open(metrics_file, 'a', encoding='utf-8') as f:
                json.dump(session.to_dict(), f, separators=(',', ':'))
                f.write('\n')
                
        except Exception as e:
            logger.error(f"Failed to save session metrics: {e}")
    
    def export_metrics(self, start_date: Optional[datetime] = None, 
                      end_date: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Export metrics for analysis.
        
        Args:
            start_date: Start date for export (None for all)
            end_date: End date for export (None for all)
            
        Returns:
            Dictionary containing exported metrics
        """
        exported_data = {
            'export_timestamp': time.time(),
            'start_date': start_date.isoformat() if start_date else None,
            'end_date': end_date.isoformat() if end_date else None,
            'sessions': [],
            'operation_statistics': self.get_operation_statistics(),
            'performance_summary': self.get_performance_summary(),
            'bottlenecks': self.identify_bottlenecks(),
            'recommendations': self.generate_optimization_recommendations()
        }
        
        # Load session data from files
        try:
            for metrics_file in self.metrics_dir.glob("sessions_*.jsonl"):
                try:
                    file_date = datetime.strptime(metrics_file.stem.split('_')[1], '%Y-%m-%d')
                    
                    # Check date range
                    if start_date and file_date < start_date:
                        continue
                    if end_date and file_date > end_date:
                        continue
                    
                    with open(metrics_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            if line.strip():
                                session_data = json.loads(line)
                                exported_data['sessions'].append(session_data)
                                
                except Exception as e:
                    logger.warning(f"Failed to read metrics file {metrics_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
        
        return exported_data
    
    def clear_metrics(self, older_than_days: int = 30) -> int:
        """
        Clear old metrics files.
        
        Args:
            older_than_days: Remove metrics older than this many days
            
        Returns:
            Number of files removed
        """
        removed_count = 0
        cutoff_date = datetime.now() - timedelta(days=older_than_days)
        
        try:
            for metrics_file in self.metrics_dir.glob("sessions_*.jsonl"):
                try:
                    file_date = datetime.strptime(metrics_file.stem.split('_')[1], '%Y-%m-%d')
                    
                    if file_date < cutoff_date:
                        metrics_file.unlink()
                        removed_count += 1
                        logger.debug(f"Removed old metrics file: {metrics_file}")
                        
                except Exception as e:
                    logger.warning(f"Failed to process metrics file {metrics_file}: {e}")
                    
        except Exception as e:
            logger.error(f"Failed to clear metrics: {e}")
        
        logger.info(f"Cleared {removed_count} old metrics files")
        return removed_count