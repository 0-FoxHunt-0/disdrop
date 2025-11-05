"""
User-Facing Performance Controls

This module provides user-friendly interfaces for performance configuration,
progress reporting, and performance profile management.
"""

import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Callable, Union
import logging
from contextlib import contextmanager

try:
    from .performance_monitor import PerformanceMonitor, PerformanceProfile
    from .performance_analyzer import PerformanceAnalyzer
    from ..logger_setup import get_logger
    from ..config_manager import ConfigManager
except ImportError:
    # Fallback for direct execution
    from performance_monitor import PerformanceMonitor, PerformanceProfile
    from performance_analyzer import PerformanceAnalyzer
    from logger_setup import get_logger
    from config_manager import ConfigManager

logger = get_logger(__name__)


@dataclass
class ProgressUpdate:
    """Progress update information."""
    operation: str
    progress_percent: float
    elapsed_time: float
    estimated_total_time: Optional[float]
    estimated_remaining_time: Optional[float]
    current_step: str
    total_steps: Optional[int] = None
    current_step_number: Optional[int] = None
    additional_info: Optional[Dict[str, Any]] = None


class ProgressReporter:
    """Real-time progress reporting with time estimation."""
    
    def __init__(self, performance_monitor: PerformanceMonitor):
        """
        Initialize progress reporter.
        
        Args:
            performance_monitor: Performance monitor instance
        """
        self.monitor = performance_monitor
        self._active_operations = {}
        self._progress_callbacks = []
        self._lock = threading.Lock()
        
    def add_progress_callback(self, callback: Callable[[ProgressUpdate], None]) -> None:
        """
        Add a callback function to receive progress updates.
        
        Args:
            callback: Function to call with progress updates
        """
        with self._lock:
            self._progress_callbacks.append(callback)
    
    def remove_progress_callback(self, callback: Callable[[ProgressUpdate], None]) -> None:
        """
        Remove a progress callback.
        
        Args:
            callback: Callback function to remove
        """
        with self._lock:
            if callback in self._progress_callbacks:
                self._progress_callbacks.remove(callback)
    
    @contextmanager
    def track_operation(self, operation_name: str, estimated_duration: Optional[float] = None,
                       total_steps: Optional[int] = None):
        """
        Context manager to track operation progress.
        
        Args:
            operation_name: Name of the operation
            estimated_duration: Estimated duration in seconds
            total_steps: Total number of steps (optional)
        """
        operation_id = f"{operation_name}_{int(time.time())}"
        start_time = time.time()
        
        # Get better time estimate from historical data
        if not estimated_duration:
            # This would be implemented to use historical data
            estimated_duration = 60.0  # Default fallback
        
        operation_info = {
            'name': operation_name,
            'start_time': start_time,
            'estimated_duration': estimated_duration,
            'total_steps': total_steps,
            'current_step': 0,
            'current_step_name': 'Starting...'
        }
        
        with self._lock:
            self._active_operations[operation_id] = operation_info
        
        try:
            tracker = OperationTracker(self, operation_id, operation_info)
            yield tracker
        finally:
            with self._lock:
                self._active_operations.pop(operation_id, None)
    
    def _notify_progress(self, progress_update: ProgressUpdate) -> None:
        """Notify all registered callbacks of progress update."""
        with self._lock:
            callbacks = self._progress_callbacks.copy()
        
        for callback in callbacks:
            try:
                callback(progress_update)
            except Exception as e:
                logger.warning(f"Progress callback failed: {e}")


class OperationTracker:
    """Tracks progress for a specific operation."""
    
    def __init__(self, reporter: ProgressReporter, operation_id: str, operation_info: Dict[str, Any]):
        """
        Initialize operation tracker.
        
        Args:
            reporter: Progress reporter instance
            operation_id: Unique operation identifier
            operation_info: Operation information dictionary
        """
        self.reporter = reporter
        self.operation_id = operation_id
        self.operation_info = operation_info
        self._last_update_time = time.time()
    
    def update_progress(self, progress_percent: float, current_step: str,
                       step_number: Optional[int] = None, **additional_info) -> None:
        """
        Update operation progress.
        
        Args:
            progress_percent: Progress percentage (0-100)
            current_step: Description of current step
            step_number: Current step number (optional)
            **additional_info: Additional information to include
        """
        current_time = time.time()
        elapsed_time = current_time - self.operation_info['start_time']
        
        # Calculate time estimates
        estimated_total_time = self.operation_info['estimated_duration']
        estimated_remaining_time = None
        
        if progress_percent > 0:
            # Update estimate based on actual progress
            estimated_total_time = elapsed_time / (progress_percent / 100.0)
            estimated_remaining_time = estimated_total_time - elapsed_time
        
        # Create progress update
        progress_update = ProgressUpdate(
            operation=self.operation_info['name'],
            progress_percent=progress_percent,
            elapsed_time=elapsed_time,
            estimated_total_time=estimated_total_time,
            estimated_remaining_time=estimated_remaining_time,
            current_step=current_step,
            total_steps=self.operation_info.get('total_steps'),
            current_step_number=step_number,
            additional_info=additional_info if additional_info else None
        )
        
        # Update operation info
        self.operation_info['current_step'] = step_number or 0
        self.operation_info['current_step_name'] = current_step
        self._last_update_time = current_time
        
        # Notify callbacks
        self.reporter._notify_progress(progress_update)
    
    def update_step(self, step_number: int, step_name: str, **additional_info) -> None:
        """
        Update current step information.
        
        Args:
            step_number: Current step number
            step_name: Name/description of current step
            **additional_info: Additional information
        """
        total_steps = self.operation_info.get('total_steps')
        progress_percent = (step_number / total_steps * 100) if total_steps else 0
        
        self.update_progress(progress_percent, step_name, step_number, **additional_info)


class PerformanceController:
    """Main controller for user-facing performance features."""
    
    def __init__(self, config_manager: ConfigManager):
        """
        Initialize performance controller.
        
        Args:
            config_manager: Configuration manager instance
        """
        self.config = config_manager
        self.monitor = PerformanceMonitor(config_manager)
        self.analyzer = PerformanceAnalyzer(self.monitor, config_manager)
        self.progress_reporter = ProgressReporter(self.monitor)
        
        # Performance settings
        self._current_profile = None
        self._user_preferences = self._load_user_preferences()
        
        # Cancellation support
        self._cancellation_requested = False
        self._cancellation_callbacks = []
        
        logger.info("Performance controller initialized")
    
    def _load_user_preferences(self) -> Dict[str, Any]:
        """Load user performance preferences from configuration."""
        return {
            'preferred_profile': self.config.get('performance_controls.preferred_profile', 'balanced'),
            'show_progress': self.config.get('performance_controls.show_progress', True),
            'show_time_estimates': self.config.get('performance_controls.show_time_estimates', True),
            'enable_cancellation': self.config.get('performance_controls.enable_cancellation', True),
            'performance_notifications': self.config.get('performance_controls.performance_notifications', True)
        }
    
    def set_performance_profile(self, profile_name: str) -> bool:
        """
        Set the active performance profile.
        
        Args:
            profile_name: Name of the profile to activate
            
        Returns:
            True if profile was set successfully
        """
        success = self.monitor.set_performance_profile(profile_name)
        if success:
            self._current_profile = self.monitor.get_performance_profile()
            logger.info(f"Performance profile set to: {profile_name}")
            
            # Log profile settings for user awareness
            if self._user_preferences.get('performance_notifications', True):
                profile = self._current_profile
                logger.info(f"Profile settings: max_evaluation_time={profile.max_quality_evaluation_time}s, "
                           f"max_iterations={profile.max_compression_iterations}, "
                           f"fast_estimation={'enabled' if profile.enable_fast_estimation else 'disabled'}")
        
        return success
    
    def get_available_profiles(self) -> List[Dict[str, Any]]:
        """
        Get available performance profiles with descriptions.
        
        Returns:
            List of profile information dictionaries
        """
        profile_names = self.monitor.get_available_profiles()
        profiles = []
        
        for name in profile_names:
            # Temporarily set profile to get details
            original_profile = self.monitor.get_performance_profile()
            self.monitor.set_performance_profile(name)
            profile = self.monitor.get_performance_profile()
            
            # Restore original profile
            if original_profile:
                self.monitor.set_performance_profile(original_profile.name)
            
            profiles.append({
                'name': name,
                'description': self._get_profile_description(profile),
                'settings': profile.to_dict() if profile else {},
                'recommended_for': self._get_profile_recommendations(name)
            })
        
        return profiles
    
    def _get_profile_description(self, profile: Optional[PerformanceProfile]) -> str:
        """Get user-friendly description for a profile."""
        if not profile:
            return "Unknown profile"
        
        descriptions = {
            'fast': 'Optimized for speed with minimal quality evaluation time',
            'balanced': 'Good balance between speed and quality with moderate evaluation time',
            'quality': 'Prioritizes quality with longer evaluation times and more iterations'
        }
        
        return descriptions.get(profile.name, f'Custom profile: {profile.name}')
    
    def _get_profile_recommendations(self, profile_name: str) -> List[str]:
        """Get recommendations for when to use each profile."""
        recommendations = {
            'fast': [
                'Batch processing of many files',
                'Quick previews and testing',
                'Time-sensitive workflows',
                'Lower-end hardware'
            ],
            'balanced': [
                'General purpose compression',
                'Most content types',
                'Moderate hardware capabilities',
                'Production workflows'
            ],
            'quality': [
                'High-quality content creation',
                'Final production outputs',
                'Powerful hardware available',
                'Quality-critical applications'
            ]
        }
        
        return recommendations.get(profile_name, ['Custom use cases'])
    
    def estimate_processing_time(self, input_file: str, target_size_mb: float,
                               platform: Optional[str] = None) -> Dict[str, Any]:
        """
        Estimate processing time for a compression operation.
        
        Args:
            input_file: Path to input file
            target_size_mb: Target output size in MB
            platform: Target platform (optional)
            
        Returns:
            Dictionary with time estimates and confidence information
        """
        # Get base estimate from monitor
        base_estimate = self.monitor.estimate_processing_time(input_file, target_size_mb)
        
        # Adjust based on current profile
        profile = self.monitor.get_performance_profile()
        if profile:
            # Fast profile reduces time, quality profile increases time
            if profile.name == 'fast':
                multiplier = 0.7
            elif profile.name == 'quality':
                multiplier = 1.5
            else:
                multiplier = 1.0
            
            base_estimate['estimated_seconds'] *= multiplier
            base_estimate['min_estimate'] *= multiplier
            base_estimate['max_estimate'] *= multiplier
        
        # Add user-friendly formatting
        estimated_seconds = base_estimate['estimated_seconds']
        
        return {
            **base_estimate,
            'formatted_estimate': self._format_duration(estimated_seconds),
            'formatted_range': f"{self._format_duration(base_estimate['min_estimate'])} - {self._format_duration(base_estimate['max_estimate'])}",
            'profile_used': profile.name if profile else 'unknown',
            'factors_considered': [
                'Historical performance data',
                'File size and complexity',
                'Current performance profile',
                'System capabilities'
            ]
        }
    
    def _format_duration(self, seconds: float) -> str:
        """Format duration in user-friendly format."""
        if seconds < 60:
            return f"{seconds:.0f} seconds"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f} minutes"
        else:
            hours = seconds / 3600
            return f"{hours:.1f} hours"
    
    def start_compression_with_progress(self, input_file: str, target_size_mb: float,
                                      platform: Optional[str] = None,
                                      progress_callback: Optional[Callable[[ProgressUpdate], None]] = None) -> str:
        """
        Start compression with progress tracking.
        
        Args:
            input_file: Path to input file
            target_size_mb: Target output size in MB
            platform: Target platform (optional)
            progress_callback: Optional callback for progress updates
            
        Returns:
            Session ID for tracking
        """
        # Add progress callback if provided
        if progress_callback:
            self.progress_reporter.add_progress_callback(progress_callback)
        
        # Start compression session
        session_id = self.monitor.start_compression_session(input_file, target_size_mb, platform)
        
        # Log user-friendly start message
        if self._user_preferences.get('performance_notifications', True):
            profile = self.monitor.get_performance_profile()
            profile_name = profile.name if profile else 'default'
            
            logger.info(f"Starting compression with {profile_name} profile")
            
            # Show time estimate
            if self._user_preferences.get('show_time_estimates', True):
                estimate = self.estimate_processing_time(input_file, target_size_mb, platform)
                logger.info(f"Estimated processing time: {estimate['formatted_estimate']} "
                           f"(confidence: {estimate['confidence']})")
        
        return session_id
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get user-friendly performance summary.
        
        Returns:
            Dictionary with performance information formatted for users
        """
        summary = self.monitor.get_performance_summary()
        
        # Add user-friendly formatting
        session_stats = summary.get('session_statistics', {})
        cache_stats = summary.get('cache_statistics', {})
        
        user_summary = {
            'current_profile': summary.get('current_profile', 'unknown'),
            'recent_performance': {
                'total_sessions': session_stats.get('total_sessions', 0),
                'success_rate': f"{session_stats.get('success_rate', 0):.1f}%",
                'average_processing_time': self._format_duration(session_stats.get('avg_session_duration', 0))
            },
            'cache_performance': {
                'hit_rate': f"{cache_stats.get('hit_rate_percent', 0):.1f}%",
                'total_lookups': cache_stats.get('total_lookups', 0),
                'efficiency_status': self._assess_cache_efficiency(cache_stats.get('hit_rate_percent', 0))
            },
            'system_status': self._get_system_status_summary(summary.get('system_statistics', {})),
            'recommendations': self._get_user_recommendations()
        }
        
        return user_summary
    
    def _assess_cache_efficiency(self, hit_rate: float) -> str:
        """Assess cache efficiency for user display."""
        if hit_rate >= 80:
            return 'Excellent'
        elif hit_rate >= 60:
            return 'Good'
        elif hit_rate >= 40:
            return 'Fair'
        elif hit_rate >= 20:
            return 'Poor'
        else:
            return 'Very Poor'
    
    def _get_system_status_summary(self, system_stats: Dict[str, Any]) -> Dict[str, str]:
        """Get user-friendly system status summary."""
        if not system_stats:
            return {'status': 'Unknown'}
        
        memory_percent = system_stats.get('memory_percent', 0)
        memory_status = 'Good'
        if memory_percent > 90:
            memory_status = 'Critical'
        elif memory_percent > 80:
            memory_status = 'High'
        elif memory_percent > 70:
            memory_status = 'Moderate'
        
        return {
            'memory_usage': f"{memory_percent:.1f}%",
            'memory_status': memory_status,
            'available_memory': f"{system_stats.get('memory_available_gb', 0):.1f} GB"
        }
    
    def _get_user_recommendations(self) -> List[str]:
        """Get user-friendly recommendations."""
        recommendations = []
        
        # Get performance summary for analysis
        summary = self.monitor.get_performance_summary()
        cache_hit_rate = summary.get('cache_statistics', {}).get('hit_rate_percent', 0)
        
        if cache_hit_rate < 30:
            recommendations.append("Consider clearing and rebuilding the performance cache for better efficiency")
        
        # Check if user is using optimal profile
        current_profile = self.monitor.get_performance_profile()
        if current_profile and current_profile.name != self._user_preferences.get('preferred_profile'):
            recommendations.append(f"Consider switching to your preferred '{self._user_preferences['preferred_profile']}' profile")
        
        # System-based recommendations
        system_stats = summary.get('system_statistics', {})
        memory_percent = system_stats.get('memory_percent', 0)
        if memory_percent > 85:
            recommendations.append("High memory usage detected - consider using 'fast' profile or closing other applications")
        
        if not recommendations:
            recommendations.append("Performance is optimal - no immediate recommendations")
        
        return recommendations
    
    def request_cancellation(self) -> None:
        """Request cancellation of current operations."""
        if not self._user_preferences.get('enable_cancellation', True):
            logger.warning("Cancellation is disabled in user preferences")
            return
        
        self._cancellation_requested = True
        logger.info("Cancellation requested by user")
        
        # Notify cancellation callbacks
        for callback in self._cancellation_callbacks:
            try:
                callback()
            except Exception as e:
                logger.warning(f"Cancellation callback failed: {e}")
    
    def is_cancellation_requested(self) -> bool:
        """Check if cancellation has been requested."""
        return self._cancellation_requested
    
    def reset_cancellation(self) -> None:
        """Reset cancellation state."""
        self._cancellation_requested = False
    
    def add_cancellation_callback(self, callback: Callable[[], None]) -> None:
        """
        Add callback to be called when cancellation is requested.
        
        Args:
            callback: Function to call on cancellation
        """
        self._cancellation_callbacks.append(callback)
    
    def remove_cancellation_callback(self, callback: Callable[[], None]) -> None:
        """
        Remove cancellation callback.
        
        Args:
            callback: Callback function to remove
        """
        if callback in self._cancellation_callbacks:
            self._cancellation_callbacks.remove(callback)
    
    def get_bottleneck_report(self) -> Dict[str, Any]:
        """
        Get user-friendly bottleneck analysis report.
        
        Returns:
            Dictionary with bottleneck information formatted for users
        """
        bottlenecks = self.analyzer.identify_performance_bottlenecks()
        
        if not bottlenecks:
            return {
                'status': 'No bottlenecks detected',
                'message': 'Your system is performing optimally',
                'recommendations': ['Continue monitoring performance regularly']
            }
        
        # Categorize bottlenecks by severity
        critical = [b for b in bottlenecks if b['severity'] == 'critical']
        high = [b for b in bottlenecks if b['severity'] == 'high']
        medium = [b for b in bottlenecks if b['severity'] == 'medium']
        
        # Generate user-friendly summary
        summary_parts = []
        if critical:
            summary_parts.append(f"{len(critical)} critical issue{'s' if len(critical) != 1 else ''}")
        if high:
            summary_parts.append(f"{len(high)} high-priority issue{'s' if len(high) != 1 else ''}")
        if medium:
            summary_parts.append(f"{len(medium)} medium-priority issue{'s' if len(medium) != 1 else ''}")
        
        status = "Issues detected: " + ", ".join(summary_parts)
        
        # Get top recommendations
        all_recommendations = []
        for bottleneck in bottlenecks[:3]:  # Top 3 bottlenecks
            all_recommendations.extend(bottleneck.get('recommendations', []))
        
        return {
            'status': status,
            'total_issues': len(bottlenecks),
            'critical_issues': len(critical),
            'high_priority_issues': len(high),
            'top_recommendations': all_recommendations[:5],  # Top 5 recommendations
            'detailed_bottlenecks': [
                {
                    'type': b['type'].replace('_', ' ').title(),
                    'severity': b['severity'].title(),
                    'description': self._get_bottleneck_description(b),
                    'impact': self._get_bottleneck_impact(b)
                }
                for b in bottlenecks[:3]  # Show top 3 detailed
            ]
        }
    
    def _get_bottleneck_description(self, bottleneck: Dict[str, Any]) -> str:
        """Get user-friendly bottleneck description."""
        bottleneck_type = bottleneck['type']
        operation = bottleneck.get('operation', 'system')
        
        descriptions = {
            'slow_operation': f"Operation '{operation}' is taking longer than expected",
            'high_failure_rate': f"Operation '{operation}' is failing frequently",
            'high_memory_usage': f"Operation '{operation}' is using excessive memory",
            'poor_cache_performance': "Cache system is not performing efficiently"
        }
        
        return descriptions.get(bottleneck_type, f"Performance issue in {operation}")
    
    def _get_bottleneck_impact(self, bottleneck: Dict[str, Any]) -> str:
        """Get user-friendly impact description."""
        severity = bottleneck['severity']
        bottleneck_type = bottleneck['type']
        
        if severity == 'critical':
            return "Severely impacting performance and user experience"
        elif severity == 'high':
            if bottleneck_type == 'slow_operation':
                return "Significantly increasing processing times"
            elif bottleneck_type == 'high_failure_rate':
                return "Causing frequent operation failures"
            else:
                return "Notably degrading system performance"
        elif severity == 'medium':
            return "Moderately affecting performance"
        else:
            return "Minor performance impact"
    
    def export_performance_report(self) -> Dict[str, Any]:
        """
        Export comprehensive performance report for user review.
        
        Returns:
            Dictionary containing complete performance analysis
        """
        optimization_report = self.analyzer.generate_optimization_report()
        user_summary = self.get_performance_summary()
        bottleneck_report = self.get_bottleneck_report()
        
        return {
            'report_timestamp': optimization_report['report_timestamp'],
            'user_summary': user_summary,
            'bottleneck_analysis': bottleneck_report,
            'optimization_potential': optimization_report['executive_summary']['optimization_potential'],
            'key_recommendations': optimization_report['next_steps'],
            'performance_trends': optimization_report['performance_trends'],
            'current_configuration': {
                'active_profile': user_summary['current_profile'],
                'user_preferences': self._user_preferences
            },
            'detailed_analysis': optimization_report  # Full technical report
        }