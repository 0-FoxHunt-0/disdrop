"""
Performance Analysis Tools

This module provides advanced analysis capabilities for performance metrics,
including trend analysis, bottleneck identification, and optimization recommendations.
"""

import json
import math
import statistics
from collections import defaultdict, Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
import logging

try:
    from .performance_monitor import CompressionSessionMetrics, OperationMetrics, PerformanceMonitor
    from ..logger_setup import get_logger
    from ..config_manager import ConfigManager
except ImportError:
    # Fallback for direct execution
    from performance_monitor import CompressionSessionMetrics, OperationMetrics, PerformanceMonitor
    from logger_setup import get_logger
    from config_manager import ConfigManager

logger = get_logger(__name__)


class PerformanceAnalyzer:
    """Advanced performance analysis and optimization recommendation system."""
    
    def __init__(self, performance_monitor: PerformanceMonitor, config_manager: ConfigManager):
        """
        Initialize performance analyzer.
        
        Args:
            performance_monitor: Performance monitor instance
            config_manager: Configuration manager instance
        """
        self.monitor = performance_monitor
        self.config = config_manager
        
        # Analysis thresholds
        self.slow_operation_threshold = self.config.get('performance_analysis.thresholds.slow_operation_seconds', 30.0)
        self.high_memory_threshold = self.config.get('performance_analysis.thresholds.high_memory_mb', 500.0)
        self.low_cache_hit_rate = self.config.get('performance_analysis.thresholds.low_cache_hit_rate', 30.0)
        self.poor_success_rate = self.config.get('performance_analysis.thresholds.poor_success_rate', 80.0)
        
        logger.info("Performance analyzer initialized")
    
    def analyze_performance_trends(self, days: int = 7) -> Dict[str, Any]:
        """
        Analyze performance trends over the specified time period.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dictionary containing trend analysis results
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Export metrics for the time period
        metrics_data = self.monitor.export_metrics(start_date, end_date)
        sessions = metrics_data.get('sessions', [])
        
        if not sessions:
            return {
                'period_days': days,
                'total_sessions': 0,
                'message': 'No session data available for analysis'
            }
        
        # Convert to session objects for analysis
        session_objects = []
        for session_data in sessions:
            try:
                session = CompressionSessionMetrics.from_dict(session_data)
                session_objects.append(session)
            except Exception as e:
                logger.debug(f"Failed to parse session data: {e}")
                continue
        
        return self._analyze_session_trends(session_objects, days)
    
    def _analyze_session_trends(self, sessions: List[CompressionSessionMetrics], days: int) -> Dict[str, Any]:
        """Analyze trends in session data."""
        successful_sessions = [s for s in sessions if s.success and s.duration]
        
        if not successful_sessions:
            return {
                'period_days': days,
                'total_sessions': len(sessions),
                'successful_sessions': 0,
                'message': 'No successful sessions for trend analysis'
            }
        
        # Time-based analysis
        daily_stats = self._calculate_daily_statistics(successful_sessions)
        
        # Performance metrics trends
        durations = [s.duration for s in successful_sessions]
        duration_trend = self._calculate_trend_slope(durations)
        
        # Size utilization trends
        utilizations = []
        for s in successful_sessions:
            if s.final_size_mb and s.target_size_mb and s.target_size_mb > 0:
                utilizations.append((s.final_size_mb / s.target_size_mb) * 100)
        
        utilization_trend = self._calculate_trend_slope(utilizations) if utilizations else 0.0
        
        # Quality trends (if available)
        quality_scores = [s.quality_score for s in successful_sessions if s.quality_score is not None]
        quality_trend = self._calculate_trend_slope(quality_scores) if quality_scores else 0.0
        
        # Platform analysis
        platform_stats = self._analyze_platform_performance(successful_sessions)
        
        # Operation performance trends
        operation_trends = self._analyze_operation_trends(successful_sessions)
        
        return {
            'period_days': days,
            'analysis_timestamp': datetime.now().isoformat(),
            'session_summary': {
                'total_sessions': len(sessions),
                'successful_sessions': len(successful_sessions),
                'success_rate': (len(successful_sessions) / len(sessions)) * 100,
                'avg_duration': statistics.mean(durations),
                'median_duration': statistics.median(durations),
                'duration_std': statistics.stdev(durations) if len(durations) > 1 else 0.0
            },
            'performance_trends': {
                'duration_trend': {
                    'slope': duration_trend,
                    'direction': self._interpret_trend(duration_trend),
                    'significance': self._assess_trend_significance(duration_trend, durations)
                },
                'utilization_trend': {
                    'slope': utilization_trend,
                    'direction': self._interpret_trend(utilization_trend),
                    'avg_utilization': statistics.mean(utilizations) if utilizations else 0.0
                },
                'quality_trend': {
                    'slope': quality_trend,
                    'direction': self._interpret_trend(quality_trend),
                    'avg_quality': statistics.mean(quality_scores) if quality_scores else None
                }
            },
            'daily_statistics': daily_stats,
            'platform_analysis': platform_stats,
            'operation_trends': operation_trends
        }
    
    def _calculate_daily_statistics(self, sessions: List[CompressionSessionMetrics]) -> Dict[str, Any]:
        """Calculate daily performance statistics."""
        daily_data = defaultdict(list)
        
        for session in sessions:
            date_key = datetime.fromtimestamp(session.start_time).strftime('%Y-%m-%d')
            daily_data[date_key].append(session)
        
        daily_stats = {}
        for date, day_sessions in daily_data.items():
            durations = [s.duration for s in day_sessions if s.duration]
            
            daily_stats[date] = {
                'session_count': len(day_sessions),
                'avg_duration': statistics.mean(durations) if durations else 0.0,
                'total_processing_time': sum(durations),
                'success_rate': (len([s for s in day_sessions if s.success]) / len(day_sessions)) * 100
            }
        
        return daily_stats
    
    def _analyze_platform_performance(self, sessions: List[CompressionSessionMetrics]) -> Dict[str, Any]:
        """Analyze performance by platform."""
        platform_data = defaultdict(list)
        
        for session in sessions:
            platform = session.platform or 'unknown'
            platform_data[platform].append(session)
        
        platform_stats = {}
        for platform, platform_sessions in platform_data.items():
            durations = [s.duration for s in platform_sessions if s.duration]
            utilizations = []
            
            for s in platform_sessions:
                if s.final_size_mb and s.target_size_mb and s.target_size_mb > 0:
                    utilizations.append((s.final_size_mb / s.target_size_mb) * 100)
            
            platform_stats[platform] = {
                'session_count': len(platform_sessions),
                'avg_duration': statistics.mean(durations) if durations else 0.0,
                'avg_utilization': statistics.mean(utilizations) if utilizations else 0.0,
                'success_rate': (len([s for s in platform_sessions if s.success]) / len(platform_sessions)) * 100
            }
        
        return platform_stats
    
    def _analyze_operation_trends(self, sessions: List[CompressionSessionMetrics]) -> Dict[str, Any]:
        """Analyze trends in operation performance."""
        operation_data = defaultdict(list)
        
        for session in sessions:
            for operation in session.operations:
                operation_data[operation.operation_name].append(operation)
        
        operation_trends = {}
        for op_name, operations in operation_data.items():
            if len(operations) < 3:  # Need sufficient data for trend analysis
                continue
            
            durations = [op.duration for op in operations]
            success_rates = [1.0 if op.success else 0.0 for op in operations]
            
            # Calculate trends
            duration_trend = self._calculate_trend_slope(durations)
            success_trend = self._calculate_trend_slope(success_rates)
            
            operation_trends[op_name] = {
                'operation_count': len(operations),
                'avg_duration': statistics.mean(durations),
                'duration_trend': {
                    'slope': duration_trend,
                    'direction': self._interpret_trend(duration_trend)
                },
                'success_rate': statistics.mean(success_rates) * 100,
                'success_trend': {
                    'slope': success_trend,
                    'direction': self._interpret_trend(success_trend)
                }
            }
        
        return operation_trends
    
    def _calculate_trend_slope(self, values: List[float]) -> float:
        """Calculate linear trend slope for a series of values."""
        if len(values) < 2:
            return 0.0
        
        n = len(values)
        x_values = list(range(n))
        
        # Calculate means
        x_mean = sum(x_values) / n
        y_mean = sum(values) / n
        
        # Calculate slope using least squares
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, values))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _interpret_trend(self, slope: float) -> str:
        """Interpret trend slope as direction."""
        if abs(slope) < 0.01:
            return 'stable'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _assess_trend_significance(self, slope: float, values: List[float]) -> str:
        """Assess the significance of a trend."""
        if len(values) < 5:
            return 'insufficient_data'
        
        # Simple significance assessment based on slope magnitude relative to data range
        data_range = max(values) - min(values)
        if data_range == 0:
            return 'no_variation'
        
        relative_slope = abs(slope) / data_range * len(values)
        
        if relative_slope > 0.5:
            return 'significant'
        elif relative_slope > 0.2:
            return 'moderate'
        else:
            return 'minimal'
    
    def identify_performance_bottlenecks(self) -> List[Dict[str, Any]]:
        """
        Identify performance bottlenecks with detailed analysis.
        
        Returns:
            List of identified bottlenecks with analysis and recommendations
        """
        bottlenecks = []
        
        # Get operation statistics
        operation_stats = self.monitor.get_operation_statistics()
        
        for op_name, stats in operation_stats.items():
            if stats['count'] < 3:  # Need sufficient data
                continue
            
            # Check for slow operations
            if stats['avg_duration'] > self.slow_operation_threshold:
                severity = self._calculate_severity(stats['avg_duration'], self.slow_operation_threshold, 120.0)
                
                bottlenecks.append({
                    'type': 'slow_operation',
                    'operation': op_name,
                    'severity': severity,
                    'metrics': {
                        'avg_duration': stats['avg_duration'],
                        'max_duration': stats['max_duration'],
                        'operation_count': stats['count']
                    },
                    'analysis': {
                        'performance_impact': self._assess_performance_impact(stats),
                        'variability': stats['max_duration'] / stats['avg_duration'] if stats['avg_duration'] > 0 else 0
                    },
                    'recommendations': self._generate_slow_operation_recommendations(op_name, stats)
                })
            
            # Check for high failure rates
            if stats['success_rate'] < self.poor_success_rate:
                severity = self._calculate_severity(100 - stats['success_rate'], 20, 50)
                
                bottlenecks.append({
                    'type': 'high_failure_rate',
                    'operation': op_name,
                    'severity': severity,
                    'metrics': {
                        'success_rate': stats['success_rate'],
                        'failure_count': stats['count'] - int(stats['count'] * stats['success_rate'] / 100),
                        'operation_count': stats['count']
                    },
                    'analysis': {
                        'reliability_impact': self._assess_reliability_impact(stats['success_rate']),
                        'failure_frequency': (100 - stats['success_rate']) / 100
                    },
                    'recommendations': self._generate_failure_rate_recommendations(op_name, stats)
                })
            
            # Check for high memory usage
            if 'avg_memory_delta_mb' in stats and stats['avg_memory_delta_mb'] > self.high_memory_threshold:
                severity = self._calculate_severity(stats['avg_memory_delta_mb'], self.high_memory_threshold, 1000.0)
                
                bottlenecks.append({
                    'type': 'high_memory_usage',
                    'operation': op_name,
                    'severity': severity,
                    'metrics': {
                        'avg_memory_delta_mb': stats['avg_memory_delta_mb'],
                        'max_memory_delta_mb': stats.get('max_memory_delta_mb', 0),
                        'operation_count': stats['count']
                    },
                    'analysis': {
                        'memory_efficiency': self._assess_memory_efficiency(stats),
                        'scalability_risk': self._assess_scalability_risk(stats['avg_memory_delta_mb'])
                    },
                    'recommendations': self._generate_memory_usage_recommendations(op_name, stats)
                })
        
        # Check cache performance
        cache_hit_rate = self.monitor.get_cache_hit_rate()
        cache_stats = self.monitor._cache_stats
        
        if cache_hit_rate < self.low_cache_hit_rate and cache_stats['total_lookups'] > 10:
            severity = self._calculate_severity(self.low_cache_hit_rate - cache_hit_rate, 10, 30)
            
            bottlenecks.append({
                'type': 'poor_cache_performance',
                'severity': severity,
                'metrics': {
                    'hit_rate': cache_hit_rate,
                    'total_lookups': cache_stats['total_lookups'],
                    'hits': cache_stats['hits'],
                    'misses': cache_stats['misses']
                },
                'analysis': {
                    'efficiency_loss': self._calculate_cache_efficiency_loss(cache_hit_rate),
                    'potential_speedup': self._calculate_potential_cache_speedup(cache_hit_rate)
                },
                'recommendations': self._generate_cache_recommendations(cache_hit_rate, cache_stats)
            })
        
        # Sort bottlenecks by severity
        severity_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        bottlenecks.sort(key=lambda x: severity_order.get(x['severity'], 4))
        
        return bottlenecks
    
    def _calculate_severity(self, value: float, threshold: float, critical_threshold: float) -> str:
        """Calculate severity level based on value and thresholds."""
        if value >= critical_threshold:
            return 'critical'
        elif value >= threshold * 2:
            return 'high'
        elif value >= threshold * 1.5:
            return 'medium'
        else:
            return 'low'
    
    def _assess_performance_impact(self, stats: Dict[str, Any]) -> str:
        """Assess the performance impact of slow operations."""
        total_time = stats['avg_duration'] * stats['count']
        
        if total_time > 600:  # More than 10 minutes total
            return 'high'
        elif total_time > 180:  # More than 3 minutes total
            return 'medium'
        else:
            return 'low'
    
    def _assess_reliability_impact(self, success_rate: float) -> str:
        """Assess the reliability impact of failure rates."""
        if success_rate < 50:
            return 'critical'
        elif success_rate < 70:
            return 'high'
        elif success_rate < 85:
            return 'medium'
        else:
            return 'low'
    
    def _assess_memory_efficiency(self, stats: Dict[str, Any]) -> str:
        """Assess memory efficiency of operations."""
        avg_memory = stats.get('avg_memory_delta_mb', 0)
        max_memory = stats.get('max_memory_delta_mb', avg_memory)
        
        # Check memory usage variability
        if max_memory > avg_memory * 3:
            return 'poor'  # High variability indicates inefficiency
        elif avg_memory > 1000:
            return 'poor'  # Very high memory usage
        elif avg_memory > 500:
            return 'moderate'
        else:
            return 'good'
    
    def _assess_scalability_risk(self, avg_memory_mb: float) -> str:
        """Assess scalability risk based on memory usage."""
        if avg_memory_mb > 2000:
            return 'high'
        elif avg_memory_mb > 1000:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_cache_efficiency_loss(self, hit_rate: float) -> float:
        """Calculate efficiency loss due to poor cache performance."""
        optimal_hit_rate = 80.0  # Assume 80% is optimal
        return max(0, optimal_hit_rate - hit_rate) / optimal_hit_rate
    
    def _calculate_potential_cache_speedup(self, hit_rate: float) -> float:
        """Calculate potential speedup from improved cache performance."""
        if hit_rate >= 80:
            return 1.0  # No significant improvement expected
        
        # Assume cache hits are 10x faster than cache misses
        current_relative_time = (hit_rate * 0.1 + (100 - hit_rate) * 1.0) / 100
        optimal_relative_time = (80 * 0.1 + 20 * 1.0) / 100
        
        return current_relative_time / optimal_relative_time
    
    def _generate_slow_operation_recommendations(self, operation: str, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations for slow operations."""
        recommendations = []
        
        avg_duration = stats['avg_duration']
        variability = stats['max_duration'] / stats['avg_duration'] if stats['avg_duration'] > 0 else 0
        
        if 'quality' in operation.lower():
            recommendations.append("Consider enabling fast quality estimation to reduce evaluation time")
            recommendations.append("Implement quality result caching to avoid redundant evaluations")
            if avg_duration > 60:
                recommendations.append("Set time limits for quality evaluations with graceful degradation")
        
        if 'compression' in operation.lower():
            recommendations.append("Optimize FFmpeg parameters for better performance")
            recommendations.append("Consider hardware acceleration if not already enabled")
            if variability > 2:
                recommendations.append("Investigate high variability in compression times")
        
        if 'cache' in operation.lower():
            recommendations.append("Review cache implementation for performance bottlenecks")
            recommendations.append("Consider cache size optimization or cleanup strategies")
        
        # General recommendations
        if avg_duration > 120:
            recommendations.append("Consider breaking down operation into smaller, parallelizable tasks")
        
        if variability > 3:
            recommendations.append("High variability detected - investigate input-dependent performance issues")
        
        return recommendations
    
    def _generate_failure_rate_recommendations(self, operation: str, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations for high failure rates."""
        recommendations = []
        
        success_rate = stats['success_rate']
        
        if success_rate < 50:
            recommendations.append("Critical failure rate - immediate investigation required")
            recommendations.append("Review error logs and implement better error handling")
        
        if 'quality' in operation.lower():
            recommendations.append("Implement fallback quality estimation methods")
            recommendations.append("Add input validation to prevent quality evaluation failures")
        
        if 'compression' in operation.lower():
            recommendations.append("Add parameter validation and fallback compression settings")
            recommendations.append("Implement progressive quality reduction for difficult inputs")
        
        recommendations.append("Add retry logic with exponential backoff for transient failures")
        recommendations.append("Implement better input validation and error recovery")
        
        return recommendations
    
    def _generate_memory_usage_recommendations(self, operation: str, stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations for high memory usage."""
        recommendations = []
        
        avg_memory = stats.get('avg_memory_delta_mb', 0)
        max_memory = stats.get('max_memory_delta_mb', avg_memory)
        
        if avg_memory > 1000:
            recommendations.append("High memory usage detected - implement memory optimization")
            recommendations.append("Consider processing large files in chunks or segments")
        
        if max_memory > avg_memory * 3:
            recommendations.append("High memory variability - investigate memory leaks or inefficient algorithms")
        
        if 'quality' in operation.lower():
            recommendations.append("Optimize quality evaluation to use less memory")
            recommendations.append("Consider sampling-based quality assessment for large files")
        
        recommendations.append("Implement memory monitoring and cleanup between operations")
        recommendations.append("Consider memory-efficient data structures and algorithms")
        
        return recommendations
    
    def _generate_cache_recommendations(self, hit_rate: float, cache_stats: Dict[str, Any]) -> List[str]:
        """Generate recommendations for cache performance."""
        recommendations = []
        
        if hit_rate < 20:
            recommendations.append("Very low cache hit rate - review cache key generation strategy")
            recommendations.append("Consider increasing cache size or adjusting TTL settings")
        elif hit_rate < 50:
            recommendations.append("Low cache hit rate - optimize cache key generation for better similarity detection")
        
        total_lookups = cache_stats['total_lookups']
        if total_lookups > 100:
            recommendations.append("High cache usage detected - consider cache performance optimization")
        
        recommendations.append("Implement cache warming strategies for common video types")
        recommendations.append("Review cache eviction policies and size limits")
        
        return recommendations
    
    def generate_optimization_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive optimization report.
        
        Returns:
            Dictionary containing detailed optimization analysis and recommendations
        """
        # Get performance trends
        trends = self.analyze_performance_trends(days=7)
        
        # Get bottlenecks
        bottlenecks = self.identify_performance_bottlenecks()
        
        # Get current performance summary
        performance_summary = self.monitor.get_performance_summary()
        
        # Generate prioritized recommendations
        recommendations = self._generate_prioritized_recommendations(bottlenecks, trends)
        
        # Calculate optimization potential
        optimization_potential = self._calculate_optimization_potential(bottlenecks, performance_summary)
        
        return {
            'report_timestamp': datetime.now().isoformat(),
            'analysis_period_days': 7,
            'executive_summary': {
                'total_bottlenecks': len(bottlenecks),
                'critical_issues': len([b for b in bottlenecks if b['severity'] == 'critical']),
                'optimization_potential': optimization_potential,
                'primary_concerns': self._identify_primary_concerns(bottlenecks)
            },
            'performance_trends': trends,
            'identified_bottlenecks': bottlenecks,
            'prioritized_recommendations': recommendations,
            'current_performance': performance_summary,
            'next_steps': self._generate_next_steps(recommendations)
        }
    
    def _generate_prioritized_recommendations(self, bottlenecks: List[Dict[str, Any]], 
                                           trends: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate prioritized optimization recommendations."""
        recommendations = []
        
        # Extract recommendations from bottlenecks
        for bottleneck in bottlenecks:
            for rec in bottleneck.get('recommendations', []):
                recommendations.append({
                    'category': bottleneck['type'],
                    'priority': bottleneck['severity'],
                    'description': rec,
                    'affected_operation': bottleneck.get('operation'),
                    'expected_impact': self._estimate_recommendation_impact(bottleneck, rec)
                })
        
        # Add trend-based recommendations
        if trends.get('performance_trends'):
            duration_trend = trends['performance_trends'].get('duration_trend', {})
            if duration_trend.get('direction') == 'increasing' and duration_trend.get('significance') in ['significant', 'moderate']:
                recommendations.append({
                    'category': 'performance_degradation',
                    'priority': 'high',
                    'description': 'Performance degradation trend detected - implement performance monitoring alerts',
                    'expected_impact': 'high'
                })
        
        # Sort by priority and expected impact
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        impact_order = {'high': 0, 'medium': 1, 'low': 2}
        
        recommendations.sort(key=lambda x: (
            priority_order.get(x['priority'], 4),
            impact_order.get(x['expected_impact'], 3)
        ))
        
        return recommendations
    
    def _estimate_recommendation_impact(self, bottleneck: Dict[str, Any], recommendation: str) -> str:
        """Estimate the impact of implementing a recommendation."""
        severity = bottleneck['severity']
        bottleneck_type = bottleneck['type']
        
        # High-impact recommendations
        if 'hardware acceleration' in recommendation.lower():
            return 'high'
        if 'caching' in recommendation.lower() and bottleneck_type == 'poor_cache_performance':
            return 'high'
        if 'time limits' in recommendation.lower() and severity in ['critical', 'high']:
            return 'high'
        
        # Medium-impact recommendations
        if 'optimization' in recommendation.lower():
            return 'medium'
        if 'fallback' in recommendation.lower():
            return 'medium'
        
        # Default to low impact
        return 'low'
    
    def _calculate_optimization_potential(self, bottlenecks: List[Dict[str, Any]], 
                                        performance_summary: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate overall optimization potential."""
        potential_speedup = 1.0
        potential_reliability_improvement = 0.0
        potential_memory_savings = 0.0
        
        for bottleneck in bottlenecks:
            if bottleneck['type'] == 'slow_operation':
                # Estimate potential speedup from fixing slow operations
                avg_duration = bottleneck['metrics']['avg_duration']
                if avg_duration > 60:
                    potential_speedup *= 1.5  # Assume 50% improvement possible
                elif avg_duration > 30:
                    potential_speedup *= 1.3  # Assume 30% improvement possible
            
            elif bottleneck['type'] == 'high_failure_rate':
                # Estimate reliability improvement
                current_success_rate = bottleneck['metrics']['success_rate']
                potential_improvement = min(95 - current_success_rate, 50)  # Cap at 50% improvement
                potential_reliability_improvement = max(potential_reliability_improvement, potential_improvement)
            
            elif bottleneck['type'] == 'high_memory_usage':
                # Estimate memory savings
                avg_memory = bottleneck['metrics']['avg_memory_delta_mb']
                potential_savings = min(avg_memory * 0.5, 500)  # Assume up to 50% savings, cap at 500MB
                potential_memory_savings = max(potential_memory_savings, potential_savings)
            
            elif bottleneck['type'] == 'poor_cache_performance':
                # Cache improvements can provide significant speedup
                hit_rate = bottleneck['metrics']['hit_rate']
                if hit_rate < 30:
                    potential_speedup *= 2.0  # Significant improvement possible
                elif hit_rate < 60:
                    potential_speedup *= 1.5  # Moderate improvement possible
        
        return {
            'estimated_speedup_factor': min(potential_speedup, 3.0),  # Cap at 3x speedup
            'estimated_reliability_improvement_percent': potential_reliability_improvement,
            'estimated_memory_savings_mb': potential_memory_savings,
            'overall_potential': self._categorize_optimization_potential(potential_speedup, potential_reliability_improvement, potential_memory_savings)
        }
    
    def _categorize_optimization_potential(self, speedup: float, reliability: float, memory: float) -> str:
        """Categorize overall optimization potential."""
        if speedup > 2.0 or reliability > 30 or memory > 300:
            return 'high'
        elif speedup > 1.5 or reliability > 15 or memory > 150:
            return 'medium'
        elif speedup > 1.2 or reliability > 5 or memory > 50:
            return 'low'
        else:
            return 'minimal'
    
    def _identify_primary_concerns(self, bottlenecks: List[Dict[str, Any]]) -> List[str]:
        """Identify primary performance concerns."""
        concerns = []
        
        critical_bottlenecks = [b for b in bottlenecks if b['severity'] == 'critical']
        if critical_bottlenecks:
            concerns.append(f"{len(critical_bottlenecks)} critical performance issues requiring immediate attention")
        
        slow_operations = [b for b in bottlenecks if b['type'] == 'slow_operation']
        if len(slow_operations) > 2:
            concerns.append(f"Multiple slow operations detected ({len(slow_operations)} operations)")
        
        high_failure_rates = [b for b in bottlenecks if b['type'] == 'high_failure_rate']
        if high_failure_rates:
            concerns.append(f"Reliability issues detected in {len(high_failure_rates)} operations")
        
        cache_issues = [b for b in bottlenecks if b['type'] == 'poor_cache_performance']
        if cache_issues:
            concerns.append("Cache performance optimization needed")
        
        if not concerns:
            concerns.append("No major performance concerns identified")
        
        return concerns
    
    def _generate_next_steps(self, recommendations: List[Dict[str, Any]]) -> List[str]:
        """Generate actionable next steps."""
        next_steps = []
        
        # Get top 3 high-priority recommendations
        high_priority = [r for r in recommendations if r['priority'] in ['critical', 'high']][:3]
        
        for i, rec in enumerate(high_priority, 1):
            next_steps.append(f"{i}. {rec['description']}")
        
        # Add monitoring recommendation
        if len(recommendations) > 3:
            next_steps.append("4. Implement continuous performance monitoring to track improvement progress")
        
        # Add review recommendation
        next_steps.append(f"{len(next_steps) + 1}. Schedule performance review in 1-2 weeks to assess optimization impact")
        
        return next_steps