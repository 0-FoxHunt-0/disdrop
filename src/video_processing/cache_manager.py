"""
Cache management and optimization system for performance caching.

This module provides advanced cache management features including automatic size management,
performance monitoring, cache warming, and persistence across application restarts.
"""

import json
import os
import threading
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable
try:
    import schedule
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False
    # Create a mock schedule for testing
    class MockSchedule:
        def every(self, interval=None):
            return self
        def hours(self):
            return self
        def day(self):
            return self
        def at(self, time):
            return self
        def do(self, func):
            return self
        def run_pending(self):
            pass
    schedule = MockSchedule()
try:
    from .performance_cache import PerformanceCache, QualityResult, CompressionResult
    from .video_fingerprinter import CompressionParams, VideoFingerprinter, CacheKeyGenerator
    from ..logger_setup import get_logger
except ImportError:
    # Fallback for direct execution
    from performance_cache import PerformanceCache, QualityResult, CompressionResult
    from video_fingerprinter import CompressionParams, VideoFingerprinter, CacheKeyGenerator
    from logger_setup import get_logger

logger = get_logger(__name__)


@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hit_rate: float
    miss_rate: float
    total_lookups: int
    cache_hits: int
    cache_misses: int
    average_response_time: float
    cache_size_mb: float
    entry_count: int
    last_cleanup: float
    warming_efficiency: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class CacheWarmingStrategy:
    """Configuration for cache warming strategies."""
    enabled: bool
    common_video_patterns: List[str]
    common_compression_params: List[Dict[str, Any]]
    warming_schedule: str  # cron-like schedule
    max_warming_time_minutes: int
    priority_content_types: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


class CacheManager:
    """Advanced cache management and optimization system."""
    
    def __init__(self, cache: Optional[PerformanceCache] = None, 
                 config_dir: Optional[str] = None,
                 max_cache_size_mb: int = 500,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the cache manager.
        
        Args:
            cache: Performance cache instance (creates new one if None)
            config_dir: Configuration directory path
            max_cache_size_mb: Maximum cache size in megabytes (only used if cache is None)
            config: Cache configuration dictionary
        """
        # Initialize cache if not provided
        if cache is None:
            cache_dir = str(config_dir) if config_dir else None
            self.cache = PerformanceCache(cache_dir, max_cache_size_mb)
        else:
            self.cache = cache
        
        # Initialize video fingerprinting components
        self.fingerprinter = VideoFingerprinter()
        self.key_generator = CacheKeyGenerator(self.fingerprinter)
        
        self.config_dir = Path(config_dir) if config_dir else Path.home() / '.disdrop' / 'cache'
        self.config_dir.mkdir(parents=True, exist_ok=True)
        
        self.metrics_file = self.config_dir / 'cache_metrics.json'
        self.config_file = self.config_dir / 'cache_config.json'
        
        # Performance monitoring
        self._metrics_history: List[CacheMetrics] = []
        self._response_times: List[float] = []
        self._lock = threading.RLock()
        
        # Background tasks
        self._cleanup_scheduler = None
        self._warming_scheduler = None
        self._monitoring_active = False
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Apply custom configuration if provided
        if config:
            self._deep_merge(self.config, config)
            self.update_configuration(config)
        
        # Initialize monitoring
        self._start_monitoring()
        
        logger.info("Cache manager initialized")
    
    def _load_configuration(self) -> Dict[str, Any]:
        """Load cache management configuration."""
        default_config = {
            'auto_cleanup': {
                'enabled': True,
                'interval_hours': 6,
                'max_age_hours': 24,
                'size_threshold_percent': 90
            },
            'warming': {
                'enabled': False,
                'common_video_patterns': ['*.mp4', '*.avi', '*.mov'],
                'common_compression_params': [
                    {'codec': 'h264', 'preset': 'medium', 'crf': 23},
                    {'codec': 'h265', 'preset': 'medium', 'crf': 28}
                ],
                'warming_schedule': '0 2 * * *',  # Daily at 2 AM
                'max_warming_time_minutes': 30,
                'priority_content_types': ['animation', 'live_action']
            },
            'monitoring': {
                'enabled': True,
                'metrics_retention_days': 30,
                'performance_alert_threshold': 0.5  # Hit rate below 50%
            },
            'optimization': {
                'auto_size_management': True,
                'target_hit_rate': 0.7,
                'eviction_strategy': 'lru',  # lru, lfu, fifo
                'compression_level': 'medium'  # low, medium, high
            }
        }
        
        try:
            if self.config_file.exists():
                with open(self.config_file, 'r') as f:
                    user_config = json.load(f)
                    # Merge with defaults
                    self._deep_merge(default_config, user_config)
            
            return default_config
            
        except Exception as e:
            logger.error(f"Error loading cache configuration: {e}")
            return default_config
    
    def _deep_merge(self, base: Dict, update: Dict) -> None:
        """Deep merge configuration dictionaries."""
        for key, value in update.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _start_monitoring(self) -> None:
        """Start background monitoring and maintenance tasks."""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        
        # Schedule automatic cleanup
        if self.config['auto_cleanup']['enabled'] and SCHEDULE_AVAILABLE:
            interval = self.config['auto_cleanup']['interval_hours']
            schedule.every(interval).hours.do(self._auto_cleanup)
        
        # Schedule cache warming
        if self.config['warming']['enabled'] and SCHEDULE_AVAILABLE:
            warming_schedule = self.config['warming']['warming_schedule']
            # Parse cron-like schedule (simplified)
            if warming_schedule == '0 2 * * *':  # Daily at 2 AM
                schedule.every().day.at("02:00").do(self._warm_cache)
        
        # Start background thread for scheduled tasks
        self._start_scheduler_thread()
        
        logger.info("Cache monitoring and maintenance started")
    
    def _start_scheduler_thread(self) -> None:
        """Start background thread for scheduled tasks."""
        def run_scheduler():
            while self._monitoring_active:
                try:
                    schedule.run_pending()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Error in scheduler thread: {e}")
        
        scheduler_thread = threading.Thread(target=run_scheduler, daemon=True)
        scheduler_thread.start()
    
    def get_cached_result_with_metrics(self, video_path: str, params: CompressionParams) -> Tuple[Optional[QualityResult], float]:
        """
        Get cached result while tracking performance metrics.
        
        Args:
            video_path: Path to the video file
            params: Compression parameters
            
        Returns:
            Tuple of (result, response_time)
        """
        start_time = time.time()
        
        try:
            result = self.cache.get_cached_quality_result(video_path, params)
            response_time = time.time() - start_time
            
            # Track response time
            with self._lock:
                self._response_times.append(response_time)
                # Keep only recent response times
                if len(self._response_times) > 1000:
                    self._response_times = self._response_times[-500:]
            
            return result, response_time
            
        except Exception as e:
            logger.error(f"Error getting cached result with metrics: {e}")
            return None, time.time() - start_time
    
    def cache_result_with_optimization(self, video_path: str, params: CompressionParams, 
                                     result: QualityResult, ttl_hours: Optional[int] = None) -> None:
        """
        Cache result with automatic optimization.
        
        Args:
            video_path: Path to the video file
            params: Compression parameters
            result: Quality result to cache
            ttl_hours: Time to live in hours (auto-calculated if None)
        """
        try:
            # Auto-calculate TTL based on result quality and confidence
            if ttl_hours is None:
                ttl_hours = self._calculate_optimal_ttl(result)
            
            self.cache.cache_quality_result(video_path, params, result, ttl_hours)
            
            # Check if cache optimization is needed
            if self.config['optimization']['auto_size_management']:
                self._check_and_optimize_cache()
            
        except Exception as e:
            logger.error(f"Error caching result with optimization: {e}")
    
    def _calculate_optimal_ttl(self, result: QualityResult) -> int:
        """Calculate optimal TTL based on result characteristics."""
        base_ttl = 24  # 24 hours default
        
        # Adjust based on confidence
        if result.confidence > 0.9:
            ttl_multiplier = 2.0  # High confidence results last longer
        elif result.confidence > 0.7:
            ttl_multiplier = 1.5
        elif result.confidence > 0.5:
            ttl_multiplier = 1.0
        else:
            ttl_multiplier = 0.5  # Low confidence results expire faster
        
        # Adjust based on evaluation method
        if result.evaluation_method == 'full':
            ttl_multiplier *= 1.5  # Full evaluations are more valuable
        elif result.evaluation_method == 'predicted':
            ttl_multiplier *= 0.7  # Predictions are less reliable
        
        return int(base_ttl * ttl_multiplier)
    
    def _check_and_optimize_cache(self) -> None:
        """Check cache performance and optimize if needed."""
        try:
            stats = self.cache.get_cache_stats()
            hit_rate = stats.get('hit_rate_percent', 0) / 100.0
            target_hit_rate = self.config['optimization']['target_hit_rate']
            
            # If hit rate is below target, consider optimization
            if hit_rate < target_hit_rate:
                logger.info(f"Cache hit rate ({hit_rate:.2%}) below target ({target_hit_rate:.2%}), optimizing...")
                self._optimize_cache_performance()
            
            # Check size limits
            size_percent = (stats.get('total_size_mb', 0) / stats.get('max_size_mb', 1)) * 100
            threshold = self.config['auto_cleanup']['size_threshold_percent']
            
            if size_percent > threshold:
                logger.info(f"Cache size ({size_percent:.1f}%) above threshold ({threshold}%), cleaning up...")
                self._auto_cleanup()
            
        except Exception as e:
            logger.error(f"Error checking and optimizing cache: {e}")
    
    def _optimize_cache_performance(self) -> None:
        """Optimize cache performance based on usage patterns."""
        try:
            # Analyze cache usage patterns
            stats = self.cache.get_cache_stats()
            
            # If hit rate is low, consider warming cache with common patterns
            if self.config['warming']['enabled']:
                self._warm_cache_selective()
            
            # Adjust eviction strategy if needed
            eviction_strategy = self.config['optimization']['eviction_strategy']
            if eviction_strategy == 'lfu':
                # Implement LFU-based optimization
                self._optimize_lfu()
            
        except Exception as e:
            logger.error(f"Error optimizing cache performance: {e}")
    
    def _auto_cleanup(self) -> None:
        """Perform automatic cache cleanup."""
        try:
            max_age_hours = self.config['auto_cleanup']['max_age_hours']
            removed_count = self.cache.cleanup_expired_entries(max_age_hours)
            
            logger.info(f"Auto cleanup removed {removed_count} expired entries")
            
            # Update metrics
            self._record_cleanup_metrics()
            
        except Exception as e:
            logger.error(f"Error in auto cleanup: {e}")
    
    def _warm_cache(self) -> None:
        """Warm cache with common video patterns."""
        if not self.config['warming']['enabled']:
            return
        
        try:
            logger.info("Starting cache warming process")
            start_time = time.time()
            max_time = self.config['warming']['max_warming_time_minutes'] * 60
            
            warming_count = 0
            patterns = self.config['warming']['common_video_patterns']
            params_list = self.config['warming']['common_compression_params']
            
            # Find videos matching common patterns
            for pattern in patterns:
                if time.time() - start_time > max_time:
                    break
                
                # This would need integration with file discovery system
                # For now, we'll simulate the warming process
                warming_count += self._warm_pattern(pattern, params_list)
            
            elapsed_time = time.time() - start_time
            logger.info(f"Cache warming completed: {warming_count} entries warmed in {elapsed_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Error in cache warming: {e}")
    
    def _warm_cache_selective(self) -> None:
        """Selectively warm cache based on recent miss patterns."""
        # This would analyze recent cache misses and pre-compute likely future requests
        # Implementation would depend on access pattern analysis
        pass
    
    def _warm_pattern(self, pattern: str, params_list: List[Dict[str, Any]]) -> int:
        """Warm cache for a specific file pattern."""
        # This would find files matching the pattern and pre-compute results
        # Implementation would depend on file system integration
        return 0
    
    def _optimize_lfu(self) -> None:
        """Optimize cache using Least Frequently Used strategy."""
        # This would implement LFU-based cache optimization
        # Current implementation uses LRU in the base cache
        pass
    
    def get_performance_metrics(self) -> CacheMetrics:
        """
        Get current cache performance metrics.
        
        Returns:
            Current cache performance metrics
        """
        try:
            stats = self.cache.get_cache_stats()
            
            with self._lock:
                avg_response_time = (
                    sum(self._response_times) / len(self._response_times)
                    if self._response_times else 0.0
                )
            
            metrics = CacheMetrics(
                hit_rate=stats.get('hit_rate_percent', 0) / 100.0,
                miss_rate=1.0 - (stats.get('hit_rate_percent', 0) / 100.0),
                total_lookups=stats.get('total_lookups', 0),
                cache_hits=stats.get('cache_hits', 0),
                cache_misses=stats.get('cache_misses', 0),
                average_response_time=avg_response_time,
                cache_size_mb=stats.get('total_size_mb', 0),
                entry_count=stats.get('total_entries', 0),
                last_cleanup=time.time(),  # Would track actual cleanup time
                warming_efficiency=0.0  # Would calculate based on warming success rate
            )
            
            # Store metrics history
            with self._lock:
                self._metrics_history.append(metrics)
                # Keep only recent metrics
                if len(self._metrics_history) > 1000:
                    self._metrics_history = self._metrics_history[-500:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            return CacheMetrics(0, 0, 0, 0, 0, 0, 0, 0, 0, 0)
    
    def _record_cleanup_metrics(self) -> None:
        """Record cleanup operation metrics."""
        try:
            metrics = self.get_performance_metrics()
            
            # Save metrics to file
            metrics_data = {
                'timestamp': time.time(),
                'metrics': metrics.to_dict()
            }
            
            # Append to metrics history file
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    history = json.load(f)
            else:
                history = []
            
            history.append(metrics_data)
            
            # Keep only recent history
            retention_days = self.config['monitoring']['metrics_retention_days']
            cutoff_time = time.time() - (retention_days * 24 * 3600)
            history = [h for h in history if h['timestamp'] > cutoff_time]
            
            with open(self.metrics_file, 'w') as f:
                json.dump(history, f, indent=2)
            
        except Exception as e:
            logger.error(f"Error recording cleanup metrics: {e}")
    
    def get_metrics_history(self, days: int = 7) -> List[CacheMetrics]:
        """
        Get historical cache metrics.
        
        Args:
            days: Number of days of history to retrieve
            
        Returns:
            List of historical cache metrics
        """
        try:
            if not self.metrics_file.exists():
                return []
            
            with open(self.metrics_file, 'r') as f:
                history = json.load(f)
            
            # Filter by time range
            cutoff_time = time.time() - (days * 24 * 3600)
            recent_history = [h for h in history if h['timestamp'] > cutoff_time]
            
            # Convert to CacheMetrics objects
            metrics_list = []
            for entry in recent_history:
                metrics_data = entry['metrics']
                metrics = CacheMetrics(**metrics_data)
                metrics_list.append(metrics)
            
            return metrics_list
            
        except Exception as e:
            logger.error(f"Error getting metrics history: {e}")
            return []
    
    def export_performance_report(self, output_path: str) -> None:
        """
        Export comprehensive performance report.
        
        Args:
            output_path: Path to save the report
        """
        try:
            current_metrics = self.get_performance_metrics()
            historical_metrics = self.get_metrics_history(30)  # Last 30 days
            cache_stats = self.cache.get_cache_stats()
            
            report = {
                'generated_at': datetime.now().isoformat(),
                'current_metrics': current_metrics.to_dict(),
                'cache_stats': cache_stats,
                'configuration': self.config,
                'historical_data': {
                    'metrics_count': len(historical_metrics),
                    'date_range': {
                        'start': datetime.fromtimestamp(
                            min(h.last_cleanup for h in historical_metrics)
                        ).isoformat() if historical_metrics else None,
                        'end': datetime.now().isoformat()
                    }
                },
                'performance_analysis': self._analyze_performance_trends(historical_metrics)
            }
            
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2)
            
            logger.info(f"Performance report exported to {output_path}")
            
        except Exception as e:
            logger.error(f"Error exporting performance report: {e}")
    
    def _analyze_performance_trends(self, metrics_history: List[CacheMetrics]) -> Dict[str, Any]:
        """Analyze performance trends from historical data."""
        if not metrics_history:
            return {}
        
        try:
            hit_rates = [m.hit_rate for m in metrics_history]
            response_times = [m.average_response_time for m in metrics_history]
            cache_sizes = [m.cache_size_mb for m in metrics_history]
            
            analysis = {
                'hit_rate_trend': {
                    'average': sum(hit_rates) / len(hit_rates),
                    'min': min(hit_rates),
                    'max': max(hit_rates),
                    'trend': 'improving' if hit_rates[-1] > hit_rates[0] else 'declining'
                },
                'response_time_trend': {
                    'average': sum(response_times) / len(response_times),
                    'min': min(response_times),
                    'max': max(response_times)
                },
                'cache_size_trend': {
                    'average': sum(cache_sizes) / len(cache_sizes),
                    'min': min(cache_sizes),
                    'max': max(cache_sizes)
                }
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing performance trends: {e}")
            return {}
    
    def update_configuration(self, new_config: Dict[str, Any]) -> None:
        """
        Update cache management configuration.
        
        Args:
            new_config: New configuration dictionary
        """
        try:
            # Merge with existing configuration
            self._deep_merge(self.config, new_config)
            
            # Save to file
            with open(self.config_file, 'w') as f:
                json.dump(self.config, f, indent=2)
            
            logger.info("Cache configuration updated")
            
        except Exception as e:
            logger.error(f"Error updating configuration: {e}")
    
    def shutdown(self) -> None:
        """Shutdown cache manager and cleanup resources."""
        try:
            self._monitoring_active = False
            
            # Save final metrics
            self._record_cleanup_metrics()
            
            logger.info("Cache manager shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during cache manager shutdown: {e}")
    
    # Methods from PerformanceCacheSystem (unified interface)
    
    def get_quality_result(self, video_path: str, params: CompressionParams) -> Tuple[Optional[QualityResult], bool, float]:
        """
        Get quality result from cache or predict from similar videos.
        
        Args:
            video_path: Path to the video file
            params: Compression parameters
            
        Returns:
            Tuple of (quality_result, is_cache_hit, response_time)
        """
        try:
            # Try exact cache match first
            result, response_time = self.get_cached_result_with_metrics(video_path, params)
            
            if result is not None:
                logger.debug(f"Exact cache hit for {video_path}")
                return result, True, response_time
            
            # Try prediction from similar videos
            predicted_result = self.cache.predict_quality_from_cache(video_path, params)
            
            if predicted_result is not None:
                logger.debug(f"Quality predicted from similar videos for {video_path}")
                return predicted_result, False, response_time
            
            logger.debug(f"No cached or predicted result for {video_path}")
            return None, False, response_time
            
        except Exception as e:
            logger.error(f"Error getting quality result from cache: {e}")
            return None, False, 0.0
    
    def cache_quality_result(self, video_path: str, params: CompressionParams, 
                           result: QualityResult) -> None:
        """
        Cache quality result with automatic optimization.
        
        Args:
            video_path: Path to the video file
            params: Compression parameters
            result: Quality result to cache
        """
        try:
            self.cache_result_with_optimization(video_path, params, result)
            logger.debug(f"Cached quality result for {video_path}")
            
        except Exception as e:
            logger.error(f"Error caching quality result: {e}")
    
    def get_compression_result(self, video_path: str, params: CompressionParams) -> Optional[CompressionResult]:
        """
        Get compression result from cache.
        
        Args:
            video_path: Path to the video file
            params: Compression parameters
            
        Returns:
            Cached compression result if found
        """
        try:
            return self.cache.get_cached_compression_result(video_path, params)
            
        except Exception as e:
            logger.error(f"Error getting compression result from cache: {e}")
            return None
    
    def cache_compression_result(self, video_path: str, params: CompressionParams, 
                               result: CompressionResult) -> None:
        """
        Cache compression result.
        
        Args:
            video_path: Path to the video file
            params: Compression parameters
            result: Compression result to cache
        """
        try:
            self.cache.cache_compression_result(video_path, params, result)
            logger.debug(f"Cached compression result for {video_path}")
            
        except Exception as e:
            logger.error(f"Error caching compression result: {e}")
    
    def find_similar_results(self, video_path: str, similarity_threshold: float = 0.8) -> List[Tuple[float, CompressionResult]]:
        """
        Find cached results from similar videos.
        
        Args:
            video_path: Path to the video file
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of (similarity_score, result) tuples
        """
        try:
            return self.cache.get_similar_video_results(video_path, similarity_threshold)
            
        except Exception as e:
            logger.error(f"Error finding similar results: {e}")
            return []
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive cache statistics.
        
        Returns:
            Dictionary containing cache statistics and metrics
        """
        try:
            base_stats = self.cache.get_cache_stats()
            performance_metrics = self.get_performance_metrics()
            
            return {
                'cache_stats': base_stats,
                'performance_metrics': performance_metrics.to_dict(),
                'system_info': {
                    'cache_dir': str(self.cache.cache_dir),
                    'max_size_mb': self.cache.max_size_bytes / (1024 * 1024),
                    'database_path': str(self.cache.db_path)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting cache statistics: {e}")
            return {}
    
    def cleanup_cache(self, max_age_hours: int = 24) -> int:
        """
        Clean up expired cache entries.
        
        Args:
            max_age_hours: Maximum age for cache entries
            
        Returns:
            Number of entries removed
        """
        try:
            return self.cache.cleanup_expired_entries(max_age_hours)
            
        except Exception as e:
            logger.error(f"Error cleaning up cache: {e}")
            return 0
    
    def clear_cache(self) -> None:
        """Clear all cache entries."""
        try:
            self.cache.clear_cache()
            logger.info("Cache cleared successfully")
            
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    def configure_cache_warming(self, enabled: bool = True, 
                              video_patterns: Optional[List[str]] = None,
                              compression_params: Optional[List[Dict[str, Any]]] = None) -> None:
        """
        Configure cache warming settings.
        
        Args:
            enabled: Whether to enable cache warming
            video_patterns: File patterns for videos to warm
            compression_params: Common compression parameters to warm
        """
        try:
            warming_config = {
                'warming': {
                    'enabled': enabled
                }
            }
            
            if video_patterns:
                warming_config['warming']['common_video_patterns'] = video_patterns
            
            if compression_params:
                warming_config['warming']['common_compression_params'] = compression_params
            
            self.update_configuration(warming_config)
            logger.info(f"Cache warming configured: enabled={enabled}")
            
        except Exception as e:
            logger.error(f"Error configuring cache warming: {e}")
    
    def optimize_cache_performance(self) -> None:
        """Manually trigger cache performance optimization."""
        try:
            self._check_and_optimize_cache()
            logger.info("Cache performance optimization completed")
            
        except Exception as e:
            logger.error(f"Error optimizing cache performance: {e}")
    
    def get_video_fingerprint(self, video_path: str) -> Optional[str]:
        """
        Get video fingerprint for debugging or analysis.
        
        Args:
            video_path: Path to the video file
            
        Returns:
            Video content hash if successful
        """
        try:
            fingerprint = self.fingerprinter.generate_fingerprint(video_path)
            return fingerprint.content_hash
            
        except Exception as e:
            logger.error(f"Error getting video fingerprint: {e}")
            return None
    
    def calculate_video_similarity(self, video_path1: str, video_path2: str) -> float:
        """
        Calculate similarity between two videos.
        
        Args:
            video_path1: Path to first video
            video_path2: Path to second video
            
        Returns:
            Similarity score between 0.0 and 1.0
        """
        try:
            fp1 = self.fingerprinter.generate_fingerprint(video_path1)
            fp2 = self.fingerprinter.generate_fingerprint(video_path2)
            
            return self.fingerprinter.calculate_similarity(fp1.perceptual_hash, fp2.perceptual_hash)
            
        except Exception as e:
            logger.error(f"Error calculating video similarity: {e}")
            return 0.0


# Convenience function for easy integration (replaces create_performance_cache_system)
def create_cache_manager(cache_dir: Optional[str] = None,
                        max_size_mb: int = 500,
                        enable_warming: bool = False) -> CacheManager:
    """
    Create a cache manager with sensible defaults.
    
    Args:
        cache_dir: Directory for cache storage
        max_size_mb: Maximum cache size in megabytes
        enable_warming: Whether to enable cache warming
        
    Returns:
        Configured CacheManager instance
    """
    config = {
        'warming': {
            'enabled': enable_warming
        },
        'auto_cleanup': {
            'enabled': True,
            'interval_hours': 6
        },
        'optimization': {
            'auto_size_management': True,
            'target_hit_rate': 0.7
        }
    }
    
    return CacheManager(config_dir=cache_dir, max_cache_size_mb=max_size_mb, config=config)