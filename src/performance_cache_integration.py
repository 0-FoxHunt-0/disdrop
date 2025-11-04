"""
Integration module for the performance caching system.

This module provides a unified interface for all caching functionality,
integrating video fingerprinting, result caching, and cache management.
"""

from typing import Optional, Dict, Any, Tuple, List
try:
    from .performance_cache import PerformanceCache, QualityResult, CompressionResult
    from .video_fingerprinter import VideoFingerprinter, CompressionParams, CacheKeyGenerator
    from .cache_manager import CacheManager, CacheMetrics
    from .logger_setup import get_logger
except ImportError:
    # Fallback for direct execution
    from performance_cache import PerformanceCache, QualityResult, CompressionResult
    from video_fingerprinter import VideoFingerprinter, CompressionParams, CacheKeyGenerator
    from cache_manager import CacheManager, CacheMetrics
    from logger_setup import get_logger

logger = get_logger(__name__)


class PerformanceCacheSystem:
    """
    Unified performance caching system that integrates all caching components.
    
    This class provides a high-level interface for caching video compression
    and quality evaluation results with automatic optimization and management.
    """
    
    def __init__(self, cache_dir: Optional[str] = None, 
                 max_cache_size_mb: int = 500,
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize the performance caching system.
        
        Args:
            cache_dir: Directory for cache storage
            max_cache_size_mb: Maximum cache size in megabytes
            config: Cache configuration dictionary
        """
        # Initialize core components
        self.cache = PerformanceCache(cache_dir, max_cache_size_mb)
        self.fingerprinter = VideoFingerprinter()
        self.key_generator = CacheKeyGenerator(self.fingerprinter)
        self.manager = CacheManager(self.cache, cache_dir)
        
        # Apply custom configuration if provided
        if config:
            self.manager.update_configuration(config)
        
        logger.info("Performance cache system initialized")
    
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
            result, response_time = self.manager.get_cached_result_with_metrics(video_path, params)
            
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
            self.manager.cache_result_with_optimization(video_path, params, result)
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
            performance_metrics = self.manager.get_performance_metrics()
            
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
    
    def export_performance_report(self, output_path: str) -> None:
        """
        Export comprehensive performance report.
        
        Args:
            output_path: Path to save the report
        """
        try:
            self.manager.export_performance_report(output_path)
            
        except Exception as e:
            logger.error(f"Error exporting performance report: {e}")
    
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
            
            self.manager.update_configuration(warming_config)
            logger.info(f"Cache warming configured: enabled={enabled}")
            
        except Exception as e:
            logger.error(f"Error configuring cache warming: {e}")
    
    def optimize_cache_performance(self) -> None:
        """Manually trigger cache performance optimization."""
        try:
            self.manager._check_and_optimize_cache()
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
    
    def shutdown(self) -> None:
        """Shutdown the caching system and cleanup resources."""
        try:
            self.manager.shutdown()
            logger.info("Performance cache system shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during cache system shutdown: {e}")


# Convenience function for easy integration
def create_performance_cache_system(cache_dir: Optional[str] = None,
                                   max_size_mb: int = 500,
                                   enable_warming: bool = False) -> PerformanceCacheSystem:
    """
    Create a performance cache system with sensible defaults.
    
    Args:
        cache_dir: Directory for cache storage
        max_size_mb: Maximum cache size in megabytes
        enable_warming: Whether to enable cache warming
        
    Returns:
        Configured PerformanceCacheSystem instance
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
    
    return PerformanceCacheSystem(cache_dir, max_size_mb, config)