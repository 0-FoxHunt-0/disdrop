"""
Integration tests for consolidated components.

Tests cache layer, quality prediction, and performance monitoring integration.
"""

import pytest
import os
import tempfile
import time
from pathlib import Path

try:
    from src.cache_manager import CacheManager, create_cache_manager
    from src.performance_cache import PerformanceCache, QualityResult, CompressionResult, CompressionParams
    from src.quality_predictor import QualityPredictor, PredictionStrategy, VideoCharacteristics
    from src.performance_monitor import PerformanceMonitor
    from src.config_manager import ConfigManager
except ImportError:
    from cache_manager import CacheManager, create_cache_manager
    from performance_cache import PerformanceCache, QualityResult, CompressionResult, CompressionParams
    from quality_predictor import QualityPredictor, PredictionStrategy, VideoCharacteristics
    from performance_monitor import PerformanceMonitor
    from config_manager import ConfigManager


class TestCacheManagerIntegration:
    """Test CacheManager unified interface."""
    
    def test_cache_manager_creation(self):
        """Test CacheManager creation with different configurations."""
        # Test with defaults
        cache_manager = create_cache_manager()
        assert cache_manager is not None
        
        # Test with custom config
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = create_cache_manager(
                cache_dir=tmpdir,
                max_size_mb=100,
                enable_warming=True
            )
            assert cache_manager.cache_dir == Path(tmpdir)
    
    def test_cache_manager_quality_result(self):
        """Test quality result caching and retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = create_cache_manager(cache_dir=tmpdir)
            
            # Create test params
            params = CompressionParams(
                codec='h264',
                bitrate=1000,
                resolution=(1920, 1080),
                fps=30.0,
                preset='medium',
                crf=23,
                additional_params={}
            )
            
            # Test caching quality result
            quality_result = QualityResult(
                vmaf_score=85.0,
                ssim_score=0.95,
                psnr_score=35.0,
                computation_time=10.0,
                evaluation_method='full',
                confidence=0.9,
                timestamp=time.time()
            )
            
            cache_manager.cache_quality_result('test_video.mp4', params, quality_result)
            
            # Test retrieval
            result, is_hit, response_time = cache_manager.get_quality_result('test_video.mp4', params)
            
            assert result is not None
            assert is_hit is True
            assert result.vmaf_score == 85.0
            assert result.ssim_score == 0.95
    
    def test_cache_manager_statistics(self):
        """Test cache statistics retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = create_cache_manager(cache_dir=tmpdir)
            
            stats = cache_manager.get_cache_statistics()
            
            assert 'cache_stats' in stats
            assert 'performance_metrics' in stats
            assert 'system_info' in stats
    
    def test_cache_manager_fingerprinting(self):
        """Test video fingerprinting functionality."""
        # Note: This test requires an actual video file
        # For now, test that methods exist and handle errors gracefully
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = create_cache_manager(cache_dir=tmpdir)
            
            # Test with non-existent file (should return None gracefully)
            fingerprint = cache_manager.get_video_fingerprint('nonexistent.mp4')
            # Should handle error gracefully
            assert fingerprint is None or isinstance(fingerprint, str)


class TestQualityPredictorIntegration:
    """Test QualityPredictor unified interface."""
    
    def test_quality_predictor_creation(self):
        """Test QualityPredictor creation."""
        predictor = QualityPredictor()
        assert predictor is not None
    
    def test_quality_predictor_characteristics_based(self):
        """Test characteristics-based prediction."""
        predictor = QualityPredictor()
        
        # Create test characteristics
        video_chars = VideoCharacteristics(
            duration=60.0,
            width=1920,
            height=1080,
            fps=30.0,
            bitrate=5000000,
            codec='h264',
            file_size_mb=50.0,
            complexity_score=0.5,
            motion_score=0.4,
            scene_changes=6
        )
        
        # Test prediction
        prediction = predictor.predict_quality(
            'test_video.mp4',
            strategy=PredictionStrategy.CHARACTERISTICS_BASED,
            video_characteristics=video_chars
        )
        
        assert prediction is not None
        assert prediction.predicted_vmaf is not None
        assert prediction.predicted_ssim is not None
        assert prediction.strategy == PredictionStrategy.CHARACTERISTICS_BASED
        assert 0 <= prediction.overall_confidence <= 1
    
    def test_quality_predictor_auto_strategy(self):
        """Test automatic strategy selection."""
        predictor = QualityPredictor()
        
        # Test auto strategy (should fall back to characteristics-based if no other data)
        prediction = predictor.predict_quality(
            'test_video.mp4',
            strategy=PredictionStrategy.AUTO
        )
        
        assert prediction is not None
        assert prediction.strategy in [
            PredictionStrategy.STATISTICAL,
            PredictionStrategy.FAST_ESTIMATION,
            PredictionStrategy.CHARACTERISTICS_BASED
        ]
    
    def test_quality_predictor_learning(self):
        """Test prediction learning from historical results."""
        predictor = QualityPredictor()
        
        # Create test characteristics
        video_chars = VideoCharacteristics(
            duration=60.0,
            width=1920,
            height=1080,
            fps=30.0,
            bitrate=5000000,
            codec='h264',
            file_size_mb=50.0,
            complexity_score=0.5,
            motion_score=0.4,
            scene_changes=6
        )
        
        # Record evaluation result
        predictor.record_evaluation_result(
            video_chars,
            actual_vmaf=85.0,
            actual_ssim=0.95,
            evaluation_time=45.0,
            evaluation_success=True
        )
        
        # Check accuracy statistics
        accuracy = predictor.get_prediction_accuracy()
        assert accuracy['total_predictions'] == 1


class TestCacheQualityPredictionIntegration:
    """Test integration between cache and quality prediction."""
    
    def test_cache_with_prediction(self):
        """Test cache integration with quality prediction."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = create_cache_manager(cache_dir=tmpdir)
            
            # Create test params
            params = CompressionParams(
                codec='h264',
                bitrate=1000,
                resolution=(1920, 1080),
                fps=30.0,
                preset='medium',
                crf=23,
                additional_params={}
            )
            
            # Test cache miss with prediction
            result, is_hit, response_time = cache_manager.get_quality_result('test_video.mp4', params)
            
            # Should return None or predicted result (not exact hit)
            # Prediction depends on similar videos in cache
            assert result is None or isinstance(result, QualityResult)


class TestPerformanceMonitoringIntegration:
    """Test performance monitoring integration."""
    
    def test_performance_monitor_with_cache(self):
        """Test performance monitor integration with cache."""
        config = ConfigManager()
        monitor = PerformanceMonitor(config)
        
        # Start session
        session_id = monitor.start_compression_session('test.mp4', 10.0, 'instagram')
        
        # Record cache operations
        monitor.record_cache_hit()
        monitor.record_cache_miss()
        
        # Get cache hit rate
        hit_rate = monitor.get_cache_hit_rate()
        assert 0 <= hit_rate <= 100
        
        # End session
        monitor.end_compression_session(True, 9.5, 85.0)
        
        # Get summary
        summary = monitor.get_performance_summary()
        assert 'cache_statistics' in summary


class TestFullIntegration:
    """Test full integration of cache, quality prediction, and performance monitoring."""
    
    def test_cache_predictor_monitor_integration(self):
        """Test full integration of all three components."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Initialize components
            config = ConfigManager()
            cache_manager = create_cache_manager(cache_dir=tmpdir)
            predictor = QualityPredictor(config)
            monitor = PerformanceMonitor(config)
            
            # Start monitoring session
            session_id = monitor.start_compression_session('test_video.mp4', 10.0, 'instagram')
            
            # Create test params
            params = CompressionParams(
                codec='h264',
                bitrate=1000,
                resolution=(1920, 1080),
                fps=30.0,
                preset='medium',
                crf=23,
                additional_params={}
            )
            
            # Test 1: Cache miss -> prediction -> cache result
            result, is_hit, response_time = cache_manager.get_quality_result('test_video.mp4', params)
            
            # Record cache operation
            if is_hit:
                monitor.record_cache_hit()
            else:
                monitor.record_cache_miss()
            
            # If cache miss, use prediction
            if result is None:
                prediction = predictor.predict_quality(
                    'test_video.mp4',
                    strategy=PredictionStrategy.CHARACTERISTICS_BASED
                )
                assert prediction is not None
                assert prediction.predicted_vmaf is not None
                
                # Convert prediction to quality result and cache it
                quality_result = QualityResult(
                    vmaf_score=prediction.predicted_vmaf,
                    ssim_score=prediction.predicted_ssim,
                    psnr_score=None,
                    computation_time=prediction.estimated_evaluation_time,
                    evaluation_method='predicted',
                    confidence=prediction.overall_confidence,
                    timestamp=time.time()
                )
                cache_manager.cache_quality_result('test_video.mp4', params, quality_result)
            
            # Test 2: Cache hit (should be faster)
            result2, is_hit2, response_time2 = cache_manager.get_quality_result('test_video.mp4', params)
            if is_hit2:
                monitor.record_cache_hit()
                assert result2 is not None
            
            # End session
            monitor.end_compression_session(True, 9.5, 85.0)
            
            # Verify statistics
            summary = monitor.get_performance_summary()
            assert 'cache_statistics' in summary
            
            cache_stats = cache_manager.get_cache_statistics()
            assert 'cache_stats' in cache_stats
            
            predictor_accuracy = predictor.get_prediction_accuracy()
            assert 'total_predictions' in predictor_accuracy
    
    def test_quality_prediction_with_cache_similarity(self):
        """Test quality prediction using cache similarity."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_manager = create_cache_manager(cache_dir=tmpdir)
            predictor = QualityPredictor()
            
            # Create test params
            params = CompressionParams(
                codec='h264',
                bitrate=1000,
                resolution=(1920, 1080),
                fps=30.0,
                preset='medium',
                crf=23,
                additional_params={}
            )
            
            # Cache a similar result first
            quality_result = QualityResult(
                vmaf_score=85.0,
                ssim_score=0.95,
                psnr_score=35.0,
                computation_time=10.0,
                evaluation_method='full',
                confidence=0.9,
                timestamp=time.time()
            )
            cache_manager.cache_quality_result('similar_video.mp4', params, quality_result)
            
            # Test prediction from similar videos
            result, is_hit, response_time = cache_manager.get_quality_result('test_video.mp4', params)
            
            # Should either hit or predict from similar videos
            assert result is None or isinstance(result, QualityResult)
    
    def test_performance_monitoring_with_predictions(self):
        """Test performance monitoring tracks prediction usage."""
        config = ConfigManager()
        predictor = QualityPredictor(config)
        monitor = PerformanceMonitor(config)
        
        # Start session
        session_id = monitor.start_compression_session('test.mp4', 10.0, 'instagram')
        
        # Make prediction with performance monitoring
        with monitor.measure_operation('quality_prediction',
                                      strategy='auto',
                                      confidence=0.85):
            prediction = predictor.predict_quality(
                'test_video.mp4',
                strategy=PredictionStrategy.AUTO
            )
        
        # End session
        monitor.end_compression_session(True, 9.5, 85.0)
        
        # Verify metrics
        summary = monitor.get_performance_summary()
        assert 'cache_statistics' in summary
        # Verify operations were recorded
        if monitor._current_session is None:
            # Check recent sessions
            assert len(monitor._recent_sessions) > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

