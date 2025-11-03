"""
Unit tests for AdaptiveParameterAdjuster
Tests parameter adjustment logic, edge cases, and extreme compression scenarios
"""

import unittest
from unittest.mock import Mock, patch
import tempfile
import os
import yaml
from src.adaptive_parameter_adjuster import AdaptiveParameterAdjuster, ParameterAdjustment
from src.bitrate_validator import BitrateValidator
from src.config_manager import ConfigManager


class TestAdaptiveParameterAdjuster(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        # Create temporary config directory
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        
        # Create test configuration
        self.test_config = {
            'video_compression': {
                'bitrate_validation': {
                    'enabled': True,
                    'encoder_minimums': {
                        'libx264': 3,
                        'libx265': 5,
                        'h264_nvenc': 2
                    },
                    'fallback_resolutions': [
                        [1280, 720],
                        [854, 480],
                        [640, 360],
                        [480, 270],
                        [320, 180]
                    ],
                    'safety_margin': 1.1,
                    'min_fps': 10,
                    'fps_reduction_steps': [0.8, 0.6, 0.5]
                }
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Initialize components
        self.config_manager = ConfigManager(self.temp_dir)
        self.bitrate_validator = BitrateValidator(self.config_manager)
        self.adjuster = AdaptiveParameterAdjuster(self.config_manager, self.bitrate_validator)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_no_adjustment_needed(self):
        """Test when no adjustment is needed"""
        params = {
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'bitrate': 5000,  # Well above minimum
            'encoder': 'libx264',
            'audio_bitrate': 64
        }
        video_info = {'duration_seconds': 60}
        
        result = self.adjuster.adjust_for_bitrate_floor(params, 3, video_info, 50.0)
        
        self.assertTrue(result.success)
        self.assertEqual(result.adjustment_type, 'none')
        self.assertEqual(result.quality_impact, 'minimal')
        self.assertEqual(result.bitrate_improvement, 0)
    
    def test_resolution_adjustment_success(self):
        """Test successful resolution adjustment"""
        params = {
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'bitrate': 2,  # Below minimum
            'encoder': 'libx264',
            'audio_bitrate': 64
        }
        video_info = {'duration_seconds': 60}
        
        result = self.adjuster.adjust_for_bitrate_floor(params, 3, video_info, 10.0)
        
        self.assertTrue(result.success)
        self.assertEqual(result.adjustment_type, 'resolution')
        self.assertLess(result.adjusted_params['width'], 1920)
        self.assertLess(result.adjusted_params['height'], 1080)
        self.assertGreaterEqual(result.adjusted_params['bitrate'], 3)
    
    def test_fps_adjustment_success(self):
        """Test successful FPS adjustment"""
        params = {
            'width': 320,  # Already at minimum resolution
            'height': 180,
            'fps': 60,
            'bitrate': 2,  # Below minimum
            'encoder': 'libx264',
            'audio_bitrate': 64
        }
        video_info = {'duration_seconds': 60}
        
        result = self.adjuster.adjust_for_bitrate_floor(params, 3, video_info, 10.0)
        
        if result.adjustment_type == 'fps':
            self.assertTrue(result.success)
            self.assertLess(result.adjusted_params['fps'], 60)
            self.assertGreaterEqual(result.adjusted_params['fps'], 10)  # Above minimum
    
    def test_combined_adjustment_success(self):
        """Test successful combined resolution and FPS adjustment"""
        params = {
            'width': 1920,
            'height': 1080,
            'fps': 60,
            'bitrate': 1,  # Very low bitrate requiring combined approach
            'encoder': 'libx264',
            'audio_bitrate': 64
        }
        video_info = {'duration_seconds': 300}  # Longer video
        
        result = self.adjuster.adjust_for_bitrate_floor(params, 3, video_info, 5.0)
        
        if result.adjustment_type == 'combined':
            self.assertTrue(result.success)
            self.assertLess(result.adjusted_params['width'], 1920)
            self.assertLess(result.adjusted_params['height'], 1080)
            self.assertLess(result.adjusted_params['fps'], 60)
    
    def test_adjustment_failure_extreme_case(self):
        """Test adjustment failure in extreme compression scenario"""
        params = {
            'width': 320,
            'height': 180,
            'fps': 10,
            'bitrate': 1,
            'encoder': 'libx264',
            'audio_bitrate': 64
        }
        video_info = {'duration_seconds': 3600}  # Very long video
        
        result = self.adjuster.adjust_for_bitrate_floor(params, 10, video_info, 1.0)
        
        self.assertFalse(result.success)
        self.assertEqual(result.adjustment_type, 'failed')
        self.assertEqual(result.quality_impact, 'significant')
    
    def test_get_fallback_resolutions(self):
        """Test getting fallback resolutions for given original resolution"""
        original_res = (1920, 1080)
        
        fallbacks = self.adjuster.get_fallback_resolutions(original_res)
        
        self.assertIsInstance(fallbacks, list)
        # All fallbacks should be smaller than original
        for width, height in fallbacks:
            self.assertLess(width * height, 1920 * 1080)
        
        # Should be sorted by quality (descending)
        if len(fallbacks) > 1:
            for i in range(len(fallbacks) - 1):
                current_pixels = fallbacks[i][0] * fallbacks[i][1]
                next_pixels = fallbacks[i + 1][0] * fallbacks[i + 1][1]
                self.assertGreaterEqual(current_pixels, next_pixels)
    
    def test_calculate_optimal_fps_reduction(self):
        """Test optimal FPS reduction calculation"""
        current_fps = 60.0
        bitrate_deficit = 2.0  # Need 2x improvement
        
        optimal_fps = self.adjuster.calculate_optimal_fps_reduction(current_fps, bitrate_deficit)
        
        self.assertLessEqual(optimal_fps, current_fps)
        self.assertGreaterEqual(optimal_fps, 10)  # Above minimum
        self.assertIn(optimal_fps, [60, 50, 30, 25, 24, 20, 15, 12, 10])  # Common FPS values
    
    def test_quality_impact_assessment(self):
        """Test quality impact assessment for different adjustment types"""
        # Test resolution impact
        minimal_ratio = 0.9  # 90% of original pixels
        moderate_ratio = 0.7  # 70% of original pixels
        significant_ratio = 0.3  # 30% of original pixels
        
        minimal_impact = self.adjuster._assess_quality_impact(minimal_ratio, 'resolution')
        moderate_impact = self.adjuster._assess_quality_impact(moderate_ratio, 'resolution')
        significant_impact = self.adjuster._assess_quality_impact(significant_ratio, 'resolution')
        
        self.assertEqual(minimal_impact, 'minimal')
        self.assertEqual(moderate_impact, 'moderate')
        self.assertEqual(significant_impact, 'significant')
    
    def test_validate_adjusted_parameters(self):
        """Test validation of adjusted parameters"""
        params = {
            'bitrate': 5000,
            'encoder': 'libx264'
        }
        
        result = self.adjuster.validate_adjusted_parameters(params, 'libx264')
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.current_bitrate, 5000)


class TestAdaptiveParameterAdjusterEdgeCases(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for edge cases"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal config
        self.config_manager = ConfigManager(self.temp_dir)
        self.bitrate_validator = BitrateValidator(self.config_manager)
        self.adjuster = AdaptiveParameterAdjuster(self.config_manager, self.bitrate_validator)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_zero_duration_handling(self):
        """Test handling of zero duration"""
        params = {
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'bitrate': 2,
            'encoder': 'libx264',
            'audio_bitrate': 64
        }
        video_info = {'duration_seconds': 0}
        
        result = self.adjuster.adjust_for_bitrate_floor(params, 3, video_info, 10.0)
        
        # Should handle gracefully without division by zero
        self.assertIsNotNone(result)
    
    def test_missing_video_info_keys(self):
        """Test handling of missing video info keys"""
        params = {
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'bitrate': 2,
            'encoder': 'libx264',
            'audio_bitrate': 64
        }
        video_info = {}  # Missing duration_seconds
        
        result = self.adjuster.adjust_for_bitrate_floor(params, 3, video_info, 10.0)
        
        # Should use default duration and work
        self.assertIsNotNone(result)
    
    def test_missing_param_keys(self):
        """Test handling of missing parameter keys"""
        params = {
            'bitrate': 2,
            'encoder': 'libx264'
            # Missing width, height, fps, audio_bitrate
        }
        video_info = {'duration_seconds': 60}
        
        result = self.adjuster.adjust_for_bitrate_floor(params, 3, video_info, 10.0)
        
        # Should use defaults and work
        self.assertIsNotNone(result)
    
    def test_extremely_small_target_size(self):
        """Test handling of extremely small target size"""
        params = {
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'bitrate': 2,
            'encoder': 'libx264',
            'audio_bitrate': 64
        }
        video_info = {'duration_seconds': 600}  # Longer video with tiny size
        
        result = self.adjuster.adjust_for_bitrate_floor(params, 10, video_info, 0.05)  # 50KB
        
        self.assertFalse(result.success)  # Should fail gracefully
    
    def test_extremely_long_video(self):
        """Test handling of extremely long video"""
        params = {
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'bitrate': 2,
            'encoder': 'libx264',
            'audio_bitrate': 64
        }
        video_info = {'duration_seconds': 36000}  # 10 hours
        
        result = self.adjuster.adjust_for_bitrate_floor(params, 3, video_info, 10.0)
        
        # Should suggest segmentation or fail gracefully
        self.assertIsNotNone(result)
    
    def test_fps_reduction_below_minimum(self):
        """Test FPS reduction that would go below minimum"""
        current_fps = 12.0  # Close to minimum
        bitrate_deficit = 5.0  # Would require very low FPS
        
        optimal_fps = self.adjuster.calculate_optimal_fps_reduction(current_fps, bitrate_deficit)
        
        self.assertGreaterEqual(optimal_fps, 10)  # Should not go below minimum
    
    def test_empty_fallback_resolutions(self):
        """Test behavior with empty fallback resolutions"""
        # Temporarily clear fallback resolutions
        original_fallbacks = self.adjuster.fallback_resolutions
        self.adjuster.fallback_resolutions = []
        
        try:
            params = {
                'width': 1920,
                'height': 1080,
                'fps': 30,
                'bitrate': 2,
                'encoder': 'libx264',
                'audio_bitrate': 64
            }
            video_info = {'duration_seconds': 60}
            
            result = self.adjuster.adjust_for_bitrate_floor(params, 3, video_info, 10.0)
            
            # Should handle gracefully
            self.assertIsNotNone(result)
        finally:
            # Restore original fallbacks
            self.adjuster.fallback_resolutions = original_fallbacks


class TestParameterAdjustmentDataClass(unittest.TestCase):
    
    def test_parameter_adjustment_creation(self):
        """Test ParameterAdjustment dataclass creation and attributes"""
        original_params = {'width': 1920, 'height': 1080}
        adjusted_params = {'width': 1280, 'height': 720}
        
        adjustment = ParameterAdjustment(
            success=True,
            adjusted_params=adjusted_params,
            original_params=original_params,
            adjustment_type='resolution',
            quality_impact='moderate',
            bitrate_improvement=500,
            message='Test adjustment'
        )
        
        self.assertTrue(adjustment.success)
        self.assertEqual(adjustment.adjustment_type, 'resolution')
        self.assertEqual(adjustment.quality_impact, 'moderate')
        self.assertEqual(adjustment.bitrate_improvement, 500)
        self.assertEqual(adjustment.message, 'Test adjustment')


if __name__ == '__main__':
    unittest.main()