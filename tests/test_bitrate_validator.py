"""
Unit tests for BitrateValidator
Tests validation logic, encoder scenarios, and configuration handling
"""

import unittest
from unittest.mock import Mock, patch
import tempfile
import os
import yaml
from src.bitrate_validator import BitrateValidator, ValidationResult, AdjustmentPlan, BitrateValidationError
from src.config_manager import ConfigManager


class TestBitrateValidator(unittest.TestCase):
    
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
                        [640, 360],
                        [480, 270],
                        [320, 180]
                    ],
                    'safety_margin': 1.1
                }
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Initialize config manager and validator
        self.config_manager = ConfigManager(self.temp_dir)
        self.validator = BitrateValidator(self.config_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_encoder_minimums_loading(self):
        """Test that encoder minimums are loaded correctly from config"""
        self.assertEqual(self.validator.encoder_minimums['libx264'], 3)
        self.assertEqual(self.validator.encoder_minimums['libx265'], 5)
        self.assertEqual(self.validator.encoder_minimums['h264_nvenc'], 2)
    
    def test_validate_bitrate_success(self):
        """Test successful bitrate validation"""
        result = self.validator.validate_bitrate(5000, 'libx264')
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.current_bitrate, 5000)
        self.assertEqual(result.minimum_required, 3)
        self.assertFalse(result.adjustment_needed)
        self.assertEqual(result.severity, 'info')
    
    def test_validate_bitrate_warning(self):
        """Test bitrate validation with warning level failure"""
        result = self.validator.validate_bitrate(2, 'libx264')
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.current_bitrate, 2)
        self.assertEqual(result.minimum_required, 3)
        self.assertTrue(result.adjustment_needed)
        self.assertEqual(result.severity, 'warning')
    
    def test_validate_bitrate_critical(self):
        """Test bitrate validation with critical level failure"""
        result = self.validator.validate_bitrate(1, 'libx264')
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.current_bitrate, 1)
        self.assertEqual(result.minimum_required, 3)
        self.assertTrue(result.adjustment_needed)
        self.assertEqual(result.severity, 'critical')
    
    def test_validate_bitrate_unknown_encoder(self):
        """Test bitrate validation with unknown encoder uses default"""
        result = self.validator.validate_bitrate(2, 'unknown_encoder')
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.minimum_required, 3)  # Default minimum
    
    def test_suggest_adjustments_resolution_reduction(self):
        """Test adjustment suggestions prioritize resolution reduction"""
        params = {
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'bitrate': 1000,
            'encoder': 'libx264',
            'audio_bitrate': 64
        }
        video_info = {'duration_seconds': 60}
        
        plan = self.validator.suggest_adjustments(params, 10.0, video_info)
        
        self.assertEqual(plan.strategy, 'resolution_reduction')
        self.assertIn('width', plan.new_params)
        self.assertIn('height', plan.new_params)
        self.assertLess(plan.new_params['width'], 1920)
        self.assertLess(plan.new_params['height'], 1080)
    
    def test_suggest_adjustments_segmentation_fallback(self):
        """Test adjustment suggestions fall back to segmentation for extreme cases"""
        params = {
            'width': 320,
            'height': 180,
            'fps': 10,
            'bitrate': 1,  # Extremely low bitrate
            'encoder': 'libx264',
            'audio_bitrate': 64
        }
        video_info = {'duration_seconds': 3600}  # Very long video
        
        plan = self.validator.suggest_adjustments(params, 1.0, video_info)
        
        self.assertEqual(plan.strategy, 'segmentation')
        self.assertEqual(plan.quality_impact, 'minimal')
    
    def test_calculate_minimum_viable_params_success(self):
        """Test calculation of minimum viable parameters"""
        video_info = {'duration_seconds': 60}
        
        params = self.validator.calculate_minimum_viable_params(video_info, 10.0, 'libx264')
        
        self.assertTrue(params['viable'])
        self.assertGreaterEqual(params['bitrate'], 3)  # Meets libx264 minimum
    
    def test_calculate_minimum_viable_params_segmentation_needed(self):
        """Test minimum viable parameters when segmentation is required"""
        video_info = {'duration_seconds': 7200}  # Very long video (2 hours)
        
        params = self.validator.calculate_minimum_viable_params(video_info, 0.5, 'libx264')  # Very small size
        
        self.assertFalse(params['viable'])
        self.assertTrue(params['requires_segmentation'])
    
    def test_get_encoder_minimum(self):
        """Test getting encoder minimum bitrate"""
        self.assertEqual(self.validator.get_encoder_minimum('libx264'), 3)
        self.assertEqual(self.validator.get_encoder_minimum('libx265'), 5)
        self.assertEqual(self.validator.get_encoder_minimum('unknown'), 3)  # Default
    
    def test_validation_enabled_check(self):
        """Test validation enabled configuration check"""
        self.assertTrue(self.validator.is_validation_enabled())
    
    def test_bitrate_validation_error_creation(self):
        """Test BitrateValidationError creation and methods"""
        video_info = {'duration': 120, 'width': 1920, 'height': 1080, 'fps': 30}
        
        error = BitrateValidationError(
            message="Test error",
            bitrate_kbps=2,
            minimum_required=5,
            encoder='libx264',
            severity='critical',
            context='test_context',
            video_info=video_info
        )
        
        self.assertEqual(error.bitrate_kbps, 2)
        self.assertEqual(error.minimum_required, 5)
        self.assertEqual(error.deficit_kbps, 3)
        self.assertEqual(error.deficit_ratio, 2.5)
        
        detailed_msg = error.get_detailed_message()
        self.assertIn('libx264', detailed_msg)
        self.assertIn('2kbps < 5kbps', detailed_msg)
        self.assertIn('test_context', detailed_msg)
        
        short_msg = error.get_short_message()
        self.assertIn('libx264 bitrate validation failed', short_msg)


class TestBitrateValidatorEdgeCases(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for edge cases"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create minimal config
        self.config_manager = ConfigManager(self.temp_dir)
        self.validator = BitrateValidator(self.config_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_zero_bitrate_handling(self):
        """Test handling of zero bitrate"""
        result = self.validator.validate_bitrate(0, 'libx264')
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.severity, 'critical')
    
    def test_negative_bitrate_handling(self):
        """Test handling of negative bitrate"""
        result = self.validator.validate_bitrate(-5, 'libx264')
        
        self.assertFalse(result.is_valid)
        self.assertEqual(result.severity, 'critical')
    
    def test_extremely_high_bitrate(self):
        """Test handling of extremely high bitrate"""
        result = self.validator.validate_bitrate(100000, 'libx264')
        
        self.assertTrue(result.is_valid)
        self.assertEqual(result.severity, 'info')
    
    def test_empty_video_info(self):
        """Test adjustment suggestions with empty video info"""
        params = {
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'bitrate': 1000,
            'encoder': 'libx264',
            'audio_bitrate': 64
        }
        
        plan = self.validator.suggest_adjustments(params, 10.0, {})
        
        # Should still work with default duration
        self.assertIsNotNone(plan.strategy)
    
    def test_zero_duration_handling(self):
        """Test handling of zero duration in calculations"""
        video_info = {'duration_seconds': 0}
        
        params = self.validator.calculate_minimum_viable_params(video_info, 10.0, 'libx264')
        
        # Should handle gracefully without division by zero
        self.assertIsNotNone(params)
    
    def test_invalid_fallback_resolutions_config(self):
        """Test handling of invalid fallback resolutions in config"""
        # Create config with invalid fallback resolutions
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        invalid_config = {
            'video_compression': {
                'bitrate_validation': {
                    'fallback_resolutions': 'invalid_format'  # Should be list
                }
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        config_manager = ConfigManager(self.temp_dir)
        validator = BitrateValidator(config_manager)
        
        # Should fall back to defaults
        self.assertIsInstance(validator.fallback_resolutions, list)
        self.assertGreater(len(validator.fallback_resolutions), 0)


if __name__ == '__main__':
    unittest.main()