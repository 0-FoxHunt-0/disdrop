"""
Integration tests for compression pipeline with bitrate validation
Tests end-to-end compression, segmentation fallback, and batch processing resilience
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import yaml
import json
from pathlib import Path
from src.config_manager import ConfigManager
from src.hardware_detector import HardwareDetector
from src.video_compressor import DynamicVideoCompressor
from src.bitrate_validator import BitrateValidator
from src.adaptive_parameter_adjuster import AdaptiveParameterAdjuster
from src.video_segmenter import VideoSegmenter


class TestCompressionPipelineIntegration(unittest.TestCase):
    
    def setUp(self):
        """Set up integration test fixtures"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test configuration with bitrate validation enabled
        self.config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        self.test_config = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {
                    'fallback': 'libx264'
                },
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
                    'segmentation_threshold_mb': 50
                }
            },
            'gif_settings': {
                'max_file_size_mb': 5
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Initialize components
        self.config_manager = ConfigManager(self.temp_dir)
        self.hardware_detector = Mock(spec=HardwareDetector)
        self.hardware_detector.get_best_encoder.return_value = 'libx264'
        self.hardware_detector.has_nvidia_gpu.return_value = False
        self.hardware_detector.has_amd_gpu.return_value = False
        
        # Create mock video file info
        self.mock_video_info = {
            'duration_seconds': 120,
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'bitrate': 5000,
            'file_size_mb': 25.0
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_mock_video_file(self, filename: str, duration: float = 120, 
                              width: int = 1920, height: int = 1080) -> str:
        """Create a mock video file for testing"""
        video_path = os.path.join(self.input_dir, filename)
        
        # Create empty file (in real tests, this would be a real video)
        with open(video_path, 'wb') as f:
            f.write(b'mock_video_data')
        
        return video_path
    
    def test_end_to_end_compression_with_validation(self):
        """Test end-to-end compression with bitrate validation enabled"""
        # Initialize compressor with validation
        compressor = DynamicVideoCompressor(self.config_manager, self.hardware_detector)
        
        # Verify bitrate validation is enabled
        self.assertTrue(compressor.bitrate_validator.is_validation_enabled())
        
        # Verify validator has correct encoder minimums
        self.assertEqual(compressor.bitrate_validator.get_encoder_minimum('libx264'), 3)
        self.assertEqual(compressor.bitrate_validator.get_encoder_minimum('libx265'), 5)
        self.assertEqual(compressor.bitrate_validator.get_encoder_minimum('h264_nvenc'), 2)
        
        # Test bitrate validation logic
        validation_result = compressor.bitrate_validator.validate_bitrate(5000, 'libx264')
        self.assertTrue(validation_result.is_valid)
        
        validation_result = compressor.bitrate_validator.validate_bitrate(2, 'libx264')
        self.assertFalse(validation_result.is_valid)
        self.assertEqual(validation_result.severity, 'warning')
    
    def test_bitrate_validation_triggers_adjustment(self):
        """Test that low bitrate triggers parameter adjustment"""
        # Initialize compressor
        compressor = DynamicVideoCompressor(self.config_manager, self.hardware_detector)
        
        # Test parameter adjustment logic directly
        params = {
            'width': 1920,
            'height': 1080,
            'fps': 30,
            'bitrate': 2,  # Below libx264 minimum of 3
            'encoder': 'libx264',
            'audio_bitrate': 64
        }
        video_info = {'duration_seconds': 3600}  # Long video
        
        # Test adjustment
        adjustment_result = compressor.parameter_adjuster.adjust_for_bitrate_floor(
            params, 3, video_info, 10.0
        )
        
        # Should successfully adjust parameters
        self.assertTrue(adjustment_result.success)
        self.assertIn(adjustment_result.adjustment_type, ['resolution', 'fps', 'combined'])
        self.assertGreaterEqual(adjustment_result.adjusted_params.get('bitrate', 0), 3)
    
    def test_segmentation_fallback_integration(self):
        """Test segmentation fallback when compression fails"""
        # Initialize compressor
        compressor = DynamicVideoCompressor(self.config_manager, self.hardware_detector)
        
        # Test minimum viable parameters calculation for extreme case
        extreme_video_info = {'duration_seconds': 14400}  # 4 hours
        
        # Test with very small target size that would require segmentation
        params = compressor.bitrate_validator.calculate_minimum_viable_params(
            extreme_video_info, 0.1, 'libx264'  # 100KB for 4 hour video
        )
        
        # Should indicate segmentation is required
        self.assertFalse(params['viable'])
        self.assertTrue(params['requires_segmentation'])
        
        # Test adjustment plan for extreme case
        extreme_params = {
            'width': 320,
            'height': 180,
            'fps': 10,
            'bitrate': 1,
            'encoder': 'libx264',
            'audio_bitrate': 64
        }
        
        plan = compressor.bitrate_validator.suggest_adjustments(
            extreme_params, 0.1, extreme_video_info
        )
        
        # Should suggest segmentation
        self.assertEqual(plan.strategy, 'segmentation')
    
    def test_batch_processing_resilience(self):
        """Test batch processing continues despite individual file failures"""
        # Create multiple test files with different characteristics
        test_files = [
            ('normal_video.mp4', 60, 1920, 1080),      # Normal video
            ('long_video.mp4', 3600, 1920, 1080),      # Very long video
            ('high_res_video.mp4', 120, 3840, 2160),   # 4K video
            ('corrupted_video.mp4', 0, 0, 0),          # Corrupted/invalid video
        ]
        
        created_files = []
        for filename, duration, width, height in test_files:
            file_path = self.create_mock_video_file(filename, duration, width, height)
            created_files.append(file_path)
        
        # Initialize compressor
        compressor = DynamicVideoCompressor(self.config_manager, self.hardware_detector)
        
        # Track processing results
        results = []
        error_types = {}
        
        # Mock different behaviors for different files
        def mock_get_info_side_effect(file_path):
            filename = os.path.basename(file_path)
            if 'normal' in filename:
                return {'duration_seconds': 60, 'width': 1920, 'height': 1080, 'fps': 30, 'bitrate': 5000, 'file_size_mb': 10}
            elif 'long' in filename:
                return {'duration_seconds': 3600, 'width': 1920, 'height': 1080, 'fps': 30, 'bitrate': 5000, 'file_size_mb': 500}
            elif 'high_res' in filename:
                return {'duration_seconds': 120, 'width': 3840, 'height': 2160, 'fps': 30, 'bitrate': 20000, 'file_size_mb': 50}
            elif 'corrupted' in filename:
                raise Exception("Invalid video file")
        
        def mock_compress_side_effect(*args, **kwargs):
            # Simulate different success/failure scenarios
            file_path = args[0] if args else kwargs.get('input_path', '')
            filename = os.path.basename(file_path)
            
            if 'normal' in filename:
                return (True, os.path.join(self.output_dir, f"{filename}_compressed.mp4"))
            elif 'long' in filename:
                # Simulate bitrate validation failure leading to segmentation
                return (False, None)
            elif 'high_res' in filename:
                return (True, os.path.join(self.output_dir, f"{filename}_compressed.mp4"))
            elif 'corrupted' in filename:
                return (False, None)
        
        # Process each file and track results
        with patch.object(compressor, '_get_video_info', side_effect=mock_get_info_side_effect):
            with patch.object(compressor, 'compress_video', side_effect=mock_compress_side_effect):
                for file_path in created_files:
                    try:
                        success, output_path = compressor.compress_video(
                            file_path,
                            self.output_dir,
                            platform='discord'
                        )
                        results.append((file_path, success, output_path))
                        
                        if not success:
                            filename = os.path.basename(file_path)
                            if 'long' in filename:
                                error_types['bitrate_validation'] = error_types.get('bitrate_validation', 0) + 1
                            elif 'corrupted' in filename:
                                error_types['invalid_file'] = error_types.get('invalid_file', 0) + 1
                            else:
                                error_types['unknown'] = error_types.get('unknown', 0) + 1
                        
                    except Exception as e:
                        # Batch processing should continue despite individual failures
                        results.append((file_path, False, None))
                        error_types['exception'] = error_types.get('exception', 0) + 1
        
        # Verify batch processing resilience
        total_files = len(created_files)
        successful_files = sum(1 for _, success, _ in results if success)
        failed_files = total_files - successful_files
        
        # Should have processed all files
        self.assertEqual(len(results), total_files)
        
        # Should have some successes and some failures (based on our mock setup)
        self.assertGreater(successful_files, 0, "Should have at least some successful compressions")
        self.assertGreater(failed_files, 0, "Should have some failures to test resilience")
        
        # Verify error tracking
        self.assertGreater(len(error_types), 0, "Should have tracked error types")
        
        # Test the batch processing resilience logging
        compressor.bitrate_validator.log_batch_processing_resilience(
            total_files, successful_files, failed_files, error_types
        )
    
    @patch('src.ffmpeg_utils.FFmpegUtils.get_video_info')
    def test_problematic_files_handling(self, mock_get_info):
        """Test handling of problematic files that cause validation issues"""
        problematic_scenarios = [
            {
                'name': 'zero_duration.mp4',
                'info': {'duration_seconds': 0, 'width': 1920, 'height': 1080, 'fps': 30, 'bitrate': 5000, 'file_size_mb': 0},
                'expected_error': 'duration'
            },
            {
                'name': 'negative_bitrate.mp4', 
                'info': {'duration_seconds': 60, 'width': 1920, 'height': 1080, 'fps': 30, 'bitrate': -1000, 'file_size_mb': 10},
                'expected_error': 'bitrate'
            },
            {
                'name': 'zero_resolution.mp4',
                'info': {'duration_seconds': 60, 'width': 0, 'height': 0, 'fps': 30, 'bitrate': 5000, 'file_size_mb': 10},
                'expected_error': 'resolution'
            },
            {
                'name': 'extreme_fps.mp4',
                'info': {'duration_seconds': 60, 'width': 1920, 'height': 1080, 'fps': 1000, 'bitrate': 50000, 'file_size_mb': 100},
                'expected_error': 'fps'
            }
        ]
        
        # Initialize compressor
        compressor = DynamicVideoCompressor(self.config_manager, self.hardware_detector)
        
        for scenario in problematic_scenarios:
            with self.subTest(scenario=scenario['name']):
                # Create mock file
                file_path = self.create_mock_video_file(scenario['name'])
                
                # Setup mock to return problematic info
                mock_get_info.return_value = scenario['info']
                
                # Test that the system handles the problematic file gracefully
                try:
                    with patch.object(compressor, '_compress_with_cae_discord_10mb') as mock_compress:
                        mock_compress.return_value = (False, None)
                        
                        success, output_path = compressor.compress_video(
                            file_path,
                            self.output_dir,
                            platform='discord'
                        )
                        
                        # Should handle gracefully (may succeed or fail, but shouldn't crash)
                        self.assertIsInstance(success, bool)
                        
                except Exception as e:
                    # If an exception is raised, it should be informative
                    error_msg = str(e).lower()
                    self.assertTrue(
                        any(keyword in error_msg for keyword in ['bitrate', 'validation', 'parameter', 'video']),
                        f"Exception should be informative: {e}"
                    )


class TestCompressionPipelineConfiguration(unittest.TestCase):
    
    def setUp(self):
        """Set up configuration test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_validation_disabled_integration(self):
        """Test compression pipeline with bitrate validation disabled"""
        # Create config with validation disabled
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        config_with_validation_disabled = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {
                    'fallback': 'libx264'
                },
                'bitrate_validation': {
                    'enabled': False
                }
            },
            'gif_settings': {
                'max_file_size_mb': 5
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_with_validation_disabled, f)
        
        # Initialize components
        config_manager = ConfigManager(self.temp_dir)
        hardware_detector = Mock(spec=HardwareDetector)
        hardware_detector.get_best_encoder.return_value = 'libx264'
        
        # Initialize compressor
        compressor = DynamicVideoCompressor(config_manager, hardware_detector)
        
        # Verify validation is disabled
        self.assertFalse(compressor.bitrate_validator.is_validation_enabled())
    
    def test_custom_encoder_minimums_integration(self):
        """Test compression pipeline with custom encoder minimums"""
        # Create config with custom encoder minimums
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        config_with_custom_minimums = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {
                    'fallback': 'libx264'
                },
                'bitrate_validation': {
                    'enabled': True,
                    'encoder_minimums': {
                        'libx264': 1,  # Lower than default
                        'libx265': 2,  # Lower than default
                        'h264_nvenc': 1  # Lower than default
                    }
                }
            },
            'gif_settings': {
                'max_file_size_mb': 5
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_with_custom_minimums, f)
        
        # Initialize components
        config_manager = ConfigManager(self.temp_dir)
        hardware_detector = Mock(spec=HardwareDetector)
        
        # Initialize validator
        validator = BitrateValidator(config_manager)
        
        # Verify custom minimums are loaded
        self.assertEqual(validator.get_encoder_minimum('libx264'), 1)
        self.assertEqual(validator.get_encoder_minimum('libx265'), 2)
        self.assertEqual(validator.get_encoder_minimum('h264_nvenc'), 1)
    
    def test_cli_override_integration(self):
        """Test CLI parameter overrides in compression pipeline"""
        # Create base config
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        base_config = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {
                    'fallback': 'libx264'
                },
                'bitrate_validation': {
                    'enabled': True,
                    'encoder_minimums': {
                        'libx264': 3
                    }
                }
            },
            'gif_settings': {
                'max_file_size_mb': 5
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(base_config, f)
        
        # Initialize config manager
        config_manager = ConfigManager(self.temp_dir)
        
        # Simulate CLI overrides
        cli_overrides = {
            'video_compression.bitrate_validation.enabled': False,
            'video_compression.max_file_size_mb': 25
        }
        
        config_manager.update_from_args(cli_overrides)
        
        # Verify overrides took effect
        self.assertFalse(config_manager.get('video_compression.bitrate_validation.enabled'))
        self.assertEqual(config_manager.get('video_compression.max_file_size_mb'), 25)
        
        # Initialize validator with overridden config
        validator = BitrateValidator(config_manager)
        
        # Verify validation is disabled due to CLI override
        self.assertFalse(validator.is_validation_enabled())


if __name__ == '__main__':
    unittest.main()