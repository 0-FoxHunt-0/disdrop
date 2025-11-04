"""
Unit tests for quality evaluation components
Tests SSIM parsing, error handling, fallback behavior, logging safety, and configuration
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import yaml
import json
import subprocess
from src.config_manager import ConfigManager
from src.quality_gates import QualityGates


class TestQualityGatesSSIMParsing(unittest.TestCase):
    """Test SSIM parsing with various FFmpeg output formats"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        self.quality_gates = QualityGates(self.config_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_ssim_parsing_standard_format(self):
        """Test SSIM parsing with standard FFmpeg output format"""
        # Standard format: n:X Y:0.xxxxx U:... V:... All:0.xxxxx (XXdB)
        stderr_output = """
        n:1 Y:0.9876 U:0.9654 V:0.9543 All:0.9691 (23.12dB)
        n:2 Y:0.9845 U:0.9632 V:0.9521 All:0.9666 (22.98dB)
        n:3 Y:0.9823 U:0.9611 V:0.9498 All:0.9644 (22.85dB)
        """
        
        ssim_score, confidence = self.quality_gates._parse_ssim_output(stderr_output)
        
        self.assertIsNotNone(ssim_score)
        self.assertGreater(ssim_score, 0.98)  # Should be average of Y values
        self.assertLessEqual(ssim_score, 1.0)
        self.assertGreater(confidence, 0.8)  # High confidence for multiple valid scores
    
    def test_ssim_parsing_alternative_format(self):
        """Test SSIM parsing with alternative Y: format"""
        stderr_output = """
        Frame 1: Y: 0.9876
        Frame 2: Y: 0.9845
        Frame 3: Y: 0.9823
        """
        
        ssim_score, confidence = self.quality_gates._parse_ssim_output(stderr_output)
        
        self.assertIsNotNone(ssim_score)
        self.assertGreater(ssim_score, 0.98)
        self.assertLessEqual(ssim_score, 1.0)
        self.assertGreater(confidence, 0.7)
    
    def test_ssim_parsing_all_format(self):
        """Test SSIM parsing with All: values"""
        stderr_output = """
        All: 0.9691
        All: 0.9666
        All: 0.9644
        """
        
        ssim_score, confidence = self.quality_gates._parse_ssim_output(stderr_output)
        
        self.assertIsNotNone(ssim_score)
        self.assertGreater(ssim_score, 0.96)
        self.assertLessEqual(ssim_score, 1.0)
        self.assertGreater(confidence, 0.7)
    
    def test_ssim_parsing_mixed_format(self):
        """Test SSIM parsing with mixed valid and invalid data"""
        stderr_output = """
        n:1 Y:0.9876 U:0.9654 V:0.9543 All:0.9691 (23.12dB)
        Some random text here
        Y: 0.9845
        Invalid line with no numbers
        All: 0.9644
        """
        
        ssim_score, confidence = self.quality_gates._parse_ssim_output(stderr_output)
        
        self.assertIsNotNone(ssim_score)
        self.assertGreater(ssim_score, 0.96)
        self.assertLessEqual(ssim_score, 1.0)
        self.assertGreater(confidence, 0.5)
    
    def test_ssim_parsing_empty_output(self):
        """Test SSIM parsing with empty output"""
        stderr_output = ""
        
        ssim_score, confidence = self.quality_gates._parse_ssim_output(stderr_output)
        
        self.assertIsNone(ssim_score)
        self.assertEqual(confidence, 0.0)
    
    def test_ssim_parsing_invalid_output(self):
        """Test SSIM parsing with invalid output"""
        stderr_output = """
        No SSIM data here
        Just random text
        Some numbers: 123 456 789
        """
        
        ssim_score, confidence = self.quality_gates._parse_ssim_output(stderr_output)
        
        self.assertIsNone(ssim_score)
        self.assertEqual(confidence, 0.0)
    
    def test_ssim_parsing_out_of_range_values(self):
        """Test SSIM parsing with out-of-range values"""
        stderr_output = """
        n:1 Y:1.5 U:0.9654 V:0.9543 All:0.9691 (23.12dB)
        n:2 Y:-0.5 U:0.9632 V:0.9521 All:0.9666 (22.98dB)
        n:3 Y:0.9823 U:0.9611 V:0.9498 All:0.9644 (22.85dB)
        """
        
        ssim_score, confidence = self.quality_gates._parse_ssim_output(stderr_output)
        
        # Should only use the valid score (0.9823)
        self.assertIsNotNone(ssim_score)
        self.assertAlmostEqual(ssim_score, 0.9823, places=4)
        self.assertLess(confidence, 0.8)  # Lower confidence due to invalid values
    
    def test_ssim_validation_outlier_detection(self):
        """Test SSIM outlier detection functionality"""
        # Create scores with outliers
        scores = [0.95, 0.96, 0.94, 0.97, 0.2, 0.95, 0.96, 1.5, 0.94]  # 0.2 and 1.5 are outliers
        
        validated_scores, confidence = self.quality_gates._validate_ssim_scores(scores)
        
        # Should remove outliers
        self.assertGreater(len(validated_scores), 0)
        self.assertLess(len(validated_scores), len(scores))
        
        # All validated scores should be in reasonable range
        for score in validated_scores:
            self.assertGreaterEqual(score, 0.9)
            self.assertLessEqual(score, 1.0)
        
        self.assertGreater(confidence, 0.0)


class TestQualityGatesVMAFParsing(unittest.TestCase):
    """Test VMAF parsing with various FFmpeg output formats"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        self.quality_gates = QualityGates(self.config_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_vmaf_parsing_json_pooled_format(self):
        """Test VMAF parsing with JSON pooled metrics format (libvmaf 2.x)"""
        stderr_output = """
        {"pooled_metrics": {"vmaf": {"mean": 85.67, "std": 2.34}}}
        """
        
        vmaf_score = self.quality_gates._parse_vmaf_output(stderr_output)
        
        self.assertIsNotNone(vmaf_score)
        self.assertAlmostEqual(vmaf_score, 85.67, places=2)
    
    def test_vmaf_parsing_json_aggregate_format(self):
        """Test VMAF parsing with JSON aggregate format"""
        stderr_output = """
        {"aggregate": {"VMAF_score": 82.45}}
        """
        
        vmaf_score = self.quality_gates._parse_vmaf_output(stderr_output)
        
        self.assertIsNotNone(vmaf_score)
        self.assertAlmostEqual(vmaf_score, 82.45, places=2)
    
    def test_vmaf_parsing_json_direct_format(self):
        """Test VMAF parsing with direct JSON vmaf field"""
        stderr_output = """
        {"vmaf": {"mean": 78.92}}
        """
        
        vmaf_score = self.quality_gates._parse_vmaf_output(stderr_output)
        
        self.assertIsNotNone(vmaf_score)
        self.assertAlmostEqual(vmaf_score, 78.92, places=2)
    
    def test_vmaf_parsing_text_format(self):
        """Test VMAF parsing with text format"""
        stderr_output = """
        VMAF score: 84.56
        """
        
        vmaf_score = self.quality_gates._parse_vmaf_output(stderr_output)
        
        self.assertIsNotNone(vmaf_score)
        self.assertAlmostEqual(vmaf_score, 84.56, places=2)
    
    def test_vmaf_parsing_multiple_json_objects(self):
        """Test VMAF parsing with multiple JSON objects"""
        stderr_output = """
        {"frame": 1, "metrics": {"vmaf": 80.1}}
        {"frame": 2, "metrics": {"vmaf": 82.3}}
        {"pooled_metrics": {"vmaf": {"mean": 81.2}}}
        """
        
        vmaf_score = self.quality_gates._parse_vmaf_output(stderr_output)
        
        self.assertIsNotNone(vmaf_score)
        self.assertAlmostEqual(vmaf_score, 81.2, places=1)  # Should use pooled metrics
    
    def test_vmaf_parsing_invalid_json(self):
        """Test VMAF parsing with invalid JSON"""
        stderr_output = """
        {"invalid": json: format}
        VMAF score: 75.43
        """
        
        vmaf_score = self.quality_gates._parse_vmaf_output(stderr_output)
        
        self.assertIsNotNone(vmaf_score)
        self.assertAlmostEqual(vmaf_score, 75.43, places=2)  # Should fall back to text parsing
    
    def test_vmaf_parsing_out_of_range(self):
        """Test VMAF parsing with out-of-range values"""
        stderr_output = """
        VMAF score: 150.0
        vmaf: -20.5
        mean: 85.67
        """
        
        vmaf_score = self.quality_gates._parse_vmaf_output(stderr_output)
        
        self.assertIsNotNone(vmaf_score)
        self.assertAlmostEqual(vmaf_score, 85.67, places=2)  # Should use valid score
    
    def test_vmaf_parsing_no_valid_data(self):
        """Test VMAF parsing with no valid VMAF data"""
        stderr_output = """
        No VMAF data here
        Just random output
        """
        
        vmaf_score = self.quality_gates._parse_vmaf_output(stderr_output)
        
        self.assertIsNone(vmaf_score)


class TestQualityGatesErrorHandling(unittest.TestCase):
    """Test error handling and fallback behavior"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        self.quality_gates = QualityGates(self.config_manager)
        
        # Create mock video files
        self.original_path = os.path.join(self.temp_dir, 'original.mp4')
        self.compressed_path = os.path.join(self.temp_dir, 'compressed.mp4')
        
        # Create empty files
        with open(self.original_path, 'wb') as f:
            f.write(b'mock_video_data')
        with open(self.compressed_path, 'wb') as f:
            f.write(b'mock_compressed_data')
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_missing_original_file(self):
        """Test handling of missing original file"""
        nonexistent_path = os.path.join(self.temp_dir, 'nonexistent.mp4')
        
        result = self.quality_gates.evaluate_quality(
            nonexistent_path, self.compressed_path
        )
        
        self.assertFalse(result['evaluation_success'])
        self.assertEqual(result['method'], 'error')
        self.assertIn('error', result['details'])
        self.assertIn('not found', result['details']['error'])
    
    def test_missing_compressed_file(self):
        """Test handling of missing compressed file"""
        nonexistent_path = os.path.join(self.temp_dir, 'nonexistent.mp4')
        
        result = self.quality_gates.evaluate_quality(
            self.original_path, nonexistent_path
        )
        
        self.assertFalse(result['evaluation_success'])
        self.assertEqual(result['method'], 'error')
        self.assertIn('error', result['details'])
        self.assertIn('not found', result['details']['error'])
    
    @patch('src.quality_gates.QualityGates._execute_ffmpeg_with_retry')
    def test_ffmpeg_execution_failure(self, mock_execute):
        """Test handling of FFmpeg execution failure"""
        # Mock FFmpeg failure
        mock_execute.return_value = {
            'success': False,
            'stdout': '',
            'stderr': 'FFmpeg error occurred',
            'returncode': 1,
            'execution_time': 1.0,
            'attempts': 3,
            'error_details': {
                'category': 'generic_ffmpeg_error',
                'is_retryable': False,
                'description': 'FFmpeg processing error'
            }
        }
        
        # Mock VMAF availability check
        with patch.object(self.quality_gates, 'check_vmaf_available', return_value=False):
            result = self.quality_gates.evaluate_quality(
                self.original_path, self.compressed_path
            )
        
        self.assertFalse(result['evaluation_success'])
        self.assertEqual(result['method'], 'ssim_only')
        self.assertIsNone(result['ssim_score'])
        self.assertEqual(result['confidence'], 0.0)
    
    def test_fallback_mode_conservative(self):
        """Test conservative fallback mode behavior"""
        # Set conservative mode
        self.quality_gates.fallback_mode = 'conservative'
        
        # Create a result that needs fallback (low confidence)
        result = {
            'vmaf_score': None,
            'ssim_score': None,
            'passes': False,
            'method': 'error',
            'confidence': 0.0,
            'evaluation_success': False,
            'details': {}
        }
        
        fallback_result = self.quality_gates._apply_fallback_behavior(
            result, vmaf_threshold=80.0, ssim_threshold=0.94
        )
        
        self.assertTrue(fallback_result['passes'])  # Conservative mode should pass
        self.assertTrue(fallback_result['details']['fallback_applied'])
        self.assertEqual(fallback_result['details']['fallback_mode'], 'conservative')
    
    def test_fallback_mode_strict(self):
        """Test strict fallback mode behavior"""
        # Set strict mode
        self.quality_gates.fallback_mode = 'strict'
        
        # Create a result that needs fallback
        result = {
            'vmaf_score': None,
            'ssim_score': None,
            'passes': True,  # Original result was passing
            'method': 'error',
            'confidence': 0.0,
            'evaluation_success': False,
            'details': {}
        }
        
        fallback_result = self.quality_gates._apply_fallback_behavior(
            result, vmaf_threshold=80.0, ssim_threshold=0.94
        )
        
        self.assertFalse(fallback_result['passes'])  # Strict mode should fail
        self.assertTrue(fallback_result['details']['fallback_applied'])
        self.assertEqual(fallback_result['details']['fallback_mode'], 'strict')
    
    def test_fallback_mode_permissive(self):
        """Test permissive fallback mode behavior"""
        # Set permissive mode
        self.quality_gates.fallback_mode = 'permissive'
        
        # Create a result that needs fallback
        result = {
            'vmaf_score': None,
            'ssim_score': None,
            'passes': False,  # Original result was failing
            'method': 'error',
            'confidence': 0.0,
            'evaluation_success': False,
            'details': {}
        }
        
        fallback_result = self.quality_gates._apply_fallback_behavior(
            result, vmaf_threshold=80.0, ssim_threshold=0.94
        )
        
        self.assertTrue(fallback_result['passes'])  # Permissive mode should pass
        self.assertTrue(fallback_result['details']['fallback_applied'])
        self.assertEqual(fallback_result['details']['fallback_mode'], 'permissive')
    
    def test_confidence_threshold_fallback(self):
        """Test fallback behavior based on confidence thresholds"""
        # Set confidence thresholds
        self.quality_gates.confidence_thresholds = {
            'minimum_acceptable': 0.5,
            'high_confidence': 0.8,
            'decision_threshold': 0.6
        }
        
        # Create result with low confidence
        result = {
            'vmaf_score': 85.0,
            'ssim_score': 0.95,
            'passes': True,
            'method': 'vmaf+ssim',
            'confidence': 0.3,  # Below minimum acceptable
            'evaluation_success': True,
            'details': {}
        }
        
        fallback_result = self.quality_gates._apply_fallback_behavior(
            result, vmaf_threshold=80.0, ssim_threshold=0.94
        )
        
        self.assertTrue(fallback_result['details']['fallback_applied'])
        self.assertEqual(fallback_result['details']['fallback_reason'], 'low_confidence')


class TestQualityGatesLoggingSafety(unittest.TestCase):
    """Test logging safety with None values and edge cases"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        self.quality_gates = QualityGates(self.config_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_safe_format_scores_with_none_values(self):
        """Test safe formatting of scores with None values"""
        # Test with both None
        vmaf_str, ssim_str = self.quality_gates._safe_format_scores(None, None)
        self.assertEqual(vmaf_str, "N/A")
        self.assertEqual(ssim_str, "N/A")
        
        # Test with VMAF None, SSIM valid
        vmaf_str, ssim_str = self.quality_gates._safe_format_scores(None, 0.95)
        self.assertEqual(vmaf_str, "N/A")
        self.assertEqual(ssim_str, "0.9500")
        
        # Test with VMAF valid, SSIM None
        vmaf_str, ssim_str = self.quality_gates._safe_format_scores(85.67, None)
        self.assertEqual(vmaf_str, "85.67")
        self.assertEqual(ssim_str, "N/A")
        
        # Test with both valid
        vmaf_str, ssim_str = self.quality_gates._safe_format_scores(85.67, 0.95)
        self.assertEqual(vmaf_str, "85.67")
        self.assertEqual(ssim_str, "0.9500")
    
    def test_safe_score_comparison_with_none(self):
        """Test safe score comparison with None values"""
        # None score should always return False
        self.assertFalse(self.quality_gates._safe_score_comparison(None, 80.0))
        self.assertFalse(self.quality_gates._safe_score_comparison(None, 0.94))
        
        # Valid scores should compare normally
        self.assertTrue(self.quality_gates._safe_score_comparison(85.0, 80.0))
        self.assertFalse(self.quality_gates._safe_score_comparison(75.0, 80.0))
        self.assertTrue(self.quality_gates._safe_score_comparison(0.95, 0.94))
        self.assertFalse(self.quality_gates._safe_score_comparison(0.93, 0.94))
    
    def test_validate_quality_result_structure(self):
        """Test quality result validation and structure completeness"""
        # Test incomplete result
        incomplete_result = {
            'vmaf_score': 85.0,
            # Missing other required fields
        }
        
        validated = self.quality_gates._validate_quality_result(incomplete_result)
        
        # Should have all required fields
        required_fields = ['vmaf_score', 'ssim_score', 'passes', 'method', 
                          'confidence', 'evaluation_success', 'details']
        for field in required_fields:
            self.assertIn(field, validated)
        
        # Should have safe default values
        self.assertIsInstance(validated['passes'], bool)
        self.assertIsInstance(validated['confidence'], float)
        self.assertIsInstance(validated['evaluation_success'], bool)
        self.assertIsInstance(validated['details'], dict)
    
    def test_validate_quality_result_score_ranges(self):
        """Test validation of score ranges"""
        # Test out-of-range VMAF score
        result_bad_vmaf = {
            'vmaf_score': 150.0,  # Invalid: > 100
            'ssim_score': 0.95,
            'passes': True,
            'method': 'vmaf+ssim',
            'confidence': 0.8,
            'evaluation_success': True,
            'details': {}
        }
        
        validated = self.quality_gates._validate_quality_result(result_bad_vmaf)
        self.assertIsNone(validated['vmaf_score'])  # Should be set to None
        
        # Test out-of-range SSIM score
        result_bad_ssim = {
            'vmaf_score': 85.0,
            'ssim_score': 1.5,  # Invalid: > 1.0
            'passes': True,
            'method': 'vmaf+ssim',
            'confidence': 0.8,
            'evaluation_success': True,
            'details': {}
        }
        
        validated = self.quality_gates._validate_quality_result(result_bad_ssim)
        self.assertIsNone(validated['ssim_score'])  # Should be set to None
        
        # Test out-of-range confidence
        result_bad_confidence = {
            'vmaf_score': 85.0,
            'ssim_score': 0.95,
            'passes': True,
            'method': 'vmaf+ssim',
            'confidence': 1.5,  # Invalid: > 1.0
            'evaluation_success': True,
            'details': {}
        }
        
        validated = self.quality_gates._validate_quality_result(result_bad_confidence)
        self.assertLessEqual(validated['confidence'], 1.0)  # Should be clamped
        self.assertGreaterEqual(validated['confidence'], 0.0)
    
    @patch('src.quality_gates.logger')
    def test_logging_with_none_values_no_exceptions(self, mock_logger):
        """Test that logging with None values doesn't raise exceptions"""
        # Create result with None values
        result = {
            'vmaf_score': None,
            'ssim_score': None,
            'passes': False,
            'method': 'error',
            'confidence': 0.0,
            'evaluation_success': False,
            'details': {'error': 'Test error'}
        }
        
        # This should not raise any exceptions
        try:
            self.quality_gates._log_quality_evaluation_result(result)
            success = True
        except Exception as e:
            success = False
            self.fail(f"Logging with None values raised exception: {e}")
        
        self.assertTrue(success)
        self.assertTrue(mock_logger.info.called)


class TestQualityGatesConfiguration(unittest.TestCase):
    """Test configuration parsing and validation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_default_configuration(self):
        """Test default configuration when no config file exists"""
        config_manager = ConfigManager(self.temp_dir)
        quality_gates = QualityGates(config_manager)
        
        # Should have reasonable defaults
        self.assertEqual(quality_gates.fallback_mode, 'conservative')
        self.assertIsInstance(quality_gates.confidence_thresholds, dict)
        self.assertIsInstance(quality_gates.ffmpeg_config, dict)
        self.assertIsInstance(quality_gates.logging_config, dict)
        self.assertIsInstance(quality_gates.ssim_config, dict)
        self.assertIsInstance(quality_gates.vmaf_config, dict)
    
    def test_custom_configuration_loading(self):
        """Test loading custom configuration"""
        # Create custom config
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        custom_config = {
            'quality_evaluation': {
                'fallback_mode': 'strict',
                'confidence_thresholds': {
                    'minimum_acceptable': 0.4,
                    'high_confidence': 0.9,
                    'decision_threshold': 0.7
                },
                'ffmpeg_execution': {
                    'timeout_seconds': 600,
                    'retry_attempts': 5
                },
                'ssim_parsing': {
                    'timeout_seconds': 450,
                    'alternative_formats': False
                }
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(custom_config, f)
        
        config_manager = ConfigManager(self.temp_dir)
        quality_gates = QualityGates(config_manager)
        
        # Should use custom values
        self.assertEqual(quality_gates.fallback_mode, 'strict')
        self.assertEqual(quality_gates.confidence_thresholds['minimum_acceptable'], 0.4)
        self.assertEqual(quality_gates.confidence_thresholds['high_confidence'], 0.9)
        self.assertEqual(quality_gates.ffmpeg_config['timeout_seconds'], 600)
        self.assertEqual(quality_gates.ffmpeg_config['retry_attempts'], 5)
        self.assertEqual(quality_gates.ssim_config['timeout_seconds'], 450)
        self.assertFalse(quality_gates.ssim_config['alternative_formats'])
    
    def test_partial_configuration_with_defaults(self):
        """Test partial configuration with fallback to defaults"""
        # Create partial config
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        partial_config = {
            'quality_evaluation': {
                'fallback_mode': 'permissive',
                'ffmpeg_execution': {
                    'timeout_seconds': 400
                    # Missing other ffmpeg settings
                }
                # Missing other sections
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(partial_config, f)
        
        config_manager = ConfigManager(self.temp_dir)
        quality_gates = QualityGates(config_manager)
        
        # Should use custom values where provided
        self.assertEqual(quality_gates.fallback_mode, 'permissive')
        self.assertEqual(quality_gates.ffmpeg_config['timeout_seconds'], 400)
        
        # Should use defaults for missing values
        self.assertEqual(quality_gates.ffmpeg_config['retry_attempts'], 3)  # Default
        self.assertIsInstance(quality_gates.confidence_thresholds, dict)  # Default structure
    
    def test_invalid_fallback_mode_handling(self):
        """Test handling of invalid fallback mode"""
        # Create config with invalid fallback mode
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        invalid_config = {
            'quality_evaluation': {
                'fallback_mode': 'invalid_mode'
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        config_manager = ConfigManager(self.temp_dir)
        quality_gates = QualityGates(config_manager)
        
        # Should still work (will use the invalid mode and log warning during fallback)
        self.assertEqual(quality_gates.fallback_mode, 'invalid_mode')
        
        # Test that fallback behavior handles invalid mode gracefully
        result = {
            'vmaf_score': None,
            'ssim_score': None,
            'passes': False,
            'method': 'error',
            'confidence': 0.0,
            'evaluation_success': False,
            'details': {}
        }
        
        # Should not raise exception and should default to conservative
        fallback_result = quality_gates._apply_fallback_behavior(
            result, vmaf_threshold=80.0, ssim_threshold=0.94
        )
        
        self.assertTrue(fallback_result['passes'])  # Should default to conservative behavior


if __name__ == '__main__':
    unittest.main()