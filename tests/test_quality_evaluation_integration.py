"""
Integration tests for CAE pipeline quality evaluation
Tests end-to-end quality evaluation, quality gate decision making, fallback behavior, and performance
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import yaml
import json
import time
from pathlib import Path
from src.config_manager import ConfigManager
from src.hardware_detector import HardwareDetector
from src.video_compressor import DynamicVideoCompressor
from src.quality_gates import QualityGates


class TestQualityEvaluationIntegration(unittest.TestCase):
    """Test end-to-end quality evaluation with real video files"""
    
    def setUp(self):
        """Set up integration test fixtures"""
        # Create temporary directories
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test configuration with quality evaluation enabled
        self.config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        self.test_config = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {
                    'fallback': 'libx264'
                },
                'quality_controls': {
                    'cae_quality_gates': {
                        'enabled': True,
                        'vmaf_threshold': 80.0,
                        'ssim_threshold': 0.94,
                        'vmaf_threshold_low_res': 75.0
                    }
                }
            },
            'quality_evaluation': {
                'fallback_mode': 'conservative',
                'confidence_thresholds': {
                    'minimum_acceptable': 0.3,
                    'high_confidence': 0.8,
                    'decision_threshold': 0.5
                },
                'ffmpeg_execution': {
                    'timeout_seconds': 60,  # Shorter for tests
                    'retry_attempts': 2
                },
                'logging': {
                    'log_parsing_attempts': True,
                    'log_performance_metrics': True
                }
            }
        }
        
        with open(self.config_file, 'w') as f:
            yaml.dump(self.test_config, f)
        
        # Initialize components
        self.config_manager = ConfigManager(self.temp_dir)
        self.hardware_detector = Mock(spec=HardwareDetector)
        self.hardware_detector.get_best_encoder.return_value = 'libx264'
        self.hardware_detector.has_nvidia_gpu.return_value = False
        
        # Create mock video files
        self.original_path = self.create_mock_video_file('original.mp4')
        self.compressed_path = self.create_mock_video_file('compressed.mp4')
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_mock_video_file(self, filename: str, size_mb: float = 1.0) -> str:
        """Create a mock video file for testing"""
        video_path = os.path.join(self.input_dir, filename)
        
        # Create file with specified size
        file_size = int(size_mb * 1024 * 1024)
        with open(video_path, 'wb') as f:
            f.write(b'mock_video_data' * (file_size // 15 + 1))
        
        return video_path
    
    @patch('src.quality_gates.QualityGates._execute_ffmpeg_with_retry')
    def test_end_to_end_quality_evaluation_success(self, mock_execute):
        """Test successful end-to-end quality evaluation"""
        # Mock successful FFmpeg execution with VMAF and SSIM output
        mock_execute.side_effect = [
            # VMAF execution result
            {
                'success': True,
                'stdout': '',
                'stderr': '{"pooled_metrics": {"vmaf": {"mean": 85.67}}}',
                'returncode': 0,
                'execution_time': 2.5,
                'attempts': 1,
                'error_details': {}
            },
            # SSIM execution result
            {
                'success': True,
                'stdout': '',
                'stderr': 'n:1 Y:0.9876 U:0.9654 V:0.9543 All:0.9691 (23.12dB)\nn:2 Y:0.9845 U:0.9632 V:0.9521 All:0.9666 (22.98dB)',
                'returncode': 0,
                'execution_time': 1.8,
                'attempts': 1,
                'error_details': {}
            }
        ]
        
        quality_gates = QualityGates(self.config_manager)
        
        # Mock VMAF availability
        with patch.object(quality_gates, 'check_vmaf_available', return_value=True):
            result = quality_gates.evaluate_quality(
                self.original_path, self.compressed_path,
                vmaf_threshold=80.0, ssim_threshold=0.94
            )
        
        # Verify successful evaluation
        self.assertTrue(result['evaluation_success'])
        self.assertEqual(result['method'], 'vmaf+ssim')
        self.assertIsNotNone(result['vmaf_score'])
        self.assertIsNotNone(result['ssim_score'])
        self.assertAlmostEqual(result['vmaf_score'], 85.67, places=2)
        self.assertGreater(result['ssim_score'], 0.98)
        self.assertTrue(result['passes'])  # Both scores above thresholds
        self.assertGreater(result['confidence'], 0.8)
        
        # Verify detailed information
        self.assertIn('vmaf_pass', result['details'])
        self.assertIn('ssim_pass', result['details'])
        self.assertTrue(result['details']['vmaf_pass'])
        self.assertTrue(result['details']['ssim_pass'])
    
    @patch('src.quality_gates.QualityGates._execute_ffmpeg_with_retry')
    def test_end_to_end_quality_evaluation_failure(self, mock_execute):
        """Test quality evaluation with failing scores"""
        # Mock FFmpeg execution with low quality scores
        mock_execute.side_effect = [
            # VMAF execution result - low score
            {
                'success': True,
                'stdout': '',
                'stderr': '{"pooled_metrics": {"vmaf": {"mean": 65.23}}}',
                'returncode': 0,
                'execution_time': 2.1,
                'attempts': 1,
                'error_details': {}
            },
            # SSIM execution result - low score
            {
                'success': True,
                'stdout': '',
                'stderr': 'n:1 Y:0.8876 U:0.8654 V:0.8543 All:0.8691 (18.12dB)\nn:2 Y:0.8845 U:0.8632 V:0.8521 All:0.8666 (17.98dB)',
                'returncode': 0,
                'execution_time': 1.9,
                'attempts': 1,
                'error_details': {}
            }
        ]
        
        quality_gates = QualityGates(self.config_manager)
        
        # Mock VMAF availability
        with patch.object(quality_gates, 'check_vmaf_available', return_value=True):
            result = quality_gates.evaluate_quality(
                self.original_path, self.compressed_path,
                vmaf_threshold=80.0, ssim_threshold=0.94
            )
        
        # Verify evaluation completed but quality failed
        self.assertTrue(result['evaluation_success'])
        self.assertEqual(result['method'], 'vmaf+ssim')
        self.assertIsNotNone(result['vmaf_score'])
        self.assertIsNotNone(result['ssim_score'])
        self.assertAlmostEqual(result['vmaf_score'], 65.23, places=2)
        self.assertLess(result['ssim_score'], 0.94)
        self.assertFalse(result['passes'])  # Both scores below thresholds
        
        # Verify individual pass/fail status
        self.assertFalse(result['details']['vmaf_pass'])
        self.assertFalse(result['details']['ssim_pass'])
    
    @patch('src.quality_gates.QualityGates._execute_ffmpeg_with_retry')
    def test_ssim_only_fallback_integration(self, mock_execute):
        """Test SSIM-only fallback when VMAF is unavailable"""
        # Mock SSIM execution result
        mock_execute.return_value = {
            'success': True,
            'stdout': '',
            'stderr': 'n:1 Y:0.9576 U:0.9454 V:0.9343 All:0.9491 (21.12dB)\nn:2 Y:0.9545 U:0.9432 V:0.9321 All:0.9466 (20.98dB)',
            'returncode': 0,
            'execution_time': 1.5,
            'attempts': 1,
            'error_details': {}
        }
        
        quality_gates = QualityGates(self.config_manager)
        
        # Mock VMAF unavailable
        with patch.object(quality_gates, 'check_vmaf_available', return_value=False):
            result = quality_gates.evaluate_quality(
                self.original_path, self.compressed_path,
                vmaf_threshold=80.0, ssim_threshold=0.94
            )
        
        # Verify SSIM-only evaluation
        self.assertTrue(result['evaluation_success'])
        self.assertEqual(result['method'], 'ssim_only')
        self.assertIsNone(result['vmaf_score'])
        self.assertIsNotNone(result['ssim_score'])
        self.assertGreater(result['ssim_score'], 0.95)
        self.assertTrue(result['passes'])  # SSIM above threshold
        self.assertGreater(result['confidence'], 0.7)  # Good confidence for SSIM-only
        
        # Verify no VMAF details
        self.assertNotIn('vmaf_pass', result['details'])
        self.assertIn('ssim_pass', result['details'])
        self.assertTrue(result['details']['ssim_pass'])


class TestQualityGateDecisionMaking(unittest.TestCase):
    """Test quality gate decision making with various scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        self.quality_gates = QualityGates(self.config_manager)
        
        # Create mock video files
        self.original_path = os.path.join(self.temp_dir, 'original.mp4')
        self.compressed_path = os.path.join(self.temp_dir, 'compressed.mp4')
        
        with open(self.original_path, 'wb') as f:
            f.write(b'mock_original_data')
        with open(self.compressed_path, 'wb') as f:
            f.write(b'mock_compressed_data')
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_quality_gate_decision_both_pass(self):
        """Test quality gate decision when both VMAF and SSIM pass"""
        # Create result where both metrics pass
        result = {
            'vmaf_score': 85.0,
            'ssim_score': 0.96,
            'passes': False,  # Will be calculated
            'method': 'vmaf+ssim',
            'confidence': 0.9,
            'evaluation_success': True,
            'details': {}
        }
        
        # Test with thresholds
        vmaf_threshold = 80.0
        ssim_threshold = 0.94
        
        # Calculate passes status
        vmaf_pass = self.quality_gates._safe_score_comparison(result['vmaf_score'], vmaf_threshold)
        ssim_pass = self.quality_gates._safe_score_comparison(result['ssim_score'], ssim_threshold)
        result['passes'] = vmaf_pass and ssim_pass
        
        self.assertTrue(result['passes'])
        self.assertTrue(vmaf_pass)
        self.assertTrue(ssim_pass)
    
    def test_quality_gate_decision_vmaf_fail(self):
        """Test quality gate decision when VMAF fails but SSIM passes"""
        # Create result where VMAF fails, SSIM passes
        result = {
            'vmaf_score': 75.0,  # Below threshold
            'ssim_score': 0.96,  # Above threshold
            'passes': False,
            'method': 'vmaf+ssim',
            'confidence': 0.8,
            'evaluation_success': True,
            'details': {}
        }
        
        vmaf_threshold = 80.0
        ssim_threshold = 0.94
        
        vmaf_pass = self.quality_gates._safe_score_comparison(result['vmaf_score'], vmaf_threshold)
        ssim_pass = self.quality_gates._safe_score_comparison(result['ssim_score'], ssim_threshold)
        result['passes'] = vmaf_pass and ssim_pass
        
        self.assertFalse(result['passes'])  # Should fail overall
        self.assertFalse(vmaf_pass)
        self.assertTrue(ssim_pass)
    
    def test_quality_gate_decision_ssim_fail(self):
        """Test quality gate decision when SSIM fails but VMAF passes"""
        # Create result where SSIM fails, VMAF passes
        result = {
            'vmaf_score': 85.0,  # Above threshold
            'ssim_score': 0.92,  # Below threshold
            'passes': False,
            'method': 'vmaf+ssim',
            'confidence': 0.8,
            'evaluation_success': True,
            'details': {}
        }
        
        vmaf_threshold = 80.0
        ssim_threshold = 0.94
        
        vmaf_pass = self.quality_gates._safe_score_comparison(result['vmaf_score'], vmaf_threshold)
        ssim_pass = self.quality_gates._safe_score_comparison(result['ssim_score'], ssim_threshold)
        result['passes'] = vmaf_pass and ssim_pass
        
        self.assertFalse(result['passes'])  # Should fail overall
        self.assertTrue(vmaf_pass)
        self.assertFalse(ssim_pass)
    
    def test_quality_gate_decision_with_none_scores(self):
        """Test quality gate decision with None scores"""
        # Test with None VMAF
        result_none_vmaf = {
            'vmaf_score': None,
            'ssim_score': 0.96,
            'passes': False,
            'method': 'vmaf+ssim',
            'confidence': 0.5,
            'evaluation_success': False,
            'details': {}
        }
        
        vmaf_pass = self.quality_gates._safe_score_comparison(result_none_vmaf['vmaf_score'], 80.0)
        ssim_pass = self.quality_gates._safe_score_comparison(result_none_vmaf['ssim_score'], 0.94)
        result_none_vmaf['passes'] = vmaf_pass and ssim_pass
        
        self.assertFalse(result_none_vmaf['passes'])  # Should fail due to None VMAF
        self.assertFalse(vmaf_pass)
        self.assertTrue(ssim_pass)
        
        # Test with None SSIM
        result_none_ssim = {
            'vmaf_score': 85.0,
            'ssim_score': None,
            'passes': False,
            'method': 'vmaf+ssim',
            'confidence': 0.5,
            'evaluation_success': False,
            'details': {}
        }
        
        vmaf_pass = self.quality_gates._safe_score_comparison(result_none_ssim['vmaf_score'], 80.0)
        ssim_pass = self.quality_gates._safe_score_comparison(result_none_ssim['ssim_score'], 0.94)
        result_none_ssim['passes'] = vmaf_pass and ssim_pass
        
        self.assertFalse(result_none_ssim['passes'])  # Should fail due to None SSIM
        self.assertTrue(vmaf_pass)
        self.assertFalse(ssim_pass)
    
    def test_confidence_score_calculation(self):
        """Test confidence score calculation for different scenarios"""
        # High confidence: both metrics available with good parsing
        confidence_high = self.quality_gates._calculate_confidence_score(
            vmaf_score=85.0, ssim_score=0.96,
            vmaf_confidence=1.0, ssim_confidence=1.0,
            method='vmaf+ssim'
        )
        self.assertGreater(confidence_high, 0.8)
        
        # Medium confidence: only VMAF available
        confidence_medium = self.quality_gates._calculate_confidence_score(
            vmaf_score=85.0, ssim_score=None,
            vmaf_confidence=1.0, ssim_confidence=0.0,
            method='vmaf+ssim'
        )
        self.assertGreater(confidence_medium, 0.5)
        self.assertLess(confidence_medium, 0.9)
        
        # Lower confidence: only SSIM available
        confidence_lower = self.quality_gates._calculate_confidence_score(
            vmaf_score=None, ssim_score=0.96,
            vmaf_confidence=0.0, ssim_confidence=1.0,
            method='vmaf+ssim'
        )
        self.assertGreater(confidence_lower, 0.3)
        self.assertLess(confidence_lower, 0.7)
        
        # No confidence: no metrics available
        confidence_none = self.quality_gates._calculate_confidence_score(
            vmaf_score=None, ssim_score=None,
            vmaf_confidence=0.0, ssim_confidence=0.0,
            method='vmaf+ssim'
        )
        self.assertEqual(confidence_none, 0.0)


class TestFallbackBehaviorIntegration(unittest.TestCase):
    """Test fallback behavior integration with compression pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create configurations for different fallback modes
        self.conservative_config = self.create_config('conservative')
        self.strict_config = self.create_config('strict')
        self.permissive_config = self.create_config('permissive')
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_config(self, fallback_mode: str) -> ConfigManager:
        """Create configuration with specified fallback mode"""
        config_dir = os.path.join(self.temp_dir, f'config_{fallback_mode}')
        os.makedirs(config_dir, exist_ok=True)
        
        config_file = os.path.join(config_dir, 'video_compression.yaml')
        config_data = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {'fallback': 'libx264'}
            },
            'quality_evaluation': {
                'fallback_mode': fallback_mode,
                'confidence_thresholds': {
                    'minimum_acceptable': 0.5,
                    'high_confidence': 0.8,
                    'decision_threshold': 0.6
                }
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        return ConfigManager(config_dir)
    
    def test_conservative_fallback_integration(self):
        """Test conservative fallback mode integration"""
        quality_gates = QualityGates(self.conservative_config)
        
        # Create evaluation failure scenario
        failed_result = {
            'vmaf_score': None,
            'ssim_score': None,
            'passes': False,
            'method': 'error',
            'confidence': 0.0,
            'evaluation_success': False,
            'details': {'error': 'FFmpeg execution failed'}
        }
        
        # Apply fallback behavior
        final_result = quality_gates._apply_fallback_behavior(
            failed_result, vmaf_threshold=80.0, ssim_threshold=0.94
        )
        
        # Conservative mode should proceed (pass quality gates)
        self.assertTrue(final_result['passes'])
        self.assertTrue(final_result['details']['fallback_applied'])
        self.assertEqual(final_result['details']['fallback_mode'], 'conservative')
        self.assertEqual(final_result['details']['fallback_reason'], 'evaluation_failed')
    
    def test_strict_fallback_integration(self):
        """Test strict fallback mode integration"""
        quality_gates = QualityGates(self.strict_config)
        
        # Create evaluation failure scenario
        failed_result = {
            'vmaf_score': None,
            'ssim_score': None,
            'passes': True,  # Original was passing
            'method': 'error',
            'confidence': 0.0,
            'evaluation_success': False,
            'details': {'error': 'FFmpeg execution failed'}
        }
        
        # Apply fallback behavior
        final_result = quality_gates._apply_fallback_behavior(
            failed_result, vmaf_threshold=80.0, ssim_threshold=0.94
        )
        
        # Strict mode should fail quality gates on evaluation failure
        self.assertFalse(final_result['passes'])
        self.assertTrue(final_result['details']['fallback_applied'])
        self.assertEqual(final_result['details']['fallback_mode'], 'strict')
        self.assertEqual(final_result['details']['fallback_reason'], 'evaluation_failed')
    
    def test_permissive_fallback_integration(self):
        """Test permissive fallback mode integration"""
        quality_gates = QualityGates(self.permissive_config)
        
        # Create low confidence scenario
        low_confidence_result = {
            'vmaf_score': 85.0,
            'ssim_score': 0.95,
            'passes': True,
            'method': 'vmaf+ssim',
            'confidence': 0.2,  # Below minimum acceptable
            'evaluation_success': True,
            'details': {}
        }
        
        # Apply fallback behavior
        final_result = quality_gates._apply_fallback_behavior(
            low_confidence_result, vmaf_threshold=80.0, ssim_threshold=0.94
        )
        
        # Permissive mode should pass on low confidence
        self.assertTrue(final_result['passes'])
        self.assertTrue(final_result['details']['fallback_applied'])
        self.assertEqual(final_result['details']['fallback_mode'], 'permissive')
        self.assertEqual(final_result['details']['fallback_reason'], 'low_confidence')
    
    def test_no_fallback_needed_integration(self):
        """Test integration when no fallback is needed"""
        quality_gates = QualityGates(self.conservative_config)
        
        # Create successful evaluation with high confidence
        successful_result = {
            'vmaf_score': 85.0,
            'ssim_score': 0.96,
            'passes': True,
            'method': 'vmaf+ssim',
            'confidence': 0.9,  # High confidence
            'evaluation_success': True,
            'details': {}
        }
        
        # Apply fallback behavior
        final_result = quality_gates._apply_fallback_behavior(
            successful_result, vmaf_threshold=80.0, ssim_threshold=0.94
        )
        
        # No fallback should be applied
        self.assertTrue(final_result['passes'])
        self.assertFalse(final_result['details']['fallback_applied'])
        self.assertEqual(final_result['vmaf_score'], 85.0)
        self.assertEqual(final_result['ssim_score'], 0.96)


class TestPerformanceAndTimeoutHandling(unittest.TestCase):
    """Test performance and timeout handling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create config with short timeouts for testing
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        config_data = {
            'quality_evaluation': {
                'ffmpeg_execution': {
                    'timeout_seconds': 2,  # Very short for testing
                    'retry_attempts': 2
                },
                'logging': {
                    'log_performance_metrics': True
                }
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        self.config_manager = ConfigManager(self.temp_dir)
        self.quality_gates = QualityGates(self.config_manager)
        
        # Create mock video files
        self.original_path = os.path.join(self.temp_dir, 'original.mp4')
        self.compressed_path = os.path.join(self.temp_dir, 'compressed.mp4')
        
        with open(self.original_path, 'wb') as f:
            f.write(b'mock_original_data')
        with open(self.compressed_path, 'wb') as f:
            f.write(b'mock_compressed_data')
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    @patch('subprocess.run')
    def test_timeout_handling(self, mock_subprocess):
        """Test FFmpeg timeout handling"""
        # Mock subprocess to raise TimeoutExpired
        mock_subprocess.side_effect = subprocess.TimeoutExpired(['ffmpeg'], timeout=2)
        
        # Execute FFmpeg command that should timeout
        result = self.quality_gates._execute_ffmpeg_with_retry(
            ['ffmpeg', '-i', 'test.mp4'], 'test operation'
        )
        
        # Should handle timeout gracefully
        self.assertFalse(result['success'])
        self.assertEqual(result['error_details']['category'], 'timeout')
        self.assertTrue(result['error_details']['is_retryable'])
        self.assertGreater(result['execution_time'], 0)
        self.assertGreater(result['attempts'], 0)
    
    @patch('subprocess.run')
    def test_retry_logic_with_transient_errors(self, mock_subprocess):
        """Test retry logic with transient errors"""
        # Mock subprocess to fail first attempt, succeed second
        mock_subprocess.side_effect = [
            # First attempt fails
            Mock(returncode=1, stdout='', stderr='resource temporarily unavailable'),
            # Second attempt succeeds
            Mock(returncode=0, stdout='success', stderr='')
        ]
        
        result = self.quality_gates._execute_ffmpeg_with_retry(
            ['ffmpeg', '-i', 'test.mp4'], 'test operation'
        )
        
        # Should succeed on retry
        self.assertTrue(result['success'])
        self.assertEqual(result['attempts'], 2)
        self.assertEqual(result['stdout'], 'success')
    
    @patch('subprocess.run')
    def test_non_retryable_error_handling(self, mock_subprocess):
        """Test handling of non-retryable errors"""
        # Mock subprocess with non-retryable error
        mock_subprocess.return_value = Mock(
            returncode=2, stdout='', stderr='invalid argument'
        )
        
        result = self.quality_gates._execute_ffmpeg_with_retry(
            ['ffmpeg', '-i', 'test.mp4'], 'test operation'
        )
        
        # Should not retry non-retryable errors
        self.assertFalse(result['success'])
        self.assertEqual(result['attempts'], 1)  # Only one attempt
        self.assertEqual(result['error_details']['category'], 'invalid_arguments')
        self.assertFalse(result['error_details']['is_retryable'])
    
    def test_ffmpeg_error_analysis(self):
        """Test FFmpeg error analysis and categorization"""
        # Test file not found error
        error_analysis = self.quality_gates._analyze_ffmpeg_error(
            'No such file or directory', 1
        )
        self.assertEqual(error_analysis['category'], 'file_not_found')
        self.assertFalse(error_analysis['is_retryable'])
        
        # Test resource busy error
        error_analysis = self.quality_gates._analyze_ffmpeg_error(
            'resource temporarily unavailable', 1
        )
        self.assertEqual(error_analysis['category'], 'resource_busy')
        self.assertTrue(error_analysis['is_retryable'])
        
        # Test VMAF filter error
        error_analysis = self.quality_gates._analyze_ffmpeg_error(
            'libvmaf filter failed', 1
        )
        self.assertEqual(error_analysis['category'], 'vmaf_filter_error')
        self.assertTrue(error_analysis['is_retryable'])
        
        # Test unknown error
        error_analysis = self.quality_gates._analyze_ffmpeg_error(
            'unknown error message', 99
        )
        self.assertEqual(error_analysis['category'], 'unknown_error')
        self.assertTrue(error_analysis['is_retryable'])
    
    @patch('time.time')
    def test_performance_metrics_logging(self, mock_time):
        """Test performance metrics logging"""
        # Mock time progression
        mock_time.side_effect = [0.0, 2.5]  # 2.5 second execution
        
        # Test performance logging
        self.quality_gates._log_performance_metrics(
            'Test operation', 2.5, {'metric1': 'value1', 'metric2': 'value2'}
        )
        
        # Should not raise exceptions (actual logging verification would require log capture)
        self.assertTrue(True)
    
    def test_diagnostic_output_creation(self):
        """Test diagnostic output creation for troubleshooting"""
        # Create mock execution result
        exec_result = {
            'success': False,
            'stdout': 'test stdout',
            'stderr': 'test stderr with error details',
            'returncode': 1,
            'execution_time': 3.2,
            'attempts': 2,
            'error_details': {
                'category': 'generic_ffmpeg_error',
                'description': 'Test error'
            }
        }
        
        cmd = ['ffmpeg', '-i', 'input.mp4', '-f', 'null', '-']
        
        diagnostic = self.quality_gates._create_diagnostic_output(
            'VMAF computation', cmd, exec_result
        )
        
        # Verify diagnostic structure
        self.assertEqual(diagnostic['operation'], 'VMAF computation')
        self.assertEqual(diagnostic['command'], 'ffmpeg -i input.mp4 -f null -')
        self.assertFalse(diagnostic['success'])
        self.assertEqual(diagnostic['execution_time'], 3.2)
        self.assertEqual(diagnostic['attempts'], 2)
        self.assertEqual(diagnostic['returncode'], 1)
        self.assertIn('error_details', diagnostic)


if __name__ == '__main__':
    unittest.main()