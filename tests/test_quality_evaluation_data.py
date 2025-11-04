"""
Test data and validation scenarios for quality evaluation
Generates test videos, FFmpeg output samples, edge cases, and automated validation
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import yaml
import json
import subprocess
import random
import string
from typing import Dict, List, Any, Optional
from src.config_manager import ConfigManager
from src.quality_gates import QualityGates


class TestVideoGenerator:
    """Generate test videos with known quality characteristics"""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.video_dir = os.path.join(temp_dir, 'test_videos')
        os.makedirs(self.video_dir, exist_ok=True)
    
    def create_test_video(self, filename: str, duration: int = 10, 
                         width: int = 1280, height: int = 720, 
                         fps: int = 30, quality: str = 'high') -> str:
        """Create a test video file with specified characteristics"""
        video_path = os.path.join(self.video_dir, filename)
        
        # Create mock video data based on parameters
        # In a real implementation, this would use FFmpeg to generate actual video
        file_size = self._calculate_expected_size(duration, width, height, fps, quality)
        
        with open(video_path, 'wb') as f:
            # Write header-like data
            f.write(b'MOCK_VIDEO_HEADER')
            f.write(f'duration={duration};width={width};height={height};fps={fps};quality={quality}'.encode())
            f.write(b'\x00' * 100)  # Padding
            
            # Write mock video data
            remaining_size = file_size - f.tell()
            if remaining_size > 0:
                chunk_size = min(8192, remaining_size)
                while remaining_size > 0:
                    write_size = min(chunk_size, remaining_size)
                    f.write(b'mock_video_data' * (write_size // 15 + 1)[:write_size])
                    remaining_size -= write_size
        
        return video_path
    
    def _calculate_expected_size(self, duration: int, width: int, height: int, 
                               fps: int, quality: str) -> int:
        """Calculate expected file size based on video parameters"""
        # Rough estimation for mock purposes
        pixels_per_frame = width * height
        frames_total = duration * fps
        
        quality_multipliers = {
            'high': 0.5,
            'medium': 0.3,
            'low': 0.1,
            'very_low': 0.05
        }
        
        multiplier = quality_multipliers.get(quality, 0.3)
        estimated_size = int(pixels_per_frame * frames_total * multiplier * 0.001)  # Rough bytes
        
        return max(1024, estimated_size)  # Minimum 1KB
    
    def create_video_set_with_known_quality(self) -> Dict[str, str]:
        """Create a set of videos with known quality relationships"""
        videos = {}
        
        # Original high quality video
        videos['original_hq'] = self.create_test_video(
            'original_high_quality.mp4', 
            duration=30, width=1920, height=1080, fps=30, quality='high'
        )
        
        # Good quality compressed version (should pass quality gates)
        videos['compressed_good'] = self.create_test_video(
            'compressed_good_quality.mp4',
            duration=30, width=1920, height=1080, fps=30, quality='medium'
        )
        
        # Poor quality compressed version (should fail quality gates)
        videos['compressed_poor'] = self.create_test_video(
            'compressed_poor_quality.mp4',
            duration=30, width=1920, height=1080, fps=30, quality='very_low'
        )
        
        # Different resolution version
        videos['compressed_720p'] = self.create_test_video(
            'compressed_720p.mp4',
            duration=30, width=1280, height=720, fps=30, quality='medium'
        )
        
        # Different frame rate version
        videos['compressed_15fps'] = self.create_test_video(
            'compressed_15fps.mp4',
            duration=30, width=1920, height=1080, fps=15, quality='medium'
        )
        
        return videos


class FFmpegOutputSampleGenerator:
    """Generate FFmpeg output samples for parsing tests"""
    
    @staticmethod
    def generate_vmaf_json_output(vmaf_score: float, frame_count: int = 100) -> str:
        """Generate realistic VMAF JSON output"""
        # Generate per-frame scores around the target score
        frame_scores = []
        for i in range(frame_count):
            # Add some variance around the target score
            variance = random.uniform(-5, 5)
            frame_score = max(0, min(100, vmaf_score + variance))
            frame_scores.append({
                "frameNum": i,
                "metrics": {
                    "vmaf": frame_score,
                    "psnr": random.uniform(20, 40),
                    "ssim": random.uniform(0.8, 1.0)
                }
            })
        
        # Create pooled metrics
        pooled_metrics = {
            "pooled_metrics": {
                "vmaf": {
                    "mean": vmaf_score,
                    "std": random.uniform(1, 5),
                    "min": min(frame['metrics']['vmaf'] for frame in frame_scores),
                    "max": max(frame['metrics']['vmaf'] for frame in frame_scores)
                }
            }
        }
        
        # Return JSON output as it would appear in stderr
        output_lines = []
        for frame in frame_scores[:5]:  # Include first 5 frames
            output_lines.append(json.dumps(frame))
        output_lines.append(json.dumps(pooled_metrics))
        
        return '\n'.join(output_lines)
    
    @staticmethod
    def generate_vmaf_text_output(vmaf_score: float) -> str:
        """Generate realistic VMAF text output"""
        return f"""
[libvmaf @ 0x7f8b8c000000] VMAF score: {vmaf_score:.2f}
[libvmaf @ 0x7f8b8c000000] VMAF mean: {vmaf_score:.2f}
[libvmaf @ 0x7f8b8c000000] VMAF std: {random.uniform(1, 5):.2f}
"""
    
    @staticmethod
    def generate_ssim_output(ssim_score: float, frame_count: int = 50) -> str:
        """Generate realistic SSIM output"""
        output_lines = []
        
        for i in range(frame_count):
            # Generate Y, U, V values around the target score
            y_score = max(0, min(1, ssim_score + random.uniform(-0.02, 0.02)))
            u_score = max(0, min(1, ssim_score + random.uniform(-0.05, 0.05)))
            v_score = max(0, min(1, ssim_score + random.uniform(-0.05, 0.05)))
            all_score = (y_score + u_score + v_score) / 3
            db_value = -10 * (1 - all_score) * 50  # Rough dB calculation
            
            output_lines.append(
                f"n:{i+1} Y:{y_score:.6f} U:{u_score:.6f} V:{v_score:.6f} "
                f"All:{all_score:.6f} ({db_value:.2f}dB)"
            )
        
        return '\n'.join(output_lines)
    
    @staticmethod
    def generate_corrupted_vmaf_output() -> str:
        """Generate corrupted/malformed VMAF output for error testing"""
        return """
[libvmaf @ 0x7f8b8c000000] Error: Invalid input
{"incomplete": json object
VMAF score: invalid_number
Random text that doesn't contain scores
"""
    
    @staticmethod
    def generate_corrupted_ssim_output() -> str:
        """Generate corrupted/malformed SSIM output for error testing"""
        return """
n:1 Y:invalid U:0.9654 V:0.9543 All:0.9691 (23.12dB)
Corrupted line with no proper format
n:2 Y:1.5 U:-0.5 V:2.0 All:0.9666 (22.98dB)
Random text without numbers
"""
    
    @staticmethod
    def generate_empty_output() -> str:
        """Generate empty output for testing"""
        return ""
    
    @staticmethod
    def generate_mixed_format_output(vmaf_score: float, ssim_score: float) -> str:
        """Generate mixed format output with both VMAF and SSIM data"""
        vmaf_json = FFmpegOutputSampleGenerator.generate_vmaf_json_output(vmaf_score, 10)
        ssim_text = FFmpegOutputSampleGenerator.generate_ssim_output(ssim_score, 10)
        
        return f"""
FFmpeg processing started...
{vmaf_json}
Processing SSIM...
{ssim_text}
Processing complete.
"""


class EdgeCaseScenarioGenerator:
    """Generate edge case test scenarios"""
    
    def __init__(self, temp_dir: str):
        self.temp_dir = temp_dir
        self.video_generator = TestVideoGenerator(temp_dir)
    
    def create_edge_case_videos(self) -> Dict[str, str]:
        """Create videos for edge case testing"""
        edge_cases = {}
        
        # Very short video (1 second)
        edge_cases['very_short'] = self.video_generator.create_test_video(
            'very_short.mp4', duration=1, width=640, height=480, fps=30
        )
        
        # Very long video (1 hour)
        edge_cases['very_long'] = self.video_generator.create_test_video(
            'very_long.mp4', duration=3600, width=1280, height=720, fps=24
        )
        
        # Unusual resolution (not 16:9)
        edge_cases['unusual_resolution'] = self.video_generator.create_test_video(
            'unusual_resolution.mp4', duration=10, width=1000, height=600, fps=25
        )
        
        # Very low resolution
        edge_cases['very_low_res'] = self.video_generator.create_test_video(
            'very_low_res.mp4', duration=10, width=160, height=120, fps=15
        )
        
        # Very high resolution (4K)
        edge_cases['very_high_res'] = self.video_generator.create_test_video(
            'very_high_res.mp4', duration=5, width=3840, height=2160, fps=30
        )
        
        # Unusual frame rate
        edge_cases['unusual_fps'] = self.video_generator.create_test_video(
            'unusual_fps.mp4', duration=10, width=1280, height=720, fps=48
        )
        
        # Zero-byte file (corrupted)
        corrupted_path = os.path.join(self.video_generator.video_dir, 'corrupted.mp4')
        with open(corrupted_path, 'wb') as f:
            pass  # Create empty file
        edge_cases['corrupted_zero_bytes'] = corrupted_path
        
        # File with invalid header
        invalid_header_path = os.path.join(self.video_generator.video_dir, 'invalid_header.mp4')
        with open(invalid_header_path, 'wb') as f:
            f.write(b'INVALID_VIDEO_HEADER' + b'\x00' * 1000)
        edge_cases['invalid_header'] = invalid_header_path
        
        return edge_cases
    
    def create_problematic_filenames(self) -> Dict[str, str]:
        """Create files with problematic names for testing"""
        problematic = {}
        
        # Filename with spaces
        problematic['spaces'] = self.video_generator.create_test_video(
            'file with spaces.mp4', duration=5
        )
        
        # Filename with special characters
        problematic['special_chars'] = self.video_generator.create_test_video(
            'file-with_special[chars].mp4', duration=5
        )
        
        # Very long filename
        long_name = 'very_long_filename_' + 'x' * 200 + '.mp4'
        try:
            problematic['long_name'] = self.video_generator.create_test_video(
                long_name, duration=5
            )
        except OSError:
            # If OS doesn't support long filenames, create a shorter alternative
            problematic['long_name'] = self.video_generator.create_test_video(
                'long_filename_alternative.mp4', duration=5
            )
        
        return problematic


class TestQualityEvaluationWithTestData(unittest.TestCase):
    """Test quality evaluation with generated test data"""
    
    def setUp(self):
        """Set up test fixtures with generated data"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        self.quality_gates = QualityGates(self.config_manager)
        
        # Generate test data
        self.video_generator = TestVideoGenerator(self.temp_dir)
        self.ffmpeg_generator = FFmpegOutputSampleGenerator()
        self.edge_case_generator = EdgeCaseScenarioGenerator(self.temp_dir)
        
        # Create test video sets
        self.known_quality_videos = self.video_generator.create_video_set_with_known_quality()
        self.edge_case_videos = self.edge_case_generator.create_edge_case_videos()
        self.problematic_filenames = self.edge_case_generator.create_problematic_filenames()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_vmaf_parsing_with_generated_samples(self):
        """Test VMAF parsing with various generated output samples"""
        test_cases = [
            (85.67, 'high_quality_vmaf'),
            (65.23, 'medium_quality_vmaf'),
            (45.12, 'low_quality_vmaf'),
            (95.89, 'very_high_quality_vmaf'),
            (25.34, 'very_low_quality_vmaf')
        ]
        
        for expected_score, test_name in test_cases:
            with self.subTest(test_name=test_name, expected_score=expected_score):
                # Test JSON format
                json_output = self.ffmpeg_generator.generate_vmaf_json_output(expected_score)
                parsed_score = self.quality_gates._parse_vmaf_output(json_output)
                
                self.assertIsNotNone(parsed_score, f"Failed to parse JSON VMAF output for {test_name}")
                self.assertAlmostEqual(parsed_score, expected_score, places=1,
                                     msg=f"VMAF score mismatch for {test_name}")
                
                # Test text format
                text_output = self.ffmpeg_generator.generate_vmaf_text_output(expected_score)
                parsed_score_text = self.quality_gates._parse_vmaf_output(text_output)
                
                self.assertIsNotNone(parsed_score_text, f"Failed to parse text VMAF output for {test_name}")
                self.assertAlmostEqual(parsed_score_text, expected_score, places=1,
                                     msg=f"VMAF text score mismatch for {test_name}")
    
    def test_ssim_parsing_with_generated_samples(self):
        """Test SSIM parsing with various generated output samples"""
        test_cases = [
            (0.9876, 'high_quality_ssim'),
            (0.9234, 'medium_quality_ssim'),
            (0.8567, 'low_quality_ssim'),
            (0.9945, 'very_high_quality_ssim'),
            (0.7823, 'very_low_quality_ssim')
        ]
        
        for expected_score, test_name in test_cases:
            with self.subTest(test_name=test_name, expected_score=expected_score):
                ssim_output = self.ffmpeg_generator.generate_ssim_output(expected_score)
                parsed_score, confidence = self.quality_gates._parse_ssim_output(ssim_output)
                
                self.assertIsNotNone(parsed_score, f"Failed to parse SSIM output for {test_name}")
                self.assertAlmostEqual(parsed_score, expected_score, places=2,
                                     msg=f"SSIM score mismatch for {test_name}")
                self.assertGreater(confidence, 0.5, f"Low confidence for {test_name}")
    
    def test_corrupted_output_handling(self):
        """Test handling of corrupted FFmpeg output"""
        # Test corrupted VMAF output
        corrupted_vmaf = self.ffmpeg_generator.generate_corrupted_vmaf_output()
        vmaf_score = self.quality_gates._parse_vmaf_output(corrupted_vmaf)
        self.assertIsNone(vmaf_score, "Should return None for corrupted VMAF output")
        
        # Test corrupted SSIM output
        corrupted_ssim = self.ffmpeg_generator.generate_corrupted_ssim_output()
        ssim_score, confidence = self.quality_gates._parse_ssim_output(corrupted_ssim)
        # May return None or a partial score, but should not crash
        if ssim_score is not None:
            self.assertGreaterEqual(ssim_score, 0.0)
            self.assertLessEqual(ssim_score, 1.0)
        
        # Test empty output
        empty_output = self.ffmpeg_generator.generate_empty_output()
        vmaf_empty = self.quality_gates._parse_vmaf_output(empty_output)
        ssim_empty, conf_empty = self.quality_gates._parse_ssim_output(empty_output)
        
        self.assertIsNone(vmaf_empty, "Should return None for empty VMAF output")
        self.assertIsNone(ssim_empty, "Should return None for empty SSIM output")
        self.assertEqual(conf_empty, 0.0, "Should return zero confidence for empty output")
    
    def test_edge_case_video_handling(self):
        """Test quality evaluation with edge case videos"""
        for case_name, video_path in self.edge_case_videos.items():
            with self.subTest(case_name=case_name):
                # Test that evaluation doesn't crash with edge case videos
                try:
                    # Mock FFmpeg execution to avoid actual processing
                    with patch.object(self.quality_gates, '_execute_ffmpeg_with_retry') as mock_exec:
                        mock_exec.return_value = {
                            'success': False,
                            'stdout': '',
                            'stderr': 'Mock error for edge case testing',
                            'returncode': 1,
                            'execution_time': 1.0,
                            'attempts': 1,
                            'error_details': {
                                'category': 'generic_ffmpeg_error',
                                'is_retryable': False,
                                'description': 'Mock error'
                            }
                        }
                        
                        result = self.quality_gates.evaluate_quality(
                            video_path, video_path  # Use same file as both original and compressed
                        )
                        
                        # Should handle gracefully without crashing
                        self.assertIsInstance(result, dict, f"Should return dict for {case_name}")
                        self.assertIn('evaluation_success', result, f"Should have evaluation_success for {case_name}")
                        self.assertIn('method', result, f"Should have method for {case_name}")
                        
                except Exception as e:
                    self.fail(f"Quality evaluation crashed with edge case {case_name}: {e}")
    
    def test_problematic_filename_handling(self):
        """Test quality evaluation with problematic filenames"""
        for case_name, video_path in self.problematic_filenames.items():
            with self.subTest(case_name=case_name):
                # Test that evaluation handles problematic filenames
                try:
                    # Mock FFmpeg execution
                    with patch.object(self.quality_gates, '_execute_ffmpeg_with_retry') as mock_exec:
                        mock_exec.return_value = {
                            'success': True,
                            'stdout': '',
                            'stderr': self.ffmpeg_generator.generate_ssim_output(0.95),
                            'returncode': 0,
                            'execution_time': 1.5,
                            'attempts': 1,
                            'error_details': {}
                        }
                        
                        # Mock VMAF unavailable to test SSIM-only path
                        with patch.object(self.quality_gates, 'check_vmaf_available', return_value=False):
                            result = self.quality_gates.evaluate_quality(
                                video_path, video_path
                            )
                            
                            # Should handle gracefully
                            self.assertIsInstance(result, dict, f"Should return dict for {case_name}")
                            self.assertTrue(result.get('evaluation_success', False), 
                                          f"Should succeed with {case_name}")
                            
                except Exception as e:
                    self.fail(f"Quality evaluation failed with problematic filename {case_name}: {e}")


class TestAutomatedQualityValidation(unittest.TestCase):
    """Automated quality evaluation validation"""
    
    def setUp(self):
        """Set up automated validation fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        self.quality_gates = QualityGates(self.config_manager)
        self.ffmpeg_generator = FFmpegOutputSampleGenerator()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_quality_threshold_validation(self):
        """Test that quality thresholds work correctly"""
        test_scenarios = [
            # (vmaf_score, ssim_score, vmaf_threshold, ssim_threshold, should_pass)
            (85.0, 0.96, 80.0, 0.94, True),   # Both pass
            (75.0, 0.96, 80.0, 0.94, False),  # VMAF fails
            (85.0, 0.92, 80.0, 0.94, False),  # SSIM fails
            (75.0, 0.92, 80.0, 0.94, False),  # Both fail
            (80.0, 0.94, 80.0, 0.94, True),   # Exactly at thresholds
            (79.9, 0.939, 80.0, 0.94, False), # Just below thresholds
        ]
        
        for vmaf_score, ssim_score, vmaf_thresh, ssim_thresh, should_pass in test_scenarios:
            with self.subTest(vmaf=vmaf_score, ssim=ssim_score, 
                            vmaf_thresh=vmaf_thresh, ssim_thresh=ssim_thresh):
                
                # Mock FFmpeg execution with specific scores
                with patch.object(self.quality_gates, '_execute_ffmpeg_with_retry') as mock_exec:
                    mock_exec.side_effect = [
                        # VMAF result
                        {
                            'success': True,
                            'stdout': '',
                            'stderr': self.ffmpeg_generator.generate_vmaf_json_output(vmaf_score),
                            'returncode': 0,
                            'execution_time': 1.0,
                            'attempts': 1,
                            'error_details': {}
                        },
                        # SSIM result
                        {
                            'success': True,
                            'stdout': '',
                            'stderr': self.ffmpeg_generator.generate_ssim_output(ssim_score),
                            'returncode': 0,
                            'execution_time': 1.0,
                            'attempts': 1,
                            'error_details': {}
                        }
                    ]
                    
                    # Create mock video files
                    original_path = os.path.join(self.temp_dir, 'original.mp4')
                    compressed_path = os.path.join(self.temp_dir, 'compressed.mp4')
                    
                    with open(original_path, 'wb') as f:
                        f.write(b'mock_original')
                    with open(compressed_path, 'wb') as f:
                        f.write(b'mock_compressed')
                    
                    # Mock VMAF availability
                    with patch.object(self.quality_gates, 'check_vmaf_available', return_value=True):
                        result = self.quality_gates.evaluate_quality(
                            original_path, compressed_path,
                            vmaf_threshold=vmaf_thresh, ssim_threshold=ssim_thresh
                        )
                    
                    # Verify threshold behavior
                    self.assertEqual(result['passes'], should_pass,
                                   f"Threshold validation failed for VMAF={vmaf_score}, SSIM={ssim_score}")
                    self.assertTrue(result['evaluation_success'],
                                  "Evaluation should succeed even if quality fails")
    
    def test_confidence_score_validation(self):
        """Test confidence score calculation validation"""
        confidence_scenarios = [
            # (vmaf_available, ssim_available, vmaf_conf, ssim_conf, method, expected_min_conf)
            (True, True, 1.0, 1.0, 'vmaf+ssim', 0.8),      # Both available, high confidence
            (True, False, 1.0, 0.0, 'vmaf+ssim', 0.5),     # Only VMAF, medium confidence
            (False, True, 0.0, 1.0, 'vmaf+ssim', 0.3),     # Only SSIM, lower confidence
            (False, False, 0.0, 0.0, 'vmaf+ssim', 0.0),    # Neither available, no confidence
            (False, True, 0.0, 1.0, 'ssim_only', 0.7),     # SSIM-only mode, good confidence
        ]
        
        for vmaf_avail, ssim_avail, vmaf_conf, ssim_conf, method, expected_min in confidence_scenarios:
            with self.subTest(method=method, vmaf_avail=vmaf_avail, ssim_avail=ssim_avail):
                vmaf_score = 85.0 if vmaf_avail else None
                ssim_score = 0.95 if ssim_avail else None
                
                confidence = self.quality_gates._calculate_confidence_score(
                    vmaf_score, ssim_score, vmaf_conf, ssim_conf, method
                )
                
                self.assertGreaterEqual(confidence, expected_min,
                                      f"Confidence too low for scenario: {method}, "
                                      f"VMAF={vmaf_avail}, SSIM={ssim_avail}")
                self.assertLessEqual(confidence, 1.0,
                                   f"Confidence too high for scenario: {method}")
    
    def test_fallback_behavior_validation(self):
        """Test fallback behavior validation across different modes"""
        fallback_scenarios = [
            # (mode, evaluation_success, confidence, original_passes, expected_final_passes)
            ('conservative', False, 0.0, False, True),   # Conservative: pass on failure
            ('conservative', True, 0.2, False, True),    # Conservative: pass on low confidence
            ('strict', False, 0.0, True, False),         # Strict: fail on evaluation failure
            ('strict', True, 0.2, True, False),          # Strict: fail on low confidence
            ('permissive', False, 0.0, False, True),     # Permissive: always pass on failure
            ('permissive', True, 0.2, False, True),      # Permissive: always pass on low confidence
            ('conservative', True, 0.8, False, False),   # High confidence: use original result
            ('strict', True, 0.8, True, True),           # High confidence: use original result
        ]
        
        for mode, eval_success, confidence, orig_passes, expected_passes in fallback_scenarios:
            with self.subTest(mode=mode, eval_success=eval_success, 
                            confidence=confidence, orig_passes=orig_passes):
                
                # Set fallback mode
                self.quality_gates.fallback_mode = mode
                self.quality_gates.confidence_thresholds = {
                    'minimum_acceptable': 0.5,
                    'high_confidence': 0.8,
                    'decision_threshold': 0.6
                }
                
                # Create test result
                result = {
                    'vmaf_score': 85.0 if eval_success else None,
                    'ssim_score': 0.95 if eval_success else None,
                    'passes': orig_passes,
                    'method': 'vmaf+ssim' if eval_success else 'error',
                    'confidence': confidence,
                    'evaluation_success': eval_success,
                    'details': {}
                }
                
                # Apply fallback behavior
                final_result = self.quality_gates._apply_fallback_behavior(
                    result, vmaf_threshold=80.0, ssim_threshold=0.94
                )
                
                # Validate fallback behavior
                self.assertEqual(final_result['passes'], expected_passes,
                               f"Fallback behavior incorrect for mode={mode}, "
                               f"eval_success={eval_success}, confidence={confidence}")
    
    def test_parsing_robustness_validation(self):
        """Test parsing robustness with various malformed inputs"""
        malformed_inputs = [
            # VMAF malformed inputs
            ('vmaf', '{"incomplete": json'),
            ('vmaf', 'VMAF score: not_a_number'),
            ('vmaf', '{"pooled_metrics": {"vmaf": "invalid"}}'),
            ('vmaf', 'Random text with no VMAF data'),
            
            # SSIM malformed inputs
            ('ssim', 'n:1 Y:invalid U:0.9654 V:0.9543'),
            ('ssim', 'Completely invalid SSIM output'),
            ('ssim', 'n:1 Y:1.5 U:-0.5 V:2.0'),  # Out of range values
            ('ssim', ''),  # Empty output
        ]
        
        for metric_type, malformed_input in malformed_inputs:
            with self.subTest(metric_type=metric_type, input_preview=malformed_input[:30]):
                try:
                    if metric_type == 'vmaf':
                        result = self.quality_gates._parse_vmaf_output(malformed_input)
                        # Should return None for malformed input, not crash
                        self.assertIsNone(result, f"Should return None for malformed VMAF input")
                    
                    elif metric_type == 'ssim':
                        result, confidence = self.quality_gates._parse_ssim_output(malformed_input)
                        # Should return None or valid score, not crash
                        if result is not None:
                            self.assertGreaterEqual(result, 0.0, "SSIM score should be >= 0")
                            self.assertLessEqual(result, 1.0, "SSIM score should be <= 1")
                        self.assertGreaterEqual(confidence, 0.0, "Confidence should be >= 0")
                        self.assertLessEqual(confidence, 1.0, "Confidence should be <= 1")
                
                except Exception as e:
                    self.fail(f"Parsing crashed with malformed {metric_type} input: {e}")


if __name__ == '__main__':
    unittest.main()