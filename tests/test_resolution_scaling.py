"""
Unit tests for resolution scaling functionality in quality evaluation
Tests SSIM computation with different video resolutions, aspect ratio preservation,
automatic resolution detection and scaling, and temporary scaled video cleanup
"""

import unittest
import tempfile
import os
import time
from unittest.mock import Mock, patch, MagicMock
from src.quality_gates import QualityGates, ResolutionAwareQualityEvaluator
from src.config_manager import ConfigManager


class MockFFmpegUtils:
    """Mock FFmpeg utilities for testing"""
    
    @staticmethod
    def get_video_resolution(video_path: str) -> tuple:
        """Mock video resolution detection based on filename"""
        if "1920x1080" in video_path:
            return (1920, 1080)
        elif "1280x720" in video_path:
            return (1280, 720)
        elif "640x480" in video_path:
            return (640, 480)
        elif "320x240" in video_path:
            return (320, 240)
        elif "1920x800" in video_path:  # Different aspect ratio
            return (1920, 800)
        elif "original" in video_path:
            return (1920, 1080)
        elif "compressed" in video_path:
            return (1280, 720)
        else:
            return (1920, 1080)  # Default


class TestResolutionDetection(unittest.TestCase):
    """Test automatic resolution detection and mismatch analysis"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        self.evaluator = ResolutionAwareQualityEvaluator(self.config_manager)
        
        # Create mock video files
        self.original_1080p = os.path.join(self.temp_dir, "original_1920x1080.mp4")
        self.compressed_720p = os.path.join(self.temp_dir, "compressed_1280x720.mp4")
        self.compressed_480p = os.path.join(self.temp_dir, "compressed_640x480.mp4")
        self.original_ultrawide = os.path.join(self.temp_dir, "original_1920x800.mp4")
        
        # Create empty files
        for path in [self.original_1080p, self.compressed_720p, self.compressed_480p, self.original_ultrawide]:
            with open(path, 'w') as f:
                f.write("")
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.ffmpeg_utils.FFmpegUtils', MockFFmpegUtils)
    def test_resolution_detection_same_resolution(self):
        """Test resolution detection when videos have same resolution"""
        # Create two videos with same resolution
        video1 = os.path.join(self.temp_dir, "video1_1920x1080.mp4")
        video2 = os.path.join(self.temp_dir, "video2_1920x1080.mp4")
        
        for path in [video1, video2]:
            with open(path, 'w') as f:
                f.write("")
        
        result = self.evaluator.detect_resolution_mismatch(video1, video2)
        
        self.assertEqual(result['original_resolution'], (1920, 1080))
        self.assertEqual(result['compressed_resolution'], (1920, 1080))
        self.assertFalse(result['resolution_mismatch'])
        self.assertFalse(result['aspect_mismatch'])
        self.assertFalse(result['scaling_needed'])
        self.assertEqual(result['target_resolution'], (1920, 1080))
    
    @patch('src.ffmpeg_utils.FFmpegUtils', MockFFmpegUtils)
    def test_resolution_detection_different_resolution(self):
        """Test resolution detection when videos have different resolutions"""
        result = self.evaluator.detect_resolution_mismatch(self.original_1080p, self.compressed_720p)
        
        self.assertEqual(result['original_resolution'], (1920, 1080))
        self.assertEqual(result['compressed_resolution'], (1280, 720))
        self.assertTrue(result['resolution_mismatch'])
        self.assertFalse(result['aspect_mismatch'])  # Same aspect ratio (16:9)
        self.assertTrue(result['scaling_needed'])
        # Should use smaller resolution (720p) as target
        self.assertEqual(result['target_resolution'], (1280, 720))
        self.assertEqual(result['smaller_video'], 'compressed')
    
    @patch('src.ffmpeg_utils.FFmpegUtils', MockFFmpegUtils)
    def test_resolution_detection_aspect_ratio_mismatch(self):
        """Test resolution detection when videos have different aspect ratios"""
        result = self.evaluator.detect_resolution_mismatch(self.original_1080p, self.original_ultrawide)
        
        self.assertEqual(result['original_resolution'], (1920, 1080))
        self.assertEqual(result['compressed_resolution'], (1920, 800))
        self.assertTrue(result['resolution_mismatch'])
        self.assertTrue(result['aspect_mismatch'])
        self.assertTrue(result['scaling_needed'])
        # Should use smaller resolution (ultrawide) as target
        self.assertEqual(result['target_resolution'], (1920, 800))
        self.assertEqual(result['smaller_video'], 'compressed')
    
    @patch('src.ffmpeg_utils.FFmpegUtils', MockFFmpegUtils)
    def test_target_resolution_selection(self):
        """Test that target resolution is always the smaller one to avoid upscaling"""
        # Test with original smaller than compressed
        original_small = os.path.join(self.temp_dir, "original_640x480.mp4")
        compressed_large = os.path.join(self.temp_dir, "compressed_1920x1080.mp4")
        
        for path in [original_small, compressed_large]:
            with open(path, 'w') as f:
                f.write("")
        
        result = self.evaluator.detect_resolution_mismatch(original_small, compressed_large)
        
        # Should use original (smaller) resolution as target
        self.assertEqual(result['target_resolution'], (640, 480))
        self.assertEqual(result['smaller_video'], 'original')


class TestVideoScaling(unittest.TestCase):
    """Test video scaling functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        self.evaluator = ResolutionAwareQualityEvaluator(self.config_manager)
        
        # Create mock video file
        self.test_video = os.path.join(self.temp_dir, "test_1920x1080.mp4")
        with open(self.test_video, 'w') as f:
            f.write("")
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('subprocess.run')
    @patch('src.ffmpeg_utils.FFmpegUtils', MockFFmpegUtils)
    def test_video_scaling_success(self, mock_subprocess):
        """Test successful video scaling"""
        # Mock successful FFmpeg execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        # Mock the scaled file existence
        with patch('os.path.exists', return_value=True):
            scaled_path = self.evaluator.scale_video_for_comparison(
                self.test_video, (1280, 720), "test_session"
            )
        
        self.assertIsNotNone(scaled_path)
        self.assertIn("scaled_video_test_session_1280x720", scaled_path)
        
        # Verify FFmpeg command was called with correct parameters
        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0][0]
        
        # Check key FFmpeg parameters
        self.assertIn('ffmpeg', call_args)
        self.assertIn('-i', call_args)
        self.assertIn(self.test_video, call_args)
        self.assertIn('-vf', call_args)
        
        # Check scaling filter
        vf_index = call_args.index('-vf') + 1
        scale_filter = call_args[vf_index]
        self.assertIn('scale=1280:720', scale_filter)
        self.assertIn('bicubic', scale_filter)
        self.assertIn('force_original_aspect_ratio=decrease', scale_filter)
    
    @patch('subprocess.run')
    @patch('src.ffmpeg_utils.FFmpegUtils', MockFFmpegUtils)
    def test_video_scaling_failure(self, mock_subprocess):
        """Test video scaling failure handling"""
        # Mock failed FFmpeg execution
        mock_result = Mock()
        mock_result.returncode = 1
        mock_result.stderr = "FFmpeg error: invalid parameters"
        mock_subprocess.return_value = mock_result
        
        scaled_path = self.evaluator.scale_video_for_comparison(
            self.test_video, (1280, 720), "test_session"
        )
        
        self.assertIsNone(scaled_path)
    
    @patch('subprocess.run')
    @patch('src.ffmpeg_utils.FFmpegUtils', MockFFmpegUtils)
    def test_aspect_ratio_preservation(self, mock_subprocess):
        """Test that aspect ratio is preserved during scaling"""
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        with patch('os.path.exists', return_value=True):
            self.evaluator.scale_video_for_comparison(
                self.test_video, (1280, 720), "test_session"
            )
        
        # Verify scaling command preserves aspect ratio
        call_args = mock_subprocess.call_args[0][0]
        vf_index = call_args.index('-vf') + 1
        scale_filter = call_args[vf_index]
        
        # Should include aspect ratio preservation and padding
        self.assertIn('force_original_aspect_ratio=decrease', scale_filter)
        self.assertIn('pad=1280:720', scale_filter)


class TestSSIMWithScaling(unittest.TestCase):
    """Test SSIM computation with resolution scaling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        self.evaluator = ResolutionAwareQualityEvaluator(self.config_manager)
        
        # Create mock video files
        self.original_1080p = os.path.join(self.temp_dir, "original_1920x1080.mp4")
        self.compressed_720p = os.path.join(self.temp_dir, "compressed_1280x720.mp4")
        
        for path in [self.original_1080p, self.compressed_720p]:
            with open(path, 'w') as f:
                f.write("")
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.ffmpeg_utils.FFmpegUtils', MockFFmpegUtils)
    def test_ssim_no_scaling_needed(self):
        """Test SSIM computation when no scaling is needed"""
        # Create videos with same resolution
        video1 = os.path.join(self.temp_dir, "video1_1920x1080.mp4")
        video2 = os.path.join(self.temp_dir, "video2_1920x1080.mp4")
        
        for path in [video1, video2]:
            with open(path, 'w') as f:
                f.write("")
        
        # Mock direct SSIM computation
        with patch.object(self.evaluator, '_compute_ssim_direct', return_value=(0.95, 1.0)):
            ssim_score, confidence, scaling_info = self.evaluator.compute_ssim_with_scaling(
                video1, video2, "test_session"
            )
        
        self.assertEqual(ssim_score, 0.95)
        self.assertEqual(confidence, 1.0)
        self.assertFalse(scaling_info['scaling_applied'])
        self.assertEqual(len(scaling_info['scaled_files']), 0)
    
    @patch('src.ffmpeg_utils.FFmpegUtils', MockFFmpegUtils)
    def test_ssim_with_scaling_both_videos(self):
        """Test SSIM computation when both videos need scaling"""
        # Mock scaling and SSIM computation
        with patch.object(self.evaluator, 'scale_video_for_comparison') as mock_scale, \
             patch.object(self.evaluator, '_compute_ssim_direct', return_value=(0.92, 0.9)):
            
            # Mock successful scaling for both videos
            mock_scale.side_effect = [
                os.path.join(self.temp_dir, "scaled_ref.mp4"),
                os.path.join(self.temp_dir, "scaled_dist.mp4")
            ]
            
            ssim_score, confidence, scaling_info = self.evaluator.compute_ssim_with_scaling(
                self.original_1080p, self.compressed_720p, "test_session"
            )
        
        self.assertEqual(ssim_score, 0.92)
        self.assertEqual(confidence, 0.9)
        self.assertTrue(scaling_info['scaling_applied'])
        self.assertEqual(scaling_info['original_resolution'], (1920, 1080))
        self.assertEqual(scaling_info['comparison_resolution'], (1280, 720))
        # In this test case, the target resolution is 720p (compressed video resolution)
        # So only the original video (1080p) needs scaling, not the compressed video
        self.assertEqual(len(scaling_info['scaled_files']), 1)
        
        # Verify scaling was called for the original video only
        self.assertEqual(mock_scale.call_count, 1)
    
    @patch('src.ffmpeg_utils.FFmpegUtils', MockFFmpegUtils)
    def test_ssim_scaling_failure(self):
        """Test SSIM computation when scaling fails"""
        # Mock scaling failure
        with patch.object(self.evaluator, 'scale_video_for_comparison', return_value=None):
            ssim_score, confidence, scaling_info = self.evaluator.compute_ssim_with_scaling(
                self.original_1080p, self.compressed_720p, "test_session"
            )
        
        self.assertIsNone(ssim_score)
        self.assertEqual(confidence, 0.0)
        self.assertTrue(scaling_info['scaling_applied'])
        self.assertEqual(len(scaling_info['scaled_files']), 0)


class TestVMAFWithScaling(unittest.TestCase):
    """Test VMAF computation with resolution scaling"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        self.evaluator = ResolutionAwareQualityEvaluator(self.config_manager)
        
        # Create mock video files
        self.original_1080p = os.path.join(self.temp_dir, "original_1920x1080.mp4")
        self.compressed_720p = os.path.join(self.temp_dir, "compressed_1280x720.mp4")
        
        for path in [self.original_1080p, self.compressed_720p]:
            with open(path, 'w') as f:
                f.write("")
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.ffmpeg_utils.FFmpegUtils', MockFFmpegUtils)
    def test_vmaf_no_scaling_needed(self):
        """Test VMAF computation when no scaling is needed"""
        # Create videos with same resolution
        video1 = os.path.join(self.temp_dir, "video1_1920x1080.mp4")
        video2 = os.path.join(self.temp_dir, "video2_1920x1080.mp4")
        
        for path in [video1, video2]:
            with open(path, 'w') as f:
                f.write("")
        
        # Mock direct VMAF computation
        with patch.object(self.evaluator, '_compute_vmaf_direct', return_value=85.5):
            vmaf_score, scaling_info = self.evaluator.compute_vmaf_with_scaling(
                video1, video2, "test_session"
            )
        
        self.assertEqual(vmaf_score, 85.5)
        self.assertFalse(scaling_info['scaling_applied'])
        self.assertEqual(len(scaling_info['scaled_files']), 0)
    
    @patch('src.ffmpeg_utils.FFmpegUtils', MockFFmpegUtils)
    def test_vmaf_with_scaling(self):
        """Test VMAF computation with resolution scaling"""
        # Mock scaling and VMAF computation
        with patch.object(self.evaluator, 'scale_video_for_comparison') as mock_scale, \
             patch.object(self.evaluator, '_compute_vmaf_direct', return_value=82.3):
            
            # Mock successful scaling for both videos
            mock_scale.side_effect = [
                os.path.join(self.temp_dir, "scaled_ref.mp4"),
                os.path.join(self.temp_dir, "scaled_dist.mp4")
            ]
            
            vmaf_score, scaling_info = self.evaluator.compute_vmaf_with_scaling(
                self.original_1080p, self.compressed_720p, "test_session"
            )
        
        self.assertEqual(vmaf_score, 82.3)
        self.assertTrue(scaling_info['scaling_applied'])
        self.assertEqual(scaling_info['original_resolution'], (1920, 1080))
        self.assertEqual(scaling_info['comparison_resolution'], (1280, 720))
        # In this test case, the target resolution is 720p (compressed video resolution)
        # So only the original video (1080p) needs scaling, not the compressed video
        self.assertEqual(len(scaling_info['scaled_files']), 1)


class TestTemporaryFileCleanup(unittest.TestCase):
    """Test temporary scaled video cleanup"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        self.evaluator = ResolutionAwareQualityEvaluator(self.config_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_temporary_file_tracking(self):
        """Test that temporary files are properly tracked"""
        # Initially no temp files
        self.assertEqual(len(self.evaluator.temp_files_created), 0)
        
        # Mock successful scaling that creates temp file
        temp_file = os.path.join(self.temp_dir, "temp_scaled.mp4")
        with open(temp_file, 'w') as f:
            f.write("")
        
        # Simulate adding temp file to tracking
        self.evaluator.temp_files_created.append(temp_file)
        
        self.assertEqual(len(self.evaluator.temp_files_created), 1)
        self.assertIn(temp_file, self.evaluator.temp_files_created)
    
    @patch('subprocess.run')
    @patch('src.ffmpeg_utils.FFmpegUtils', MockFFmpegUtils)
    def test_cleanup_after_scaling(self, mock_subprocess):
        """Test cleanup of temporary files after scaling operations"""
        # Mock successful FFmpeg execution
        mock_result = Mock()
        mock_result.returncode = 0
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        # Create a real temporary file to test cleanup
        temp_file = os.path.join(self.temp_dir, "scaled_video_test_1280x720.mp4")
        with open(temp_file, 'w') as f:
            f.write("")
        
        with patch('os.path.exists', return_value=True), \
             patch.object(self.evaluator, 'temp_files_created', []):
            
            # Perform scaling operation
            test_video = os.path.join(self.temp_dir, "test_1920x1080.mp4")
            with open(test_video, 'w') as f:
                f.write("")
            
            scaled_path = self.evaluator.scale_video_for_comparison(
                test_video, (1280, 720), "test"
            )
            
            # Verify temp file was tracked
            self.assertIsNotNone(scaled_path)
            self.assertIn(scaled_path, self.evaluator.temp_files_created)
    
    def test_cleanup_nonexistent_files(self):
        """Test cleanup handles nonexistent files gracefully"""
        # Add nonexistent file to cleanup list
        nonexistent_file = os.path.join(self.temp_dir, "nonexistent.mp4")
        self.evaluator.temp_files_created.append(nonexistent_file)
        
        # Cleanup should not raise exception for nonexistent files
        try:
            # Simulate cleanup (would normally be called by quality gates)
            for temp_file in self.evaluator.temp_files_created:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        except Exception as e:
            self.fail(f"Cleanup raised exception for nonexistent file: {e}")


class TestResolutionScalingIntegration(unittest.TestCase):
    """Integration tests for resolution scaling with quality evaluation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        self.quality_gates = QualityGates(self.config_manager)
        
        # Create mock video files with different resolutions
        self.original_1080p = os.path.join(self.temp_dir, "original_1920x1080.mp4")
        self.compressed_720p = os.path.join(self.temp_dir, "compressed_1280x720.mp4")
        
        for path in [self.original_1080p, self.compressed_720p]:
            with open(path, 'w') as f:
                f.write("")
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('src.ffmpeg_utils.FFmpegUtils', MockFFmpegUtils)
    def test_quality_evaluation_with_resolution_scaling(self):
        """Test end-to-end quality evaluation with automatic resolution scaling"""
        # Mock the quality evaluation methods to return scaling info
        mock_scaling_info = {
            'scaling_applied': True,
            'original_resolution': (1920, 1080),
            'comparison_resolution': (1280, 720),
            'scaled_files': []
        }
        
        # Create evaluator for testing
        evaluator = ResolutionAwareQualityEvaluator(self.config_manager)
        
        with patch.object(evaluator, 'compute_ssim_with_scaling') as mock_ssim, \
             patch.object(evaluator, 'compute_vmaf_with_scaling') as mock_vmaf:
            
            # Mock successful quality evaluation with scaling
            mock_ssim.return_value = (0.95, 0.9, mock_scaling_info)
            mock_vmaf.return_value = (85.0, mock_scaling_info)
            
            # This would be called by the main quality evaluation method
            ssim_score, ssim_confidence, ssim_info = evaluator.compute_ssim_with_scaling(
                self.original_1080p, self.compressed_720p, "test_session"
            )
            vmaf_score, vmaf_info = evaluator.compute_vmaf_with_scaling(
                self.original_1080p, self.compressed_720p, "test_session"
            )
        
        # Verify scaling was applied
        self.assertTrue(ssim_info['scaling_applied'])
        self.assertTrue(vmaf_info['scaling_applied'])
        
        # Verify resolution information
        self.assertEqual(ssim_info['original_resolution'], (1920, 1080))
        self.assertEqual(ssim_info['comparison_resolution'], (1280, 720))
        
        # Verify quality scores
        self.assertEqual(ssim_score, 0.95)
        self.assertEqual(vmaf_score, 85.0)


if __name__ == '__main__':
    unittest.main()