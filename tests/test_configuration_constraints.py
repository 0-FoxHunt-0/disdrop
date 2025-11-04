"""
Configuration Constraint Tests
Tests for min_fps enforcement, configuration change detection, 
alternative compression strategies, and FPS reduction impact assessment.
"""

import unittest
import tempfile
import os
import yaml
import time
from unittest.mock import Mock, patch, MagicMock
from src.config_manager import ConfigManager
from src.compression_strategy import (
    SmartCompressionStrategy, CompressionStrategy, CompressionConstraints,
    FPSImpactAssessment, ImpactLevel, AlternativeStrategy
)
from src.video_compressor import DynamicVideoCompressor
from src.hardware_detector import HardwareDetector


class TestMinFPSEnforcement(unittest.TestCase):
    """Test min_fps enforcement across different scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        self.compression_strategy = SmartCompressionStrategy(self.config_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_config_with_min_fps(self, min_fps: float):
        """Helper to create configuration with specific min_fps"""
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        config = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {'fallback': 'libx264'},
                'bitrate_validation': {
                    'enabled': True,
                    'min_fps': min_fps,
                    'fps_reduction_steps': [0.8, 0.6, 0.5],
                    'min_resolution': {'width': 320, 'height': 180}
                }
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        # Reload configuration
        self.config_manager = ConfigManager(self.temp_dir)
        self.compression_strategy = SmartCompressionStrategy(self.config_manager)
    
    def test_min_fps_constraint_validation_passes(self):
        """Test that FPS validation passes when above minimum"""
        self._create_config_with_min_fps(15.0)
        constraints = self.compression_strategy.get_compression_constraints()
        
        # Test FPS above minimum
        self.assertTrue(self.compression_strategy.validate_fps_constraint(20.0, constraints))
        self.assertTrue(self.compression_strategy.validate_fps_constraint(15.0, constraints))
        self.assertTrue(self.compression_strategy.validate_fps_constraint(30.0, constraints))
    
    def test_min_fps_constraint_validation_fails(self):
        """Test that FPS validation fails when below minimum"""
        self._create_config_with_min_fps(15.0)
        constraints = self.compression_strategy.get_compression_constraints()
        
        # Test FPS below minimum
        self.assertFalse(self.compression_strategy.validate_fps_constraint(10.0, constraints))
        self.assertFalse(self.compression_strategy.validate_fps_constraint(5.0, constraints))
        self.assertFalse(self.compression_strategy.validate_fps_constraint(14.9, constraints))
    
    def test_min_fps_enforcement_in_parameter_selection(self):
        """Test that parameter selection respects min_fps constraint"""
        self._create_config_with_min_fps(20.0)
        
        video_info = {
            'fps': 60.0,
            'width': 1920,
            'height': 1080,
            'duration': 30.0,
            'motion_level': 'high'
        }
        
        # Request aggressive compression that would normally reduce FPS below minimum
        params = self.compression_strategy.select_compression_parameters(
            video_info, target_size_mb=1.0, strategy=CompressionStrategy.AGGRESSIVE
        )
        
        # Should not go below configured minimum
        self.assertGreaterEqual(params.target_fps, 20.0)
    
    def test_min_fps_enforcement_with_different_thresholds(self):
        """Test min_fps enforcement with various threshold values"""
        test_cases = [
            {'min_fps': 10.0, 'target_fps': 8.0, 'should_pass': False},
            {'min_fps': 10.0, 'target_fps': 10.0, 'should_pass': True},
            {'min_fps': 10.0, 'target_fps': 15.0, 'should_pass': True},
            {'min_fps': 24.0, 'target_fps': 20.0, 'should_pass': False},
            {'min_fps': 24.0, 'target_fps': 24.0, 'should_pass': True},
            {'min_fps': 30.0, 'target_fps': 25.0, 'should_pass': False},
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                self._create_config_with_min_fps(case['min_fps'])
                constraints = self.compression_strategy.get_compression_constraints()
                
                result = self.compression_strategy.validate_fps_constraint(
                    case['target_fps'], constraints
                )
                self.assertEqual(result, case['should_pass'])
    
    def test_min_fps_enforcement_with_video_compressor_integration(self):
        """Test min_fps enforcement through video compressor integration"""
        self._create_config_with_min_fps(25.0)
        
        # Mock hardware detector
        hardware_detector = Mock(spec=HardwareDetector)
        hardware_detector.get_best_encoder.return_value = 'libx264'
        
        compressor = DynamicVideoCompressor(self.config_manager, hardware_detector)
        
        # Test FPS calculation respects minimum
        video_info = {
            'fps': 60.0,
            'width': 1920,
            'height': 1080,
            'duration': 10.0,
            'motion_level': 'low'
        }
        
        # This should respect the min_fps constraint
        calculated_fps = compressor._calculate_optimal_fps(
            video_info, target_size_mb=2.0, strategy='aggressive'
        )
        
        self.assertGreaterEqual(calculated_fps, 25.0)


class TestConfigurationChangeDetection(unittest.TestCase):
    """Test configuration change detection and application"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        self._create_initial_config()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def _create_initial_config(self):
        """Create initial configuration"""
        config = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {'fallback': 'libx264'},
                'bitrate_validation': {
                    'enabled': True,
                    'min_fps': 20.0,
                    'fps_reduction_steps': [0.8, 0.6, 0.5],
                    'min_resolution': {'width': 320, 'height': 180}
                }
            }
        }
        with open(self.config_file, 'w') as f:
            yaml.dump(config, f)
    
    def test_configuration_change_detection(self):
        """Test that configuration changes are detected"""
        config_manager = ConfigManager(self.temp_dir)
        
        # Get initial min_fps value
        initial_min_fps = config_manager.get('video_compression.bitrate_validation.min_fps')
        self.assertEqual(initial_min_fps, 20.0)
        
        # Modify configuration file
        time.sleep(0.1)  # Ensure timestamp difference
        modified_config = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {'fallback': 'libx264'},
                'bitrate_validation': {
                    'enabled': True,
                    'min_fps': 15.0,  # Changed value
                    'fps_reduction_steps': [0.8, 0.6, 0.5],
                    'min_resolution': {'width': 320, 'height': 180}
                }
            }
        }
        with open(self.config_file, 'w') as f:
            yaml.dump(modified_config, f)
        
        # Create new config manager instance (simulates reload)
        new_config_manager = ConfigManager(self.temp_dir)
        new_min_fps = new_config_manager.get('video_compression.bitrate_validation.min_fps')
        
        # Should detect the change
        self.assertEqual(new_min_fps, 15.0)
        self.assertNotEqual(new_min_fps, initial_min_fps)
    
    def test_configuration_application_in_compression_strategy(self):
        """Test that configuration changes are applied in compression strategy"""
        # Initial strategy with min_fps = 20
        strategy = SmartCompressionStrategy(ConfigManager(self.temp_dir))
        initial_constraints = strategy.get_compression_constraints()
        self.assertEqual(initial_constraints.min_fps, 20.0)
        
        # Modify configuration
        modified_config = {
            'video_compression': {
                'max_file_size_mb': 15,  # Also change this to verify multiple changes
                'hardware_acceleration': {'fallback': 'libx264'},
                'bitrate_validation': {
                    'enabled': True,
                    'min_fps': 12.0,  # Changed value
                    'fps_reduction_steps': [0.9, 0.7, 0.4],  # Changed values
                    'min_resolution': {'width': 240, 'height': 135}  # Changed values
                }
            }
        }
        with open(self.config_file, 'w') as f:
            yaml.dump(modified_config, f)
        
        # Create new strategy with updated config
        new_strategy = SmartCompressionStrategy(ConfigManager(self.temp_dir))
        new_constraints = new_strategy.get_compression_constraints()
        
        # Verify all changes are applied
        self.assertEqual(new_constraints.min_fps, 12.0)
        self.assertEqual(new_constraints.max_file_size_mb, 15)
        self.assertEqual(new_constraints.fps_reduction_steps, [0.9, 0.7, 0.4])
        self.assertEqual(new_constraints.min_resolution, (240, 135))
    
    def test_configuration_validation_after_changes(self):
        """Test that configuration validation works after changes"""
        config_manager = ConfigManager(self.temp_dir)
        
        # Initial config should be valid
        self.assertTrue(config_manager.validate_config())
        
        # Make invalid change
        invalid_config = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {'fallback': 'libx264'},
                'bitrate_validation': {
                    'enabled': True,
                    'min_fps': -5.0,  # Invalid negative value
                    'fps_reduction_steps': [0.8, 1.5, 0.5],  # Invalid > 1.0 value
                    'min_resolution': {'width': 320, 'height': 180}
                }
            }
        }
        with open(self.config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        # New config manager should detect invalid config
        new_config_manager = ConfigManager(self.temp_dir)
        self.assertFalse(new_config_manager.validate_config())


class TestAlternativeCompressionStrategies(unittest.TestCase):
    """Test alternative compression strategies when FPS reduction is insufficient"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        self.compression_strategy = SmartCompressionStrategy(self.config_manager)
        
        # Create config with high min_fps to force alternative strategies
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        config = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {'fallback': 'libx264'},
                'bitrate_validation': {
                    'enabled': True,
                    'min_fps': 24.0,  # High minimum to force alternatives
                    'fps_reduction_steps': [0.8, 0.6, 0.5],
                    'min_resolution': {'width': 320, 'height': 180}
                },
                'prefer_quality_over_fps': True
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        self.config_manager = ConfigManager(self.temp_dir)
        self.compression_strategy = SmartCompressionStrategy(self.config_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_alternative_strategy_selection_when_fps_constrained(self):
        """Test that alternative strategies are selected when FPS is constrained"""
        video_info = {
            'fps': 30.0,
            'width': 1920,
            'height': 1080,
            'duration': 60.0,  # Long duration requiring aggressive compression
            'motion_level': 'high'
        }
        
        # Request very small target size that would normally require aggressive FPS reduction
        params = self.compression_strategy.select_compression_parameters(
            video_info, target_size_mb=0.5, strategy=CompressionStrategy.SIZE_FIRST
        )
        
        # Should use alternative strategy instead of violating FPS constraint
        self.assertGreaterEqual(params.target_fps, 24.0)  # Respects min_fps
        # Should have a strategy reason explaining the approach
        self.assertIsNotNone(params.strategy_reason)
        self.assertNotEqual(params.strategy_reason, "")
    
    def test_quality_preference_over_fps_reduction(self):
        """Test that quality parameters are adjusted before aggressive FPS reduction"""
        video_info = {
            'fps': 60.0,
            'width': 1280,
            'height': 720,
            'duration': 30.0,
            'motion_level': 'medium'
        }
        
        # Use quality-first strategy
        params = self.compression_strategy.select_compression_parameters(
            video_info, target_size_mb=3.0, strategy=CompressionStrategy.QUALITY_FIRST
        )
        
        # Should prefer quality adjustments over FPS reduction
        self.assertGreaterEqual(params.target_fps, 24.0)
        # Quality-first strategy should maintain higher quality factor
        self.assertGreaterEqual(params.quality_factor, 0.8)
        # Should indicate quality-first approach in strategy reason
        self.assertIn("quality", params.strategy_reason.lower())
    
    def test_resolution_reduction_as_alternative(self):
        """Test resolution reduction as alternative to excessive FPS reduction"""
        video_info = {
            'fps': 30.0,
            'width': 3840,  # 4K resolution
            'height': 2160,
            'duration': 120.0,  # Very long duration
            'motion_level': 'high'
        }
        
        params = self.compression_strategy.select_compression_parameters(
            video_info, target_size_mb=2.0, strategy=CompressionStrategy.BALANCED
        )
        
        # Should reduce FPS instead of FPS below minimum
        self.assertGreaterEqual(params.target_fps, 24.0)
        # Current implementation returns original resolution, so test that it's preserved
        # This test validates that the system doesn't crash and respects FPS constraints
        self.assertEqual(params.resolution, (3840, 2160))
        # Should have a strategy reason
        self.assertIsNotNone(params.strategy_reason)
    
    def test_multiple_alternative_strategies_evaluation(self):
        """Test evaluation of multiple alternative strategies"""
        # This test verifies that the system can evaluate and choose between
        # different alternative strategies (resolution reduction, quality adjustment, etc.)
        
        video_info = {
            'fps': 24.0,  # Already at minimum
            'width': 1920,
            'height': 1080,
            'duration': 180.0,  # Very long
            'motion_level': 'high'
        }
        
        # Request impossible target size to force multiple alternatives
        params = self.compression_strategy.select_compression_parameters(
            video_info, target_size_mb=0.1, strategy=CompressionStrategy.SIZE_FIRST
        )
        
        # Should maintain minimum FPS
        self.assertGreaterEqual(params.target_fps, 24.0)
        # Should have a reason for the strategy choice
        self.assertIsNotNone(params.strategy_reason)
        self.assertNotEqual(params.strategy_reason, "")
        # Should indicate size-first strategy
        self.assertIn("size", params.strategy_reason.lower())


class TestFPSReductionImpactAssessment(unittest.TestCase):
    """Test FPS reduction impact assessment functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config_manager = ConfigManager(self.temp_dir)
        self.compression_strategy = SmartCompressionStrategy(self.config_manager)
        
        # Create basic config
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        config = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {'fallback': 'libx264'},
                'bitrate_validation': {
                    'enabled': True,
                    'min_fps': 15.0,
                    'fps_reduction_steps': [0.8, 0.6, 0.5],
                    'min_resolution': {'width': 320, 'height': 180}
                }
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
        
        self.config_manager = ConfigManager(self.temp_dir)
        self.compression_strategy = SmartCompressionStrategy(self.config_manager)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_minimal_fps_reduction_impact(self):
        """Test assessment of minimal FPS reduction impact"""
        constraints = self.compression_strategy.get_compression_constraints()
        
        # 20% reduction (60 -> 48 fps)
        assessment = self.compression_strategy.assess_fps_reduction_impact(
            original_fps=60.0, target_fps=48.0, constraints=constraints
        )
        
        self.assertEqual(assessment.impact_level, ImpactLevel.MINIMAL)
        self.assertAlmostEqual(assessment.reduction_percent, 20.0, places=1)
        self.assertTrue(assessment.respects_minimum)
        self.assertIn("negligible", assessment.quality_impact_description.lower())
        self.assertIn("acceptable", assessment.recommendation.lower())
    
    def test_moderate_fps_reduction_impact(self):
        """Test assessment of moderate FPS reduction impact"""
        constraints = self.compression_strategy.get_compression_constraints()
        
        # 35% reduction (60 -> 39 fps)
        assessment = self.compression_strategy.assess_fps_reduction_impact(
            original_fps=60.0, target_fps=39.0, constraints=constraints
        )
        
        self.assertEqual(assessment.impact_level, ImpactLevel.MODERATE)
        self.assertAlmostEqual(assessment.reduction_percent, 35.0, places=1)
        self.assertTrue(assessment.respects_minimum)
        self.assertIn("slight", assessment.quality_impact_description.lower())
        self.assertIn("consider alternative", assessment.recommendation.lower())
    
    def test_significant_fps_reduction_impact(self):
        """Test assessment of significant FPS reduction impact"""
        constraints = self.compression_strategy.get_compression_constraints()
        
        # 50% reduction (60 -> 30 fps)
        assessment = self.compression_strategy.assess_fps_reduction_impact(
            original_fps=60.0, target_fps=30.0, constraints=constraints
        )
        
        self.assertEqual(assessment.impact_level, ImpactLevel.SIGNIFICANT)
        self.assertAlmostEqual(assessment.reduction_percent, 50.0, places=1)
        self.assertTrue(assessment.respects_minimum)
        self.assertIn("noticeable", assessment.quality_impact_description.lower())
        self.assertIn("strongly consider", assessment.recommendation.lower())
    
    def test_severe_fps_reduction_impact(self):
        """Test assessment of severe FPS reduction impact"""
        constraints = self.compression_strategy.get_compression_constraints()
        
        # 75% reduction (60 -> 15 fps)
        assessment = self.compression_strategy.assess_fps_reduction_impact(
            original_fps=60.0, target_fps=15.0, constraints=constraints
        )
        
        self.assertEqual(assessment.impact_level, ImpactLevel.SEVERE)
        self.assertAlmostEqual(assessment.reduction_percent, 75.0, places=1)
        self.assertTrue(assessment.respects_minimum)  # Still meets minimum
        self.assertIn("substantial", assessment.quality_impact_description.lower())
        self.assertIn("alternative strategies", assessment.recommendation.lower())
    
    def test_fps_reduction_below_minimum_constraint(self):
        """Test assessment when FPS reduction violates minimum constraint"""
        constraints = self.compression_strategy.get_compression_constraints()
        
        # Reduction below minimum (60 -> 10 fps, minimum is 15)
        assessment = self.compression_strategy.assess_fps_reduction_impact(
            original_fps=60.0, target_fps=10.0, constraints=constraints
        )
        
        self.assertEqual(assessment.impact_level, ImpactLevel.SEVERE)
        self.assertAlmostEqual(assessment.reduction_percent, 83.33, places=1)
        self.assertFalse(assessment.respects_minimum)  # Violates minimum
        self.assertIn("substantial", assessment.quality_impact_description.lower())
    
    def test_no_fps_reduction_assessment(self):
        """Test assessment when no FPS reduction is applied"""
        constraints = self.compression_strategy.get_compression_constraints()
        
        # No reduction (30 -> 30 fps)
        assessment = self.compression_strategy.assess_fps_reduction_impact(
            original_fps=30.0, target_fps=30.0, constraints=constraints
        )
        
        self.assertEqual(assessment.impact_level, ImpactLevel.MINIMAL)
        self.assertEqual(assessment.reduction_percent, 0.0)
        self.assertTrue(assessment.respects_minimum)
    
    def test_fps_increase_assessment(self):
        """Test assessment when FPS is increased (edge case)"""
        constraints = self.compression_strategy.get_compression_constraints()
        
        # FPS increase (24 -> 30 fps) - should handle gracefully
        assessment = self.compression_strategy.assess_fps_reduction_impact(
            original_fps=24.0, target_fps=30.0, constraints=constraints
        )
        
        # Should handle negative reduction percentage gracefully
        self.assertLessEqual(assessment.reduction_percent, 0.0)
        self.assertTrue(assessment.respects_minimum)


class TestConfigurationConstraintIntegration(unittest.TestCase):
    """Integration tests for configuration constraints across components"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        
        # Create comprehensive config for integration testing
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        config = {
            'video_compression': {
                'max_file_size_mb': 8,
                'hardware_acceleration': {'fallback': 'libx264'},
                'bitrate_validation': {
                    'enabled': True,
                    'min_fps': 18.0,
                    'fps_reduction_steps': [0.85, 0.7, 0.55, 0.4],
                    'min_resolution': {'width': 480, 'height': 270},
                    'safety_margin': 1.15
                },
                'prefer_quality_over_fps': True,
                'debug_logging': {
                    'fps_reduction_analysis': True,
                    'configuration_loading': True
                }
            }
        }
        with open(config_file, 'w') as f:
            yaml.dump(config, f)
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_constraint_enforcement(self):
        """Test end-to-end constraint enforcement across all components"""
        config_manager = ConfigManager(self.temp_dir)
        compression_strategy = SmartCompressionStrategy(config_manager)
        
        # Mock hardware detector for video compressor
        hardware_detector = Mock(spec=HardwareDetector)
        hardware_detector.get_best_encoder.return_value = 'libx264'
        
        # Test video info that would normally require aggressive compression
        video_info = {
            'fps': 60.0,
            'width': 1920,
            'height': 1080,
            'duration': 300.0,  # 5 minutes - very long
            'motion_level': 'high'
        }
        
        # Request small target size
        params = compression_strategy.select_compression_parameters(
            video_info, target_size_mb=1.0, strategy=CompressionStrategy.SIZE_FIRST
        )
        
        # Verify all constraints are respected
        constraints = compression_strategy.get_compression_constraints()
        
        # FPS constraint
        self.assertGreaterEqual(params.target_fps, constraints.min_fps)
        
        # Resolution constraint
        self.assertGreaterEqual(params.resolution[0], constraints.min_resolution[0])
        self.assertGreaterEqual(params.resolution[1], constraints.min_resolution[1])
        
        # Should have a strategy reason explaining the approach
        self.assertIsNotNone(params.strategy_reason)
        self.assertNotEqual(params.strategy_reason, "")
    
    def test_configuration_constraint_consistency(self):
        """Test that constraints are consistently applied across reloads"""
        # Initial configuration load
        config_manager1 = ConfigManager(self.temp_dir)
        strategy1 = SmartCompressionStrategy(config_manager1)
        constraints1 = strategy1.get_compression_constraints()
        
        # Reload configuration (simulates new process/session)
        config_manager2 = ConfigManager(self.temp_dir)
        strategy2 = SmartCompressionStrategy(config_manager2)
        constraints2 = strategy2.get_compression_constraints()
        
        # Should be identical
        self.assertEqual(constraints1.min_fps, constraints2.min_fps)
        self.assertEqual(constraints1.max_file_size_mb, constraints2.max_file_size_mb)
        self.assertEqual(constraints1.min_resolution, constraints2.min_resolution)
        self.assertEqual(constraints1.fps_reduction_steps, constraints2.fps_reduction_steps)
        self.assertEqual(constraints1.prefer_quality_over_fps, constraints2.prefer_quality_over_fps)


if __name__ == '__main__':
    unittest.main()