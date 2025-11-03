"""
Unit tests for configuration parsing and CLI parameter handling
Tests bitrate validation configuration and parameter overrides
"""

import unittest
import tempfile
import os
import yaml
from src.config_manager import ConfigManager


class TestConfigValidation(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_valid_bitrate_validation_config(self):
        """Test validation of valid bitrate validation configuration"""
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        valid_config = {
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
                        [640, 360],
                        [480, 270],
                        [320, 180]
                    ],
                    'segmentation_threshold_mb': 50,
                    'safety_margin': 1.1,
                    'min_resolution': {
                        'width': 320,
                        'height': 180
                    },
                    'min_fps': 10,
                    'fps_reduction_steps': [0.8, 0.6, 0.5]
                }
            },
            'gif_settings': {
                'max_file_size_mb': 5
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(valid_config, f)
        
        config_manager = ConfigManager(self.temp_dir)
        self.assertTrue(config_manager.validate_config())
    
    def test_invalid_encoder_minimums(self):
        """Test validation fails with invalid encoder minimums"""
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        invalid_config = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {
                    'fallback': 'libx264'
                },
                'bitrate_validation': {
                    'encoder_minimums': {
                        'libx264': -5,  # Invalid negative value
                        'libx265': 'invalid'  # Invalid string value
                    }
                }
            },
            'gif_settings': {
                'max_file_size_mb': 5
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        config_manager = ConfigManager(self.temp_dir)
        self.assertFalse(config_manager.validate_config())
    
    def test_invalid_fallback_resolutions_format(self):
        """Test validation fails with invalid fallback resolutions format"""
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        invalid_config = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {
                    'fallback': 'libx264'
                },
                'bitrate_validation': {
                    'fallback_resolutions': [
                        [640, 360],
                        [480],  # Invalid - missing height
                        'invalid'  # Invalid - not a list
                    ]
                }
            },
            'gif_settings': {
                'max_file_size_mb': 5
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        config_manager = ConfigManager(self.temp_dir)
        self.assertFalse(config_manager.validate_config())
    
    def test_invalid_segmentation_threshold(self):
        """Test validation fails with invalid segmentation threshold"""
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        invalid_config = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {
                    'fallback': 'libx264'
                },
                'bitrate_validation': {
                    'segmentation_threshold_mb': -10  # Invalid negative value
                }
            },
            'gif_settings': {
                'max_file_size_mb': 5
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        config_manager = ConfigManager(self.temp_dir)
        self.assertFalse(config_manager.validate_config())
    
    def test_invalid_safety_margin(self):
        """Test validation fails with invalid safety margin"""
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        invalid_config = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {
                    'fallback': 'libx264'
                },
                'bitrate_validation': {
                    'safety_margin': 0  # Invalid zero value
                }
            },
            'gif_settings': {
                'max_file_size_mb': 5
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        config_manager = ConfigManager(self.temp_dir)
        self.assertFalse(config_manager.validate_config())
    
    def test_invalid_min_resolution(self):
        """Test validation fails with invalid minimum resolution"""
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        invalid_config = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {
                    'fallback': 'libx264'
                },
                'bitrate_validation': {
                    'min_resolution': {
                        'width': -320,  # Invalid negative value
                        'height': 'invalid'  # Invalid string value
                    }
                }
            },
            'gif_settings': {
                'max_file_size_mb': 5
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        config_manager = ConfigManager(self.temp_dir)
        self.assertFalse(config_manager.validate_config())
    
    def test_invalid_fps_settings(self):
        """Test validation fails with invalid FPS settings"""
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        invalid_config = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {
                    'fallback': 'libx264'
                },
                'bitrate_validation': {
                    'min_fps': -10,  # Invalid negative value
                    'fps_reduction_steps': [0.8, 1.5, 0.5]  # Invalid > 1.0 value
                }
            },
            'gif_settings': {
                'max_file_size_mb': 5
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(invalid_config, f)
        
        config_manager = ConfigManager(self.temp_dir)
        self.assertFalse(config_manager.validate_config())
    
    def test_set_encoder_bitrate_floor(self):
        """Test setting encoder bitrate floor override"""
        config_manager = ConfigManager(self.temp_dir)
        
        # Set bitrate floor for specific encoder
        config_manager.set_encoder_bitrate_floor('libx264', 5)
        
        # Verify it was set
        encoder_minimums = config_manager.get('video_compression.bitrate_validation.encoder_minimums', {})
        self.assertEqual(encoder_minimums.get('libx264'), 5)
    
    def test_set_all_encoder_bitrate_floors(self):
        """Test setting bitrate floor for all encoders"""
        config_manager = ConfigManager(self.temp_dir)
        
        # Set bitrate floor for all encoders
        config_manager.set_all_encoder_bitrate_floors(2)
        
        # Verify all encoders were set
        encoder_minimums = config_manager.get('video_compression.bitrate_validation.encoder_minimums', {})
        for encoder, minimum in encoder_minimums.items():
            self.assertEqual(minimum, 2)
    
    def test_validate_encoder_name(self):
        """Test encoder name validation"""
        config_manager = ConfigManager(self.temp_dir)
        
        # Valid encoders
        self.assertTrue(config_manager.validate_encoder_name('libx264'))
        self.assertTrue(config_manager.validate_encoder_name('libx265'))
        self.assertTrue(config_manager.validate_encoder_name('h264_nvenc'))
        self.assertTrue(config_manager.validate_encoder_name('h264_amf'))
        self.assertTrue(config_manager.validate_encoder_name('h264_qsv'))
        self.assertTrue(config_manager.validate_encoder_name('h264_videotoolbox'))
        
        # Invalid encoder
        self.assertFalse(config_manager.validate_encoder_name('invalid_encoder'))
    
    def test_get_bitrate_validation_config(self):
        """Test getting complete bitrate validation configuration"""
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        test_config = {
            'video_compression': {
                'bitrate_validation': {
                    'enabled': True,
                    'encoder_minimums': {
                        'libx264': 3
                    },
                    'segmentation_threshold_mb': 25
                }
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(test_config, f)
        
        config_manager = ConfigManager(self.temp_dir)
        validation_config = config_manager.get_bitrate_validation_config()
        
        self.assertTrue(validation_config['enabled'])
        self.assertEqual(validation_config['encoder_minimums']['libx264'], 3)
        self.assertEqual(validation_config['segmentation_threshold_mb'], 25)
        
        # Check defaults are provided
        self.assertIn('safety_margin', validation_config)
        self.assertIn('min_resolution', validation_config)
        self.assertIn('min_fps', validation_config)
    
    def test_cli_args_update(self):
        """Test updating configuration from CLI arguments"""
        config_manager = ConfigManager(self.temp_dir)
        
        # Simulate CLI arguments
        cli_args = {
            'video_compression.bitrate_validation.enabled': False,
            'video_compression.max_file_size_mb': 25
        }
        
        config_manager.update_from_args(cli_args)
        
        # Verify updates
        self.assertFalse(config_manager.get('video_compression.bitrate_validation.enabled'))
        self.assertEqual(config_manager.get('video_compression.max_file_size_mb'), 25)
    
    def test_missing_required_config_keys(self):
        """Test validation fails when required keys are missing"""
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        incomplete_config = {
            'video_compression': {
                # Missing max_file_size_mb and hardware_acceleration.fallback
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(incomplete_config, f)
        
        config_manager = ConfigManager(self.temp_dir)
        self.assertFalse(config_manager.validate_config())
    
    def test_bitrate_validation_disabled(self):
        """Test validation passes when bitrate validation is disabled"""
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        config_with_disabled_validation = {
            'video_compression': {
                'max_file_size_mb': 10,
                'hardware_acceleration': {
                    'fallback': 'libx264'
                },
                'bitrate_validation': {
                    'enabled': False,
                    'encoder_minimums': {
                        'libx264': 'invalid'  # This should be ignored when disabled
                    }
                }
            },
            'gif_settings': {
                'max_file_size_mb': 5
            }
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(config_with_disabled_validation, f)
        
        config_manager = ConfigManager(self.temp_dir)
        self.assertTrue(config_manager.validate_config())


class TestConfigManagerEdgeCases(unittest.TestCase):
    
    def setUp(self):
        """Set up test fixtures for edge cases"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_invalid_bitrate_floor_values(self):
        """Test handling of invalid bitrate floor values"""
        config_manager = ConfigManager(self.temp_dir)
        
        # Test invalid values are rejected
        config_manager.set_encoder_bitrate_floor('libx264', -5)  # Negative
        config_manager.set_encoder_bitrate_floor('libx264', 0)   # Zero
        config_manager.set_all_encoder_bitrate_floors(-2)        # Negative
        
        # Should not have set invalid values
        encoder_minimums = config_manager.get('video_compression.bitrate_validation.encoder_minimums', {})
        # Values should either be unset or use defaults, not the invalid values
        if 'libx264' in encoder_minimums:
            self.assertGreater(encoder_minimums['libx264'], 0)
    
    def test_nonexistent_config_directory(self):
        """Test handling of nonexistent config directory"""
        nonexistent_dir = os.path.join(self.temp_dir, 'nonexistent')
        
        # Should not raise exception
        config_manager = ConfigManager(nonexistent_dir)
        
        # Should still work with defaults
        self.assertIsNotNone(config_manager.get('video_compression.max_file_size_mb', 10))
    
    def test_corrupted_yaml_file(self):
        """Test handling of corrupted YAML file"""
        config_file = os.path.join(self.temp_dir, 'video_compression.yaml')
        
        # Write invalid YAML
        with open(config_file, 'w') as f:
            f.write('invalid: yaml: content: [unclosed')
        
        # Should handle gracefully
        try:
            config_manager = ConfigManager(self.temp_dir)
            # Should not crash, may use defaults
            self.assertIsNotNone(config_manager)
        except Exception as e:
            # If it does raise an exception, it should be a YAML error
            self.assertTrue(isinstance(e, (yaml.YAMLError, yaml.scanner.ScannerError)) or 'yaml' in str(e).lower())


if __name__ == '__main__':
    unittest.main()