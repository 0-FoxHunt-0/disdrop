"""
Configuration Manager for Video Compressor
Handles loading and managing configuration from YAML files and CLI arguments
"""

import os
import importlib.resources as pkg_resources
import yaml
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.config = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all configuration files from the config directory"""
        try:
            config_files = [
                'video_compression.yaml',
                'gif_settings.yaml',
                'logging.yaml'
            ]
            
            for config_file in config_files:
                # 1) Prefer explicit external config dir
                config_path = os.path.join(self.config_dir, config_file)
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as file:
                        config_data = yaml.safe_load(file)
                        if config_data:
                            self.config.update(config_data)
                        logger.debug(f"Loaded config from {config_path}")
                    continue

                # 2) Fall back to packaged defaults under installed package dir
                try:
                    package_dir = os.path.abspath(os.path.dirname(__file__))
                    packaged_path = os.path.join(package_dir, 'config', config_file)
                    if os.path.exists(packaged_path):
                        with open(packaged_path, 'r', encoding='utf-8') as file:
                            config_data = yaml.safe_load(file)
                            if config_data:
                                self.config.update(config_data)
                            logger.debug(f"Loaded packaged default config from {packaged_path}")
                        continue
                except Exception as e:
                    logger.debug(f"Error checking packaged config for {config_file}: {e}")

                # 3) If neither found, keep existing defaults
                logger.warning(f"Config file not found in '{self.config_dir}' or packaged defaults: {config_file}")
                    
        except Exception as e:
            logger.error(f"Error loading configuration: {e}")
            raise
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation
        Example: get('video_compression.quality.crf')
        """
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            logger.debug(f"Configuration key '{key_path}' not found, using default: {default}")
            return default
    
    def get_platform_config(self, platform: str, config_type: str = 'video_compression') -> Dict[str, Any]:
        """Get platform-specific configuration"""
        platform_config = self.get(f'{config_type}.platforms.{platform}', {})
        if not platform_config:
            logger.warning(f"Platform '{platform}' not found in {config_type} config")
        return platform_config
    
    def update_from_args(self, args_dict: Dict[str, Any]):
        """Update configuration with command line arguments"""
        for key, value in args_dict.items():
            if value is not None:
                self._set_nested_value(key, value)
                logger.debug(f"Updated config from CLI: {key} = {value}")
    
    def _set_nested_value(self, key_path: str, value: Any):
        """Set nested configuration value using dot notation"""
        keys = key_path.split('.')
        config_section = self.config
        
        # Navigate to the parent of the target key
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
        
        # Set the final value
        config_section[keys[-1]] = value
    
    def get_hardware_acceleration(self) -> str:
        """Get appropriate hardware acceleration codec"""
        nvidia_codec = self.get('video_compression.hardware_acceleration.nvidia')
        amd_codec = self.get('video_compression.hardware_acceleration.amd')
        fallback_codec = self.get('video_compression.hardware_acceleration.fallback')
        
        # This will be enhanced by hardware detection
        return fallback_codec
    
    def validate_config(self) -> bool:
        """Validate that required configuration values are present"""
        required_keys = [
            'video_compression.max_file_size_mb',
            'gif_settings.max_file_size_mb',
            'video_compression.hardware_acceleration.fallback'
        ]
        
        for key in required_keys:
            if self.get(key) is None:
                logger.error(f"Required configuration key missing: {key}")
                return False
        
        logger.info("Configuration validation passed")
        return True
    
    def get_temp_dir(self) -> str:
        """Get temporary directory path"""
        temp_dir = self.get('video_compression.temp_dir', './temp')
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir 