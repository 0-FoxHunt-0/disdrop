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
        
        # Validate bitrate validation configuration if enabled
        if self.get('video_compression.bitrate_validation.enabled', True):
            if not self._validate_bitrate_validation_config():
                return False
        
        logger.info("Configuration validation passed")
        return True
    
    def _validate_bitrate_validation_config(self) -> bool:
        """Validate bitrate validation specific configuration"""
        # Validate encoder minimums
        encoder_minimums = self.get('video_compression.bitrate_validation.encoder_minimums', {})
        if not isinstance(encoder_minimums, dict):
            logger.error("bitrate_validation.encoder_minimums must be a dictionary")
            return False
        
        # Validate encoder minimum values are positive numbers
        for encoder, minimum in encoder_minimums.items():
            if not isinstance(minimum, (int, float)) or minimum <= 0:
                logger.error(f"Invalid encoder minimum for {encoder}: {minimum} (must be positive number)")
                return False
        
        # Validate fallback resolutions format
        fallback_resolutions = self.get('video_compression.bitrate_validation.fallback_resolutions', [])
        if fallback_resolutions and not isinstance(fallback_resolutions, list):
            logger.error("bitrate_validation.fallback_resolutions must be a list")
            return False
        
        for res in fallback_resolutions:
            if not isinstance(res, list) or len(res) != 2:
                logger.error(f"Invalid fallback resolution format: {res} (must be [width, height])")
                return False
            if not all(isinstance(dim, int) and dim > 0 for dim in res):
                logger.error(f"Invalid fallback resolution dimensions: {res} (must be positive integers)")
                return False
        
        # Validate segmentation threshold
        segmentation_threshold = self.get('video_compression.bitrate_validation.segmentation_threshold_mb')
        if segmentation_threshold is not None:
            if not isinstance(segmentation_threshold, (int, float)) or segmentation_threshold <= 0:
                logger.error(f"Invalid segmentation_threshold_mb: {segmentation_threshold} (must be positive number)")
                return False
        
        # Validate safety margin
        safety_margin = self.get('video_compression.bitrate_validation.safety_margin')
        if safety_margin is not None:
            if not isinstance(safety_margin, (int, float)) or safety_margin <= 0:
                logger.error(f"Invalid safety_margin: {safety_margin} (must be positive number)")
                return False
        
        # Validate minimum resolution constraints
        min_resolution = self.get('video_compression.bitrate_validation.min_resolution', {})
        if min_resolution:
            if not isinstance(min_resolution, dict):
                logger.error("bitrate_validation.min_resolution must be a dictionary")
                return False
            
            width = min_resolution.get('width')
            height = min_resolution.get('height')
            
            if width is not None and (not isinstance(width, int) or width <= 0):
                logger.error(f"Invalid min_resolution.width: {width} (must be positive integer)")
                return False
            
            if height is not None and (not isinstance(height, int) or height <= 0):
                logger.error(f"Invalid min_resolution.height: {height} (must be positive integer)")
                return False
        
        # Validate minimum FPS
        min_fps = self.get('video_compression.bitrate_validation.min_fps')
        if min_fps is not None:
            if not isinstance(min_fps, (int, float)) or min_fps <= 0:
                logger.error(f"Invalid min_fps: {min_fps} (must be positive number)")
                return False
        
        # Validate FPS reduction steps
        fps_reduction_steps = self.get('video_compression.bitrate_validation.fps_reduction_steps', [])
        if fps_reduction_steps:
            if not isinstance(fps_reduction_steps, list):
                logger.error("bitrate_validation.fps_reduction_steps must be a list")
                return False
            
            for step in fps_reduction_steps:
                if not isinstance(step, (int, float)) or step <= 0 or step > 1:
                    logger.error(f"Invalid FPS reduction step: {step} (must be between 0 and 1)")
                    return False
        
        return True
    
    def set_encoder_bitrate_floor(self, encoder: str, bitrate_kbps: int):
        """Set runtime bitrate floor override for a specific encoder"""
        if not isinstance(bitrate_kbps, (int, float)) or bitrate_kbps <= 0:
            logger.error(f"Invalid bitrate floor for {encoder}: {bitrate_kbps} (must be positive number)")
            return
        
        encoder_minimums = self.get('video_compression.bitrate_validation.encoder_minimums', {})
        encoder_minimums[encoder] = bitrate_kbps
        self._set_nested_value('video_compression.bitrate_validation.encoder_minimums', encoder_minimums)
        logger.info(f"Set bitrate floor override for {encoder}: {bitrate_kbps}kbps")
    
    def set_all_encoder_bitrate_floors(self, bitrate_kbps: int):
        """Set runtime bitrate floor override for all encoders"""
        if not isinstance(bitrate_kbps, (int, float)) or bitrate_kbps <= 0:
            logger.error(f"Invalid bitrate floor: {bitrate_kbps} (must be positive number)")
            return
        
        # Get current encoder minimums or use defaults
        encoder_minimums = self.get('video_compression.bitrate_validation.encoder_minimums', {
            'libx264': 3,
            'libx265': 5,
            'h264_nvenc': 2,
            'h264_amf': 2,
            'h264_qsv': 2,
            'h264_videotoolbox': 2
        })
        
        # Override all encoder minimums with the specified floor
        for encoder in encoder_minimums.keys():
            encoder_minimums[encoder] = bitrate_kbps
        
        self._set_nested_value('video_compression.bitrate_validation.encoder_minimums', encoder_minimums)
        logger.info(f"Set bitrate floor override for all encoders: {bitrate_kbps}kbps")
    
    def validate_encoder_name(self, encoder: str) -> bool:
        """Validate that an encoder name is supported"""
        supported_encoders = [
            'libx264', 'libx265', 'h264_nvenc', 'h264_amf', 
            'h264_qsv', 'h264_videotoolbox'
        ]
        return encoder in supported_encoders
    
    def get_bitrate_validation_config(self) -> Dict[str, Any]:
        """Get complete bitrate validation configuration"""
        return {
            'enabled': self.get('video_compression.bitrate_validation.enabled', True),
            'encoder_minimums': self.get('video_compression.bitrate_validation.encoder_minimums', {}),
            'fallback_resolutions': self.get('video_compression.bitrate_validation.fallback_resolutions', []),
            'segmentation_threshold_mb': self.get('video_compression.bitrate_validation.segmentation_threshold_mb', 50),
            'safety_margin': self.get('video_compression.bitrate_validation.safety_margin', 1.1),
            'min_resolution': self.get('video_compression.bitrate_validation.min_resolution', {'width': 320, 'height': 180}),
            'min_fps': self.get('video_compression.bitrate_validation.min_fps', 10),
            'fps_reduction_steps': self.get('video_compression.bitrate_validation.fps_reduction_steps', [0.8, 0.6, 0.5])
        }
    
    def get_temp_dir(self) -> str:
        """Return the package-root temp directory path (<package_root>/temp).

        Ignores any user-configured temp_dir to enforce a consistent location
        outside of output directories and independent of CWD.
        """
        try:
            # Resolve application base dir using existing helper
            from .logger_setup import get_app_base_dir
            base_dir = get_app_base_dir()
        except Exception:
            # Fallback to current working directory if helper fails
            base_dir = os.getcwd()

        temp_dir = os.path.join(base_dir, 'temp')
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir 