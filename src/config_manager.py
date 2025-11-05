"""
Configuration Manager for Video Compressor
Handles loading and managing configuration from YAML files and CLI arguments
"""

import os
import importlib.resources as pkg_resources
import yaml
import logging
from typing import Dict, Any, Optional, List

logger = logging.getLogger(__name__)

class ConfigManager:
    def __init__(self, config_dir: str = "config"):
        self.config_dir = config_dir
        self.config = {}
        self._config_file_timestamps = {}  # Track file modification times
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
                loaded_path = None
                
                # 1) Prefer explicit external config dir
                config_path = os.path.join(self.config_dir, config_file)
                if os.path.exists(config_path):
                    with open(config_path, 'r', encoding='utf-8') as file:
                        config_data = yaml.safe_load(file)
                        if config_data:
                            self.config.update(config_data)
                        loaded_path = config_path
                        # Track file modification time for change detection
                        self._config_file_timestamps[config_file] = os.path.getmtime(config_path)
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
                            loaded_path = packaged_path
                            # Track file modification time for change detection
                            self._config_file_timestamps[config_file] = os.path.getmtime(packaged_path)
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
        overrides_applied = []
        for key, value in args_dict.items():
            if value is not None:
                old_value = self.get(key)
                self._set_nested_value(key, value)
                overrides_applied.append(f"{key}: {old_value} → {value}")
                logger.info(f"Configuration override applied: {key} = {value} (was: {old_value})")
        
        if overrides_applied:
            logger.info(f"Applied {len(overrides_applied)} CLI configuration overrides")
        else:
            logger.debug("No CLI configuration overrides to apply")
    
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
            'libx264': 10,
            'libx265': 15,
            'h264_nvenc': 8,
            'h264_amf': 8,
            'h264_qsv': 8,
            'h264_videotoolbox': 8
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
    
    def apply_quality_evaluation_overrides(self, args):
        """Apply quality evaluation CLI overrides to configuration"""
        if hasattr(args, 'quality_fallback_mode') and args.quality_fallback_mode:
            self._set_nested_value('quality_evaluation.fallback_mode', args.quality_fallback_mode)
            logger.info(f"Quality evaluation fallback mode override: {args.quality_fallback_mode}")
        
        if hasattr(args, 'quality_debug') and args.quality_debug:
            # Enable detailed debugging
            self._set_nested_value('quality_evaluation.logging.log_parsing_attempts', True)
            self._set_nested_value('quality_evaluation.logging.log_performance_metrics', True)
            self._set_nested_value('quality_evaluation.ffmpeg_execution.capture_performance_metrics', True)
            self._set_nested_value('quality_evaluation.error_handling.log_parsing_details', True)
            logger.info("Quality evaluation debug mode enabled")
        
        if hasattr(args, 'quality_timeout') and args.quality_timeout:
            # Apply timeout to all quality operations
            self._set_nested_value('quality_evaluation.ffmpeg_execution.timeout_seconds', args.quality_timeout)
            self._set_nested_value('quality_evaluation.ssim_parsing.timeout_seconds', args.quality_timeout)
            self._set_nested_value('quality_evaluation.vmaf_parsing.timeout_seconds', args.quality_timeout)
            logger.info(f"Quality evaluation timeout override: {args.quality_timeout}s")
        
        if hasattr(args, 'quality_retry_attempts') and args.quality_retry_attempts:
            # Apply retry attempts to all quality operations
            self._set_nested_value('quality_evaluation.ffmpeg_execution.retry_attempts', args.quality_retry_attempts)
            self._set_nested_value('quality_evaluation.ssim_parsing.retry_attempts', args.quality_retry_attempts)
            logger.info(f"Quality evaluation retry attempts override: {args.quality_retry_attempts}")
    
    def apply_debug_logging_overrides(self, debug_enabled: bool = False, performance_enabled: bool = False):
        """Apply debug logging CLI overrides to configuration"""
        if debug_enabled:
            # Enable comprehensive debug logging
            self._set_nested_value('video_compression.debug_logging.enabled', True)
            self._set_nested_value('video_compression.debug_logging.compression_decisions', True)
            self._set_nested_value('video_compression.debug_logging.fps_reduction_analysis', True)
            self._set_nested_value('video_compression.debug_logging.configuration_loading', True)
            self._set_nested_value('video_compression.debug_logging.parameter_changes', True)
            self._set_nested_value('video_compression.debug_logging.session_tracking', True)
            
            # Enable decision logging
            self._set_nested_value('video_compression.debug_logging.decision_logging.strategy_selection', True)
            self._set_nested_value('video_compression.debug_logging.decision_logging.encoder_selection', True)
            self._set_nested_value('video_compression.debug_logging.decision_logging.quality_parameter_decisions', True)
            self._set_nested_value('video_compression.debug_logging.decision_logging.resolution_decisions', True)
            self._set_nested_value('video_compression.debug_logging.decision_logging.bitrate_calculations', True)
            
            # Enable configuration logging
            self._set_nested_value('video_compression.debug_logging.config_logging.active_values', True)
            self._set_nested_value('video_compression.debug_logging.config_logging.overrides_applied', True)
            self._set_nested_value('video_compression.debug_logging.config_logging.validation_results', True)
            self._set_nested_value('video_compression.debug_logging.config_logging.file_changes', True)
            
            # Enable session logging
            self._set_nested_value('video_compression.debug_logging.session_logging.detailed_summary', True)
            self._set_nested_value('video_compression.debug_logging.session_logging.decision_history', True)
            self._set_nested_value('video_compression.debug_logging.session_logging.parameter_evolution', True)
            
            logger.info("Comprehensive debug logging enabled via CLI override")
        
        if performance_enabled or debug_enabled:
            # Enable performance metrics logging
            self._set_nested_value('video_compression.debug_logging.performance_metrics', True)
            self._set_nested_value('video_compression.debug_logging.performance_logging.operation_timing', True)
            self._set_nested_value('video_compression.debug_logging.performance_logging.ffmpeg_performance', True)
            self._set_nested_value('video_compression.debug_logging.performance_logging.quality_evaluation_timing', True)
            self._set_nested_value('video_compression.debug_logging.session_logging.performance_summary', True)
            
            logger.info("Performance metrics logging enabled via CLI override")
    
    def check_for_config_changes(self) -> bool:
        """Check if any configuration files have been modified since last load"""
        try:
            config_files = [
                'video_compression.yaml',
                'gif_settings.yaml', 
                'logging.yaml'
            ]
            
            for config_file in config_files:
                # Check external config first
                config_path = os.path.join(self.config_dir, config_file)
                if os.path.exists(config_path):
                    current_mtime = os.path.getmtime(config_path)
                    stored_mtime = self._config_file_timestamps.get(config_file, 0)
                    if current_mtime > stored_mtime:
                        logger.info(f"Configuration file {config_file} has been modified")
                        return True
                else:
                    # Check packaged config
                    try:
                        package_dir = os.path.abspath(os.path.dirname(__file__))
                        packaged_path = os.path.join(package_dir, 'config', config_file)
                        if os.path.exists(packaged_path):
                            current_mtime = os.path.getmtime(packaged_path)
                            stored_mtime = self._config_file_timestamps.get(config_file, 0)
                            if current_mtime > stored_mtime:
                                logger.info(f"Packaged configuration file {config_file} has been modified")
                                return True
                    except Exception:
                        pass
            
            return False
        except Exception as e:
            logger.warning(f"Error checking for config changes: {e}")
            return False
    
    def reload_config_if_changed(self) -> bool:
        """Reload configuration if files have been modified. Returns True if reloaded."""
        if self.check_for_config_changes():
            logger.info("Reloading configuration due to file changes")
            old_config = self.config.copy()
            self.config = {}
            self._config_file_timestamps = {}
            self._load_all_configs()
            
            # Log what changed
            self._log_config_changes(old_config, self.config)
            return True
        return False
    
    def _log_config_changes(self, old_config: Dict[str, Any], new_config: Dict[str, Any]):
        """Log configuration changes for debugging"""
        try:
            # Check key configuration values that affect compression
            key_paths = [
                'video_compression.bitrate_validation.min_fps',
                'video_compression.bitrate_validation.fps_reduction_steps',
                'video_compression.bitrate_validation.min_resolution',
                'video_compression.max_file_size_mb',
                'video_compression.quality.crf',
                'video_compression.quality.preset'
            ]
            
            for key_path in key_paths:
                old_value = self._get_nested_value(old_config, key_path)
                new_value = self._get_nested_value(new_config, key_path)
                
                if old_value != new_value:
                    logger.info(f"Configuration changed: {key_path}: {old_value} → {new_value}")
                    
        except Exception as e:
            logger.debug(f"Error logging config changes: {e}")
    
    def _get_nested_value(self, config: Dict[str, Any], key_path: str) -> Any:
        """Get nested value from config dict using dot notation"""
        keys = key_path.split('.')
        value = config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return None
    
    def log_active_configuration(self):
        """Log active configuration values for debugging"""
        logger.info("=== Active Configuration Values ===")
        
        # Configuration source information
        logger.info(f"Configuration directory: {self.config_dir}")
        if self._config_file_timestamps:
            logger.info("Loaded configuration files:")
            for config_file, timestamp in self._config_file_timestamps.items():
                import datetime
                dt = datetime.datetime.fromtimestamp(timestamp)
                logger.info(f"  {config_file}: {dt.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Video compression settings
        logger.info("Video Compression Settings:")
        logger.info(f"  Max file size: {self.get('video_compression.max_file_size_mb', 'not set')} MB")
        logger.info(f"  Min FPS: {self.get('video_compression.bitrate_validation.min_fps', 'not set')}")
        logger.info(f"  FPS reduction steps: {self.get('video_compression.bitrate_validation.fps_reduction_steps', 'not set')}")
        
        min_res = self.get('video_compression.bitrate_validation.min_resolution', {})
        if min_res:
            logger.info(f"  Min resolution: {min_res.get('width', '?')}x{min_res.get('height', '?')}")
        else:
            logger.info(f"  Min resolution: not set")
        
        logger.info(f"  CRF: {self.get('video_compression.quality.crf', 'not set')}")
        logger.info(f"  Preset: {self.get('video_compression.quality.preset', 'not set')}")
        logger.info(f"  Tune: {self.get('video_compression.quality.tune', 'not set')}")
        
        # Hardware acceleration
        logger.info("Hardware Acceleration:")
        logger.info(f"  NVIDIA: {self.get('video_compression.hardware_acceleration.nvidia', 'not set')}")
        logger.info(f"  AMD: {self.get('video_compression.hardware_acceleration.amd', 'not set')}")
        logger.info(f"  Fallback: {self.get('video_compression.hardware_acceleration.fallback', 'not set')}")
        
        # Bitrate validation
        bitrate_config = self.get_bitrate_validation_config()
        logger.info("Bitrate Validation:")
        logger.info(f"  Enabled: {bitrate_config.get('enabled', 'not set')}")
        logger.info(f"  Safety margin: {bitrate_config.get('safety_margin', 'not set')}")
        
        encoder_mins = bitrate_config.get('encoder_minimums', {})
        if encoder_mins:
            logger.info("  Encoder minimums (kbps):")
            for encoder, min_bitrate in encoder_mins.items():
                logger.info(f"    {encoder}: {min_bitrate}")
        else:
            logger.info("  Encoder minimums: not set")
        
        # Fallback resolutions
        fallback_res = bitrate_config.get('fallback_resolutions', [])
        if fallback_res:
            logger.info("  Fallback resolutions:")
            for i, (width, height) in enumerate(fallback_res):
                logger.info(f"    {i+1}. {width}x{height}")
        else:
            logger.info("  Fallback resolutions: not set")
        
        # Platform configurations
        platforms = self.get('video_compression.platforms', {})
        if platforms:
            logger.info("Platform Configurations:")
            for platform, config in platforms.items():
                max_size = config.get('max_file_size_mb', 'inherited')
                resolution = f"{config.get('max_width', '?')}x{config.get('max_height', '?')}"
                bitrate = config.get('bitrate', 'not set')
                fps = config.get('fps', 'not set')
                logger.info(f"  {platform}: {resolution}, {bitrate}, {fps}fps, {max_size}MB")
        
        logger.info("=== End Configuration ===")
    
    def validate_configuration_values(self) -> List[str]:
        """Validate configuration values and return list of issues"""
        issues = []
        
        # Validate min_fps
        min_fps = self.get('video_compression.bitrate_validation.min_fps')
        if min_fps is not None:
            if not isinstance(min_fps, (int, float)) or min_fps <= 0:
                issues.append(f"Invalid min_fps: {min_fps} (must be positive number)")
            elif min_fps > 60:
                issues.append(f"Unusually high min_fps: {min_fps} (consider values between 1-30)")
        
        # Validate FPS reduction steps
        fps_steps = self.get('video_compression.bitrate_validation.fps_reduction_steps', [])
        if fps_steps:
            if not isinstance(fps_steps, list):
                issues.append("fps_reduction_steps must be a list")
            else:
                for i, step in enumerate(fps_steps):
                    if not isinstance(step, (int, float)) or step <= 0 or step > 1:
                        issues.append(f"Invalid FPS reduction step[{i}]: {step} (must be between 0 and 1)")
                
                # Check that steps are in descending order (more aggressive reduction)
                if len(fps_steps) > 1:
                    for i in range(1, len(fps_steps)):
                        if fps_steps[i] >= fps_steps[i-1]:
                            issues.append(f"FPS reduction steps should be in descending order: {fps_steps}")
                            break
        
        # Validate CRF
        crf = self.get('video_compression.quality.crf')
        if crf is not None:
            if not isinstance(crf, (int, float)) or crf < 0 or crf > 51:
                issues.append(f"Invalid CRF: {crf} (must be between 0-51)")
            elif crf > 35:
                issues.append(f"High CRF value: {crf} (may result in poor quality, consider 18-28)")
        
        # Validate preset
        preset = self.get('video_compression.quality.preset')
        if preset is not None:
            valid_presets = ['ultrafast', 'superfast', 'veryfast', 'faster', 'fast', 'medium', 'slow', 'slower', 'veryslow']
            if preset not in valid_presets:
                issues.append(f"Invalid preset: {preset} (must be one of: {', '.join(valid_presets)})")
        
        # Validate max file size
        max_size = self.get('video_compression.max_file_size_mb')
        if max_size is not None:
            if not isinstance(max_size, (int, float)) or max_size <= 0:
                issues.append(f"Invalid max_file_size_mb: {max_size} (must be positive number)")
            elif max_size > 1000:
                issues.append(f"Very large max_file_size_mb: {max_size} (consider if this is intentional)")
        
        # Validate bitrate settings
        encoder_minimums = self.get('video_compression.bitrate_validation.encoder_minimums', {})
        if encoder_minimums:
            if not isinstance(encoder_minimums, dict):
                issues.append("encoder_minimums must be a dictionary")
            else:
                for encoder, min_bitrate in encoder_minimums.items():
                    if not isinstance(min_bitrate, (int, float)) or min_bitrate <= 0:
                        issues.append(f"Invalid minimum bitrate for {encoder}: {min_bitrate} (must be positive)")
                    elif min_bitrate > 10000:
                        issues.append(f"Very high minimum bitrate for {encoder}: {min_bitrate}kbps")
        
        # Validate resolution settings
        min_resolution = self.get('video_compression.bitrate_validation.min_resolution', {})
        if min_resolution:
            if not isinstance(min_resolution, dict):
                issues.append("min_resolution must be a dictionary")
            else:
                width = min_resolution.get('width')
                height = min_resolution.get('height')
                
                if width is not None:
                    if not isinstance(width, int) or width <= 0:
                        issues.append(f"Invalid min_resolution.width: {width} (must be positive integer)")
                    elif width < 64:
                        issues.append(f"Very low min_resolution.width: {width} (may cause encoding issues)")
                
                if height is not None:
                    if not isinstance(height, int) or height <= 0:
                        issues.append(f"Invalid min_resolution.height: {height} (must be positive integer)")
                    elif height < 48:
                        issues.append(f"Very low min_resolution.height: {height} (may cause encoding issues)")
        
        # Validate safety margin
        safety_margin = self.get('video_compression.bitrate_validation.safety_margin')
        if safety_margin is not None:
            if not isinstance(safety_margin, (int, float)) or safety_margin <= 0:
                issues.append(f"Invalid safety_margin: {safety_margin} (must be positive number)")
            elif safety_margin < 1.0:
                issues.append(f"Low safety_margin: {safety_margin} (values < 1.0 may cause size overruns)")
            elif safety_margin > 2.0:
                issues.append(f"High safety_margin: {safety_margin} (may result in unnecessarily small files)")
        
        # Validate hardware acceleration settings
        hw_nvidia = self.get('video_compression.hardware_acceleration.nvidia')
        hw_amd = self.get('video_compression.hardware_acceleration.amd')
        hw_fallback = self.get('video_compression.hardware_acceleration.fallback')
        
        valid_encoders = ['libx264', 'libx265', 'h264_nvenc', 'h264_amf', 'h264_qsv', 'h264_videotoolbox']
        
        for hw_type, encoder in [('nvidia', hw_nvidia), ('amd', hw_amd), ('fallback', hw_fallback)]:
            if encoder and encoder not in valid_encoders:
                issues.append(f"Invalid {hw_type} encoder: {encoder} (must be one of: {', '.join(valid_encoders)})")
        
        return issues
    
    def validate_compression_parameters(self, fps: float, bitrate: int, width: int, height: int) -> List[str]:
        """Validate specific compression parameters against configuration constraints"""
        issues = []
        
        # Check FPS against minimum
        min_fps = self.get('video_compression.bitrate_validation.min_fps', 20)
        if fps < min_fps:
            issues.append(f"FPS {fps} is below configured minimum {min_fps}")
        
        # Check resolution against minimum
        min_resolution = self.get('video_compression.bitrate_validation.min_resolution', {})
        min_width = min_resolution.get('width', 320)
        min_height = min_resolution.get('height', 180)
        
        if width < min_width:
            issues.append(f"Width {width} is below configured minimum {min_width}")
        if height < min_height:
            issues.append(f"Height {height} is below configured minimum {min_height}")
        
        # Check bitrate against encoder minimums
        encoder_minimums = self.get('video_compression.bitrate_validation.encoder_minimums', {})
        if encoder_minimums:
            # Find the most restrictive minimum (highest value)
            max_minimum = max(encoder_minimums.values()) if encoder_minimums.values() else 0
            if bitrate < max_minimum:
                issues.append(f"Bitrate {bitrate}kbps may be below encoder minimums (highest: {max_minimum}kbps)")
        
        # Calculate bits per pixel and warn if too low
        pixels_per_sec = width * height * fps
        if pixels_per_sec > 0:
            bpp = (bitrate * 1000) / pixels_per_sec
            if bpp < 0.02:  # Very low BPP threshold
                issues.append(f"Very low bits-per-pixel: {bpp:.4f} (may cause severe quality degradation)")
            elif bpp < 0.05:  # Low BPP threshold
                issues.append(f"Low bits-per-pixel: {bpp:.4f} (may cause quality issues)")
        
        return issues
    
    def log_parameter_validation(self, fps: float, bitrate: int, width: int, height: int, encoder: str = None):
        """Log validation results for compression parameters"""
        logger.info(f"Validating compression parameters: {width}x{height}@{fps}fps, {bitrate}kbps")
        if encoder:
            logger.info(f"Target encoder: {encoder}")
        
        issues = self.validate_compression_parameters(fps, bitrate, width, height)
        
        if issues:
            logger.warning("Parameter validation issues:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        else:
            logger.info("All parameters within configured constraints")
        
        # Log additional context
        pixels_per_sec = width * height * fps
        if pixels_per_sec > 0:
            bpp = (bitrate * 1000) / pixels_per_sec
            logger.info(f"Calculated bits-per-pixel: {bpp:.4f}")
        
        # Check against encoder-specific minimums if encoder specified
        if encoder:
            encoder_minimums = self.get('video_compression.bitrate_validation.encoder_minimums', {})
            encoder_min = encoder_minimums.get(encoder)
            if encoder_min:
                if bitrate >= encoder_min:
                    logger.info(f"Bitrate meets {encoder} minimum: {bitrate}kbps >= {encoder_min}kbps")
                else:
                    logger.warning(f"Bitrate below {encoder} minimum: {bitrate}kbps < {encoder_min}kbps")

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