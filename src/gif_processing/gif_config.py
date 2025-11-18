"""
GIF Configuration Helper
Provides centralized config access with validation and sensible defaults
"""

from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class GifConfigHelper:
    """Helper class for accessing GIF configuration with validation and defaults"""
    
    def __init__(self, config_manager):
        """
        Initialize config helper.
        
        Args:
            config_manager: ConfigManager instance
        """
        self.config = config_manager
    
    def get_optimization_config(self) -> Dict[str, Any]:
        """
        Get optimization configuration settings.
        
        Returns:
            Dictionary with optimization settings
        """
        gif_config = self.config.get('gif_settings', {}) or {}
        opt_config = gif_config.get('optimization', {}) or {}
        size_pred = opt_config.get('size_prediction', {}) or {}
        near_target_cfg = opt_config.get('near_target', {}) or {}
        
        return {
            'fps': int(gif_config.get('fps', 20)),
            'width': int(gif_config.get('width', 360)),
            'height': int(gif_config.get('height', -1)),
            'colors': int(gif_config.get('colors', gif_config.get('palette_size', 256))),
            'dither': str(gif_config.get('dither', 'floyd_steinberg')),
            'lossy': int(gif_config.get('lossy', 60)),
            'stats_mode': str(gif_config.get('stats_mode', 'diff')),
            'max_file_size_mb': float(gif_config.get('max_file_size_mb', 10.0)),
            'max_duration_seconds': float(gif_config.get('max_duration_seconds', 30.0)),
            'allow_aggressive_compression': bool(opt_config.get('allow_aggressive_compression', True)),
            'use_source_reencoding': bool(opt_config.get('use_source_reencoding', True)),
            'size_prediction_enabled': bool(size_pred.get('enabled', True)),
            'near_target': {
                'threshold_percent': float(near_target_cfg.get('threshold_percent', 15.0)),
                'mode': str(near_target_cfg.get('mode', 'both')),
                'max_attempts': int(near_target_cfg.get('max_attempts', 60)),
                'fine_tune_threshold_percent': float(near_target_cfg.get('fine_tune_threshold_percent', 10.0)),
                'absolute_mb_threshold': float(near_target_cfg.get('absolute_mb_threshold', 0.3)),
            }
        }
    
    def get_performance_config(self) -> Dict[str, Any]:
        """
        Get performance configuration settings.
        
        Returns:
            Dictionary with performance settings
        """
        perf_cfg = self.config.get('gif_settings.performance', {}) or {}
        gifsicle_perf = self.config.get('gif_settings.gifsicle_performance', {}) or {}
        
        fast_mode = bool(gifsicle_perf.get('fast_mode', True))
        
        return {
            'fast_mode': fast_mode,
            'gifsicle_time_budget_seconds': int(gifsicle_perf.get('gifsicle_time_budget_seconds', 25 if fast_mode else 45)),
            'gifsicle_max_candidates': int(gifsicle_perf.get('gifsicle_max_candidates', 48 if fast_mode else 120)),
            'gifsicle_optimize_level': int(gifsicle_perf.get('gifsicle_optimize_level', 2 if fast_mode else 3)),
            'skip_gifsicle_far_over_ratio': float(gifsicle_perf.get('skip_gifsicle_far_over_ratio', 0.35 if fast_mode else 0.5)),
            'near_target_max_runs': int(gifsicle_perf.get('near_target_max_runs', 12 if fast_mode else 24))
        }
    
    def get_long_clip_guardrails(self) -> Dict[str, Any]:
        """
        Get guardrail settings for very long clips to keep FFmpeg within time budgets.
        """
        defaults = {
            'enabled': True,
            'duration_seconds': 45.0,
            'frame_budget': 900.0,
            'palette_fps_cap': 14,
            'palette_stats_mode': 'diff',
            'palette_max_colors': 208,
            'mpdecimate': {'hi': 768, 'lo': 64, 'frac': 0.4},
            'reencode_max_workers': 1,
            'reencode_strategy_limit': 2,
            'segment_duration_ratio': 0.5,
            'segment_frame_budget_ratio': 0.6,
        }
        cfg = self.config.get('gif_settings.performance.long_clip_guardrails', {}) or {}
        mp_cfg = cfg.get('mpdecimate', {}) or {}
        
        return {
            'enabled': bool(cfg.get('enabled', defaults['enabled'])),
            'duration_seconds': float(cfg.get('duration_seconds', defaults['duration_seconds'])),
            'frame_budget': float(cfg.get('frame_budget', defaults['frame_budget'])),
            'palette_fps_cap': int(cfg.get('palette_fps_cap', defaults['palette_fps_cap'])),
            'palette_stats_mode': str(cfg.get('palette_stats_mode', defaults['palette_stats_mode'])),
            'palette_max_colors': int(cfg.get('palette_max_colors', defaults['palette_max_colors'])),
            'mpdecimate': {
                'hi': int(mp_cfg.get('hi', defaults['mpdecimate']['hi'])),
                'lo': int(mp_cfg.get('lo', defaults['mpdecimate']['lo'])),
                'frac': float(mp_cfg.get('frac', defaults['mpdecimate']['frac'])),
            },
            'reencode_max_workers': int(cfg.get('reencode_max_workers', defaults['reencode_max_workers'])),
            'reencode_strategy_limit': int(cfg.get('reencode_strategy_limit', defaults['reencode_strategy_limit'])),
            'segment_duration_ratio': float(cfg.get('segment_duration_ratio', defaults['segment_duration_ratio'])),
            'segment_frame_budget_ratio': float(cfg.get('segment_frame_budget_ratio', defaults['segment_frame_budget_ratio'])),
        }
    
    def get_quality_optimization_config(self) -> Dict[str, Any]:
        """
        Get quality optimization configuration.
        
        Returns:
            Dictionary with quality optimization settings
        """
        qo_cfg = self.config.get('gif_settings.quality_optimization', {}) or {}
        
        return {
            'enabled': bool(qo_cfg.get('enabled', True)),
            'max_attempts_per_stage': int(qo_cfg.get('max_attempts_per_stage', 4)),
            'target_size_efficiency': float(qo_cfg.get('target_size_efficiency', 0.95)),
            'early_stop': {
                'enabled': bool(qo_cfg.get('early_stop', {}).get('enabled', True)),
                'min_target_utilization': float(qo_cfg.get('early_stop', {}).get('min_target_utilization', 0.92)),
                'min_quality_score': float(qo_cfg.get('early_stop', {}).get('min_quality_score', 7.5))
            },
            'skip_uplift': {
                'enabled': bool(qo_cfg.get('skip_uplift', {}).get('enabled', True)),
                'min_size_headroom': float(qo_cfg.get('skip_uplift', {}).get('min_size_headroom', 0.04)),
                'max_extra_attempts': int(qo_cfg.get('skip_uplift', {}).get('max_extra_attempts', 0))
            }
        }
    
    def get_quality_stages(self) -> List[Dict[str, Any]]:
        """
        Get quality optimization stages.
        
        Returns:
            List of quality stage dictionaries
        """
        qo_cfg = self.config.get('gif_settings.quality_optimization', {}) or {}
        stages = qo_cfg.get('quality_stages', [])
        
        if not stages:
            # Default stages if none configured
            stages = [
                {
                    'name': 'Maximum Quality',
                    'colors': 256,
                    'fps_multiplier': 1.0,
                    'resolution_multiplier': 1.0,
                    'lossy': 0,
                    'dither': 'floyd_steinberg'
                },
                {
                    'name': 'High Quality',
                    'colors': 224,
                    'fps_multiplier': 0.98,
                    'resolution_multiplier': 0.98,
                    'lossy': 20,
                    'dither': 'floyd_steinberg'
                },
                {
                    'name': 'Medium Quality',
                    'colors': 160,
                    'fps_multiplier': 0.9,
                    'resolution_multiplier': 0.92,
                    'lossy': 60,
                    'dither': 'bayer'
                }
            ]
        
        return stages
    
    def get_quality_floors(self) -> Dict[str, Any]:
        """
        Get quality floor constraints.
        
        Returns:
            Dictionary with quality floor settings
        """
        floors = self.config.get('gif_settings.quality_floors', {}) or {}
        
        return {
            'enforce': bool(floors.get('enforce', True)),
            'min_width': int(floors.get('min_width', 320)),
            'min_fps': int(floors.get('min_fps', 18)),
            'min_width_aggressive': int(floors.get('min_width_aggressive', 240)),
            'min_fps_aggressive': int(floors.get('min_fps_aggressive', 12))
        }
    
    def get_segmentation_config(self) -> Dict[str, Any]:
        """
        Get segmentation configuration.
        
        Returns:
            Dictionary with segmentation settings
        """
        seg_cfg = self.config.get('gif_settings.segmentation', {}) or {}
        base_durations = seg_cfg.get('base_durations', {}) or {}
        quality_scaling = seg_cfg.get('quality_scaling', {}) or {}
        estimation = seg_cfg.get('estimation', {}) or {}
        
        return {
            'prefer_single_file_first': bool(seg_cfg.get('prefer_single_file_first', True)),
            'size_threshold_multiplier': float(seg_cfg.get('size_threshold_multiplier', 2.5)),
            'fallback_duration_limit': float(seg_cfg.get('fallback_duration_limit', 180)),
            'min_segment_duration': float(seg_cfg.get('min_segment_duration', 12)),
            'max_segment_duration': float(seg_cfg.get('max_segment_duration', 35)),
            'base_durations': {
                'short_video_max': float(base_durations.get('short_video_max', 40)),
                'short_segment_duration': float(base_durations.get('short_segment_duration', 18)),
                'medium_video_max': float(base_durations.get('medium_video_max', 80)),
                'medium_segment_duration': float(base_durations.get('medium_segment_duration', 22)),
                'long_video_max': float(base_durations.get('long_video_max', 120)),
                'long_segment_duration': float(base_durations.get('long_segment_duration', 25)),
                'very_long_segment_duration': float(base_durations.get('very_long_segment_duration', 28))
            },
            'quality_scaling': {
                'enabled': bool(quality_scaling.get('enabled', True)),
                'long_segment_fps_reduction': float(quality_scaling.get('long_segment_fps_reduction', 0.9)),
                'long_segment_color_reduction': float(quality_scaling.get('long_segment_color_reduction', 0.9))
            },
            'estimation': {
                'default_fps': int(estimation.get('default_fps', 20)),
                'default_colors': int(estimation.get('default_colors', 256)),
                'default_lossy': int(estimation.get('default_lossy', 60)),
                'aggressive_fps': int(estimation.get('aggressive_fps', 15)),
                'aggressive_colors': int(estimation.get('aggressive_colors', 192)),
                'aggressive_lossy': int(estimation.get('aggressive_lossy', 100))
            }
        }
    
    def get_platform_settings(self, platform: str) -> Dict[str, Any]:
        """
        Get platform-specific settings.
        
        Args:
            platform: Platform name (e.g., 'discord', 'twitter', 'slack')
        
        Returns:
            Dictionary with platform-specific settings
        """
        platforms = self.config.get('gif_settings.platforms', {}) or {}
        platform_cfg = platforms.get(platform, {}) or {}
        
        # Get base optimization config as defaults
        base_config = self.get_optimization_config()
        
        # Override with platform-specific settings
        result = base_config.copy()
        result.update({
            'max_width': int(platform_cfg.get('max_width', base_config['width'])),
            'max_height': int(platform_cfg.get('max_height', base_config['height'])),
            'max_duration': float(platform_cfg.get('max_duration', base_config['max_duration_seconds'])),
            'max_file_size_mb': float(platform_cfg.get('max_file_size_mb', base_config['max_file_size_mb'])),
            'colors': int(platform_cfg.get('colors', base_config['colors'])),
            'dither': str(platform_cfg.get('dither', base_config['dither'])),
            'lossy': int(platform_cfg.get('lossy', base_config['lossy']))
        })
        
        return result
    
    def get_adaptive_config(self) -> Dict[str, Any]:
        """
        Get adaptive optimization configuration.
        
        Returns:
            Dictionary with adaptive settings
        """
        adaptive_fps = self.config.get('gif_settings.adaptive_fps', {}) or {}
        adaptive_mpdecimate = self.config.get('gif_settings.adaptive_mpdecimate', {}) or {}
        strategies = self.config.get('gif_settings.optimization_strategies', {}) or {}
        
        return {
            'fps': {
                'enabled': bool(adaptive_fps.get('enabled', True)),
                'motion_analysis': bool(adaptive_fps.get('motion_analysis', True)),
                'fps_range': list(adaptive_fps.get('fps_range', [1.0, 0.9, 0.8, 0.7, 0.6]))
            },
            'mpdecimate': {
                'enabled': bool(adaptive_mpdecimate.get('enabled', True)),
                'motion_adaptive': bool(adaptive_mpdecimate.get('motion_adaptive', True)),
                'frac_range': list(adaptive_mpdecimate.get('frac_range', [0.1, 0.2, 0.3, 0.4, 0.5]))
            },
            'strategies': {
                'variable_fps': bool(strategies.get('variable_fps', True)),
                'motion_based_optimization': bool(strategies.get('motion_based_optimization', True)),
                'content_aware_palette': bool(strategies.get('content_aware_palette', True)),
                'progressive_quality': bool(strategies.get('progressive_quality', True))
            }
        }
    
    def get_single_gif_config(self) -> Dict[str, Any]:
        """
        Get single GIF feasibility check configuration.
        
        Returns:
            Dictionary with single GIF settings
        """
        single_gif = self.config.get('gif_settings.single_gif', {}) or {}
        quick_feasibility = single_gif.get('quick_feasibility', {}) or {}
        
        return {
            'quick_feasibility': {
                'enabled': bool(quick_feasibility.get('enabled', True)),
                'width': int(quick_feasibility.get('width', 240)),
                'fps': int(quick_feasibility.get('fps', 12))
            }
        }
    
    def get_multiprocessing_config(self) -> Dict[str, Any]:
        """
        Get multiprocessing configuration.
        
        Returns:
            Dictionary with multiprocessing settings
        """
        mp_cfg = self.config.get('gif_settings.multiprocessing', {}) or {}
        
        return {
            'max_concurrent_segments': int(mp_cfg.get('max_concurrent_segments', 4)),
            'enabled': bool(mp_cfg.get('enabled', True)),
            'use_dynamic_analysis': bool(mp_cfg.get('use_dynamic_analysis', True)),
            'analysis_mode': str(mp_cfg.get('analysis_mode', 'recommended'))
        }


