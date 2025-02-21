from typing import Dict, Tuple, List

# Quality thresholds
QUALITY_THRESHOLDS = {
    'min_scale_factor': 0.2,
    'min_colors': 32,
    'max_lossy': 100,
    'min_dimension': 64,
    'min_fps': 5,
    'acceptable_overshoot': 1.1
}

# Optimization strategies
OPTIMIZATION_STRATEGIES: Dict[str, List[Tuple]] = {
    'quality_steps': [
        (1.0, 256, 20, 'floyd_steinberg'),
        (0.9, 192, 30, 'floyd_steinberg'),
        (0.8, 128, 40, 'sierra2'),
        (0.7, 96, 60, 'bayer'),
        (0.5, 64, 80, 'bayer'),
        (0.3, 48, 90, 'bayer'),
        (0.2, 32, 100, 'bayer')
    ],
}

# Resource limits
RESOURCE_LIMITS = {
    'memory_threshold': 85.0,  # Percent
    'cpu_threshold': 90.0,     # Percent
    'max_threads': 8,
    'memory_per_thread': 500   # MB
}

# Cache settings
CACHE_CONFIG = {
    'frame_cache_size': 100,
    'frame_cache_ttl': 300,    # seconds
    'dimension_cache_size': 100,
    'dimension_cache_ttl': 900  # seconds
}

# FFmpeg settings
FFMPEG_CONFIG = {
    'timeout': 120,            # seconds
    'max_retries': 3,
    'retry_delay': 1.0,       # seconds
    'thread_count': 4
}
