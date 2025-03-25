# config.py

import os
from pathlib import Path

# Get the src directory path
SRC_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
# Get the project root directory (parent of src)
BASE_DIR = SRC_DIR.parent

# Define the directories relative to the project root
INPUT_DIR = BASE_DIR / 'input'
OUTPUT_DIR = BASE_DIR / 'output'
LOG_DIR = BASE_DIR / 'logs'
TEMP_FILE_DIR = BASE_DIR / 'temp'  # Changed from 'temp_files' to 'temp'

# Optimization settings for videos and gifs
VIDEO_COMPRESSION = {
    'scale_factor': 0.8,
    'crf': 23,
    'gpu_acceleration': True,
    'fallback_settings': {
        'scale_factor': 0.8,
        'crf': 26,
        'preset': 'medium'
    },
    'min_size_mb': 10.0,      # Target maximum size for gifs in MB
    'fps_range': (15, 30),
    'quality_presets': ['faster', 'medium', 'slower']
}

VIDEO_SETTINGS = {
    'gpu': {
        'enabled': True,
        'encoders': ['h264_nvenc', 'hevc_nvenc'],
        'presets': {
            'quality': {'preset': 'p7', 'crf': 18},
            'balanced': {'preset': 'p4', 'crf': 23},
            'speed': {'preset': 'p1', 'crf': 28}
        }
    },
    'cpu': {
        'encoders': ['libx264', 'libx265'],
        'presets': {
            'quality': {'preset': 'slower', 'crf': 18},
            'balanced': {'preset': 'medium', 'crf': 23},
            'speed': {'preset': 'veryfast', 'crf': 28}
        }
    },
    'general': {
        'scale_factor': 0.8,
        'max_retries': 3,
        'timeout': 3600
    }
}

GIF_COMPRESSION = {
    'fps_range': (20, 15),    # Good middle ground for quality vs size
    'colors': 256,            # Keep max colors for better quality
    # Balanced compression (lower than 55 but higher than 20)
    'lossy_value': 35,
    'min_size_mb': 10,        # Return to 10MB limit as required
    'min_width': 320,         # Keep higher resolution minimum width
    'min_height': 240,        # Keep higher resolution minimum height
    'quality_priority': True  # New flag to optimize for quality within size constraints
}

# Strategies for multiple pass overs after failed attempts - updated for better quality
GIF_PASS_OVERS = [
    {  # First pass: try higher FPS with full colors but higher lossy value
        'fps_range': (18, 15),
        'colors': 256,
        'lossy_value': 40,
        'scale_factor': 0.9,
    },
    {  # Second pass: reduce FPS slightly, keep colors, increase compression
        'fps_range': (15, 15),
        'colors': 224,
        'lossy_value': 50,
        'scale_factor': 0.8,
    },
    {  # Third pass: reduce FPS more, lower colors slightly
        'fps_range': (12, 12),
        'colors': 192,
        'lossy_value': 60,
        'scale_factor': 0.7,
    },
    {  # Fourth pass: further reductions while maintaining recognizability
        'fps_range': (10, 10),
        'colors': 160,
        'lossy_value': 70,
        'scale_factor': 0.6,
    },
    {  # Last resort: aggressive optimization
        'fps_range': (8, 8),
        'colors': 128,
        'lossy_value': 80,
        'scale_factor': 0.5,
    },
]

# Additional settings for file handling
GIF_SIZE_TO_SKIP = 100  # Max size for gifs before skipping gif optimization with gifsicle

# File naming and logging settings
LOG_FILE = LOG_DIR / 'processing.log'
FFPMEG_LOG_FILE = LOG_DIR / 'ffmpeg.log'
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mkv', '.mov', '.wmv']
