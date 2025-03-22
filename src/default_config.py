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
    'min_size_mb': 15.0,      # Target maximum size for gifs in MB
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
    'fps_range': (10, 30),  # Frame rate range for gif optimization
    'colors': 224,          # Lowered from 256 to avoid file size increase in lossless mode
    'lossy_value': 0,       # Prioritize scaling over lossy compression
    'min_size_mb': 10.0,    # Target maximum size for gifs in MB
    'min_width': 0,
    'min_height': 120,
    'quality': 95,          # Increased from 85 to preserve quality
    'scale_priority': True  # New flag to prioritize scaling over other reductions
}

# Strategies for multiple pass overs after failed attempts - updated to prioritize scaling
GIF_PASS_OVERS = [
    {
        'fps_range': (15, 15),
        'colors': 224,
        'lossy_value': 0,
        'scale_factor': 0.85,
    },
    {
        'fps_range': (15, 12),
        'colors': 192,
        'lossy_value': 0,
        'scale_factor': 0.7,
    },
    {
        'fps_range': (12, 12),
        'colors': 192,
        'lossy_value': 30,
        'scale_factor': 0.6,
    },
    {
        'fps_range': (12, 10),
        'colors': 160,
        'lossy_value': 50,
        'scale_factor': 0.5,
    },
    {
        'fps_range': (10, 10),
        'colors': 128,
        'lossy_value': 60,
        'scale_factor': 0.4,
    },
]

# Additional settings for file handling
GIF_SIZE_TO_SKIP = 100  # Max size for gifs before skipping gif optimization with gifsicle

# File naming and logging settings
LOG_FILE = LOG_DIR / 'processing.log'
FFPMEG_LOG_FILE = LOG_DIR / 'ffmpeg.log'
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.avi', '.mkv', '.mov', '.wmv']
