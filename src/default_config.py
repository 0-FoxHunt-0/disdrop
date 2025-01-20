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
# Temporary directory for storing intermediate files
TEMP_FILE_DIR = BASE_DIR / 'temp_files'

# Ensure that the directories exist
for directory in [INPUT_DIR, OUTPUT_DIR, LOG_DIR, TEMP_FILE_DIR]:
    directory = Path(directory)
    if not directory.exists():
        os.makedirs(directory)

# Optimization settings for videos and gifs
VIDEO_COMPRESSION = {
    'scale_factor': 0.8,
    'crf': 23,
    'gpu_acceleration': True,
    'fallback_settings': {
        'scale_factor': 0.8,
        'crf': 26,
        'preset': 'medium'
    }
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
    'fps_range': (10, 15),  # Frame rate range for gif optimization
    'colors': 256,          # Number of colors for gif optimization
    'lossy_value': 55,      # Lossy value for gif compression
    'min_size_mb': 10,      # Target maximum size for gifs in MB
    'min_width': 0,
    'min_height': 120
}

# Strategies for multiple pass overs after failed attempts
GIF_PASS_OVERS = [
    {
        'fps_range': (10, 15),
        'colors': 192,
        'lossy_value': 55,
        'scale_factor': 0.8,
    },
    {
        'fps_range': (10, 15),
        'colors': 192,
        'lossy_value': 80,
        'scale_factor': 0.8,
    },
    {
        'fps_range': (10, 15),
        'colors': 128,
        'lossy_value': 80,
        'scale_factor': 0.8,
    },
    {
        'fps_range': (10, 15),
        'colors': 64,
        'lossy_value': 80,
        'scale_factor': 0.8,
    },
    {
        'fps_range': (5, 10),
        'colors': 64,
        'lossy_value': 80,
        'scale_factor': 0.5,
    },
]

# Additional settings for file handling
GIF_SIZE_TO_SKIP = 90  # Max size for gifs before skipping gif optimization with gifsicle

# File naming and logging settings
LOG_FILE = LOG_DIR / 'disdrop.log'
FFPMEG_LOG_FILE = LOG_DIR / 'ffmpeg.log'
SUPPORTED_VIDEO_FORMATS = ['.mp4', '.mkv',
                           '.m4v', '.avi', '.mov', '.wmv', '.flv']
