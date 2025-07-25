#!/usr/bin/env python3
"""
Video Compressor - Main Entry Point
Compress videos and create GIFs optimized for social media platforms

When run without arguments, automatically processes videos from the 'input' folder:
1. Converts all videos to optimized MP4 format (if not already MP4)
2. Generates and optimizes GIFs from the processed videos
3. Ensures all outputs are under 10MB and not corrupted
4. Supports graceful shutdown with Ctrl+C
"""

import sys
import os

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cli import main

if __name__ == '__main__':
    main() 