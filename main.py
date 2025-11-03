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

# Force UTF-8 encoding for console output
if sys.platform.startswith('win'):
    # Set console code page to UTF-8 on Windows
    os.system('chcp 65001 > nul')
    # Reconfigure stdout to use UTF-8
    if hasattr(sys.stdout, 'reconfigure'):
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except Exception:
            pass
    if hasattr(sys.stderr, 'reconfigure'):
        try:
            sys.stderr.reconfigure(encoding='utf-8')
        except Exception:
            pass

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.cli import main

if __name__ == '__main__':
    main() 