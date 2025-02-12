# DisDrop: Video and GIF Compression Tool

DisDrop is a powerful tool designed to compress and optimize videos and GIFs, enabling them to meet the 10MB file size constraints commonly imposed by social media platforms. It features adaptive compression, hardware acceleration, and intelligent file handling.

## Features

### Video Optimization
- **Smart Compression**:
  - Multi-stage compression with dynamic quality adjustment
  - Two-pass encoding for large files (>100MB)
  - Adaptive bitrate and CRF based on file characteristics
  - Progressive quality reduction until target size is reached
  - Maintains maximum possible quality while achieving target size

- **Intelligent Scaling**:
  - Dynamic resolution scaling based on content
  - Preserves aspect ratio and quality
  - Extreme compression mode for near-target files
  - Smart dimension adjustment for optimal quality

- **Optimization Features**:
  - Skip already compressed files meeting size requirements
  - Process files in size order (largest first)
  - Temporary file handling for safe processing
  - Multiple compression attempts with increasing aggressiveness

### Hardware Acceleration
- NVIDIA GPU (NVENC) support for faster processing
- Automatic fallback to CPU if GPU unavailable
- Optimized encoding presets for both GPU and CPU

### System Features
- Detailed progress logging with color coding
- Efficient temp file management
- Error recovery and cleanup
- Support for interrupted processing

## Requirements

### System Requirements
- Python 3.8+
- FFmpeg with NVENC support (optional, for GPU acceleration)
- 4GB RAM minimum (8GB+ recommended)
- NVIDIA GPU (optional, for hardware acceleration)

### Python Dependencies
- `concurrent.futures`
- `dataclasses`
- `functools`
- `logging`
- `pathlib`
- `shutil`
- `subprocess`
- `argparse`
- `signal`

## Installation

1. Clone this repository:
   ```bash
   git clone <[repository-url](https://github.com/0-FoxHunt-0/disdrop.git)>
   cd <disdrop>
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure ffmpeg, ffprobe, and gifsicle are installed and added to your PATH.

## Configuration

All configurations are stored in `default_config.py`. Users can modify this file to suit their specific requirements, such as changing input/output paths or adjusting compression quality.

## Usage

Run the main script:
```bash
python main.py [options]
```

### Options
- `--debug`: Enable debug logging.
- `--no-gpu`: Disable GPU acceleration.
- `--input-dir`: Specify a custom input directory.
- `--output-dir`: Specify a custom output directory.

### Example
Optimize files with GPU acceleration and custom directories:
```bash
python main.py --input-dir ./my_videos --output-dir ./optimized_files
```

## File Structure
```
project/
├── main.py                # Main entry point of the application
├── gpu_acceleration.py    # GPU setup and checks for NVENC
├── gif_optimization.py    # GIF processing and optimization logic
├── video_optimization.py  # Video compression and conversion logic
├── logging_system.py      # Logging and debugging utilities
├── temp_file_manager.py   # Temporary file management
├── default_config.py      # Configuration settings
└── logs/                  # Logs directory
└── input/                 # Default input directory
└── output/                # Default output directory
└── temp_files/            # Temporary files directory
```

## Error Handling and Cleanup

The tool supports graceful shutdown through SIGINT or SIGTERM signals. Temporary files are automatically cleaned up at the end of processing or upon an error. However, in rare cases, such as abrupt system shutdowns or permission issues, manual cleanup of the temporary directory might be required.

## Contributing

Currently, there is no formal contributing system in place. However, feel free to submit issues and pull requests for improvements. Contributions are welcome, and a CONTRIBUTING.md file may be added in the future to provide guidelines.

## License

Currently, no license is applied to this project. Users should seek permission before using or modifying this code. A license may be added in the future to clarify usage terms.

