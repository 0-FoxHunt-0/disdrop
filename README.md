# Video and GIF Optimization Tool

This repository offers a powerful tool for reducing file sizes and maintaining quality in video and GIF content, making it ideal for creators, developers, and teams handling large media files. The tool supports GPU acceleration, multi-pass optimization, and detailed logging for a seamless processing experience.

## Features

- **Video Optimization**:
  - Compress videos using ffmpeg with support for NVENC (GPU acceleration).
  - Multi-pass processing for fine-tuned compression.
  - Dynamic bitrate and CRF adjustments for optimal size and quality.

- **GIF Optimization**:
  - Convert videos to GIFs with high-quality palettes and scaling.
  - Optimize GIFs using gifsicle with configurable settings.
  - Support for multi-pass attempts for failed files.

- **Hardware Acceleration**:
  - Leverages NVIDIA GPU and NVENC for faster processing, significantly reducing encoding time and enabling smoother handling of high-resolution videos compared to CPU-only processing.
  - Falls back to CPU processing if GPU is unavailable.

- **Robust Logging**:
  - Logs system and processing events in color-coded and file-based formats.
  - Dedicated logs for ffmpeg commands.

- **Error Handling**:
  - Graceful shutdown and cleanup of temporary files.
  - Detailed error reporting for failed files.

## Requirements

### Dependencies
- [ffmpeg](https://ffmpeg.org/) (with NVENC support for GPU acceleration)
- [ffprobe](https://ffmpeg.org/ffprobe.html)
- [gifsicle](https://www.lcdf.org/gifsicle/)
- Python 3.8+

### Python Libraries
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
   git clone <repository-url>
   cd <repository-name>
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

