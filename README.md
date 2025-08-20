# Video Compressor & GIF Generator

A comprehensive Python tool for compressing videos and creating optimized GIFs for social media platforms, with intelligent hardware detection and automated workflows.

## Features

- **Video Compression**: Optimize videos for various social media platforms
- **GIF Generation**: Create high-quality GIFs from videos with iterative quality optimization
- **Hardware Acceleration**: Automatic detection and utilization of GPU acceleration
- **Platform Optimization**: Tailored settings for Twitter, Instagram, TikTok, YouTube Shorts, Facebook, Discord, and Slack
- **Batch Processing**: Process multiple files automatically
- **Automated Workflow**: Monitor input directory for new files
- **Quality Optimization**: Iterative GIF optimization to maximize quality while staying under size limits

## Installation

1. Clone the repository:

```bash
git clone <repository-url>
cd disdrop
```

2. Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Ensure FFmpeg is installed on your system:
   - **Windows**: Download from https://ffmpeg.org/download.html
   - **macOS**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg` (Ubuntu/Debian)

## Quick Start

### Zero-argument auto mode

```bash
python main.py
```

Watches `input/` and processes new files to `output/` (defaults: max size 10 MB, check every 5s). Stop with Ctrl+C.

### Basic Video Compression

```bash
python main.py compress input.mp4 output.mp4 --platform instagram
```

### Basic GIF Creation

```bash
python main.py gif input.mp4 output.gif --platform twitter
```

### Quality-Optimized GIF Creation

```bash
python main.py gif input.mp4 output.gif --max-size 5.0
```

### Quality-optimized GIF (explicit target)

```bash
python main.py quality-gif input.mp4 output.gif --target-size 5.0 --quality-preference balanced [--platform discord]
```

### Optimize an existing GIF

```bash
python main.py optimize-gif input.gif output.gif
```

### Batch Processing

```bash
python main.py batch-compress "*.mp4" --output-dir compressed/ --platform instagram --parallel 4
python main.py batch-gif "*.mp4" --output-dir gifs/ --platform twitter --duration 8 --parallel 4
```

### Automated workflow (explicit)

```bash
python main.py auto --check-interval 10 --max-size 10.0
```

## How It Works

### Video Compression

- Content analysis (ffprobe) informs encoder, bitrate, FPS, and resolution
- Hardware acceleration is used when available with robust software fallback
- If a single output cannot reasonably meet target size, the system may invoke the video segmenter as a last resort

Video segmentation behavior (configurable in `config/video_compression.yaml`):

- Default trigger: estimated size ≥ 3.0× target, or extremely long videos (≥ 10 minutes)
- Segment duration stays within bounds and is tuned by content complexity and motion

### GIF Creation and Optimization

- Generates high-quality GIFs with a palette pipeline, `mpdecimate`, and aspect-ratio preservation (`force_original_aspect_ratio=decrease`, `setsar=1`)
- If `--max-size` is given, an iterative optimizer searches multiple quality stages to land under the target with best visual fidelity
- Platforms supported by CLI: `twitter`, `discord` (8 MB), `slack`. Internal presets also exist for Telegram/Reddit
- GIF segmentation is adaptive: long/complex source clips can be split into multiple GIFs; nested segmentation is avoided

### Iterative Quality Optimization for GIFs

The tool includes an advanced iterative optimization system that automatically tries to generate the best quality GIF possible while staying within size limits. This feature performs multiple optimization attempts with different quality levels to find the optimal balance.

#### How It Works

1. **Quality Stages**: The system tries multiple quality levels in order:

   - **Maximum Quality**: Highest resolution, FPS, colors, and best dithering
   - **High Quality**: Good balance of quality and size
   - **Medium Quality**: Moderate settings for smaller files
   - **Standard Quality**: Basic settings as fallback

2. **Iterative Attempts**: For each quality stage:

   - Attempts multiple GIF generation methods (FFmpeg palette, direct conversion, OpenCV+PIL)
   - If the result is under the size limit, calculates a quality score
   - If too large, reduces parameters and tries again
   - Continues until finding the best quality result within size limits

3. **Quality Scoring**: Each successful attempt is scored based on:

   - Color depth (0-256 colors)
   - Frame rate quality (up to 30fps)
   - Resolution quality (up to 800x800)
   - Size efficiency (how close to target size)
   - Compression quality (lossy settings)
   - Dithering quality (Floyd-Steinberg vs Bayer)

4. **Best Result Selection**: Chooses the result with the highest quality score that meets size requirements

## CLI Overview

### Global Options

- `--log-level [DEBUG|INFO|WARNING|ERROR]`, `--debug` (verbose console + file logs)
- `--temp-dir PATH` (defaults to `./temp`)
- `--output-dir PATH` (global output directory for all modes)
- `--max-size MB` (global size cap applied to both video and GIF where relevant)
- `--force-software` (bypass hardware encoders)

### Commands

- `compress <in> <out> [--platform ...] [--encoder ...] [--quality CRF] [--bitrate 1000k] [--resolution WxH] [--fps N]`
- `gif <in> <out> [--platform twitter|discord|slack] [--start S] [--duration S] [--max-size MB] [--fps N] [--width W] [--height H] [--colors C]`
- `quality-gif <in> <out> --target-size MB [--quality-preference quality|balanced|size] [--platform ...] [--start] [--duration]`
- `optimize-gif <in> <out>`
- `batch-compress <glob> --output-dir DIR [--platform ...] [--suffix _compressed] [--parallel N]`
- `batch-gif <glob> --output-dir DIR [--platform ...] [--duration S] [--parallel N]`
- `hardware-info`
- `config show|validate`
- `auto [--check-interval S] [--max-size MB]`

## Configuration

Configuration lives in `config/`:

- `video_compression.yaml`: max size (default 10 MB), hardware encoders, conservative segmentation policy, platform presets
- `gif_settings.yaml`: default max size (10 MB), FPS/resolution, palette/dither/lossy, quality-optimization stages, segmentation heuristics
- `logging.yaml`: console/file logging formats and levels (logs in `logs/`)

### Examples (abridged)

```yaml
# video_compression.yaml
video_compression:
  max_file_size_mb: 10
  segmentation:
    size_threshold_multiplier: 3.0
    fallback_duration_limit: 600

# gif_settings.yaml
gif_settings:
  max_file_size_mb: 10
  fps: 20
  width: 360
  height: 360
  quality_optimization:
    enabled: true
```

CLI overrides map into config at runtime, for example:

- `--max-size 8.0` → `video_compression.max_file_size_mb` and `gif_settings.max_file_size_mb`
- `--fps 20` → `video_compression.platforms.custom.fps` and `gif_settings.fps`
- `--width/--height/--colors` → `gif_settings.width/height/colors`

### Custom Platform Configuration

Add custom platform settings in `gif_settings.yaml`:

```yaml
platforms:
  custom_platform:
    max_width: 500
    max_height: 500
    max_duration: 12
    colors: 128
    max_file_size_mb: 6
    dither: "floyd_steinberg"
    lossy: 30
```

## Platform-Specific Optimization

The tool automatically applies platform-specific optimizations:

- **Twitter**: 506x506 max, 15s duration, optimized for web viewing
- **Discord**: 400x400 max, 10s duration, 8MB limit for free users
- **Slack**: 360x360 max, 8s duration, conservative compression
- **Instagram**: 1080x1080 max, optimized for mobile viewing
- **TikTok**: 1080x1920 max, vertical format optimization

## Hardware Acceleration

The tool automatically detects and utilizes available hardware:

- **NVIDIA GPU**: NVENC encoder for fast compression
- **AMD GPU**: AMF encoder support
- **Intel GPU**: QSV encoder for integrated graphics
- **Apple Silicon**: VideoToolbox for M1/M2 Macs

## Examples

### Example 1: Social Media Content Creation

```bash
# Create Instagram-optimized video
python main.py compress raw_video.mp4 instagram_video.mp4 --platform instagram

# Create Twitter GIF with quality optimization
python main.py quality-gif raw_video.mp4 twitter_gif.gif --target-size 5.0 --quality-preference balanced --platform twitter
```

### Example 2: Batch Processing

```bash
# Process all videos in directory for multiple platforms
python main.py batch-compress "videos/*.mp4" --output-dir instagram/ --platform instagram
python main.py batch-compress "videos/*.mp4" --output-dir twitter/ --platform twitter
python main.py batch-gif "videos/*.mp4" --output-dir gifs/ --platform discord --duration 8
```

### Example 3: Automated Workflow

```bash
# Start automated processing
python main.py auto --check-interval 30 --max-size 8.0

# Place videos in input/ directory and they'll be processed automatically
```

## Performance Tips

1. **Use Hardware Acceleration**: The tool automatically detects and uses GPU acceleration when available
2. **Batch Processing**: Use batch commands for multiple files to save time
3. **Quality Optimization**: Use the quality-gif command for best results when file size is critical
4. **Platform-Specific Settings**: Always specify the target platform for optimal results
5. **Parallel Processing**: Use `--parallel` flag for batch operations on multi-core systems
6. **Valid existing outputs in `output/` are reused to save time**

## Troubleshooting

### Common Issues

- **FFmpeg not found**: Install FFmpeg and ensure it's in your system PATH
- **Hardware encoder errors**: app falls back to software; use `--force-software` to skip hardware
- **Outputs slightly over target**: try `quality-gif --target-size ...` or reduce FPS/scale/colors
- **Large file sizes**: Use quality optimization or reduce target resolution
- **Poor quality**: Increase quality settings or use quality-gif command

### Debug Mode

Enable debug logging for detailed information:

```bash
python main.py compress input.mp4 output.mp4 --log-level DEBUG
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- FFmpeg for video processing capabilities
- OpenCV for computer vision features
- Pillow for image processing
- PyYAML for configuration management
