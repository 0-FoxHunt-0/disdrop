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
cd video-compressor
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

## Advanced Features

### Iterative Quality Optimization for GIFs

The tool now includes an advanced iterative optimization system that automatically tries to generate the best quality GIF possible while staying within size limits. This feature performs multiple optimization attempts with different quality levels to find the optimal balance.

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

#### Configuration

The quality optimization can be configured in `config/gif_settings.yaml`:

```yaml
gif_settings:
  quality_optimization:
    enabled: true  # Enable/disable the feature
    max_attempts_per_stage: 5  # Attempts per quality level
    target_size_efficiency: 0.9  # Target 90% of max size for best quality
    quality_stages:
      - name: "Maximum Quality"
        colors: 256
        fps_multiplier: 2.0
        resolution_multiplier: 1.5
        lossy: 0
        dither: "floyd_steinberg"
      - name: "High Quality"
        colors: 256
        fps_multiplier: 1.5
        resolution_multiplier: 1.2
        lossy: 20
        dither: "floyd_steinberg"
      # ... more stages
```

#### Usage Examples

```bash
# Basic quality optimization
python main.py gif input.mp4 output.gif --max-size 8.0

# Platform-specific optimization
python main.py gif input.mp4 output.gif --platform discord --max-size 8.0

# Custom time range with quality optimization
python main.py gif input.mp4 output.gif --max-size 5.0 --start 10.0 --duration 15.0

# Using the example script
python example_quality_optimization.py
```

#### Example Output

```
Starting quality optimization process (target size: 8.0MB)
Quality optimization stage 1/4: Maximum Quality
  Attempt 1/5
    Success! Size: 7.2MB, Quality Score: 0.85
    New best result! Quality score: 0.85
Maximum quality achieved within size limit (90%) - stopping optimization
Final result: 7.2MB with quality score 0.85
```

#### Python API Usage

```python
from src.gif_optimizer_advanced import AdvancedGifOptimizer
from src.config_manager import ConfigManager

# Initialize
config = ConfigManager()
optimizer = AdvancedGifOptimizer(config)

# Create quality-optimized GIF
result = optimizer.optimize_gif_with_quality_target(
    input_video="input.mp4",
    output_path="output.gif",
    max_size_mb=5.0,
    quality_preference='balanced',
    platform='twitter'
)

print(f"Quality score: {result['quality_score']:.2f}/10")
print(f"Size efficiency: {result['size_efficiency']:.1%}")
```

### Platform-Specific Optimization

The tool automatically applies platform-specific optimizations:

- **Twitter**: 506x506 max, 15s duration, optimized for web viewing
- **Discord**: 400x400 max, 10s duration, 8MB limit for free users
- **Slack**: 360x360 max, 8s duration, conservative compression
- **Instagram**: 1080x1080 max, optimized for mobile viewing
- **TikTok**: 1080x1920 max, vertical format optimization

### Hardware Acceleration

The tool automatically detects and utilizes available hardware:

- **NVIDIA GPU**: NVENC encoder for fast compression
- **AMD GPU**: AMF encoder support
- **Intel GPU**: QSV encoder for integrated graphics
- **Apple Silicon**: VideoToolbox for M1/M2 Macs

### Batch Processing

Process multiple files at once:

```bash
# Batch compress videos
python main.py batch-compress "*.mp4" --output-dir compressed/ --platform instagram

# Batch create GIFs
python main.py batch-gif "*.mp4" --output-dir gifs/ --platform twitter --duration 10

# Parallel processing
python main.py batch-compress "*.mp4" --output-dir compressed/ --parallel 4
```

### Automated Workflow

Monitor a directory for new files and process them automatically:

```bash
# Start automated workflow
python main.py auto --check-interval 10 --max-size 10.0
```

This will:
1. Monitor the `input/` directory for new video files
2. Automatically compress them based on platform settings
3. Create optimized GIFs if requested
4. Move results to `output/` directory
5. Clean up temporary files

## Configuration

The tool uses YAML configuration files in the `config/` directory:

- `video_compression.yaml`: Video compression settings
- `gif_settings.yaml`: GIF generation settings
- `logging.yaml`: Logging configuration

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

## Command Line Interface

### Video Compression Commands

```bash
# Basic compression
python main.py compress input.mp4 output.mp4 --platform instagram

# Custom quality settings
python main.py compress input.mp4 output.mp4 --quality 23 --bitrate 1000k

# Custom resolution
python main.py compress input.mp4 output.mp4 --resolution 1080x1080 --fps 30
```

### GIF Commands

```bash
# Basic GIF creation
python main.py gif input.mp4 output.gif --platform twitter

# Quality-optimized GIF
python main.py quality-gif input.mp4 output.gif --target-size 5.0 --quality-preference balanced

# Custom time range
python main.py gif input.mp4 output.gif --start 10.0 --duration 15.0

# Optimize existing GIF
python main.py optimize-gif input.gif output.gif
```

### Utility Commands

```bash
# Show hardware information
python main.py hardware-info

# Show configuration
python main.py config show

# Validate configuration
python main.py config validate
```

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

## Troubleshooting

### Common Issues

1. **FFmpeg not found**: Install FFmpeg and ensure it's in your system PATH
2. **GPU acceleration not working**: Check if your GPU drivers are up to date
3. **Large file sizes**: Use quality optimization or reduce target resolution
4. **Poor quality**: Increase quality settings or use quality-gif command

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
