# Video Compressor & GIF Generator

A comprehensive Python tool for compressing videos and creating optimized GIFs for social media platforms, with intelligent hardware detection and automated workflows.

## What's New - Simplified CLI

The CLI has been streamlined for easier use while maintaining full functionality:

✨ **Smart Auto-Detection**

- Glob patterns (`*.mp4`) automatically trigger batch mode
- Unified `gif` command auto-detects operation type (create/optimize/quality)
- No need for separate `batch-compress` or `batch-gif` commands

⚡ **Command Aliases**

- `c` / `v` / `video` → `compress`
- `g` / `a` / `anim` → `gif`
- `w` / `watch` / `m` / `monitor` → `auto`
- `hw` / `i` / `info` → `hardware-info`
- `cfg` → `config`, `ch` → `cache`
- `open` → Open application folders in file explorer

🎯 **Short Flags**

- `-p` → `--platform`
- `-s` → `--max-size`
- `-o` → `--output-dir`
- `-i` → `--input-dir`
- `-j` → `--parallel`
- `-d` → `--duration`
- `-q` → `--quality`
- `-f` → `--fps`
- `-v` → `--debug`

**Example:** `python main.py g "*.mp4" -o gifs/ -p discord -j 3`

All old commands still work for backward compatibility!

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

5. (Optional) Install globally for use from anywhere:

```bash
pip install --user -e .
```

After installation, you can use `disdrop` command from anywhere instead of `python main.py`:

```bash
disdrop auto           # Start automated workflow
disdrop open -i -o     # Open input and output folders
disdrop hw             # Show hardware info
```

## Quick Start

### Zero-argument auto mode

```bash
python main.py
# or use short alias:
python main.py w
```

Watches `input/` and processes new files to `output/` (defaults: max size 10 MB, check every 5s). Stop with Ctrl+C.

### Basic Video Compression

```bash
# Long form
python main.py compress input.mp4 output.mp4 --platform instagram

# Short form with aliases
python main.py c input.mp4 output.mp4 -p instagram
```

### Batch Video Compression (Auto-Detected)

```bash
# Glob patterns automatically trigger batch mode
python main.py c "*.mp4" -o compressed/ -p tiktok -j 4
```

### Basic GIF Creation

```bash
# Long form
python main.py gif input.mp4 output.gif --platform twitter

# Short form with aliases
python main.py g input.mp4 output.gif -p twitter -d 10
```

### Batch GIF Creation (Auto-Detected)

```bash
# Glob patterns automatically trigger batch mode
python main.py g "*.mp4" -o gifs/ -p discord -j 3
```

### Quality-Optimized GIF Creation

```bash
# With max-size (iterative optimization)
python main.py g input.mp4 output.gif -s 5.0 -p twitter

# With target-size (advanced optimization)
python main.py g input.mp4 output.gif --target-size 5.0 --quality-preference balanced
```

### Optimize an existing GIF

```bash
# Auto-detected when both input and output are GIFs
python main.py g input.gif output.gif
```

### Automated workflow

```bash
# Long form
python main.py auto --check-interval 10 -s 8

# Short aliases
python main.py w --check-interval 10 -s 8
python main.py watch --check-interval 10 -s 8
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

### Smart Features

- **Auto-Batch Detection**: Glob patterns (`*.mp4`) automatically trigger batch mode
- **Unified GIF Command**: Single `gif` command auto-detects optimization type
- **Command Aliases**: Short versions for faster typing
- **Short Flags**: Single-letter options for common flags

### Global Options

| Long Form           | Short | Description                             |
| ------------------- | ----- | --------------------------------------- |
| `--debug`           | `-v`  | Verbose console + file logs             |
| `--max-size MB`     | `-s`  | Global size cap for output files        |
| `--output-dir PATH` | `-o`  | Output directory for all modes          |
| `--input-dir PATH`  | `-i`  | Input directory for auto mode           |
| `--temp-dir PATH`   |       | Temporary directory (default: `./temp`) |
| `--force-software`  |       | Bypass hardware encoders                |
| `--no-cache`        |       | Don't use success cache                 |

### Commands and Aliases

| Command         | Aliases                      | Description              |
| --------------- | ---------------------------- | ------------------------ |
| `compress`      | `c`, `v`, `video`            | Compress video file(s)   |
| `gif`           | `g`, `a`, `anim`             | Create/optimize GIF(s)   |
| `auto`          | `w`, `watch`, `m`, `monitor` | Watch input folder       |
| `open`          | -                            | Open app folders         |
| `hardware-info` | `hw`, `i`, `info`            | Show hardware info       |
| `config`        | `cfg`                        | Configuration management |
| `cache`         | `ch`                         | Cache operations         |

### Command Details

#### compress / c / v / video

Compress video file(s). Auto-detects batch mode from glob patterns.

```bash
compress <input> [output] [options]
```

**Options:**

- `-p, --platform` - Target platform (instagram, twitter, tiktok, youtube_shorts, facebook)
- `-q, --quality CRF` - Quality setting (lower = better)
- `-f, --fps N` - Target frame rate
- `--encoder` - Force specific encoder
- `--bitrate` - Target bitrate (e.g., 1000k)
- `--resolution` - Target resolution (e.g., 1080x1080)
- `--suffix` - Suffix for batch output files (default: \_compressed)
- `-j, --parallel N` - Number of parallel processes for batch

#### gif / g / a / anim

Create or optimize GIF(s). Auto-detects operation type and batch mode.

```bash
gif <input> [output] [options]
```

**Options:**

- `-p, --platform` - Target platform (twitter, discord, slack)
- `-s, --max-size MB` - Max file size (enables quality optimization)
- `--target-size MB` - Target size for advanced optimization
- `-d, --duration S` - Duration in seconds
- `--start S` - Start time in seconds
- `-f, --fps N` - Frame rate
- `--width W` - Width in pixels
- `--height H` - Height in pixels
- `--colors C` - Number of colors in palette
- `--quality-preference` - Strategy: quality, balanced, size
- `-j, --parallel N` - Number of parallel processes for batch

#### auto / w / watch / m / monitor

Watch input folder and automatically process new files.

```bash
auto [options]
```

**Options:**

- `--check-interval S` - Check frequency in seconds (default: 5)
- `-s, --max-size MB` - Max output size (default: 10.0)
- `-i, --input-dir PATH` - Input directory (default: ./input)
- `-o, --output-dir PATH` - Output directory (default: ./output)
- `--no-cache` - Don't use success cache

#### open

Open application folders in file explorer. Useful for quickly accessing input, output, logs, or config directories.

```bash
open [options]
```

**Options:**

- `-i, --input` - Open input folder
- `-o, --output` - Open output folder
- `-l, --logs` - Open logs folder
- `-c, --config` - Open config folder

**Examples:**

```bash
# Open single folder
disdrop open -i                # Open input folder
disdrop open -o                # Open output folder

# Open multiple folders at once
disdrop open -i -o -l          # Open input, output, and logs folders
```

Note: Folders are created if they don't exist. Works cross-platform (Windows, macOS, Linux).

#### Other Commands

- `hardware-info` / `hw` / `i` / `info` - Display hardware acceleration info
- `config` / `cfg` `show|validate` - Configuration management
- `cache` / `ch` `clear|stats|validate` - Cache operations

### Quick Reference Card

Common tasks with shortest syntax:

```bash
# Compress single video for Instagram
python main.py c video.mp4 out.mp4 -p instagram

# Batch compress all videos for TikTok
python main.py c "*.mp4" -o compressed/ -p tiktok -j 4

# Create GIF for Twitter
python main.py g video.mp4 out.gif -p twitter -s 5

# Batch create GIFs for Discord
python main.py g "*.mp4" -o gifs/ -p discord -j 3

# Optimize existing GIF
python main.py g input.gif output.gif

# Watch folder for new files
python main.py w

# Open folders in file explorer
python main.py open -i -o

# Check hardware capabilities
python main.py hw

# Show configuration
python main.py cfg show

# Clear cache
python main.py ch clear
```

## Configuration

Default configuration files are packaged under `disdrop/config/` and used automatically. To override, use `--config-dir PATH` or set `DISDROP_BASE_DIR` to change the runtime base (for logs/input/output/temp).

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
# Create Instagram-optimized video (long form)
python main.py compress raw_video.mp4 instagram_video.mp4 --platform instagram

# Create Instagram-optimized video (short form)
python main.py c raw_video.mp4 instagram_video.mp4 -p instagram

# Create Twitter GIF with quality optimization (long form)
python main.py gif raw_video.mp4 twitter_gif.gif --target-size 5.0 --quality-preference balanced -p twitter

# Create Twitter GIF with quality optimization (short form)
python main.py g raw_video.mp4 twitter_gif.gif --target-size 5.0 -p twitter
```

### Example 2: Batch Processing (Auto-Detected)

```bash
# Process all videos for Instagram (auto-detects batch mode from glob pattern)
python main.py c "videos/*.mp4" -o instagram/ -p instagram -j 4

# Process all videos for Twitter
python main.py c "videos/*.mp4" -o twitter/ -p twitter -j 4

# Create GIFs for Discord from all videos
python main.py g "videos/*.mp4" -o gifs/ -p discord -d 8 -j 3
```

### Example 3: Automated Workflow

```bash
# Start automated processing (long form)
python main.py auto --check-interval 30 -s 8.0

# Start automated processing (short aliases)
python main.py w --check-interval 30 -s 8.0
python main.py watch --check-interval 30 -s 8.0

# Place videos in input/ directory and they'll be processed automatically
```

### Example 4: Quick Commands with Aliases

```bash
# Check hardware capabilities
python main.py hw

# Show configuration
python main.py cfg show

# Clear cache
python main.py ch clear

# Optimize existing GIF (auto-detected)
python main.py g input.gif output.gif
```

## Performance Tips

1. **Use Hardware Acceleration**: The tool automatically detects and uses GPU acceleration when available
2. **Batch Processing**: Use glob patterns (`*.mp4`) for automatic batch processing
3. **Short Commands**: Use aliases (`c`, `g`, `w`) and short flags (`-p`, `-s`, `-o`) for faster typing
4. **Quality Optimization**: The unified `gif` command automatically optimizes based on your flags
5. **Platform-Specific Settings**: Always specify the target platform (`-p`) for optimal results
6. **Parallel Processing**: Use `-j N` flag for batch operations on multi-core systems
7. **Valid existing outputs in `output/` are reused to save time**

## Troubleshooting

### Common Issues

- **FFmpeg not found**: Install FFmpeg and ensure it's in your system PATH
- **Hardware encoder errors**: App falls back to software; use `--force-software` to skip hardware detection
- **Outputs slightly over target**: Use `--target-size` for advanced optimization or reduce FPS/scale/colors
- **Large file sizes**: Use `-s` or `--max-size` to enable automatic quality optimization
- **Poor quality**: Increase quality settings or use `--target-size` with `--quality-preference quality`
- **Batch mode not working**: Ensure glob patterns are quoted (`"*.mp4"` not `*.mp4`)

### Debug Mode

Enable debug logging for detailed information:

```bash
# Long form
python main.py compress input.mp4 output.mp4 --debug

# Short form
python main.py c input.mp4 output.mp4 -v
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
