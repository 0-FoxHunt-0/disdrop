# 🎥 DisDrop: Advanced Video & GIF Optimization

![DisDrop Logo](assets/logo.png)

A powerful tool for intelligent video and GIF compression, optimized for Discord and social media file size limits.

[![GitHub license](https://img.shields.io/github/license/0-FoxHunt-0/disdrop)](https://github.com/0-FoxHunt-0/disdrop/blob/main/LICENSE)
[![GitHub issues](https://img.shields.io/github/issues/0-FoxHunt-0/disdrop)](https://github.com/0-FoxHunt-0/disdrop/issues)
[![GitHub stars](https://img.shields.io/github/stars/0-FoxHunt-0/disdrop)](https://github.com/0-FoxHunt-0/disdrop/stargazers)

## ✨ Features

### 🎯 Intelligent Optimization
- **Smart Content Analysis**
  - Frame-by-frame motion detection
  - Color palette optimization
  - Adaptive quality settings
  - Content-aware compression

- **Progressive Quality Control**
  - Quality thresholds to prevent over-compression
  - Frame rate optimization based on motion
  - Dynamic color reduction
  - Adaptive dithering selection

### 🚀 Performance
- **Hardware Acceleration**
  - NVIDIA GPU (NVENC) support
  - Multi-threaded processing
  - Batch optimization
  - Automatic CPU fallback

- **Resource Management**
  - Memory-efficient processing
  - Automatic cleanup
  - Temporary file handling
  - Process monitoring

### 🛠 Advanced Features
- **Optimization Strategies**
  - Multiple compression passes
  - Frame similarity detection
  - Temporal optimization
  - Color palette analysis

- **Quality Preservation**
  - Minimum quality thresholds
  - Smart dimension scaling
  - Optimal dithering selection
  - Frame disposal optimization

## 📋 Requirements

### System
- Python 3.8+
- FFmpeg with NVENC support (optional)
- 4GB RAM (8GB+ recommended)
- NVIDIA GPU (optional)

### Dependencies
```bash
pip install -r requirements.txt
```

## 🚀 Quick Start

1. Clone the repository:
```bash
git clone https://github.com/0-FoxHunt-0/disdrop.git
cd disdrop
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run DisDrop:
```bash
python run.py [options]
```

## 💻 Usage

### Basic Usage
```bash
python run.py
```

### Advanced Options
```bash
python run.py --debug           # Enable debug logging
python run.py --no-gpu          # Disable GPU acceleration
python run.py --input-dir DIR   # Custom input directory
python run.py --output-dir DIR  # Custom output directory
python run.py --verbose         # Detailed logging
```

## 📁 Directory Structure
```
disdrop/
├── src/                # Source code
│   ├── gif_optimization.py
│   ├── video_optimization.py
│   └── ...
├── input/              # Input files
├── output/             # Processed files
├── temp/               # Temporary files
├── logs/               # Log files
└── requirements.txt    # Dependencies
```

## 🔧 Configuration

Configure settings in `src/default_config.py`:

```python
GIF_COMPRESSION = {
    'fps_range': (10, 15),
    'colors': 256,
    'lossy_value': 55,
    'min_size_mb': 10
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit pull requests.

1. Fork the repo
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⭐ Support

If you find DisDrop helpful, please consider giving it a star on GitHub!

[https://github.com/0-FoxHunt-0/disdrop](https://github.com/0-FoxHunt-0/disdrop)

## 📞 Contact

For bugs or feature requests, please use the [issue tracker](https://github.com/0-FoxHunt-0/disdrop/issues).

---

Made with ❤️ by FoxHunt

