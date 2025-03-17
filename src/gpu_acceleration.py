# gpu_acceleration.py
# Enhanced GPU setup and acceleration for video processing and GIF creation

import logging
import os
import platform
import subprocess
import threading
import tempfile
from enum import Enum
from typing import Dict, Any, Optional, Tuple

from .logging_system import log_function_call, performance_monitor, get_logger, setup_ffmpeg_logging, run_ffmpeg_command
from .default_config import FFPMEG_LOG_FILE

# Module-level logger
logger = get_logger('gpu')


class GPUType(Enum):
    """Supported GPU types for acceleration."""
    NVIDIA = 'nvidia'
    INTEL = 'intel'
    AMD = 'amd'
    OTHER = 'other'
    NONE = 'none'


class GPUCapabilities:
    """Detailed GPU capabilities information."""

    def __init__(self):
        self.gpu_type = GPUType.NONE
        self.model = "Unknown"
        self.vram_mb = 0
        self.supports_hwaccel = False
        self.cuda_available = False
        self.nvenc_available = False
        self.qsv_available = False
        self.amf_available = False
        self.vaapi_available = False
        self.recommended_concurrency = 1
        self.max_resolution = (0, 0)

    def __str__(self) -> str:
        return (
            f"GPU: {self.gpu_type.value} ({self.model})\n"
            f"VRAM: {self.vram_mb}MB\n"
            f"HW Acceleration: {'Supported' if self.supports_hwaccel else 'Not Supported'}\n"
            f"CUDA: {'Available' if self.cuda_available else 'Not Available'}\n"
            f"Encoders: {', '.join(self.get_available_encoders())}"
        )

    def get_available_encoders(self) -> list:
        """Get list of available encoders."""
        encoders = []
        if self.nvenc_available:
            encoders.append("NVENC")
        if self.qsv_available:
            encoders.append("QSV")
        if self.amf_available:
            encoders.append("AMF")
        if self.vaapi_available:
            encoders.append("VAAPI")
        return encoders if encoders else ["None"]

    def get_optimal_settings(self) -> Dict[str, Any]:
        """Get optimal encoding settings based on GPU capabilities."""
        settings = {
            'use_hw_accel': self.supports_hwaccel,
            'max_concurrent_encodes': self.recommended_concurrency,
            'preferred_encoder': None,
            'max_resolution': self.max_resolution,
        }

        # Set preferred encoder based on available options
        if self.nvenc_available:
            settings['preferred_encoder'] = 'nvenc'
        elif self.qsv_available:
            settings['preferred_encoder'] = 'qsv'
        elif self.vaapi_available:
            settings['preferred_encoder'] = 'vaapi'
        elif self.amf_available:
            settings['preferred_encoder'] = 'amf'

        return settings


# Singleton to avoid multiple detections
class GPUManager:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if not self._initialized:
            self.capabilities = GPUCapabilities()
            self.detected = False
            self._detection_lock = threading.Lock()
            self._detection_complete = False
            self._initialized = True

    def detect_gpu(self, force_refresh: bool = False) -> GPUCapabilities:
        """Detect GPU capabilities with proper caching and locking."""
        # Use thread lock to avoid multiple detections
        with self._detection_lock:
            if self._detection_complete and not force_refresh:
                return self.capabilities

            # Only log if this is the first detection or forced refresh
            if not self._detection_complete or force_refresh:
                logger.info("Detecting GPU capabilities...")

            # Try to detect in order of market share and detection reliability
            try:
                if self._detect_nvidia_gpu():
                    self.capabilities.gpu_type = GPUType.NVIDIA
                elif self._detect_intel_gpu():
                    self.capabilities.gpu_type = GPUType.INTEL
                elif self._detect_amd_gpu():
                    self.capabilities.gpu_type = GPUType.AMD

                # Detect available encoders regardless of GPU type
                self._detect_available_encoders()

                # Determine if GPU can accelerate based on encoders
                self.capabilities.supports_hwaccel = any([
                    self.capabilities.nvenc_available,
                    self.capabilities.qsv_available,
                    self.capabilities.amf_available,
                    self.capabilities.vaapi_available
                ])

                # Set recommended concurrency based on GPU capabilities
                self._set_recommended_concurrency()

                logger.debug(
                    f"GPU detection completed: {self.capabilities.gpu_type.value}")
                if self.capabilities.supports_hwaccel:
                    logger.debug(
                        f"Hardware acceleration available via: {self.capabilities.get_available_encoders()}")
                else:
                    logger.debug("No hardware acceleration available")

            except Exception as e:
                logger.error(f"Error during GPU detection: {e}", exc_info=True)
                self.capabilities.gpu_type = GPUType.NONE
                self.capabilities.supports_hwaccel = False

            self._detection_complete = True
            self.detected = True
            return self.capabilities

    def _detect_nvidia_gpu(self) -> bool:
        """Detect NVIDIA GPU and update capabilities."""
        try:
            # Check if FFmpeg can detect NVENC first - most reliable on Windows
            # Use run_ffmpeg_command to properly capture and log output
            logger.debug("Checking for NVENC support via FFmpeg")
            temp_encoder_file = os.path.join(
                tempfile.gettempdir(), "ffmpeg_encoders_nvidia.txt")

            # Run ffmpeg command to directly get encoder list
            try:
                encoder_output = subprocess.check_output(
                    ['ffmpeg', '-encoders', '-hide_banner'],
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )

                # Check if NVENC is in the encoder list
                if 'h264_nvenc' in encoder_output:
                    self.capabilities.nvenc_available = True
                    self.capabilities.model = "NVIDIA GPU (NVENC available)"
                    self.capabilities.cuda_available = True
                    logger.debug("NVENC available via FFmpeg encoders list")
                    return True
            except:
                logger.debug(
                    "Could not directly check FFmpeg encoders, trying alternative methods")

            # Try importing pycuda for detailed info
            try:
                import pycuda.driver as cuda
                cuda.init()
                device_count = cuda.Device.count()
                if device_count > 0:
                    device = cuda.Device(0)
                    self.capabilities.gpu_type = GPUType.NVIDIA
                    self.capabilities.model = device.name()
                    self.capabilities.vram_mb = device.total_memory() // 1024 // 1024
                    self.capabilities.cuda_available = True
                    logger.debug(
                        f"Detected NVIDIA GPU: {self.capabilities.model}")
                    return True
            except (ImportError, ModuleNotFoundError):
                logger.debug("pycuda not available, trying nvidia-smi")
            except Exception as e:
                logger.debug(f"pycuda detection failed: {e}")

            # Try nvidia-smi as fallback
            try:
                import json
                nvidia_smi_output = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=name,memory.total',
                        '--format=csv,noheader,nounits'],
                    universal_newlines=True
                )
                if nvidia_smi_output.strip():
                    gpu_info = nvidia_smi_output.strip().split(',')
                    self.capabilities.gpu_type = GPUType.NVIDIA
                    self.capabilities.model = gpu_info[0].strip()
                    self.capabilities.vram_mb = int(gpu_info[1].strip())
                    self.capabilities.cuda_available = True
                    logger.debug(
                        f"Detected NVIDIA GPU via nvidia-smi: {self.capabilities.model}")
                    return True
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("nvidia-smi not available")
            except Exception as e:
                logger.debug(f"nvidia-smi detection failed: {e}")

            # Try checking for CUDA libraries as fallback
            try:
                nvcc_output = subprocess.check_output(
                    ['nvcc', '--version'],
                    universal_newlines=True,
                    stderr=subprocess.DEVNULL
                )
                if 'cuda' in nvcc_output.lower():
                    self.capabilities.cuda_available = True
                    self.capabilities.model = "NVIDIA GPU (details unavailable)"
                    logger.debug("CUDA available via nvcc")
                    return True
            except (subprocess.SubprocessError, FileNotFoundError):
                logger.debug("NVCC not available")

            # Last resort - check ffmpeg.log file if available
            try:
                if os.path.exists(FFPMEG_LOG_FILE):
                    with open(FFPMEG_LOG_FILE, 'r') as f:
                        log_content = f.read()
                        if 'h264_nvenc' in log_content:
                            self.capabilities.nvenc_available = True
                            self.capabilities.model = "NVIDIA GPU (NVENC available)"
                            logger.debug("NVENC available via FFmpeg log file")
                            return True
            except:
                pass

            return False

        except Exception as e:
            logger.error(f"Error detecting NVIDIA GPU: {e}")
            return False

    def _detect_intel_gpu(self) -> bool:
        """Detect Intel GPU and capabilities."""
        try:
            # Try FFmpeg first to check QSV availability directly
            try:
                encoder_output = subprocess.check_output(
                    ['ffmpeg', '-encoders', '-hide_banner'],
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )

                # Check if Intel QSV is in the encoder list
                if 'h264_qsv' in encoder_output:
                    self.capabilities.qsv_available = True
                    self.capabilities.model = "Intel GPU (QSV available)"
                    self.capabilities.supports_hwaccel = True
                    logger.debug(
                        "Intel QSV available via FFmpeg encoders list")
                    return True
            except:
                logger.debug(
                    "Could not directly check FFmpeg encoders for QSV, trying alternative methods")

            system = platform.system()

            if system == 'Windows':
                # Use PowerShell on Windows
                ps_cmd = "Get-WmiObject -Query \"SELECT * FROM Win32_VideoController WHERE Name LIKE '%Intel%' AND Name LIKE '%Graphics%'\" | Select-Object Name, AdapterRAM | ConvertTo-Csv -NoTypeInformation"
                output = subprocess.check_output(
                    ['powershell', '-Command', ps_cmd], text=True)

                lines = output.strip().split('\n')
                if len(lines) > 1:
                    import csv
                    from io import StringIO
                    reader = csv.DictReader(StringIO('\n'.join(lines)))
                    for row in reader:
                        if 'Name' in row and 'Intel' in row['Name']:
                            self.capabilities.model = row['Name']
                            try:
                                ram = int(row.get('AdapterRAM', '0'))
                                self.capabilities.vram_mb = ram // (
                                    1024 * 1024)
                            except ValueError:
                                pass
                            self.capabilities.supports_hwaccel = True
                            return True

            elif system == 'Linux':
                # Use lspci on Linux
                output = subprocess.check_output(['lspci', '-v'], text=True)
                for line in output.splitlines():
                    if 'Intel' in line and ('VGA' in line or 'Graphics' in line):
                        self.capabilities.model = line.split(':')[-1].strip()
                        self.capabilities.supports_hwaccel = True
                        return True

            # Last resort, check if dxgkrnl.sys is loaded on Windows (Intel graphics driver)
            if system == 'Windows':
                try:
                    output = subprocess.check_output(
                        ['driverquery'], text=True)
                    if 'igdkmd64.sys' in output or 'dxgkrnl.sys' in output:
                        self.capabilities.model = "Intel Graphics (based on system drivers)"
                        self.capabilities.supports_hwaccel = True
                        return True
                except:
                    pass

        except Exception as e:
            logger.debug(f"Error detecting Intel GPU: {e}")

        return False

    def _detect_amd_gpu(self) -> bool:
        """Detect AMD GPU and capabilities."""
        try:
            # Try FFmpeg first to check AMF availability directly
            try:
                encoder_output = subprocess.check_output(
                    ['ffmpeg', '-encoders', '-hide_banner'],
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )

                # Check if AMD AMF is in the encoder list
                if 'h264_amf' in encoder_output:
                    self.capabilities.amf_available = True
                    self.capabilities.model = "AMD GPU (AMF available)"
                    self.capabilities.supports_hwaccel = True
                    logger.debug("AMD AMF available via FFmpeg encoders list")
                    return True
            except:
                logger.debug(
                    "Could not directly check FFmpeg encoders for AMF, trying alternative methods")

            system = platform.system()

            if system == 'Windows':
                # Use PowerShell on Windows
                ps_cmd = "Get-WmiObject -Query \"SELECT * FROM Win32_VideoController WHERE Name LIKE '%AMD%' OR Name LIKE '%Radeon%'\" | Select-Object Name, AdapterRAM | ConvertTo-Csv -NoTypeInformation"
                output = subprocess.check_output(
                    ['powershell', '-Command', ps_cmd], text=True)

                lines = output.strip().split('\n')
                if len(lines) > 1:
                    import csv
                    from io import StringIO
                    reader = csv.DictReader(StringIO('\n'.join(lines)))
                    for row in reader:
                        if 'Name' in row and ('AMD' in row['Name'] or 'Radeon' in row['Name']):
                            self.capabilities.model = row['Name']
                            try:
                                ram = int(row.get('AdapterRAM', '0'))
                                self.capabilities.vram_mb = ram // (
                                    1024 * 1024)
                            except ValueError:
                                pass
                            self.capabilities.supports_hwaccel = True
                            return True

            elif system == 'Linux':
                # Use lspci on Linux
                output = subprocess.check_output(['lspci', '-v'], text=True)
                for line in output.splitlines():
                    if ('AMD' in line or 'Radeon' in line) and ('VGA' in line or 'Graphics' in line):
                        self.capabilities.model = line.split(':')[-1].strip()
                        self.capabilities.supports_hwaccel = True
                        return True

            # Check for AMD display drivers on Windows
            if system == 'Windows':
                try:
                    output = subprocess.check_output(
                        ['driverquery'], text=True)
                    if 'amdkmdap.sys' in output or 'atikmdag.sys' in output:
                        self.capabilities.model = "AMD Graphics (based on system drivers)"
                        self.capabilities.supports_hwaccel = True
                        return True
                except:
                    pass

        except Exception as e:
            logger.debug(f"Error detecting AMD GPU: {e}")

        return False

    def _detect_available_encoders(self):
        """Detect available encoders."""
        # Detect FFmpeg acceleration support
        self._detect_ffmpeg_support()

    def _detect_ffmpeg_support(self):
        """Detect FFmpeg hardware acceleration support."""
        try:
            # Get encoders directly from FFmpeg rather than through log files
            try:
                encoder_output = subprocess.check_output(
                    ['ffmpeg', '-encoders', '-hide_banner'],
                    stderr=subprocess.STDOUT,
                    universal_newlines=True
                )

                # Check for hardware encoders directly in the output
                if 'h264_nvenc' in encoder_output:
                    self.capabilities.nvenc_available = True
                    logger.debug("NVENC support detected in FFmpeg encoders")

                if 'h264_qsv' in encoder_output:
                    self.capabilities.qsv_available = True
                    logger.debug(
                        "QuickSync support detected in FFmpeg encoders")

                if 'h264_amf' in encoder_output:
                    self.capabilities.amf_available = True
                    logger.debug("AMD AMF support detected in FFmpeg encoders")

                if 'h264_vaapi' in encoder_output:
                    self.capabilities.vaapi_available = True
                    logger.debug("VA-API support detected in FFmpeg encoders")

                available_encoders = self.capabilities.get_available_encoders()
                if available_encoders and available_encoders[0] != "None":
                    logger.debug(
                        f"Detected encoders: {', '.join(available_encoders)}")

                return
            except:
                logger.debug(
                    "Could not directly check FFmpeg encoders, falling back to log file method")

            # Legacy method using log files as fallback
            ffmpeg_logger = setup_ffmpeg_logging()
            ffmpeg_logger.info(
                "Detecting FFmpeg hardware acceleration support")

            # Use temporary files to capture output
            temp_encoder_file = os.path.join(
                tempfile.gettempdir(), "ffmpeg_encoders.txt")
            temp_decoder_file = os.path.join(
                tempfile.gettempdir(), "ffmpeg_decoders.txt")

            # Run the commands
            run_ffmpeg_command(['ffmpeg', '-encoders', '-v', 'info',
                               '-hide_banner', '-y', '-f', 'null', temp_encoder_file])
            run_ffmpeg_command(['ffmpeg', '-decoders', '-v', 'info',
                               '-hide_banner', '-y', '-f', 'null', temp_decoder_file])

            # Read outputs from the ffmpeg.log file since our run_ffmpeg_command redirects there
            if os.path.exists(FFPMEG_LOG_FILE):
                with open(FFPMEG_LOG_FILE, 'r') as f:
                    log_content = f.read()
                    # Extract the relevant sections
                    if "FFmpeg encoders output:" in log_content:
                        encoders_output = log_content.split("FFmpeg encoders output:")[
                            1].split("FFmpeg encoders error output:")[0]
                    else:
                        encoders_output = log_content

                    # Check for hardware acceleration support
                    detected_encoders = []

                    # NVIDIA NVENC support
                    if self.capabilities.gpu_type == GPUType.NVIDIA or self.capabilities.gpu_type == GPUType.NONE:
                        if 'h264_nvenc' in encoders_output:
                            self.capabilities.nvenc_available = True
                            detected_encoders.append('NVENC')
                            ffmpeg_logger.info("NVENC support detected")

                    # Intel QuickSync support
                    if self.capabilities.gpu_type == GPUType.INTEL or self.capabilities.gpu_type == GPUType.NONE:
                        if 'h264_qsv' in encoders_output:
                            self.capabilities.qsv_available = True
                            ffmpeg_logger.info("QuickSync support detected")

                    # AMD AMF support
                    if self.capabilities.gpu_type == GPUType.AMD or self.capabilities.gpu_type == GPUType.NONE:
                        if 'h264_amf' in encoders_output:
                            self.capabilities.amf_available = True
                            ffmpeg_logger.info("AMD AMF support detected")

                    # VA-API support (Linux)
                    if 'h264_vaapi' in encoders_output:
                        self.capabilities.vaapi_available = True
                        ffmpeg_logger.info("VA-API support detected")

                    ffmpeg_logger.info(
                        f"Detected encoders: {self.capabilities.get_available_encoders()}")

        except Exception as e:
            logger.error(f"Error detecting FFmpeg support: {e}", exc_info=True)

    def _set_recommended_concurrency(self):
        """Calculate recommended concurrency based on GPU capabilities."""
        # Start with conservative defaults
        if self.capabilities.gpu_type == GPUType.NONE:
            # CPU only
            self.capabilities.recommended_concurrency = max(
                1, (os.cpu_count() or 4) // 2)
            return

        # Calculate based on GPU type and VRAM
        if self.capabilities.gpu_type == GPUType.NVIDIA:
            # NVIDIA recommendations
            if self.capabilities.vram_mb >= 8000:  # 8GB+
                self.capabilities.recommended_concurrency = 3
            elif self.capabilities.vram_mb >= 4000:  # 4GB+
                self.capabilities.recommended_concurrency = 2
            else:
                self.capabilities.recommended_concurrency = 1

        elif self.capabilities.gpu_type == GPUType.INTEL:
            # Intel integrated GPUs typically have less VRAM and power
            self.capabilities.recommended_concurrency = 1

        elif self.capabilities.gpu_type == GPUType.AMD:
            # AMD recommendations
            if self.capabilities.vram_mb >= 6000:  # 6GB+
                self.capabilities.recommended_concurrency = 2
            else:
                self.capabilities.recommended_concurrency = 1

        else:
            # Unknown GPU - be conservative
            self.capabilities.recommended_concurrency = 1

    def get_ffmpeg_options(self, input_path: Optional[str] = None) -> Dict[str, Any]:
        """Get optimized FFmpeg options based on detected GPU."""
        options = {
            'input_options': [],
            'output_options': [],
            'filter_options': {}
        }

        if not self.detected:
            self.detect_gpu()

        if not self.capabilities.supports_hwaccel:
            # CPU-only optimizations
            cpu_count = os.cpu_count() or 4
            threads = min(cpu_count - 1, 8) if cpu_count > 1 else 1
            options['output_options'].extend(['-threads', str(threads)])
            return options

        # Add hardware acceleration options
        if self.capabilities.gpu_type == GPUType.NVIDIA and self.capabilities.nvenc_available:
            options['input_options'].extend(
                ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])
            options['filter_options']['scale_suffix'] = '_cuda'

        elif self.capabilities.gpu_type == GPUType.INTEL and self.capabilities.qsv_available:
            options['input_options'].extend(
                ['-hwaccel', 'qsv', '-hwaccel_output_format', 'qsv'])
            options['filter_options']['scale_suffix'] = '_qsv'

        elif self.capabilities.vaapi_available:
            # VA-API typically needs device specification on Linux
            options['input_options'].extend(
                ['-hwaccel', 'vaapi', '-hwaccel_output_format', 'vaapi'])
            options['filter_options']['scale_suffix'] = '_vaapi'

        elif self.capabilities.gpu_type == GPUType.AMD and self.capabilities.amf_available:
            # AMF acceleration
            # AMF doesn't need special scale filter
            options['filter_options']['scale_suffix'] = ''

        return options


@performance_monitor
def setup_gpu_acceleration():
    """Prepare the environment for GPU-accelerated video processing."""
    logger.info("Setting up GPU acceleration")
    gpu_manager = GPUManager()
    capabilities = gpu_manager.detect_gpu()

    # Set detection flag so we don't log again
    gpu_manager.detected = True

    # If no hardware acceleration was detected, try one more direct check with FFmpeg
    if not capabilities.supports_hwaccel:
        logger.debug(
            "No GPU acceleration detected by primary methods, trying direct FFmpeg check")
        try:
            # Direct check for hardware encoders
            encoder_output = subprocess.check_output(
                ['ffmpeg', '-encoders', '-hide_banner'],
                stderr=subprocess.STDOUT,
                universal_newlines=True
            )

            # Check for any hardware encoder
            if any(encoder in encoder_output for encoder in ['h264_nvenc', 'h264_qsv', 'h264_amf', 'h264_vaapi']):
                logger.debug("Found hardware encoders via direct FFmpeg check")

                # Set specific encoder flags
                if 'h264_nvenc' in encoder_output:
                    capabilities.nvenc_available = True
                    capabilities.gpu_type = GPUType.NVIDIA
                    capabilities.model = "NVIDIA GPU (detected from FFmpeg)"
                elif 'h264_qsv' in encoder_output:
                    capabilities.qsv_available = True
                    capabilities.gpu_type = GPUType.INTEL
                    capabilities.model = "Intel GPU (detected from FFmpeg)"
                elif 'h264_amf' in encoder_output:
                    capabilities.amf_available = True
                    capabilities.gpu_type = GPUType.AMD
                    capabilities.model = "AMD GPU (detected from FFmpeg)"

                # Set hardware acceleration flag
                capabilities.supports_hwaccel = True
        except:
            logger.debug("Direct FFmpeg check failed")

    if not capabilities.supports_hwaccel:
        logger.warning(
            "Falling back to CPU processing as GPU acceleration is not supported or available.")
        return False
    else:
        logger.success(
            f"GPU acceleration is enabled using {capabilities.gpu_type.value.upper()} acceleration")
        logger.info(f"GPU model: {capabilities.model}")
        logger.info(
            f"Available encoders: {', '.join(capabilities.get_available_encoders())}")
        return True


def get_optimal_settings() -> Dict[str, Any]:
    """Get optimal processing settings based on detected GPU."""
    logger.debug("Getting optimal GPU settings")
    gpu_manager = GPUManager()
    capabilities = gpu_manager.detect_gpu()
    settings = capabilities.get_optimal_settings()
    logger.debug(f"Optimal GPU settings: {settings}")
    return settings


def get_ffmpeg_hwaccel_args(input_path: Optional[str] = None) -> list:
    """Get FFmpeg hardware acceleration arguments for the command line."""
    logger.debug("Getting FFmpeg hardware acceleration arguments")
    gpu_manager = GPUManager()
    options = gpu_manager.get_ffmpeg_options(input_path)
    logger.debug(f"Hardware acceleration options: {options['input_options']}")
    # Return just the input options as a flat list
    return options['input_options']
