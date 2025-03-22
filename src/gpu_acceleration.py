# gpu_acceleration.py
# Enhanced GPU setup and acceleration for video processing and GIF creation

import logging
import os
import platform
import subprocess
import threading
import tempfile
import time
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

    @log_function_call
    def detect_gpu(self, force_refresh: bool = False) -> GPUCapabilities:
        """Detect GPU capabilities with proper caching and locking."""
        # Use thread lock to avoid multiple detections
        with self._detection_lock:
            if self._detection_complete and not force_refresh:
                return self.capabilities

            logger.info("Detecting GPU capabilities...")

            # Try to detect in order of market share and detection reliability
            try:
                # First, check if the system has a GPU
                if self._detect_any_gpu():
                    logger.debug("Found a GPU in the system")

                # Then try to identify the type
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

                # If we detected a GPU but couldn't determine if it supports hw accel,
                # enable basic acceleration anyway on Windows
                if self.capabilities.gpu_type != GPUType.NONE and not self.capabilities.supports_hwaccel:
                    if platform.system() == 'Windows':
                        logger.debug(
                            "Enabling basic hardware acceleration for detected GPU")
                        self.capabilities.supports_hwaccel = True

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

    def _detect_any_gpu(self) -> bool:
        """Basic detection of any GPU in the system."""
        try:
            # For Windows, use DirectX diagnostics
            if platform.system() == 'Windows':
                try:
                    # Use DxDiag for basic detection
                    import tempfile
                    temp_file = tempfile.NamedTemporaryFile(
                        delete=False, suffix='.txt')
                    temp_file.close()

                    subprocess.run(['dxdiag', '/t', temp_file.name],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)

                    # Wait for dxdiag to complete
                    time.sleep(2)

                    with open(temp_file.name, 'r', encoding='utf-8', errors='ignore') as f:
                        dxdiag_content = f.read()

                    # Clean up temp file
                    try:
                        os.unlink(temp_file.name)
                    except:
                        pass

                    # Look for Display section which indicates a GPU
                    if "Display Devices" in dxdiag_content or "Display Device" in dxdiag_content:
                        # Try to extract manufacturer
                        import re
                        vendor_match = re.search(
                            r'Card name: ([^\r\n]+)', dxdiag_content)
                        if vendor_match:
                            vendor = vendor_match.group(1).strip()
                            if "NVIDIA" in vendor:
                                self.capabilities.gpu_type = GPUType.NVIDIA
                                self.capabilities.model = vendor
                            elif "AMD" in vendor or "Radeon" in vendor:
                                self.capabilities.gpu_type = GPUType.AMD
                                self.capabilities.model = vendor
                            elif "Intel" in vendor:
                                self.capabilities.gpu_type = GPUType.INTEL
                                self.capabilities.model = vendor
                            else:
                                self.capabilities.gpu_type = GPUType.OTHER
                                self.capabilities.model = vendor

                            logger.debug(f"Detected GPU via dxdiag: {vendor}")
                            return True
                        # If we can't extract vendor but found Display section, assume there's a GPU
                        else:
                            self.capabilities.gpu_type = GPUType.OTHER
                            self.capabilities.model = "Unknown GPU"
                            logger.debug("Detected generic GPU via dxdiag")
                            return True
                except Exception as e:
                    logger.debug(f"DxDiag GPU detection failed: {e}")

                # Try using PowerShell to get video controller info
                try:
                    ps_cmd = "Get-WmiObject -Query \"SELECT * FROM Win32_VideoController\" | Select-Object Name, AdapterRAM | ConvertTo-Csv -NoTypeInformation"
                    output = subprocess.check_output(
                        ['powershell', '-Command', ps_cmd], text=True)

                    lines = output.strip().split('\n')
                    if len(lines) > 1:
                        import csv
                        from io import StringIO
                        reader = csv.DictReader(StringIO('\n'.join(lines)))
                        for row in reader:
                            if 'Name' in row:
                                gpu_name = row['Name']
                                self.capabilities.model = gpu_name

                                # Determine GPU type from name
                                if "NVIDIA" in gpu_name:
                                    self.capabilities.gpu_type = GPUType.NVIDIA
                                elif "AMD" in gpu_name or "Radeon" in gpu_name:
                                    self.capabilities.gpu_type = GPUType.AMD
                                elif "Intel" in gpu_name:
                                    self.capabilities.gpu_type = GPUType.INTEL
                                else:
                                    self.capabilities.gpu_type = GPUType.OTHER

                                logger.debug(
                                    f"Detected GPU via WMI: {gpu_name}")
                                return True
                except Exception as e:
                    logger.debug(f"PowerShell GPU detection failed: {e}")

            # For Linux, use lspci
            elif platform.system() == 'Linux':
                try:
                    output = subprocess.check_output(
                        ['lspci', '-v'], text=True)
                    for line in output.splitlines():
                        if 'VGA' in line or '3D' in line or 'Display' in line:
                            self.capabilities.model = line.split(
                                ':')[-1].strip()

                            # Determine GPU type from lspci output
                            if "NVIDIA" in line:
                                self.capabilities.gpu_type = GPUType.NVIDIA
                            elif "AMD" in line or "Radeon" in line or "ATI" in line:
                                self.capabilities.gpu_type = GPUType.AMD
                            elif "Intel" in line:
                                self.capabilities.gpu_type = GPUType.INTEL
                            else:
                                self.capabilities.gpu_type = GPUType.OTHER

                            logger.debug(
                                f"Detected GPU via lspci: {self.capabilities.model}")
                            return True
                except Exception as e:
                    logger.debug(f"lspci GPU detection failed: {e}")

            # For macOS
            elif platform.system() == 'Darwin':
                try:
                    output = subprocess.check_output(
                        ['system_profiler', 'SPDisplaysDataType'], text=True)
                    if "Chipset Model" in output:
                        import re
                        model_match = re.search(
                            r'Chipset Model: ([^\n]+)', output)
                        if model_match:
                            model = model_match.group(1).strip()
                            self.capabilities.model = model

                            # Determine GPU type
                            if "NVIDIA" in model:
                                self.capabilities.gpu_type = GPUType.NVIDIA
                            elif "AMD" in model or "Radeon" in model:
                                self.capabilities.gpu_type = GPUType.AMD
                            elif "Intel" in model:
                                self.capabilities.gpu_type = GPUType.INTEL
                            else:
                                self.capabilities.gpu_type = GPUType.OTHER

                            logger.debug(
                                f"Detected GPU via system_profiler: {model}")
                            return True
                except Exception as e:
                    logger.debug(f"system_profiler GPU detection failed: {e}")

            # Last resort: Check if FFmpeg has any hardware encoders available
            try:
                temp_encoder_file = os.path.join(
                    tempfile.gettempdir(), "ffmpeg_encoders_any.txt")
                run_ffmpeg_command(['ffmpeg', '-encoders', '-v', 'info',
                                   '-hide_banner', '-y', '-f', 'null', temp_encoder_file])

                with open(FFPMEG_LOG_FILE, 'r') as f:
                    log_content = f.read()
                    hw_encoders = ['nvenc', 'qsv',
                                   'amf', 'vaapi', 'videotoolbox']

                    for encoder in hw_encoders:
                        if encoder in log_content.lower():
                            self.capabilities.model = f"Unknown GPU with {encoder} support"
                            self.capabilities.gpu_type = GPUType.OTHER
                            self.capabilities.supports_hwaccel = True
                            logger.debug(
                                f"Detected GPU via FFmpeg {encoder} support")
                            return True
            except Exception as e:
                logger.debug(f"FFmpeg encoder detection failed: {e}")

            return False
        except Exception as e:
            logger.error(f"Error in _detect_any_gpu: {e}")
            return False

    def _detect_nvidia_gpu(self) -> bool:
        """Detect NVIDIA GPU and update capabilities."""
        try:
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
                # On Windows, check if nvidia-smi exists before calling it
                if platform.system() == 'Windows':
                    import subprocess
                    try:
                        # First just check if nvidia-smi exists at all
                        subprocess.run(['where', 'nvidia-smi'],
                                       stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE,
                                       check=True,
                                       text=True)
                    except (subprocess.SubprocessError, FileNotFoundError):
                        logger.debug("nvidia-smi not found in PATH")
                        # Try default install location
                        nvidia_smi_paths = [
                            r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
                            r"C:\Windows\System32\nvidia-smi.exe"
                        ]
                        for path in nvidia_smi_paths:
                            if os.path.exists(path):
                                logger.debug(f"Found nvidia-smi at {path}")
                                try:
                                    output = subprocess.check_output(
                                        [path, '--query-gpu=name,memory.total',
                                            '--format=csv,noheader,nounits'],
                                        universal_newlines=True
                                    )
                                    if output.strip():
                                        gpu_info = output.strip().split(',')
                                        self.capabilities.gpu_type = GPUType.NVIDIA
                                        self.capabilities.model = gpu_info[0].strip(
                                        )
                                        self.capabilities.vram_mb = int(
                                            gpu_info[1].strip())
                                        self.capabilities.cuda_available = True
                                        logger.debug(
                                            f"Detected NVIDIA GPU via {path}: {self.capabilities.model}")
                                        return True
                                except Exception as e:
                                    logger.debug(
                                        f"Failed to use nvidia-smi at {path}: {e}")

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

            # Alternative detection method for Windows - check registry
            if platform.system() == 'Windows':
                try:
                    import winreg
                    with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Control\Video") as key:
                        i = 0
                        while True:
                            try:
                                subkey_name = winreg.EnumKey(key, i)
                                with winreg.OpenKey(key, f"{subkey_name}\\0000") as subkey:
                                    try:
                                        provider_name, _ = winreg.QueryValueEx(
                                            subkey, "ProviderName")
                                        device_desc, _ = winreg.QueryValueEx(
                                            subkey, "Device Description")
                                        if "NVIDIA" in provider_name:
                                            self.capabilities.gpu_type = GPUType.NVIDIA
                                            self.capabilities.model = device_desc
                                            self.capabilities.supports_hwaccel = True
                                            self.capabilities.nvenc_available = True
                                            logger.debug(
                                                f"Detected NVIDIA GPU via registry: {device_desc}")
                                            return True
                                    except FileNotFoundError:
                                        pass
                                i += 1
                            except (WindowsError, FileNotFoundError):
                                break
                except Exception as e:
                    logger.debug(f"Registry detection failed: {e}")

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

            # Check if FFmpeg can detect NVENC
            # Use run_ffmpeg_command to properly capture and log output
            ffmpeg_logger = setup_ffmpeg_logging()
            ffmpeg_logger.info("Checking for NVENC support via FFmpeg")

            # Run ffmpeg command to check for NVENC support
            temp_encoder_file = os.path.join(
                tempfile.gettempdir(), "ffmpeg_encoders_nvidia.txt")
            run_ffmpeg_command(['ffmpeg', '-encoders', '-v', 'info',
                               '-hide_banner', '-y', '-f', 'null', temp_encoder_file])

            # Check ffmpeg.log for the h264_nvenc encoder
            nvenc_available = False
            with open(FFPMEG_LOG_FILE, 'r') as f:
                log_content = f.read()
                if 'h264_nvenc' in log_content:
                    nvenc_available = True

            if nvenc_available:
                self.capabilities.nvenc_available = True
                self.capabilities.model = "NVIDIA GPU (NVENC available)"
                logger.debug("NVENC available via FFmpeg")
                ffmpeg_logger.info("NVENC support detected")
                return True

            return False

        except Exception as e:
            logger.error(f"Error detecting NVIDIA GPU: {e}")
            return False

    def _detect_intel_gpu(self) -> bool:
        """Detect Intel GPU and capabilities."""
        try:
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

        except:
            pass

        return False

    def _detect_amd_gpu(self) -> bool:
        """Detect AMD GPU and capabilities."""
        try:
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

        except:
            pass

        return False

    def _detect_available_encoders(self):
        """Detect available encoders."""
        # Detect FFmpeg acceleration support
        self._detect_ffmpeg_support()

    def _detect_ffmpeg_support(self):
        """Detect FFmpeg hardware acceleration support."""
        try:
            # Import the FFmpeg logging function
            ffmpeg_logger = setup_ffmpeg_logging()

            # Log the detection attempt
            ffmpeg_logger.info(
                "Detecting FFmpeg hardware acceleration support")

            # Get encoders and decoders output by running commands through our logging wrapper
            encoders_output = ""
            encoders_error = ""
            decoders_output = ""
            decoders_error = ""

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
            with open(FFPMEG_LOG_FILE, 'r') as f:
                log_content = f.read()
                # Extract the relevant sections
                if "FFmpeg encoders output:" in log_content:
                    encoders_output = log_content.split("FFmpeg encoders output:")[
                        1].split("FFmpeg encoders error output:")[0]
                if "FFmpeg encoders error output:" in log_content:
                    encoders_error = log_content.split("FFmpeg encoders error output:")[1].split("FFmpeg decoders output:")[
                        0] if "FFmpeg decoders output:" in log_content else log_content.split("FFmpeg encoders error output:")[1]
                if "FFmpeg decoders output:" in log_content:
                    decoders_output = log_content.split("FFmpeg decoders output:")[1].split("FFmpeg decoders error output:")[
                        0] if "FFmpeg decoders error output:" in log_content else log_content.split("FFmpeg decoders output:")[1]

            # Check for hardware acceleration support
            detected_encoders = []

            # NVIDIA NVENC support
            if self.capabilities.gpu_type == GPUType.NVIDIA:
                if 'h264_nvenc' in encoders_output:
                    self.capabilities.nvenc_available = True
                    detected_encoders.append('NVENC')
                    ffmpeg_logger.info("NVENC support detected")

            # Intel QuickSync support
            if self.capabilities.gpu_type == GPUType.INTEL or 'h264_qsv' in encoders_output:
                if 'h264_qsv' in encoders_output:
                    self.capabilities.qsv_available = True
                    ffmpeg_logger.info("QuickSync support detected")

            # AMD AMF support
            if self.capabilities.gpu_type == GPUType.AMD or 'h264_amf' in encoders_output:
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


@log_function_call
def check_gpu_support():
    """Check if the system has GPU acceleration support."""
    logger.debug("Checking GPU support")
    gpu_manager = GPUManager()
    capabilities = gpu_manager.detect_gpu()
    result = capabilities.supports_hwaccel
    logger.debug(f"GPU support check result: {result}")
    return result


@performance_monitor
def setup_gpu_acceleration():
    """Prepare the environment for GPU-accelerated video processing."""
    logger.info("Setting up GPU acceleration")
    gpu_manager = GPUManager()
    capabilities = gpu_manager.detect_gpu()

    # Set detection flag so we don't log again
    gpu_manager.detected = True

    if not capabilities.supports_hwaccel:
        logger.warning(
            "Falling back to CPU processing as GPU acceleration is not supported or available.")
        return False
    else:
        gpu_type_value = capabilities.gpu_type.value if capabilities.gpu_type else "unknown"
        logger.success(
            f"GPU acceleration is enabled using {gpu_type_value.upper() if gpu_type_value else 'UNKNOWN'} acceleration")
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
