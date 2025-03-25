#!/usr/bin/env python3
import os
import sys
import platform
import logging
import subprocess
import shutil
from enum import Enum, auto
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Set

from src.logging_system import LoggingSystem


class GPUType(Enum):
    """Enumeration of supported GPU types."""
    NVIDIA = auto()
    AMD = auto()
    INTEL = auto()
    APPLE = auto()
    GENERIC = auto()


class AccelerationType(Enum):
    """Enumeration of supported acceleration frameworks."""
    CUDA = auto()
    ROCM = auto()
    OPENCL = auto()
    ONEAPI = auto()
    DIRECTML = auto()
    METAL = auto()
    CPU = auto()  # Fallback when no GPU acceleration is available


class GPUDetector:
    """
    A comprehensive class to detect GPU acceleration support across different operating systems.

    This class follows the DRY principle and is designed to be easily maintainable
    and integrated into other modules. It detects various GPU acceleration options:
    - NVIDIA CUDA
    - AMD ROCm
    - Intel OneAPI
    - OpenCL
    - DirectX/DirectML (Windows)
    - Metal (macOS)

    If no GPU acceleration is available, it defaults to CPU processing.
    """

    def __init__(self, config=None):
        """
        Initialize the GPU detector with logging setup.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}

        # Get the logging system
        self.logging_system = LoggingSystem(self.config)
        self.logger = self.logging_system.get_logger('gpu_detector')

        # Start a section for GPU detection
        self.logging_system.start_new_log_section(
            "GPU Detection Initialization")

        self.os_type = platform.system().lower()
        self.detected_gpus = set()
        self.available_accelerations = set()
        self._detection_performed = False

    def detect(self) -> Tuple[Set[GPUType], Set[AccelerationType]]:
        """
        Detect available GPUs and acceleration frameworks.

        Returns:
            Tuple[Set[GPUType], Set[AccelerationType]]: Sets of detected GPU types and
            acceleration frameworks.
        """
        if self._detection_performed:
            return self.detected_gpus, self.available_accelerations

        # Start a new log section for GPU detection
        self.logging_system.start_new_log_section("GPU Detection Process")
        self.logger.info(
            f"Detecting GPUs and acceleration frameworks on {self.os_type}")

        # Always ensure CPU is available as fallback
        self.available_accelerations.add(AccelerationType.CPU)

        # Detect based on operating system
        try:
            if self.os_type == 'windows':
                self._detect_on_windows()
            elif self.os_type == 'darwin':
                self._detect_on_macos()
            elif self.os_type == 'linux':
                self._detect_on_linux()
            else:
                self.logger.warning(
                    f"Unsupported OS: {self.os_type}. Using CPU processing.")

            # Cross-platform detections
            self._detect_opencl()

            self._detection_performed = True

        except Exception as e:
            self.logger.error(
                f"Error during GPU detection: {e}", exc_info=True)
            # Ensure we can still continue with CPU
            self.available_accelerations = {AccelerationType.CPU}

        return self.detected_gpus, self.available_accelerations

    def get_preferred_acceleration(self) -> AccelerationType:
        """
        Get the preferred acceleration method based on available options.

        Returns:
            AccelerationType: The recommended acceleration framework to use
        """
        if not self._detection_performed:
            self.detect()

        # Preferred order (generally by performance)
        preferences = [
            AccelerationType.CUDA,     # NVIDIA CUDA generally fastest when available
            AccelerationType.METAL,    # Apple Metal on macOS
            AccelerationType.ROCM,     # AMD ROCm
            AccelerationType.DIRECTML,  # DirectML on Windows
            AccelerationType.ONEAPI,   # Intel OneAPI
            AccelerationType.OPENCL,   # OpenCL as a more generic option
            AccelerationType.CPU       # CPU as fallback
        ]

        for accel in preferences:
            if accel in self.available_accelerations:
                self.logger.success(
                    f"Selected {accel.name} as preferred acceleration method")
                return accel

        self.logger.info(
            "Using CPU acceleration (no GPU acceleration available)")
        return AccelerationType.CPU

    def _detect_on_windows(self) -> None:
        """Detect GPUs on Windows systems."""
        self._detect_nvidia_windows()
        self._detect_amd_windows()
        self._detect_intel_windows()
        self._detect_directml()

    def _detect_on_macos(self) -> None:
        """Detect GPUs on macOS systems."""
        # Check for Metal support on macOS
        if self._run_command("system_profiler SPDisplaysDataType"):
            output = self._run_command("system_profiler SPDisplaysDataType")

            if output:
                if any(gpu in output.lower() for gpu in ["amd", "radeon"]):
                    self.detected_gpus.add(GPUType.AMD)
                if "nvidia" in output.lower():
                    self.detected_gpus.add(GPUType.NVIDIA)
                if "intel" in output.lower():
                    self.detected_gpus.add(GPUType.INTEL)
                if any(apple_gpu in output.lower() for apple_gpu in ["apple", "m1", "m2", "m3"]):
                    self.detected_gpus.add(GPUType.APPLE)

                # Metal is available on all modern Macs
                self.available_accelerations.add(AccelerationType.METAL)
                self.logger.success("Metal acceleration is available")

    def _detect_on_linux(self) -> None:
        """Detect GPUs on Linux systems."""
        self._detect_nvidia_linux()
        self._detect_amd_linux()
        self._detect_intel_linux()

    def _detect_nvidia_windows(self) -> None:
        """Detect NVIDIA GPUs on Windows with improved RTX support."""
        # Look for NVIDIA CUDA on Windows using multiple detection methods
        try:
            # Try nvidia-smi first as it's faster and more reliable
            nvml_path = shutil.which("nvidia-smi")
            if nvml_path:
                # On Windows, check common nvidia-smi paths
                nvidia_smi_paths = [
                    nvml_path,  # Use the found path
                    r"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe",
                    r"C:\Windows\System32\nvidia-smi.exe"
                ]

                for path in nvidia_smi_paths:
                    try:
                        # Use a short timeout to avoid hanging
                        output = self._run_command(
                            f'"{path}" --query-gpu=name,memory.total --format=csv,noheader,nounits')
                        if output and not "failed" in output.lower():
                            self.detected_gpus.add(GPUType.NVIDIA)
                            self.available_accelerations.add(
                                AccelerationType.CUDA)

                            # Check for RTX GPUs which definitely support NVENC
                            if 'RTX' in output:
                                self.logger.success(
                                    "NVIDIA RTX GPU detected with NVENC support")
                            else:
                                self.logger.success(
                                    "NVIDIA GPU detected with CUDA support")

                            return True
                    except Exception as e:
                        self.logger.debug(
                            f"Failed to use nvidia-smi at {path}: {e}")
                        continue

            # If nvidia-smi fails, try dxdiag
            dxdiag_output = self._run_command("dxdiag /t")
            # Look for dxdiag files that may have been created
            self.logging_system.find_and_move_dxdiag_file()

            if dxdiag_output and "nvidia" in dxdiag_output.lower():
                self.detected_gpus.add(GPUType.NVIDIA)
                self.logger.info("NVIDIA GPU detected via dxdiag")

                # Check for RTX series which definitely supports NVENC
                if 'RTX' in dxdiag_output:
                    self.available_accelerations.add(AccelerationType.CUDA)
                    self.logger.success(
                        "NVIDIA RTX GPU with NVENC support detected")
                return True

            # Try PowerShell as a fallback
            ps_cmd = "Get-WmiObject -Query \"SELECT * FROM Win32_VideoController WHERE Name LIKE '%NVIDIA%'\" | Select-Object Name | ConvertTo-Csv -NoTypeInformation"
            output = self._run_command(f'powershell -Command "{ps_cmd}"')

            if output and "nvidia" in output.lower():
                self.detected_gpus.add(GPUType.NVIDIA)
                self.logger.info("NVIDIA GPU detected via WMI")

                # Check for RTX series which definitely supports NVENC
                if 'RTX' in output:
                    self.available_accelerations.add(AccelerationType.CUDA)
                    self.logger.success(
                        "NVIDIA RTX GPU with NVENC support detected")
                return True

            # Final check - use FFmpeg to see if NVENC is available
            self._detect_ffmpeg_nvenc_support()

            if AccelerationType.CUDA in self.available_accelerations:
                return True

        except Exception as e:
            self.logger.error(f"Error detecting NVIDIA GPU: {e}")

        return False

    def _detect_ffmpeg_nvenc_support(self) -> bool:
        """Check if FFmpeg can use NVENC."""
        try:
            # Check for NVENC support using FFmpeg directly
            output = self._run_command("ffmpeg -encoders")
            if output and ('h264_nvenc' in output.lower() or 'hevc_nvenc' in output.lower()):
                self.detected_gpus.add(GPUType.NVIDIA)
                self.available_accelerations.add(AccelerationType.CUDA)
                self.logger.success(
                    "NVENC support detected via FFmpeg encoders")
                return True

            return False
        except Exception as e:
            self.logger.debug(f"Error checking NVENC support: {e}")
            return False

    def _detect_amd_windows(self) -> None:
        """Detect AMD GPUs on Windows."""
        # Check for AMD GPUs
        dxdiag_output = self._run_command("dxdiag /t")
        # Look for dxdiag files that may have been created
        self.logging_system.find_and_move_dxdiag_file()

        if dxdiag_output and "amd" in dxdiag_output.lower():
            self.detected_gpus.add(GPUType.AMD)
            self.logger.info("AMD GPU detected")

            # Check for ROCm
            if shutil.which("rocm-smi"):
                self.available_accelerations.add(AccelerationType.ROCM)
                self.logger.success("AMD ROCm acceleration is available")

    def _detect_intel_windows(self) -> None:
        """Detect Intel GPUs on Windows."""
        dxdiag_output = self._run_command("dxdiag /t")
        # Look for dxdiag files that may have been created
        self.logging_system.find_and_move_dxdiag_file()

        if dxdiag_output and "intel" in dxdiag_output.lower():
            self.detected_gpus.add(GPUType.INTEL)
            self.logger.info("Intel GPU detected")

            # Check for OneAPI
            if os.environ.get("ONEAPI_ROOT") or shutil.which("sycl-ls"):
                self.available_accelerations.add(AccelerationType.ONEAPI)
                self.logger.success("Intel OneAPI acceleration is available")

    def _detect_directml(self) -> None:
        """Detect DirectML support on Windows."""
        if self.os_type == 'windows':
            # If we have any GPU and we're on Windows 10/11, DirectML should be available
            try:
                windows_version = int(platform.version().split('.')[0])
                if self.detected_gpus and windows_version >= 10:
                    self.available_accelerations.add(AccelerationType.DIRECTML)
                    self.logger.success("DirectML acceleration is available")
            except (ValueError, IndexError) as e:
                self.logger.warning(
                    f"Failed to determine Windows version: {e}")
                # Fall back to checking if we're on Windows 10 or higher without the version number
                if self.detected_gpus and any(ver in platform.version() for ver in ['10.', '11.']):
                    self.available_accelerations.add(AccelerationType.DIRECTML)
                    self.logger.success(
                        "DirectML acceleration is available (fallback detection)")

    def _detect_nvidia_linux(self) -> None:
        """Detect NVIDIA GPUs on Linux."""
        # Check for NVIDIA GPU
        if os.path.exists("/proc/driver/nvidia/version") or shutil.which("nvidia-smi"):
            output = self._run_command("nvidia-smi")
            if output and not "failed" in output.lower():
                self.detected_gpus.add(GPUType.NVIDIA)

                # Check for CUDA
                if shutil.which("nvcc") or os.path.exists("/usr/local/cuda"):
                    self.available_accelerations.add(AccelerationType.CUDA)
                    self.logger.success(
                        "NVIDIA GPU with CUDA support detected")

    def _detect_amd_linux(self) -> None:
        """Detect AMD GPUs on Linux."""
        # Check for AMD GPU on Linux
        if os.path.exists("/sys/class/drm/"):
            gpu_devices = self._run_command("lspci | grep -E 'VGA|3D|Display'")
            if gpu_devices and "amd" in gpu_devices.lower():
                self.detected_gpus.add(GPUType.AMD)

                # Check for ROCm
                if shutil.which("rocm-smi") or os.path.exists("/opt/rocm"):
                    self.available_accelerations.add(AccelerationType.ROCM)
                    self.logger.success("AMD GPU with ROCm support detected")

    def _detect_intel_linux(self) -> None:
        """Detect Intel GPUs on Linux."""
        # Check for Intel GPU on Linux
        gpu_devices = self._run_command("lspci | grep -E 'VGA|3D|Display'")
        if gpu_devices and "intel" in gpu_devices.lower():
            self.detected_gpus.add(GPUType.INTEL)

            # Check for OneAPI
            if os.environ.get("ONEAPI_ROOT") or shutil.which("sycl-ls"):
                self.available_accelerations.add(AccelerationType.ONEAPI)
                self.logger.success("Intel GPU with OneAPI support detected")

    def _detect_opencl(self) -> None:
        """Detect OpenCL support (cross-platform)."""
        # Check for OpenCL - either through system library or specific implementation
        opencl_indicators = [
            # Windows-specific
            os.path.exists("C:\\Windows\\System32\\OpenCL.dll"),
            # Linux-specific
            os.path.exists(
                "/usr/lib/libOpenCL.so") or os.path.exists("/usr/lib64/libOpenCL.so"),
            # macOS-specific
            os.path.exists("/System/Library/Frameworks/OpenCL.framework"),
            # Cross-platform - check if command exists
            shutil.which("clinfo") is not None
        ]

        if any(opencl_indicators):
            self.available_accelerations.add(AccelerationType.OPENCL)
            self.logger.success("OpenCL acceleration is available")

    def _run_command(self, command: str) -> Optional[str]:
        """
        Run a shell command and return its output with improved timeout handling.

        Args:
            command: Command to run

        Returns:
            str: Command output if successful, None otherwise
        """
        try:
            # Use a timeout to prevent hanging on problematic commands
            process = subprocess.run(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=4,  # 4 second timeout
                shell=True
            )

            # Look for and move any dxdiag files if this was a dxdiag command
            if 'dxdiag' in command.lower():
                self.logging_system.find_and_move_dxdiag_file()

            if process.returncode == 0:
                return process.stdout
            else:
                self.logger.debug(
                    f"Command '{command}' failed with code {process.returncode}")
                return None
        except subprocess.TimeoutExpired:
            self.logger.debug(f"Command '{command}' timed out after 4 seconds")
            # Still look for dxdiag files even if the command timed out
            if 'dxdiag' in command.lower():
                self.logging_system.find_and_move_dxdiag_file()
            return None
        except Exception as e:
            self.logger.debug(f"Error running command '{command}': {e}")
            return None

    def get_device_info(self) -> Dict:
        """
        Get detailed information about detected devices.

        Returns:
            Dict: Dictionary containing detailed information about detected devices
        """
        if not self._detection_performed:
            self.detect()

        device_info = {
            "os_type": self.os_type,
            "detected_gpus": [gpu.name for gpu in self.detected_gpus],
            "available_accelerations": [accel.name for accel in self.available_accelerations],
            "preferred_acceleration": self.get_preferred_acceleration().name,
            "detailed_info": {}
        }

        # Add detailed information based on GPU type
        if GPUType.NVIDIA in self.detected_gpus:
            nvidia_info = self._run_command(
                "nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv")
            if nvidia_info:
                device_info["detailed_info"]["nvidia"] = nvidia_info

        if GPUType.AMD in self.detected_gpus and self.os_type == "linux":
            amd_info = self._run_command(
                "rocm-smi --showproductname --showdriverversion")
            if amd_info:
                device_info["detailed_info"]["amd"] = amd_info

        return device_info


# Example usage
if __name__ == "__main__":
    config = {
        'logging': {
            'level': 'INFO',
            'console_level': 'INFO',
            'file_level': 'DEBUG',
            'directory': './logs',
            'clear_logs': True
        }
    }

    # Initialize the logging system
    logging_system = LoggingSystem(config)
    logger = logging_system.get_logger('main')

    # Start with a section header for GPU detection
    logging_system.start_new_log_section("GPU Detection")

    logger.info("Initializing GPU detector")
    detector = GPUDetector(config)

    # Detect GPU capabilities
    gpu_types, accel_types = detector.detect()
    preferred = detector.get_preferred_acceleration()

    logger.info(f"Detected GPU types: {[gpu.name for gpu in gpu_types]}")
    logger.info(
        f"Available acceleration: {[accel.name for accel in accel_types]}")
    logger.success(f"Preferred acceleration: {preferred.name}")

    # Get detailed device info
    device_info = detector.get_device_info()
    logger.info("\nDevice info:")
    for key, value in device_info.items():
        if key != "detailed_info":
            logger.info(f"  {key}: {value}")

    if device_info["detailed_info"]:
        logger.info("\nDetailed GPU information:")
        for gpu, info in device_info["detailed_info"].items():
            logger.info(f"  {gpu.upper()}:")
            for line in info.strip().split('\n'):
                logger.info(f"    {line}")
