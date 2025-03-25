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

    def __init__(self):
        """Initialize the GPU detector with logging setup."""
        self.logger = logging.getLogger(__name__)
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
                return accel

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
                self.logger.info("Metal acceleration is available")

    def _detect_on_linux(self) -> None:
        """Detect GPUs on Linux systems."""
        self._detect_nvidia_linux()
        self._detect_amd_linux()
        self._detect_intel_linux()

    def _detect_nvidia_windows(self) -> None:
        """Detect NVIDIA GPUs on Windows."""
        # Look for NVIDIA CUDA on Windows
        nvml_path = shutil.which("nvidia-smi")
        if nvml_path:
            output = self._run_command("nvidia-smi")
            if output and not "failed" in output.lower():
                self.detected_gpus.add(GPUType.NVIDIA)
                self.available_accelerations.add(AccelerationType.CUDA)
                self.logger.info("NVIDIA GPU with CUDA support detected")

    def _detect_amd_windows(self) -> None:
        """Detect AMD GPUs on Windows."""
        # Check for AMD GPUs
        dxdiag_output = self._run_command("dxdiag /t")
        if dxdiag_output and "amd" in dxdiag_output.lower():
            self.detected_gpus.add(GPUType.AMD)
            self.logger.info("AMD GPU detected")

            # Check for ROCm
            if shutil.which("rocm-smi"):
                self.available_accelerations.add(AccelerationType.ROCM)
                self.logger.info("AMD ROCm acceleration is available")

    def _detect_intel_windows(self) -> None:
        """Detect Intel GPUs on Windows."""
        dxdiag_output = self._run_command("dxdiag /t")
        if dxdiag_output and "intel" in dxdiag_output.lower():
            self.detected_gpus.add(GPUType.INTEL)
            self.logger.info("Intel GPU detected")

            # Check for OneAPI
            if os.environ.get("ONEAPI_ROOT") or shutil.which("sycl-ls"):
                self.available_accelerations.add(AccelerationType.ONEAPI)
                self.logger.info("Intel OneAPI acceleration is available")

    def _detect_directml(self) -> None:
        """Detect DirectML support on Windows."""
        if self.os_type == 'windows':
            # If we have any GPU and we're on Windows 10/11, DirectML should be available
            if self.detected_gpus and int(platform.version().split('.')[0]) >= 10:
                self.available_accelerations.add(AccelerationType.DIRECTML)
                self.logger.info("DirectML acceleration is available")

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
                    self.logger.info("NVIDIA GPU with CUDA support detected")

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
                    self.logger.info("AMD GPU with ROCm support detected")

    def _detect_intel_linux(self) -> None:
        """Detect Intel GPUs on Linux."""
        # Check for Intel GPU on Linux
        gpu_devices = self._run_command("lspci | grep -E 'VGA|3D|Display'")
        if gpu_devices and "intel" in gpu_devices.lower():
            self.detected_gpus.add(GPUType.INTEL)

            # Check for OneAPI
            if os.environ.get("ONEAPI_ROOT") or shutil.which("sycl-ls"):
                self.available_accelerations.add(AccelerationType.ONEAPI)
                self.logger.info("Intel GPU with OneAPI support detected")

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
            self.logger.info("OpenCL acceleration is available")

    def _run_command(self, command: str) -> Optional[str]:
        """
        Run a system command and return the output.

        Args:
            command: The command to run

        Returns:
            Optional[str]: Command output or None if command failed
        """
        try:
            result = subprocess.run(
                command,
                shell=True,
                check=False,
                capture_output=True,
                text=True,
                timeout=5  # Avoid hanging on problematic commands
            )
            return result.stdout
        except (subprocess.SubprocessError, OSError) as e:
            self.logger.debug(f"Command '{command}' failed: {e}")
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
    logging.basicConfig(level=logging.INFO)

    detector = GPUDetector()
    gpu_types, accel_types = detector.detect()

    print(f"Detected GPU types: {[gpu.name for gpu in gpu_types]}")
    print(f"Available acceleration: {[accel.name for accel in accel_types]}")
    print(
        f"Preferred acceleration: {detector.get_preferred_acceleration().name}")

    # Get detailed device info
    device_info = detector.get_device_info()
    print("\nDevice info:")
    for key, value in device_info.items():
        if key != "detailed_info":
            print(f"  {key}: {value}")

    if device_info["detailed_info"]:
        print("\nDetailed GPU information:")
        for gpu, info in device_info["detailed_info"].items():
            print(f"  {gpu.upper()}:")
            for line in info.strip().split('\n'):
                print(f"    {line}")
