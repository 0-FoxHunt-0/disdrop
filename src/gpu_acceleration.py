# gpu_acceleration.py
# Handles GPU setup and checks for hardware acceleration.

import logging
import subprocess

from logging_system import log_function_call


@log_function_call
def check_gpu_support():
    """Check if the system has an NVIDIA GPU and NVENC support."""
    try:
        # Check for NVIDIA GPU using nvidia-smi
        gpu_info = subprocess.check_output(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], text=True).strip()
        logging.info(f"Detected GPU: {gpu_info}")

        # Check for NVENC support in FFmpeg
        nvenc_check = subprocess.run(
            ['ffmpeg', '-encoders'], capture_output=True, text=True)
        if 'h264_nvenc' in nvenc_check.stdout:
            logging.info("NVENC support is available.")
            return True
        else:
            logging.warning(
                "NVENC support is not available in the current FFmpeg build.")
            return False
    except FileNotFoundError:
        logging.error(
            "nvidia-smi or FFmpeg not found. Ensure they are installed and in the PATH.")
        return False
    except subprocess.CalledProcessError as e:
        logging.error(f"Error checking GPU support: {e}", exc_info=True)
        return False


@log_function_call
def setup_gpu_acceleration():
    """Prepare the environment for GPU-accelerated video processing."""
    gpu_supported = check_gpu_support()
    if not gpu_supported:
        logging.warning(
            "Falling back to CPU processing as GPU is not supported or available.")
        return False
    else:
        logging.info("GPU acceleration is enabled and ready to use.")
        return True
