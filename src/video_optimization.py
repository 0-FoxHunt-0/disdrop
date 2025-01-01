# video_optimization.py

import logging
import os
import shutil
import subprocess
from pathlib import Path

from default_config import (INPUT_DIR, LOG_DIR, OUTPUT_DIR, TEMP_FILE_DIR,
                            VIDEO_COMPRESSION)
from logging_system import log_function_call, run_ffmpeg_command


def get_video_dimensions(video_path):
    """Retrieve the width and height of the video."""
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0',
        str(video_path)
    ]
    try:
        output = subprocess.check_output(command, text=True).strip()
        width, height = map(int, output.split('x'))
        return width, height
    except subprocess.CalledProcessError:
        logging.error(f"Failed to get dimensions for {video_path}")
        return None, None


def get_file_size(file_path):
    """Get the size of a file in MB."""
    return os.path.getsize(file_path) / (1024 * 1024)


def has_audio_stream(video_path):
    """Check if video has an audio stream."""
    command = ['ffprobe', '-i', str(video_path), '-show_streams',
               '-select_streams', 'a', '-loglevel', 'error']
    try:
        output = subprocess.check_output(command, stderr=subprocess.STDOUT)
        return len(output) > 0
    except subprocess.CalledProcessError:
        return False


def calculate_target_bitrate(width, height):
    """Calculate appropriate bitrate based on video dimensions."""
    pixels = width * height
    base_bitrate = 4000  # 4Mbps base for 1080p
    target_bitrate = min(base_bitrate,
                         int(base_bitrate * pixels / (1920 * 1080)))
    return max(500, target_bitrate)  # Ensure minimum 500kbps


def get_frame_rate(video_path):
    """Retrieve the frame rate of the video."""
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    try:
        output = subprocess.check_output(command, text=True).strip()
        num, den = map(int, output.split('/'))
        # Handle cases where frame rate is an integer
        return num / den if den != 0 else num
    except subprocess.CalledProcessError:
        logging.error(f"Failed to get frame rate for {video_path}")
        return None


@log_function_call
def compress_video(input_path, output_path, scale_factor, crf, use_gpu=True):
    """
    Compress video with improved quality control and size management.
    Returns (success, final_size_mb)
    """
    width, height = get_video_dimensions(input_path)
    if not width or not height:
        return False, float('inf')

    # Get frame rate
    frame_rate = get_frame_rate(input_path)
    if frame_rate is None:
        return False, float('inf')

    # If frame rate is above 30, reduce to 30
    if frame_rate > 30:
        framerate_filter = ',fps=fps=30'
    else:
        framerate_filter = ''

    # Handle small videos
    if min(width, height) < 120:
        new_width, new_height = width, height
    else:
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)

        # Ensure minimum dimension of 120px
        if min(new_width, new_height) < 120:
            scale = 120 / min(new_width, new_height)
            new_width = int(new_width * scale)
            new_height = int(new_height * scale)

    # Ensure dimensions are even
    new_width += new_width % 2
    new_height += new_height % 2

    # Calculate appropriate bitrate
    target_bitrate = calculate_target_bitrate(new_width, new_height)
    bitrate = f"{target_bitrate}k"
    buffer_size = f"{target_bitrate * 2}k"

    # Check for audio stream
    has_audio = has_audio_stream(input_path)

    # Prepare scaling filter with padding
    filter_string = (
        f"scale={new_width}:{new_height}:force_original_aspect_ratio=decrease,"
        f"pad={new_width}:{new_height}:(ow-iw)/2:(oh-ih)/2{framerate_filter}"
    )

    if use_gpu:
        command = [
            'ffmpeg', '-hwaccel', 'cuda', '-i', str(input_path),
            '-vf', filter_string,
            '-c:v', 'h264_nvenc',
            '-preset', 'p7',  # Most efficient preset for NVENC
            '-b:v', bitrate,
            '-maxrate', bitrate,
            '-bufsize', buffer_size,
            '-profile:v', 'main',
            '-rc', 'vbr',  # Variable bitrate mode
            '-cq', str(crf),
            '-c:a', 'copy' if has_audio else 'none',
            '-movflags', '+faststart',
            '-y', str(output_path)
        ]
    else:
        command = [
            'ffmpeg', '-i', str(input_path),
            '-vf', filter_string,
            '-c:v', 'libx264',
            '-preset', 'veryslow',
            '-b:v', bitrate,
            '-maxrate', bitrate,
            '-bufsize', buffer_size,
            '-profile:v', 'main',
            '-crf', str(crf),
            '-c:a', 'copy' if has_audio else 'none',
            '-movflags', '+faststart',
            '-y', str(output_path)
        ]

    try:
        success = run_ffmpeg_command(command)
        if success:
            final_size = get_file_size(output_path)
            logging.info(f"Video compressed to {
                         final_size:.2f} MB: {output_path}")
            return True, final_size
        return False, float('inf')
    except Exception as e:
        logging.error(f"Error compressing video: {e}")
        return False, float('inf')


def compress_video_pass1(input_path, temp_output_path, scale_factor, crf, use_gpu=True):
    # This pass focuses on keeping quality relatively high while reducing size
    # Use a slightly lower CRF for better quality in first pass
    crf -= 1  # Less aggressive than before
    return compress_video(input_path, temp_output_path, scale_factor, crf, use_gpu)


def compress_video_pass2(input_path, final_output_path, scale_factor, crf, use_gpu=True):
    # This pass focuses on reducing size further but less aggressively
    # Use a slightly higher CRF for more compression in second pass
    crf += 1  # Less aggressive than before
    return compress_video(input_path, final_output_path, scale_factor, crf, use_gpu)


def process_videos(gpu_supported=False):
    """Process all videos in the input directory for compression."""
    failed_files = []
    input_dir = Path(INPUT_DIR)
    temp_dir = Path(TEMP_FILE_DIR)
    output_dir = Path(OUTPUT_DIR)
    min_size_mb = VIDEO_COMPRESSION['min_size_mb']

    for video_file in input_dir.glob('*.mp4'):
        # Check if the video has an audio stream
        has_audio = has_audio_stream(video_file)

        # Prepare names for potential output files
        compressed_video_name = video_file.stem + ".mp4"

        if has_audio and (output_dir / compressed_video_name).exists():
            logging.info(f"Video {
                         video_file.name} already has a compressed version in the output directory. Skipping.")
            continue

        original_size = get_file_size(video_file)
        logging.info(f"Processing video: {video_file}")
        logging.info(f"Original size: {original_size:.2f} MB")

        if original_size < min_size_mb:
            logging.info(f"Video is already smaller than {
                         min_size_mb} MB. Copying to output directory.")
            shutil.copy2(video_file, output_dir / compressed_video_name)
            continue

        # Initial compression settings
        crf = VIDEO_COMPRESSION['crf']
        scale_factor = VIDEO_COMPRESSION['scale_factor']

        best_size = float('inf')
        temp_file_1 = temp_dir / f"temp_pass1_{video_file.name}"
        temp_file_2 = temp_dir / f"temp_pass2_{video_file.name}"
        final_output = output_dir / compressed_video_name

        while best_size > min_size_mb:
            # First Pass: Quality over size but less aggressive
            success, size_pass1 = compress_video_pass1(
                video_file, temp_file_1, scale_factor, crf, gpu_supported)
            if not success:
                logging.error(
                    f"First pass compression failed for {video_file}")
                failed_files.append(video_file)
                break

            # Second Pass: Size reduction but less aggressive
            success, size_pass2 = compress_video_pass2(
                temp_file_1, temp_file_2, scale_factor, crf, gpu_supported)
            if success:
                if size_pass2 < best_size:
                    best_size = size_pass2
                    # Clean up previous best if exists
                    if final_output.exists():
                        final_output.unlink()
                    # Only move the file if it's below min_size_mb
                    if best_size <= min_size_mb:
                        shutil.move(str(temp_file_2), str(final_output))
                        logging.info(f"Video encoded to {
                                     best_size:.2f} MB, moved to {final_output}")
                    else:
                        logging.info(f"Current compression to {
                                     best_size:.2f} MB, not moving as it exceeds {min_size_mb} MB")
                else:
                    logging.info(f"Current compression did not improve size. Best size: {
                                 best_size:.2f} MB")
                # Clean up temp file from first pass
                if temp_file_1.exists():
                    temp_file_1.unlink()
            else:
                logging.error(
                    f"Second pass compression failed for {video_file}")
                failed_files.append(video_file)
                # Clean up both temp files if failure occurred
                if temp_file_1.exists():
                    temp_file_1.unlink()
                if temp_file_2.exists():
                    temp_file_2.unlink()
                break

            # If we've hit the minimum size or can't compress further, break
            if best_size <= min_size_mb:
                break

            # Adjust compression parameters if needed
            if crf < 35:
                crf += 2
                logging.info(f"Trying higher CRF: {crf}")
            else:
                scale_factor -= 0.1
                crf = VIDEO_COMPRESSION['crf']  # Reset CRF when changing scale
                logging.info(f"Trying smaller scale: {scale_factor:.1f}")

        # Ensure cleanup if the loop was broken due to failure or if the size constraint wasn't met
        if temp_file_1.exists():
            temp_file_1.unlink()
        if temp_file_2.exists():
            temp_file_2.unlink()

    return failed_files
