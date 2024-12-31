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
        success = run_ffmpeg_command(command, str(LOG_DIR / 'ffmpeg.log'))
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


@log_function_call
def process_videos(gpu_supported=False):
    """
    Process all videos in the input directory for compression.
    Only deletes videos with no audio after successful GIF creation.
    """
    failed_files = []
    input_dir = Path(INPUT_DIR)
    temp_dir = Path(TEMP_FILE_DIR)
    output_dir = Path(OUTPUT_DIR)
    min_size_mb = VIDEO_COMPRESSION['min_size_mb']

    for video_file in input_dir.glob('*.mp4'):
        # Store audio status early
        has_audio = has_audio_stream(video_file)
        logging.info(f"Processing video {
                     video_file.name} - Has audio: {has_audio}")

        # Prepare names for potential output files
        compressed_video_name = video_file.stem + "_compressed.mp4"
        compressed_video_path = output_dir / compressed_video_name
        gif_output_path = output_dir / (video_file.stem + "_fps.gif")

        # Check if files already exist
        video_exists = compressed_video_path.exists()
        gif_exists = gif_output_path.exists()

        if has_audio:
            # For videos with audio, keep both video and gif
            if video_exists and gif_exists:
                logging.info(f"Both compressed video and GIF already exist for {
                             video_file.name}. Skipping.")
                continue
        else:
            # For videos without audio, we only need the gif
            if gif_exists:
                logging.info(
                    f"GIF already exists for no-audio video {video_file.name}. Skipping.")
                continue

        # Process video if needed
        if not video_exists:
            original_size = get_file_size(video_file)
            logging.info(f"Processing video: {video_file}")
            logging.info(f"Original size: {original_size:.2f} MB")

            success = compress_video(
                video_file,
                compressed_video_path,
                VIDEO_COMPRESSION['scale_factor'],
                VIDEO_COMPRESSION['crf'],
                gpu_supported
            )

            if not success:
                failed_files.append(video_file)
                continue

        # Process GIF if needed
        if not gif_exists:
            source_video = compressed_video_path if video_exists else video_file
            logging.info(f"Creating GIF from video: {source_video}")

            success = process_single_gif(
                source_video,
                gif_output_path,
                (0, 120),  # min dimensions
                is_video=True
            )

            if not success:
                failed_files.append(video_file)
                continue

            # Only delete the compressed video if:
            # 1. The video has no audio
            # 2. GIF creation was successful
            # 3. The compressed video exists
            if not has_audio and video_exists and compressed_video_path.exists():
                try:
                    compressed_video_path.unlink()
                    logging.info(f"Deleted video with no audio after successful GIF creation: {
                                 compressed_video_path}")
                except Exception as e:
                    logging.error(f"Error deleting video {
                                  compressed_video_path}: {e}")

    return failed_files
