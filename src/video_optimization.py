# video_optimization.py

import logging
import os
import shutil
import subprocess
from pathlib import Path

from default_config import (INPUT_DIR, OUTPUT_DIR, SUPPORTED_VIDEO_FORMATS,
                            TEMP_FILE_DIR, VIDEO_COMPRESSION)
from logging_system import log_function_call, run_ffmpeg_command
from temp_file_manager import TempFileManager  # Added import


def get_video_dimensions(video_path):
    """Retrieve the width and height of the video."""
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0',
        str(video_path)
    ]
    try:
        # Replace direct subprocess call with run_ffmpeg_command
        if run_ffmpeg_command(command):
            output = subprocess.check_output(command, text=True).strip()
            width, height = map(int, output.split('x'))
            return width, height
        return None, None
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
        # Replace direct subprocess call with run_ffmpeg_command
        if run_ffmpeg_command(command):
            output = subprocess.check_output(command, stderr=subprocess.STDOUT)
            return len(output) > 0
        return False
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
        # Replace direct subprocess call with run_ffmpeg_command
        if run_ffmpeg_command(command):
            output = subprocess.check_output(command, text=True).strip()
            num, den = map(int, output.split('/'))
            return num / den if den != 0 else num
        return None
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


def convert_to_mp4(input_path, output_path, use_gpu=True):
    """Convert video to MP4 format while maintaining quality."""
    try:
        if use_gpu:
            command = [
                'ffmpeg', '-hwaccel', 'cuda',
                '-i', str(input_path),
                '-c:v', 'h264_nvenc',
                '-preset', 'p7',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-movflags', '+faststart',
                '-y', str(output_path)
            ]
        else:
            command = [
                'ffmpeg', '-i', str(input_path),
                '-c:v', 'libx264',
                '-preset', 'medium',
                '-crf', '18',
                '-c:a', 'aac',
                '-b:a', '192k',
                '-movflags', '+faststart',
                '-y', str(output_path)
            ]

        logging.info(f"Converting {input_path.name} to MP4...")
        success = run_ffmpeg_command(command)
        if success:
            logging.info(f"Successfully converted {input_path.name} to MP4")
            return True
        return False
    except Exception as e:
        logging.error(f"Error converting video to MP4: {e}")
        return False


def compress_video_pass1(input_path, temp_output_path, scale_factor, crf, use_gpu=True):
    """First pass focuses on analyzing the video and creating a quality baseline."""
    width, height = get_video_dimensions(input_path)
    if not width or not height:
        return False, float('inf')

    # Calculate bitrate based on scaled dimensions
    new_width = int(width * scale_factor)
    new_height = int(height * scale_factor)
    target_bitrate = calculate_target_bitrate(new_width, new_height)

    # First pass uses 2-pass encoding to analyze the video
    if use_gpu:
        command = [
            'ffmpeg', '-hwaccel', 'cuda', '-i', str(input_path),
            '-vf', f'scale={new_width}:{new_height}',
            '-c:v', 'h264_nvenc',
            '-preset', 'p7',
            '-b:v', f'{target_bitrate}k',
            '-pass', '1',
            '-f', 'null',
            '/dev/null'  # Use NUL on Windows
        ]
    else:
        command = [
            'ffmpeg', '-i', str(input_path),
            '-vf', f'scale={new_width}:{new_height}',
            '-c:v', 'libx264',
            '-preset', 'medium',
            '-b:v', f'{target_bitrate}k',
            '-pass', '1',
            '-f', 'null',
            '/dev/null'  # Use NUL on Windows
        ]

    success = run_ffmpeg_command(command)
    if not success:
        return False, float('inf')

    # Now do the actual first pass encode
    return compress_video(input_path, temp_output_path, scale_factor, crf, use_gpu)


def compress_video_pass2(input_path, final_output_path, scale_factor, crf, use_gpu=True):
    """Second pass focuses on achieving target size with quality constraints."""
    # Increase CRF more significantly for second pass
    adjusted_crf = min(crf + 4, 51)  # Ensure we don't exceed max CRF of 51
    return compress_video(input_path, final_output_path, scale_factor, adjusted_crf, use_gpu)


def process_videos(gpu_supported=False):
    """Process all videos from input directory with consolidated logic."""
    failed_files = []
    input_dir = Path(INPUT_DIR)
    temp_dir = Path(TEMP_FILE_DIR)
    output_dir = Path(OUTPUT_DIR)

    # First pass: Convert all non-MP4 videos to MP4 in input directory
    for format in SUPPORTED_VIDEO_FORMATS:
        if format != '.mp4':
            for video_file in input_dir.glob(f'*{format}'):
                temp_file = temp_dir / f"temp_{video_file.stem}.mp4"
                final_output = input_dir / f"{video_file.stem}.mp4"
                TempFileManager.register(temp_file)

                if not final_output.exists():
                    success = convert_to_mp4(
                        video_file, temp_file, gpu_supported)
                    if success:
                        shutil.move(str(temp_file), str(final_output))
                    else:
                        failed_files.append(video_file)

                TempFileManager.unregister(temp_file)

    # Second pass: Process all MP4 files from input directory
    for video_file in input_dir.glob('*.[mM][pP]4'):
        temp_file_1 = temp_dir / f"temp_pass1_{video_file.name}"
        temp_file_2 = temp_dir / f"temp_pass2_{video_file.name}"
        TempFileManager.register(temp_file_1)
        TempFileManager.register(temp_file_2)

        has_audio = has_audio_stream(video_file)
        compressed_video_name = f"{video_file.stem}.mp4"
        final_output = output_dir / compressed_video_name

        if final_output.exists():
            logging.info(
                f"Video {video_file.name} already processed. Skipping.")
            continue

        # Compression strategy
        crf = VIDEO_COMPRESSION['crf']
        scale_factor = VIDEO_COMPRESSION['scale_factor']
        best_size = float('inf')
        attempts = 0

        # Get original dimensions
        width, height = get_video_dimensions(video_file)
        if not width or not height:
            logging.error(f"Failed to get dimensions for {video_file}")
            failed_files.append(video_file)
            continue

        while best_size > min_size_mb:
            success, size_pass1 = compress_video_pass1(
                video_file, temp_file_1, scale_factor, crf, gpu_supported)

            if not success:
                logging.warning(
                    f"First pass compression failed for {video_file}")
                failed_files.append(video_file)
                break

            success, size_pass2 = compress_video_pass2(
                temp_file_1, temp_file_2, scale_factor, crf, gpu_supported)

            if not success:
                logging.warning(
                    f"Second pass compression failed for {video_file}")
                failed_files.append(video_file)
                break

            if size_pass2 < best_size:
                best_size = size_pass2
                if best_size <= min_size_mb:
                    # Check if dimensions meet the minimum requirements
                    new_width = int(width * scale_factor)
                    new_height = int(height * scale_factor)
                    if new_width >= min_width and new_height >= min_height:
                        shutil.move(str(temp_file_2), str(final_output))
                        logging.info(f"Video successfully compressed to {
                                     best_size:.2f} MB")
                        break
                    else:
                        logging.warning(
                            f"Video dimensions after scaling do not meet minimum size requirements. Continuing compression.")

            # Adjust compression parameters dynamically
            attempts += 1

            if best_size > min_size_mb * 1.5:  # If significantly over target
                if crf < 35:  # Gradually increase CRF for quality reduction
                    crf += 2
                else:
                    scale_factor *= 0.9  # Reduce scale if CRF is already high
            elif best_size > min_size_mb:  # If slightly over target
                if crf < 31:  # Moderate increase in CRF
                    crf += 1
                elif scale_factor > 0.5:  # Moderate scale reduction if CRF increase isn't enough
                    scale_factor *= 0.95
            else:
                # If under target or just right, we might slightly improve quality or maintain
                # If we've increased CRF, try to lower it back
                if crf > VIDEO_COMPRESSION['crf']:
                    # But not below initial value
                    crf = max(crf - 1, VIDEO_COMPRESSION['crf'])
                elif scale_factor < 1.0:  # If scale was reduced, try to increase slightly
                    # But not above original scale
                    scale_factor = min(scale_factor * 1.05, 1.0)

            logging.info(f"Attempt {attempts}: CRF={
                         crf}, scale={scale_factor:.2f}")

        # Cleanup
        TempFileManager.unregister(temp_file_1)
        TempFileManager.unregister(temp_file_2)
        if temp_file_1.exists():
            temp_file_1.unlink()
        if temp_file_2.exists():
            temp_file_2.unlink()

    return failed_files
