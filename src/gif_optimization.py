# gif_optimization.py
import logging
import os
import subprocess
from multiprocessing import Pool
from pathlib import Path

from default_config import GIF_COMPRESSION, INPUT_DIR, OUTPUT_DIR, TEMP_FILE_DIR
from logging_system import log_function_call, run_ffmpeg_command

# Global variable to track failed files
failed_files = []


def calculate_new_dimensions(current_width, current_height, scale_factor):
    """
    Calculate new dimensions while maintaining aspect ratio.
    Uses the larger dimension to determine scaling to ensure proper size reduction.

    Args:
        current_width (int): Original width of the image/video
        current_height (int): Original height of the image/video
        scale_factor (float): Factor to scale the dimensions by

    Returns:
        tuple: New width and height maintaining aspect ratio
    """
    if current_width is None or current_height is None:
        return 320, 240  # Default fallback dimensions

    # Calculate new dimensions
    new_width = max(int(current_width * scale_factor), 1)
    new_height = max(int(current_height * scale_factor), 1)

    # If either dimension is below minimum, scale up maintaining aspect ratio
    min_dimension = 120
    if new_width < min_dimension or new_height < min_dimension:
        # Calculate scaling factors for both dimensions
        width_scale = min_dimension / new_width
        height_scale = min_dimension / new_height

        # Use the larger scaling factor to ensure minimum dimension is met
        final_scale = max(width_scale, height_scale)

        new_width = max(int(new_width * final_scale), min_dimension)
        new_height = max(int(new_height * final_scale), min_dimension)

    # Ensure even dimensions for video encoding
    new_width += new_width % 2
    new_height += new_height % 2

    return new_width, new_height


def get_gif_dimensions(gif_path):
    """
    Get the dimensions of a GIF file using ffprobe.

    Args:
        gif_path (Path): Path to the GIF file

    Returns:
        tuple: Width and height of the GIF, or None if dimensions cannot be determined
    """
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height', '-of', 'csv=s=x:p=0',
        str(gif_path)
    ]
    try:
        output = subprocess.check_output(command, text=True).strip()
        width, height = map(int, output.split('x'))
        return width, height
    except (subprocess.CalledProcessError, ValueError) as e:
        logging.error(f"Error getting GIF dimensions: {e}")
        return None, None


def has_audio_stream(video_path):
    """
    Check if a video has an audio stream using ffprobe.
    Uses a more robust method to detect audio streams.

    Args:
        video_path (Path): Path to the video file

    Returns:
        bool: True if the video has an audio stream, False otherwise
    """
    command = [
        'ffprobe',
        '-v', 'error',
        '-select_streams', 'a',
        '-show_entries', 'stream=codec_type',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        str(video_path)
    ]
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=True)
        # Check if any audio stream was found
        has_audio = bool(result.stdout.strip())
        logging.info(f"Audio stream detection for {video_path}: {
                     'Found' if has_audio else 'Not found'}")
        return has_audio
    except subprocess.CalledProcessError as e:
        logging.error(f"Error checking audio stream for video {
                      video_path}: {e.stderr}")
        return False


def get_file_size(file_path):
    """Get the file size in megabytes."""
    try:
        return Path(file_path).stat().st_size / (1024 * 1024)
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return float('inf')


@log_function_call
def generate_palette(video_path, palette_path, fps, dimensions):
    """Generate a palette for gif creation."""
    from default_config import LOG_FILE
    command = [
        'ffmpeg', '-i', str(video_path),
        '-vf', f'fps={fps},scale={dimensions[0]
                                  }:{dimensions[1]}:flags=lanczos,palettegen',
        '-y', str(palette_path)
    ]
    return run_ffmpeg_command(command, LOG_FILE)


@log_function_call
def create_gif_from_video(video_path, palette_path, output_gif, fps, dimensions):
    """Create a gif from a video file using a generated palette."""
    from default_config import LOG_FILE
    command = [
        'ffmpeg', '-i', str(video_path), '-i', str(palette_path),
        '-lavfi', f'fps={fps},scale={dimensions[0]}:{
            dimensions[1]}:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer',
        '-y', str(output_gif)
    ]
    return run_ffmpeg_command(command, LOG_FILE)


@log_function_call
def optimize_gif_with_gifsicle(input_gif, output_gif, colors, lossy):
    """Optimize a gif using gifsicle."""
    command = [
        'gifsicle', '--optimize=3', '--colors', str(colors),
        '--lossy=' + str(lossy), '--no-conserve-memory',
        str(input_gif), '-o', str(output_gif)
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True)
        optimized_size = get_file_size(output_gif)
        logging.info(f"GIF optimized to size: {optimized_size:.2f} MB")
        return optimized_size
    except subprocess.CalledProcessError as e:
        logging.error(f"Error optimizing GIF: {e.stderr}")
        return float('inf')


@log_function_call
def process_single_gif(file_path, output_path, min_dimensions, *, is_video=False):
    """
    Process a single gif or video by iterating over fps and scaling options.
    Maintains aspect ratio throughout the optimization process.
    """
    # Get initial dimensions
    if is_video:
        width, height = get_video_dimensions(file_path)
    else:
        width, height = get_gif_dimensions(file_path)

    if width is None or height is None:
        logging.error(f"Could not get dimensions for {file_path}")
        return False

    temp_dir = Path(TEMP_FILE_DIR)
    output_dir = Path(OUTPUT_DIR)

    # Ensure directories exist
    temp_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    fps_range = range(GIF_COMPRESSION['fps_range'][1],
                      GIF_COMPRESSION['fps_range'][0] - 1, -1)

    def process_fps(fps):
        nonlocal width, height
        palette_path = temp_dir / f"{file_path.stem}_{fps}.png"
        temp_gif = temp_dir / f"{file_path.stem}_{fps}.gif"

        if is_video:
            # Generate palette and create GIF from video
            if not generate_palette(file_path, palette_path, fps, (width, height)):
                return False
            if not create_gif_from_video(file_path, palette_path, temp_gif, fps, (width, height)):
                return False
        else:
            # If the input is already a gif, use it directly for the first iteration
            temp_gif = file_path

        size = get_file_size(temp_gif)
        target_size = GIF_COMPRESSION['min_size_mb']

        # Loop until the gif is under target size or minimum size is reached
        while size > target_size:
            # Calculate scaling factor based on file size
            if size > 50:
                scale_factor = 0.3
            elif size > 40:
                scale_factor = 0.4
            elif size > 30:
                scale_factor = 0.5
            elif size > 20:
                scale_factor = 0.8
            else:
                scale_factor = 0.9

            # Calculate new dimensions maintaining aspect ratio
            width, height = calculate_new_dimensions(
                width, height, scale_factor)

            if is_video:
                if not generate_palette(file_path, palette_path, fps, (width, height)):
                    return False
                if not create_gif_from_video(file_path, palette_path, temp_gif, fps, (width, height)):
                    return False

            size = get_file_size(temp_gif)

            # Break if we can't reduce size further while maintaining minimum dimensions
            if min(width, height) <= 120:
                break

        if size <= target_size:
            optimize_gif_with_gifsicle(
                temp_gif,
                output_path,
                colors=GIF_COMPRESSION['colors'],
                lossy=GIF_COMPRESSION['lossy_value']
            )
            return True

        return False

    success = any(process_fps(fps) for fps in fps_range)

    # Clean up temporary files
    for temp_file in temp_dir.glob(f"{file_path.stem}_*"):
        try:
            temp_file.unlink()
        except Exception as e:
            logging.warning(f"Failed to delete temporary file {
                            temp_file}: {e}")

    return success


@log_function_call
def process_gifs():
    """Process all gifs in the input directory for optimization."""
    global failed_files
    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)
    min_dimensions = (GIF_COMPRESSION['min_width'],
                      GIF_COMPRESSION['min_height'])

    # Process all optimized videos in output directory
    for video_file in output_dir.glob('*.mp4'):
        output_gif = output_dir / (video_file.stem + "_fps.gif")
        logging.info(f"Processing video for GIF creation: {video_file}")

        success = process_single_gif(
            video_file,
            output_gif,
            min_dimensions,
            is_video=True
        )

        if success:
            # Check if the video has an audio stream
            if not has_audio_stream(video_file):
                try:
                    # Delete the video file if no audio stream
                    os.remove(video_file)
                    logging.info(f"Deleted video without audio: {video_file}")
                except Exception as e:
                    logging.error(f"Error deleting video {
                                  video_file}: {e}", exc_info=True)
        if not success and video_file not in failed_files:
            failed_files.append(video_file)

    # Process all gifs in input directory
    for gif_file in input_dir.glob('*.gif'):
        output_gif = output_dir / (gif_file.stem + "_fps.gif")
        logging.info(f"Processing GIF for optimization: {gif_file}")

        success = process_single_gif(
            gif_file,
            output_gif,
            min_dimensions,
            is_video=False
        )

        if not success and gif_file not in failed_files:
            failed_files.append(gif_file)

    # Remove duplicates before returning the list
    failed_files = list(set(failed_files))
    return failed_files
