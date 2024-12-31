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


def has_audio_stream(video_path):
    """Check if a video has an audio stream using ffmpeg."""
    command = ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries',
               'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    try:
        result = subprocess.run(command, check=True,
                                text=True, capture_output=True)
        # Check if 'audio' is mentioned in the output
        return 'audio' in result.stdout
    except subprocess.CalledProcessError as e:
        logging.error(f"Error checking audio stream for video {
                      video_path}: {e.stderr}")
        return False  # Assume no audio if there's an error


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
    """Process a single gif or video by iterating over fps and scaling options."""
    width, height = min_dimensions
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
            generate_palette(file_path, palette_path, fps, (width, height))
            create_gif_from_video(file_path, palette_path,
                                  temp_gif, fps, (width, height))
        else:
            # If the input is already a gif, use it directly
            temp_gif = file_path

        size = get_file_size(temp_gif)
        target_size = GIF_COMPRESSION['min_size_mb']

        # Loop until the gif is under target size or minimum size is reached
        while size > target_size:
            logging.warning(
                f"GIF exceeds {target_size}MB, attempting to scale down (fps={fps}).")

            # Calculate scaling factor based on file size
            if size > 50:
                scale_factor = 0.3
            elif size > 40:
                scale_factor = 0.4
            elif size > 30:
                scale_factor = 0.5
            elif size > 20:
                scale_factor = 0.8
            elif size > 10:
                scale_factor = 0.9
            else:
                scale_factor = 0.95

            # Scale both dimensions
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)

            # Ensure minimum dimension of 120px but adjust proportionally
            if min(new_width, new_height) < 120:
                scale = 120 / min(new_width, new_height)
                if new_width < new_height:
                    # Ensure at least 121 width
                    new_width = max(int(width * scale), 121)
                    # Scale height proportionally
                    new_height = max(
                        int(height * scale * (new_width / width)), 121)
                else:
                    # Ensure at least 121 height
                    new_height = max(int(height * scale), 121)
                    # Scale width proportionally
                    new_width = max(
                        int(width * scale * (new_height / height)), 121)

            # Avoid exact 120 if possible
            if new_height == 120:
                new_height += 1  # Increment by 1 to avoid exact 120
            if new_width == 120:
                new_width += 1  # Increment by 1 to avoid exact 120

            width, height = new_width, new_height

            if is_video:
                generate_palette(file_path, palette_path, fps, (width, height))
                create_gif_from_video(
                    file_path, palette_path, temp_gif, fps, (width, height))

            size = get_file_size(temp_gif)

        if size <= target_size or min(width, height) < 120:
            logging.info(f"GIF is under target size or at minimum dimensions: {
                         output_path}, size: {size:.2f} MB")
            optimized_size = optimize_gif_with_gifsicle(
                temp_gif, output_path,
                colors=GIF_COMPRESSION['colors'],
                lossy=GIF_COMPRESSION['lossy_value']
            )
            logging.info(f"Optimized GIF saved: {
                         output_path}, size: {optimized_size:.2f} MB")
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
