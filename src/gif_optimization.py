# gif_optimization.py
import logging
import os
import shutil
import subprocess
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from default_config import (GIF_COMPRESSION, GIF_SIZE_TO_SKIP, INPUT_DIR, OUTPUT_DIR,
                            TEMP_FILE_DIR)
from logging_system import log_function_call, run_ffmpeg_command
from video_optimization import get_video_dimensions

# Global variable to track failed files
failed_files = []


def has_audio_stream(video_path):
    """Check if a video has an audio stream using ffmpeg."""
    command = ['ffprobe', '-v', 'error', '-select_streams', 'a', '-show_entries',
               'stream=codec_type', '-of', 'default=noprint_wrappers=1:nokey=1', str(video_path)]
    try:
        result = subprocess.run(command, check=True,
                                text=True, capture_output=True)
        return 'audio' in result.stdout
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
def generate_palette(file_path, palette_path, fps, dimensions):
    """Generate a palette for gif creation."""
    command = [
        'ffmpeg', '-i', str(file_path),
        '-vf', f'fps={fps},scale={dimensions[0]
                                  }:{dimensions[1]}:flags=lanczos,palettegen',
        '-y', str(palette_path)
    ]
    logging.info(f"GENERATED_PALETTE: {fps}fps - ({dimensions[0]}x{
        dimensions[1]}) - {Path(file_path).name}")
    return run_ffmpeg_command(command)


@log_function_call
def create_gif_from_video(file_path, palette_path, output_gif, fps, dimensions):
    """Create a gif from a video file using a generated palette."""
    command = [
        'ffmpeg', '-i', str(file_path), '-i', str(palette_path),
        '-lavfi', f'fps={fps},scale={dimensions[0]}:{
            dimensions[1]}:flags=lanczos[x];[x][1:v]paletteuse=dither=bayer',
        '-y', str(output_gif)
    ]
    gif_size = get_file_size(output_gif)
    logging.info(f"GENERATED_GIF: {fps}fps - {gif_size:.2f} MB - ({dimensions[0]}x{
        dimensions[1]}) - {Path(file_path).name}")
    return run_ffmpeg_command(command)


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


# Inside gif_optimization.py

def process_file(file_path, output_path, is_video):
    MIN_HEIGHT = GIF_COMPRESSION['min_height']

    # Get video dimensions, handling potential None values
    width, height = get_video_dimensions(file_path)
    if width is None or height is None:
        logging.error(f"Failed to retrieve dimensions for {file_path}")
        failed_files.append(file_path)
        return

    scale_factor = 1.0

    while True:
        logging.info(f"Processing {'video' if is_video else 'GIF'} {
                     file_path} with scale={scale_factor:.2f}")
        fps_range = range(
            max(GIF_COMPRESSION['fps_range']),
            min(GIF_COMPRESSION['fps_range']) - 1,
            -1
        )

        try:
            with Pool(processes=len(fps_range)) as pool:
                # Create input arguments for each FPS value
                process_args = [
                    (str(file_path), str(output_path), is_video, fps, scale_factor)
                    for fps in fps_range
                ]

                # Create an iterator for the results
                async_results = pool.imap_unordered(process_gif, process_args)

                # Process the first result
                first_result = next(async_results)
                if first_result[2] is not None:  # If processing succeeded
                    first_size = first_result[1]

                    if first_size > 50:
                        scale_factor *= 0.3
                    elif first_size > 40:
                        scale_factor *= 0.4
                    elif first_size > 30:
                        scale_factor *= 0.5
                    elif first_size > 20:
                        scale_factor *= 0.8
                    elif first_size > 10:
                        scale_factor *= 0.9
                    else:
                        scale_factor *= 0.95

                    if first_size > GIF_COMPRESSION['min_size_mb']:
                        logging.info(f"First optimization resulted in {first_size:.2f}MB, "
                                     f"exceeding limit of {
                                         GIF_COMPRESSION['min_size_mb']}MB. "
                                     "Terminating current batch and reducing scale.")
                        # Clean up the temporary file from first result
                        if first_result[2] and Path(first_result[2]).exists():
                            os.remove(first_result[2])
                        # Terminate the pool
                        pool.terminate()
                        pool.join()
                        # Adjust scale factor and continue to next iteration
                        continue

                # If we get here, first result was good or failed
                # Collect all results including the first one
                results = [first_result] + list(async_results)

        except KeyboardInterrupt:
            logging.warning(
                "Script interrupted by user. Closing multiprocessing pool.")
            pool.terminate()
            pool.join()
            raise

        valid_results = [res for res in results if res[2] is not None]

        new_height = int(height * scale_factor)
        if new_height < MIN_HEIGHT:
            logging.warning(f"Failed to optimize {
                file_path} due to minimum dimension constraints")
            failed_files.append(file_path)
            break  # Exit the loop if we can't meet the minimum height requirement

        if not valid_results:
            if scale_factor < 0.1:
                logging.error(f"Could not create GIF for {file_path}")
                failed_files.append(file_path)
                break
            continue

        smallest_size = min(size for _, size, _ in valid_results)

        # Get the best result (highest FPS within size limit)
        valid_gifs = [
            (fps, size, path)
            for fps, size, path in valid_results
            if size <= GIF_COMPRESSION['min_size_mb']
        ]

        if valid_gifs:
            best_gif = max(valid_gifs, key=lambda x: x[0])
            shutil.move(best_gif[2], output_path)
            # Assuming 'width' and 'height' are available here
            dimensions = (int(width * scale_factor),
                          int(height * scale_factor))
            logging.info(f"{best_gif[0]}fps - ({dimensions[0]
                                                }x{dimensions[1]}) - {Path(file_path).name}")

            # Remove from failed_files if successful
            if file_path in failed_files:
                failed_files.remove(file_path)

            # Clean up temporary files
            for _, _, path in valid_results:
                if path != best_gif[2]:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
            break
        else:
            # If no valid GIF was created, keep in failed_files
            if file_path not in failed_files:
                failed_files.append(file_path)


def process_gif(input_args):
    """
    Process a single GIF with given parameters.
    Takes a tuple of arguments to work with Pool.imap_unordered.

    Args:
        input_args: tuple containing (file_path, output_path, is_video, fps, scale_factor)

    Returns:
        tuple: (fps, size, path) where size is float('inf') if processing failed
    """
    file_path, output_path, is_video, fps, scale_factor = input_args
    temp_dir = Path(TEMP_FILE_DIR)
    base_name = Path(output_path).stem
    temp_gif_path = temp_dir / f"{base_name}_{fps}_{scale_factor:.2f}.gif"

    width, height = get_video_dimensions(file_path)
    new_width, new_height = int(
        width * scale_factor), int(height * scale_factor)

    if new_height < 120:
        logging.warning(f"GIF_HEIGHT_TOO_SMALL {
                        new_height}px for {Path(file_path).name}")
        return (fps, float('inf'), None)

    try:
        if not is_video:  # Handling GIF input
            # Directly scale down the GIF using gifsicle
            command = [
                'gifsicle', '--scale', f'{scale_factor}',
                '--colors', str(GIF_COMPRESSION['colors']),
                '--lossy=' + str(GIF_COMPRESSION['lossy_value']),
                '--optimize=3',
                str(file_path), '-o', str(temp_gif_path)
            ]
            subprocess.run(command, check=True, capture_output=True, text=True)
        else:  # For videos, keep the existing logic
            palette_path = temp_dir / \
                f"palette_{fps}_{scale_factor}_{os.getpid()}.png"
            generate_palette(file_path, palette_path,
                             fps, (new_width, new_height))
            create_gif_from_video(file_path, palette_path,
                                  temp_gif_path, fps, (new_width, new_height))

        if not temp_gif_path.is_file():
            logging.error(f"GIF_CREATION_FAILED for {Path(file_path).name}")
            return (fps, float('inf'), None)

        initial_size = get_file_size(temp_gif_path)

        if initial_size > 90:
            logging.info(f"GIF_OVERSIZED initial_size={initial_size:.2f}MB for {
                fps}fps - ({new_width}x{new_height}) - {GIF_COMPRESSION['colors']} - {GIF_COMPRESSION['lossy_value']}, skipping optimization")
            return (fps, initial_size, str(temp_gif_path))

        optimized_size = optimize_gif_with_gifsicle(
            temp_gif_path, temp_gif_path, GIF_COMPRESSION['colors'], GIF_COMPRESSION['lossy_value'])

        if optimized_size:
            logging.info(f"GIF_OPTIMIZED size={optimized_size:.2f}MB fps={
                fps} - ({new_width}x{new_height}) - {GIF_COMPRESSION['colors']} - {GIF_COMPRESSION['lossy_value']}")
        else:
            logging.error(f"GIF_OPTIMIZATION_FAILED for {
                          fps}fps - {Path(file_path).name}")
        return (fps, optimized_size, str(temp_gif_path))

    except Exception as e:
        logging.error(f"GIF_PROCESSING_ERROR {
                      str(e)} for {Path(file_path).name}")
        if temp_gif_path.exists():
            temp_gif_path.unlink()
        return (fps, float('inf'), None)


def process_gifs():
    global failed_files
    failed_files = []

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    # Ensure directories exist
    Path(TEMP_FILE_DIR).mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process video files from output directory
    for video_file in output_dir.glob('*.mp4'):
        # Ensure original name here
        output_gif = output_dir / f"{video_file.stem}.gif"
        if not output_gif.exists():
            process_file(video_file, output_gif, is_video=True)
        else:
            logging.info(f"Skipping {video_file.name} as {
                         output_gif.name} already exists.")

    # Process GIF files from input directory
    for gif_file in input_dir.glob('*.gif'):
        # Ensure original name here
        output_gif = output_dir / f"{gif_file.stem}.gif"
        if not output_gif.exists():
            process_file(gif_file, output_gif, is_video=False)
        else:
            logging.info(f"Skipping {gif_file.name} as {
                         output_gif.name} already exists.")

    return list(set(failed_files))
