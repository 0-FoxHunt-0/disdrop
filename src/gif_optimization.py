# gif_optimization.py
import logging
import os
import shutil
import subprocess
from functools import partial
from multiprocessing import Pool
from pathlib import Path

from default_config import (GIF_COMPRESSION, INPUT_DIR, OUTPUT_DIR,
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


def process_file(file_path, output_path, is_video):
    MIN_HEIGHT = 120  # Minimum acceptable height

    # Get video dimensions, handling potential None values
    width, height = get_video_dimensions(file_path)
    if width is None or height is None:
        logging.error(f"Failed to retrieve dimensions for {file_path}")
        failed_files.append(file_path)
        return

    scale_factor = 1.0

    while True:
        logging.info(f"Processing {file_path} with scale={scale_factor:.2f}")
        fps_range = range(15, 9, -1)  # 15 to 10 fps

        try:
            with Pool(processes=len(fps_range)) as pool:
                results = pool.starmap(partial(process_gif, file_path, output_path, is_video),
                                       [(fps, scale_factor) for fps in fps_range])
        except KeyboardInterrupt:
            logging.warning(
                "Script interrupted by user. Closing multiprocessing pool.")
            pool.terminate()
            pool.join()
            raise

        valid_results = [res for res in results if res[2]
                         is not None]  # res is (fps, size, path)

        if not valid_results:
            if scale_factor < 0.1:
                logging.error(f"Could not create GIF for {file_path}")
                failed_files.append(file_path)
                break
            continue

        smallest_size = min(size for _, size, _ in valid_results)
        if smallest_size > 50:
            scale_factor *= 0.3
        elif smallest_size > 30:
            scale_factor *= 0.5
        elif smallest_size > 20:
            scale_factor *= 0.7
        elif smallest_size > 15:
            scale_factor *= 0.85
        else:
            scale_factor *= 0.95

        valid_gifs = [(fps, size, path) for fps, size,
                      path in valid_results if size <= GIF_COMPRESSION['min_size_mb']]

        if valid_gifs:
            best_gif = max(valid_gifs, key=lambda x: x[0])
            shutil.move(best_gif[2], output_path)
            logging.info(f"GIF optimized: {output_path}, size={
                         best_gif[1]:.2f}MB, fps={best_gif[0]}")

            # Clean up temporary files
            for _, _, path in valid_results:
                if path != best_gif[2]:
                    try:
                        os.remove(path)
                    except FileNotFoundError:
                        pass
            break  # GIF creation successful, move to next file
        else:
            # Here we check if we've hit the minimum dimensions
            new_height = int(height * scale_factor)
            if new_height < MIN_HEIGHT:
                logging.error(f"Failed to optimize {
                              file_path} due to minimum dimension constraints")
                failed_files.append(file_path)
                break

            if scale_factor < 0.1:
                logging.error(f"Failed to optimize {file_path}")
                failed_files.append(file_path)
                break

            logging.info(f"Retrying with scale={
                         scale_factor:.2f} for {file_path}")

    # Handle video files without audio
    if is_video and not has_audio_stream(file_path):
        try:
            os.remove(file_path)
            logging.info(f"Deleted video without audio: {file_path}")
        except Exception as e:
            logging.error(f"Error deleting video {
                          file_path}: {e}", exc_info=True)


def process_gif(file_path, output_path, is_video, fps, scale_factor):
    """
    Process a single GIF or video with given parameters.
    Returns tuple of (fps, size, path) where size is float('inf') if processing failed
    """
    temp_dir = Path(TEMP_FILE_DIR)
    base_name = Path(output_path).stem
    temp_gif_path = temp_dir / f"{base_name}_{fps}_{scale_factor:.2f}.gif"
    palette_path = temp_dir / f"palette_{fps}_{scale_factor}_{os.getpid()}.png"

    width, height = get_video_dimensions(file_path)
    new_width, new_height = int(
        width * scale_factor), int(height * scale_factor)

    if new_height < 120:
        logging.warning(f"GIF_HEIGHT_TOO_SMALL {
                        new_height}px for {Path(file_path).name}")
        failed_files.append(output_path)
        return (fps, float('inf'), None)

    try:
        if is_video:
            generate_palette(file_path, palette_path,
                             fps, (new_width, new_height))
            create_gif_from_video(file_path, palette_path,
                                  temp_gif_path, fps, (new_width, new_height))
        else:  # For GIFs, adjust the command to directly process the GIF
            command = [
                'ffmpeg', '-i', str(file_path),
                '-filter_complex', f'[0:v]fps={fps},scale={new_width}:{
                    new_height}:flags=lanczos,setpts=PTS-STARTPTS[v];[v][1:v]paletteuse',
                '-i', str(palette_path),
                '-map', '[v]', '-y', str(temp_gif_path)
            ]
            if not run_ffmpeg_command(command):
                logging.error(f"GIF_CREATION_FAILED for {
                              Path(file_path).name}")
                return (fps, float('inf'), None)

        if not temp_gif_path.is_file():
            logging.error(f"GIF_CREATION_FAILED for {Path(file_path).name}")
            return (fps, float('inf'), None)

        try:
            os.remove(palette_path)
        except FileNotFoundError:
            pass

        initial_size = get_file_size(temp_gif_path)

        if initial_size > 90:
            logging.info(f"GIF_OVERSIZED initial_size={initial_size:.2f}MB for {
                         Path(file_path).name}, skipping optimization")
            return (fps, initial_size, str(temp_gif_path))

        optimized_size = optimize_gif_with_gifsicle(
            temp_gif_path, temp_gif_path, GIF_COMPRESSION['colors'], GIF_COMPRESSION['lossy_value'])

        logging.info(f"GIF_OPTIMIZED size={optimized_size:.2f}MB fps={
                     fps} for {Path(file_path).name}")
        return (fps, optimized_size, str(temp_gif_path))

    except Exception as e:
        logging.error(f"GIF_PROCESSING_ERROR {
                      str(e)} for {Path(file_path).name}")
        for file in [temp_gif_path, palette_path]:
            try:
                os.remove(file)
            except FileNotFoundError:
                pass
        return (fps, float('inf'), None)


@log_function_call
def process_gifs():
    """Process all gifs and videos for optimization, skipping existing outputs."""
    global failed_files

    input_dir = Path(INPUT_DIR)
    output_dir = Path(OUTPUT_DIR)

    # Ensure directories exist
    Path(TEMP_FILE_DIR).mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process video files from output directory
    for video_file in output_dir.glob('*.mp4'):
        output_gif = output_dir / (video_file.stem + ".gif")
        # Check if the output file already exists
        if not output_gif.exists():
            process_file(video_file, output_gif, is_video=True)
        else:
            logging.info(f"Skipping {video_file.name} as {
                         output_gif.name} already exists.")

    # Process GIF files from input directory
    for gif_file in input_dir.glob('*.gif'):
        output_gif = output_dir / (gif_file.stem + ".gif")
        # Check if the output file already exists
        if not output_gif.exists():
            process_file(gif_file, output_gif, is_video=False)
        else:
            logging.info(f"Skipping {gif_file.name} as {
                         output_gif.name} already exists.")

    # Remove duplicates before returning the list
    failed_files = list(set(failed_files))
    return failed_files
