#!/usr/bin/env python3
import os
import logging
import hashlib
import json
from pathlib import Path
import subprocess
import shutil
import concurrent.futures
from typing import Dict, Optional, List, Set, Tuple

# Import GPU detector
from src.gpu_detector import GPUDetector, AccelerationType


class GIFProcessor:
    """
    Advanced GIF processor class that handles GIF detection, optimization, and conversion
    using efficient parameters.

    Features:
    - Automatic detection of all GIF files in input directory
    - Intelligent optimization of GIF files to reduce file size
    - Smart caching to avoid reprocessing files with identical settings
    """

    # Define GIF file extension
    GIF_EXTENSION = '.gif'

    def __init__(self, config: Dict = None):
        """
        Initialize the GIF processor with configuration.

        Args:
            config: Configuration dictionary for the processor
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Set up directories
        self.input_dir = Path(self.config.get(
            'directories', {}).get('input', './input'))
        self.output_dir = Path(self.config.get(
            'directories', {}).get('output', './output'))
        self.temp_dir = Path(self.config.get(
            'directories', {}).get('temp', './temp'))

        # Get processing settings specific to GIF from config
        gif_processing = self.config.get('gif', {}).get('processing', {})

        # Get batch processing settings
        self.batch_size = gif_processing.get('batch_size', 1)
        self.num_threads = gif_processing.get('threads', 1)

        # Get quality settings
        self.max_size_mb = gif_processing.get('max_size_mb', 8)
        self.max_dimension = gif_processing.get('max_dimension', 720)

        # Ensure directories exist
        self._ensure_directories()

        # Flag for shutdown detection - will be set by ResourceManager
        self.shutdown_requested = False

        # Cache for processed files to avoid duplicate processing
        self.processed_cache_file = self.output_dir / '.processed_gif_cache.json'
        self.processed_cache = self._load_processed_cache()

    def _ensure_directories(self) -> None:
        """Ensure all required directories exist."""
        for directory in [self.input_dir, self.output_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            self.logger.debug(f"Ensured directory exists: {directory}")

    def _load_processed_cache(self) -> Dict:
        """
        Load the cache of processed files.

        Returns:
            Dict: Cache of processed files with their settings hash
        """
        if self.processed_cache_file.exists():
            try:
                with open(self.processed_cache_file, 'r') as f:
                    return json.load(f)
            except (json.JSONDecodeError, IOError) as e:
                self.logger.warning(f"Could not load processed cache: {e}")
                return {}
        return {}

    def _save_processed_cache(self) -> None:
        """Save the cache of processed files."""
        try:
            with open(self.processed_cache_file, 'w') as f:
                json.dump(self.processed_cache, f)
        except IOError as e:
            self.logger.warning(f"Could not save processed cache: {e}")

    def _get_settings_hash(self, file_path: Path) -> str:
        """
        Create a hash of the file and current processing settings.

        Args:
            file_path: Path to the file

        Returns:
            str: Hash representing the file and current settings
        """
        # Get file modification time and size
        stat = file_path.stat()
        file_info = f"{file_path.name}:{stat.st_mtime}:{stat.st_size}"

        # Get relevant settings from config that would affect output
        gif_config = self.config.get('gif', {})
        settings = {
            'target_size_mb': gif_config.get('target_size_mb', 10),
            'min_width': gif_config.get('min_width', 160),
            'colors': gif_config.get('colors', 256),
            'optimize_level': gif_config.get('optimize_level', 3),
        }

        # Create a hash of file info and settings
        settings_str = f"{file_info}:{json.dumps(settings, sort_keys=True)}"
        return hashlib.md5(settings_str.encode()).hexdigest()

    def find_gif_files(self) -> List[Path]:
        """
        Find all GIF files in the input directory.

        Returns:
            List[Path]: List of GIF files
        """
        gif_files = []

        for file_path in self.input_dir.glob('**/*'):
            if file_path.is_file() and file_path.suffix.lower() == self.GIF_EXTENSION:
                gif_files.append(file_path)

        self.logger.info(f"Found {len(gif_files)} GIF files")
        return gif_files

    def optimize_gif(self, input_file: Path) -> Optional[Path]:
        """
        Optimize a GIF file to reduce its size while maintaining reasonable quality.

        Args:
            input_file: Path to the input GIF file

        Returns:
            Optional[Path]: Path to the optimized file, or None if optimization failed
        """
        # Determine output filename and path
        output_file = self.output_dir / f"optimized_{input_file.name}"

        # Check if this file already has been processed with current settings
        settings_hash = self._get_settings_hash(input_file)
        if input_file.name in self.processed_cache and self.processed_cache[input_file.name] == settings_hash:
            self.logger.info(
                f"Skipping already processed file with same settings: {input_file}")
            if output_file.exists():
                return output_file
            # If cache says processed but file doesn't exist, reprocess
            self.logger.warning(
                f"Cache indicates file was processed but output doesn't exist: {output_file}")

        # Get gif parameters from config
        gif_config = self.config.get('gif', {})
        target_size_mb = gif_config.get('target_size_mb', 10)
        min_width = gif_config.get('min_width', 160)
        colors = gif_config.get('colors', 256)
        optimize_level = gif_config.get('optimize_level', 3)

        # Create temporary file for processing
        temp_output = self.temp_dir / f"temp_{input_file.name}"

        try:
            # Use gifsicle for GIF optimization
            # First pass: Optimize and resize if needed
            resize_arg = []

            # Check the current size of the GIF
            file_size_mb = input_file.stat().st_size / (1024*1024)

            # Only resize if the file is larger than target size
            if file_size_mb > target_size_mb:
                # Get the dimensions of the input GIF using gifsicle
                cmd = ["gifsicle", "--info", str(input_file)]
                result = subprocess.run(
                    cmd, capture_output=True, text=True, check=True)

                # Parse dimensions
                width = None
                for line in result.stdout.splitlines():
                    if "logical screen" in line:
                        # Extract dimensions like "logical screen 800x600"
                        parts = line.split()
                        if len(parts) >= 3:
                            dimensions = parts[2].split('x')
                            if len(dimensions) == 2:
                                width = int(dimensions[0])
                                break

                # Calculate resize if width is available
                if width and width > min_width:
                    # Start with a modest reduction and increase based on how much over target we are
                    resize_factor = 0.8
                    if file_size_mb > target_size_mb * 3:
                        resize_factor = 0.6
                    elif file_size_mb > target_size_mb * 2:
                        resize_factor = 0.7

                    new_width = max(int(width * resize_factor), min_width)
                    resize_arg = ["--resize", f"{new_width}x"]

            # Optimize command
            optimize_arg = f"--optimize={optimize_level}"
            colors_arg = f"--colors={colors}"

            cmd = ["gifsicle", "--no-warnings", optimize_arg, colors_arg]
            cmd.extend(resize_arg)
            cmd.extend(["-o", str(temp_output), str(input_file)])

            self.logger.info(f"Optimizing GIF: {input_file} to {temp_output}")
            self.logger.debug(f"Gifsicle command: {' '.join(cmd)}")

            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                text=True
            )

            # Check if the file was reduced enough
            if temp_output.exists():
                temp_size_mb = temp_output.stat().st_size / (1024*1024)

                # If still too large, try more aggressive optimization
                if temp_size_mb > target_size_mb and colors > 64:
                    # Try with reduced colors
                    reduced_colors = max(64, colors // 2)
                    colors_arg = f"--colors={reduced_colors}"

                    cmd = ["gifsicle", "--no-warnings",
                           optimize_arg, colors_arg]
                    cmd.extend(resize_arg)
                    cmd.extend(["-o", str(temp_output), str(input_file)])

                    self.logger.info(
                        f"Re-optimizing GIF with fewer colors: {input_file}")
                    self.logger.debug(f"Gifsicle command: {' '.join(cmd)}")

                    result = subprocess.run(
                        cmd,
                        check=True,
                        capture_output=True,
                        text=True
                    )

            # Move the temp file to the output location
            shutil.move(str(temp_output), str(output_file))

            # Save to processed cache
            self.processed_cache[input_file.name] = settings_hash
            self._save_processed_cache()

            # Calculate and log space savings
            original_size = input_file.stat().st_size / (1024*1024)
            new_size = output_file.stat().st_size / (1024*1024)
            savings_percent = (1 - (new_size / original_size)) * 100

            self.logger.info(
                f"GIF optimization completed. Size reduction: {original_size:.2f}MB -> {new_size:.2f}MB ({savings_percent:.1f}% saved)")

            return output_file

        except subprocess.CalledProcessError as e:
            self.logger.error(
                f"Gifsicle error during optimization: {e.stderr}")
            if temp_output.exists():
                temp_output.unlink()
            return None
        except Exception as e:
            self.logger.error(
                f"Error during GIF optimization: {e}", exc_info=True)
            if temp_output.exists():
                temp_output.unlink()
            return None

    def _process_gif_batch(self, batch_files: List[Path]) -> Dict[Path, Optional[Path]]:
        """
        Process a batch of GIF files.

        Args:
            batch_files: List of GIF files to process

        Returns:
            Dict[Path, Optional[Path]]: Mapping of input paths to output paths
        """
        results = {}

        for gif_file in batch_files:
            # Optimize the GIF file
            optimized = self.optimize_gif(gif_file)
            results[gif_file] = optimized

        return results

    def process_gifs(self) -> Dict[Path, Optional[Path]]:
        """
        Process all GIF files found in the input directory.

        Returns:
            Dict[Path, Optional[Path]]: Dictionary mapping input paths to output paths
        """
        # Find all GIF files
        gif_files = self.find_gif_files()

        # Return early if no files found
        if not gif_files:
            self.logger.info("No GIF files found to process")
            return {}

        # Dictionary to store results
        results = {}

        # Process files based on threading configuration
        if self.batch_size > 1 and self.num_threads > 1:
            # Split files into batches
            batches = [gif_files[i:i + self.batch_size]
                       for i in range(0, len(gif_files), self.batch_size)]

            self.logger.info(
                f"Processing {len(gif_files)} GIFs in {len(batches)} batches using {self.num_threads} threads")

            # Process batches with thread pool
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
                # Submit batch processing tasks
                future_to_batch = {executor.submit(
                    self._process_gif_batch, batch): batch for batch in batches}

                # Collect results as they complete
                for future in concurrent.futures.as_completed(future_to_batch):
                    if self.shutdown_requested:
                        executor.shutdown(wait=False)
                        self.logger.info(
                            "Shutdown requested, cancelling remaining tasks")
                        break

                    batch_results = future.result()
                    results.update(batch_results)
        else:
            # Process files sequentially
            self.logger.info(f"Processing {len(gif_files)} GIFs sequentially")

            # Process each file
            for gif_file in gif_files:
                if self.shutdown_requested:
                    self.logger.info("Shutdown requested, stopping processing")
                    break

                result = self._process_gif(gif_file)
                results[gif_file] = result

        # Clean up temporary files
        for temp_file in self.temp_dir.glob(f'*{self.GIF_EXTENSION}'):
            if temp_file.is_file():
                try:
                    temp_file.unlink()
                    self.logger.debug(
                        f"Cleaned up temporary file: {temp_file}")
                except OSError as e:
                    self.logger.warning(
                        f"Could not remove temporary file {temp_file}: {e}")

        return results

    def _process_gif(self, gif_file: Path) -> Optional[Path]:
        """
        Process a single GIF file.

        Args:
            gif_file: Path to the GIF file

        Returns:
            Optional[Path]: Path to the optimized GIF, or None if processing failed
        """
        try:
            # Skip if file doesn't exist
            if not gif_file.exists():
                self.logger.warning(f"File does not exist: {gif_file}")
                return None

            # Check for shutdown request
            if self.shutdown_requested:
                return None

            # Set output path
            output_file = self.output_dir / f"optimized_{gif_file.name}"

            # Optimize the GIF
            success = self._optimize_gif(gif_file, output_file)

            if success and output_file.exists():
                return output_file
            else:
                return None

        except Exception as e:
            self.logger.error(
                f"Error processing GIF {gif_file}: {e}", exc_info=True)
            return None

    def _optimize_gif(self, input_file: Path, output_file: Path) -> bool:
        """
        Optimize a GIF file to reduce size while maintaining quality.

        Args:
            input_file: Path to the input GIF
            output_file: Path to save the optimized GIF

        Returns:
            bool: True if optimization was successful, False otherwise
        """
        # Implementation details will depend on the specific optimization method
        # This is a placeholder that should be replaced with actual implementation
        try:
            # Check for shutdown request
            if self.shutdown_requested:
                return False

            # Get original file size
            original_size = input_file.stat().st_size / (1024 * 1024)  # Size in MB

            # If already under max size, just copy
            if original_size <= self.max_size_mb:
                shutil.copy(input_file, output_file)
                self.logger.info(
                    f"GIF already under max size ({original_size:.2f} MB), copied to output")
                return True

            # TODO: Implement actual GIF optimization logic
            # For now, just copy the file
            shutil.copy(input_file, output_file)
            return True

        except Exception as e:
            self.logger.error(f"Error optimizing GIF: {e}", exc_info=True)
            return False

    def batch_process(self) -> Dict[Path, Optional[Path]]:
        """
        Process all GIF files in batch mode.

        Returns:
            Dict[Path, Optional[Path]]: Dictionary mapping input paths to output paths
        """
        return self.process_gifs()


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simple config for testing
    config = {
        'directories': {
            'input': './input',
            'output': './output',
            'temp': './temp'
        },
        'gif': {
            'target_size_mb': 5,
            'min_width': 200,
            'colors': 128,
            'optimize_level': 3,
            'processing': {
                'threads': 1,
                'batch_size': 1
            }
        }
    }

    processor = GIFProcessor(config)

    # Process all GIFs
    results = processor.batch_process()

    # Print results
    print("\nProcessing results:")
    for input_file, output_file in results.items():
        status = "SUCCESS" if output_file else "FAILED"
        print(
            f"{input_file.name} → {output_file.name if output_file else 'N/A'} : {status}")
