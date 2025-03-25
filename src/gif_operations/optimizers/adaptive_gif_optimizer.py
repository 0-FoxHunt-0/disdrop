import os
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Tuple, Dict, Optional, List, Union, NamedTuple
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import uuid

import numpy as np
from PIL import Image, ImageSequence

from src.default_config import TEMP_FILE_DIR
from src.gif_operations.file_processor import FileProcessor
from src.gif_operations.enhanced_analyzer import EnhancedGIFAnalysis, EnhancedAnalyzer
from src.logging_system import ModernLogStyle, log_gif_progress, run_ffmpeg_command

# Configure logger
logger = logging.getLogger(__name__)


class OptimizationStep(NamedTuple):
    """Represents a single step in the optimization process"""
    scale_factor: float
    colors: int
    lossy_value: int
    dither_method: str
    quality_estimate: float
    expected_size_reduction: float


class OptimizationResult(NamedTuple):
    """Result of an optimization operation"""
    success: bool
    file_path: Optional[Path]
    size_mb: float
    settings: Dict
    quality_score: float
    error: Optional[str] = None


class AdaptiveGIFOptimizer:
    """Advanced GIF optimizer with perceptual quality preservation"""

    def __init__(self, max_workers: int = 4):
        self.analyzer = EnhancedAnalyzer(max_workers=max_workers)
        self.file_processor = FileProcessor()
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

        # Optimization strategy constants
        self.MIN_ACCEPTABLE_QUALITY = 0.7
        self.MAX_LOSSY_VALUE = 150  # Limit max lossy value to preserve quality
        self.MIN_COLORS = 32  # Minimum colors to preserve basic quality

        # Create base temp directory if it doesn't exist
        os.makedirs(TEMP_FILE_DIR, exist_ok=True)

    async def optimize(self, input_path: Path, output_path: Path, target_size_mb: float) -> OptimizationResult:
        """Optimize GIF with adaptive quality-aware approach

        Args:
            input_path: Path to the input GIF
            output_path: Path where optimized GIF should be saved
            target_size_mb: Target size in megabytes

        Returns:
            OptimizationResult with optimization details
        """
        try:
            # Log the start of the optimization process
            logger.info(f"Starting adaptive GIF optimization for {input_path}")
            log_gif_progress(
                f"Analyzing GIF content: {input_path.name}", "processing")

            # Analyze the GIF content
            analysis = await self.analyzer.analyze_gif(input_path)

            # Log analysis results
            logger.info(
                f"GIF analysis results - Motion: {analysis.motion_score:.2f}, Complexity: {analysis.complexity_score:.2f}, Colors: {analysis.optimal_colors}")

            # Start with a simple lossless optimization
            log_gif_progress(
                f"Applying lossless optimization...", "optimizing")
            lossless_path = self._create_temp_path("lossless")
            lossless_result = await self._apply_lossless_optimization(input_path, lossless_path)

            if not lossless_result.success:
                logger.error(f"Lossless optimization failed for {input_path}")
                log_gif_progress(f"Lossless optimization failed!", "error")
                return OptimizationResult(
                    success=False,
                    file_path=None,
                    size_mb=0.0,
                    settings={},
                    quality_score=0.0,
                    error="Lossless optimization failed"
                )

            # Check if lossless optimization is sufficient
            current_size = lossless_result.size_mb
            original_size = self.file_processor.get_file_size(input_path)

            # Log the size reduction
            reduction_percent = ((original_size - current_size) /
                                 original_size * 100) if original_size > 0 else 0
            logger.info(
                f"Lossless optimization: {original_size:.2f}MB → {current_size:.2f}MB ({reduction_percent:.1f}% reduction)")
            log_gif_progress(
                f"Lossless result: {current_size:.2f}MB ({reduction_percent:.1f}% reduction)", "optimizing")

            if current_size <= target_size_mb:
                logger.info(
                    f"Lossless optimization sufficient: {current_size:.2f}MB <= {target_size_mb:.2f}MB")
                log_gif_progress(
                    f"Target size achieved with lossless optimization!", "success")
                shutil.copy2(lossless_path, output_path)
                return OptimizationResult(
                    success=True,
                    file_path=output_path,
                    size_mb=current_size,
                    settings=lossless_result.settings,
                    quality_score=1.0  # Perfect quality with lossless
                )

            # Generate optimization steps based on content analysis
            log_gif_progress(
                f"Generating optimization strategy...", "processing")
            optimization_steps = self._generate_optimization_steps(
                analysis, current_size, target_size_mb
            )

            # Log the optimization plan
            logger.info(
                f"Generated {len(optimization_steps)} optimization steps")
            if len(optimization_steps) > 0:
                log_gif_progress(
                    f"Testing {len(optimization_steps)} optimization strategies", "optimizing")
            else:
                log_gif_progress(
                    f"No viable optimization strategies found", "warning")

            # Try optimization steps until target size is reached
            best_result = lossless_result
            best_quality_score = 1.0

            for step_num, step in enumerate(optimization_steps):
                logger.info(f"Trying optimization step {step_num+1}/{len(optimization_steps)}: "
                            f"scale={step.scale_factor:.2f}, colors={step.colors}, "
                            f"lossy={step.lossy_value}, dither={step.dither_method}")

                # Update progress log
                log_gif_progress(
                    f"Step {step_num+1}/{len(optimization_steps)}: scale={step.scale_factor:.2f}, colors={step.colors}, lossy={step.lossy_value}",
                    "optimizing"
                )

                step_output = self._create_temp_path(f"step_{step_num}")

                # Apply this optimization step
                result = await self._apply_optimization_step(
                    lossless_path, step_output, step
                )

                if not result.success:
                    logger.warning(
                        f"Optimization step {step_num+1} failed, continuing with next step")
                    log_gif_progress(
                        f"Step {step_num+1} failed, trying next approach", "warning")
                    continue

                # Check if this step reached the target size
                if result.size_mb <= target_size_mb:
                    logger.info(f"Target size reached at step {step_num+1}: "
                                f"{result.size_mb:.2f}MB <= {target_size_mb:.2f}MB")

                    log_gif_progress(
                        f"Target size reached: {result.size_mb:.2f}MB with quality {step.quality_estimate:.2f}",
                        "success"
                    )

                    # Copy to output path
                    shutil.copy2(step_output, output_path)
                    return OptimizationResult(
                        success=True,
                        file_path=output_path,
                        size_mb=result.size_mb,
                        settings=result.settings,
                        quality_score=step.quality_estimate
                    )

                # Keep track of best result (closest to target size with acceptable quality)
                if (result.size_mb < best_result.size_mb and
                        step.quality_estimate >= self.MIN_ACCEPTABLE_QUALITY):
                    best_result = result
                    best_quality_score = step.quality_estimate
                    logger.info(
                        f"New best result: {result.size_mb:.2f}MB with quality {step.quality_estimate:.2f}")
                    log_gif_progress(
                        f"New best result: {result.size_mb:.2f}MB (quality: {step.quality_estimate:.2f})", "optimizing")

            # If we couldn't reach target size with acceptable quality, use best result
            if best_result.file_path and os.path.exists(best_result.file_path):
                logger.warning(f"Target size not reached, but using best result: "
                               f"{best_result.size_mb:.2f}MB (target: {target_size_mb:.2f}MB) "
                               f"with quality score {best_quality_score:.2f}")

                log_gif_progress(
                    f"Using best compromise: {best_result.size_mb:.2f}MB (quality: {best_quality_score:.2f})",
                    "warning"
                )

                shutil.copy2(best_result.file_path, output_path)
                return OptimizationResult(
                    success=True,
                    file_path=output_path,
                    size_mb=best_result.size_mb,
                    settings=best_result.settings,
                    quality_score=best_quality_score
                )

            # As a last resort, try a more aggressive approach if all else failed
            logger.warning(
                f"Regular optimization steps failed, trying aggressive optimization")
            log_gif_progress(
                f"Trying aggressive optimization as last resort", "warning")

            aggressive_path = self._create_temp_path("aggressive")
            aggressive_result = await self._apply_aggressive_optimization(
                input_path, aggressive_path, target_size_mb
            )

            if aggressive_result.success and aggressive_result.file_path:
                log_gif_progress(
                    f"Aggressive optimization result: {aggressive_result.size_mb:.2f}MB",
                    "warning" if aggressive_result.size_mb > target_size_mb else "success"
                )

                shutil.copy2(aggressive_result.file_path, output_path)
                return OptimizationResult(
                    success=True,
                    file_path=output_path,
                    size_mb=aggressive_result.size_mb,
                    settings=aggressive_result.settings,
                    quality_score=aggressive_result.quality_score
                )

            # If all optimization approaches failed
            logger.error(
                f"All optimization approaches failed for {input_path}")
            log_gif_progress(f"All optimization approaches failed", "error")

            return OptimizationResult(
                success=False,
                file_path=None,
                size_mb=0.0,
                settings={},
                quality_score=0.0,
                error="All optimization approaches failed"
            )

        except Exception as e:
            logger.exception(f"Error optimizing GIF {input_path}: {e}")
            log_gif_progress(f"Optimization error: {str(e)}", "error")

            return OptimizationResult(
                success=False,
                file_path=None,
                size_mb=0.0,
                settings={},
                quality_score=0.0,
                error=str(e)
            )

    async def _apply_lossless_optimization(self, input_path: Path, output_path: Path) -> OptimizationResult:
        """Apply lossless optimization using gifsicle"""
        try:
            # Run gifsicle with optimize=3 (maximum) but no lossy compression or color reduction
            cmd = [
                'gifsicle',
                '--optimize=3',
                '--no-warnings',
                '--no-conserve-memory',
                str(input_path),
                '-o', str(output_path)
            ]

            logger.info(f"Running lossless gifsicle command: {' '.join(cmd)}")

            # Run in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            process = await loop.run_in_executor(
                self.executor,
                lambda: subprocess.run(cmd, capture_output=True, text=True)
            )

            if process.returncode != 0:
                logger.error(f"Gifsicle lossless error: {process.stderr}")
                return OptimizationResult(
                    success=False,
                    file_path=None,
                    size_mb=0.0,
                    settings={},
                    quality_score=0.0,
                    error=process.stderr
                )

            # Get file size
            size_mb = self.file_processor.get_file_size(output_path)
            logger.info(f"Lossless optimization complete: {size_mb:.2f}MB")

            return OptimizationResult(
                success=True,
                file_path=output_path,
                size_mb=size_mb,
                settings={"optimize": 3, "lossy": 0,
                          "colors": 256, "scale": 1.0},
                quality_score=1.0  # Lossless = perfect quality
            )

        except Exception as e:
            logger.exception(f"Error in lossless optimization: {e}")
            return OptimizationResult(
                success=False,
                file_path=None,
                size_mb=0.0,
                settings={},
                quality_score=0.0,
                error=str(e)
            )

    async def _apply_optimization_step(self, input_path: Path, output_path: Path,
                                       step: OptimizationStep) -> OptimizationResult:
        """Apply a specific optimization step"""
        try:
            # Prepare gifsicle command
            cmd = [
                'gifsicle',
                '--optimize=3',
                '--no-warnings',
                '--no-conserve-memory',
                '--colors', str(step.colors)
            ]

            # Add lossy parameter if needed
            if step.lossy_value > 0:
                cmd.extend(['--lossy', str(step.lossy_value)])

            # Add scale parameter if needed
            if step.scale_factor < 1.0:
                cmd.extend(['--scale', f"{step.scale_factor:.3f}"])

            # Add dither method if not 'none'
            if step.dither_method != 'none':
                if ':' in step.dither_method:
                    # For bayer with scale parameter - handle both formats (bayer-scale and bayer_scale)
                    dither_parts = step.dither_method.split(':')
                    dither_type = dither_parts[0]

                    # Add the basic dither type
                    cmd.append(f'--dither={dither_type}')

                    # Handle any additional parameters - older gifsicle doesn't support combined parameters
                    if len(dither_parts) > 1:
                        # For bayer scale, add as a separate parameter
                        if 'bayer-scale=' in dither_parts[1] or 'bayer_scale=' in dither_parts[1]:
                            # Extract the value regardless of the format used
                            if 'bayer-scale=' in dither_parts[1]:
                                scale_value = dither_parts[1].split(
                                    'bayer-scale=')[1]
                            else:
                                scale_value = dither_parts[1].split(
                                    'bayer_scale=')[1]

                            # Add as a separate parameter
                            cmd.extend(['--scale', scale_value])
                else:
                    # Simple dither method
                    cmd.append(f'--dither={step.dither_method}')

            # Add input and output paths
            cmd.extend([str(input_path), '-o', str(output_path)])

            logger.info(f"Running gifsicle command: {' '.join(cmd)}")

            # Run in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            process = await loop.run_in_executor(
                self.executor,
                lambda: subprocess.run(cmd, capture_output=True, text=True)
            )

            if process.returncode != 0:
                logger.error(f"Gifsicle optimization error: {process.stderr}")
                return OptimizationResult(
                    success=False,
                    file_path=None,
                    size_mb=0.0,
                    settings={},
                    quality_score=0.0,
                    error=process.stderr
                )

            # Get file size
            size_mb = self.file_processor.get_file_size(output_path)

            # Calculate size reduction from original
            original_size = self.file_processor.get_file_size(input_path)
            reduction_percent = ((original_size - size_mb) /
                                 original_size * 100) if original_size > 0 else 0

            logger.info(
                f"Optimization step result: {size_mb:.2f}MB ({reduction_percent:.1f}% reduction)")

            # Create settings dict
            settings = {
                "optimize": 3,
                "lossy": step.lossy_value,
                "colors": step.colors,
                "scale": step.scale_factor,
                "dither": step.dither_method
            }

            return OptimizationResult(
                success=True,
                file_path=output_path,
                size_mb=size_mb,
                settings=settings,
                quality_score=step.quality_estimate
            )

        except Exception as e:
            logger.exception(f"Error in optimization step: {e}")
            return OptimizationResult(
                success=False,
                file_path=None,
                size_mb=0.0,
                settings={},
                quality_score=0.0,
                error=str(e)
            )

    async def _apply_aggressive_optimization(self, input_path: Path, output_path: Path,
                                             target_size_mb: float) -> OptimizationResult:
        """Apply aggressive optimization as a last resort"""
        try:
            # Get original size to calculate required reduction
            original_size = self.file_processor.get_file_size(input_path)
            reduction_factor = target_size_mb / original_size if original_size > 0 else 0.5

            logger.info(
                f"Applying aggressive optimization - need {(1-reduction_factor)*100:.1f}% reduction")
            log_gif_progress(
                f"Aggressive optimization - target: {target_size_mb:.2f}MB", "warning")

            # Calculate aggressive parameters
            # MODIFIED: Increase minimum scale and color values to reduce artifacts
            scale_factor = max(0.6, min(1.0, reduction_factor ** 0.5))
            colors = max(self.MIN_COLORS, int(256 * reduction_factor))
            # Ensure at least 32 colors to prevent black artifacts
            colors = max(32, colors)
            # MODIFIED: Reduce maximum lossy value to prevent black artifacts
            lossy = min(80, int(150 * (1 - reduction_factor)))

            # Log the aggressive parameters
            logger.info(
                f"Aggressive parameters: scale={scale_factor:.2f}, colors={colors}, lossy={lossy}")

            # Prepare gifsicle command
            cmd = [
                'gifsicle',
                '--optimize=3',
                '--no-warnings',
                '--no-conserve-memory',
                '--colors', str(colors),
                '--lossy', str(lossy),
                '--scale', f"{scale_factor:.3f}",
                '--dither=ordered',  # MODIFIED: Changed to ordered dithering for fewer artifacts
                str(input_path),
                '-o', str(output_path)
            ]

            logger.info(
                f"Running aggressive gifsicle command: {' '.join(cmd)}")

            # Run in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            process = await loop.run_in_executor(
                self.executor,
                lambda: subprocess.run(cmd, capture_output=True, text=True)
            )

            if process.returncode != 0:
                logger.error(
                    f"Aggressive optimization error: {process.stderr}")
                return OptimizationResult(
                    success=False,
                    file_path=None,
                    size_mb=0.0,
                    settings={},
                    quality_score=0.0,
                    error=process.stderr
                )

            # Get file size
            size_mb = self.file_processor.get_file_size(output_path)

            # Log the result
            reduction_achieved = (
                (original_size - size_mb) / original_size * 100) if original_size > 0 else 0
            logger.info(
                f"Aggressive optimization result: {size_mb:.2f}MB ({reduction_achieved:.1f}% reduction)")

            # Estimate quality (will be low for aggressive optimization)
            quality_score = max(0.1, min(0.6, reduction_factor))

            settings = {
                "optimize": 3,
                "lossy": lossy,
                "colors": colors,
                "scale": scale_factor,
                "dither": "ordered",
                "mode": "aggressive"
            }

            return OptimizationResult(
                success=True,
                file_path=output_path,
                size_mb=size_mb,
                settings=settings,
                quality_score=quality_score
            )

        except Exception as e:
            logger.exception(f"Error in aggressive optimization: {e}")
            return OptimizationResult(
                success=False,
                file_path=None,
                size_mb=0.0,
                settings={},
                quality_score=0.0,
                error=str(e)
            )

    def _generate_optimization_steps(self, analysis: EnhancedGIFAnalysis,
                                     current_size: float, target_size_mb: float) -> List[OptimizationStep]:
        """Generate a series of optimization steps based on file analysis and target size"""
        steps = []
        reduction_needed = (current_size - target_size_mb) / \
            current_size if current_size > target_size_mb else 0

        # Default progression of quality steps from least to most aggressive
        # MODIFIED: Updated dithering options to reduce black artifacts
        quality_steps = [
            # Very high quality (minimal loss)
            OptimizationStep(
                scale_factor=1.0,
                colors=256,
                lossy_value=20,
                dither_method='none',  # No dithering for highest quality
                quality_estimate=0.95,
                expected_size_reduction=0.2
            ),
            # High quality
            OptimizationStep(
                scale_factor=1.0,
                colors=192,
                lossy_value=40,
                dither_method='none',  # No dithering to prevent artifacts
                quality_estimate=0.9,
                expected_size_reduction=0.3
            ),
            # Medium-high quality
            OptimizationStep(
                scale_factor=0.95,
                colors=128,
                lossy_value=60,
                dither_method='ordered',  # Changed to ordered dithering to reduce artifacts
                quality_estimate=0.8,
                expected_size_reduction=0.5
            ),
            # Medium quality
            OptimizationStep(
                scale_factor=0.9,
                colors=96,
                lossy_value=80,
                dither_method='ordered',  # Changed to ordered dithering to reduce artifacts
                quality_estimate=0.7,
                expected_size_reduction=0.6
            ),
            # Low quality (last resort)
            OptimizationStep(
                scale_factor=0.8,
                colors=64,
                lossy_value=100,
                dither_method='ordered',  # Changed to ordered dithering to reduce artifacts
                quality_estimate=0.5,
                expected_size_reduction=0.7
            )
        ]

        # If we need more aggressive reduction, add more steps
        if reduction_needed > 0.7:
            # MODIFIED: Increased minimum colors to prevent black artifacts
            quality_steps.append(
                OptimizationStep(
                    scale_factor=0.7,
                    colors=48,  # Increased from 32 to reduce black artifacts
                    lossy_value=120,  # Reduced from a higher value
                    dither_method='ordered',  # Changed to ordered dithering to reduce artifacts
                    quality_estimate=0.4,
                    expected_size_reduction=0.8
                )
            )

        # For extremely large GIFs we might need a very aggressive option
        if reduction_needed > 0.9:
            # MODIFIED: Increased minimum colors and reduced lossy value
            quality_steps.append(
                OptimizationStep(
                    scale_factor=0.5,
                    colors=32,  # Increased from 16 to reduce black artifacts
                    lossy_value=150,  # Reduced from 200 to reduce artifacts
                    dither_method='ordered',  # Changed dithering method
                    quality_estimate=0.2,
                    expected_size_reduction=0.95
                )
            )

        # Filter steps based on required reduction
        filtered_steps = [
            step for step in quality_steps
            if step.expected_size_reduction >= reduction_needed
        ]

        # Sort by expected quality (highest first)
        filtered_steps.sort(key=lambda s: s.quality_estimate, reverse=True)

        # Log the filtered steps
        logger.info(
            f"Filtered to {len(filtered_steps)} viable steps based on required {reduction_needed*100:.1f}% reduction")

        # Always include at least one step
        if not filtered_steps and quality_steps:
            filtered_steps = [quality_steps[0]]
            logger.info(
                f"No steps met reduction target, including first step as fallback")

        return filtered_steps

    def _get_simpler_dither(self, dither_method: str) -> str:
        """Get a simplified dither method that consumes less resources"""
        if dither_method in ['none', '']:
            return 'none'
        # MODIFIED: Changed to use ordered dithering instead of bayer for better quality
        if dither_method == 'floyd-steinberg':
            return 'ordered'  # Less resource-intensive alternative
        elif ':' in dither_method:
            # For complex methods, use ordered instead
            return 'ordered'
        else:
            return dither_method

    def _create_temp_path(self, prefix: str) -> Path:
        """Create a temporary file path"""
        return Path(TEMP_FILE_DIR) / f"{prefix}_{uuid.uuid4().hex}.gif"
