"""
PIL/Pillow-based GIF Compression
Provides advanced compression methods using PIL for palette quantization,
frame optimization, and post-processing
"""

import os
from typing import Dict, Any, Optional, Tuple, List
import logging
from PIL import Image, ImageSequence, ImagePalette
import math

logger = logging.getLogger(__name__)


class PILCompressor:
    """PIL-based GIF compression utilities"""
    
    @staticmethod
    def quantize_with_adaptive_palette(gif_path: str, output_path: str, max_colors: int = 256,
                                      method: str = 'median_cut') -> bool:
        """
        Quantize GIF using PIL's adaptive palette methods.
        
        Args:
            gif_path: Input GIF path
            output_path: Output GIF path
            max_colors: Maximum number of colors (2-256)
            method: Quantization method ('median_cut' or 'octree')
        
        Returns:
            True if successful
        """
        try:
            max_colors = max(2, min(256, int(max_colors)))
            
            # PIL quantization methods: 0=median cut, 1=maximum coverage, 2=fast octree, 3=libimagequant
            # Use median cut (0) as default, or fast octree (2) if requested
            if method == 'octree':
                quantize_method = 2  # Fast octree
            else:
                quantize_method = 0  # Median cut
            
            frames = []
            durations = []
            disposal_methods = []
            
            with Image.open(gif_path) as img:
                # Collect all frames and their metadata
                for frame in ImageSequence.Iterator(img):
                    frame_copy = frame.copy()
                    frames.append(frame_copy)
                    
                    # Get frame duration
                    duration = frame_copy.info.get('duration', 100)
                    durations.append(duration)
                    
                    # Get disposal method
                    disposal = frame_copy.info.get('disposal', 2)
                    disposal_methods.append(disposal)
                
                if not frames:
                    logger.warning("No frames found in GIF")
                    return False
                
                # Create a composite image for palette generation (sample frames)
                # This helps create a better global palette
                sample_frames = frames[::max(1, len(frames) // 10)]  # Sample ~10 frames
                if not sample_frames:
                    sample_frames = frames[:1]
                
                # Create a temporary composite for palette analysis
                composite_width = max(f.size[0] for f in sample_frames)
                composite_height = sum(f.size[1] for f in sample_frames)
                composite = Image.new('RGB', (composite_width, composite_height))
                
                y_offset = 0
                for frame in sample_frames:
                    # Convert to RGB if needed
                    if frame.mode != 'RGB':
                        frame_rgb = frame.convert('RGB')
                    else:
                        frame_rgb = frame
                    composite.paste(frame_rgb, (0, y_offset))
                    y_offset += frame.size[1]
                
                # Generate optimized palette from composite
                quantized_composite = composite.quantize(colors=max_colors, method=quantize_method, dither=Image.Dither.NONE)
                palette = quantized_composite.getpalette()
                
                # Apply palette to all frames
                optimized_frames = []
                for i, frame in enumerate(frames):
                    # Convert frame to RGB first
                    if frame.mode != 'RGB':
                        frame_rgb = frame.convert('RGB')
                    else:
                        frame_rgb = frame
                    
                    # Quantize with the shared palette
                    # Create a temporary image with the palette to use as reference
                    palette_img = Image.new('P', (1, 1))
                    palette_img.putpalette(palette)
                    quantized = frame_rgb.quantize(palette=palette_img, dither=Image.Dither.FLOYDSTEINBERG)
                    optimized_frames.append(quantized)
                
                # Save optimized GIF
                if optimized_frames:
                    optimized_frames[0].save(
                        output_path,
                        'GIF',
                        save_all=True,
                        append_images=optimized_frames[1:],
                        duration=durations,
                        loop=0,
                        disposal=disposal_methods,
                        optimize=True
                    )
                    
                    return os.path.exists(output_path) and os.path.getsize(output_path) > 0
            
            return False
        except Exception as e:
            logger.debug(f"PIL adaptive palette quantization failed: {e}")
            return False
    
    @staticmethod
    def optimize_frame_disposal(gif_path: str, output_path: str) -> bool:
        """
        Optimize frame disposal methods to reduce file size.
        Analyzes frames and sets optimal disposal methods.
        
        Returns:
            True if successful
        """
        try:
            frames = []
            durations = []
            
            with Image.open(gif_path) as img:
                prev_frame = None
                for frame in ImageSequence.Iterator(img):
                    frame_copy = frame.copy()
                    
                    # Determine optimal disposal method
                    if prev_frame is None:
                        # First frame: use no disposal (0) or keep (1)
                        disposal = 1
                    else:
                        # Compare with previous frame to determine if we can use previous disposal
                        # For simplicity, use background disposal (2) for most cases
                        # This allows better compression
                        disposal = 2
                    
                    frame_copy.info['disposal'] = disposal
                    frames.append(frame_copy)
                    durations.append(frame_copy.info.get('duration', 100))
                    prev_frame = frame_copy
                
                if frames:
                    frames[0].save(
                        output_path,
                        'GIF',
                        save_all=True,
                        append_images=frames[1:],
                        duration=durations,
                        loop=0,
                        optimize=True
                    )
                    return os.path.exists(output_path)
            
            return False
        except Exception as e:
            logger.debug(f"PIL frame disposal optimization failed: {e}")
            return False
    
    @staticmethod
    def remove_duplicate_frames(gif_path: str, output_path: str, similarity_threshold: float = 0.95) -> bool:
        """
        Remove duplicate or very similar frames using PIL.
        
        Args:
            gif_path: Input GIF path
            output_path: Output GIF path
            similarity_threshold: Threshold for frame similarity (0.0-1.0)
        
        Returns:
            True if successful
        """
        try:
            frames = []
            durations = []
            disposal_methods = []
            
            with Image.open(gif_path) as img:
                prev_frame_data = None
                
                for frame in ImageSequence.Iterator(img):
                    frame_copy = frame.copy()
                    
                    # Convert to RGB for comparison
                    if frame_copy.mode != 'RGB':
                        frame_rgb = frame_copy.convert('RGB')
                    else:
                        frame_rgb = frame_copy
                    
                    # Calculate frame hash/similarity
                    # Simple approach: compare histograms
                    if prev_frame_data is not None:
                        # Calculate histogram similarity
                        hist1 = frame_rgb.histogram()
                        hist2 = prev_frame_data['histogram']
                        
                        # Normalize histograms
                        total1 = sum(hist1)
                        total2 = sum(hist2)
                        if total1 > 0 and total2 > 0:
                            hist1_norm = [h / total1 for h in hist1]
                            hist2_norm = [h / total2 for h in hist2]
                            
                            # Calculate correlation
                            similarity = sum(min(h1, h2) for h1, h2 in zip(hist1_norm, hist2_norm))
                            
                            if similarity >= similarity_threshold:
                                # Skip this frame (duplicate)
                                # Add its duration to the previous frame
                                if frames:
                                    durations[-1] += frame_copy.info.get('duration', 100)
                                continue
                    
                    # Keep this frame
                    frames.append(frame_copy)
                    durations.append(frame_copy.info.get('duration', 100))
                    disposal_methods.append(frame_copy.info.get('disposal', 2))
                    
                    # Store for next comparison
                    prev_frame_data = {
                        'histogram': frame_rgb.histogram()
                    }
            
            if frames:
                frames[0].save(
                    output_path,
                    'GIF',
                    save_all=True,
                    append_images=frames[1:],
                    duration=durations,
                    loop=0,
                    disposal=disposal_methods,
                    optimize=True
                )
                return os.path.exists(output_path)
            
            return False
        except Exception as e:
            logger.debug(f"PIL duplicate frame removal failed: {e}")
            return False
    
    @staticmethod
    def optimize_palette_entries(gif_path: str, output_path: str) -> bool:
        """
        Optimize palette by removing unused color entries.
        
        Returns:
            True if successful
        """
        try:
            frames = []
            durations = []
            disposal_methods = []
            
            with Image.open(gif_path) as img:
                for frame in ImageSequence.Iterator(img):
                    frame_copy = frame.copy()
                    frames.append(frame_copy)
                    durations.append(frame_copy.info.get('duration', 100))
                    disposal_methods.append(frame_copy.info.get('disposal', 2))
            
            if frames:
                # PIL's optimize=True already handles palette optimization
                frames[0].save(
                    output_path,
                    'GIF',
                    save_all=True,
                    append_images=frames[1:],
                    duration=durations,
                    loop=0,
                    disposal=disposal_methods,
                    optimize=True
                )
                return os.path.exists(output_path)
            
            return False
        except Exception as e:
            logger.debug(f"PIL palette optimization failed: {e}")
            return False
    
    @staticmethod
    def compress_gif(gif_path: str, output_path: str, max_colors: Optional[int] = None,
                    remove_duplicates: bool = True, optimize_disposal: bool = True) -> Tuple[bool, float]:
        """
        Comprehensive PIL-based GIF compression.
        
        Args:
            gif_path: Input GIF path
            output_path: Output GIF path
            max_colors: Optional maximum colors for quantization
            remove_duplicates: Whether to remove duplicate frames
            optimize_disposal: Whether to optimize disposal methods
        
        Returns:
            Tuple of (success, size_reduction_percent)
        """
        try:
            original_size = os.path.getsize(gif_path)
            
            # Step 1: Remove duplicates if requested
            temp_path1 = output_path
            if remove_duplicates:
                temp_path1 = output_path + '.dedup.gif'
                if not PILCompressor.remove_duplicate_frames(gif_path, temp_path1):
                    temp_path1 = gif_path  # Fallback to original
            
            # Step 2: Optimize disposal methods if requested
            temp_path2 = temp_path1
            if optimize_disposal and temp_path1 != gif_path:
                temp_path2 = output_path + '.disposal.gif'
                if not PILCompressor.optimize_frame_disposal(temp_path1, temp_path2):
                    temp_path2 = temp_path1  # Fallback
            
            # Step 3: Quantize palette if requested
            final_path = temp_path2
            if max_colors and max_colors < 256:
                if not PILCompressor.quantize_with_adaptive_palette(temp_path2, output_path, max_colors):
                    final_path = temp_path2  # Fallback
                else:
                    final_path = output_path
            else:
                # Just optimize palette entries
                if not PILCompressor.optimize_palette_entries(temp_path2, output_path):
                    final_path = temp_path2
                else:
                    final_path = output_path
            
            # Cleanup temp files
            for temp_file in [temp_path1, temp_path2]:
                if temp_file != gif_path and temp_file != output_path and os.path.exists(temp_file):
                    try:
                        os.remove(temp_file)
                    except Exception:
                        pass
            
            if final_path == output_path and os.path.exists(output_path):
                new_size = os.path.getsize(output_path)
                reduction_pct = ((original_size - new_size) / original_size * 100) if original_size > 0 else 0
                return True, reduction_pct
            
            return False, 0.0
        except Exception as e:
            logger.debug(f"PIL comprehensive compression failed: {e}")
            return False, 0.0

