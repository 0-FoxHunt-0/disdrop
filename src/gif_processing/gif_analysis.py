"""
GIF Analysis Tools
Provides content analysis for adaptive optimization (motion, similarity, complexity)
"""

import os
import subprocess
import shutil
import time
from typing import Dict, Any, List, Optional
import logging
import numpy as np
from PIL import Image
import cv2

from .gif_utils import temp_dir_context, get_gif_info

logger = logging.getLogger(__name__)


def _calculate_frame_histogram(frame_array: np.ndarray) -> np.ndarray:
    """Calculate color histogram for frame comparison"""
    # Convert to HSV for better perceptual comparison
    hsv = cv2.cvtColor(frame_array, cv2.COLOR_RGB2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
    return cv2.normalize(hist, hist).flatten()


def _compare_histograms(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """Compare two histograms for similarity"""
    return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)


def analyze_motion(gif_path: str, original_fps: float, temp_dir: str) -> Dict[str, Any]:
    """
    Analyze motion content to determine optimal FPS values.
    
    Args:
        gif_path: Path to GIF file
        original_fps: Original FPS of the GIF
        temp_dir: Temporary directory for frame extraction
    
    Returns:
        Dict with 'optimal_fps', 'fps_range', 'motion_level' (low/medium/high)
    """
    try:
        # Quick motion analysis using ffmpeg
        info = get_gif_info(gif_path)
        duration = info.get('duration', 0)
        
        if duration <= 0:
            # Fallback: use default analysis
            return {
                'optimal_fps': original_fps,
                'fps_range': [original_fps, original_fps * 0.9, original_fps * 0.8, original_fps * 0.7, original_fps * 0.6],
                'motion_level': 'medium',
                'avg_motion': 0.5
            }
        
        # Extract frames at intervals for motion analysis
        with temp_dir_context("motion_analysis", temp_dir) as temp_frames_dir:
            try:
                # Extract sample frames (every 0.5 seconds)
                sample_cmd = [
                    'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', gif_path,
                    '-vf', 'fps=2',  # 2 fps = every 0.5 seconds
                    '-frames:v', '20',  # Limit to 20 frames max
                    os.path.join(temp_frames_dir, 'frame_%03d.png')
                ]
                result = subprocess.run(sample_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                if result.returncode != 0:
                    raise ValueError(f"FFmpeg frame extraction failed: {result.stderr.decode('utf-8', errors='replace')[:200]}")
                
                # Analyze extracted frames
                frame_files = sorted([f for f in os.listdir(temp_frames_dir) if f.endswith('.png')])
                if len(frame_files) < 2:
                    raise ValueError("Not enough frames extracted")
                
                motion_scores = []
                prev_frame = None
                
                for frame_file in frame_files:
                    frame_path = os.path.join(temp_frames_dir, frame_file)
                    try:
                        frame = Image.open(frame_path)
                        frame_array = np.array(frame.convert('RGB'))
                        
                        if prev_frame is not None:
                            diff = np.abs(frame_array.astype(float) - prev_frame.astype(float))
                            motion_score = np.mean(diff) / 255.0
                            motion_scores.append(motion_score)
                        
                        prev_frame = frame_array
                    except Exception as e:
                        logger.debug(f"Failed to analyze frame {frame_file}: {e}")
                        continue
                
                if not motion_scores:
                    raise ValueError("No motion scores calculated")
                
                avg_motion = np.mean(motion_scores)
                max_motion = np.max(motion_scores)
                motion_variance = np.var(motion_scores)
            
            except Exception as e:
                logger.debug(f"Motion analysis frame extraction failed: {e}")
                raise
        
        # Determine motion level and optimal FPS
        if avg_motion > 0.15 or max_motion > 0.3:
            motion_level = 'high'
            optimal_fps_mult = 1.0  # Keep original FPS for high motion
        elif avg_motion > 0.05 or motion_variance > 0.01:
            motion_level = 'medium'
            optimal_fps_mult = 0.9  # Slight reduction for medium motion
        else:
            motion_level = 'low'
            optimal_fps_mult = 0.7  # More reduction for low motion
        
        optimal_fps = max(6, min(20, original_fps * optimal_fps_mult))
        
        # Generate FPS range based on motion level
        if motion_level == 'high':
            fps_range = [original_fps, original_fps * 0.95, original_fps * 0.9, original_fps * 0.85, original_fps * 0.8]
        elif motion_level == 'medium':
            fps_range = [original_fps * 0.9, original_fps * 0.8, original_fps * 0.7, original_fps * 0.6, original_fps * 0.5]
        else:  # low
            fps_range = [original_fps * 0.7, original_fps * 0.6, original_fps * 0.5, original_fps * 0.4, original_fps * 0.3]
        
        # Ensure all FPS values are within valid range
        fps_range = [max(6, min(20, fps)) for fps in fps_range]
        
        return {
            'optimal_fps': optimal_fps,
            'fps_range': fps_range,
            'motion_level': motion_level,
            'avg_motion': float(avg_motion),
            'max_motion': float(max_motion),
            'motion_variance': float(motion_variance)
        }
        
    except Exception as e:
        logger.debug(f"Motion analysis failed, using defaults: {e}")
        # Fallback to conservative defaults
        return {
            'optimal_fps': original_fps * 0.9,
            'fps_range': [original_fps, original_fps * 0.9, original_fps * 0.8, original_fps * 0.7, original_fps * 0.6],
            'motion_level': 'medium',
            'avg_motion': 0.5
        }


def analyze_frame_similarity(gif_path: str, temp_dir: str) -> Dict[str, Any]:
    """
    Analyze frame similarity to determine optimal mpdecimate parameters.
    
    Args:
        gif_path: Path to GIF file
        temp_dir: Temporary directory for frame extraction
    
    Returns:
        Dict with 'optimal_frac', 'frac_range', 'similarity_level' (low/medium/high)
    """
    try:
        # Extract frames at intervals for similarity analysis
        with temp_dir_context("similarity_analysis", temp_dir) as temp_frames_dir:
            try:
                # Extract sample frames (every 0.3 seconds for better analysis)
                sample_cmd = [
                    'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', gif_path,
                    '-vf', 'fps=3.33',  # ~3.33 fps = every 0.3 seconds
                    '-frames:v', '30',  # Limit to 30 frames max
                    os.path.join(temp_frames_dir, 'frame_%03d.png')
                ]
                result = subprocess.run(sample_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                if result.returncode != 0:
                    raise ValueError(f"FFmpeg frame extraction failed: {result.stderr.decode('utf-8', errors='replace')[:200]}")
                
                # Analyze extracted frames
                frame_files = sorted([f for f in os.listdir(temp_frames_dir) if f.endswith('.png')])
                if len(frame_files) < 3:
                    raise ValueError("Not enough frames extracted")
                
                similarity_scores = []
                prev_frame = None
                
                for frame_file in frame_files:
                    frame_path = os.path.join(temp_frames_dir, frame_file)
                    try:
                        frame = Image.open(frame_path)
                        frame_array = np.array(frame.convert('RGB'))
                        
                        if prev_frame is not None:
                            # Calculate similarity using histogram comparison
                            hist1 = _calculate_frame_histogram(prev_frame)
                            hist2 = _calculate_frame_histogram(frame_array)
                            similarity = _compare_histograms(hist1, hist2)
                            similarity_scores.append(similarity)
                        
                        prev_frame = frame_array
                    except Exception as e:
                        logger.debug(f"Failed to analyze frame {frame_file}: {e}")
                        continue
                
                if not similarity_scores:
                    raise ValueError("No similarity scores calculated")
                
                avg_similarity = np.mean(similarity_scores)
                min_similarity = np.min(similarity_scores)
                similarity_variance = np.var(similarity_scores)
            
            except Exception as e:
                logger.debug(f"Frame similarity analysis frame extraction failed: {e}")
                raise
        
        # Determine similarity level and optimal frac
        # High similarity (static scenes) = more aggressive duplicate removal (lower frac)
        # Low similarity (dynamic scenes) = conservative duplicate removal (higher frac)
        if avg_similarity > 0.9 or min_similarity > 0.85:
            similarity_level = 'high'  # Very similar frames (static)
            optimal_frac = 0.1  # Aggressive duplicate removal
            frac_range = [0.1, 0.15, 0.2, 0.25, 0.3]
        elif avg_similarity > 0.7 or similarity_variance < 0.01:
            similarity_level = 'medium'  # Moderately similar
            optimal_frac = 0.2  # Moderate duplicate removal
            frac_range = [0.2, 0.25, 0.3, 0.35, 0.4]
        else:
            similarity_level = 'low'  # Very different frames (dynamic)
            optimal_frac = 0.3  # Conservative duplicate removal
            frac_range = [0.3, 0.35, 0.4, 0.45, 0.5]
        
        return {
            'optimal_frac': optimal_frac,
            'frac_range': frac_range,
            'similarity_level': similarity_level,
            'avg_similarity': float(avg_similarity),
            'min_similarity': float(min_similarity),
            'similarity_variance': float(similarity_variance)
        }
        
    except Exception as e:
        logger.debug(f"Frame similarity analysis failed, using defaults: {e}")
        # Fallback to conservative defaults
        return {
            'optimal_frac': 0.3,
            'frac_range': [0.1, 0.2, 0.3, 0.4, 0.5],
            'similarity_level': 'medium',
            'avg_similarity': 0.7
        }


def analyze_scene_complexity(gif_path: str, temp_dir: str) -> Dict[str, Any]:
    """
    Analyze scene complexity to determine optimal palette and dithering settings.
    
    Args:
        gif_path: Path to GIF file
        temp_dir: Temporary directory for frame extraction
    
    Returns:
        Dict with 'stats_mode' ('full' or 'diff'), 'dither' strategy, and 'complexity_level'
    """
    try:
        # Extract frames at intervals for complexity analysis
        with temp_dir_context("complexity_analysis", temp_dir) as temp_frames_dir:
            try:
                # Extract sample frames (every 0.4 seconds for complexity analysis)
                sample_cmd = [
                    'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', gif_path,
                    '-vf', 'fps=2.5',  # ~2.5 fps = every 0.4 seconds
                    '-frames:v', '25',  # Limit to 25 frames max
                    os.path.join(temp_frames_dir, 'frame_%03d.png')
                ]
                result = subprocess.run(sample_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                if result.returncode != 0:
                    raise ValueError(f"FFmpeg frame extraction failed: {result.stderr.decode('utf-8', errors='replace')[:200]}")
                
                # Analyze extracted frames
                frame_files = sorted([f for f in os.listdir(temp_frames_dir) if f.endswith('.png')])
                if len(frame_files) < 3:
                    raise ValueError("Not enough frames extracted")
                
                color_complexities = []
                gradient_scores = []
                prev_frame = None
                
                for frame_file in frame_files:
                    frame_path = os.path.join(temp_frames_dir, frame_file)
                    try:
                        frame = Image.open(frame_path)
                        frame_array = np.array(frame.convert('RGB'))
                        
                        # Calculate color complexity
                        unique_colors = len(set(tuple(pixel) for pixel in frame_array.reshape(-1, 3)))
                        color_complexity = min(unique_colors / 1000.0, 10.0)
                        color_complexities.append(color_complexity)
                        
                        # Calculate gradient score (indicates smooth color transitions)
                        if prev_frame is not None:
                            # Calculate spatial gradients to detect smooth vs flat areas
                            gray = np.mean(frame_array, axis=2).astype(np.uint8)
                            grad_x = np.abs(np.diff(gray, axis=1))
                            grad_y = np.abs(np.diff(gray, axis=0))
                            gradient_score = (np.mean(grad_x) + np.mean(grad_y)) / 255.0
                            gradient_scores.append(gradient_score)
                        
                        prev_frame = frame_array
                    except Exception as e:
                        logger.debug(f"Failed to analyze frame {frame_file}: {e}")
                        continue
                
                if not color_complexities:
                    raise ValueError("No complexity scores calculated")
                
                avg_color_complexity = np.mean(color_complexities)
                max_color_complexity = np.max(color_complexities)
                avg_gradient = np.mean(gradient_scores) if gradient_scores else 0.5
            
            except Exception as e:
                logger.debug(f"Scene complexity analysis frame extraction failed: {e}")
                raise
        
        # Determine complexity level and optimal settings
        # High complexity (many colors, gradients) = use 'full' stats_mode and floyd_steinberg
        # Low complexity (few colors, flat areas) = use 'diff' stats_mode and bayer
        if avg_color_complexity > 7.0 or max_color_complexity > 8.5 or avg_gradient > 0.3:
            complexity_level = 'high'  # Complex scenes with many colors/gradients
            stats_mode = 'full'  # Full stats for better color accuracy
            dither = 'floyd_steinberg'  # Better for gradients
        elif avg_color_complexity > 4.0 or avg_gradient > 0.15:
            complexity_level = 'medium'  # Moderate complexity
            stats_mode = 'diff'  # Diff stats for efficiency
            dither = 'floyd_steinberg'  # Still use floyd for moderate gradients
        else:
            complexity_level = 'low'  # Simple scenes with few colors
            stats_mode = 'diff'  # Diff stats for efficiency
            dither = 'bayer'  # Bayer is fine for flat colors
        
        return {
            'stats_mode': stats_mode,
            'dither': dither,
            'complexity_level': complexity_level,
            'avg_color_complexity': float(avg_color_complexity),
            'max_color_complexity': float(max_color_complexity),
            'avg_gradient': float(avg_gradient)
        }
        
    except Exception as e:
        logger.debug(f"Scene complexity analysis failed, using defaults: {e}")
        # Fallback to conservative defaults
        return {
            'stats_mode': 'diff',
            'dither': 'sierra2_4a',
            'complexity_level': 'medium',
            'avg_color_complexity': 5.0
        }


def analyze_motion_segments(gif_path: str, original_fps: float, temp_dir: str) -> List[Dict[str, Any]]:
    """
    Analyze motion across GIF duration to identify segments with different motion levels.
    Returns segments with recommended FPS for each.
    
    Args:
        gif_path: Path to GIF file
        original_fps: Original FPS of the GIF
        temp_dir: Temporary directory for frame extraction
    
    Returns:
        List of dicts with 'start_time', 'end_time', 'motion_level', 'recommended_fps'
    """
    try:
        info = get_gif_info(gif_path)
        duration = info.get('duration', 0)
        
        if duration <= 0 or duration < 2.0:
            # Too short for segmentation, return single segment
            return [{
                'start_time': 0.0,
                'end_time': duration if duration > 0 else 10.0,
                'motion_level': 'medium',
                'recommended_fps': original_fps * 0.9
            }]
        
        # Extract frames at intervals for motion analysis
        with temp_dir_context("motion_segments", temp_dir) as temp_frames_dir:
            try:
                # Extract frames more frequently for better segmentation (every 0.2 seconds)
                sample_cmd = [
                    'ffmpeg', '-hide_banner', '-loglevel', 'error', '-y',
                    '-i', gif_path,
                    '-vf', 'fps=5',  # 5 fps = every 0.2 seconds
                    '-frames:v', '50',  # Limit to 50 frames max
                    os.path.join(temp_frames_dir, 'frame_%03d.png')
                ]
                result = subprocess.run(sample_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30)
                if result.returncode != 0:
                    raise ValueError(f"FFmpeg frame extraction failed: {result.stderr.decode('utf-8', errors='replace')[:200]}")
                
                # Analyze extracted frames
                frame_files = sorted([f for f in os.listdir(temp_frames_dir) if f.endswith('.png')])
                if len(frame_files) < 3:
                    raise ValueError("Not enough frames extracted")
                
                motion_scores = []
                prev_frame = None
                
                for frame_file in frame_files:
                    frame_path = os.path.join(temp_frames_dir, frame_file)
                    try:
                        frame = Image.open(frame_path)
                        frame_array = np.array(frame.convert('RGB'))
                        
                        if prev_frame is not None:
                            diff = np.abs(frame_array.astype(float) - prev_frame.astype(float))
                            motion_score = np.mean(diff) / 255.0
                            motion_scores.append(motion_score)
                        
                        prev_frame = frame_array
                    except Exception as e:
                        logger.debug(f"Failed to analyze frame {frame_file}: {e}")
                        continue
                
                if not motion_scores:
                    raise ValueError("No motion scores calculated")
            
            except Exception as e:
                logger.debug(f"Motion segment analysis frame extraction failed: {e}")
                raise
        
        # Group motion scores into segments
        # Use a simple approach: group consecutive frames with similar motion
        # Each motion score represents approximately 0.2 seconds (5fps extraction)
        segments = []
        sample_interval = 0.2  # 5fps = 0.2 seconds per sample
        
        current_segment_start = 0.0
        current_segment_motion = []
        segment_threshold = 0.1  # Group frames within 0.1 motion difference
        
        for i, motion in enumerate(motion_scores):
            if not current_segment_motion:
                current_segment_motion.append(motion)
            else:
                avg_motion = np.mean(current_segment_motion)
                # If motion is significantly different, start new segment
                if abs(motion - avg_motion) > segment_threshold and len(current_segment_motion) >= 2:
                    # Finalize current segment
                    seg_avg_motion = np.mean(current_segment_motion)
                    seg_end_time = current_segment_start + (len(current_segment_motion) * sample_interval)
                    
                    # Determine motion level and FPS
                    if seg_avg_motion > 0.15:
                        motion_level = 'high'
                        recommended_fps = original_fps * 1.0  # Keep full FPS
                    elif seg_avg_motion > 0.05:
                        motion_level = 'medium'
                        recommended_fps = original_fps * 0.85  # Slight reduction
                    else:
                        motion_level = 'low'
                        recommended_fps = original_fps * 0.6  # More reduction
                    
                    segments.append({
                        'start_time': current_segment_start,
                        'end_time': seg_end_time,
                        'motion_level': motion_level,
                        'recommended_fps': max(6, min(20, recommended_fps)),
                        'avg_motion': float(seg_avg_motion)
                    })
                    
                    # Start new segment
                    current_segment_start = seg_end_time
                    current_segment_motion = [motion]
                else:
                    current_segment_motion.append(motion)
        
        # Add final segment
        if current_segment_motion:
            seg_avg_motion = np.mean(current_segment_motion)
            seg_end_time = duration
            
            if seg_avg_motion > 0.15:
                motion_level = 'high'
                recommended_fps = original_fps * 1.0
            elif seg_avg_motion > 0.05:
                motion_level = 'medium'
                recommended_fps = original_fps * 0.85
            else:
                motion_level = 'low'
                recommended_fps = original_fps * 0.6
            
            segments.append({
                'start_time': current_segment_start,
                'end_time': seg_end_time,
                'motion_level': motion_level,
                'recommended_fps': max(6, min(20, recommended_fps)),
                'avg_motion': float(seg_avg_motion)
            })
        
        # Merge very short segments (< 0.5 seconds) with adjacent segments
        merged_segments = []
        for i, seg in enumerate(segments):
            seg_duration = seg['end_time'] - seg['start_time']
            if seg_duration < 0.5 and merged_segments:
                # Merge with previous segment
                prev_seg = merged_segments[-1]
                prev_seg['end_time'] = seg['end_time']
                # Recalculate average motion and FPS
                total_motion = (prev_seg.get('avg_motion', 0) + seg['avg_motion']) / 2
                if total_motion > 0.15:
                    prev_seg['motion_level'] = 'high'
                    prev_seg['recommended_fps'] = original_fps * 1.0
                elif total_motion > 0.05:
                    prev_seg['motion_level'] = 'medium'
                    prev_seg['recommended_fps'] = original_fps * 0.85
                else:
                    prev_seg['motion_level'] = 'low'
                    prev_seg['recommended_fps'] = original_fps * 0.6
                prev_seg['recommended_fps'] = max(6, min(20, prev_seg['recommended_fps']))
                prev_seg['avg_motion'] = total_motion
            else:
                merged_segments.append(seg)
        
        return merged_segments if merged_segments else [{
            'start_time': 0.0,
            'end_time': duration,
            'motion_level': 'medium',
            'recommended_fps': original_fps * 0.9
        }]
        
    except Exception as e:
        logger.debug(f"Motion segment analysis failed, using single segment: {e}")
        # Fallback to single segment
        return [{
            'start_time': 0.0,
            'end_time': 10.0,
            'motion_level': 'medium',
            'recommended_fps': original_fps * 0.9
        }]





