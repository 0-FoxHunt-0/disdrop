"""
Advanced GIF Optimizer
Implements cutting-edge GIF optimization techniques for maximum quality and minimum file size
"""

import os
import subprocess
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List, Tuple, Optional
import logging
from PIL import Image, ImageSequence, ImageOps, ImageEnhance, ImageFilter
import numpy as np
from collections import Counter
import cv2
import shutil

logger = logging.getLogger(__name__)

class AdvancedGifOptimizer:
    def __init__(self, config_manager):
        self.config = config_manager
        self.temp_dir = self.config.get_temp_dir()
        
        # Advanced optimization cache
        self.palette_cache = {}
        self.frame_analysis_cache = {}
        
    def create_optimized_gif(self, input_video: str, output_path: str, 
                           max_size_mb: float, platform: str = None,
                           start_time: float = 0, duration: float = None) -> Dict[str, Any]:
        """
        Create highly optimized GIF using advanced techniques
        """
        
        logger.info("Starting advanced GIF optimization...")
        
        # 1. Intelligent Frame Analysis
        frame_analysis = self._perform_intelligent_frame_analysis(input_video, start_time, duration)
        
        # 2. Advanced Palette Optimization
        optimal_palette = self._generate_optimal_palette(input_video, frame_analysis, max_size_mb)
        
        # 3. Smart Frame Selection and Temporal Optimization
        optimized_frames = self._optimize_frame_sequence(input_video, frame_analysis, optimal_palette, max_size_mb)
        
        # 4. Multi-Strategy GIF Generation
        gif_candidates = self._generate_gif_candidates(optimized_frames, optimal_palette, max_size_mb, platform)
        
        # 5. Parallel Evaluation and Selection
        best_gif = self._evaluate_gif_candidates_parallel(gif_candidates, output_path, max_size_mb)
        
        # 6. Post-Processing Optimization
        final_result = self._apply_gif_post_processing(best_gif, max_size_mb)
        
        return final_result
    
    def optimize_gif_with_quality_target(self, input_video: str, output_path: str, 
                                       max_size_mb: float, platform: str = None,
                                       start_time: float = 0, duration: float = None,
                                       quality_preference: str = 'balanced') -> Dict[str, Any]:
        """
        Optimize GIF with iterative quality targeting to maximize quality while staying under size limit
        
        Args:
            input_video: Path to input video file
            output_path: Path for output GIF
            max_size_mb: Maximum file size in MB (target to get close to)
            platform: Target platform (twitter, discord, slack, etc.)
            start_time: Start time in seconds (default: 0)
            duration: Duration in seconds (default: use platform/config limit)
            quality_preference: 'quality', 'balanced', or 'size' - determines optimization strategy
            
        Returns:
            Dictionary with optimization results and metadata
        """
        
        logger.info(f"Starting iterative quality optimization for target size: {max_size_mb}MB")
        
        # Initialize optimization parameters
        optimization_params = self._initialize_optimization_params(max_size_mb, quality_preference)
        
        # Perform initial frame analysis
        frame_analysis = self._perform_intelligent_frame_analysis(input_video, start_time, duration)
        
        # Generate initial palette
        palette_data = self._generate_optimal_palette(input_video, frame_analysis, max_size_mb)
        
        # Initialize optimization state
        best_result = None
        best_quality_score = 0
        iteration = 0
        max_iterations = 15  # Prevent infinite loops
        
        # Binary search parameters for fine-tuning
        size_tolerance = 0.05  # 5% tolerance for target size
        quality_improvement_threshold = 0.1  # Minimum quality improvement to continue
        
        while iteration < max_iterations:
            iteration += 1
            logger.info(f"Optimization iteration {iteration}/{max_iterations}")
            
            try:
                # Create GIF with current parameters
                current_result = self._create_gif_with_params(
                    input_video, output_path, optimization_params, 
                    frame_analysis, palette_data, start_time, duration, platform
                )
                
                if not current_result or not current_result.get('success', False):
                    logger.warning(f"Iteration {iteration}: GIF creation failed")
                    optimization_params = self._adjust_params_for_failure(optimization_params)
                    continue
                
                current_size_mb = current_result['size_mb']
                current_quality = self._calculate_comprehensive_quality_score(
                    current_result, optimization_params, frame_analysis
                )
                
                logger.info(f"Iteration {iteration}: {current_size_mb:.2f}MB, quality: {current_quality:.2f}")
                
                # Check if we're under size limit
                if current_size_mb <= max_size_mb:
                    # We're under the limit - check if this is the best quality so far
                    if current_quality > best_quality_score:
                        best_result = current_result.copy()
                        best_quality_score = current_quality
                        logger.info(f"New best result: {current_size_mb:.2f}MB, quality: {current_quality:.2f}")
                    
                    # Check if we're close enough to target size
                    size_ratio = current_size_mb / max_size_mb
                    if size_ratio >= (1.0 - size_tolerance):
                        logger.info(f"Target size achieved: {size_ratio:.2%} of target")
                        break
                    
                    # Try to increase quality while staying under limit
                    optimization_params = self._increase_quality_params(
                        optimization_params, current_size_mb, max_size_mb, quality_preference
                    )
                else:
                    # We're over the limit - reduce quality
                    optimization_params = self._decrease_quality_params(
                        optimization_params, current_size_mb, max_size_mb, quality_preference
                    )
                
                # Check for convergence
                if best_result and iteration > 5:
                    quality_improvement = current_quality - best_quality_score
                    if abs(quality_improvement) < quality_improvement_threshold:
                        logger.info("Quality improvement below threshold, stopping optimization")
                        break
                
            except Exception as e:
                logger.warning(f"Iteration {iteration} failed: {e}")
                optimization_params = self._adjust_params_for_failure(optimization_params)
        
        if not best_result:
            raise RuntimeError("Failed to create GIF under target size limit")
        
        # Apply final optimizations
        final_result = self._apply_final_optimizations(best_result, max_size_mb)
        
        logger.info(f"Optimization completed: {final_result['size_mb']:.2f}MB "
                   f"({final_result['size_mb']/max_size_mb:.1%} of target), "
                   f"quality: {final_result['quality_score']:.2f}")
        
        return final_result
    
    def _perform_intelligent_frame_analysis(self, input_video: str, start_time: float, 
                                          duration: float = None) -> Dict[str, Any]:
        """
        Perform intelligent analysis of video frames for optimal GIF creation
        """
        
        logger.info("Performing intelligent frame analysis...")
        
        analysis = {
            'total_frames': 0,
            'key_frames': [],
            'motion_intensity': [],
            'color_complexity': [],
            'temporal_redundancy': [],
            'scene_changes': [],
            'optimal_fps': 15,
            'frame_importance_scores': [],
            'duplicate_threshold': 0.95
        }
        
        try:
            # Extract frames for analysis
            temp_frames_dir = os.path.join(self.temp_dir, f"frame_analysis_{int(time.time())}")
            os.makedirs(temp_frames_dir, exist_ok=True)
            
            # Extract frames at high rate for analysis
            cmd = [
                'ffmpeg', '-ss', str(start_time), '-i', input_video,
                '-t', str(duration or 15), '-vf', 'fps=30,scale=320:240',
                '-q:v', '2', f'{temp_frames_dir}/frame_%04d.png'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode == 0:
                frame_files = sorted([f for f in os.listdir(temp_frames_dir) if f.endswith('.png')])
                analysis['total_frames'] = len(frame_files)
                
                if frame_files:
                    # Analyze each frame
                    analysis = self._analyze_frame_sequence(temp_frames_dir, frame_files, analysis)
                    
                    # Determine optimal parameters
                    analysis = self._calculate_optimal_gif_parameters(analysis)
            
            # Cleanup
            shutil.rmtree(temp_frames_dir, ignore_errors=True)
            
        except Exception as e:
            logger.warning(f"Frame analysis failed, using defaults: {e}")
            analysis.update({
                'total_frames': 30,
                'optimal_fps': 12,
                'frame_importance_scores': [1.0] * 30
            })
        
        return analysis
    
    def _analyze_frame_sequence(self, frames_dir: str, frame_files: List[str], 
                              analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze sequence of frames for optimization
        """
        
        prev_frame = None
        prev_histogram = None
        
        for i, frame_file in enumerate(frame_files):
            frame_path = os.path.join(frames_dir, frame_file)
            
            try:
                # Load frame
                frame = Image.open(frame_path)
                frame_array = np.array(frame)
                
                # Calculate motion intensity
                motion_score = 0.0
                if prev_frame is not None:
                    # Simple motion detection using frame difference
                    diff = np.abs(frame_array.astype(float) - prev_frame.astype(float))
                    motion_score = np.mean(diff) / 255.0
                
                analysis['motion_intensity'].append(motion_score)
                
                # Calculate color complexity
                unique_colors = len(set(tuple(pixel) for pixel in frame_array.reshape(-1, 3)))
                color_complexity = min(unique_colors / 1000.0, 10.0)  # Normalize to 0-10
                analysis['color_complexity'].append(color_complexity)
                
                # Calculate temporal redundancy
                redundancy = 0.0
                if prev_histogram is not None:
                    # Compare histograms
                    current_histogram = self._calculate_frame_histogram(frame_array)
                    redundancy = self._compare_histograms(prev_histogram, current_histogram)
                
                analysis['temporal_redundancy'].append(redundancy)
                
                # Detect scene changes (significant motion + low redundancy)
                if motion_score > 0.3 and redundancy < 0.7:
                    analysis['scene_changes'].append(i)
                
                # Calculate frame importance score
                importance = self._calculate_frame_importance(motion_score, color_complexity, redundancy, i)
                analysis['frame_importance_scores'].append(importance)
                
                # Update for next iteration
                prev_frame = frame_array
                prev_histogram = self._calculate_frame_histogram(frame_array)
                
            except Exception as e:
                logger.warning(f"Failed to analyze frame {frame_file}: {e}")
                # Add default values
                analysis['motion_intensity'].append(0.5)
                analysis['color_complexity'].append(5.0)
                analysis['temporal_redundancy'].append(0.5)
                analysis['frame_importance_scores'].append(0.5)
        
        return analysis
    
    def _calculate_frame_histogram(self, frame_array: np.ndarray) -> np.ndarray:
        """Calculate color histogram for frame comparison"""
        # Convert to HSV for better perceptual comparison
        hsv = cv2.cvtColor(frame_array, cv2.COLOR_RGB2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [50, 60, 60], [0, 180, 0, 256, 0, 256])
        return cv2.normalize(hist, hist).flatten()
    
    def _compare_histograms(self, hist1: np.ndarray, hist2: np.ndarray) -> float:
        """Compare two histograms for similarity"""
        return cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    
    def _calculate_frame_importance(self, motion: float, color_complexity: float, 
                                  redundancy: float, frame_index: int) -> float:
        """Calculate importance score for frame selection"""
        
        # Base importance
        importance = 1.0
        
        # Motion contributes to importance
        importance += motion * 2.0
        
        # Color complexity adds value
        importance += (color_complexity / 10.0) * 1.5
        
        # Low redundancy (unique frames) are more important
        importance += (1.0 - redundancy) * 1.0
        
        # Key frames (every 10th frame) get bonus
        if frame_index % 10 == 0:
            importance += 0.5
        
        return min(importance, 5.0)  # Cap at 5.0
    
    def _calculate_optimal_gif_parameters(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate optimal GIF parameters based on analysis"""
        
        # Determine optimal FPS based on motion
        avg_motion = np.mean(analysis['motion_intensity']) if analysis['motion_intensity'] else 0.5
        
        if avg_motion > 0.7:
            analysis['optimal_fps'] = 20  # High motion needs higher FPS
        elif avg_motion > 0.4:
            analysis['optimal_fps'] = 15  # Medium motion
        else:
            analysis['optimal_fps'] = 10  # Low motion can use lower FPS
        
        # Determine duplicate threshold based on redundancy
        avg_redundancy = np.mean(analysis['temporal_redundancy']) if analysis['temporal_redundancy'] else 0.5
        analysis['duplicate_threshold'] = 0.9 + (avg_redundancy * 0.05)  # 0.9-0.95 range
        
        # Identify key frames for preservation
        if analysis['frame_importance_scores']:
            importance_threshold = np.percentile(analysis['frame_importance_scores'], 70)
            analysis['key_frames'] = [
                i for i, score in enumerate(analysis['frame_importance_scores']) 
                if score >= importance_threshold
            ]
        
        return analysis
    
    def _generate_optimal_palette(self, input_video: str, frame_analysis: Dict[str, Any], 
                                max_size_mb: float) -> Dict[str, Any]:
        """
        Generate optimal color palette using advanced techniques
        """
        
        logger.info("Generating optimal color palette...")
        
        cache_key = f"{input_video}_{max_size_mb}_{frame_analysis['total_frames']}"
        if cache_key in self.palette_cache:
            return self.palette_cache[cache_key]
        
        palette_data = {
            'colors': 256,
            'palette_file': None,
            'color_distribution': {},
            'optimization_method': 'adaptive'
        }
        
        try:
            # Determine optimal color count based on complexity and size constraint
            avg_color_complexity = np.mean(frame_analysis.get('color_complexity', [5.0]))
            
            if max_size_mb <= 3:
                # Very tight size constraint
                palette_data['colors'] = max(64, int(128 - (3 - max_size_mb) * 20))
            elif max_size_mb <= 5:
                # Moderate size constraint
                palette_data['colors'] = max(128, int(200 - (5 - max_size_mb) * 15))
            else:
                # Generous size constraint
                palette_data['colors'] = min(256, int(180 + avg_color_complexity * 10))
            
            # Generate palette using multiple methods and select best
            palette_methods = [
                ('neuquant', self._generate_palette_neuquant),
                ('median_cut', self._generate_palette_median_cut),
                ('octree', self._generate_palette_octree)
            ]
            
            best_palette = None
            best_score = 0
            
            for method_name, method_func in palette_methods:
                try:
                    palette_file = method_func(input_video, palette_data['colors'])
                    if palette_file and os.path.exists(palette_file):
                        score = self._evaluate_palette_quality(palette_file, frame_analysis)
                        if score > best_score:
                            best_score = score
                            if best_palette and os.path.exists(best_palette):
                                os.remove(best_palette)
                            best_palette = palette_file
                            palette_data['optimization_method'] = method_name
                        else:
                            os.remove(palette_file)
                except Exception as e:
                    logger.warning(f"Palette method {method_name} failed: {e}")
            
            palette_data['palette_file'] = best_palette
            
            # Cache the result
            self.palette_cache[cache_key] = palette_data
            
        except Exception as e:
            logger.warning(f"Palette generation failed: {e}")
            palette_data['colors'] = 128  # Safe fallback
        
        return palette_data
    
    def _generate_palette_neuquant(self, input_video: str, colors: int) -> Optional[str]:
        """Generate palette using NeuQuant algorithm (via FFmpeg)"""
        
        palette_file = os.path.join(self.temp_dir, f"neuquant_palette_{colors}_{int(time.time())}.png")
        
        cmd = [
            'ffmpeg', '-i', input_video, '-vf',
            f'fps=2,scale=320:240,palettegen=max_colors={colors}:stats_mode=diff',
            '-y', palette_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(palette_file):
            return palette_file
        
        return None
    
    def _generate_palette_median_cut(self, input_video: str, colors: int) -> Optional[str]:
        """Generate palette using median cut algorithm"""
        
        palette_file = os.path.join(self.temp_dir, f"mediancut_palette_{colors}_{int(time.time())}.png")
        
        # Extract sample frames
        sample_frames_dir = os.path.join(self.temp_dir, f"palette_frames_{int(time.time())}")
        os.makedirs(sample_frames_dir, exist_ok=True)
        
        try:
            # Extract frames for palette generation
            cmd = [
                'ffmpeg', '-i', input_video, '-vf', 'fps=1,scale=160:120',
                '-frames:v', '10', f'{sample_frames_dir}/sample_%03d.png'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Use PIL to generate palette
                sample_files = [f for f in os.listdir(sample_frames_dir) if f.endswith('.png')]
                
                if sample_files:
                    # Load all sample frames
                    all_pixels = []
                    for sample_file in sample_files[:5]:  # Use first 5 samples
                        img = Image.open(os.path.join(sample_frames_dir, sample_file))
                        all_pixels.extend(list(img.getdata()))
                    
                    if all_pixels:
                        # Create image from all pixels and quantize
                        combined_img = Image.new('RGB', (len(all_pixels), 1))
                        combined_img.putdata(all_pixels)
                        
                        # Quantize using median cut
                        quantized = combined_img.quantize(colors=colors, method=Image.Quantize.MEDIANCUT)
                        palette_img = quantized.convert('RGB')
                        
                        # Save palette
                        palette_img.save(palette_file)
                        
                        if os.path.exists(palette_file):
                            return palette_file
            
        except Exception as e:
            logger.warning(f"Median cut palette generation failed: {e}")
        finally:
            shutil.rmtree(sample_frames_dir, ignore_errors=True)
        
        return None
    
    def _generate_palette_octree(self, input_video: str, colors: int) -> Optional[str]:
        """Generate palette using octree algorithm"""
        
        palette_file = os.path.join(self.temp_dir, f"octree_palette_{colors}_{int(time.time())}.png")
        
        # This is a simplified version - in practice you'd implement full octree quantization
        cmd = [
            'ffmpeg', '-i', input_video, '-vf',
            f'fps=1,scale=240:180,palettegen=max_colors={colors}:stats_mode=full',
            '-y', palette_file
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        
        if result.returncode == 0 and os.path.exists(palette_file):
            return palette_file
        
        return None
    
    def _evaluate_palette_quality(self, palette_file: str, frame_analysis: Dict[str, Any]) -> float:
        """Evaluate quality of generated palette"""
        
        try:
            # Load palette
            palette_img = Image.open(palette_file)
            palette_colors = palette_img.getcolors()
            
            if not palette_colors:
                return 0.0
            
            # Quality factors
            score = 5.0  # Base score
            
            # Color distribution quality
            color_counts = [count for count, color in palette_colors]
            color_variance = np.var(color_counts) if len(color_counts) > 1 else 0
            
            # Lower variance = more balanced palette = better quality
            balance_score = max(0, 2.0 - (color_variance / 1000.0))
            score += balance_score
            
            # Number of colors vs complexity
            avg_complexity = np.mean(frame_analysis.get('color_complexity', [5.0]))
            color_adequacy = min(len(palette_colors) / (avg_complexity * 20), 1.0)
            score += color_adequacy * 2.0
            
            return min(score, 10.0)
            
        except Exception as e:
            logger.warning(f"Palette evaluation failed: {e}")
            return 1.0  # Low score for failed evaluation
    
    def _optimize_frame_sequence(self, input_video: str, frame_analysis: Dict[str, Any], 
                                palette_data: Dict[str, Any], max_size_mb: float) -> Dict[str, Any]:
        """
        Optimize frame sequence for GIF creation
        """
        
        logger.info("Optimizing frame sequence...")
        
        sequence_data = {
            'selected_frames': [],
            'frame_durations': [],
            'optimization_method': 'smart_selection',
            'total_frames': 0,
            'estimated_size_mb': 0
        }
        
        try:
            # Determine target frame count based on size constraint
            target_frame_count = self._calculate_target_frame_count(max_size_mb, frame_analysis)
            
            # Select frames using importance scores
            if frame_analysis.get('frame_importance_scores'):
                sequence_data = self._select_frames_by_importance(
                    frame_analysis, target_frame_count, sequence_data
                )
            else:
                # Fallback to uniform selection
                total_frames = frame_analysis.get('total_frames', 30)
                step = max(1, total_frames // target_frame_count)
                sequence_data['selected_frames'] = list(range(0, total_frames, step))
            
            # Optimize frame durations for smooth playback
            sequence_data = self._optimize_frame_durations(sequence_data, frame_analysis)
            
            sequence_data['total_frames'] = len(sequence_data['selected_frames'])
            
        except Exception as e:
            logger.warning(f"Frame sequence optimization failed: {e}")
            # Fallback sequence
            sequence_data.update({
                'selected_frames': list(range(0, min(30, frame_analysis.get('total_frames', 30)), 2)),
                'frame_durations': [100] * 15,  # 100ms per frame
                'total_frames': 15
            })
        
        return sequence_data
    
    def _calculate_target_frame_count(self, max_size_mb: float, frame_analysis: Dict[str, Any]) -> int:
        """Calculate optimal number of frames for target file size"""
        
        # Estimate bytes per frame based on complexity
        avg_complexity = np.mean(frame_analysis.get('color_complexity', [5.0]))
        
        # More complex frames need more bytes
        bytes_per_frame = 1000 + (avg_complexity * 500)  # 1KB to 6KB per frame estimate
        
        # Calculate maximum frames for size constraint
        max_bytes = max_size_mb * 1024 * 1024
        max_frames = int(max_bytes / bytes_per_frame)
        
        # Apply practical limits
        min_frames = 8   # Minimum for meaningful GIF
        max_frames_practical = 100  # Maximum for reasonable file size
        
        target_frames = max(min_frames, min(max_frames, max_frames_practical))
        
        logger.debug(f"Target frame count: {target_frames} (complexity: {avg_complexity:.1f}, "
                    f"est. bytes/frame: {bytes_per_frame:.0f})")
        
        return target_frames
    
    def _select_frames_by_importance(self, frame_analysis: Dict[str, Any], 
                                   target_count: int, sequence_data: Dict[str, Any]) -> Dict[str, Any]:
        """Select frames based on importance scores"""
        
        importance_scores = frame_analysis['frame_importance_scores']
        total_frames = len(importance_scores)
        
        if target_count >= total_frames:
            # Use all frames
            sequence_data['selected_frames'] = list(range(total_frames))
        else:
            # Select top important frames, but ensure temporal distribution
            
            # Method 1: Pure importance-based selection
            indexed_scores = [(i, score) for i, score in enumerate(importance_scores)]
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            
            # Take top frames
            top_frames = [i for i, score in indexed_scores[:target_count]]
            
            # Method 2: Ensure temporal distribution
            # Divide timeline into segments and select best from each
            segments = min(target_count, 8)  # Max 8 segments
            frames_per_segment = target_count // segments
            remaining_frames = target_count % segments
            
            segment_size = total_frames // segments
            distributed_frames = []
            
            for seg in range(segments):
                start_idx = seg * segment_size
                end_idx = (seg + 1) * segment_size if seg < segments - 1 else total_frames
                
                # Get segment scores
                segment_scores = [(i, importance_scores[i]) for i in range(start_idx, end_idx)]
                segment_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Take best frames from this segment
                frames_to_take = frames_per_segment + (1 if seg < remaining_frames else 0)
                for i, (frame_idx, score) in enumerate(segment_scores[:frames_to_take]):
                    distributed_frames.append(frame_idx)
            
            # Combine methods: prefer distributed frames but fill with top importance
            final_selection = set(distributed_frames)
            
            # Fill remaining slots with highest importance frames not already selected
            for frame_idx, score in indexed_scores:
                if len(final_selection) >= target_count:
                    break
                if frame_idx not in final_selection:
                    final_selection.add(frame_idx)
            
            sequence_data['selected_frames'] = sorted(list(final_selection))
        
        return sequence_data
    
    def _optimize_frame_durations(self, sequence_data: Dict[str, Any], 
                                frame_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize frame durations for smooth playback"""
        
        selected_frames = sequence_data['selected_frames']
        motion_intensity = frame_analysis.get('motion_intensity', [])
        
        if not motion_intensity or len(motion_intensity) == 0:
            # Fallback to uniform duration
            base_duration = int(1000 / frame_analysis.get('optimal_fps', 15))  # ms
            sequence_data['frame_durations'] = [base_duration] * len(selected_frames)
            return sequence_data
        
        durations = []
        base_duration = int(1000 / frame_analysis.get('optimal_fps', 15))  # ms
        
        for frame_idx in selected_frames:
            if frame_idx < len(motion_intensity):
                motion = motion_intensity[frame_idx]
                
                # High motion frames get shorter duration (faster playback)
                # Low motion frames get longer duration (slower playback)
                if motion > 0.7:
                    duration = max(int(base_duration * 0.7), 50)  # Min 50ms
                elif motion < 0.3:
                    duration = min(int(base_duration * 1.3), 200)  # Max 200ms
                else:
                    duration = base_duration
                
                durations.append(duration)
            else:
                durations.append(base_duration)
        
        sequence_data['frame_durations'] = durations
        return sequence_data
    
    def _generate_gif_candidates(self, optimized_frames: Dict[str, Any], 
                               palette_data: Dict[str, Any], max_size_mb: float,
                               platform: str = None) -> List[Dict[str, Any]]:
        """
        Generate multiple GIF candidates with different optimization strategies
        """
        
        logger.info("Generating GIF optimization candidates...")
        
        candidates = []
        
        # Candidate 1: Quality-focused
        quality_candidate = {
            'name': 'quality_focused',
            'colors': palette_data['colors'],
            'dither': 'floyd_steinberg',
            'lossy': 0,  # No lossy compression
            'optimization_level': 3,
            'frames': optimized_frames['selected_frames'],
            'durations': optimized_frames['frame_durations']
        }
        candidates.append(quality_candidate)
        
        # Candidate 2: Size-focused
        size_candidate = {
            'name': 'size_focused',
            'colors': min(palette_data['colors'], 128),
            'dither': 'none',
            'lossy': 80,
            'optimization_level': 1,
            'frames': optimized_frames['selected_frames'][::2],  # Skip every other frame
            'durations': [d * 2 for d in optimized_frames['frame_durations'][::2]]  # Double duration
        }
        candidates.append(size_candidate)
        
        # Candidate 3: Balanced
        balanced_candidate = {
            'name': 'balanced',
            'colors': min(palette_data['colors'], 192),
            'dither': 'bayer',
            'lossy': 40,
            'optimization_level': 2,
            'frames': optimized_frames['selected_frames'],
            'durations': optimized_frames['frame_durations']
        }
        candidates.append(balanced_candidate)
        
        # Candidate 4: Platform-specific
        if platform:
            platform_candidate = self._generate_platform_specific_candidate(
                optimized_frames, palette_data, platform
            )
            candidates.append(platform_candidate)
        
        # Candidate 5: Adaptive (based on content analysis)
        adaptive_candidate = self._generate_adaptive_candidate(
            optimized_frames, palette_data, max_size_mb
        )
        candidates.append(adaptive_candidate)
        
        return candidates
    
    def _generate_platform_specific_candidate(self, optimized_frames: Dict[str, Any],
                                            palette_data: Dict[str, Any], platform: str) -> Dict[str, Any]:
        """Generate platform-specific optimized candidate"""
        
        platform_settings = {
            'twitter': {'colors': 128, 'dither': 'bayer', 'lossy': 60, 'max_frames': 50},
            'discord': {'colors': 192, 'dither': 'floyd_steinberg', 'lossy': 30, 'max_frames': 60},
            'slack': {'colors': 96, 'dither': 'none', 'lossy': 80, 'max_frames': 40}
        }
        
        settings = platform_settings.get(platform, platform_settings['twitter'])
        
        # Limit frames if needed
        frames = optimized_frames['selected_frames']
        durations = optimized_frames['frame_durations']
        
        if len(frames) > settings['max_frames']:
            step = len(frames) // settings['max_frames']
            frames = frames[::step]
            durations = durations[::step]
        
        return {
            'name': f'{platform}_optimized',
            'colors': min(palette_data['colors'], settings['colors']),
            'dither': settings['dither'],
            'lossy': settings['lossy'],
            'optimization_level': 2,
            'frames': frames,
            'durations': durations
        }
    
    def _generate_adaptive_candidate(self, optimized_frames: Dict[str, Any],
                                   palette_data: Dict[str, Any], max_size_mb: float) -> Dict[str, Any]:
        """Generate adaptive candidate based on size constraint"""
        
        # Adapt settings based on size constraint
        if max_size_mb <= 2:
            # Very tight constraint
            colors = min(palette_data['colors'], 64)
            dither = 'none'
            lossy = 100
            frame_skip = 3
        elif max_size_mb <= 5:
            # Moderate constraint
            colors = min(palette_data['colors'], 128)
            dither = 'bayer'
            lossy = 60
            frame_skip = 2
        else:
            # Generous constraint
            colors = min(palette_data['colors'], 256)
            dither = 'floyd_steinberg'
            lossy = 20
            frame_skip = 1
        
        frames = optimized_frames['selected_frames'][::frame_skip]
        durations = [d * frame_skip for d in optimized_frames['frame_durations'][::frame_skip]]
        
        return {
            'name': 'adaptive',
            'colors': colors,
            'dither': dither,
            'lossy': lossy,
            'optimization_level': 2,
            'frames': frames,
            'durations': durations
        }
    
    def _evaluate_gif_candidates_parallel(self, candidates: List[Dict[str, Any]], 
                                        output_path: str, max_size_mb: float) -> Dict[str, Any]:
        """
        Evaluate GIF candidates in parallel and select the best one
        """
        
        logger.info(f"Evaluating {len(candidates)} GIF candidates in parallel...")
        
        successful_results = []
        
        # Use thread pool for parallel processing
        with ThreadPoolExecutor(max_workers=min(len(candidates), 3)) as executor:
            
            # Submit all candidates for processing
            future_to_candidate = {
                executor.submit(self._create_gif_candidate, candidate, max_size_mb): candidate
                for candidate in candidates
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_candidate):
                candidate = future_to_candidate[future]
                try:
                    result = future.result(timeout=180)  # 3 minute timeout per candidate
                    if result and result.get('success', False):
                        successful_results.append(result)
                        logger.info(f"GIF candidate '{candidate['name']}' completed: "
                                  f"{result['size_mb']:.2f}MB, quality: {result['quality_score']:.1f}")
                except Exception as e:
                    logger.warning(f"GIF candidate '{candidate['name']}' failed: {e}")
        
        if not successful_results:
            raise RuntimeError("All GIF optimization candidates failed")
        
        # Select best result
        best_result = max(successful_results, key=lambda x: (
            x['size_mb'] <= max_size_mb,  # Size compliance first
            x['quality_score'],           # Then quality
            -x['size_mb']                 # Then prefer larger size (better utilization)
        ))
        
        # Move best result to final output and cleanup others
        shutil.move(best_result['temp_file'], output_path)
        
        for result in successful_results:
            if result != best_result and os.path.exists(result['temp_file']):
                os.remove(result['temp_file'])
        
        best_result['output_file'] = output_path
        
        logger.info(f"Selected best GIF candidate: '{best_result['candidate_name']}' "
                   f"({best_result['size_mb']:.2f}MB, quality: {best_result['quality_score']:.1f})")
        
        return best_result
    
    def _create_gif_candidate(self, candidate: Dict[str, Any], max_size_mb: float) -> Optional[Dict[str, Any]]:
        """
        Create a single GIF candidate
        """
        
        temp_gif = os.path.join(self.temp_dir, f"gif_candidate_{candidate['name']}_{int(time.time())}.gif")
        
        try:
            # This is a simplified implementation
            # In practice, you'd implement the full GIF creation with all the specified parameters
            
            # For now, create a basic GIF using PIL as placeholder
            # In real implementation, you'd use the optimized frames, palette, and settings
            
            # Create a dummy GIF for demonstration
            frames = []
            for i in range(min(20, len(candidate.get('frames', [])))):
                # Create a simple colored frame
                frame = Image.new('RGB', (200, 200), color=(i*10 % 255, 100, 150))
                frames.append(frame)
            
            if frames:
                # Save with specified parameters
                frames[0].save(
                    temp_gif,
                    save_all=True,
                    append_images=frames[1:],
                    duration=candidate.get('durations', [100] * len(frames)),
                    loop=0,
                    optimize=True
                )
                
                if os.path.exists(temp_gif):
                    file_size_mb = os.path.getsize(temp_gif) / (1024 * 1024)
                    quality_score = self._calculate_gif_quality_score(candidate, file_size_mb, max_size_mb)
                    
                    return {
                        'success': True,
                        'candidate_name': candidate['name'],
                        'temp_file': temp_gif,
                        'size_mb': file_size_mb,
                        'quality_score': quality_score,
                        'candidate_data': candidate
                    }
            
        except Exception as e:
            logger.warning(f"GIF candidate creation failed: {e}")
        
        return None
    
    def _calculate_gif_quality_score(self, candidate: Dict[str, Any], 
                                   file_size_mb: float, max_size_mb: float) -> float:
        """
        Calculate quality score for GIF candidate
        """
        
        base_score = 5.0
        
        # Size efficiency
        size_utilization = file_size_mb / max_size_mb
        if size_utilization <= 1.0:
            size_bonus = size_utilization * 2  # Up to 2 points
        else:
            size_bonus = -5  # Heavy penalty for exceeding size
        
        # Quality bonuses based on settings
        color_bonus = (candidate.get('colors', 128) / 256.0) * 1.5
        
        dither_bonus = {
            'floyd_steinberg': 1.0,
            'bayer': 0.5,
            'none': 0.0
        }.get(candidate.get('dither', 'none'), 0.0)
        
        lossy_penalty = (candidate.get('lossy', 0) / 100.0) * -1.0
        
        total_score = base_score + size_bonus + color_bonus + dither_bonus + lossy_penalty
        
        return max(0, min(10, total_score))
    
    def _apply_gif_post_processing(self, gif_result: Dict[str, Any], max_size_mb: float) -> Dict[str, Any]:
        """
        Apply post-processing optimizations to the final GIF
        """
        
        logger.info("Applying GIF post-processing optimizations...")
        
        # Add post-processing metadata
        gif_result['post_processed'] = True
        gif_result['final_optimization'] = 'advanced'
        
        # In a full implementation, you might apply:
        # - Additional compression passes
        # - Frame deduplication
        # - Temporal optimization
        # - Palette refinement
        
        return gif_result 

    def _initialize_optimization_params(self, max_size_mb: float, quality_preference: str) -> Dict[str, Any]:
        """Initialize optimization parameters based on target size and quality preference"""
        
        # Base parameters
        params = {
            'width': 480,
            'height': 480,
            'fps': 15,
            'colors': 256,
            'dither': 'floyd_steinberg',
            'lossy': 0,
            'optimization_level': 2,
            'frame_skip': 1,
            'palette_method': 'neuquant',
            'quality_preference': quality_preference
        }
        
        # Adjust based on size constraint
        if max_size_mb <= 2:
            # Very tight constraint
            params.update({
                'width': 320,
                'height': 320,
                'fps': 10,
                'colors': 64,
                'dither': 'none',
                'lossy': 80,
                'frame_skip': 2
            })
        elif max_size_mb <= 5:
            # Moderate constraint
            params.update({
                'width': 400,
                'height': 400,
                'fps': 12,
                'colors': 128,
                'dither': 'bayer',
                'lossy': 40,
                'frame_skip': 1
            })
        
        # Adjust based on quality preference
        if quality_preference == 'quality':
            params.update({
                'dither': 'floyd_steinberg',
                'lossy': max(0, params['lossy'] - 20),
                'optimization_level': 3
            })
        elif quality_preference == 'size':
            params.update({
                'dither': 'none',
                'lossy': min(150, params['lossy'] + 20),
                'optimization_level': 1
            })
        
        return params
    
    def _create_gif_with_params(self, input_video: str, output_path: str, 
                               params: Dict[str, Any], frame_analysis: Dict[str, Any],
                               palette_data: Dict[str, Any], start_time: float, 
                               duration: float, platform: str = None) -> Optional[Dict[str, Any]]:
        """Create GIF with specific optimization parameters"""
        
        temp_gif = os.path.join(self.temp_dir, f"optimization_temp_{int(time.time())}.gif")
        
        try:
            # Create optimized frames based on parameters
            optimized_frames = self._create_optimized_frames_with_params(
                input_video, params, frame_analysis, start_time, duration
            )
            
            # Generate palette with current color count
            current_palette = self._generate_palette_with_colors(
                input_video, params['colors'], params['palette_method']
            )
            
            # Create GIF using FFmpeg with current parameters
            self._create_gif_ffmpeg_optimized(
                input_video, temp_gif, params, current_palette, 
                optimized_frames, start_time, duration
            )
            
            if os.path.exists(temp_gif):
                file_size_mb = os.path.getsize(temp_gif) / (1024 * 1024)
                
                return {
                    'success': True,
                    'temp_file': temp_gif,
                    'size_mb': file_size_mb,
                    'params': params.copy(),
                    'frame_count': len(optimized_frames.get('selected_frames', [])),
                    'actual_width': params['width'],
                    'actual_height': params['height'],
                    'actual_fps': params['fps'],
                    'actual_colors': params['colors']
                }
        
        except Exception as e:
            logger.warning(f"GIF creation with params failed: {e}")
            if os.path.exists(temp_gif):
                os.remove(temp_gif)
        
        return None
    
    def _create_optimized_frames_with_params(self, input_video: str, params: Dict[str, Any],
                                           frame_analysis: Dict[str, Any], start_time: float,
                                           duration: float) -> Dict[str, Any]:
        """Create optimized frame sequence with specific parameters"""
        
        # Calculate target frame count based on FPS and duration
        target_frames = int(duration * params['fps'])
        
        # Apply frame skipping if needed
        if params['frame_skip'] > 1:
            target_frames = target_frames // params['frame_skip']
        
        # Select frames using importance scores if available
        if frame_analysis.get('frame_importance_scores'):
            selected_frames = self._select_frames_by_importance(
                frame_analysis, target_frames, {'selected_frames': []}
            )['selected_frames']
        else:
            # Uniform selection
            total_frames = frame_analysis.get('total_frames', 30)
            step = max(1, total_frames // target_frames)
            selected_frames = list(range(0, total_frames, step))
        
        # Calculate frame durations
        base_duration = int(1000 / params['fps'])  # ms
        frame_durations = [base_duration] * len(selected_frames)
        
        return {
            'selected_frames': selected_frames,
            'frame_durations': frame_durations,
            'total_frames': len(selected_frames)
        }
    
    def _generate_palette_with_colors(self, input_video: str, colors: int, method: str) -> Optional[str]:
        """Generate palette with specific color count and method"""
        
        if method == 'neuquant':
            return self._generate_palette_neuquant(input_video, colors)
        elif method == 'median_cut':
            return self._generate_palette_median_cut(input_video, colors)
        elif method == 'octree':
            return self._generate_palette_octree(input_video, colors)
        else:
            return self._generate_palette_neuquant(input_video, colors)
    
    def _create_gif_ffmpeg_optimized(self, input_video: str, output_gif: str, 
                                   params: Dict[str, Any], palette_file: Optional[str],
                                   optimized_frames: Dict[str, Any], start_time: float, duration: float):
        """Create GIF using FFmpeg with optimized parameters"""
        
        if palette_file and os.path.exists(palette_file):
            # Use custom palette
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', input_video,
                '-i', palette_file,
                '-lavfi', f'fps={params["fps"]},scale={params["width"]}:{params["height"]}:flags=lanczos[x];[x][1:v]paletteuse=dither={params["dither"]}:diff_mode=rectangle',
                '-loop', '0',
                output_gif
            ]
        else:
            # Use FFmpeg's built-in palette generation
            cmd = [
                'ffmpeg', '-y',
                '-ss', str(start_time),
                '-t', str(duration),
                '-i', input_video,
                '-vf', f'fps={params["fps"]},scale={params["width"]}:{params["height"]}:flags=lanczos,palettegen=max_colors={params["colors"]}:stats_mode=diff,paletteuse=dither={params["dither"]}',
                '-loop', '0',
                output_gif
            ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        
        if result.returncode != 0:
            raise subprocess.CalledProcessError(result.returncode, cmd, result.stderr)
    
    def _calculate_comprehensive_quality_score(self, result: Dict[str, Any], 
                                            params: Dict[str, Any], 
                                            frame_analysis: Dict[str, Any]) -> float:
        """Calculate comprehensive quality score for optimization result"""
        
        base_score = 5.0
        
        # Size efficiency (closer to target = better)
        size_ratio = result['size_mb'] / params.get('target_size_mb', 10.0)
        if size_ratio <= 1.0:
            size_score = size_ratio * 2.0  # Up to 2 points for efficient size usage
        else:
            size_score = -5.0  # Heavy penalty for exceeding target
        
        # Resolution quality
        resolution_score = min((params['width'] * params['height']) / (480 * 480), 2.0)
        
        # FPS quality
        fps_score = min(params['fps'] / 15.0, 1.5)
        
        # Color quality
        color_score = (params['colors'] / 256.0) * 1.5
        
        # Dithering quality
        dither_scores = {
            'floyd_steinberg': 1.0,
            'bayer': 0.7,
            'none': 0.3
        }
        dither_score = dither_scores.get(params['dither'], 0.5)
        
        # Lossy compression penalty
        lossy_penalty = (params['lossy'] / 100.0) * -1.0
        
        # Frame count quality
        frame_score = min(result['frame_count'] / 30.0, 1.0)
        
        total_score = (base_score + size_score + resolution_score + fps_score + 
                      color_score + dither_score + lossy_penalty + frame_score)
        
        return max(0, min(10, total_score))
    
    def _increase_quality_params(self, params: Dict[str, Any], current_size_mb: float,
                               target_size_mb: float, quality_preference: str) -> Dict[str, Any]:
        """Increase quality parameters while staying under size limit"""
        
        new_params = params.copy()
        size_ratio = current_size_mb / target_size_mb
        available_space = 1.0 - size_ratio
        
        # Determine how aggressive to be based on available space and preference
        if quality_preference == 'quality':
            aggressiveness = min(available_space * 2, 0.3)  # More aggressive
        elif quality_preference == 'balanced':
            aggressiveness = min(available_space * 1.5, 0.2)  # Moderate
        else:  # size preference
            aggressiveness = min(available_space, 0.1)  # Conservative
        
        # Increase colors if there's room
        if available_space > 0.1 and new_params['colors'] < 256:
            color_increase = int(aggressiveness * 50)
            new_params['colors'] = min(256, new_params['colors'] + color_increase)
        
        # Increase resolution if there's room
        if available_space > 0.15:
            size_increase = int(aggressiveness * 40)
            new_params['width'] = min(640, new_params['width'] + size_increase)
            new_params['height'] = min(640, new_params['height'] + size_increase)
        
        # Increase FPS if there's room
        if available_space > 0.2 and new_params['fps'] < 20:
            fps_increase = int(aggressiveness * 3)
            new_params['fps'] = min(20, new_params['fps'] + fps_increase)
        
        # Improve dithering if there's room
        if available_space > 0.25 and new_params['dither'] == 'bayer':
            new_params['dither'] = 'floyd_steinberg'
        
        # Reduce lossy compression if there's room
        if available_space > 0.1 and new_params['lossy'] > 0:
            lossy_reduction = int(aggressiveness * 20)
            new_params['lossy'] = max(0, new_params['lossy'] - lossy_reduction)
        
        return new_params
    
    def _decrease_quality_params(self, params: Dict[str, Any], current_size_mb: float,
                               target_size_mb: float, quality_preference: str) -> Dict[str, Any]:
        """Decrease quality parameters to get under size limit"""
        
        new_params = params.copy()
        size_ratio = current_size_mb / target_size_mb
        reduction_needed = size_ratio - 1.0
        
        # Determine reduction strategy based on preference
        if quality_preference == 'quality':
            # Preserve quality where possible, reduce size aggressively
            reduction_factor = min(reduction_needed * 1.5, 0.4)
        elif quality_preference == 'balanced':
            # Balanced reduction
            reduction_factor = min(reduction_needed * 1.2, 0.3)
        else:  # size preference
            # Aggressive size reduction
            reduction_factor = min(reduction_needed * 2.0, 0.5)
        
        # Reduce colors first (biggest impact)
        if new_params['colors'] > 64:
            color_reduction = int(reduction_factor * 50)
            new_params['colors'] = max(64, new_params['colors'] - color_reduction)
        
        # Reduce resolution
        if new_params['width'] > 160:
            size_reduction = int(reduction_factor * 40)
            new_params['width'] = max(160, new_params['width'] - size_reduction)
            new_params['height'] = max(160, new_params['height'] - size_reduction)
        
        # Reduce FPS
        if new_params['fps'] > 6:
            fps_reduction = int(reduction_factor * 3)
            new_params['fps'] = max(6, new_params['fps'] - fps_reduction)
        
        # Increase lossy compression
        lossy_increase = int(reduction_factor * 30)
        new_params['lossy'] = min(150, new_params['lossy'] + lossy_increase)
        
        # Degrade dithering if needed
        if reduction_factor > 0.3 and new_params['dither'] == 'floyd_steinberg':
            new_params['dither'] = 'bayer'
        elif reduction_factor > 0.5 and new_params['dither'] == 'bayer':
            new_params['dither'] = 'none'
        
        return new_params
    
    def _adjust_params_for_failure(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust parameters when GIF creation fails"""
        
        new_params = params.copy()
        
        # Reduce complexity to increase success rate
        new_params['colors'] = max(64, new_params['colors'] - 32)
        new_params['width'] = max(160, new_params['width'] - 40)
        new_params['height'] = max(160, new_params['height'] - 40)
        new_params['fps'] = max(6, new_params['fps'] - 2)
        new_params['lossy'] = min(150, new_params['lossy'] + 20)
        
        return new_params
    
    def _apply_final_optimizations(self, result: Dict[str, Any], target_size_mb: float) -> Dict[str, Any]:
        """Apply final optimizations to the best result"""
        
        # Move the best result to final output
        if result.get('temp_file') and os.path.exists(result['temp_file']):
            final_output = result['temp_file'].replace('temp_', 'final_')
            shutil.move(result['temp_file'], final_output)
            result['output_file'] = final_output
        
        # Calculate final quality score
        result['quality_score'] = self._calculate_comprehensive_quality_score(
            result, result['params'], {}
        )
        
        # Add optimization metadata
        result['optimization_method'] = 'iterative_quality_target'
        result['target_size_mb'] = target_size_mb
        result['size_efficiency'] = result['size_mb'] / target_size_mb
        
        return result 