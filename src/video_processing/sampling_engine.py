"""
Sampling Engine for Efficient Video Analysis
Implements frame sampling algorithms for representative quality assessment
"""

import os
import subprocess
import json
import logging
import time
import random
import math
from typing import Dict, Any, Optional, List, Tuple, NamedTuple
from dataclasses import dataclass
from pathlib import Path
import tempfile
import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VideoSegment:
    """Represents a video segment for sampling."""
    start_time: float
    end_time: float
    duration: float
    segment_type: str  # 'temporal', 'scene', 'complexity'
    complexity_score: Optional[float] = None
    motion_score: Optional[float] = None


@dataclass
class FrameInfo:
    """Information about a specific frame."""
    timestamp: float
    frame_number: int
    complexity_score: float
    motion_score: float
    scene_id: int
    is_keyframe: bool = False


@dataclass
class SceneTransition:
    """Information about scene boundaries."""
    timestamp: float
    frame_number: int
    transition_score: float
    scene_before: int
    scene_after: int


class SamplingEngine:
    """Efficiently sample video content for fast quality assessment."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        
        # Configuration parameters
        self.temporal_interval_seconds = self._get_config('temporal_interval_seconds', 5.0)
        self.max_samples_per_scene = self._get_config('max_samples_per_scene', 3)
        self.complexity_threshold = self._get_config('complexity_threshold', 0.3)
        self.scene_change_threshold = self._get_config('scene_change_threshold', 0.4)
        self.motion_threshold = self._get_config('motion_threshold', 0.2)
        
        # Sampling limits
        self.max_total_samples = self._get_config('max_total_samples', 50)
        self.min_samples_per_minute = self._get_config('min_samples_per_minute', 2)
        self.max_analysis_duration = self._get_config('max_analysis_duration', 60.0)
        
        logger.info("Sampling Engine initialized")
    
    def _get_config(self, key: str, default: Any) -> Any:
        """Get configuration value with fallback to default."""
        if self.config:
            return self.config.get(f'sampling_engine.{key}', default)
        return default
    
    def generate_representative_samples(
        self, 
        video_path: str, 
        target_duration: float
    ) -> List[VideoSegment]:
        """Generate representative video segments for quality assessment.
        
        Args:
            video_path: Path to video file
            target_duration: Target total duration for sampling
            
        Returns:
            List of VideoSegment objects representing optimal sampling regions
        """
        logger.info(f"Generating representative samples for {os.path.basename(video_path)}")
        
        # Get video information
        video_info = self._get_video_info(video_path)
        if not video_info:
            logger.error(f"Could not get video info for {video_path}")
            return []
        
        total_duration = video_info['duration']
        fps = video_info.get('fps', 30.0)
        
        # Limit analysis duration for performance
        analysis_duration = min(target_duration, self.max_analysis_duration, total_duration)
        
        logger.debug(f"Video duration: {total_duration:.1f}s, analysis duration: {analysis_duration:.1f}s")
        
        # Generate different types of samples
        temporal_samples = self._generate_temporal_samples(total_duration, analysis_duration)
        scene_samples = self._generate_scene_based_samples(video_path, analysis_duration)
        complexity_samples = self._generate_complexity_based_samples(video_path, analysis_duration)
        
        # Combine and optimize sample selection
        all_samples = temporal_samples + scene_samples + complexity_samples
        optimized_samples = self._optimize_sample_selection(all_samples, target_duration)
        
        logger.info(f"Generated {len(optimized_samples)} representative samples "
                   f"covering {sum(s.duration for s in optimized_samples):.1f}s")
        
        return optimized_samples
    
    def _generate_temporal_samples(self, total_duration: float, analysis_duration: float) -> List[VideoSegment]:
        """Generate samples at regular temporal intervals."""
        samples = []
        
        # Calculate interval based on target coverage
        num_intervals = max(1, int(analysis_duration / self.temporal_interval_seconds))
        actual_interval = analysis_duration / num_intervals
        
        for i in range(num_intervals):
            start_time = i * actual_interval
            end_time = min(start_time + 2.0, analysis_duration)  # 2-second samples
            
            if end_time > start_time:
                samples.append(VideoSegment(
                    start_time=start_time,
                    end_time=end_time,
                    duration=end_time - start_time,
                    segment_type='temporal'
                ))
        
        logger.debug(f"Generated {len(samples)} temporal samples")
        return samples
    
    def _generate_scene_based_samples(self, video_path: str, analysis_duration: float) -> List[VideoSegment]:
        """Generate samples from different video scenes."""
        samples = []
        
        try:
            # Detect scene boundaries
            scene_transitions = self.identify_scene_boundaries(video_path)
            
            if not scene_transitions:
                logger.debug("No scene transitions detected, using temporal fallback")
                return []
            
            # Filter transitions within analysis duration
            valid_transitions = [t for t in scene_transitions if t.timestamp <= analysis_duration]
            
            if not valid_transitions:
                return []
            
            # Sample from each scene
            current_scene_start = 0.0
            
            for transition in valid_transitions:
                scene_duration = transition.timestamp - current_scene_start
                
                if scene_duration >= 1.0:  # Only sample scenes longer than 1 second
                    # Sample from middle of scene
                    sample_start = current_scene_start + scene_duration * 0.3
                    sample_end = min(sample_start + 2.0, transition.timestamp)
                    
                    if sample_end > sample_start:
                        samples.append(VideoSegment(
                            start_time=sample_start,
                            end_time=sample_end,
                            duration=sample_end - sample_start,
                            segment_type='scene'
                        ))
                
                current_scene_start = transition.timestamp
            
            # Sample from final scene
            final_scene_duration = analysis_duration - current_scene_start
            if final_scene_duration >= 1.0:
                sample_start = current_scene_start + final_scene_duration * 0.3
                sample_end = min(sample_start + 2.0, analysis_duration)
                
                if sample_end > sample_start:
                    samples.append(VideoSegment(
                        start_time=sample_start,
                        end_time=sample_end,
                        duration=sample_end - sample_start,
                        segment_type='scene'
                    ))
        
        except Exception as e:
            logger.warning(f"Scene-based sampling failed: {e}")
        
        logger.debug(f"Generated {len(samples)} scene-based samples")
        return samples
    
    def _generate_complexity_based_samples(self, video_path: str, analysis_duration: float) -> List[VideoSegment]:
        """Generate samples focusing on quality-critical regions."""
        samples = []
        
        try:
            # Analyze video complexity at regular intervals
            complexity_analysis = self._analyze_video_complexity(video_path, analysis_duration)
            
            if not complexity_analysis:
                return []
            
            # Find high-complexity regions
            high_complexity_regions = []
            
            for timestamp, complexity in complexity_analysis:
                if complexity >= self.complexity_threshold:
                    high_complexity_regions.append((timestamp, complexity))
            
            # Group nearby high-complexity points into regions
            if high_complexity_regions:
                regions = self._group_complexity_regions(high_complexity_regions)
                
                for region_start, region_end, avg_complexity in regions:
                    if region_end - region_start >= 0.5:  # At least 0.5 seconds
                        sample_duration = min(3.0, region_end - region_start)
                        
                        samples.append(VideoSegment(
                            start_time=region_start,
                            end_time=region_start + sample_duration,
                            duration=sample_duration,
                            segment_type='complexity',
                            complexity_score=avg_complexity
                        ))
        
        except Exception as e:
            logger.warning(f"Complexity-based sampling failed: {e}")
        
        logger.debug(f"Generated {len(samples)} complexity-based samples")
        return samples
    
    def _optimize_sample_selection(self, all_samples: List[VideoSegment], target_duration: float) -> List[VideoSegment]:
        """Optimize sample selection to avoid overlap and meet duration target."""
        if not all_samples:
            return []
        
        # Sort samples by start time
        sorted_samples = sorted(all_samples, key=lambda s: s.start_time)
        
        # Remove overlapping samples, preferring higher-priority types
        type_priority = {'complexity': 3, 'scene': 2, 'temporal': 1}
        
        optimized = []
        last_end_time = -1.0
        
        for sample in sorted_samples:
            # Check for overlap with previous sample
            if sample.start_time >= last_end_time:
                optimized.append(sample)
                last_end_time = sample.end_time
            else:
                # Handle overlap - keep higher priority sample
                if optimized and type_priority.get(sample.segment_type, 0) > type_priority.get(optimized[-1].segment_type, 0):
                    # Replace last sample with higher priority one
                    optimized[-1] = sample
                    last_end_time = sample.end_time
        
        # Limit total duration
        total_duration = 0.0
        final_samples = []
        
        for sample in optimized:
            if total_duration + sample.duration <= target_duration:
                final_samples.append(sample)
                total_duration += sample.duration
            else:
                # Truncate last sample if needed
                remaining_duration = target_duration - total_duration
                if remaining_duration > 0.5:  # Only if significant duration remains
                    truncated_sample = VideoSegment(
                        start_time=sample.start_time,
                        end_time=sample.start_time + remaining_duration,
                        duration=remaining_duration,
                        segment_type=sample.segment_type,
                        complexity_score=sample.complexity_score
                    )
                    final_samples.append(truncated_sample)
                break
        
        return final_samples
    
    def extract_temporal_samples(self, video_path: str, interval_seconds: float) -> List[VideoSegment]:
        """Extract frames at regular intervals across video duration.
        
        Args:
            video_path: Path to video file
            interval_seconds: Time interval between samples
            
        Returns:
            List of VideoSegment objects at regular intervals
        """
        logger.debug(f"Extracting temporal samples every {interval_seconds}s from {os.path.basename(video_path)}")
        
        video_info = self._get_video_info(video_path)
        if not video_info:
            return []
        
        duration = video_info['duration']
        samples = []
        
        current_time = 0.0
        while current_time < duration:
            sample_end = min(current_time + 1.0, duration)  # 1-second samples
            
            samples.append(VideoSegment(
                start_time=current_time,
                end_time=sample_end,
                duration=sample_end - current_time,
                segment_type='temporal'
            ))
            
            current_time += interval_seconds
        
        logger.debug(f"Generated {len(samples)} temporal samples")
        return samples
    
    def identify_scene_boundaries(self, video_path: str) -> List[SceneTransition]:
        """Identify scene boundaries using FFmpeg scene detection.
        
        Args:
            video_path: Path to video file
            
        Returns:
            List of SceneTransition objects marking scene changes
        """
        logger.debug(f"Identifying scene boundaries in {os.path.basename(video_path)}")
        
        transitions = []
        
        try:
            # Use FFmpeg scene detection filter
            cmd = [
                'ffmpeg', '-i', video_path, '-vf', f'select=gt(scene\\,{self.scene_change_threshold})',
                '-vsync', 'vfr', '-f', 'null', '-'
            ]
            
            # Alternative approach: use ffprobe with scene detection
            probe_cmd = [
                'ffprobe', '-f', 'lavfi', '-i', 
                f'movie={video_path},select=gt(scene\\,{self.scene_change_threshold})',
                '-show_entries', 'packet=pts_time', '-of', 'csv=p=0', '-v', 'quiet'
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0 and result.stdout.strip():
                timestamps = []
                for line in result.stdout.strip().split('\n'):
                    try:
                        timestamp = float(line.strip())
                        timestamps.append(timestamp)
                    except ValueError:
                        continue
                
                # Convert timestamps to SceneTransition objects
                for i, timestamp in enumerate(timestamps):
                    transitions.append(SceneTransition(
                        timestamp=timestamp,
                        frame_number=int(timestamp * 30),  # Assume 30fps
                        transition_score=self.scene_change_threshold,
                        scene_before=i,
                        scene_after=i + 1
                    ))
            
            else:
                # Fallback: create artificial scene boundaries at regular intervals
                logger.debug("Scene detection failed, using temporal fallback")
                video_info = self._get_video_info(video_path)
                if video_info:
                    duration = video_info['duration']
                    scene_interval = 30.0  # 30-second scenes
                    
                    current_time = scene_interval
                    scene_id = 0
                    
                    while current_time < duration:
                        transitions.append(SceneTransition(
                            timestamp=current_time,
                            frame_number=int(current_time * 30),
                            transition_score=0.5,
                            scene_before=scene_id,
                            scene_after=scene_id + 1
                        ))
                        
                        current_time += scene_interval
                        scene_id += 1
        
        except Exception as e:
            logger.warning(f"Scene boundary detection failed: {e}")
        
        logger.debug(f"Identified {len(transitions)} scene transitions")
        return transitions
    
    def sample_across_scenes(self, scenes: List[SceneTransition], samples_per_scene: int) -> List[VideoSegment]:
        """Sample representative segments from each detected scene.
        
        Args:
            scenes: List of scene transitions
            samples_per_scene: Number of samples to take from each scene
            
        Returns:
            List of VideoSegment objects distributed across scenes
        """
        logger.debug(f"Sampling {samples_per_scene} segments from each of {len(scenes)} scenes")
        
        samples = []
        
        if not scenes:
            return samples
        
        # Add implicit scene at the beginning
        scene_boundaries = [0.0] + [s.timestamp for s in scenes]
        
        for i in range(len(scene_boundaries) - 1):
            scene_start = scene_boundaries[i]
            scene_end = scene_boundaries[i + 1]
            scene_duration = scene_end - scene_start
            
            if scene_duration < 1.0:  # Skip very short scenes
                continue
            
            # Generate samples within this scene
            for j in range(samples_per_scene):
                # Distribute samples evenly within scene
                relative_position = (j + 0.5) / samples_per_scene
                sample_start = scene_start + relative_position * scene_duration * 0.8  # Use 80% of scene
                sample_end = min(sample_start + 1.5, scene_end)  # 1.5-second samples
                
                if sample_end > sample_start:
                    samples.append(VideoSegment(
                        start_time=sample_start,
                        end_time=sample_end,
                        duration=sample_end - sample_start,
                        segment_type='scene'
                    ))
        
        logger.debug(f"Generated {len(samples)} scene-distributed samples")
        return samples
    
    def select_quality_critical_frames(self, video_path: str, num_frames: int) -> List[FrameInfo]:
        """Select frames that are critical for quality assessment.
        
        Args:
            video_path: Path to video file
            num_frames: Number of frames to select
            
        Returns:
            List of FrameInfo objects for quality-critical frames
        """
        logger.debug(f"Selecting {num_frames} quality-critical frames from {os.path.basename(video_path)}")
        
        frames = []
        
        try:
            # Get video information
            video_info = self._get_video_info(video_path)
            if not video_info:
                return frames
            
            duration = video_info['duration']
            fps = video_info.get('fps', 30.0)
            total_frames = int(duration * fps)
            
            # Strategy 1: Key frames (I-frames)
            keyframes = self._detect_keyframes(video_path, num_frames // 3)
            
            # Strategy 2: High-complexity frames
            complexity_frames = self._select_high_complexity_frames(video_path, num_frames // 3)
            
            # Strategy 3: Motion boundary frames
            motion_frames = self._select_motion_boundary_frames(video_path, num_frames // 3)
            
            # Combine and deduplicate
            all_candidates = keyframes + complexity_frames + motion_frames
            
            # Remove duplicates and sort by timestamp
            unique_frames = {}
            for frame in all_candidates:
                key = int(frame.timestamp * 10)  # 0.1s precision
                if key not in unique_frames or frame.complexity_score > unique_frames[key].complexity_score:
                    unique_frames[key] = frame
            
            frames = sorted(unique_frames.values(), key=lambda f: f.timestamp)[:num_frames]
        
        except Exception as e:
            logger.warning(f"Quality-critical frame selection failed: {e}")
        
        logger.debug(f"Selected {len(frames)} quality-critical frames")
        return frames
    
    def _get_video_info(self, video_path: str) -> Optional[Dict[str, Any]]:
        """Get basic video information using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json', 
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                return None
            
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                return None
            
            # Extract info
            duration = float(data.get('format', {}).get('duration', 0))
            fps_str = video_stream.get('r_frame_rate', '30/1')
            
            # Parse fps
            if '/' in fps_str:
                num, den = fps_str.split('/')
                fps = float(num) / float(den) if float(den) != 0 else 30.0
            else:
                fps = float(fps_str)
            
            return {
                'duration': duration,
                'fps': fps,
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0))
            }
            
        except Exception as e:
            logger.debug(f"Error getting video info: {e}")
            return None
    
    def _analyze_video_complexity(self, video_path: str, max_duration: float) -> List[Tuple[float, float]]:
        """Analyze video complexity at regular intervals."""
        complexity_data = []
        
        try:
            # Sample complexity every 5 seconds
            interval = 5.0
            current_time = 0.0
            
            while current_time < max_duration:
                # Extract frame and analyze complexity
                complexity = self._estimate_frame_complexity(video_path, current_time)
                if complexity is not None:
                    complexity_data.append((current_time, complexity))
                
                current_time += interval
        
        except Exception as e:
            logger.debug(f"Complexity analysis failed: {e}")
        
        return complexity_data
    
    def _estimate_frame_complexity(self, video_path: str, timestamp: float) -> Optional[float]:
        """Estimate complexity of frame at given timestamp."""
        try:
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # Extract frame
            cmd = [
                'ffmpeg', '-v', 'quiet', '-ss', str(timestamp), '-i', video_path,
                '-vframes', '1', '-y', temp_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            
            if result.returncode == 0 and os.path.exists(temp_path):
                # Analyze frame complexity using OpenCV
                try:
                    img = cv2.imread(temp_path, cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        # Use Laplacian variance as complexity measure
                        laplacian_var = cv2.Laplacian(img, cv2.CV_64F).var()
                        # Normalize to 0-1 range (empirical scaling)
                        complexity = min(1.0, laplacian_var / 1000.0)
                        return complexity
                except Exception:
                    pass
            
            # Cleanup
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        
        except Exception:
            pass
        
        return None
    
    def _group_complexity_regions(self, complexity_points: List[Tuple[float, float]]) -> List[Tuple[float, float, float]]:
        """Group nearby high-complexity points into regions."""
        if not complexity_points:
            return []
        
        regions = []
        current_region_start = complexity_points[0][0]
        current_region_end = complexity_points[0][0]
        current_region_complexities = [complexity_points[0][1]]
        
        for timestamp, complexity in complexity_points[1:]:
            if timestamp - current_region_end <= 10.0:  # Within 10 seconds
                current_region_end = timestamp
                current_region_complexities.append(complexity)
            else:
                # Finish current region
                avg_complexity = sum(current_region_complexities) / len(current_region_complexities)
                regions.append((current_region_start, current_region_end, avg_complexity))
                
                # Start new region
                current_region_start = timestamp
                current_region_end = timestamp
                current_region_complexities = [complexity]
        
        # Add final region
        if current_region_complexities:
            avg_complexity = sum(current_region_complexities) / len(current_region_complexities)
            regions.append((current_region_start, current_region_end, avg_complexity))
        
        return regions
    
    def _detect_keyframes(self, video_path: str, num_frames: int) -> List[FrameInfo]:
        """Detect I-frames (keyframes) in the video."""
        frames = []
        
        try:
            # Use ffprobe to detect keyframes
            cmd = [
                'ffprobe', '-v', 'quiet', '-select_streams', 'v:0',
                '-show_entries', 'packet=pts_time,flags', '-of', 'csv=p=0',
                video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                keyframe_times = []
                
                for line in result.stdout.strip().split('\n'):
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            timestamp = float(parts[0])
                            flags = parts[1]
                            
                            if 'K' in flags:  # Keyframe flag
                                keyframe_times.append(timestamp)
                        except ValueError:
                            continue
                
                # Select subset of keyframes
                if keyframe_times:
                    step = max(1, len(keyframe_times) // num_frames)
                    selected_times = keyframe_times[::step][:num_frames]
                    
                    for i, timestamp in enumerate(selected_times):
                        frames.append(FrameInfo(
                            timestamp=timestamp,
                            frame_number=int(timestamp * 30),  # Assume 30fps
                            complexity_score=0.5,  # Default
                            motion_score=0.0,
                            scene_id=0,
                            is_keyframe=True
                        ))
        
        except Exception as e:
            logger.debug(f"Keyframe detection failed: {e}")
        
        return frames
    
    def _select_high_complexity_frames(self, video_path: str, num_frames: int) -> List[FrameInfo]:
        """Select frames with high spatial complexity."""
        frames = []
        
        try:
            # Analyze complexity at regular intervals
            video_info = self._get_video_info(video_path)
            if not video_info:
                return frames
            
            duration = video_info['duration']
            analysis_points = min(50, int(duration))  # Analyze up to 50 points
            
            complexity_data = []
            for i in range(analysis_points):
                timestamp = (i + 0.5) * duration / analysis_points
                complexity = self._estimate_frame_complexity(video_path, timestamp)
                if complexity is not None:
                    complexity_data.append((timestamp, complexity))
            
            # Sort by complexity and select top frames
            complexity_data.sort(key=lambda x: x[1], reverse=True)
            selected = complexity_data[:num_frames]
            
            for i, (timestamp, complexity) in enumerate(selected):
                frames.append(FrameInfo(
                    timestamp=timestamp,
                    frame_number=int(timestamp * video_info.get('fps', 30)),
                    complexity_score=complexity,
                    motion_score=0.0,
                    scene_id=0,
                    is_keyframe=False
                ))
        
        except Exception as e:
            logger.debug(f"High-complexity frame selection failed: {e}")
        
        return frames
    
    def _select_motion_boundary_frames(self, video_path: str, num_frames: int) -> List[FrameInfo]:
        """Select frames at motion boundaries (start/end of motion)."""
        frames = []
        
        try:
            # This is a simplified implementation
            # In practice, you'd analyze motion vectors or frame differences
            
            video_info = self._get_video_info(video_path)
            if not video_info:
                return frames
            
            duration = video_info['duration']
            
            # Sample at regular intervals and assume motion boundaries
            for i in range(num_frames):
                timestamp = (i + 0.5) * duration / num_frames
                
                frames.append(FrameInfo(
                    timestamp=timestamp,
                    frame_number=int(timestamp * video_info.get('fps', 30)),
                    complexity_score=0.4,
                    motion_score=0.6,  # Assume motion boundary
                    scene_id=0,
                    is_keyframe=False
                ))
        
        except Exception as e:
            logger.debug(f"Motion boundary frame selection failed: {e}")
        
        return frames