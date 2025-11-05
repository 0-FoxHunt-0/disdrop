"""
Content Analysis Engine
Implements comprehensive video content analysis for adaptive quality optimization
"""

import os
import subprocess
import json
import math
import time
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import dataclass
from enum import Enum
import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)


class ContentType(Enum):
    """Video content type classification."""
    ANIMATION = "animation"
    LIVE_ACTION = "live_action"
    MIXED = "mixed"
    SCREEN_RECORDING = "screen_recording"
    GAMING = "gaming"


@dataclass
class ContentAnalysis:
    """Comprehensive content analysis results."""
    content_type: ContentType
    motion_complexity: float  # 0-10 scale
    spatial_complexity: float  # 0-10 scale
    scene_count: int
    average_scene_duration: float
    color_complexity: float  # 0-10 scale
    noise_level: float  # 0-10 scale
    recommended_encoding_profile: str
    
    # Additional analysis data
    temporal_stability: float  # 0-10 scale
    edge_density: float  # 0-10 scale
    texture_complexity: float  # 0-10 scale
    motion_vectors_analysis: Dict[str, Any]
    scene_transitions: List[float]
    quality_critical_regions: List[Tuple[float, float]]  # (start_time, end_time)


@dataclass
class SceneAnalysis:
    """Individual scene analysis results."""
    start_time: float
    end_time: float
    duration: float
    motion_level: float
    spatial_complexity: float
    color_variance: float
    edge_density: float
    is_quality_critical: bool


class ContentAnalysisEngine:
    """Advanced content analysis engine for video optimization."""
    
    def __init__(self, config_manager, temp_dir: str = None):
        self.config = config_manager
        self.temp_dir = temp_dir or config_manager.get_temp_dir()
        
        # Analysis configuration
        self.max_analysis_duration = self.config.get('content_analysis.max_duration_seconds', 120)
        self.sample_frame_count = self.config.get('content_analysis.sample_frame_count', 20)
        self.scene_detection_threshold = self.config.get('content_analysis.scene_threshold', 0.3)
        
        # Cache for expensive operations
        self._analysis_cache = {}
        
    def analyze_content(self, video_path: str) -> ContentAnalysis:
        """
        Perform comprehensive content analysis.
        
        Args:
            video_path: Path to video file
            
        Returns:
            ContentAnalysis object with complete analysis results
        """
        cache_key = f"{video_path}_{os.path.getmtime(video_path)}"
        if cache_key in self._analysis_cache:
            logger.debug(f"Using cached content analysis for {os.path.basename(video_path)}")
            return self._analysis_cache[cache_key]
        
        logger.info(f"Starting comprehensive content analysis for {os.path.basename(video_path)}")
        start_time = time.time()
        
        try:
            # Get basic video information
            video_info = self._get_video_info(video_path)
            
            # Detect content type
            content_type = self._detect_content_type(video_path, video_info)
            
            # Scene detection and analysis
            scene_transitions = self._detect_scene_changes(video_path)
            scene_analyses = self._analyze_scenes(video_path, scene_transitions)
            
            # Motion analysis
            motion_analysis = self._analyze_motion_complexity(video_path, video_info)
            
            # Spatial complexity analysis
            spatial_analysis = self._analyze_spatial_complexity(video_path, scene_transitions)
            
            # Color and texture analysis
            color_complexity = self._analyze_color_complexity(video_path)
            texture_complexity = self._analyze_texture_complexity(video_path)
            
            # Noise level detection
            noise_level = self._detect_noise_level(video_path)
            
            # Edge density analysis
            edge_density = self._analyze_edge_density(video_path)
            
            # Temporal stability analysis
            temporal_stability = self._analyze_temporal_stability(video_path, motion_analysis)
            
            # Quality critical regions identification
            quality_critical_regions = self._identify_quality_critical_regions(scene_analyses)
            
            # Generate encoding profile recommendation
            encoding_profile = self._recommend_encoding_profile(
                content_type, motion_analysis['complexity'], spatial_analysis['complexity']
            )
            
            # Compile results
            analysis = ContentAnalysis(
                content_type=content_type,
                motion_complexity=motion_analysis['complexity'],
                spatial_complexity=spatial_analysis['complexity'],
                scene_count=len(scene_transitions),
                average_scene_duration=self._calculate_average_scene_duration(scene_transitions, video_info['duration']),
                color_complexity=color_complexity,
                noise_level=noise_level,
                recommended_encoding_profile=encoding_profile,
                temporal_stability=temporal_stability,
                edge_density=edge_density,
                texture_complexity=texture_complexity,
                motion_vectors_analysis=motion_analysis,
                scene_transitions=scene_transitions,
                quality_critical_regions=quality_critical_regions
            )
            
            # Cache results
            self._analysis_cache[cache_key] = analysis
            
            analysis_duration = time.time() - start_time
            logger.info(f"Content analysis completed in {analysis_duration:.2f}s")
            logger.info(f"Analysis results: type={content_type.value}, motion={motion_analysis['complexity']:.1f}, "
                       f"spatial={spatial_analysis['complexity']:.1f}, scenes={len(scene_transitions)}")
            
            return analysis
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            # Return fallback analysis
            return self._create_fallback_analysis(video_path)
    
    def _get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Extract basic video information using ffprobe."""
        try:
            cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', video_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                raise RuntimeError(f"ffprobe failed: {result.stderr}")
            
            data = json.loads(result.stdout)
            
            # Find video stream
            video_stream = None
            for stream in data.get('streams', []):
                if stream.get('codec_type') == 'video':
                    video_stream = stream
                    break
            
            if not video_stream:
                raise RuntimeError("No video stream found")
            
            return {
                'duration': float(data.get('format', {}).get('duration', 0)),
                'width': int(video_stream.get('width', 0)),
                'height': int(video_stream.get('height', 0)),
                'fps': eval(video_stream.get('r_frame_rate', '30/1')),
                'codec': video_stream.get('codec_name', 'unknown'),
                'bitrate': int(video_stream.get('bit_rate', 0)),
                'pixel_format': video_stream.get('pix_fmt', 'unknown')
            }
            
        except Exception as e:
            logger.warning(f"Failed to get video info: {e}")
            return {
                'duration': 60.0, 'width': 1920, 'height': 1080, 'fps': 30.0,
                'codec': 'unknown', 'bitrate': 0, 'pixel_format': 'unknown'
            }
    
    def _detect_content_type(self, video_path: str, video_info: Dict[str, Any]) -> ContentType:
        """
        Detect content type using multiple heuristics.
        
        Args:
            video_path: Path to video file
            video_info: Basic video information
            
        Returns:
            Detected ContentType
        """
        try:
            # Sample frames for analysis
            sample_frames = self._extract_sample_frames(video_path, num_frames=5)
            
            animation_score = 0.0
            live_action_score = 0.0
            screen_recording_score = 0.0
            gaming_score = 0.0
            
            for frame in sample_frames:
                # Analyze frame characteristics
                frame_analysis = self._analyze_frame_characteristics(frame)
                
                # Animation indicators
                if frame_analysis['color_palette_limited']:
                    animation_score += 2.0
                if frame_analysis['sharp_edges']:
                    animation_score += 1.0
                if frame_analysis['flat_colors']:
                    animation_score += 1.5
                
                # Live action indicators
                if frame_analysis['natural_textures']:
                    live_action_score += 2.0
                if frame_analysis['continuous_gradients']:
                    live_action_score += 1.0
                if frame_analysis['noise_present']:
                    live_action_score += 1.0
                
                # Screen recording indicators
                if frame_analysis['text_detected']:
                    screen_recording_score += 2.0
                if frame_analysis['ui_elements']:
                    screen_recording_score += 1.5
                if frame_analysis['high_contrast']:
                    screen_recording_score += 1.0
                
                # Gaming indicators
                if frame_analysis['hud_elements']:
                    gaming_score += 2.0
                if frame_analysis['particle_effects']:
                    gaming_score += 1.5
                if frame_analysis['rapid_changes']:
                    gaming_score += 1.0
            
            # Determine content type based on scores
            scores = {
                ContentType.ANIMATION: animation_score,
                ContentType.LIVE_ACTION: live_action_score,
                ContentType.SCREEN_RECORDING: screen_recording_score,
                ContentType.GAMING: gaming_score
            }
            
            max_score = max(scores.values())
            if max_score < 3.0:
                return ContentType.MIXED
            
            # Check for mixed content
            high_scores = [content_type for content_type, score in scores.items() if score > max_score * 0.7]
            if len(high_scores) > 1:
                return ContentType.MIXED
            
            return max(scores, key=scores.get)
            
        except Exception as e:
            logger.warning(f"Content type detection failed: {e}")
            return ContentType.LIVE_ACTION  # Safe default
    
    def _extract_sample_frames(self, video_path: str, num_frames: int = 5) -> List[np.ndarray]:
        """Extract sample frames for analysis."""
        frames = []
        
        try:
            video_info = self._get_video_info(video_path)
            duration = video_info['duration']
            
            for i in range(num_frames):
                # Extract frame at regular intervals
                timestamp = (i + 1) * duration / (num_frames + 1)
                frame_path = os.path.join(self.temp_dir, f"sample_frame_{i}.png")
                
                cmd = [
                    'ffmpeg', '-ss', str(timestamp), '-i', video_path,
                    '-vframes', '1', '-y', frame_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                
                if result.returncode == 0 and os.path.exists(frame_path):
                    # Load frame with OpenCV
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        frames.append(frame)
                    
                    # Cleanup
                    os.remove(frame_path)
            
        except Exception as e:
            logger.warning(f"Frame extraction failed: {e}")
        
        return frames
    
    def _analyze_frame_characteristics(self, frame: np.ndarray) -> Dict[str, bool]:
        """Analyze frame characteristics for content type detection."""
        try:
            # Convert to different color spaces for analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            
            # Color palette analysis
            unique_colors = len(np.unique(frame.reshape(-1, frame.shape[-1]), axis=0))
            total_pixels = frame.shape[0] * frame.shape[1]
            color_palette_limited = unique_colors < total_pixels * 0.1
            
            # Edge analysis
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.sum(edges > 0) / total_pixels
            sharp_edges = edge_density > 0.15
            
            # Color flatness analysis
            color_variance = np.var(frame, axis=(0, 1))
            flat_colors = np.mean(color_variance) < 500
            
            # Texture analysis
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            natural_textures = laplacian_var > 100
            
            # Gradient analysis
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            continuous_gradients = np.std(gradient_magnitude) > 20
            
            # Noise detection
            noise_level = cv2.fastNlMeansDenoising(gray).var()
            noise_present = noise_level > 50
            
            # Text detection (simplified)
            text_detected = self._detect_text_regions(gray)
            
            # UI elements detection (simplified)
            ui_elements = self._detect_ui_elements(frame)
            
            # High contrast detection
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            high_contrast = (hist[0] + hist[255]) > total_pixels * 0.1
            
            # HUD elements detection (gaming)
            hud_elements = self._detect_hud_elements(frame)
            
            # Particle effects detection (gaming)
            particle_effects = self._detect_particle_effects(frame)
            
            # Rapid changes detection (would need temporal analysis)
            rapid_changes = False  # Placeholder
            
            return {
                'color_palette_limited': color_palette_limited,
                'sharp_edges': sharp_edges,
                'flat_colors': flat_colors,
                'natural_textures': natural_textures,
                'continuous_gradients': continuous_gradients,
                'noise_present': noise_present,
                'text_detected': text_detected,
                'ui_elements': ui_elements,
                'high_contrast': high_contrast,
                'hud_elements': hud_elements,
                'particle_effects': particle_effects,
                'rapid_changes': rapid_changes
            }
            
        except Exception as e:
            logger.warning(f"Frame characteristic analysis failed: {e}")
            return {key: False for key in [
                'color_palette_limited', 'sharp_edges', 'flat_colors', 'natural_textures',
                'continuous_gradients', 'noise_present', 'text_detected', 'ui_elements',
                'high_contrast', 'hud_elements', 'particle_effects', 'rapid_changes'
            ]}
    
    def _detect_text_regions(self, gray_frame: np.ndarray) -> bool:
        """Detect text regions in frame (simplified implementation)."""
        try:
            # Use morphological operations to detect text-like regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            morph = cv2.morphologyEx(gray_frame, cv2.MORPH_CLOSE, kernel)
            
            # Find contours that might be text
            contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_like_contours = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                
                # Text-like characteristics
                if 0.1 < aspect_ratio < 10 and w > 10 and h > 5:
                    text_like_contours += 1
            
            return text_like_contours > 5
            
        except Exception:
            return False
    
    def _detect_ui_elements(self, frame: np.ndarray) -> bool:
        """Detect UI elements like buttons, menus, etc."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect rectangular regions (common in UI)
            edges = cv2.Canny(gray, 50, 150)
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rectangular_regions = 0
            for contour in contours:
                # Approximate contour to polygon
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's roughly rectangular
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(contour)
                    if w > 20 and h > 10:  # Minimum size for UI element
                        rectangular_regions += 1
            
            return rectangular_regions > 3
            
        except Exception:
            return False
    
    def _detect_hud_elements(self, frame: np.ndarray) -> bool:
        """Detect HUD elements common in gaming content."""
        try:
            # Look for elements typically found at screen edges
            h, w = frame.shape[:2]
            
            # Check corners and edges for HUD-like elements
            corner_regions = [
                frame[0:h//4, 0:w//4],  # Top-left
                frame[0:h//4, 3*w//4:w],  # Top-right
                frame[3*h//4:h, 0:w//4],  # Bottom-left
                frame[3*h//4:h, 3*w//4:w]  # Bottom-right
            ]
            
            hud_indicators = 0
            for region in corner_regions:
                # Convert to grayscale
                gray_region = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                
                # Look for high contrast elements (typical in HUD)
                _, binary = cv2.threshold(gray_region, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                white_pixels = np.sum(binary == 255)
                total_pixels = region.shape[0] * region.shape[1]
                
                if 0.1 < white_pixels / total_pixels < 0.9:  # Mixed content suggests HUD
                    hud_indicators += 1
            
            return hud_indicators >= 2
            
        except Exception:
            return False
    
    def _detect_particle_effects(self, frame: np.ndarray) -> bool:
        """Detect particle effects common in gaming content."""
        try:
            # Look for small, bright, scattered elements
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect bright spots
            _, bright_spots = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
            
            # Find small contours (potential particles)
            contours, _ = cv2.findContours(bright_spots, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            small_bright_objects = 0
            for contour in contours:
                area = cv2.contourArea(contour)
                if 1 < area < 50:  # Small objects
                    small_bright_objects += 1
            
            return small_bright_objects > 20
            
        except Exception:
            return False
    
    def _detect_scene_changes(self, video_path: str) -> List[float]:
        """Detect scene changes using FFmpeg scene detection."""
        try:
            cmd = [
                'ffmpeg', '-i', video_path, '-vf',
                f'select=gt(scene\\,{self.scene_detection_threshold}),showinfo',
                '-f', 'null', '-'
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True, 
                timeout=self.max_analysis_duration, encoding='utf-8', errors='replace'
            )
            
            scene_changes = [0.0]  # Always include start
            
            for line in result.stderr.split('\n'):
                if 'pts_time:' in line:
                    try:
                        time_str = line.split('pts_time:')[1].split()[0]
                        scene_changes.append(float(time_str))
                    except (IndexError, ValueError):
                        continue
            
            return sorted(scene_changes)
            
        except Exception as e:
            logger.warning(f"Scene detection failed: {e}")
            return [0.0]  # Fallback to single scene
    
    def _analyze_scenes(self, video_path: str, scene_transitions: List[float]) -> List[SceneAnalysis]:
        """Analyze individual scenes for characteristics."""
        scenes = []
        video_info = self._get_video_info(video_path)
        duration = video_info['duration']
        
        for i, start_time in enumerate(scene_transitions):
            end_time = scene_transitions[i + 1] if i + 1 < len(scene_transitions) else duration
            scene_duration = end_time - start_time
            
            if scene_duration < 0.5:  # Skip very short scenes
                continue
            
            try:
                # Analyze scene characteristics
                scene_analysis = self._analyze_single_scene(video_path, start_time, end_time)
                scenes.append(scene_analysis)
                
            except Exception as e:
                logger.warning(f"Scene analysis failed for scene {i}: {e}")
                # Add fallback scene analysis
                scenes.append(SceneAnalysis(
                    start_time=start_time,
                    end_time=end_time,
                    duration=scene_duration,
                    motion_level=5.0,
                    spatial_complexity=5.0,
                    color_variance=5.0,
                    edge_density=5.0,
                    is_quality_critical=False
                ))
        
        return scenes
    
    def _analyze_single_scene(self, video_path: str, start_time: float, end_time: float) -> SceneAnalysis:
        """Analyze characteristics of a single scene."""
        duration = end_time - start_time
        mid_time = start_time + duration / 2
        
        # Extract frame from middle of scene
        frame_path = os.path.join(self.temp_dir, f"scene_analysis_{start_time:.2f}.png")
        
        try:
            cmd = [
                'ffmpeg', '-ss', str(mid_time), '-i', video_path,
                '-vframes', '1', '-y', frame_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(frame_path):
                frame = cv2.imread(frame_path)
                
                if frame is not None:
                    # Analyze frame characteristics
                    motion_level = self._estimate_motion_from_frame(frame)
                    spatial_complexity = self._calculate_frame_spatial_complexity(frame)
                    color_variance = self._calculate_color_variance(frame)
                    edge_density = self._calculate_edge_density(frame)
                    
                    # Determine if scene is quality critical
                    is_quality_critical = (
                        spatial_complexity > 7.0 or 
                        edge_density > 0.2 or 
                        color_variance > 8.0
                    )
                    
                    os.remove(frame_path)
                    
                    return SceneAnalysis(
                        start_time=start_time,
                        end_time=end_time,
                        duration=duration,
                        motion_level=motion_level,
                        spatial_complexity=spatial_complexity,
                        color_variance=color_variance,
                        edge_density=edge_density,
                        is_quality_critical=is_quality_critical
                    )
            
        except Exception as e:
            logger.warning(f"Single scene analysis failed: {e}")
        
        finally:
            if os.path.exists(frame_path):
                os.remove(frame_path)
        
        # Fallback analysis
        return SceneAnalysis(
            start_time=start_time,
            end_time=end_time,
            duration=duration,
            motion_level=5.0,
            spatial_complexity=5.0,
            color_variance=5.0,
            edge_density=0.1,
            is_quality_critical=False
        )
    
    def _analyze_motion_complexity(self, video_path: str, video_info: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze motion complexity using optical flow and motion vectors."""
        try:
            # Use FFmpeg to extract motion information
            cmd = [
                'ffmpeg', '-i', video_path, '-vf', 'select=not(mod(n\\,30))',
                '-vsync', 'vfr', '-f', 'null', '-'
            ]
            
            result = subprocess.run(
                cmd, capture_output=True, text=True,
                timeout=min(60, self.max_analysis_duration), encoding='utf-8', errors='replace'
            )
            
            # Analyze motion based on frame processing complexity
            # This is a simplified approach - in practice, you'd analyze actual motion vectors
            processing_lines = len([line for line in result.stderr.split('\n') if 'frame=' in line])
            
            # Estimate motion complexity based on processing characteristics
            base_complexity = min(processing_lines / 10.0, 10.0)
            
            # Adjust based on video characteristics
            fps = video_info.get('fps', 30.0)
            if fps > 50:
                base_complexity *= 1.2  # High FPS suggests motion content
            elif fps < 25:
                base_complexity *= 0.8  # Low FPS suggests less motion
            
            return {
                'complexity': min(base_complexity, 10.0),
                'average_motion': base_complexity,
                'motion_variance': base_complexity * 0.3,
                'high_motion_scenes': [],
                'static_scenes': []
            }
            
        except Exception as e:
            logger.warning(f"Motion analysis failed: {e}")
            return {
                'complexity': 5.0,
                'average_motion': 5.0,
                'motion_variance': 2.0,
                'high_motion_scenes': [],
                'static_scenes': []
            }
    
    def _analyze_spatial_complexity(self, video_path: str, scene_transitions: List[float]) -> Dict[str, Any]:
        """Analyze spatial complexity across scenes."""
        complexities = []
        
        try:
            for i, scene_time in enumerate(scene_transitions[:5]):  # Limit to first 5 scenes
                frame_path = os.path.join(self.temp_dir, f"spatial_analysis_{i}.png")
                
                cmd = [
                    'ffmpeg', '-ss', str(scene_time), '-i', video_path,
                    '-vframes', '1', '-y', frame_path
                ]
                
                result = subprocess.run(cmd, capture_output=True, timeout=30)
                
                if result.returncode == 0 and os.path.exists(frame_path):
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        complexity = self._calculate_frame_spatial_complexity(frame)
                        complexities.append(complexity)
                    
                    os.remove(frame_path)
            
            if complexities:
                avg_complexity = np.mean(complexities)
                max_complexity = np.max(complexities)
                variance = np.var(complexities)
            else:
                avg_complexity = max_complexity = 5.0
                variance = 1.0
            
            return {
                'complexity': avg_complexity,
                'max_complexity': max_complexity,
                'variance': variance,
                'per_scene': complexities
            }
            
        except Exception as e:
            logger.warning(f"Spatial complexity analysis failed: {e}")
            return {
                'complexity': 5.0,
                'max_complexity': 5.0,
                'variance': 1.0,
                'per_scene': [5.0]
            }
    
    def _calculate_frame_spatial_complexity(self, frame: np.ndarray) -> float:
        """Calculate spatial complexity of a single frame."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Calculate multiple complexity metrics
            
            # 1. Laplacian variance (edge/detail density)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # 2. Gradient magnitude
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            gradient_complexity = np.mean(gradient_magnitude)
            
            # 3. Texture complexity using Local Binary Patterns
            texture_complexity = self._calculate_texture_complexity_simple(gray)
            
            # 4. Frequency domain complexity
            freq_complexity = self._calculate_frequency_complexity(gray)
            
            # Combine metrics (normalize to 0-10 scale)
            complexity = (
                min(laplacian_var / 100, 10) * 0.3 +
                min(gradient_complexity / 50, 10) * 0.3 +
                texture_complexity * 0.2 +
                freq_complexity * 0.2
            )
            
            return min(complexity, 10.0)
            
        except Exception as e:
            logger.warning(f"Frame spatial complexity calculation failed: {e}")
            return 5.0
    
    def _calculate_texture_complexity_simple(self, gray_frame: np.ndarray) -> float:
        """Calculate texture complexity using simple statistical measures."""
        try:
            # Use standard deviation of local regions
            kernel_size = 5
            kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
            local_mean = cv2.filter2D(gray_frame.astype(np.float32), -1, kernel)
            local_variance = cv2.filter2D((gray_frame.astype(np.float32) - local_mean)**2, -1, kernel)
            
            texture_complexity = np.mean(np.sqrt(local_variance))
            return min(texture_complexity / 20, 10.0)  # Normalize to 0-10
            
        except Exception:
            return 5.0
    
    def _calculate_frequency_complexity(self, gray_frame: np.ndarray) -> float:
        """Calculate complexity in frequency domain."""
        try:
            # Apply FFT
            f_transform = np.fft.fft2(gray_frame)
            f_shift = np.fft.fftshift(f_transform)
            magnitude_spectrum = np.log(np.abs(f_shift) + 1)
            
            # High frequency content indicates complexity
            h, w = magnitude_spectrum.shape
            center_h, center_w = h // 2, w // 2
            
            # Extract high frequency region (outer part of spectrum)
            mask = np.zeros((h, w), dtype=bool)
            y, x = np.ogrid[:h, :w]
            mask_inner = (x - center_w)**2 + (y - center_h)**2 < (min(h, w) // 4)**2
            mask = ~mask_inner
            
            high_freq_energy = np.mean(magnitude_spectrum[mask])
            return min(high_freq_energy / 5, 10.0)  # Normalize to 0-10
            
        except Exception:
            return 5.0
    
    def _analyze_color_complexity(self, video_path: str) -> float:
        """Analyze color complexity of the video."""
        try:
            # Extract a few sample frames
            sample_frames = self._extract_sample_frames(video_path, num_frames=3)
            
            if not sample_frames:
                return 5.0
            
            color_complexities = []
            
            for frame in sample_frames:
                # Convert to HSV for better color analysis
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Calculate color histogram
                hist_h = cv2.calcHist([hsv], [0], None, [180], [0, 180])
                hist_s = cv2.calcHist([hsv], [1], None, [256], [0, 256])
                hist_v = cv2.calcHist([hsv], [2], None, [256], [0, 256])
                
                # Color diversity (number of significant colors)
                significant_colors_h = np.sum(hist_h > np.max(hist_h) * 0.01)
                significant_colors_s = np.sum(hist_s > np.max(hist_s) * 0.01)
                
                # Color variance
                color_variance = np.var(frame, axis=(0, 1))
                avg_color_variance = np.mean(color_variance)
                
                # Combine metrics
                complexity = (
                    min(significant_colors_h / 20, 10) * 0.4 +
                    min(significant_colors_s / 30, 10) * 0.3 +
                    min(avg_color_variance / 1000, 10) * 0.3
                )
                
                color_complexities.append(complexity)
            
            return np.mean(color_complexities)
            
        except Exception as e:
            logger.warning(f"Color complexity analysis failed: {e}")
            return 5.0
    
    def _analyze_texture_complexity(self, video_path: str) -> float:
        """Analyze texture complexity of the video."""
        try:
            sample_frames = self._extract_sample_frames(video_path, num_frames=3)
            
            if not sample_frames:
                return 5.0
            
            texture_complexities = []
            
            for frame in sample_frames:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                texture_complexity = self._calculate_texture_complexity_simple(gray)
                texture_complexities.append(texture_complexity)
            
            return np.mean(texture_complexities)
            
        except Exception as e:
            logger.warning(f"Texture complexity analysis failed: {e}")
            return 5.0
    
    def _detect_noise_level(self, video_path: str) -> float:
        """Detect noise level in the video."""
        try:
            # Extract a sample frame for noise analysis
            sample_frames = self._extract_sample_frames(video_path, num_frames=1)
            
            if not sample_frames:
                return 2.0
            
            frame = sample_frames[0]
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use denoising to estimate noise level
            denoised = cv2.fastNlMeansDenoising(gray)
            noise = cv2.absdiff(gray, denoised)
            
            noise_level = np.mean(noise)
            return min(noise_level / 10, 10.0)  # Normalize to 0-10
            
        except Exception as e:
            logger.warning(f"Noise detection failed: {e}")
            return 2.0
    
    def _analyze_edge_density(self, video_path: str) -> float:
        """Analyze edge density in the video."""
        try:
            sample_frames = self._extract_sample_frames(video_path, num_frames=3)
            
            if not sample_frames:
                return 0.1
            
            edge_densities = []
            
            for frame in sample_frames:
                edge_density = self._calculate_edge_density(frame)
                edge_densities.append(edge_density)
            
            return np.mean(edge_densities)
            
        except Exception as e:
            logger.warning(f"Edge density analysis failed: {e}")
            return 0.1
    
    def _calculate_edge_density(self, frame: np.ndarray) -> float:
        """Calculate edge density of a frame."""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            
            total_pixels = frame.shape[0] * frame.shape[1]
            edge_pixels = np.sum(edges > 0)
            
            return edge_pixels / total_pixels
            
        except Exception:
            return 0.1
    
    def _analyze_temporal_stability(self, video_path: str, motion_analysis: Dict[str, Any]) -> float:
        """Analyze temporal stability (inverse of motion complexity)."""
        motion_complexity = motion_analysis.get('complexity', 5.0)
        motion_variance = motion_analysis.get('motion_variance', 2.0)
        
        # Higher motion = lower stability
        stability = 10.0 - (motion_complexity * 0.7 + motion_variance * 0.3)
        return max(0.0, min(stability, 10.0))
    
    def _identify_quality_critical_regions(self, scene_analyses: List[SceneAnalysis]) -> List[Tuple[float, float]]:
        """Identify regions that are critical for quality preservation."""
        critical_regions = []
        
        for scene in scene_analyses:
            if scene.is_quality_critical:
                critical_regions.append((scene.start_time, scene.end_time))
        
        return critical_regions
    
    def _recommend_encoding_profile(self, content_type: ContentType, motion_complexity: float, 
                                   spatial_complexity: float) -> str:
        """Recommend encoding profile based on content analysis."""
        
        if content_type == ContentType.ANIMATION:
            if spatial_complexity > 7.0:
                return "animation_high_detail"
            elif motion_complexity > 7.0:
                return "animation_high_motion"
            else:
                return "animation_standard"
        
        elif content_type == ContentType.SCREEN_RECORDING:
            return "screen_recording"
        
        elif content_type == ContentType.GAMING:
            if motion_complexity > 8.0:
                return "gaming_high_action"
            else:
                return "gaming_standard"
        
        elif content_type == ContentType.LIVE_ACTION:
            if motion_complexity > 7.0 and spatial_complexity > 7.0:
                return "live_action_complex"
            elif motion_complexity > 7.0:
                return "live_action_high_motion"
            elif spatial_complexity > 7.0:
                return "live_action_high_detail"
            else:
                return "live_action_standard"
        
        else:  # MIXED
            # Use balanced approach for mixed content
            if motion_complexity > 6.0 or spatial_complexity > 6.0:
                return "mixed_complex"
            else:
                return "mixed_standard"
    
    def _calculate_average_scene_duration(self, scene_transitions: List[float], total_duration: float) -> float:
        """Calculate average scene duration."""
        if len(scene_transitions) <= 1:
            return total_duration
        
        durations = []
        for i in range(len(scene_transitions) - 1):
            duration = scene_transitions[i + 1] - scene_transitions[i]
            durations.append(duration)
        
        # Add final scene duration
        if scene_transitions:
            final_duration = total_duration - scene_transitions[-1]
            if final_duration > 0:
                durations.append(final_duration)
        
        return np.mean(durations) if durations else total_duration
    
    def _estimate_motion_from_frame(self, frame: np.ndarray) -> float:
        """Estimate motion level from a single frame (simplified approach)."""
        try:
            # This is a simplified approach - ideally you'd compare consecutive frames
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Use gradient magnitude as a proxy for potential motion
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
            
            motion_estimate = np.mean(gradient_magnitude) / 20  # Normalize
            return min(motion_estimate, 10.0)
            
        except Exception:
            return 5.0
    
    def _calculate_color_variance(self, frame: np.ndarray) -> float:
        """Calculate color variance in a frame."""
        try:
            color_variance = np.var(frame, axis=(0, 1))
            avg_variance = np.mean(color_variance)
            return min(avg_variance / 500, 10.0)  # Normalize to 0-10
            
        except Exception:
            return 5.0
    
    def _create_fallback_analysis(self, video_path: str) -> ContentAnalysis:
        """Create fallback analysis when full analysis fails."""
        logger.warning("Using fallback content analysis")
        
        return ContentAnalysis(
            content_type=ContentType.LIVE_ACTION,
            motion_complexity=5.0,
            spatial_complexity=5.0,
            scene_count=1,
            average_scene_duration=60.0,
            color_complexity=5.0,
            noise_level=2.0,
            recommended_encoding_profile="live_action_standard",
            temporal_stability=5.0,
            edge_density=0.1,
            texture_complexity=5.0,
            motion_vectors_analysis={'complexity': 5.0, 'average_motion': 5.0, 'motion_variance': 2.0},
            scene_transitions=[0.0],
            quality_critical_regions=[]
        )