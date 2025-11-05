"""
Adaptive Quality Strategy
Main orchestrator for content-aware video compression optimization
"""

from typing import Dict, Any, List, Tuple, Optional
import logging
import time

try:
    from .content_analysis_engine import ContentAnalysisEngine, ContentAnalysis, ContentType
    from .content_aware_encoding_profiles import ContentAwareEncodingProfiles, EncodingProfile, BitrateAllocation, DynamicParameters
    from .quality_size_tradeoff_analyzer import QualitySizeTradeoffAnalyzer, TradeoffAnalysis, QualitySizeOption, TradeoffStrategy
except ImportError:
    from content_analysis_engine import ContentAnalysisEngine, ContentAnalysis, ContentType
    from content_aware_encoding_profiles import ContentAwareEncodingProfiles, EncodingProfile, BitrateAllocation, DynamicParameters
    from quality_size_tradeoff_analyzer import QualitySizeTradeoffAnalyzer, TradeoffAnalysis, QualitySizeOption, TradeoffStrategy

logger = logging.getLogger(__name__)


class AdaptiveQualityStrategy:
    """
    Main adaptive quality strategy that orchestrates content analysis,
    profile selection, and quality-size trade-off optimization.
    """
    
    def __init__(self, config_manager, temp_dir: str = None):
        self.config = config_manager
        
        # Initialize components
        self.content_analyzer = ContentAnalysisEngine(config_manager, temp_dir)
        self.encoding_profiles = ContentAwareEncodingProfiles(config_manager)
        self.tradeoff_analyzer = QualitySizeTradeoffAnalyzer(config_manager)
        
        # Configuration
        self.enable_content_analysis = self.config.get('adaptive_quality.enable_content_analysis', True)
        self.enable_dynamic_profiles = self.config.get('adaptive_quality.enable_dynamic_profiles', True)
        self.enable_tradeoff_analysis = self.config.get('adaptive_quality.enable_tradeoff_analysis', True)
        self.default_strategy = TradeoffStrategy(
            self.config.get('adaptive_quality.default_strategy', 'balanced')
        )
        
        # Cache for analysis results
        self._analysis_cache = {}
    
    def analyze_and_optimize(self, video_path: str, target_size_mb: float,
                           original_resolution: Tuple[int, int],
                           duration_seconds: float,
                           strategy: TradeoffStrategy = None) -> Dict[str, Any]:
        """
        Perform complete adaptive quality analysis and optimization.
        
        Args:
            video_path: Path to input video
            target_size_mb: Target file size in MB
            original_resolution: Original video resolution (width, height)
            duration_seconds: Video duration in seconds
            strategy: Preferred trade-off strategy
            
        Returns:
            Complete optimization results with recommendations
        """
        start_time = time.time()
        strategy = strategy or self.default_strategy
        
        logger.info(f"Starting adaptive quality analysis for {video_path}")
        logger.info(f"Target: {target_size_mb:.1f}MB, Duration: {duration_seconds:.1f}s, Strategy: {strategy.value}")
        
        results = {
            'video_path': video_path,
            'target_size_mb': target_size_mb,
            'duration_seconds': duration_seconds,
            'strategy': strategy.value,
            'analysis_time': 0.0,
            'content_analysis': None,
            'selected_profile': None,
            'bitrate_allocation': None,
            'dynamic_parameters': None,
            'tradeoff_analysis': None,
            'recommended_option': None,
            'user_notification': None,
            'success': False,
            'error': None
        }
        
        try:
            # Step 1: Content Analysis
            if self.enable_content_analysis:
                logger.info("Performing content analysis...")
                content_analysis = self.content_analyzer.analyze_content(video_path)
                results['content_analysis'] = content_analysis
                
                logger.info(f"Content analysis complete: type={content_analysis.content_type.value}, "
                           f"motion={content_analysis.motion_complexity:.1f}, "
                           f"spatial={content_analysis.spatial_complexity:.1f}")
            else:
                # Create fallback analysis
                content_analysis = self._create_fallback_content_analysis(video_path, duration_seconds)
                results['content_analysis'] = content_analysis
                logger.info("Using fallback content analysis (content analysis disabled)")
            
            # Step 2: Profile Selection
            if self.enable_dynamic_profiles:
                logger.info("Selecting optimal encoding profile...")
                profile = self.encoding_profiles.select_encoding_profile(content_analysis)
                results['selected_profile'] = profile
                
                logger.info(f"Selected profile: {profile.name} - {profile.description}")
            else:
                # Use default profile
                profile = self._get_default_profile()
                results['selected_profile'] = profile
                logger.info("Using default encoding profile (dynamic profiles disabled)")
            
            # Step 3: Bitrate Allocation
            logger.info("Calculating optimal bitrate allocation...")
            bitrate_allocation = self.encoding_profiles.calculate_optimal_bitrate_distribution(
                content_analysis, target_size_mb, duration_seconds, profile
            )
            results['bitrate_allocation'] = bitrate_allocation
            
            logger.info(f"Bitrate allocation: video={bitrate_allocation.video_bitrate_kbps}kbps, "
                       f"audio={bitrate_allocation.audio_bitrate_kbps}kbps, "
                       f"predicted_quality={bitrate_allocation.predicted_quality_score:.2f}")
            
            # Step 4: Dynamic Parameter Generation
            logger.info("Generating dynamic encoding parameters...")
            dynamic_params = self.encoding_profiles.generate_dynamic_parameters(
                content_analysis, profile, bitrate_allocation
            )
            results['dynamic_parameters'] = dynamic_params
            
            if dynamic_params.adjustments_made:
                logger.info(f"Dynamic adjustments: {', '.join(dynamic_params.adjustments_made)}")
            
            # Step 5: Trade-off Analysis
            if self.enable_tradeoff_analysis:
                logger.info("Performing quality-size trade-off analysis...")
                tradeoff_analysis = self.tradeoff_analyzer.analyze_tradeoffs(
                    content_analysis, target_size_mb, duration_seconds, 
                    original_resolution, strategy
                )
                results['tradeoff_analysis'] = tradeoff_analysis
                
                # Find recommended option
                recommended_option = None
                for option in tradeoff_analysis.options:
                    if option.option_id == tradeoff_analysis.recommended_option_id:
                        recommended_option = option
                        break
                
                results['recommended_option'] = recommended_option
                
                logger.info(f"Trade-off analysis complete: {len(tradeoff_analysis.options)} options generated")
                if recommended_option:
                    logger.info(f"Recommended: {recommended_option.description} "
                               f"({recommended_option.estimated_size_mb:.1f}MB, "
                               f"quality={recommended_option.quality_level.value})")
                
                # Generate user notification
                user_notification = self.tradeoff_analyzer.generate_user_notification_summary(tradeoff_analysis)
                results['user_notification'] = user_notification
            
            results['success'] = True
            
        except Exception as e:
            logger.error(f"Adaptive quality analysis failed: {e}")
            results['error'] = str(e)
            results['success'] = False
        
        finally:
            results['analysis_time'] = time.time() - start_time
            logger.info(f"Adaptive quality analysis completed in {results['analysis_time']:.2f}s")
        
        return results
    
    def get_encoding_parameters(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract encoding parameters from analysis results.
        
        Args:
            analysis_results: Results from analyze_and_optimize()
            
        Returns:
            Dictionary of encoding parameters ready for FFmpeg
        """
        if not analysis_results.get('success'):
            return self._get_fallback_encoding_parameters()
        
        content_analysis = analysis_results.get('content_analysis')
        profile = analysis_results.get('selected_profile')
        bitrate_allocation = analysis_results.get('bitrate_allocation')
        dynamic_params = analysis_results.get('dynamic_parameters')
        recommended_option = analysis_results.get('recommended_option')
        
        # Start with base parameters
        params = {
            'encoder': 'libx264',  # Default
            'preset': 'medium',
            'tune': 'film',
            'crf': 23,
            'video_bitrate': 1000,
            'audio_bitrate': 128,
            'video_filters': []
        }
        
        # Apply profile settings
        if profile:
            if profile.codec_preference:
                params['encoder'] = profile.codec_preference[0]
            params['preset'] = profile.preset
            params['tune'] = profile.tune
            params['crf'] = profile.crf_base
            
            # Apply psychovisual tuning
            params.update(profile.psychovisual_tuning)
        
        # Apply bitrate allocation
        if bitrate_allocation:
            params['video_bitrate'] = bitrate_allocation.video_bitrate_kbps
            params['audio_bitrate'] = bitrate_allocation.audio_bitrate_kbps
        
        # Apply dynamic parameters
        if dynamic_params:
            params['crf'] = dynamic_params.adjusted_crf
            params['preset'] = dynamic_params.adjusted_preset
            params['gop_size'] = dynamic_params.adjusted_gop_size
            params['b_frames'] = dynamic_params.adjusted_b_frames
            params['video_filters'] = dynamic_params.video_filters
        
        # Apply recommended option settings if available
        if recommended_option:
            params['video_bitrate'] = recommended_option.video_bitrate_kbps
            params['audio_bitrate'] = recommended_option.audio_bitrate_kbps
            params['resolution'] = recommended_option.resolution
            params['fps'] = recommended_option.fps
            params['encoder'] = recommended_option.codec
            params['preset'] = recommended_option.preset
        
        return params
    
    def get_quality_size_options(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Get user-friendly quality-size options from analysis results.
        
        Args:
            analysis_results: Results from analyze_and_optimize()
            
        Returns:
            List of user-friendly option dictionaries
        """
        tradeoff_analysis = analysis_results.get('tradeoff_analysis')
        if not tradeoff_analysis:
            return []
        
        options = []
        for option in tradeoff_analysis.options:
            option_dict = {
                'id': option.option_id,
                'name': option.description,
                'size_mb': option.estimated_size_mb,
                'quality_level': option.quality_level.value,
                'quality_score': option.predicted_quality_score,
                'encoding_time_minutes': option.encoding_time_estimate / 60.0,
                'pros': option.pros,
                'cons': option.cons,
                'recommended_for': option.recommended_for,
                'is_recommended': option.option_id == tradeoff_analysis.recommended_option_id,
                'parameters': {
                    'video_bitrate_kbps': option.video_bitrate_kbps,
                    'audio_bitrate_kbps': option.audio_bitrate_kbps,
                    'resolution': option.resolution,
                    'fps': option.fps,
                    'codec': option.codec,
                    'preset': option.preset
                }
            }
            options.append(option_dict)
        
        return options
    
    def get_user_notifications(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get user notifications and recommendations from analysis results.
        
        Args:
            analysis_results: Results from analyze_and_optimize()
            
        Returns:
            Dictionary with user notifications
        """
        notifications = {
            'warnings': [],
            'suggestions': [],
            'technical_notes': [],
            'alternatives': [],
            'summary': None
        }
        
        tradeoff_analysis = analysis_results.get('tradeoff_analysis')
        if tradeoff_analysis:
            notifications['warnings'] = tradeoff_analysis.warnings
            notifications['suggestions'] = tradeoff_analysis.suggestions
            notifications['technical_notes'] = tradeoff_analysis.technical_notes
            notifications['alternatives'] = tradeoff_analysis.alternative_approaches
        
        user_notification = analysis_results.get('user_notification')
        if user_notification:
            notifications['summary'] = user_notification
        
        return notifications
    
    def select_option_by_id(self, analysis_results: Dict[str, Any], option_id: str) -> Optional[Dict[str, Any]]:
        """
        Select a specific option by ID and return its parameters.
        
        Args:
            analysis_results: Results from analyze_and_optimize()
            option_id: ID of the option to select
            
        Returns:
            Encoding parameters for the selected option, or None if not found
        """
        tradeoff_analysis = analysis_results.get('tradeoff_analysis')
        if not tradeoff_analysis:
            return None
        
        for option in tradeoff_analysis.options:
            if option.option_id == option_id:
                return {
                    'encoder': option.codec,
                    'preset': option.preset,
                    'video_bitrate': option.video_bitrate_kbps,
                    'audio_bitrate': option.audio_bitrate_kbps,
                    'resolution': option.resolution,
                    'fps': option.fps,
                    'estimated_size_mb': option.estimated_size_mb,
                    'predicted_quality': option.predicted_quality_score,
                    'encoding_time_estimate': option.encoding_time_estimate
                }
        
        return None
    
    def _create_fallback_content_analysis(self, video_path: str, duration_seconds: float) -> ContentAnalysis:
        """Create fallback content analysis when analysis is disabled."""
        from .content_analysis_engine import ContentAnalysis, ContentType
        
        return ContentAnalysis(
            content_type=ContentType.LIVE_ACTION,
            motion_complexity=5.0,
            spatial_complexity=5.0,
            scene_count=max(1, int(duration_seconds / 30)),  # Estimate 1 scene per 30 seconds
            average_scene_duration=30.0,
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
    
    def _get_default_profile(self) -> EncodingProfile:
        """Get default encoding profile when dynamic profiles are disabled."""
        return self.encoding_profiles.profiles.get('live_action_standard')
    
    def _get_fallback_encoding_parameters(self) -> Dict[str, Any]:
        """Get fallback encoding parameters when analysis fails."""
        return {
            'encoder': 'libx264',
            'preset': 'medium',
            'tune': 'film',
            'crf': 23,
            'video_bitrate': 1000,
            'audio_bitrate': 128,
            'video_filters': [],
            'gop_size': 60,
            'b_frames': 3
        }
    
    def analyze_content_complexity(self, video_path: str) -> Dict[str, Any]:
        """
        Analyze content complexity (public interface for external use).
        
        Args:
            video_path: Path to video file
            
        Returns:
            Content complexity analysis results
        """
        try:
            content_analysis = self.content_analyzer.analyze_content(video_path)
            
            return {
                'content_type': content_analysis.content_type.value,
                'motion_complexity': content_analysis.motion_complexity,
                'spatial_complexity': content_analysis.spatial_complexity,
                'temporal_stability': content_analysis.temporal_stability,
                'scene_count': content_analysis.scene_count,
                'average_scene_duration': content_analysis.average_scene_duration,
                'color_complexity': content_analysis.color_complexity,
                'noise_level': content_analysis.noise_level,
                'edge_density': content_analysis.edge_density,
                'texture_complexity': content_analysis.texture_complexity,
                'recommended_profile': content_analysis.recommended_encoding_profile,
                'quality_critical_regions': content_analysis.quality_critical_regions
            }
            
        except Exception as e:
            logger.error(f"Content complexity analysis failed: {e}")
            return {
                'content_type': 'live_action',
                'motion_complexity': 5.0,
                'spatial_complexity': 5.0,
                'temporal_stability': 5.0,
                'scene_count': 1,
                'average_scene_duration': 60.0,
                'color_complexity': 5.0,
                'noise_level': 2.0,
                'edge_density': 0.1,
                'texture_complexity': 5.0,
                'recommended_profile': 'live_action_standard',
                'quality_critical_regions': []
            }
    
    def calculate_optimal_bitrate_distribution(self, content_complexity: Dict[str, Any],
                                             target_size_mb: float, duration_seconds: float) -> Dict[str, Any]:
        """
        Calculate optimal bitrate distribution (public interface for external use).
        
        Args:
            content_complexity: Content complexity analysis results
            target_size_mb: Target file size in MB
            duration_seconds: Video duration in seconds
            
        Returns:
            Bitrate allocation results
        """
        try:
            # Convert complexity dict back to ContentAnalysis object
            try:
                from .content_analysis_engine import ContentType
            except ImportError:
                from content_analysis_engine import ContentType
            
            content_analysis = ContentAnalysis(
                content_type=ContentType(content_complexity.get('content_type', 'live_action')),
                motion_complexity=content_complexity.get('motion_complexity', 5.0),
                spatial_complexity=content_complexity.get('spatial_complexity', 5.0),
                scene_count=content_complexity.get('scene_count', 1),
                average_scene_duration=content_complexity.get('average_scene_duration', 60.0),
                color_complexity=content_complexity.get('color_complexity', 5.0),
                noise_level=content_complexity.get('noise_level', 2.0),
                recommended_encoding_profile=content_complexity.get('recommended_profile', 'live_action_standard'),
                temporal_stability=content_complexity.get('temporal_stability', 5.0),
                edge_density=content_complexity.get('edge_density', 0.1),
                texture_complexity=content_complexity.get('texture_complexity', 5.0),
                motion_vectors_analysis={'complexity': content_complexity.get('motion_complexity', 5.0)},
                scene_transitions=[0.0],
                quality_critical_regions=content_complexity.get('quality_critical_regions', [])
            )
            
            # Select profile and calculate bitrate
            profile = self.encoding_profiles.select_encoding_profile(content_analysis)
            bitrate_allocation = self.encoding_profiles.calculate_optimal_bitrate_distribution(
                content_analysis, target_size_mb, duration_seconds, profile
            )
            
            return {
                'video_bitrate_kbps': bitrate_allocation.video_bitrate_kbps,
                'audio_bitrate_kbps': bitrate_allocation.audio_bitrate_kbps,
                'total_bitrate_kbps': bitrate_allocation.total_bitrate_kbps,
                'complexity_factor': bitrate_allocation.complexity_factor,
                'motion_adjustment': bitrate_allocation.motion_adjustment,
                'spatial_adjustment': bitrate_allocation.spatial_adjustment,
                'content_type_adjustment': bitrate_allocation.content_type_adjustment,
                'predicted_quality_score': bitrate_allocation.predicted_quality_score,
                'confidence_level': bitrate_allocation.confidence_level
            }
            
        except Exception as e:
            logger.error(f"Bitrate distribution calculation failed: {e}")
            # Return fallback allocation
            total_bitrate = (target_size_mb * 8 * 1024) / duration_seconds  # kbps
            return {
                'video_bitrate_kbps': int(total_bitrate * 0.9),
                'audio_bitrate_kbps': int(total_bitrate * 0.1),
                'total_bitrate_kbps': int(total_bitrate),
                'complexity_factor': 1.0,
                'motion_adjustment': 0.0,
                'spatial_adjustment': 0.0,
                'content_type_adjustment': 0.0,
                'predicted_quality_score': 0.7,
                'confidence_level': 0.5
            }
    
    def generate_quality_size_options(self, content_complexity: Dict[str, Any],
                                    target_size_mb: float, duration_seconds: float,
                                    original_resolution: Tuple[int, int]) -> List[Dict[str, Any]]:
        """
        Generate quality-size options (public interface for external use).
        
        Args:
            content_complexity: Content complexity analysis results
            target_size_mb: Target file size in MB
            duration_seconds: Video duration in seconds
            original_resolution: Original video resolution (width, height)
            
        Returns:
            List of quality-size options
        """
        try:
            # Convert complexity dict back to ContentAnalysis object
            try:
                from .content_analysis_engine import ContentType
            except ImportError:
                from content_analysis_engine import ContentType
            
            content_analysis = ContentAnalysis(
                content_type=ContentType(content_complexity.get('content_type', 'live_action')),
                motion_complexity=content_complexity.get('motion_complexity', 5.0),
                spatial_complexity=content_complexity.get('spatial_complexity', 5.0),
                scene_count=content_complexity.get('scene_count', 1),
                average_scene_duration=content_complexity.get('average_scene_duration', 60.0),
                color_complexity=content_complexity.get('color_complexity', 5.0),
                noise_level=content_complexity.get('noise_level', 2.0),
                recommended_encoding_profile=content_complexity.get('recommended_profile', 'live_action_standard'),
                temporal_stability=content_complexity.get('temporal_stability', 5.0),
                edge_density=content_complexity.get('edge_density', 0.1),
                texture_complexity=content_complexity.get('texture_complexity', 5.0),
                motion_vectors_analysis={'complexity': content_complexity.get('motion_complexity', 5.0)},
                scene_transitions=[0.0],
                quality_critical_regions=content_complexity.get('quality_critical_regions', [])
            )
            
            # Generate trade-off analysis
            tradeoff_analysis = self.tradeoff_analyzer.analyze_tradeoffs(
                content_analysis, target_size_mb, duration_seconds, original_resolution
            )
            
            # Convert to user-friendly format
            options = []
            for option in tradeoff_analysis.options:
                options.append({
                    'id': option.option_id,
                    'description': option.description,
                    'estimated_size_mb': option.estimated_size_mb,
                    'quality_level': option.quality_level.value,
                    'quality_score': option.predicted_quality_score,
                    'video_bitrate_kbps': option.video_bitrate_kbps,
                    'audio_bitrate_kbps': option.audio_bitrate_kbps,
                    'resolution': option.resolution,
                    'fps': option.fps,
                    'codec': option.codec,
                    'preset': option.preset,
                    'encoding_time_minutes': option.encoding_time_estimate / 60.0,
                    'pros': option.pros,
                    'cons': option.cons,
                    'recommended_for': option.recommended_for,
                    'is_recommended': option.option_id == tradeoff_analysis.recommended_option_id
                })
            
            return options
            
        except Exception as e:
            logger.error(f"Quality-size options generation failed: {e}")
            return []