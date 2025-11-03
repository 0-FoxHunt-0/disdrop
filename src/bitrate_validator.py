"""
Bitrate Validation Module
Validates calculated bitrates against encoder minimum requirements and provides adjustment suggestions
"""

import logging
from typing import Dict, Any, Optional, Tuple, List
from dataclasses import dataclass
try:
    from .config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

logger = logging.getLogger(__name__)


class BitrateValidationError(Exception):
    """Custom exception for bitrate validation failures"""
    
    def __init__(self, message: str, bitrate_kbps: int, minimum_required: int, 
                 encoder: str, severity: str = 'error', context: str = None, 
                 video_info: Dict[str, Any] = None):
        super().__init__(message)
        self.bitrate_kbps = bitrate_kbps
        self.minimum_required = minimum_required
        self.encoder = encoder
        self.severity = severity
        self.context = context or "unknown"
        self.video_info = video_info or {}
        
        # Calculate additional diagnostic information
        self.deficit_kbps = minimum_required - bitrate_kbps
        self.deficit_ratio = minimum_required / max(bitrate_kbps, 1)
    
    def get_detailed_message(self) -> str:
        """Get detailed error message with context-specific suggestions"""
        base_msg = f"Bitrate validation failed for {self.encoder}: {self.bitrate_kbps}kbps < {self.minimum_required}kbps minimum"
        
        # Add context information
        if self.context != "unknown":
            base_msg += f" (Context: {self.context})"
        
        # Add deficit information
        base_msg += f"\nDeficit: {self.deficit_kbps}kbps ({self.deficit_ratio:.1f}x below minimum)"
        
        # Add video-specific context if available
        if self.video_info:
            duration = self.video_info.get('duration', 0)
            resolution = f"{self.video_info.get('width', 'unknown')}x{self.video_info.get('height', 'unknown')}"
            fps = self.video_info.get('fps', 'unknown')
            base_msg += f"\nVideo: {resolution}@{fps}fps, {duration:.1f}s duration"
        
        # Context-specific suggestions
        if self.severity == 'critical':
            if self.deficit_ratio > 3.0:
                suggestions = [
                    "• CRITICAL: Bitrate is extremely low - consider segmentation",
                    "• Reduce resolution to 320x180 or lower",
                    "• Lower frame rate to 10-15 fps",
                    "• Split video into multiple smaller files",
                    "• Increase target file size if possible"
                ]
            else:
                suggestions = [
                    "• Try reducing video resolution significantly",
                    "• Consider lowering frame rate to 15-20 fps", 
                    "• Enable video segmentation to split into smaller files",
                    "• Use a different encoder with lower minimum requirements"
                ]
        else:
            suggestions = [
                "• Try reducing video resolution slightly",
                "• Consider lowering frame rate",
                "• Enable automatic parameter adjustment", 
                "• Use segmentation for very long videos"
            ]
        
        # Add encoder-specific suggestions
        if self.encoder == 'libx264':
            suggestions.append("• Try hardware encoder (h264_nvenc/h264_amf) for lower minimums")
        elif self.encoder in ['h264_nvenc', 'h264_amf', 'h264_qsv']:
            suggestions.append("• Hardware encoder already in use - consider segmentation")
        
        return f"{base_msg}\n\nSuggested solutions:\n" + "\n".join(suggestions)
    
    def get_short_message(self) -> str:
        """Get concise error message for logging"""
        return f"{self.encoder} bitrate validation failed: {self.bitrate_kbps}kbps < {self.minimum_required}kbps (deficit: {self.deficit_kbps}kbps)"


@dataclass
class ValidationResult:
    """Result of bitrate validation"""
    is_valid: bool
    current_bitrate: int
    minimum_required: int
    adjustment_needed: bool
    severity: str  # 'warning', 'critical'
    message: str


@dataclass
class AdjustmentPlan:
    """Plan for adjusting parameters to meet bitrate requirements"""
    strategy: str  # 'resolution_reduction', 'fps_reduction', 'segmentation'
    new_params: Dict[str, Any]
    expected_bitrate: int
    quality_impact: str  # 'minimal', 'moderate', 'significant'
    fallback_available: bool


class BitrateValidator:
    """Validates bitrates against encoder minimum requirements"""
    
    def __init__(self, config: ConfigManager):
        self.config = config
        
        # Load encoder minimums from config or use defaults
        self.encoder_minimums = self._load_encoder_minimums()
        
        # Load fallback resolutions from config or use defaults
        self.fallback_resolutions = self._load_fallback_resolutions()
        
        logger.debug(f"BitrateValidator initialized with encoder minimums: {self.encoder_minimums}")
    
    def _load_encoder_minimums(self) -> Dict[str, int]:
        """Load encoder minimum bitrate requirements from configuration"""
        config_minimums = self.config.get('video_compression.bitrate_validation.encoder_minimums', {})
        
        # Default minimums based on encoder capabilities
        defaults = {
            'libx264': 3,      # kbps - FFmpeg libx264 minimum
            'libx265': 5,      # kbps - HEVC requires higher minimum
            'h264_nvenc': 2,   # kbps - NVENC can go lower
            'h264_amf': 2,     # kbps - AMD AMF similar to NVENC
            'h264_qsv': 2,     # kbps - Intel QSV similar to NVENC
            'h264_videotoolbox': 2,  # kbps - Apple VideoToolbox
        }
        
        # Merge config with defaults, config takes precedence
        encoder_minimums = defaults.copy()
        encoder_minimums.update(config_minimums)
        
        return encoder_minimums
    
    def _load_fallback_resolutions(self) -> List[Tuple[int, int]]:
        """Load fallback resolution candidates from configuration"""
        config_resolutions = self.config.get('video_compression.bitrate_validation.fallback_resolutions', [])
        
        # Default fallback resolutions for extreme compression scenarios
        defaults = [
            (320, 180),   # Ultra-low for extreme cases
            (426, 240),   # 240p
            (480, 270),   # 270p
            (640, 360),   # 360p
            (854, 480),   # 480p
        ]
        
        # Convert config format to tuples if provided
        if config_resolutions:
            try:
                fallback_resolutions = [(res[0], res[1]) for res in config_resolutions]
            except (IndexError, TypeError):
                logger.warning("Invalid fallback_resolutions format in config, using defaults")
                fallback_resolutions = defaults
        else:
            fallback_resolutions = defaults
        
        return fallback_resolutions
    
    def validate_bitrate(self, bitrate_kbps: int, encoder: str) -> ValidationResult:
        """
        Validate bitrate against encoder minimum requirements
        
        Args:
            bitrate_kbps: Calculated bitrate in kbps
            encoder: Encoder name (e.g., 'libx264', 'h264_nvenc')
            
        Returns:
            ValidationResult with validation status and details
        """
        # Get minimum requirement for this encoder
        min_required = self.encoder_minimums.get(encoder, 3)  # Default to 3kbps if unknown
        
        is_valid = bitrate_kbps >= min_required
        adjustment_needed = not is_valid
        
        if is_valid:
            severity = 'info'
            message = f"Bitrate {bitrate_kbps}kbps meets {encoder} minimum requirement ({min_required}kbps)"
        else:
            # Determine severity based on how far below minimum we are
            deficit_ratio = min_required / max(bitrate_kbps, 1)
            if deficit_ratio > 2.0:
                severity = 'critical'
                message = f"Bitrate {bitrate_kbps}kbps critically below {encoder} minimum ({min_required}kbps). Severe quality degradation expected."
            else:
                severity = 'warning'
                message = f"Bitrate {bitrate_kbps}kbps below {encoder} minimum ({min_required}kbps). Quality may be compromised."
        
        return ValidationResult(
            is_valid=is_valid,
            current_bitrate=bitrate_kbps,
            minimum_required=min_required,
            adjustment_needed=adjustment_needed,
            severity=severity,
            message=message
        )
    
    def suggest_adjustments(self, params: Dict[str, Any], target_size_mb: float, 
                          video_info: Dict[str, Any]) -> AdjustmentPlan:
        """
        Suggest parameter adjustments to meet bitrate requirements
        
        Args:
            params: Current encoding parameters
            target_size_mb: Target file size in MB
            video_info: Video information (duration, etc.)
            
        Returns:
            AdjustmentPlan with suggested changes
        """
        current_width = params.get('width', 1920)
        current_height = params.get('height', 1080)
        current_fps = params.get('fps', 30)
        current_bitrate = params.get('bitrate', 1000)
        encoder = params.get('encoder', 'libx264')
        duration = video_info.get('duration_seconds', 60)
        audio_bitrate = params.get('audio_bitrate', 64)
        
        # Calculate minimum viable bitrate for this encoder
        min_bitrate = self.encoder_minimums.get(encoder, 3)
        
        # Try resolution reduction first (least quality impact for most content)
        for width, height in self.fallback_resolutions:
            if width < current_width or height < current_height:
                # Calculate what bitrate we'd get with this resolution
                new_bitrate = self._calculate_target_bitrate(target_size_mb, duration, audio_bitrate)
                
                if new_bitrate >= min_bitrate:
                    new_params = params.copy()
                    new_params.update({
                        'width': width,
                        'height': height,
                        'bitrate': new_bitrate
                    })
                    
                    # Determine quality impact
                    resolution_ratio = (width * height) / (current_width * current_height)
                    if resolution_ratio > 0.7:
                        quality_impact = 'minimal'
                    elif resolution_ratio > 0.4:
                        quality_impact = 'moderate'
                    else:
                        quality_impact = 'significant'
                    
                    return AdjustmentPlan(
                        strategy='resolution_reduction',
                        new_params=new_params,
                        expected_bitrate=new_bitrate,
                        quality_impact=quality_impact,
                        fallback_available=True
                    )
        
        # Try FPS reduction if resolution reduction isn't sufficient
        for fps_reduction in [0.8, 0.6, 0.5]:  # 80%, 60%, 50% of original FPS
            new_fps = max(10, current_fps * fps_reduction)  # Don't go below 10fps
            
            # Recalculate bitrate with reduced FPS (proportional reduction)
            fps_ratio = new_fps / current_fps
            adjusted_bitrate = int(current_bitrate * fps_ratio)
            
            if adjusted_bitrate >= min_bitrate:
                new_params = params.copy()
                new_params.update({
                    'fps': new_fps,
                    'bitrate': adjusted_bitrate
                })
                
                quality_impact = 'moderate' if fps_reduction >= 0.6 else 'significant'
                
                return AdjustmentPlan(
                    strategy='fps_reduction',
                    new_params=new_params,
                    expected_bitrate=adjusted_bitrate,
                    quality_impact=quality_impact,
                    fallback_available=True
                )
        
        # If neither resolution nor FPS reduction works, suggest segmentation
        return AdjustmentPlan(
            strategy='segmentation',
            new_params=params,  # Keep original params for segments
            expected_bitrate=current_bitrate,
            quality_impact='minimal',  # Segmentation preserves quality
            fallback_available=True
        )
    
    def calculate_minimum_viable_params(self, video_info: Dict[str, Any], 
                                      size_limit: float, encoder: str = 'libx264') -> Dict[str, Any]:
        """
        Calculate minimum viable parameters that meet encoder requirements
        
        Args:
            video_info: Video information including duration
            size_limit: Target size limit in MB
            encoder: Target encoder
            
        Returns:
            Dictionary with minimum viable parameters
        """
        duration = video_info.get('duration_seconds', 60)
        min_bitrate = self.encoder_minimums.get(encoder, 3)
        audio_bitrate = 64  # Conservative audio bitrate
        
        # Calculate what video bitrate we can afford
        available_bitrate = self._calculate_target_bitrate(size_limit, duration, audio_bitrate)
        
        if available_bitrate >= min_bitrate:
            # We can meet minimum requirements with reasonable resolution
            return {
                'bitrate': available_bitrate,
                'audio_bitrate': audio_bitrate,
                'encoder': encoder,
                'viable': True
            }
        
        # Find the smallest resolution that allows minimum bitrate
        for width, height in reversed(self.fallback_resolutions):  # Start with smallest
            # Calculate required size for minimum bitrate
            required_size = self._calculate_required_size(min_bitrate, duration, audio_bitrate)
            
            if required_size <= size_limit * 1.1:  # Allow 10% margin
                return {
                    'width': width,
                    'height': height,
                    'bitrate': min_bitrate,
                    'audio_bitrate': audio_bitrate,
                    'encoder': encoder,
                    'viable': True,
                    'requires_segmentation': False
                }
        
        # If even smallest resolution doesn't work, segmentation is needed
        return {
            'bitrate': min_bitrate,
            'audio_bitrate': audio_bitrate,
            'encoder': encoder,
            'viable': False,
            'requires_segmentation': True
        }
    
    def _calculate_target_bitrate(self, target_size_mb: float, duration_s: float, audio_kbps: int) -> int:
        """Calculate target video bitrate for given constraints"""
        total_bits = target_size_mb * 8 * 1024 * 1024
        video_kbps = int(max((total_bits / max(duration_s, 1.0) / 1000) - audio_kbps, 64))
        return video_kbps
    
    def _calculate_required_size(self, video_bitrate_kbps: int, duration_s: float, audio_kbps: int) -> float:
        """Calculate required file size for given bitrates"""
        total_kbps = video_bitrate_kbps + audio_kbps
        total_bits = total_kbps * 1000 * duration_s
        size_mb = total_bits / (8 * 1024 * 1024)
        return size_mb
    
    def is_validation_enabled(self) -> bool:
        """Check if bitrate validation is enabled in configuration"""
        return self.config.get('video_compression.bitrate_validation.enabled', True)
    
    def get_encoder_minimum(self, encoder: str) -> int:
        """Get minimum bitrate requirement for specific encoder"""
        return self.encoder_minimums.get(encoder, 3)
    
    def log_validation_result(self, result: ValidationResult, context: str = "", 
                            video_info: Dict[str, Any] = None):
        """Log validation result with appropriate level and enhanced context"""
        prefix = f"[{context}] " if context else ""
        
        # Add video context to message if available
        video_context = ""
        if video_info:
            duration = video_info.get('duration', 0)
            resolution = f"{video_info.get('width', 'unknown')}x{video_info.get('height', 'unknown')}"
            fps = video_info.get('fps', 'unknown')
            video_context = f" ({resolution}@{fps}fps, {duration:.1f}s)"
        
        message = f"{prefix}{result.message}{video_context}"
        
        if result.severity == 'critical':
            logger.error(message)
            # Log additional diagnostic information for critical failures
            if result.current_bitrate > 0:
                deficit = result.minimum_required - result.current_bitrate
                deficit_ratio = result.minimum_required / result.current_bitrate
                logger.error(f"{prefix}Bitrate deficit: {deficit}kbps ({deficit_ratio:.1f}x below minimum)")
        elif result.severity == 'warning':
            logger.warning(message)
        else:
            logger.info(message)
    
    def log_fallback_strategy_used(self, strategy: str, original_params: Dict[str, Any], 
                                 new_params: Dict[str, Any], context: str = "", 
                                 reason: str = None):
        """Log when a fallback strategy is used with enhanced details"""
        prefix = f"[{context}] " if context else ""
        reason_suffix = f" (Reason: {reason})" if reason else ""
        
        if strategy == 'resolution_reduction':
            orig_res = f"{original_params.get('width', 'unknown')}x{original_params.get('height', 'unknown')}"
            new_res = f"{new_params.get('width', 'unknown')}x{new_params.get('height', 'unknown')}"
            orig_bitrate = original_params.get('bitrate', 'unknown')
            new_bitrate = new_params.get('bitrate', 'unknown')
            
            logger.warning(f"{prefix}FALLBACK: Resolution reduction {orig_res} → {new_res}{reason_suffix}")
            logger.info(f"{prefix}Bitrate adjustment: {orig_bitrate}kbps → {new_bitrate}kbps")
            
            # Calculate quality impact
            if (isinstance(original_params.get('width'), int) and isinstance(original_params.get('height'), int) and
                isinstance(new_params.get('width'), int) and isinstance(new_params.get('height'), int)):
                orig_pixels = original_params['width'] * original_params['height']
                new_pixels = new_params['width'] * new_params['height']
                reduction_pct = (1 - new_pixels / orig_pixels) * 100
                logger.info(f"{prefix}Quality impact: {reduction_pct:.1f}% pixel reduction")
            
        elif strategy == 'fps_reduction':
            orig_fps = original_params.get('fps', 'unknown')
            new_fps = new_params.get('fps', 'unknown')
            orig_bitrate = original_params.get('bitrate', 'unknown')
            new_bitrate = new_params.get('bitrate', 'unknown')
            
            logger.warning(f"{prefix}FALLBACK: FPS reduction {orig_fps} → {new_fps}{reason_suffix}")
            logger.info(f"{prefix}Bitrate adjustment: {orig_bitrate}kbps → {new_bitrate}kbps")
            
            # Calculate temporal quality impact
            if (isinstance(orig_fps, (int, float)) and isinstance(new_fps, (int, float)) and orig_fps > 0):
                reduction_pct = (1 - new_fps / orig_fps) * 100
                logger.info(f"{prefix}Quality impact: {reduction_pct:.1f}% frame rate reduction")
            
        elif strategy == 'segmentation':
            logger.warning(f"{prefix}FALLBACK: Video segmentation{reason_suffix}")
            logger.info(f"{prefix}Quality impact: Minimal (preserves original quality in segments)")
            
            # Log segmentation parameters if available
            if 'segment_duration' in new_params:
                logger.info(f"{prefix}Segment duration: {new_params['segment_duration']}s")
            if 'estimated_segments' in new_params:
                logger.info(f"{prefix}Estimated segments: {new_params['estimated_segments']}")
                
        elif strategy == 'encoder_fallback':
            orig_encoder = original_params.get('encoder', 'unknown')
            new_encoder = new_params.get('encoder', 'unknown')
            logger.warning(f"{prefix}FALLBACK: Encoder change {orig_encoder} → {new_encoder}{reason_suffix}")
            
        else:
            logger.warning(f"{prefix}FALLBACK: {strategy}{reason_suffix}")
    
    def log_validation_failure_summary(self, failures: List[ValidationResult], 
                                     context: str = "batch_processing"):
        """Log summary of validation failures for batch processing"""
        if not failures:
            return
            
        total_failures = len(failures)
        critical_failures = sum(1 for f in failures if f.severity == 'critical')
        warning_failures = total_failures - critical_failures
        
        logger.error(f"[{context}] Bitrate validation summary: {total_failures} failures "
                    f"({critical_failures} critical, {warning_failures} warnings)")
        
        # Group failures by encoder
        encoder_failures = {}
        for failure in failures:
            encoder = getattr(failure, 'encoder', 'unknown')
            if encoder not in encoder_failures:
                encoder_failures[encoder] = {'critical': 0, 'warning': 0}
            encoder_failures[encoder][failure.severity] += 1
        
        for encoder, counts in encoder_failures.items():
            logger.error(f"[{context}] {encoder}: {counts['critical']} critical, {counts['warning']} warnings")
        
        # Log common suggestions
        if critical_failures > 0:
            logger.error(f"[{context}] Suggestions for critical failures:")
            logger.error(f"[{context}] • Enable segmentation: --enable-segmentation")
            logger.error(f"[{context}] • Lower bitrate floor: --bitrate-floor 1")
            logger.error(f"[{context}] • Use hardware encoders: --force-hardware")
        
        if warning_failures > 0:
            logger.warning(f"[{context}] Suggestions for warnings:")
            logger.warning(f"[{context}] • Enable auto-adjustment: --enable-auto-adjustment")
            logger.warning(f"[{context}] • Reduce target resolution: --max-width 720")
    
    def create_validation_error(self, result: ValidationResult, encoder: str = "unknown", 
                              context: str = None, video_info: Dict[str, Any] = None) -> BitrateValidationError:
        """Create a BitrateValidationError from validation result with enhanced context"""
        return BitrateValidationError(
            message=result.message,
            bitrate_kbps=result.current_bitrate,
            minimum_required=result.minimum_required,
            encoder=encoder,
            severity=result.severity,
            context=context,
            video_info=video_info
        )
    
    def get_validation_summary(self, results: List[ValidationResult]) -> Dict[str, Any]:
        """Get summary of multiple validation results"""
        total = len(results)
        valid = sum(1 for r in results if r.is_valid)
        warnings = sum(1 for r in results if r.severity == 'warning')
        critical = sum(1 for r in results if r.severity == 'critical')
        
        return {
            'total_validations': total,
            'valid_count': valid,
            'warning_count': warnings,
            'critical_count': critical,
            'success_rate': (valid / total * 100) if total > 0 else 0,
            'all_valid': valid == total
        }
    
    def log_batch_processing_resilience(self, total_files: int, successful: int, 
                                      failed: int, error_types: Dict[str, int]):
        """Log batch processing resilience and error recovery information"""
        if total_files == 0:
            return
            
        success_rate = (successful / total_files) * 100
        
        logger.info(f"=== BATCH PROCESSING RESILIENCE REPORT ===")
        logger.info(f"Total files processed: {total_files}")
        logger.info(f"Success rate: {success_rate:.1f}% ({successful} successful, {failed} failed)")
        
        if failed > 0:
            logger.info(f"Error recovery: Processing continued despite {failed} failures")
            logger.info("Error breakdown by category:")
            
            for error_type, count in error_types.items():
                percentage = (count / failed) * 100
                logger.info(f"  • {error_type}: {count} files ({percentage:.1f}% of failures)")
            
            # Provide actionable recommendations based on error patterns
            if error_types.get('bitrate_validation', 0) > 0:
                logger.info("Recommendation: Consider enabling segmentation for bitrate validation failures")
            if error_types.get('encoder', 0) > 0:
                logger.info("Recommendation: Check FFmpeg installation and try software encoding")
            if error_types.get('memory', 0) > 0:
                logger.info("Recommendation: Reduce video resolution or enable segmentation")
        
        if successful > 0:
            logger.info(f"Batch processing completed successfully with {success_rate:.1f}% success rate")
        else:
            logger.error("Batch processing failed - no files processed successfully")
            logger.error("Check individual file errors and system configuration")