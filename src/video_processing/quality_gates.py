"""
Quality Gates Module
Implements VMAF and SSIM quality measurements for CAE pipeline
"""

import os
import subprocess
import json
import logging
import re
import time
import uuid
import threading
from typing import Dict, Any, Optional, Tuple, List
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class QualityGates:
    """Objective quality measurement using VMAF and SSIM."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        self._vmaf_available = None
        
        # Initialize debug log manager and VMAF filter builder
        self.debug_log_manager = DebugLogManager(config_manager)
        self.vmaf_filter_builder = VMAFFilterBuilder(self.debug_log_manager)
        
        # Load configuration settings
        self.fallback_mode = self._get_fallback_mode()
        self.confidence_thresholds = self._get_confidence_thresholds()
        self.ffmpeg_config = self._get_ffmpeg_config()
        self.logging_config = self._get_logging_config()
        self.ssim_config = self._get_ssim_config()
        self.vmaf_config = self._get_vmaf_config()
        
        # Clean up any legacy debug files from previous versions
        self.cleanup_legacy_debug_files()
    
    def _get_fallback_mode(self) -> str:
        """Get configured fallback mode for quality evaluation failures.
        
        Returns:
            Fallback mode: 'conservative', 'permissive', or 'strict'
        """
        if self.config:
            return self.config.get('quality_evaluation.fallback_mode', 'conservative')
        return 'conservative'
    
    def _get_confidence_thresholds(self) -> Dict[str, float]:
        """Get confidence thresholds for decision making.
        
        Returns:
            Dictionary with confidence thresholds for different decisions
        """
        if self.config:
            return {
                'minimum_acceptable': self.config.get('quality_evaluation.confidence_thresholds.minimum_acceptable', 0.3),
                'high_confidence': self.config.get('quality_evaluation.confidence_thresholds.high_confidence', 0.8),
                'decision_threshold': self.config.get('quality_evaluation.confidence_thresholds.decision_threshold', 0.5)
            }
        return {
            'minimum_acceptable': 0.3,
            'high_confidence': 0.8,
            'decision_threshold': 0.5
        }
    
    def _get_ffmpeg_config(self) -> Dict[str, Any]:
        """Get FFmpeg execution configuration.
        
        Returns:
            Dictionary with FFmpeg execution settings
        """
        if self.config:
            return {
                'timeout_seconds': self.config.get('quality_evaluation.ffmpeg_execution.timeout_seconds', 300),
                'retry_attempts': self.config.get('quality_evaluation.ffmpeg_execution.retry_attempts', 3),
                'retry_delay_seconds': self.config.get('quality_evaluation.ffmpeg_execution.retry_delay_seconds', 2),
                'enable_detailed_errors': self.config.get('quality_evaluation.ffmpeg_execution.enable_detailed_errors', True),
                'capture_performance_metrics': self.config.get('quality_evaluation.ffmpeg_execution.capture_performance_metrics', False)
            }
        return {
            'timeout_seconds': 300,
            'retry_attempts': 3,
            'retry_delay_seconds': 2,
            'enable_detailed_errors': True,
            'capture_performance_metrics': False
        }
    
    def _get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration for quality evaluation.
        
        Returns:
            Dictionary with logging settings
        """
        if self.config:
            return {
                'enable_structured_logging': self.config.get('quality_evaluation.logging.enable_structured_logging', True),
                'log_performance_metrics': self.config.get('quality_evaluation.logging.log_performance_metrics', False),
                'log_parsing_attempts': self.config.get('quality_evaluation.logging.log_parsing_attempts', False),
                'log_confidence_breakdown': self.config.get('quality_evaluation.logging.log_confidence_breakdown', True)
            }
        return {
            'enable_structured_logging': True,
            'log_performance_metrics': False,
            'log_parsing_attempts': False,
            'log_confidence_breakdown': True
        }
    
    def _get_ssim_config(self) -> Dict[str, Any]:
        """Get SSIM parsing configuration.
        
        Returns:
            Dictionary with SSIM parsing settings
        """
        if self.config:
            return {
                'timeout_seconds': self.config.get('quality_evaluation.ssim_parsing.timeout_seconds', 300),
                'retry_attempts': self.config.get('quality_evaluation.ssim_parsing.retry_attempts', 3),
                'alternative_formats': self.config.get('quality_evaluation.ssim_parsing.alternative_formats', True),
                'enable_outlier_detection': self.config.get('quality_evaluation.ssim_parsing.validation.enable_outlier_detection', True),
                'min_samples_for_outlier_detection': self.config.get('quality_evaluation.ssim_parsing.validation.min_samples_for_outlier_detection', 4),
                'iqr_multiplier': self.config.get('quality_evaluation.ssim_parsing.validation.iqr_multiplier', 1.5)
            }
        return {
            'timeout_seconds': 300,
            'retry_attempts': 3,
            'alternative_formats': True,
            'enable_outlier_detection': True,
            'min_samples_for_outlier_detection': 4,
            'iqr_multiplier': 1.5
        }
    
    def _get_vmaf_config(self) -> Dict[str, Any]:
        """Get VMAF parsing configuration.
        
        Returns:
            Dictionary with VMAF parsing settings
        """
        if self.config:
            default_patterns = [
                'VMAF\\s+score[:\\s]+([\\d.]+)',
                'vmaf[:\\s]+([\\d.]+)',
                'mean[:\\s]+([\\d.]+)'
            ]
            return {
                'timeout_seconds': self.config.get('quality_evaluation.vmaf_parsing.timeout_seconds', 300),
                'json_fallback': self.config.get('quality_evaluation.vmaf_parsing.json_fallback', True),
                'prefer_json_format': self.config.get('quality_evaluation.vmaf_parsing.prefer_json_format', True),
                'text_parsing_patterns': self.config.get('quality_evaluation.vmaf_parsing.text_parsing_patterns', default_patterns)
            }
        return {
            'timeout_seconds': 300,
            'json_fallback': True,
            'prefer_json_format': True,
            'text_parsing_patterns': [
                'VMAF\\s+score[:\\s]+([\\d.]+)',
                'vmaf[:\\s]+([\\d.]+)',
                'mean[:\\s]+([\\d.]+)'
            ]
        }
    
    def _generate_thread_safe_session_id(self) -> str:
        """Generate a unique session ID safe for concurrent execution.
        
        Returns:
            Unique session identifier combining timestamp, thread ID, and UUID component
        """
        # Use nanosecond precision timestamp + thread ID + UUID component for guaranteed uniqueness
        # Use time.perf_counter() for higher precision and add a counter-like component
        perf_counter = int(time.perf_counter() * 1000000000)  # nanosecond precision
        thread_id = threading.get_ident()
        uuid_short = uuid.uuid4().hex[:8]  # Use 8 chars for better uniqueness
        return f"{perf_counter}_{thread_id}_{uuid_short}"
    
    def _apply_fallback_behavior(self, result: Dict[str, Any], vmaf_threshold: float, 
                                ssim_threshold: float) -> Dict[str, Any]:
        """Apply fallback behavior when quality evaluation fails or has low confidence.
        
        Args:
            result: Quality evaluation result
            vmaf_threshold: VMAF threshold that was used
            ssim_threshold: SSIM threshold that was used
            
        Returns:
            Updated result with fallback behavior applied
        """
        confidence = result.get('confidence', 0.0)
        evaluation_success = result.get('evaluation_success', False)
        original_passes = result.get('passes', False)
        
        # Determine if fallback behavior should be applied
        needs_fallback = False
        fallback_reason = None
        
        if not evaluation_success:
            needs_fallback = True
            fallback_reason = 'evaluation_failed'
        elif confidence < self.confidence_thresholds['minimum_acceptable']:
            needs_fallback = True
            fallback_reason = 'low_confidence'
        
        if not needs_fallback:
            # No fallback needed, return original result
            result['details']['fallback_applied'] = False
            return result
        
        # Apply fallback behavior based on mode
        fallback_passes = original_passes  # Default to original result
        
        if self.fallback_mode == 'conservative':
            # Conservative: proceed with current parameters when evaluation fails
            # Assume quality is acceptable if we can't measure it properly
            fallback_passes = True
            fallback_decision = 'proceed_with_current_parameters'
            
        elif self.fallback_mode == 'permissive':
            # Permissive: assume quality passes when evaluation fails
            # Most lenient approach - always pass quality gates on failure
            fallback_passes = True
            fallback_decision = 'assume_quality_passes'
            
        elif self.fallback_mode == 'strict':
            # Strict: fail quality gates when evaluation fails
            # Most conservative approach - require successful evaluation
            fallback_passes = False
            fallback_decision = 'fail_on_evaluation_failure'
        
        else:
            # Unknown mode, default to conservative
            logger.warning(f"Unknown fallback mode '{self.fallback_mode}', using conservative")
            fallback_passes = True
            fallback_decision = 'default_conservative'
        
        # Update result with fallback information
        result['passes'] = fallback_passes
        result['details']['fallback_applied'] = True
        result['details']['fallback_mode'] = self.fallback_mode
        result['details']['fallback_reason'] = fallback_reason
        result['details']['fallback_decision'] = fallback_decision
        result['details']['original_passes'] = original_passes
        result['details']['confidence_threshold'] = self.confidence_thresholds['minimum_acceptable']
        
        # Log fallback behavior
        logger.info(f"Quality evaluation fallback applied: mode={self.fallback_mode}, "
                   f"reason={fallback_reason}, confidence={confidence:.3f}, "
                   f"original_passes={original_passes}, fallback_passes={fallback_passes}")
        
        return result
    
    def _log_quality_evaluation_start(self, original_path: str, compressed_path: str, 
                                     vmaf_threshold: float, ssim_threshold: float,
                                     eval_height: Optional[int] = None) -> None:
        """Log structured information about quality evaluation start."""
        logger.info(f"Starting quality evaluation: {os.path.basename(compressed_path)}")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Quality evaluation parameters:")
            logger.debug(f"  Original: {original_path}")
            logger.debug(f"  Compressed: {compressed_path}")
            logger.debug(f"  VMAF threshold: {vmaf_threshold}")
            logger.debug(f"  SSIM threshold: {ssim_threshold}")
            logger.debug(f"  Eval height: {eval_height or 'native'}")
            logger.debug(f"  Fallback mode: {self.fallback_mode}")
    
    def _log_quality_evaluation_result(self, result: Dict[str, Any]) -> None:
        """Log structured quality evaluation results with comprehensive performance metrics and scaling information."""
        vmaf_str, ssim_str = self._safe_format_scores(result['vmaf_score'], result['ssim_score'])
        
        # Basic result logging with enhanced information
        logger.info(f"Quality eval: method={result['method']}, "
                   f"VMAF={vmaf_str}, SSIM={ssim_str}, "
                   f"passes={result['passes']}, confidence={result['confidence']:.3f}, "
                   f"eval_success={result['evaluation_success']}")
        
        # Log resolution scaling information if present
        if result.get('scaling_applied'):
            orig_res = result.get('original_resolution', 'unknown')
            comp_res = result.get('comparison_resolution', 'unknown')
            logger.info(f"Resolution scaling applied: {orig_res} → {comp_res}")
        
        # Enhanced structured logging for quality evaluation failures
        if not result['evaluation_success']:
            self._log_quality_evaluation_failure(result)
        
        # Debug-level detailed logging with comprehensive metrics
        if logger.isEnabledFor(logging.DEBUG):
            self._log_detailed_quality_evaluation_results(result)
    
    def _log_quality_evaluation_failure(self, result: Dict[str, Any]) -> None:
        """Log detailed error messages for quality evaluation failures with structured information.
        
        Args:
            result: Quality evaluation result dictionary
        """
        details = result.get('details', {})
        error_msg = details.get('error', 'Unknown error')
        
        logger.error("=== QUALITY EVALUATION FAILURE ===")
        logger.error(f"Method attempted: {result.get('method', 'unknown')}")
        logger.error(f"Error: {error_msg}")
        
        # Log resolution-specific failure information
        if result.get('scaling_applied'):
            logger.error("Resolution scaling information:")
            logger.error(f"  Original resolution: {result.get('original_resolution', 'unknown')}")
            logger.error(f"  Target resolution: {result.get('comparison_resolution', 'unknown')}")
            logger.error(f"  Scaling applied: {result.get('scaling_applied', False)}")
        
        # Log specific failure types with detailed context
        if 'ssim' in error_msg.lower() and 'resolution' in error_msg.lower():
            self._log_ssim_resolution_mismatch_error(result, details)
        elif 'vmaf' in error_msg.lower():
            self._log_vmaf_computation_error(result, details)
        
        # Log confidence scoring context
        confidence_breakdown = details.get('confidence_breakdown', {})
        if confidence_breakdown:
            logger.error("Confidence breakdown:")
            for metric, available in confidence_breakdown.items():
                logger.error(f"  {metric}: {available}")
        
        # Log fallback behavior if applied
        if details.get('fallback_applied'):
            logger.error(f"Fallback behavior: {details.get('fallback_mode', 'unknown')} "
                        f"(reason: {details.get('fallback_reason', 'unknown')})")
        
        logger.error("=== END QUALITY EVALUATION FAILURE ===")
    
    def _log_ssim_resolution_mismatch_error(self, result: Dict[str, Any], details: Dict[str, Any]) -> None:
        """Log detailed error messages for SSIM resolution mismatches.
        
        Args:
            result: Quality evaluation result
            details: Error details dictionary
        """
        logger.error("SSIM RESOLUTION MISMATCH DETAILS:")
        
        # Log resolution information
        orig_res = result.get('original_resolution')
        comp_res = result.get('comparison_resolution')
        
        if orig_res and comp_res:
            orig_w, orig_h = orig_res
            comp_w, comp_h = comp_res
            
            logger.error(f"  Original video: {orig_w}x{orig_h} ({orig_w * orig_h:,} pixels)")
            logger.error(f"  Compressed video: {comp_w}x{comp_h} ({comp_w * comp_h:,} pixels)")
            
            # Calculate aspect ratios
            orig_aspect = orig_w / orig_h if orig_h > 0 else 0
            comp_aspect = comp_w / comp_h if comp_h > 0 else 0
            
            logger.error(f"  Aspect ratios: {orig_aspect:.3f} vs {comp_aspect:.3f}")
            
            # Determine resolution difference type
            if orig_w != comp_w or orig_h != comp_h:
                if orig_aspect != comp_aspect:
                    logger.error("  Issue: Both resolution and aspect ratio mismatch")
                    logger.error("  Recommendation: Check video encoding settings for aspect ratio preservation")
                else:
                    logger.error("  Issue: Resolution mismatch with same aspect ratio")
                    logger.error("  Recommendation: Videos can be scaled for comparison")
            
            # Log scaling attempt results if available
            if result.get('scaling_applied'):
                target_res = result.get('comparison_resolution')
                if target_res:
                    target_w, target_h = target_res
                    logger.error(f"  Scaling attempted to: {target_w}x{target_h}")
                    logger.error("  Scaling failed - check scaling implementation")
            else:
                logger.error("  No scaling attempted - automatic scaling may resolve this issue")
        
        # Log recovery suggestions
        logger.error("RECOVERY SUGGESTIONS:")
        logger.error("  1. Enable automatic resolution scaling")
        logger.error("  2. Pre-scale videos to matching resolution")
        logger.error("  3. Use VMAF instead of SSIM (handles resolution differences better)")
        logger.error("  4. Check video encoding pipeline for resolution consistency")
    
    def _log_vmaf_computation_error(self, result: Dict[str, Any], details: Dict[str, Any]) -> None:
        """Log detailed error messages for VMAF computation failures.
        
        Args:
            result: Quality evaluation result
            details: Error details dictionary
        """
        logger.error("VMAF COMPUTATION ERROR DETAILS:")
        
        error_msg = details.get('error', '')
        
        # Categorize VMAF error types
        if 'not available' in error_msg.lower() or 'not found' in error_msg.lower():
            logger.error("  Issue: VMAF filter not available in FFmpeg")
            logger.error("  Cause: FFmpeg not compiled with libvmaf support")
            logger.error("  Recovery: Install FFmpeg with libvmaf or use SSIM fallback")
        elif 'timeout' in error_msg.lower():
            logger.error("  Issue: VMAF computation timed out")
            logger.error("  Cause: Video too complex or system overloaded")
            logger.error("  Recovery: Increase timeout, reduce resolution, or use fewer threads")
        elif 'parameter' in error_msg.lower() or 'syntax' in error_msg.lower():
            logger.error("  Issue: VMAF filter parameter error")
            logger.error("  Cause: Malformed filter parameters or path escaping issues")
            logger.error("  Recovery: Check filter parameter formatting and file paths")
        elif 'model' in error_msg.lower():
            logger.error("  Issue: VMAF model error")
            logger.error("  Cause: Model file not found or corrupted")
            logger.error("  Recovery: Use default model or check model installation")
        else:
            logger.error(f"  Issue: Generic VMAF error - {error_msg}")
            logger.error("  Recovery: Check FFmpeg logs and video compatibility")
        
        # Log resolution context for VMAF
        if result.get('original_resolution') and result.get('comparison_resolution'):
            orig_res = result['original_resolution']
            comp_res = result['comparison_resolution']
            logger.error(f"  Video resolutions: {orig_res} vs {comp_res}")
            
            if orig_res != comp_res:
                logger.error("  Note: Resolution mismatch may contribute to VMAF failure")
                logger.error("  Suggestion: Enable automatic resolution scaling")
    
    def _log_detailed_quality_evaluation_results(self, result: Dict[str, Any]) -> None:
        """Log comprehensive debug-level quality evaluation results.
        
        Args:
            result: Quality evaluation result dictionary
        """
        details = result.get('details', {})
        
        logger.debug("=== DETAILED QUALITY EVALUATION RESULTS ===")
        logger.debug(f"Evaluation method: {result.get('method', 'unknown')}")
        logger.debug(f"Overall success: {result['evaluation_success']}")
        logger.debug(f"Quality passes: {result['passes']}")
        logger.debug(f"Confidence score: {result['confidence']:.4f}")
        
        # Log individual metric results
        if result.get('vmaf_score') is not None:
            vmaf_pass = details.get('vmaf_pass', False)
            vmaf_threshold = details.get('vmaf_threshold', 'N/A')
            vmaf_confidence = details.get('vmaf_confidence', 0.0)
            logger.debug(f"VMAF: {result['vmaf_score']:.2f} (threshold: {vmaf_threshold}, "
                        f"pass: {vmaf_pass}, confidence: {vmaf_confidence:.3f})")
        
        if result.get('ssim_score') is not None:
            ssim_pass = details.get('ssim_pass', False)
            ssim_threshold = details.get('ssim_threshold', 'N/A')
            ssim_confidence = details.get('ssim_confidence', 0.0)
            logger.debug(f"SSIM: {result['ssim_score']:.4f} (threshold: {ssim_threshold}, "
                        f"pass: {ssim_pass}, confidence: {ssim_confidence:.3f})")
        
        # Log resolution and scaling information
        if result.get('scaling_applied'):
            logger.debug("Resolution scaling details:")
            logger.debug(f"  Original resolution: {result.get('original_resolution', 'unknown')}")
            logger.debug(f"  Comparison resolution: {result.get('comparison_resolution', 'unknown')}")
            logger.debug(f"  Scaling applied: {result['scaling_applied']}")
        
        # Log confidence breakdown
        confidence_breakdown = details.get('confidence_breakdown', {})
        if confidence_breakdown:
            logger.debug("Confidence breakdown:")
            for component, value in confidence_breakdown.items():
                logger.debug(f"  {component}: {value}")
        
        # Log fallback information
        if details.get('fallback_applied'):
            logger.debug("Fallback behavior:")
            logger.debug(f"  Mode: {details.get('fallback_mode', 'unknown')}")
            logger.debug(f"  Reason: {details.get('fallback_reason', 'unknown')}")
            logger.debug(f"  Decision: {details.get('fallback_decision', 'unknown')}")
            logger.debug(f"  Original result: {details.get('original_passes', 'unknown')}")
        
        # Log evaluation timing if available
        if 'evaluation_time' in details:
            logger.debug(f"Evaluation time: {details['evaluation_time']:.3f}s")
        
        # Log any additional context
        if 'error' in details:
            logger.debug(f"Error details: {details['error']}")
        
        logger.debug("=== END DETAILED RESULTS ===")
    
    def _calculate_confidence_score_with_logging(self, vmaf_score: Optional[float], ssim_score: Optional[float], 
                                               vmaf_confidence: float = 1.0, ssim_confidence: float = 1.0,
                                               method: str = 'error') -> Tuple[float, Dict[str, Any]]:
        """Calculate confidence score with detailed logging of the calculation process.
        
        Args:
            vmaf_score: VMAF score (0-100) or None
            ssim_score: SSIM score (0-1) or None
            vmaf_confidence: Confidence in VMAF parsing (0-1)
            ssim_confidence: Confidence in SSIM parsing (0-1)
            method: Evaluation method used
            
        Returns:
            Tuple of (overall_confidence_score, confidence_breakdown_dict)
        """
        confidence_breakdown = {
            'vmaf_available': vmaf_score is not None,
            'ssim_available': ssim_score is not None,
            'vmaf_confidence': vmaf_confidence,
            'ssim_confidence': ssim_confidence,
            'method': method,
            'calculation_details': {}
        }
        
        if method == 'error':
            confidence_breakdown['calculation_details']['reason'] = 'evaluation_failed'
            return 0.0, confidence_breakdown
        
        # Calculate confidence based on method and available metrics
        if method == 'vmaf+ssim':
            if vmaf_score is not None and ssim_score is not None:
                # Both metrics available - high confidence
                base_confidence = (vmaf_confidence + ssim_confidence) / 2
                confidence_breakdown['calculation_details'] = {
                    'reason': 'dual_metrics_available',
                    'base_confidence': base_confidence,
                    'vmaf_weight': 0.5,
                    'ssim_weight': 0.5
                }
                final_confidence = base_confidence
            elif vmaf_score is not None:
                # Only VMAF available - moderate confidence
                base_confidence = vmaf_confidence * 0.8
                confidence_breakdown['calculation_details'] = {
                    'reason': 'vmaf_only_in_dual_mode',
                    'base_confidence': vmaf_confidence,
                    'penalty_factor': 0.8,
                    'penalty_reason': 'missing_ssim_metric'
                }
                final_confidence = base_confidence
            elif ssim_score is not None:
                # Only SSIM available - lower confidence
                base_confidence = ssim_confidence * 0.6
                confidence_breakdown['calculation_details'] = {
                    'reason': 'ssim_only_in_dual_mode',
                    'base_confidence': ssim_confidence,
                    'penalty_factor': 0.6,
                    'penalty_reason': 'missing_vmaf_metric'
                }
                final_confidence = base_confidence
            else:
                # No metrics available - no confidence
                confidence_breakdown['calculation_details']['reason'] = 'no_metrics_available'
                final_confidence = 0.0
        
        elif method == 'ssim_only':
            if ssim_score is not None:
                # SSIM-only mode with successful parsing
                base_confidence = ssim_confidence * 0.9
                confidence_breakdown['calculation_details'] = {
                    'reason': 'ssim_only_mode',
                    'base_confidence': ssim_confidence,
                    'mode_factor': 0.9,
                    'mode_reason': 'single_metric_mode'
                }
                final_confidence = base_confidence
            else:
                confidence_breakdown['calculation_details']['reason'] = 'ssim_parsing_failed'
                final_confidence = 0.0
        
        else:
            confidence_breakdown['calculation_details']['reason'] = f'unknown_method_{method}'
            final_confidence = 0.0
        
        # Log confidence calculation if debug logging is enabled
        if logger.isEnabledFor(logging.DEBUG) and self.logging_config.get('log_confidence_breakdown', True):
            self._log_confidence_calculation(final_confidence, confidence_breakdown)
        
        return final_confidence, confidence_breakdown
    
    def _log_confidence_calculation(self, final_confidence: float, breakdown: Dict[str, Any]) -> None:
        """Log detailed confidence score calculation for debugging.
        
        Args:
            final_confidence: Final calculated confidence score
            breakdown: Confidence calculation breakdown
        """
        logger.debug("=== CONFIDENCE SCORE CALCULATION ===")
        logger.debug(f"Final confidence: {final_confidence:.4f}")
        logger.debug(f"Method: {breakdown['method']}")
        logger.debug(f"VMAF available: {breakdown['vmaf_available']}")
        logger.debug(f"SSIM available: {breakdown['ssim_available']}")
        
        calc_details = breakdown.get('calculation_details', {})
        if calc_details:
            logger.debug("Calculation details:")
            for key, value in calc_details.items():
                logger.debug(f"  {key}: {value}")
        
        logger.debug("=== END CONFIDENCE CALCULATION ===")
    
    def _add_confidence_scoring_to_result(self, result: Dict[str, Any], 
                                        vmaf_confidence: float = 1.0, 
                                        ssim_confidence: float = 1.0) -> Dict[str, Any]:
        """Add confidence scoring with detailed breakdown to quality evaluation result.
        
        Args:
            result: Quality evaluation result dictionary
            vmaf_confidence: VMAF parsing confidence
            ssim_confidence: SSIM parsing confidence
            
        Returns:
            Updated result with confidence scoring
        """
        # Calculate confidence with detailed logging
        confidence, confidence_breakdown = self._calculate_confidence_score_with_logging(
            result.get('vmaf_score'),
            result.get('ssim_score'),
            vmaf_confidence,
            ssim_confidence,
            result.get('method', 'error')
        )
        
        # Update result with confidence information
        result['confidence'] = confidence
        
        # Ensure details dictionary exists
        if 'details' not in result:
            result['details'] = {}
        
        # Add detailed confidence breakdown
        result['details']['confidence_breakdown'] = confidence_breakdown
        result['details']['vmaf_confidence'] = vmaf_confidence
        result['details']['ssim_confidence'] = ssim_confidence
        
        return result
    
    def _log_parsing_attempt(self, metric_name: str, strategy: str, success: bool, 
                           details: Optional[str] = None) -> None:
        """Log parsing attempt for debugging purposes."""
        if (self.logging_config['log_parsing_attempts'] and 
            logger.isEnabledFor(logging.DEBUG)):
            status = "SUCCESS" if success else "FAILED"
            log_msg = f"{metric_name} parsing {strategy}: {status}"
            if details:
                log_msg += f" - {details}"
            logger.debug(log_msg)
    
    def _log_performance_metrics(self, operation: str, execution_time: float, 
                               additional_metrics: Optional[Dict[str, Any]] = None) -> None:
        """Log performance metrics for quality computations."""
        if (self.ffmpeg_config['capture_performance_metrics'] or 
            self.logging_config['log_performance_metrics']):
            logger.info(f"Performance: {operation} completed in {execution_time:.2f}s")
            
            if (additional_metrics and 
                self.logging_config['log_performance_metrics'] and 
                logger.isEnabledFor(logging.DEBUG)):
                for key, value in additional_metrics.items():
                    logger.debug(f"  {key}: {value}")
    
    def _create_diagnostic_output(self, operation: str, cmd: List[str], 
                                 exec_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create diagnostic output for troubleshooting parsing issues."""
        diagnostic = {
            'operation': operation,
            'command': ' '.join(cmd),
            'success': exec_result['success'],
            'execution_time': exec_result['execution_time'],
            'attempts': exec_result['attempts'],
            'returncode': exec_result['returncode']
        }
        
        if not exec_result['success']:
            diagnostic['error_details'] = exec_result['error_details']
        
        # Include stderr/stdout for debugging if enabled
        if logger.isEnabledFor(logging.DEBUG):
            diagnostic['stderr_length'] = len(exec_result['stderr'])
            diagnostic['stdout_length'] = len(exec_result['stdout'])
            
            # Include first/last few lines of output for context
            stderr_lines = exec_result['stderr'].split('\n')
            if len(stderr_lines) > 10:
                diagnostic['stderr_preview'] = {
                    'first_5_lines': stderr_lines[:5],
                    'last_5_lines': stderr_lines[-5:]
                }
            else:
                diagnostic['stderr_preview'] = stderr_lines
        
        return diagnostic
    
    def _execute_ffmpeg_with_retry(self, cmd: List[str], operation_name: str, 
                                  timeout_override: Optional[int] = None) -> Dict[str, Any]:
        """Execute FFmpeg command with robust error handling and retry logic.
        
        Args:
            cmd: FFmpeg command as list of strings
            operation_name: Human-readable name for the operation (for logging)
            timeout_override: Optional timeout override in seconds
            
        Returns:
            Dictionary with execution results:
                - success: bool
                - stdout: str
                - stderr: str
                - returncode: int
                - execution_time: float
                - attempts: int
                - error_details: Dict[str, Any]
                - complete_ffmpeg_output: str (for enhanced diagnostics)
        """
        timeout = timeout_override or self.ffmpeg_config['timeout_seconds']
        max_attempts = self.ffmpeg_config['retry_attempts']
        retry_delay = self.ffmpeg_config['retry_delay_seconds']
        
        result = {
            'success': False,
            'stdout': '',
            'stderr': '',
            'returncode': -1,
            'execution_time': 0.0,
            'attempts': 0,
            'error_details': {},
            'complete_ffmpeg_output': ''  # Enhanced diagnostics
        }
        
        # Track all attempts for comprehensive error reporting
        all_attempts = []
        
        for attempt in range(1, max_attempts + 1):
            result['attempts'] = attempt
            start_time = time.time()
            attempt_info = {
                'attempt_number': attempt,
                'start_time': start_time,
                'timeout_used': timeout
            }
            
            try:
                logger.debug(f"FFmpeg {operation_name} attempt {attempt}/{max_attempts}: {' '.join(cmd[:3])}...")
                
                # Enhanced process execution with better error capture
                process_result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    encoding='utf-8',
                    errors='replace'
                )
                
                execution_time = time.time() - start_time
                result['execution_time'] = execution_time
                result['stdout'] = process_result.stdout
                result['stderr'] = process_result.stderr
                result['returncode'] = process_result.returncode
                
                # Capture complete output for diagnostics
                complete_output = f"=== STDOUT ===\n{process_result.stdout}\n=== STDERR ===\n{process_result.stderr}"
                result['complete_ffmpeg_output'] = complete_output
                
                # Update attempt info
                attempt_info.update({
                    'execution_time': execution_time,
                    'returncode': process_result.returncode,
                    'stdout_length': len(process_result.stdout),
                    'stderr_length': len(process_result.stderr),
                    'success': process_result.returncode == 0
                })
                
                if self.ffmpeg_config['capture_performance_metrics']:
                    logger.debug(f"FFmpeg {operation_name} completed in {execution_time:.2f}s")
                
                if process_result.returncode == 0:
                    result['success'] = True
                    attempt_info['final_result'] = 'success'
                    all_attempts.append(attempt_info)
                    
                    if attempt > 1:
                        logger.info(f"FFmpeg {operation_name} succeeded on attempt {attempt}")
                    return result
                else:
                    # Enhanced error analysis with more context
                    error_analysis = self._analyze_ffmpeg_error_enhanced(
                        process_result.stderr, 
                        process_result.stdout,
                        process_result.returncode,
                        operation_name,
                        attempt
                    )
                    result['error_details'] = error_analysis
                    attempt_info['error_analysis'] = error_analysis
                    
                    if not error_analysis['is_retryable']:
                        logger.warning(f"FFmpeg {operation_name} failed with non-retryable error: {error_analysis['category']}")
                        attempt_info['final_result'] = 'non_retryable_failure'
                        all_attempts.append(attempt_info)
                        break
                    
                    if attempt < max_attempts:
                        logger.warning(f"FFmpeg {operation_name} attempt {attempt} failed (retryable): {error_analysis['category']}")
                        attempt_info['final_result'] = 'retryable_failure'
                        all_attempts.append(attempt_info)
                        time.sleep(retry_delay)
                    else:
                        logger.error(f"FFmpeg {operation_name} failed after {max_attempts} attempts")
                        attempt_info['final_result'] = 'final_failure'
                        all_attempts.append(attempt_info)
                
            except subprocess.TimeoutExpired as e:
                execution_time = time.time() - start_time
                result['execution_time'] = execution_time
                
                # Enhanced timeout error details with recovery mechanisms
                timeout_details = {
                    'category': 'timeout',
                    'is_retryable': True,
                    'severity': 'medium',
                    'description': f'Operation timed out after {timeout} seconds',
                    'suggested_action': 'Increase timeout or reduce video complexity',
                    'timeout_seconds': timeout,
                    'operation_name': operation_name,
                    'attempt_number': attempt,
                    'partial_stdout': getattr(e, 'stdout', '') or '',
                    'partial_stderr': getattr(e, 'stderr', '') or '',
                    'recovery_mechanisms': ['increase_timeout', 'reduce_resolution', 'use_fewer_threads', 'split_video_segments'],
                    'timeout_analysis': self._analyze_timeout_cause(operation_name, timeout, execution_time, e)
                }
                result['error_details'] = timeout_details
                
                # Update attempt info for timeout
                attempt_info.update({
                    'execution_time': execution_time,
                    'error_type': 'timeout',
                    'timeout_seconds': timeout,
                    'final_result': 'timeout'
                })
                
                logger.warning(f"FFmpeg {operation_name} timed out after {timeout}s on attempt {attempt}")
                
                if attempt < max_attempts:
                    # Increase timeout for retry with better progression
                    old_timeout = timeout
                    timeout = min(int(timeout * 1.5), 600)  # Cap at 10 minutes
                    logger.info(f"Increasing timeout from {old_timeout}s to {timeout}s for retry")
                    attempt_info['timeout_increased_to'] = timeout
                    all_attempts.append(attempt_info)
                    time.sleep(retry_delay)
                else:
                    logger.error(f"FFmpeg {operation_name} timed out after {max_attempts} attempts")
                    all_attempts.append(attempt_info)
                
            except Exception as e:
                execution_time = time.time() - start_time
                result['execution_time'] = execution_time
                
                # Enhanced system error details
                system_error_details = {
                    'category': 'system_error',
                    'is_retryable': False,
                    'description': str(e),
                    'exception_type': type(e).__name__,
                    'operation_name': operation_name,
                    'attempt_number': attempt
                }
                result['error_details'] = system_error_details
                
                # Update attempt info for system error
                attempt_info.update({
                    'execution_time': execution_time,
                    'error_type': 'system_error',
                    'exception_type': type(e).__name__,
                    'exception_message': str(e),
                    'final_result': 'system_error'
                })
                
                logger.error(f"FFmpeg {operation_name} system error on attempt {attempt}: {e}")
                all_attempts.append(attempt_info)
                break
        
        # Add comprehensive attempt history to result for enhanced diagnostics
        result['attempt_history'] = all_attempts
        
        # Log comprehensive failure summary if all attempts failed
        if not result['success']:
            self._log_comprehensive_ffmpeg_failure(operation_name, cmd, result, all_attempts)
        
        return result
    
    def _analyze_timeout_cause(self, operation_name: str, timeout_seconds: int, 
                              execution_time: float, timeout_exception: subprocess.TimeoutExpired) -> Dict[str, Any]:
        """Analyze timeout cause and provide specific recommendations.
        
        Args:
            operation_name: Name of the operation that timed out
            timeout_seconds: Configured timeout value
            execution_time: Actual execution time before timeout
            timeout_exception: The timeout exception object
            
        Returns:
            Dictionary with timeout analysis and recommendations
        """
        analysis = {
            'timeout_type': 'unknown',
            'likely_cause': 'unknown',
            'recommended_timeout': timeout_seconds * 2,  # Default: double the timeout
            'alternative_approaches': []
        }
        
        # Analyze timeout characteristics
        timeout_ratio = execution_time / timeout_seconds if timeout_seconds > 0 else 1.0
        
        if 'vmaf' in operation_name.lower():
            analysis['timeout_type'] = 'vmaf_computation'
            if timeout_ratio >= 0.95:  # Very close to timeout
                analysis['likely_cause'] = 'complex_video_analysis'
                analysis['recommended_timeout'] = min(timeout_seconds * 3, 1800)  # Cap at 30 minutes
                analysis['alternative_approaches'] = [
                    'reduce_evaluation_resolution',
                    'use_fewer_vmaf_threads',
                    'split_video_into_segments',
                    'fallback_to_ssim_only'
                ]
            else:
                analysis['likely_cause'] = 'system_resource_contention'
                analysis['recommended_timeout'] = timeout_seconds * 2
                analysis['alternative_approaches'] = [
                    'reduce_concurrent_operations',
                    'increase_system_priority',
                    'free_system_memory'
                ]
        elif 'ssim' in operation_name.lower():
            analysis['timeout_type'] = 'ssim_computation'
            analysis['likely_cause'] = 'video_processing_complexity'
            analysis['recommended_timeout'] = min(timeout_seconds * 2, 900)  # Cap at 15 minutes
            analysis['alternative_approaches'] = [
                'reduce_evaluation_resolution',
                'use_frame_sampling',
                'simplify_ssim_computation'
            ]
        else:
            analysis['timeout_type'] = 'general_ffmpeg'
            analysis['likely_cause'] = 'ffmpeg_processing_complexity'
            analysis['recommended_timeout'] = timeout_seconds * 2
            analysis['alternative_approaches'] = [
                'use_hardware_acceleration',
                'reduce_output_quality',
                'use_faster_preset'
            ]
        
        # Add partial output analysis if available
        partial_stderr = getattr(timeout_exception, 'stderr', '') or ''
        partial_stdout = getattr(timeout_exception, 'stdout', '') or ''
        
        if partial_stderr or partial_stdout:
            analysis['partial_output_available'] = True
            analysis['partial_output_length'] = len(partial_stderr) + len(partial_stdout)
            
            # Look for progress indicators in partial output
            combined_partial = f"{partial_stderr} {partial_stdout}".lower()
            if 'frame=' in combined_partial or 'time=' in combined_partial:
                analysis['progress_detected'] = True
                analysis['likely_cause'] = 'slow_but_progressing'
            else:
                analysis['progress_detected'] = False
                analysis['likely_cause'] = 'stuck_or_hanging'
        else:
            analysis['partial_output_available'] = False
        
        return analysis
    
    def _analyze_ffmpeg_error(self, stderr: str, returncode: int) -> Dict[str, Any]:
        """Analyze FFmpeg error output to categorize and determine retry strategy.
        
        Args:
            stderr: FFmpeg stderr output
            returncode: Process return code
            
        Returns:
            Dictionary with error analysis:
                - category: str (error category)
                - is_retryable: bool (whether retry might help)
                - description: str (human-readable description)
                - specific_error: Optional[str] (specific error if identified)
        """
        return self._analyze_ffmpeg_error_enhanced(stderr, '', returncode, 'unknown', 1)
    
    def _analyze_ffmpeg_error_enhanced(self, stderr: str, stdout: str, returncode: int, 
                                     operation_name: str, attempt_number: int) -> Dict[str, Any]:
        """Enhanced FFmpeg error analysis with comprehensive categorization and context.
        
        Args:
            stderr: FFmpeg stderr output
            stdout: FFmpeg stdout output
            returncode: Process return code
            operation_name: Name of the operation being performed
            attempt_number: Current attempt number
            
        Returns:
            Dictionary with enhanced error analysis:
                - category: str (error category)
                - is_retryable: bool (whether retry might help)
                - description: str (human-readable description)
                - specific_error: Optional[str] (specific error if identified)
                - severity: str (critical, high, medium, low)
                - suggested_action: str (recommended action)
                - context: Dict[str, Any] (additional context)
                - recovery_mechanisms: List[str] (possible recovery strategies)
        """
        stderr_lower = stderr.lower()
        stdout_lower = stdout.lower()
        combined_output = f"{stderr} {stdout}".lower()
        
        # Enhanced error patterns with severity and suggested actions
        error_patterns = [
            # Critical non-retryable errors (file/format issues)
            {
                'patterns': ['no such file', 'file not found', 'cannot open'],
                'category': 'file_not_found',
                'is_retryable': False,
                'severity': 'critical',
                'description': 'Input file not found or inaccessible',
                'suggested_action': 'Verify file path and permissions',
                'recovery_mechanisms': ['verify_file_exists', 'check_permissions', 'use_absolute_path']
            },
            {
                'patterns': ['invalid data found', 'invalid argument', 'unsupported format'],
                'category': 'invalid_format',
                'is_retryable': False,
                'severity': 'critical',
                'description': 'Invalid or unsupported file format',
                'suggested_action': 'Check file format compatibility',
                'recovery_mechanisms': ['validate_file_format', 'try_alternative_decoder', 'convert_input_format']
            },
            {
                'patterns': ['permission denied', 'access denied'],
                'category': 'permission_error',
                'is_retryable': False,
                'severity': 'high',
                'description': 'File permission or access error',
                'suggested_action': 'Check file and directory permissions',
                'recovery_mechanisms': ['fix_file_permissions', 'use_different_output_location', 'run_with_elevated_privileges']
            },
            {
                'patterns': ['no space left', 'disk full'],
                'category': 'disk_space',
                'is_retryable': False,
                'severity': 'high',
                'description': 'Insufficient disk space',
                'suggested_action': 'Free up disk space or use different output location',
                'recovery_mechanisms': ['cleanup_temp_files', 'use_different_drive', 'reduce_output_quality']
            },
            
            # VMAF-specific errors (enhanced categorization)
            {
                'patterns': ['libvmaf.*not found', 'vmaf.*not available', 'unknown filter.*vmaf'],
                'category': 'vmaf_not_available',
                'is_retryable': False,
                'severity': 'critical',
                'description': 'VMAF filter not available in FFmpeg build',
                'suggested_action': 'Install FFmpeg with libvmaf support or use alternative quality metrics',
                'recovery_mechanisms': ['fallback_to_ssim', 'install_libvmaf', 'use_external_vmaf_tool']
            },
            {
                'patterns': ['vmaf.*model.*not found', 'vmaf.*model.*error'],
                'category': 'vmaf_model_error',
                'is_retryable': True,
                'severity': 'high',
                'description': 'VMAF model file not found or corrupted',
                'suggested_action': 'Check VMAF model installation or use default model',
                'recovery_mechanisms': ['use_default_model', 'download_vmaf_models', 'specify_model_path']
            },
            {
                'patterns': ['vmaf.*resolution.*mismatch', 'vmaf.*dimension.*error'],
                'category': 'vmaf_resolution_mismatch',
                'is_retryable': True,
                'severity': 'medium',
                'description': 'Video resolution mismatch for VMAF computation',
                'suggested_action': 'Ensure both videos have compatible resolutions',
                'recovery_mechanisms': ['auto_scale_videos', 'use_smaller_resolution', 'force_resolution_match']
            },
            # VMAF filter parameter errors (NEW - Enhanced for task 9.1)
            {
                'patterns': ['no option name near', 'invalid filter.*vmaf', 'malformed.*vmaf', 'syntax error.*vmaf'],
                'category': 'vmaf_filter_parameter_error',
                'is_retryable': True,
                'severity': 'high',
                'description': 'VMAF filter parameter syntax error',
                'suggested_action': 'Check VMAF filter parameter formatting and escaping',
                'recovery_mechanisms': ['regenerate_filter_params', 'escape_special_characters', 'use_simplified_params', 'validate_filter_syntax']
            },
            {
                'patterns': ['log_path.*error', 'cannot.*write.*log', 'log.*permission'],
                'category': 'vmaf_log_path_error',
                'is_retryable': True,
                'severity': 'medium',
                'description': 'VMAF log file path or permission error',
                'suggested_action': 'Check log file path and directory permissions',
                'recovery_mechanisms': ['create_log_directory', 'use_temp_log_location', 'fix_log_permissions', 'disable_vmaf_logging']
            },
            {
                'patterns': ['vmaf.*timeout', 'vmaf.*killed', 'vmaf.*interrupted'],
                'category': 'vmaf_computation_timeout',
                'is_retryable': True,
                'severity': 'medium',
                'description': 'VMAF computation timed out or was interrupted',
                'suggested_action': 'Increase timeout or reduce video complexity',
                'recovery_mechanisms': ['increase_timeout', 'reduce_resolution', 'use_fewer_threads', 'split_video_segments']
            },
            {
                'patterns': ['libvmaf', 'vmaf filter'],
                'category': 'vmaf_filter_error',
                'is_retryable': True,
                'severity': 'medium',
                'description': 'VMAF filter execution error',
                'suggested_action': 'Check video compatibility and try with different parameters',
                'recovery_mechanisms': ['retry_with_different_params', 'fallback_to_ssim', 'use_alternative_quality_metric']
            },
            
            # Resource-related retryable errors
            {
                'patterns': ['resource temporarily unavailable', 'device busy'],
                'category': 'resource_busy',
                'is_retryable': True,
                'severity': 'medium',
                'description': 'System resources temporarily unavailable',
                'suggested_action': 'Wait and retry, or reduce concurrent operations',
                'recovery_mechanisms': ['wait_and_retry', 'reduce_thread_count', 'queue_operation', 'use_lower_priority']
            },
            {
                'patterns': ['connection refused', 'network unreachable', 'timeout'],
                'category': 'network_error',
                'is_retryable': True,
                'severity': 'medium',
                'description': 'Network connectivity issue',
                'suggested_action': 'Check network connection and retry',
                'recovery_mechanisms': ['retry_with_backoff', 'use_local_resources', 'check_network_connectivity']
            },
            {
                'patterns': ['memory allocation failed', 'out of memory'],
                'category': 'memory_error',
                'is_retryable': True,
                'severity': 'high',
                'description': 'Memory allocation failure',
                'suggested_action': 'Reduce video resolution or close other applications',
                'recovery_mechanisms': ['reduce_resolution', 'free_memory', 'use_streaming_mode', 'split_processing']
            },
            
            # Filter-specific errors
            {
                'patterns': ['ssim filter', 'ssim stats'],
                'category': 'ssim_filter_error',
                'is_retryable': True,
                'severity': 'medium',
                'description': 'SSIM filter execution error',
                'suggested_action': 'Check video compatibility for SSIM computation',
                'recovery_mechanisms': ['retry_with_scaling', 'use_alternative_ssim_method', 'fallback_to_basic_quality_check']
            },
            
            # Codec and encoding errors
            {
                'patterns': ['encoder.*not found', 'codec.*not available'],
                'category': 'codec_not_available',
                'is_retryable': False,
                'severity': 'high',
                'description': 'Required codec or encoder not available',
                'suggested_action': 'Use different codec or install required encoder',
                'recovery_mechanisms': ['use_fallback_codec', 'install_codec', 'use_software_encoder']
            }
        ]
        
        # Check for specific error patterns
        matched_pattern = None
        for error_pattern in error_patterns:
            for pattern in error_pattern['patterns']:
                if pattern in combined_output:
                    matched_pattern = error_pattern.copy()
                    matched_pattern['specific_error'] = pattern
                    break
            if matched_pattern:
                break
        
        # If no specific pattern matched, use default categorization
        if not matched_pattern:
            if returncode == 1:
                matched_pattern = {
                    'category': 'generic_ffmpeg_error',
                    'is_retryable': True,
                    'severity': 'medium',
                    'description': 'Generic FFmpeg processing error',
                    'suggested_action': 'Check command parameters and input files',
                    'specific_error': None,
                    'recovery_mechanisms': ['retry_with_different_params', 'validate_input_files', 'check_ffmpeg_version']
                }
            elif returncode == 2:
                matched_pattern = {
                    'category': 'invalid_arguments',
                    'is_retryable': False,
                    'severity': 'high',
                    'description': 'Invalid command line arguments',
                    'suggested_action': 'Review FFmpeg command syntax',
                    'specific_error': None,
                    'recovery_mechanisms': ['validate_command_syntax', 'use_simplified_command', 'check_ffmpeg_documentation']
                }
            else:
                matched_pattern = {
                    'category': 'unknown_error',
                    'is_retryable': True,
                    'severity': 'medium',
                    'description': f'Unknown error (return code: {returncode})',
                    'suggested_action': 'Check FFmpeg logs for more details',
                    'specific_error': None,
                    'recovery_mechanisms': ['increase_logging_verbosity', 'retry_with_different_approach', 'check_system_resources']
                }
        
        # Add context information
        context = {
            'operation_name': operation_name,
            'attempt_number': attempt_number,
            'return_code': returncode,
            'stderr_length': len(stderr),
            'stdout_length': len(stdout),
            'has_stderr_content': len(stderr.strip()) > 0,
            'has_stdout_content': len(stdout.strip()) > 0
        }
        
        # Add output samples for debugging (first and last lines)
        if stderr.strip():
            stderr_lines = [line.strip() for line in stderr.split('\n') if line.strip()]
            context['stderr_first_line'] = stderr_lines[0] if stderr_lines else ''
            context['stderr_last_line'] = stderr_lines[-1] if stderr_lines else ''
            context['stderr_line_count'] = len(stderr_lines)
        
        if stdout.strip():
            stdout_lines = [line.strip() for line in stdout.split('\n') if line.strip()]
            context['stdout_first_line'] = stdout_lines[0] if stdout_lines else ''
            context['stdout_last_line'] = stdout_lines[-1] if stdout_lines else ''
            context['stdout_line_count'] = len(stdout_lines)
        
        # Enhance retry recommendation based on attempt number and error type
        if matched_pattern['is_retryable'] and attempt_number >= 2:
            if matched_pattern['category'] in ['vmaf_filter_error', 'resource_busy', 'memory_error']:
                # These errors might resolve with different parameters
                matched_pattern['suggested_action'] += f' (attempt {attempt_number})'
            elif attempt_number >= 3:
                # After multiple attempts, suggest alternative approaches
                matched_pattern['suggested_action'] += ' or consider alternative approach'
        
        # Build final result with enhanced error classification
        result = {
            'category': matched_pattern['category'],
            'is_retryable': matched_pattern['is_retryable'],
            'severity': matched_pattern['severity'],
            'description': matched_pattern['description'],
            'suggested_action': matched_pattern['suggested_action'],
            'specific_error': matched_pattern.get('specific_error'),
            'recovery_mechanisms': matched_pattern.get('recovery_mechanisms', []),
            'context': context,
            'error_classification': self._classify_error_for_recovery(matched_pattern, context, attempt_number)
        }
        
        return result
    
    def _classify_error_for_recovery(self, error_pattern: Dict[str, Any], 
                                   context: Dict[str, Any], attempt_number: int) -> Dict[str, Any]:
        """Classify error for recovery strategy selection.
        
        Args:
            error_pattern: Matched error pattern information
            context: Error context information
            attempt_number: Current attempt number
            
        Returns:
            Dictionary with error classification for recovery
        """
        classification = {
            'error_type': error_pattern['category'],
            'retry_recommended': error_pattern['is_retryable'],
            'urgency': error_pattern['severity'],
            'recovery_priority': 'medium',
            'auto_recovery_possible': False,
            'requires_user_intervention': False
        }
        
        # Determine recovery priority and auto-recovery possibility
        if error_pattern['category'] in ['vmaf_filter_parameter_error', 'vmaf_log_path_error']:
            classification['recovery_priority'] = 'high'
            classification['auto_recovery_possible'] = True
            classification['requires_user_intervention'] = False
        elif error_pattern['category'] in ['vmaf_not_available', 'codec_not_available']:
            classification['recovery_priority'] = 'high'
            classification['auto_recovery_possible'] = False
            classification['requires_user_intervention'] = True
        elif error_pattern['category'] in ['resource_busy', 'memory_error', 'vmaf_computation_timeout']:
            classification['recovery_priority'] = 'medium'
            classification['auto_recovery_possible'] = True
            classification['requires_user_intervention'] = False
        elif error_pattern['category'] in ['file_not_found', 'permission_error', 'disk_space']:
            classification['recovery_priority'] = 'high'
            classification['auto_recovery_possible'] = False
            classification['requires_user_intervention'] = True
        
        # Adjust based on attempt number
        if attempt_number >= 3:
            classification['recovery_priority'] = 'low'
            classification['auto_recovery_possible'] = False
        
        # Add specific recovery recommendations
        recovery_mechanisms = error_pattern.get('recovery_mechanisms', [])
        if recovery_mechanisms:
            # Prioritize recovery mechanisms based on error type and attempt number
            if attempt_number == 1:
                classification['recommended_recovery'] = recovery_mechanisms[0] if recovery_mechanisms else None
            elif attempt_number == 2 and len(recovery_mechanisms) > 1:
                classification['recommended_recovery'] = recovery_mechanisms[1]
            else:
                classification['recommended_recovery'] = recovery_mechanisms[-1] if recovery_mechanisms else None
        
        return classification
    
    def _log_comprehensive_ffmpeg_failure(self, operation_name: str, cmd: List[str], 
                                         result: Dict[str, Any], all_attempts: List[Dict[str, Any]]) -> None:
        """Log comprehensive FFmpeg failure information for enhanced diagnostics.
        
        Args:
            operation_name: Name of the failed operation
            cmd: FFmpeg command that was executed
            result: Final execution result
            all_attempts: List of all attempt information
        """
        logger.error(f"=== COMPREHENSIVE FFMPEG FAILURE REPORT ===")
        logger.error(f"Operation: {operation_name}")
        logger.error(f"Total attempts: {len(all_attempts)}")
        logger.error(f"Total execution time: {result['execution_time']:.2f}s")
        
        # Log command (truncated for readability)
        cmd_str = ' '.join(cmd)
        if len(cmd_str) > 200:
            cmd_str = cmd_str[:200] + '...'
        logger.error(f"Command: {cmd_str}")
        
        # Log final error details
        error_details = result.get('error_details', {})
        logger.error(f"Final error category: {error_details.get('category', 'unknown')}")
        logger.error(f"Error severity: {error_details.get('severity', 'unknown')}")
        logger.error(f"Suggested action: {error_details.get('suggested_action', 'none')}")
        
        # Log attempt summary
        logger.error("=== ATTEMPT SUMMARY ===")
        for i, attempt in enumerate(all_attempts, 1):
            attempt_result = attempt.get('final_result', 'unknown')
            execution_time = attempt.get('execution_time', 0)
            error_type = attempt.get('error_type', 'unknown')
            
            logger.error(f"Attempt {i}: {attempt_result} ({execution_time:.2f}s) - {error_type}")
            
            if 'error_analysis' in attempt:
                error_analysis = attempt['error_analysis']
                logger.error(f"  Category: {error_analysis.get('category', 'unknown')}")
                logger.error(f"  Retryable: {error_analysis.get('is_retryable', False)}")
                if error_analysis.get('specific_error'):
                    logger.error(f"  Specific error: {error_analysis['specific_error']}")
        
        # Log output samples for debugging
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("=== OUTPUT SAMPLES FOR DEBUGGING ===")
            
            # Log stderr samples
            stderr = result.get('stderr', '')
            if stderr.strip():
                stderr_lines = [line.strip() for line in stderr.split('\n') if line.strip()]
                logger.debug(f"STDERR ({len(stderr_lines)} lines):")
                
                # First few lines
                for i, line in enumerate(stderr_lines[:3]):
                    logger.debug(f"  {i+1}: {line}")
                
                if len(stderr_lines) > 6:
                    logger.debug("  ...")
                
                # Last few lines
                for i, line in enumerate(stderr_lines[-3:]):
                    logger.debug(f"  {len(stderr_lines)-2+i}: {line}")
            
            # Log stdout samples
            stdout = result.get('stdout', '')
            if stdout.strip():
                stdout_lines = [line.strip() for line in stdout.split('\n') if line.strip()]
                logger.debug(f"STDOUT ({len(stdout_lines)} lines):")
                
                # First few lines
                for i, line in enumerate(stdout_lines[:3]):
                    logger.debug(f"  {i+1}: {line}")
                
                if len(stdout_lines) > 6:
                    logger.debug("  ...")
                
                # Last few lines
                for i, line in enumerate(stdout_lines[-3:]):
                    logger.debug(f"  {len(stdout_lines)-2+i}: {line}")
        
        logger.error("=== END FAILURE REPORT ===")
    
    def _safe_format_scores(self, vmaf_score: Optional[float], ssim_score: Optional[float]) -> Tuple[str, str]:
        """Safely format quality scores for logging.
        
        Args:
            vmaf_score: VMAF score (0-100) or None
            ssim_score: SSIM score (0-1) or None
            
        Returns:
            Tuple of (vmaf_str, ssim_str) formatted for display
        """
        vmaf_str = f"{vmaf_score:.2f}" if vmaf_score is not None else "N/A"
        ssim_str = f"{ssim_score:.4f}" if ssim_score is not None else "N/A"
        return vmaf_str, ssim_str
    
    def _validate_quality_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and ensure quality result structure completeness.
        
        Args:
            result: Quality evaluation result dictionary
            
        Returns:
            Validated result with all required keys and safe values
        """
        # Ensure all required keys exist with safe defaults
        validated_result = {
            'vmaf_score': result.get('vmaf_score'),
            'ssim_score': result.get('ssim_score'),
            'passes': bool(result.get('passes', False)),
            'method': result.get('method', 'error'),
            'confidence': float(result.get('confidence', 0.0)),
            'evaluation_success': bool(result.get('evaluation_success', False)),
            'details': result.get('details', {})
        }
        
        # Validate score ranges if they exist
        if validated_result['vmaf_score'] is not None:
            # VMAF should be 0-100
            if not (0 <= validated_result['vmaf_score'] <= 100):
                logger.warning(f"VMAF score {validated_result['vmaf_score']} outside expected range [0-100]")
                validated_result['vmaf_score'] = None
        
        if validated_result['ssim_score'] is not None:
            # SSIM should be 0-1
            if not (0 <= validated_result['ssim_score'] <= 1):
                logger.warning(f"SSIM score {validated_result['ssim_score']} outside expected range [0-1]")
                validated_result['ssim_score'] = None
        
        # Validate confidence score
        if not (0 <= validated_result['confidence'] <= 1):
            logger.warning(f"Confidence score {validated_result['confidence']} outside expected range [0-1]")
            validated_result['confidence'] = max(0.0, min(1.0, validated_result['confidence']))
        
        # Ensure details is a dictionary
        if not isinstance(validated_result['details'], dict):
            validated_result['details'] = {}
        
        return validated_result
    
    def _safe_score_comparison(self, score: Optional[float], threshold: float) -> bool:
        """Safely compare a score against a threshold, handling None values.
        
        Args:
            score: Quality score or None
            threshold: Threshold value to compare against
            
        Returns:
            True if score exists and meets threshold, False otherwise
        """
        return score is not None and score >= threshold
    
    def check_vmaf_available(self) -> bool:
        """Check if libvmaf is available in FFmpeg (cached)."""
        if self._vmaf_available is None:
            try:
                from ..ffmpeg_utils import FFmpegUtils
                self._vmaf_available = FFmpegUtils.check_libvmaf_available()
                if self._vmaf_available:
                    logger.info("libvmaf filter detected in FFmpeg")
                else:
                    logger.warning("libvmaf not available; will use SSIM fallback")
            except Exception as e:
                logger.warning(f"Failed to check libvmaf availability: {e}")
                self._vmaf_available = False
        return self._vmaf_available
    
    def _calculate_confidence_score(self, vmaf_score: Optional[float], ssim_score: Optional[float], 
                                   vmaf_confidence: float = 1.0, ssim_confidence: float = 1.0,
                                   method: str = 'error') -> float:
        """Calculate overall confidence score for quality evaluation.
        
        Args:
            vmaf_score: VMAF score (0-100) or None
            ssim_score: SSIM score (0-1) or None
            vmaf_confidence: Confidence in VMAF parsing (0-1)
            ssim_confidence: Confidence in SSIM parsing (0-1)
            method: Evaluation method used
            
        Returns:
            Overall confidence score (0-1)
        """
        if method == 'error':
            return 0.0
        
        # Base confidence factors
        vmaf_available = vmaf_score is not None
        ssim_available = ssim_score is not None
        
        if method == 'vmaf+ssim':
            if vmaf_available and ssim_available:
                # Both metrics available - high confidence
                return (vmaf_confidence + ssim_confidence) / 2
            elif vmaf_available:
                # Only VMAF available - moderate confidence
                return vmaf_confidence * 0.8
            elif ssim_available:
                # Only SSIM available - lower confidence
                return ssim_confidence * 0.6
            else:
                # No metrics available - no confidence
                return 0.0
        
        elif method == 'ssim_only':
            if ssim_available:
                # SSIM-only mode with successful parsing
                return ssim_confidence * 0.9  # Slightly lower than dual-metric
            else:
                return 0.0
        
        return 0.0
    
    def _track_evaluation_success(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Track evaluation success separately from quality pass/fail.
        
        Args:
            result: Quality evaluation result
            
        Returns:
            Updated result with evaluation_success field
        """
        # Evaluation is successful if we got at least one valid metric
        has_vmaf = result.get('vmaf_score') is not None
        has_ssim = result.get('ssim_score') is not None
        
        evaluation_success = has_vmaf or has_ssim
        result['evaluation_success'] = evaluation_success
        
        # Add evaluation details
        if 'details' not in result:
            result['details'] = {}
        
        result['details']['has_vmaf'] = has_vmaf
        result['details']['has_ssim'] = has_ssim
        result['details']['evaluation_attempted'] = result.get('method') != 'error'
        
        return result

    def evaluate_quality(self, original_path: str, compressed_path: str,
                        vmaf_threshold: float = 80.0, ssim_threshold: float = 0.94,
                        eval_height: Optional[int] = None) -> Dict[str, Any]:
        """Evaluate compressed video quality against original.
        
        Args:
            original_path: Reference video
            compressed_path: Compressed video to evaluate
            vmaf_threshold: Minimum VMAF score (0-100)
            ssim_threshold: Minimum SSIM score (0-1)
            eval_height: Downscale eval to this height (e.g., 540); None = native
        
        Returns:
            Dict with keys:
                - vmaf_score: float or None
                - ssim_score: float or None
                - passes: bool (meets thresholds)
                - method: 'vmaf+ssim', 'ssim_only', or 'error'
                - confidence: float (0-1, confidence in evaluation)
                - evaluation_success: bool (whether evaluation completed successfully)
                - details: additional info including confidence breakdown
        """
        # Log evaluation start with structured information
        self._log_quality_evaluation_start(original_path, compressed_path, 
                                         vmaf_threshold, ssim_threshold, eval_height)
        
        evaluation_start_time = time.time()
        
        result = {
            'vmaf_score': None,
            'ssim_score': None,
            'passes': False,
            'method': 'error',
            'confidence': 0.0,
            'evaluation_success': False,
            'scaling_applied': False,
            'original_resolution': None,
            'comparison_resolution': None,
            'details': {}
        }
        
        try:
            # Validate inputs
            if not os.path.exists(original_path):
                result['details']['error'] = f"Original not found: {original_path}"
                return result
            if not os.path.exists(compressed_path):
                result['details']['error'] = f"Compressed not found: {compressed_path}"
                return result
            
            # Check VMAF availability
            has_vmaf = self.check_vmaf_available()
            
            if has_vmaf:
                # Run VMAF and SSIM concurrently
                logger.debug("Starting concurrent VMAF and SSIM evaluation")
                concurrent_start_time = time.time()
                
                with ThreadPoolExecutor(max_workers=2) as executor:
                    # Submit both tasks with separate start times
                    vmaf_start_time = time.time()
                    ssim_start_time = time.time()
                    
                    vmaf_future = executor.submit(
                        self._compute_vmaf, original_path, compressed_path, eval_height
                    )
                    ssim_future = executor.submit(
                        self._compute_ssim, original_path, compressed_path, eval_height
                    )
                    
                    # Collect results with independent error handling
                    vmaf_score = None
                    ssim_score = None
                    ssim_confidence = 0.0
                    vmaf_time = 0.0
                    ssim_time = 0.0
                    
                    # Wait for VMAF result
                    try:
                        vmaf_score = vmaf_future.result()
                        vmaf_time = time.time() - vmaf_start_time
                        logger.debug(f"VMAF computation completed in {vmaf_time:.2f}s")
                    except Exception as e:
                        logger.error(f"VMAF computation failed in thread: {e}", exc_info=True)
                        vmaf_time = time.time() - vmaf_start_time
                        # vmaf_score remains None
                    
                    # Wait for SSIM result
                    try:
                        ssim_score, ssim_confidence = ssim_future.result()
                        ssim_time = time.time() - ssim_start_time
                        logger.debug(f"SSIM computation completed in {ssim_time:.2f}s")
                    except Exception as e:
                        logger.error(f"SSIM computation failed in thread: {e}", exc_info=True)
                        ssim_time = time.time() - ssim_start_time
                        # ssim_score remains None, ssim_confidence remains 0.0
                
                # Calculate timing after both futures complete
                total_time = time.time() - evaluation_start_time
                concurrent_time = time.time() - concurrent_start_time
                
                # Log concurrency benefit
                if vmaf_score is not None or ssim_score is not None:
                    estimated_sequential_time = (vmaf_time if vmaf_score is not None else 0) + \
                                                   (ssim_time if ssim_score is not None else 0)
                    if estimated_sequential_time > 0:
                        time_saved = estimated_sequential_time - concurrent_time
                        if time_saved > 0:
                            logger.debug(f"Concurrent evaluation saved approximately {time_saved:.2f}s "
                                       f"(concurrent: {concurrent_time:.2f}s vs sequential: {estimated_sequential_time:.2f}s)")
                
                result['vmaf_score'] = vmaf_score
                result['ssim_score'] = ssim_score
                result['method'] = 'vmaf+ssim'
                
                # Both must pass - use safe comparison
                vmaf_pass = self._safe_score_comparison(vmaf_score, vmaf_threshold)
                ssim_pass = self._safe_score_comparison(ssim_score, ssim_threshold)
                result['passes'] = vmaf_pass and ssim_pass
                
                # Calculate confidence scores with detailed logging
                vmaf_confidence = 1.0 if vmaf_score is not None else 0.0
                result = self._add_confidence_scoring_to_result(result, vmaf_confidence, ssim_confidence)
                
                # Safe dictionary access for details
                result['details']['vmaf_threshold'] = vmaf_threshold
                result['details']['ssim_threshold'] = ssim_threshold
                result['details']['vmaf_pass'] = vmaf_pass
                result['details']['ssim_pass'] = ssim_pass
                
            else:
                # SSIM-only fallback
                ssim_score, ssim_confidence = self._compute_ssim(original_path, compressed_path, eval_height)
                result['ssim_score'] = ssim_score
                result['method'] = 'ssim_only'
                result['passes'] = self._safe_score_comparison(ssim_score, ssim_threshold)
                
                # Calculate confidence for SSIM-only mode with detailed logging
                result = self._add_confidence_scoring_to_result(result, 0.0, ssim_confidence)
                
                result['details']['ssim_threshold'] = ssim_threshold
            
            if eval_height:
                result['details']['eval_height'] = eval_height
            
            # Track evaluation success and add confidence thresholds
            result = self._track_evaluation_success(result)
            
            # Log performance metrics
            evaluation_time = time.time() - evaluation_start_time
            self._log_performance_metrics("Quality evaluation", evaluation_time, {
                'method': result['method'],
                'vmaf_available': result['vmaf_score'] is not None,
                'ssim_available': result['ssim_score'] is not None,
                'confidence': result['confidence']
            })
            
        except Exception as e:
            evaluation_time = time.time() - evaluation_start_time
            logger.error(f"Quality evaluation failed after {evaluation_time:.2f}s: {e}")
            result['details']['error'] = str(e)
            result['evaluation_success'] = False
            result['confidence'] = 0.0
        
        # Validate and ensure result structure completeness before returning
        validated_result = self._validate_quality_result(result)
        
        # Ensure evaluation_success is tracked in validated result
        if 'evaluation_success' not in validated_result:
            validated_result = self._track_evaluation_success(validated_result)
        
        # Apply fallback behavior if needed
        final_result = self._apply_fallback_behavior(validated_result, vmaf_threshold, ssim_threshold)
        
        # Log final structured results
        self._log_quality_evaluation_result(final_result)
        
        return final_result
    
    def _cleanup_vmaf_debug_files(self, debug_files: List[str], success: bool = True) -> None:
        """Clean up VMAF debug files after processing based on retention policy.
        
        Args:
            debug_files: List of debug file paths to clean up
            success: Whether the operation was successful
        """
        # Get debug file configuration
        debug_config = self._get_debug_file_config()
        
        if not debug_config['cleanup_enabled']:
            logger.debug("Debug file cleanup disabled by configuration")
            return
        
        # Check if we should clean up based on success/failure
        should_cleanup = (success and debug_config['cleanup_on_success']) or \
                        (not success and debug_config['cleanup_on_failure'])
        
        if not should_cleanup:
            logger.debug(f"Debug file cleanup skipped (success={success}, "
                        f"cleanup_on_success={debug_config['cleanup_on_success']}, "
                        f"cleanup_on_failure={debug_config['cleanup_on_failure']})")
            return
        
        retention_policy = debug_config['retention_policy']
        
        if retention_policy == 'immediate':
            # Clean up immediately
            for debug_file in debug_files:
                self._remove_debug_file(debug_file)
        else:
            # For other policies, just log that files are being kept
            logger.debug(f"Debug files kept due to retention policy '{retention_policy}': "
                        f"{[os.path.basename(f) for f in debug_files]}")
            
            # Perform general cleanup based on retention policy
            self._perform_debug_file_rotation()
    
    def _get_debug_file_config(self) -> Dict[str, Any]:
        """Get debug file configuration with defaults."""
        if self.config:
            return {
                'cleanup_enabled': self.config.get('quality_evaluation.debug_files.cleanup_enabled', True),
                'retention_policy': self.config.get('quality_evaluation.debug_files.retention_policy', 'immediate'),
                'max_files_per_session': self.config.get('quality_evaluation.debug_files.max_files_per_session', 10),
                'max_age_hours': self.config.get('quality_evaluation.debug_files.max_age_hours', 24),
                'cleanup_on_success': self.config.get('quality_evaluation.debug_files.cleanup_on_success', True),
                'cleanup_on_failure': self.config.get('quality_evaluation.debug_files.cleanup_on_failure', False)
            }
        return {
            'cleanup_enabled': True,
            'retention_policy': 'immediate',
            'max_files_per_session': 10,
            'max_age_hours': 24,
            'cleanup_on_success': True,
            'cleanup_on_failure': False
        }
    
    def _remove_debug_file(self, debug_file: str) -> None:
        """Remove a single debug file with error handling."""
        try:
            if os.path.exists(debug_file):
                os.remove(debug_file)
                logger.debug(f"Cleaned up VMAF debug file: {os.path.basename(debug_file)}")
        except Exception as e:
            logger.debug(f"Could not remove VMAF debug file {debug_file}: {e}")
    
    def _perform_debug_file_rotation(self) -> None:
        """Perform debug file rotation based on retention policy."""
        try:
            from ..logger_setup import get_default_logs_dir
            logs_dir = get_default_logs_dir()
            debug_config = self._get_debug_file_config()
            
            # Find all VMAF debug files
            import glob
            vmaf_debug_pattern = os.path.join(logs_dir, "vmaf_debug_*.json")
            vmaf_text_pattern = os.path.join(logs_dir, "vmaf_debug_text_*.log")
            
            debug_files = glob.glob(vmaf_debug_pattern) + glob.glob(vmaf_text_pattern)
            
            if not debug_files:
                return
            
            retention_policy = debug_config['retention_policy']
            max_files = debug_config['max_files_per_session']
            max_age_hours = debug_config['max_age_hours']
            
            # Sort by modification time (newest first)
            debug_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            
            files_to_remove = []
            
            if retention_policy == 'session':
                # Keep only the most recent N files
                if len(debug_files) > max_files:
                    files_to_remove = debug_files[max_files:]
            
            elif retention_policy in ['daily', 'weekly']:
                # Remove files older than max_age_hours
                import time
                current_time = time.time()
                max_age_seconds = max_age_hours * 3600
                
                for debug_file in debug_files:
                    try:
                        file_age = current_time - os.path.getmtime(debug_file)
                        if file_age > max_age_seconds:
                            files_to_remove.append(debug_file)
                    except Exception:
                        continue
            
            # Remove old files
            for debug_file in files_to_remove:
                self._remove_debug_file(debug_file)
            
            if files_to_remove:
                logger.debug(f"Debug file rotation: removed {len(files_to_remove)} old files "
                           f"(policy: {retention_policy})")
        
        except Exception as e:
            logger.debug(f"Debug file rotation failed: {e}")
    
    def cleanup_legacy_debug_files(self) -> None:
        """Clean up any legacy debug files created in the main directory.
        
        This method should be called during initialization to clean up files
        that may have been created by previous versions.
        """
        try:
            # Look for files that might have been created in the main directory
            legacy_files = ["-", "vmaf_debug.json", "vmaf_debug.log"]
            
            for filename in legacy_files:
                if os.path.exists(filename):
                    try:
                        os.remove(filename)
                        logger.info(f"Cleaned up legacy debug file: {filename}")
                    except Exception as e:
                        logger.warning(f"Could not remove legacy debug file {filename}: {e}")
            
            # Also check for any files matching VMAF debug patterns in current directory
            import glob
            legacy_patterns = ["vmaf_debug_*.json", "vmaf_debug_*.log", "vmaf_debug_text_*.log", "*vmaf_debug*.json", "*vmaf_debug*.log"]
            
            for pattern in legacy_patterns:
                for filepath in glob.glob(pattern):
                    try:
                        os.remove(filepath)
                        logger.info(f"Cleaned up legacy debug file: {filepath}")
                    except Exception as e:
                        logger.warning(f"Could not remove legacy debug file {filepath}: {e}")
        
        except Exception as e:
            logger.debug(f"Legacy debug file cleanup failed: {e}")
    
    def _parse_vmaf_output(self, stderr_output: str) -> Optional[float]:
        """Robust VMAF parsing with automatic format detection and comprehensive error handling.
        
        Args:
            stderr_output: FFmpeg stderr output containing VMAF statistics
            
        Returns:
            VMAF score (0-100) or None if parsing fails
        """
        parsing_attempts = []
        
        # Detect output format for optimized parsing strategy
        detected_format = self._detect_vmaf_output_format(stderr_output)
        self._log_parsing_attempt("VMAF", "format_detection", True, f"Detected format: {detected_format}")
        
        # Strategy selection based on detected format
        strategies = []
        if detected_format == 'xml':
            strategies = ['xml', 'json', 'text', 'numeric']
        elif detected_format == 'json':
            strategies = ['json', 'xml', 'text', 'numeric']
        elif detected_format == 'text':
            strategies = ['text', 'json', 'xml', 'numeric']
        else:
            # Unknown format - try all strategies
            strategies = ['xml', 'json', 'text', 'numeric']
        
        # Execute parsing strategies in optimized order
        for strategy in strategies:
            try:
                if strategy == 'xml':
                    self._log_parsing_attempt("VMAF", "strategy_xml", False, "Starting XML parsing")
                    xml_score = self._parse_vmaf_xml_format(stderr_output)
                    validated_score = self._validate_vmaf_score(xml_score)
                    if validated_score is not None:
                        parsing_attempts.append("strategy_xml_success")
                        self._log_parsing_attempt("VMAF", "strategy_xml", True, f"Score: {validated_score:.2f}")
                        return validated_score
                    parsing_attempts.append("strategy_xml_no_data")
                    self._log_parsing_attempt("VMAF", "strategy_xml", False, "No valid XML VMAF data found")
                
                elif strategy == 'json':
                    self._log_parsing_attempt("VMAF", "strategy_json", False, "Starting JSON parsing")
                    json_score = self._parse_vmaf_json_format(stderr_output)
                    validated_score = self._validate_vmaf_score(json_score)
                    if validated_score is not None:
                        parsing_attempts.append("strategy_json_success")
                        self._log_parsing_attempt("VMAF", "strategy_json", True, f"Score: {validated_score:.2f}")
                        return validated_score
                    parsing_attempts.append("strategy_json_no_data")
                    self._log_parsing_attempt("VMAF", "strategy_json", False, "No valid JSON VMAF data found")
                
                elif strategy == 'text':
                    self._log_parsing_attempt("VMAF", "strategy_text", False, "Starting text pattern parsing")
                    text_score = self._parse_vmaf_text_format(stderr_output)
                    validated_score = self._validate_vmaf_score(text_score)
                    if validated_score is not None:
                        parsing_attempts.append("strategy_text_success")
                        self._log_parsing_attempt("VMAF", "strategy_text", True, f"Score: {validated_score:.2f}")
                        return validated_score
                    parsing_attempts.append("strategy_text_no_data")
                    self._log_parsing_attempt("VMAF", "strategy_text", False, "No valid text VMAF data found")
                
                elif strategy == 'numeric':
                    self._log_parsing_attempt("VMAF", "strategy_numeric", False, "Starting numeric extraction")
                    numeric_score = self._parse_vmaf_numeric_format(stderr_output)
                    validated_score = self._validate_vmaf_score(numeric_score)
                    if validated_score is not None:
                        parsing_attempts.append("strategy_numeric_success")
                        self._log_parsing_attempt("VMAF", "strategy_numeric", True, f"Score: {validated_score:.2f}")
                        return validated_score
                    parsing_attempts.append("strategy_numeric_no_data")
                    self._log_parsing_attempt("VMAF", "strategy_numeric", False, "No valid numeric VMAF data found")
                
            except Exception as e:
                parsing_attempts.append(f"strategy_{strategy}_error:{str(e)[:50]}")
                self._log_parsing_attempt("VMAF", f"strategy_{strategy}", False, f"Exception: {str(e)[:100]}")
        
        # All strategies failed - log comprehensive debugging information
        logger.warning(f"VMAF parsing failed with all strategies. Detected format: {detected_format}")
        self._log_vmaf_parsing_failure(stderr_output, parsing_attempts)
        
        return None

    def _parse_vmaf_json_format(self, stderr_output: str) -> Optional[float]:
        """Parse VMAF output in JSON format.
        
        Args:
            stderr_output: FFmpeg stderr output containing JSON VMAF data
            
        Returns:
            VMAF score (0-100) or None if parsing fails
        """
        try:
            # Look for complete JSON objects in the output
            lines = stderr_output.split('\n')
            json_objects_found = 0
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Find potential JSON content
                json_start = line.find('{')
                if json_start >= 0:
                    json_objects_found += 1
                    try:
                        # Try to parse from the first { to end of line
                        json_str = line[json_start:]
                        vmaf_data = json.loads(json_str)
                        
                        # Look for pooled metrics (libvmaf 2.x format)
                        if 'pooled_metrics' in vmaf_data and 'vmaf' in vmaf_data['pooled_metrics']:
                            vmaf_score = vmaf_data['pooled_metrics']['vmaf']['mean']
                            return float(vmaf_score)
                        
                        # Look for aggregate metrics (alternative format)
                        if 'aggregate' in vmaf_data and 'VMAF_score' in vmaf_data['aggregate']:
                            vmaf_score = vmaf_data['aggregate']['VMAF_score']
                            return float(vmaf_score)
                        
                        # Look for direct vmaf field
                        if 'vmaf' in vmaf_data:
                            if isinstance(vmaf_data['vmaf'], dict) and 'mean' in vmaf_data['vmaf']:
                                vmaf_score = vmaf_data['vmaf']['mean']
                            else:
                                vmaf_score = vmaf_data['vmaf']
                            return float(vmaf_score)
                        
                    except json.JSONDecodeError as e:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"JSON decode error in line: {str(e)[:100]}")
                        continue
                    except (KeyError, TypeError, ValueError) as e:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"JSON structure error: {str(e)[:100]}")
                        continue
            
            return None
            
        except Exception as e:
            logger.debug(f"JSON parsing error: {str(e)[:100]}")
            return None

    def _parse_vmaf_text_format(self, stderr_output: str) -> Optional[float]:
        """Parse VMAF output in text format using regex patterns.
        
        Args:
            stderr_output: FFmpeg stderr output containing text VMAF data
            
        Returns:
            VMAF score (0-100) or None if parsing fails
        """
        try:
            # Use configured text parsing patterns or defaults
            patterns = self.vmaf_config.get('text_parsing_patterns', [
                r'VMAF\s+score[:\s]+([\d.]+)',
                r'vmaf[:\s]+([\d.]+)',
                r'mean[:\s]+([\d.]+)'
            ])
            
            # Compile patterns
            compiled_patterns = []
            for pattern_str in patterns:
                try:
                    compiled_patterns.append(re.compile(pattern_str, re.IGNORECASE))
                except re.error as e:
                    logger.debug(f"Invalid regex pattern '{pattern_str}': {e}")
                    continue
            
            # Search for matches
            for pattern in compiled_patterns:
                matches = pattern.findall(stderr_output)
                if matches:
                    # Take the last match (usually the final/aggregate score)
                    try:
                        vmaf_score = float(matches[-1])
                        if 0 <= vmaf_score <= 100:  # Validate VMAF range
                            return vmaf_score
                    except (ValueError, TypeError):
                        continue
            
            return None
            
        except Exception as e:
            logger.debug(f"Text parsing error: {str(e)[:100]}")
            return None

    def _parse_vmaf_numeric_format(self, stderr_output: str) -> Optional[float]:
        """Parse VMAF output using numeric extraction from VMAF-related lines.
        
        Args:
            stderr_output: FFmpeg stderr output containing numeric VMAF data
            
        Returns:
            VMAF score (0-100) or None if parsing fails
        """
        try:
            lines = stderr_output.split('\n')
            potential_scores = []
            
            for line in lines:
                line = line.strip().lower()
                if 'vmaf' in line or 'mean' in line:
                    # Extract all numbers from VMAF-related lines
                    numbers = re.findall(r'\b(\d+\.?\d*)\b', line)
                    for num_str in numbers:
                        try:
                            num = float(num_str)
                            # VMAF scores are typically 0-100
                            if 0 <= num <= 100:
                                potential_scores.append(num)
                        except ValueError:
                            continue
            
            if potential_scores:
                # Use the highest score (often the most reliable)
                return max(potential_scores)
            
            return None
            
        except Exception as e:
            logger.debug(f"Numeric parsing error: {str(e)[:100]}")
            return None

    def _parse_vmaf_xml_format(self, stderr_output: str) -> Optional[float]:
        """Parse VMAF output in XML format from libvmaf filter.
        
        Args:
            stderr_output: FFmpeg stderr output that may contain XML VMAF data
            
        Returns:
            VMAF score (0-100) or None if parsing fails
        """
        try:
            import xml.etree.ElementTree as ET
            
            # Look for XML content in the output
            lines = stderr_output.split('\n')
            xml_content = []
            in_xml = False
            xml_start_patterns = ['<?xml', '<VMAF', '<vmaf', '<libvmaf']
            
            for line in lines:
                line_stripped = line.strip()
                
                # Check if this line starts XML content
                if not in_xml:
                    for pattern in xml_start_patterns:
                        if pattern.lower() in line_stripped.lower():
                            in_xml = True
                            xml_content.append(line)
                            break
                else:
                    xml_content.append(line)
                    # Check if XML content ends (simple heuristic)
                    if '</VMAF>' in line or '</vmaf>' in line or '</libvmaf>' in line:
                        break
            
            if not xml_content:
                self._log_parsing_attempt("VMAF", "xml_format", False, "No XML content found")
                return None
            
            # Join XML content and try to parse
            xml_string = '\n'.join(xml_content)
            
            # Try to parse the XML
            try:
                root = ET.fromstring(xml_string)
            except ET.ParseError:
                # If direct parsing fails, try to extract XML from within the content
                # Look for complete XML documents
                xml_start = xml_string.find('<?xml')
                if xml_start == -1:
                    # Look for root elements without XML declaration
                    for pattern in ['<VMAF', '<vmaf', '<libvmaf']:
                        xml_start = xml_string.lower().find(pattern.lower())
                        if xml_start != -1:
                            break
                
                if xml_start == -1:
                    self._log_parsing_attempt("VMAF", "xml_format", False, "No valid XML start found")
                    return None
                
                # Extract from XML start to end
                xml_substring = xml_string[xml_start:]
                root = ET.fromstring(xml_substring)
            
            # Parse frame-level VMAF data and calculate mean
            vmaf_scores = []
            
            # Strategy 1: Look for aggregate/pooled metrics first (preferred)
            for metric in root.findall('.//metric'):
                if metric.get('name', '').lower() == 'vmaf':
                    try:
                        # Try mean attribute first, then value, then text content
                        score_str = metric.get('mean') or metric.get('value') or metric.text
                        if score_str:
                            score = float(score_str)
                            if 0 <= score <= 100:
                                # Aggregate scores are preferred, return immediately
                                self._log_parsing_attempt("VMAF", "xml_format_aggregate", True, 
                                                        f"Found aggregate score: {score:.2f}")
                                return score
                    except (ValueError, TypeError):
                        continue
            
            # Strategy 2: Look for frame elements with VMAF scores as attributes
            for frame in root.findall('.//frame'):
                try:
                    # Check for vmaf attribute
                    vmaf_attr = frame.get('vmaf')
                    if vmaf_attr:
                        score = float(vmaf_attr)
                        if 0 <= score <= 100:
                            vmaf_scores.append(score)
                except (ValueError, TypeError):
                    continue
                
                # Also check for child vmaf elements
                vmaf_elem = frame.find('vmaf') or frame.find('VMAF')
                if vmaf_elem is not None:
                    try:
                        score_str = vmaf_elem.text or vmaf_elem.get('value', '')
                        if score_str:
                            score = float(score_str)
                            if 0 <= score <= 100:
                                vmaf_scores.append(score)
                    except (ValueError, TypeError):
                        continue
            
            # Strategy 3: Look for aggregate/pooled sections with vmaf elements
            for section in root.findall('.//aggregate') + root.findall('.//pooled') + root.findall('.//pooled_metrics'):
                vmaf_elem = section.find('vmaf') or section.find('VMAF')
                if vmaf_elem is not None:
                    try:
                        score_str = vmaf_elem.text or vmaf_elem.get('mean', '') or vmaf_elem.get('value', '')
                        if score_str:
                            score = float(score_str)
                            if 0 <= score <= 100:
                                # Aggregate scores are preferred, return immediately
                                self._log_parsing_attempt("VMAF", "xml_format_aggregate", True, 
                                                        f"Found aggregate score: {score:.2f}")
                                return score
                    except (ValueError, TypeError):
                        continue
            
            # Strategy 4: Look for any element with 'vmaf' in name and numeric content
            for elem in root.iter():
                if 'vmaf' in elem.tag.lower():
                    try:
                        score_str = elem.text or elem.get('value', '') or elem.get('mean', '')
                        if score_str:
                            score = float(score_str)
                            if 0 <= score <= 100:
                                vmaf_scores.append(score)
                    except (ValueError, TypeError):
                        continue
            
            # Calculate mean from frame-level scores if we have them
            if vmaf_scores:
                mean_score = sum(vmaf_scores) / len(vmaf_scores)
                self._log_parsing_attempt("VMAF", "xml_format_frames", True, 
                                        f"Calculated mean from {len(vmaf_scores)} frame scores: {mean_score:.2f}")
                return mean_score
            
            self._log_parsing_attempt("VMAF", "xml_format", False, "No VMAF scores found in XML structure")
            return None
            
        except ImportError:
            self._log_parsing_attempt("VMAF", "xml_format", False, "xml.etree.ElementTree not available")
            return None
        except ET.ParseError as e:
            self._log_parsing_attempt("VMAF", "xml_format", False, f"XML parsing error: {str(e)[:100]}")
            return None
        except Exception as e:
            self._log_parsing_attempt("VMAF", "xml_format", False, f"Unexpected error: {str(e)[:100]}")
            return None

    def _detect_vmaf_output_format(self, output: str) -> str:
        """Detect the format of VMAF output for automatic parsing strategy selection.
        
        Args:
            output: FFmpeg stderr output containing VMAF data
            
        Returns:
            Format type: 'xml', 'json', 'text', or 'unknown'
        """
        output_lower = output.lower()
        
        # Check for XML format indicators
        xml_indicators = ['<?xml', '<vmaf', '<libvmaf', '</vmaf>', '</libvmaf>']
        if any(indicator in output_lower for indicator in xml_indicators):
            return 'xml'
        
        # Check for JSON format indicators
        json_indicators = ['"pooled_metrics"', '"vmaf"', '"aggregate"', '"mean"']
        if ('{' in output and '}' in output and 
            any(indicator in output_lower for indicator in json_indicators)):
            return 'json'
        
        # Check for text format indicators
        text_indicators = ['vmaf score:', 'vmaf:', 'mean:']
        if any(indicator in output_lower for indicator in text_indicators):
            return 'text'
        
        return 'unknown'

    def _validate_vmaf_score(self, score: Optional[float]) -> Optional[float]:
        """Validate VMAF score is within expected range (0-100).
        
        Args:
            score: VMAF score to validate
            
        Returns:
            Validated score or None if invalid
        """
        if score is None:
            return None
        
        try:
            score_float = float(score)
            if 0 <= score_float <= 100:
                return score_float
            else:
                logger.warning(f"VMAF score {score_float:.2f} outside valid range [0-100]")
                return None
        except (ValueError, TypeError):
            logger.warning(f"Invalid VMAF score format: {score}")
            return None

    def _log_vmaf_parsing_failure(self, output: str, parsing_attempts: List[str]) -> None:
        """Log detailed information about VMAF parsing failure for debugging.
        
        Args:
            output: The FFmpeg output that failed to parse
            parsing_attempts: List of parsing strategies that were attempted
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        
        # Log parsing failure summary
        logger.debug(f"VMAF parsing failed after {len(parsing_attempts)} attempts: {parsing_attempts}")
        
        # Detect and log the apparent format
        detected_format = self._detect_vmaf_output_format(output)
        logger.debug(f"Detected output format: {detected_format}")
        
        # Log output characteristics for debugging
        output_lines = output.split('\n')
        non_empty_lines = [line.strip() for line in output_lines if line.strip()]
        
        logger.debug(f"Output characteristics: {len(output_lines)} total lines, "
                    f"{len(non_empty_lines)} non-empty lines")
        
        # Log sample of output for debugging (first and last few lines)
        if non_empty_lines:
            sample_size = min(3, len(non_empty_lines))
            logger.debug(f"Output sample (first {sample_size} lines): {non_empty_lines[:sample_size]}")
            
            if len(non_empty_lines) > sample_size:
                logger.debug(f"Output sample (last {sample_size} lines): {non_empty_lines[-sample_size:]}")
        
        # Log specific format-related issues
        if detected_format == 'xml':
            logger.debug("XML format detected but parsing failed - check XML structure")
        elif detected_format == 'json':
            logger.debug("JSON format detected but parsing failed - check JSON structure")
        elif detected_format == 'text':
            logger.debug("Text format detected but parsing failed - check text patterns")
        else:
            logger.debug("Unknown format - output may be malformed or in unexpected format")

    def _compute_vmaf(self, ref_path: str, dist_path: str, 
                     eval_height: Optional[int] = None) -> Optional[float]:
        """Compute VMAF score using FFmpeg libvmaf filter with resolution scaling support.
        
        Returns VMAF score (0-100) or None on error.
        """
        try:
            # Create session ID for temporary files
            session_id = self._generate_thread_safe_session_id()
            
            # Initialize resolution-aware evaluator
            resolution_evaluator = ResolutionAwareQualityEvaluator(self.config)
            
            try:
                # Use resolution-aware VMAF computation
                vmaf_score, scaling_info = resolution_evaluator.compute_vmaf_with_scaling(
                    ref_path, dist_path, session_id
                )
                
                # Log scaling decisions
                if scaling_info['scaling_applied']:
                    logger.info(f"VMAF resolution scaling applied: "
                               f"{scaling_info['original_resolution']} → {scaling_info['comparison_resolution']}")
                    
                    # Log scaled files for debugging
                    if scaling_info['scaled_files']:
                        logger.debug(f"Temporary scaled files created: {len(scaling_info['scaled_files'])}")
                
                return vmaf_score
                
            finally:
                # Always clean up temporary files
                resolution_evaluator.cleanup_temporary_files()
            
        except Exception as e:
            logger.error(f"VMAF computation with scaling failed: {e}")
            
            # Fallback to legacy method if scaling fails
            logger.debug("Falling back to legacy VMAF computation method")
            return self._compute_vmaf_legacy(ref_path, dist_path, eval_height)
    
    def _compute_vmaf_legacy(self, ref_path: str, dist_path: str, 
                            eval_height: Optional[int] = None) -> Optional[float]:
        """Legacy VMAF computation method (fallback for compatibility).
        
        Returns VMAF score (0-100) or None on error.
        """
        debug_files_created = []
        
        try:
            # Always scale both videos to same resolution for comparison
            # Use eval_height or the smaller of the two video heights
            if not eval_height:
                # Get both video resolutions
                from ..ffmpeg_utils import FFmpegUtils
                ref_w, ref_h = FFmpegUtils.get_video_resolution(ref_path)
                dist_w, dist_h = FFmpegUtils.get_video_resolution(dist_path)
                eval_height = min(ref_h, dist_h)
            
            # Calculate adaptive timeout based on video characteristics
            adaptive_timeout = self._calculate_vmaf_timeout(ref_path, dist_path, eval_height)
            logger.debug(f"VMAF computation timeout set to {adaptive_timeout}s based on video characteristics")
            
            # Build filter chain: scale both to same resolution, then libvmaf
            scale_filter = f"scale=-2:{eval_height}:flags=bicubic,"
            
            # Create unique session ID for debug files
            session_id = self._generate_thread_safe_session_id()
            
            # Strategy 1: Try JSON format first (most reliable parsing)
            vmaf_debug_file = self.vmaf_filter_builder.get_safe_log_filename("vmaf", session_id, "json")
            debug_files_created.append(vmaf_debug_file)
            
            # Build filter complex using VMAFFilterBuilder
            filter_complex = self.vmaf_filter_builder.build_complete_filter_complex(
                ref_path, dist_path, vmaf_debug_file, "json", eval_height, 4
            )
            
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', dist_path,
                '-i', ref_path,
                '-filter_complex', filter_complex,
                '-f', 'null', '-'
            ]
            
            # Execute with adaptive timeout and enhanced error handling
            exec_result = self._execute_ffmpeg_with_retry(cmd, "VMAF computation (JSON legacy)", adaptive_timeout)
            
            if exec_result['success']:
                # Use robust parsing with multiple strategies
                vmaf_score = self._parse_vmaf_output(exec_result['stderr'])
                validated_score = self._validate_vmaf_score(vmaf_score)
                if validated_score is not None:
                    logger.info(f"VMAF computation successful (JSON format): {validated_score:.2f}")
                    # Clean up debug files on success
                    self.debug_log_manager.cleanup_debug_files(session_id, ["vmaf"])
                    return validated_score
                
                logger.debug("VMAF JSON parsing failed, trying XML format fallback")
            else:
                # Enhanced error classification for JSON attempt
                error_details = exec_result['error_details']
                if error_details.get('category') == 'timeout':
                    logger.warning(f"VMAF JSON computation timed out after {adaptive_timeout}s")
                elif error_details.get('category') == 'vmaf_not_available':
                    logger.error("VMAF filter not available - cannot compute quality scores")
                    self.debug_log_manager.cleanup_debug_files(session_id, ["vmaf"])
                    return None
                else:
                    logger.debug(f"VMAF JSON format failed: {error_details.get('description', 'Unknown error')}")
            
            # Strategy 2: Fallback to XML format if JSON failed
            vmaf_debug_file_xml = self.vmaf_filter_builder.get_safe_log_filename("vmaf", session_id, "xml")
            debug_files_created.append(vmaf_debug_file_xml)
            
            # Build XML filter complex using VMAFFilterBuilder
            filter_complex_xml = self.vmaf_filter_builder.build_complete_filter_complex(
                ref_path, dist_path, vmaf_debug_file_xml, "xml", eval_height, 4
            )
            
            cmd_xml = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', dist_path,
                '-i', ref_path,
                '-filter_complex', filter_complex_xml,
                '-f', 'null', '-'
            ]
            
            # Execute XML format with adaptive timeout
            exec_result_xml = self._execute_ffmpeg_with_retry(cmd_xml, "VMAF computation (XML legacy)", adaptive_timeout)
            
            if exec_result_xml['success']:
                vmaf_score = self._parse_vmaf_output(exec_result_xml['stderr'])
                validated_score = self._validate_vmaf_score(vmaf_score)
                if validated_score is not None:
                    logger.info(f"VMAF computation successful (XML format): {validated_score:.2f}")
                    # Clean up debug files on success
                    self.debug_log_manager.cleanup_debug_files(session_id, ["vmaf"])
                    return validated_score
                
                logger.debug("VMAF XML parsing failed, trying text format fallback")
            else:
                # Enhanced error classification for XML attempt
                error_details = exec_result_xml['error_details']
                if error_details.get('category') == 'timeout':
                    logger.warning(f"VMAF XML computation timed out after {adaptive_timeout}s")
                else:
                    logger.debug(f"VMAF XML format failed: {error_details.get('description', 'Unknown error')}")
            
            # Strategy 3: Final fallback to text format if both JSON and XML failed
            vmaf_debug_file_text = self.vmaf_filter_builder.get_safe_log_filename("vmaf", session_id, "text")
            debug_files_created.append(vmaf_debug_file_text)
            
            # Build text filter complex using VMAFFilterBuilder
            filter_complex_fallback = self.vmaf_filter_builder.build_complete_filter_complex(
                ref_path, dist_path, vmaf_debug_file_text, "text", eval_height, 4
            )
            
            cmd_fallback = [
                'ffmpeg', '-hide_banner', '-loglevel', 'info',  # More verbose for text parsing
                '-i', dist_path,
                '-i', ref_path,
                '-filter_complex', filter_complex_fallback,
                '-f', 'null', '-'
            ]
            
            # Execute fallback with adaptive timeout
            exec_result_fallback = self._execute_ffmpeg_with_retry(cmd_fallback, "VMAF computation (text legacy)", adaptive_timeout)
            
            if exec_result_fallback['success']:
                vmaf_score = self._parse_vmaf_output(exec_result_fallback['stderr'])
                validated_score = self._validate_vmaf_score(vmaf_score)
                if validated_score is not None:
                    logger.info(f"VMAF computation successful (text format): {validated_score:.2f}")
                    # Clean up debug files on success
                    self.debug_log_manager.cleanup_debug_files(session_id, ["vmaf"])
                    return validated_score
            else:
                # Enhanced error classification for text attempt
                error_details = exec_result_fallback['error_details']
                if error_details.get('category') == 'timeout':
                    logger.warning(f"VMAF text computation timed out after {adaptive_timeout}s")
            
            # All attempts failed - provide comprehensive error analysis
            self._log_comprehensive_vmaf_failure(
                [exec_result, exec_result_xml, exec_result_fallback],
                adaptive_timeout,
                ref_path,
                dist_path,
                eval_height
            )
            
            # Clean up debug files on failure
            self.debug_log_manager.cleanup_debug_files(session_id, ["vmaf"])
            
            return None
            
        except Exception as e:
            logger.error(f"VMAF computation system error: {e}")
            import traceback
            logger.debug(f"VMAF computation exception traceback: {traceback.format_exc()}")
            
            # Clean up debug files on exception
            try:
                self.debug_log_manager.cleanup_debug_files(session_id, ["vmaf"])
            except Exception as cleanup_error:
                logger.debug(f"Failed to cleanup debug files after exception: {cleanup_error}")
            
            return None
    
    def _calculate_vmaf_timeout(self, ref_path: str, dist_path: str, eval_height: int) -> int:
        """Calculate adaptive timeout for VMAF computation based on video characteristics.
        
        Args:
            ref_path: Reference video path
            dist_path: Distorted video path  
            eval_height: Evaluation height
            
        Returns:
            Timeout in seconds
        """
        try:
            from ..ffmpeg_utils import FFmpegUtils
            
            # Get video durations
            ref_duration = FFmpegUtils.get_video_duration(ref_path)
            dist_duration = FFmpegUtils.get_video_duration(dist_path)
            max_duration = max(ref_duration, dist_duration)
            
            # Base timeout calculation: ~2-3 seconds per second of video
            base_timeout = max_duration * 2.5
            
            # Adjust for resolution (higher resolution = more processing time)
            if eval_height >= 1080:
                resolution_multiplier = 2.0
            elif eval_height >= 720:
                resolution_multiplier = 1.5
            elif eval_height >= 480:
                resolution_multiplier = 1.2
            else:
                resolution_multiplier = 1.0
            
            # Apply resolution multiplier
            adjusted_timeout = base_timeout * resolution_multiplier
            
            # Apply configured minimum and maximum bounds
            min_timeout = self.vmaf_config.get('timeout_seconds', 300)
            max_timeout = min_timeout * 3  # Allow up to 3x the configured timeout
            
            final_timeout = max(min_timeout, min(int(adjusted_timeout), max_timeout))
            
            logger.debug(f"VMAF timeout calculation: duration={max_duration:.1f}s, "
                        f"height={eval_height}px, base={base_timeout:.1f}s, "
                        f"adjusted={adjusted_timeout:.1f}s, final={final_timeout}s")
            
            return final_timeout
            
        except Exception as e:
            logger.debug(f"Failed to calculate adaptive VMAF timeout: {e}")
            # Fallback to configured timeout
            return self.vmaf_config.get('timeout_seconds', 300)
    
    def _log_comprehensive_vmaf_failure(self, exec_results: List[Dict[str, Any]], 
                                       timeout_used: int, ref_path: str, 
                                       dist_path: str, eval_height: int) -> None:
        """Log comprehensive VMAF failure analysis for enhanced diagnostics.
        
        Args:
            exec_results: List of execution results from all attempts
            timeout_used: Timeout value that was used
            ref_path: Reference video path
            dist_path: Distorted video path
            eval_height: Evaluation height used
        """
        logger.error("=== COMPREHENSIVE VMAF FAILURE ANALYSIS ===")
        logger.error(f"Reference video: {os.path.basename(ref_path)}")
        logger.error(f"Distorted video: {os.path.basename(dist_path)}")
        logger.error(f"Evaluation height: {eval_height}px")
        logger.error(f"Timeout used: {timeout_used}s")
        
        # Analyze failure patterns
        failure_categories = []
        timeout_count = 0
        retryable_count = 0
        
        for i, result in enumerate(exec_results):
            format_name = ['JSON', 'XML', 'Text'][i]
            error_details = result.get('error_details', {})
            category = error_details.get('category', 'unknown')
            
            failure_categories.append(category)
            
            if category == 'timeout':
                timeout_count += 1
            elif error_details.get('is_retryable', False):
                retryable_count += 1
            
            logger.error(f"{format_name} attempt: {category} "
                        f"(retryable: {error_details.get('is_retryable', False)}, "
                        f"severity: {error_details.get('severity', 'unknown')})")
        
        # Provide failure analysis and recommendations
        if timeout_count == len(exec_results):
            logger.error("ANALYSIS: All attempts timed out - video may be too complex or system overloaded")
            logger.error("RECOMMENDATION: Try with lower resolution or increase timeout configuration")
        elif 'vmaf_not_available' in failure_categories:
            logger.error("ANALYSIS: VMAF filter not available in FFmpeg build")
            logger.error("RECOMMENDATION: Install FFmpeg with libvmaf support or use SSIM-only quality evaluation")
        elif retryable_count > 0:
            logger.error("ANALYSIS: Transient failures detected - may succeed on retry")
            logger.error("RECOMMENDATION: System resources may be constrained")
        else:
            logger.error("ANALYSIS: Persistent failures across all formats")
            logger.error("RECOMMENDATION: Check video file compatibility and FFmpeg installation")
        
        logger.error("=== END VMAF FAILURE ANALYSIS ===")
        
        # Log to debug level for additional context
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("VMAF failure context:")
            logger.debug(f"  Failure categories: {failure_categories}")
            logger.debug(f"  Timeout attempts: {timeout_count}/{len(exec_results)}")
            logger.debug(f"  Retryable attempts: {retryable_count}/{len(exec_results)}")
            
            # Log execution times
            for i, result in enumerate(exec_results):
                format_name = ['JSON', 'XML', 'Text'][i]
                exec_time = result.get('execution_time', 0)
                attempts = result.get('attempts', 0)
                logger.debug(f"  {format_name}: {exec_time:.2f}s over {attempts} attempts")
    
    def _validate_ssim_scores(self, scores: List[float]) -> Tuple[List[float], float]:
        """Validate and sanitize SSIM scores with outlier detection.
        
        Args:
            scores: List of raw SSIM scores
            
        Returns:
            Tuple of (validated_scores, confidence_score)
        """
        if not scores:
            return [], 0.0
        
        # Basic range validation (SSIM should be 0-1)
        valid_scores = [score for score in scores if 0 <= score <= 1]
        
        if not valid_scores:
            logger.warning(f"No valid SSIM scores found in range [0,1] from {len(scores)} raw scores")
            return [], 0.0
        
        # Outlier detection using IQR method (if enabled)
        if (self.ssim_config['enable_outlier_detection'] and 
            len(valid_scores) >= self.ssim_config['min_samples_for_outlier_detection']):
            sorted_scores = sorted(valid_scores)
            n = len(sorted_scores)
            q1_idx = n // 4
            q3_idx = 3 * n // 4
            q1 = sorted_scores[q1_idx]
            q3 = sorted_scores[q3_idx]
            iqr = q3 - q1
            
            # Define outlier bounds using configured multiplier
            iqr_multiplier = self.ssim_config['iqr_multiplier']
            lower_bound = q1 - iqr_multiplier * iqr
            upper_bound = q3 + iqr_multiplier * iqr
            
            # Filter outliers
            filtered_scores = [score for score in valid_scores 
                             if lower_bound <= score <= upper_bound]
            
            outliers_removed = len(valid_scores) - len(filtered_scores)
            if outliers_removed > 0:
                logger.debug(f"Removed {outliers_removed} SSIM outliers outside [{lower_bound:.4f}, {upper_bound:.4f}]")
            
            final_scores = filtered_scores if filtered_scores else valid_scores
        else:
            final_scores = valid_scores
        
        # Calculate confidence based on:
        # 1. Percentage of valid scores from raw input
        # 2. Consistency (low standard deviation)
        # 3. Number of samples
        
        validity_ratio = len(valid_scores) / len(scores)
        
        if len(final_scores) > 1:
            import statistics
            std_dev = statistics.stdev(final_scores)
            consistency_score = max(0, 1 - (std_dev * 10))  # Penalize high variance
        else:
            consistency_score = 0.5  # Moderate confidence for single score
        
        sample_size_score = min(1.0, len(final_scores) / 10)  # Full confidence at 10+ samples
        
        confidence = (validity_ratio * 0.4 + consistency_score * 0.4 + sample_size_score * 0.2)
        
        logger.debug(f"SSIM validation: {len(final_scores)}/{len(scores)} scores, "
                    f"confidence={confidence:.3f} (validity={validity_ratio:.3f}, "
                    f"consistency={consistency_score:.3f}, samples={sample_size_score:.3f})")
        
        return final_scores, confidence

    def _parse_ssim_output(self, stderr_output: str) -> Tuple[Optional[float], float]:
        """Robust SSIM parsing with multiple fallback strategies and validation.
        
        Args:
            stderr_output: FFmpeg stderr output containing SSIM statistics
            
        Returns:
            Tuple of (average_ssim_score, confidence_score)
        """
        parsing_attempts = []
        all_raw_scores = []
        
        # Strategy 1: Standard format - "n:X Y:0.xxxxx U:... V:... All:0.xxxxx (XXdB)"
        try:
            self._log_parsing_attempt("SSIM", "strategy1_standard", False, "Starting standard format parsing")
            
            pattern1 = re.compile(r'n:\d+\s+Y:([\d.]+)')
            matches = pattern1.findall(stderr_output)
            if matches:
                scores = [float(match) for match in matches]
                all_raw_scores.extend(scores)
                parsing_attempts.append(f"strategy1_success:{len(matches)}_frames")
                self._log_parsing_attempt("SSIM", "strategy1_standard", True, 
                                        f"Found {len(matches)} Y-channel scores")
            else:
                parsing_attempts.append("strategy1_no_matches")
                self._log_parsing_attempt("SSIM", "strategy1_standard", False, "No Y-channel matches found")
                
        except Exception as e:
            parsing_attempts.append(f"strategy1_error:{str(e)[:50]}")
            self._log_parsing_attempt("SSIM", "strategy1_standard", False, f"Exception: {str(e)[:100]}")
        
        # Strategy 2: Alternative format - look for "Y:" followed by decimal
        if not all_raw_scores:  # Only try if strategy 1 failed
            try:
                self._log_parsing_attempt("SSIM", "strategy2_alternative", False, "Starting alternative Y: format parsing")
                
                pattern2 = re.compile(r'Y:\s*([\d.]+)')
                matches = pattern2.findall(stderr_output)
                if matches:
                    scores = [float(match) for match in matches]
                    all_raw_scores.extend(scores)
                    parsing_attempts.append(f"strategy2_success:{len(matches)}_y_scores")
                    self._log_parsing_attempt("SSIM", "strategy2_alternative", True, 
                                            f"Found {len(matches)} Y scores")
                else:
                    parsing_attempts.append("strategy2_no_matches")
                    self._log_parsing_attempt("SSIM", "strategy2_alternative", False, "No Y: matches found")
                    
            except Exception as e:
                parsing_attempts.append(f"strategy2_error:{str(e)[:50]}")
                self._log_parsing_attempt("SSIM", "strategy2_alternative", False, f"Exception: {str(e)[:100]}")
        
        # Strategy 3: Look for "All:" values which represent overall SSIM
        if not all_raw_scores:  # Only try if previous strategies failed
            try:
                pattern3 = re.compile(r'All:\s*([\d.]+)')
                matches = pattern3.findall(stderr_output)
                if matches:
                    scores = [float(match) for match in matches]
                    all_raw_scores.extend(scores)
                    parsing_attempts.append(f"strategy3_success:{len(matches)}_all_scores")
                    logger.debug(f"SSIM strategy 3 found {len(matches)} All scores")
                else:
                    parsing_attempts.append("strategy3_no_matches")
            except Exception as e:
                parsing_attempts.append(f"strategy3_error:{str(e)[:50]}")
        
        # Strategy 4: Line-by-line parsing with fuzzy matching (last resort)
        if not all_raw_scores:
            try:
                lines = stderr_output.split('\n')
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Look for lines containing SSIM-like patterns
                    if any(keyword in line.lower() for keyword in ['ssim', 'y:', 'n:']):
                        # Extract all decimal numbers from the line
                        decimals = re.findall(r'\b(0\.\d+|\d+\.\d+)\b', line)
                        for decimal in decimals:
                            try:
                                score = float(decimal)
                                # Only accept reasonable SSIM values
                                if 0 <= score <= 1:
                                    all_raw_scores.append(score)
                            except ValueError:
                                continue
                
                if all_raw_scores:
                    parsing_attempts.append(f"strategy4_success:{len(all_raw_scores)}_fuzzy_scores")
                    logger.debug(f"SSIM strategy 4 (fuzzy) found {len(all_raw_scores)} scores")
                else:
                    parsing_attempts.append("strategy4_no_scores_found")
            except Exception as e:
                parsing_attempts.append(f"strategy4_error:{str(e)[:50]}")
        
        # Validate and sanitize the collected scores
        if all_raw_scores:
            validated_scores, confidence = self._validate_ssim_scores(all_raw_scores)
            
            if validated_scores:
                avg_score = sum(validated_scores) / len(validated_scores)
                logger.debug(f"SSIM final result: {avg_score:.4f} from {len(validated_scores)} validated scores, "
                           f"confidence={confidence:.3f}")
                return avg_score, confidence
            else:
                logger.warning(f"All {len(all_raw_scores)} SSIM scores failed validation")
                return None, 0.0
        
        # All strategies failed
        logger.warning(f"SSIM parsing failed with all strategies. Attempts: {parsing_attempts}")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"SSIM stderr output for debugging:\n{stderr_output}")
        
        return None, 0.0

    def _compute_ssim(self, ref_path: str, dist_path: str,
                     eval_height: Optional[int] = None) -> Tuple[Optional[float], float]:
        """Compute SSIM score using FFmpeg ssim filter with resolution scaling support.
        
        Returns:
            Tuple of (ssim_score, confidence_score)
        """
        try:
            # Create session ID for temporary files
            session_id = self._generate_thread_safe_session_id()
            
            # Initialize resolution-aware evaluator
            resolution_evaluator = ResolutionAwareQualityEvaluator(self.config)
            
            try:
                # Use resolution-aware SSIM computation
                ssim_score, confidence, scaling_info = resolution_evaluator.compute_ssim_with_scaling(
                    ref_path, dist_path, session_id
                )
                
                # Log scaling decisions
                if scaling_info['scaling_applied']:
                    logger.info(f"SSIM resolution scaling applied: "
                               f"{scaling_info['original_resolution']} → {scaling_info['comparison_resolution']}")
                    
                    # Log scaled files for debugging
                    if scaling_info['scaled_files']:
                        logger.debug(f"Temporary scaled files created: {len(scaling_info['scaled_files'])}")
                
                return ssim_score, confidence
                
            finally:
                # Always clean up temporary files
                resolution_evaluator.cleanup_temporary_files()
            
        except Exception as e:
            logger.error(f"SSIM computation with scaling failed: {e}")
            
            # Fallback to legacy method if scaling fails
            logger.debug("Falling back to legacy SSIM computation method")
            return self._compute_ssim_legacy(ref_path, dist_path, eval_height)
    
    def _compute_ssim_legacy(self, ref_path: str, dist_path: str,
                            eval_height: Optional[int] = None) -> Tuple[Optional[float], float]:
        """Legacy SSIM computation method (fallback for compatibility).
        
        Returns:
            Tuple of (ssim_score, confidence_score)
        """
        try:
            # Always scale both videos to same resolution for comparison
            if not eval_height:
                # Get both video resolutions
                from ..ffmpeg_utils import FFmpegUtils
                ref_w, ref_h = FFmpegUtils.get_video_resolution(ref_path)
                dist_w, dist_h = FFmpegUtils.get_video_resolution(dist_path)
                eval_height = min(ref_h, dist_h)
            
            # Build filter chain: scale both to same resolution
            scale_filter = f"scale=-2:{eval_height}:flags=bicubic,"
            
            filter_complex = (
                f"[0:v]{scale_filter}setpts=PTS-STARTPTS[dist];"
                f"[1:v]{scale_filter}setpts=PTS-STARTPTS[ref];"
                f"[dist][ref]ssim=stats_file=-"
            )
            
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', dist_path,
                '-i', ref_path,
                '-filter_complex', filter_complex,
                '-f', 'null', '-'
            ]
            
            # Execute with robust error handling
            exec_result = self._execute_ffmpeg_with_retry(cmd, "SSIM computation (legacy)")
            
            if exec_result['success']:
                # Use robust parsing with multiple strategies and validation
                return self._parse_ssim_output(exec_result['stderr'])
            else:
                # Log detailed error information
                if self.ffmpeg_config['enable_detailed_errors']:
                    logger.error(f"SSIM computation failed: {exec_result['error_details']}")
                else:
                    logger.warning(f"SSIM computation failed: {exec_result['error_details'].get('description', 'Unknown error')}")
                
                return None, 0.0
            
        except Exception as e:
            logger.error(f"SSIM computation system error: {e}")
            return None, 0.0


class ResolutionAwareQualityEvaluator:
    """Handles quality evaluation with automatic resolution scaling for SSIM and VMAF computation."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        self.temp_files_created = []
        
    def detect_resolution_mismatch(self, original_path: str, compressed_path: str) -> Dict[str, Any]:
        """Detect resolution mismatch between original and compressed videos.
        
        Args:
            original_path: Path to original video
            compressed_path: Path to compressed video
            
        Returns:
            Dictionary with resolution information and mismatch details
        """
        try:
            from ..ffmpeg_utils import FFmpegUtils
            
            # Get resolutions
            orig_w, orig_h = FFmpegUtils.get_video_resolution(original_path)
            comp_w, comp_h = FFmpegUtils.get_video_resolution(compressed_path)
            
            # Calculate aspect ratios
            orig_aspect = orig_w / orig_h if orig_h > 0 else 1.0
            comp_aspect = comp_w / comp_h if comp_h > 0 else 1.0
            
            # Determine if there's a mismatch
            resolution_mismatch = (orig_w != comp_w) or (orig_h != comp_h)
            aspect_mismatch = abs(orig_aspect - comp_aspect) > 0.01  # 1% tolerance
            
            # Determine target resolution (use smaller to avoid upscaling artifacts)
            if orig_w * orig_h <= comp_w * comp_h:
                target_w, target_h = orig_w, orig_h
                smaller_video = "original"
            else:
                target_w, target_h = comp_w, comp_h
                smaller_video = "compressed"
            
            result = {
                'original_resolution': (orig_w, orig_h),
                'compressed_resolution': (comp_w, comp_h),
                'original_aspect_ratio': orig_aspect,
                'compressed_aspect_ratio': comp_aspect,
                'resolution_mismatch': resolution_mismatch,
                'aspect_mismatch': aspect_mismatch,
                'target_resolution': (target_w, target_h),
                'smaller_video': smaller_video,
                'scaling_needed': resolution_mismatch
            }
            
            # Log detailed resolution analysis
            self._log_resolution_analysis(result)
            
            return result
            
        except Exception as e:
            logger.error(f"Resolution detection failed: {e}")
            # Return safe defaults
            return {
                'original_resolution': (1920, 1080),
                'compressed_resolution': (1920, 1080),
                'original_aspect_ratio': 16/9,
                'compressed_aspect_ratio': 16/9,
                'resolution_mismatch': False,
                'aspect_mismatch': False,
                'target_resolution': (1920, 1080),
                'smaller_video': 'original',
                'scaling_needed': False
            }
    
    def _log_resolution_analysis(self, analysis: Dict[str, Any]) -> None:
        """Log detailed resolution analysis for quality evaluation decisions.
        
        Args:
            analysis: Resolution analysis result dictionary
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        
        logger.debug("=== RESOLUTION ANALYSIS FOR QUALITY EVALUATION ===")
        
        orig_w, orig_h = analysis['original_resolution']
        comp_w, comp_h = analysis['compressed_resolution']
        target_w, target_h = analysis['target_resolution']
        
        logger.debug(f"Original video: {orig_w}x{orig_h} ({orig_w * orig_h:,} pixels)")
        logger.debug(f"Compressed video: {comp_w}x{comp_h} ({comp_w * comp_h:,} pixels)")
        logger.debug(f"Target resolution: {target_w}x{target_h} ({target_w * target_h:,} pixels)")
        
        # Log aspect ratio information
        orig_aspect = analysis['original_aspect_ratio']
        comp_aspect = analysis['compressed_aspect_ratio']
        logger.debug(f"Aspect ratios: original={orig_aspect:.3f}, compressed={comp_aspect:.3f}")
        
        # Log mismatch analysis
        if analysis['resolution_mismatch']:
            logger.debug("Resolution mismatch detected - scaling will be required")
            
            # Calculate resolution difference percentages
            orig_pixels = orig_w * orig_h
            comp_pixels = comp_w * comp_h
            pixel_diff_pct = abs(orig_pixels - comp_pixels) / max(orig_pixels, comp_pixels) * 100
            
            logger.debug(f"Pixel count difference: {pixel_diff_pct:.1f}%")
            
            if analysis['aspect_mismatch']:
                aspect_diff_pct = abs(orig_aspect - comp_aspect) / max(orig_aspect, comp_aspect) * 100
                logger.debug(f"Aspect ratio difference: {aspect_diff_pct:.1f}%")
                logger.debug("WARNING: Aspect ratio mismatch may indicate encoding issues")
            
            # Log scaling strategy
            smaller_video = analysis['smaller_video']
            logger.debug(f"Scaling strategy: Use {smaller_video} video resolution as target")
            logger.debug(f"Reason: Avoid upscaling artifacts by using smaller resolution")
            
            # Log scaling requirements
            orig_needs_scaling = (orig_w, orig_h) != (target_w, target_h)
            comp_needs_scaling = (comp_w, comp_h) != (target_w, target_h)
            
            logger.debug("Scaling requirements:")
            logger.debug(f"  Original video needs scaling: {orig_needs_scaling}")
            logger.debug(f"  Compressed video needs scaling: {comp_needs_scaling}")
            
        else:
            logger.debug("No resolution mismatch - direct comparison possible")
        
        logger.debug("=== END RESOLUTION ANALYSIS ===")
    
    def _log_quality_scaling_decision(self, metric_type: str, resolution_analysis: Dict[str, Any], 
                                     target_resolution: Tuple[int, int]) -> None:
        """Log detailed scaling decisions during quality evaluation.
        
        Args:
            metric_type: Type of quality metric (SSIM, VMAF)
            resolution_analysis: Resolution analysis results
            target_resolution: Target resolution for scaling
        """
        logger.info(f"=== {metric_type} RESOLUTION SCALING DECISION ===")
        
        orig_res = resolution_analysis['original_resolution']
        comp_res = resolution_analysis['compressed_resolution']
        target_w, target_h = target_resolution
        
        logger.info(f"Input resolutions:")
        logger.info(f"  Original: {orig_res[0]}x{orig_res[1]} ({orig_res[0] * orig_res[1]:,} pixels)")
        logger.info(f"  Compressed: {comp_res[0]}x{comp_res[1]} ({comp_res[0] * comp_res[1]:,} pixels)")
        logger.info(f"Target resolution: {target_w}x{target_h} ({target_w * target_h:,} pixels)")
        
        # Log scaling strategy reasoning
        smaller_video = resolution_analysis['smaller_video']
        logger.info(f"Scaling strategy: Use {smaller_video} video resolution as target")
        
        # Log quality impact assessment
        orig_pixels = orig_res[0] * orig_res[1]
        comp_pixels = comp_res[0] * comp_res[1]
        target_pixels = target_w * target_h
        
        orig_impact = (target_pixels - orig_pixels) / orig_pixels * 100 if orig_pixels > 0 else 0
        comp_impact = (target_pixels - comp_pixels) / comp_pixels * 100 if comp_pixels > 0 else 0
        
        logger.info(f"Quality impact assessment:")
        logger.info(f"  Original video: {orig_impact:+.1f}% pixel change")
        logger.info(f"  Compressed video: {comp_impact:+.1f}% pixel change")
        
        # Log aspect ratio considerations
        if resolution_analysis['aspect_mismatch']:
            orig_aspect = resolution_analysis['original_aspect_ratio']
            comp_aspect = resolution_analysis['compressed_aspect_ratio']
            logger.info(f"Aspect ratio mismatch detected:")
            logger.info(f"  Original: {orig_aspect:.3f}, Compressed: {comp_aspect:.3f}")
            logger.info(f"  Scaling will preserve aspect ratios using padding if necessary")
        
        # Log metric-specific considerations
        if metric_type == "SSIM":
            logger.info(f"SSIM considerations:")
            logger.info(f"  - SSIM requires identical resolutions for comparison")
            logger.info(f"  - Scaling may slightly affect SSIM accuracy")
            logger.info(f"  - Using bicubic interpolation for best quality preservation")
        elif metric_type == "VMAF":
            logger.info(f"VMAF considerations:")
            logger.info(f"  - VMAF can handle different resolutions but scaling improves accuracy")
            logger.info(f"  - Scaling reduces computational complexity")
            logger.info(f"  - Target resolution chosen to minimize quality loss")
        
        logger.info(f"=== END {metric_type} SCALING DECISION ===")
    
    def _log_scaling_decision(self, video_type: str, original_res: Tuple[int, int], 
                             target_res: Tuple[int, int], scaling_method: str) -> None:
        """Log resolution scaling decisions during quality evaluation.
        
        Args:
            video_type: Type of video being scaled ('original' or 'compressed')
            original_res: Original video resolution
            target_res: Target resolution for scaling
            scaling_method: Scaling method being used
        """
        orig_w, orig_h = original_res
        target_w, target_h = target_res
        
        logger.info(f"SCALING DECISION: {video_type} video")
        logger.info(f"  From: {orig_w}x{orig_h} ({orig_w * orig_h:,} pixels)")
        logger.info(f"  To: {target_w}x{target_h} ({target_w * target_h:,} pixels)")
        logger.info(f"  Method: {scaling_method}")
        
        # Calculate scaling factors
        scale_factor_w = target_w / orig_w if orig_w > 0 else 1.0
        scale_factor_h = target_h / orig_h if orig_h > 0 else 1.0
        
        logger.info(f"  Scale factors: width={scale_factor_w:.3f}, height={scale_factor_h:.3f}")
        
        # Determine scaling type
        if scale_factor_w > 1.0 or scale_factor_h > 1.0:
            logger.info("  Scaling type: UPSCALING (may introduce artifacts)")
        elif scale_factor_w < 1.0 or scale_factor_h < 1.0:
            logger.info("  Scaling type: DOWNSCALING (preserves quality)")
        else:
            logger.info("  Scaling type: NO SCALING (same resolution)")
        
        # Log quality impact assessment
        pixel_reduction = 1.0 - (target_w * target_h) / (orig_w * orig_h)
        if pixel_reduction > 0:
            logger.info(f"  Quality impact: {pixel_reduction * 100:.1f}% pixel reduction")
        elif pixel_reduction < 0:
            logger.info(f"  Quality impact: {abs(pixel_reduction) * 100:.1f}% pixel increase (upscaling)")
        else:
            logger.info("  Quality impact: No pixel count change")
    
    def scale_video_for_comparison(self, video_path: str, target_resolution: Tuple[int, int], 
                                  session_id: str) -> Optional[str]:
        """Scale video to target resolution for quality comparison.
        
        Args:
            video_path: Path to video to scale
            target_resolution: Target (width, height)
            session_id: Unique session ID for temporary file naming
            
        Returns:
            Path to scaled video file or None on error
        """
        try:
            target_w, target_h = target_resolution
            
            # Generate temporary scaled video path
            import tempfile
            import os
            from ..logger_setup import get_default_logs_dir
            
            temp_dir = get_default_logs_dir()
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create temporary file with proper extension
            video_ext = os.path.splitext(video_path)[1] or '.mp4'
            temp_filename = f"scaled_video_{session_id}_{target_w}x{target_h}{video_ext}"
            temp_path = os.path.join(temp_dir, temp_filename)
            
            # Build scaling command with aspect ratio preservation
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', video_path,
                '-vf', f'scale={target_w}:{target_h}:flags=bicubic:force_original_aspect_ratio=decrease,pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2',
                '-c:v', 'libx264', '-preset', 'fast', '-crf', '18',
                '-an',  # No audio needed for quality comparison
                '-y',   # Overwrite output file
                temp_path
            ]
            
            # Log scaling decision with detailed information
            from ..ffmpeg_utils import FFmpegUtils
            original_w, original_h = FFmpegUtils.get_video_resolution(video_path)
            self._log_scaling_decision("video", (original_w, original_h), target_resolution, "bicubic with aspect ratio preservation")
            
            # Execute scaling command
            import subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout for scaling
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode == 0 and os.path.exists(temp_path):
                # Track temporary file for cleanup
                self.temp_files_created.append(temp_path)
                logger.debug(f"Video scaled successfully: {os.path.basename(temp_path)}")
                return temp_path
            else:
                logger.error(f"Video scaling failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Video scaling error: {e}")
            return None
    
    def compute_ssim_with_scaling(self, ref_path: str, dist_path: str, 
                                 session_id: str) -> Tuple[Optional[float], float, Dict[str, Any]]:
        """Compute SSIM with automatic resolution scaling if needed.
        
        Args:
            ref_path: Reference video path
            dist_path: Distorted video path
            session_id: Unique session ID for temporary files
            
        Returns:
            Tuple of (ssim_score, confidence, scaling_info)
        """
        scaling_info = {
            'scaling_applied': False,
            'original_resolution': None,
            'comparison_resolution': None,
            'scaled_files': []
        }
        
        try:
            # Detect resolution mismatch
            resolution_analysis = self.detect_resolution_mismatch(ref_path, dist_path)
            scaling_info['original_resolution'] = resolution_analysis['original_resolution']
            scaling_info['comparison_resolution'] = resolution_analysis['compressed_resolution']
            
            if not resolution_analysis['scaling_needed']:
                # No scaling needed, use original paths
                logger.debug("No resolution scaling needed for SSIM computation")
                ssim_score, confidence = self._compute_ssim_direct(ref_path, dist_path)
                return ssim_score, confidence, scaling_info
            
            # Scaling needed
            target_resolution = resolution_analysis['target_resolution']
            scaling_info['scaling_applied'] = True
            scaling_info['comparison_resolution'] = target_resolution
            
            # Log detailed scaling decision
            self._log_quality_scaling_decision("SSIM", resolution_analysis, target_resolution)
            
            # Determine which videos need scaling
            ref_needs_scaling = resolution_analysis['original_resolution'] != target_resolution
            dist_needs_scaling = resolution_analysis['compressed_resolution'] != target_resolution
            
            # Scale videos as needed
            scaled_ref_path = ref_path
            scaled_dist_path = dist_path
            
            if ref_needs_scaling:
                scaled_ref_path = self.scale_video_for_comparison(ref_path, target_resolution, f"{session_id}_ref")
                if scaled_ref_path:
                    scaling_info['scaled_files'].append(scaled_ref_path)
                else:
                    logger.error("Failed to scale reference video for SSIM")
                    return None, 0.0, scaling_info
            
            if dist_needs_scaling:
                scaled_dist_path = self.scale_video_for_comparison(dist_path, target_resolution, f"{session_id}_dist")
                if scaled_dist_path:
                    scaling_info['scaled_files'].append(scaled_dist_path)
                else:
                    logger.error("Failed to scale distorted video for SSIM")
                    return None, 0.0, scaling_info
            
            # Compute SSIM with scaled videos
            ssim_score, confidence = self._compute_ssim_direct(scaled_ref_path, scaled_dist_path)
            
            logger.debug(f"SSIM computed with scaling: {ssim_score}, confidence: {confidence}")
            return ssim_score, confidence, scaling_info
            
        except Exception as e:
            logger.error(f"SSIM computation with scaling failed: {e}")
            return None, 0.0, scaling_info
    
    def compute_vmaf_with_scaling(self, ref_path: str, dist_path: str, 
                                 session_id: str) -> Tuple[Optional[float], Dict[str, Any]]:
        """Compute VMAF with automatic resolution scaling if needed.
        
        Args:
            ref_path: Reference video path
            dist_path: Distorted video path
            session_id: Unique session ID for temporary files
            
        Returns:
            Tuple of (vmaf_score, scaling_info)
        """
        scaling_info = {
            'scaling_applied': False,
            'original_resolution': None,
            'comparison_resolution': None,
            'scaled_files': []
        }
        
        try:
            # Detect resolution mismatch
            resolution_analysis = self.detect_resolution_mismatch(ref_path, dist_path)
            scaling_info['original_resolution'] = resolution_analysis['original_resolution']
            scaling_info['comparison_resolution'] = resolution_analysis['compressed_resolution']
            
            if not resolution_analysis['scaling_needed']:
                # No scaling needed, use original paths
                logger.debug("No resolution scaling needed for VMAF computation")
                return self._compute_vmaf_direct(ref_path, dist_path), scaling_info
            
            # Scaling needed
            target_resolution = resolution_analysis['target_resolution']
            scaling_info['scaling_applied'] = True
            scaling_info['comparison_resolution'] = target_resolution
            
            # Log detailed scaling decision
            self._log_quality_scaling_decision("VMAF", resolution_analysis, target_resolution)
            
            # Determine which videos need scaling
            ref_needs_scaling = resolution_analysis['original_resolution'] != target_resolution
            dist_needs_scaling = resolution_analysis['compressed_resolution'] != target_resolution
            
            # Scale videos as needed
            scaled_ref_path = ref_path
            scaled_dist_path = dist_path
            
            if ref_needs_scaling:
                scaled_ref_path = self.scale_video_for_comparison(ref_path, target_resolution, f"{session_id}_ref")
                if scaled_ref_path:
                    scaling_info['scaled_files'].append(scaled_ref_path)
                else:
                    logger.error("Failed to scale reference video for VMAF")
                    return None, scaling_info
            
            if dist_needs_scaling:
                scaled_dist_path = self.scale_video_for_comparison(dist_path, target_resolution, f"{session_id}_dist")
                if scaled_dist_path:
                    scaling_info['scaled_files'].append(scaled_dist_path)
                else:
                    logger.error("Failed to scale distorted video for VMAF")
                    return None, scaling_info
            
            # Compute VMAF with scaled videos
            vmaf_score = self._compute_vmaf_direct(scaled_ref_path, scaled_dist_path)
            
            logger.debug(f"VMAF computed with scaling: {vmaf_score}")
            return vmaf_score, scaling_info
            
        except Exception as e:
            logger.error(f"VMAF computation with scaling failed: {e}")
            return None, scaling_info
    
    def _compute_ssim_direct(self, ref_path: str, dist_path: str) -> Tuple[Optional[float], float]:
        """Direct SSIM computation assuming videos have matching resolutions."""
        try:
            # Simple SSIM filter without scaling (videos should already be same resolution)
            filter_complex = (
                "[0:v]setpts=PTS-STARTPTS[dist];"
                "[1:v]setpts=PTS-STARTPTS[ref];"
                "[dist][ref]ssim=stats_file=-"
            )
            
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'error',
                '-i', dist_path,
                '-i', ref_path,
                '-filter_complex', filter_complex,
                '-f', 'null', '-'
            ]
            
            # Execute command
            import subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode == 0:
                ssim_output = result.stderr if result.stderr else result.stdout
                ssim_score, confidence = self._parse_ssim_output_simple(ssim_output or "")
                if ssim_score is None:
                    # Parsing failed - log detailed information
                    logger.error(f"SSIM parsing failed: Unable to parse FFmpeg output")
                    logger.debug(f"FFmpeg command: {' '.join(cmd)}")
                    logger.debug(f"FFmpeg returncode: {result.returncode}")
                    logger.debug(f"FFmpeg stdout length: {len(result.stdout or '')} chars")
                    logger.debug(f"FFmpeg stderr length: {len(result.stderr or '')} chars")
                    if result.stderr:
                        logger.debug(f"FFmpeg stderr output (first 1000 chars):\n{result.stderr[:1000]}")
                    elif result.stdout:
                        logger.debug(f"FFmpeg stdout output (first 1000 chars):\n{result.stdout[:1000]}")
                    else:
                        logger.debug("FFmpeg produced no SSIM output on stdout or stderr")
                return ssim_score, confidence
            else:
                logger.error(f"Direct SSIM computation failed (returncode {result.returncode})")
                logger.debug(f"FFmpeg command: {' '.join(cmd)}")
                if result.stderr:
                    logger.error(f"FFmpeg stderr: {result.stderr[:500]}")
                if result.stdout:
                    logger.debug(f"FFmpeg stdout: {result.stdout[:500]}")
                return None, 0.0
                
        except Exception as e:
            logger.error(f"Direct SSIM computation error: {e}", exc_info=True)
            return None, 0.0
    
    def _compute_vmaf_direct(self, ref_path: str, dist_path: str) -> Optional[float]:
        """Direct VMAF computation assuming videos have matching resolutions."""
        try:
            # Simple VMAF filter without scaling (videos should already be same resolution)
            filter_complex = (
                "[0:v]setpts=PTS-STARTPTS[dist];"
                "[1:v]setpts=PTS-STARTPTS[ref];"
                "[dist][ref]libvmaf"
            )
            
            cmd = [
                'ffmpeg', '-hide_banner', '-loglevel', 'info',
                '-i', dist_path,
                '-i', ref_path,
                '-filter_complex', filter_complex,
                '-f', 'null', '-'
            ]
            
            # Execute command
            import subprocess
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300,
                encoding='utf-8',
                errors='replace'
            )
            
            if result.returncode == 0:
                return self._parse_vmaf_output_simple(result.stderr)
            else:
                logger.error(f"Direct VMAF computation failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Direct VMAF computation error: {e}")
            return None
    
    def _parse_ssim_output_simple(self, stderr: str) -> Tuple[Optional[float], float]:
        """Simple SSIM output parsing."""
        try:
            import re
            # Look for SSIM Y value in stderr
            ssim_pattern = r'SSIM Y:([0-9.]+)'
            match = re.search(ssim_pattern, stderr)
            
            if match:
                ssim_score = float(match.group(1))
                if 0.0 <= ssim_score <= 1.0:
                    logger.debug(f"SSIM parsing successful: found score {ssim_score:.4f}")
                    return ssim_score, 0.9  # High confidence for successful parsing
                else:
                    logger.warning(f"SSIM parsing found invalid score: {ssim_score} (not in range 0-1)")
            
            # If no match found, try alternative patterns
            logger.debug(f"SSIM parsing: primary pattern not found, trying alternatives")
            
            # Try alternative patterns similar to _parse_ssim_output
            alternative_patterns = [
                (r'n:\d+\s+Y:([\d.]+)', 'strategy1_standard'),
                (r'Y:\s*([\d.]+)', 'strategy2_alternative'),
                (r'All:\s*([\d.]+)', 'strategy3_all'),
            ]
            
            for pattern, strategy_name in alternative_patterns:
                matches = re.findall(pattern, stderr)
                if matches:
                    try:
                        scores = [float(m) for m in matches]
                        valid_scores = [s for s in scores if 0.0 <= s <= 1.0]
                        if valid_scores:
                            avg_score = sum(valid_scores) / len(valid_scores)
                            logger.debug(f"SSIM parsing successful with {strategy_name}: found {len(valid_scores)} scores, avg={avg_score:.4f}")
                            return avg_score, 0.8  # Slightly lower confidence for alternative patterns
                    except (ValueError, ZeroDivisionError) as e:
                        logger.debug(f"SSIM parsing error with {strategy_name}: {e}")
                        continue
            
            # All parsing attempts failed
            logger.warning(f"SSIM parsing failed: No valid SSIM scores found in output")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"SSIM stderr output for debugging:\n{stderr[:500]}")
            return None, 0.0
            
        except Exception as e:
            logger.error(f"SSIM parsing error: {e}", exc_info=True)
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"SSIM stderr output that caused error:\n{stderr[:500]}")
            return None, 0.0
    
    def _parse_vmaf_output_simple(self, stderr: str) -> Optional[float]:
        """Simple VMAF output parsing."""
        try:
            import re
            # Look for VMAF score in stderr
            vmaf_patterns = [
                r'VMAF score:\s*([0-9.]+)',
                r'vmaf=([0-9.]+)',
                r'mean:\s*([0-9.]+)'
            ]
            
            for pattern in vmaf_patterns:
                match = re.search(pattern, stderr, re.IGNORECASE)
                if match:
                    vmaf_score = float(match.group(1))
                    if 0.0 <= vmaf_score <= 100.0:
                        return vmaf_score
            
            return None
            
        except Exception as e:
            logger.debug(f"VMAF parsing error: {e}")
            return None
    
    def cleanup_temporary_files(self) -> None:
        """Clean up all temporary scaled video files created during evaluation."""
        import time as time_module
        
        for temp_file in self.temp_files_created:
            try:
                if os.path.exists(temp_file):
                    # On Windows, files may be locked briefly after subprocess closes
                    # Retry with small delays to handle file locking
                    max_retries = 3
                    retry_delay = 0.1  # 100ms
                    
                    for attempt in range(max_retries):
                        try:
                            os.remove(temp_file)
                            logger.debug(f"Cleaned up temporary scaled video: {os.path.basename(temp_file)}")
                            break
                        except (OSError, PermissionError) as e:
                            if attempt < max_retries - 1:
                                time_module.sleep(retry_delay * (attempt + 1))
                            else:
                                # Last attempt failed, log but don't raise
                                logger.debug(f"Could not remove temporary file {temp_file} after {max_retries} attempts: {e}")
            except Exception as e:
                logger.debug(f"Error cleaning up temporary file {temp_file}: {e}")
        
        self.temp_files_created.clear()


class DebugLogManager:
    """Manages debug log file paths and ensures proper location in logs/ directory."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        self._logs_dir = None
    
    def get_logs_directory(self) -> str:
        """Get the logs directory path, creating it if necessary."""
        if self._logs_dir is None:
            from ..logger_setup import get_default_logs_dir
            self._logs_dir = get_default_logs_dir()
        
        # Ensure logs directory exists
        os.makedirs(self._logs_dir, exist_ok=True)
        return self._logs_dir
    
    def get_debug_log_path(self, log_type: str, session_id: str, format_ext: str = "log") -> str:
        """Generate a proper debug log file path in the logs/ directory.
        
        Args:
            log_type: Type of debug log (e.g., 'vmaf', 'ssim')
            session_id: Unique session identifier (timestamp or UUID)
            format_ext: File extension (e.g., 'json', 'xml', 'log')
            
        Returns:
            Full path to debug log file in logs/ directory
        """
        logs_dir = self.get_logs_directory()
        
        # Sanitize inputs to prevent path traversal
        safe_log_type = self._sanitize_filename_component(log_type)
        safe_session_id = self._sanitize_filename_component(session_id)
        safe_format_ext = self._sanitize_filename_component(format_ext)
        
        # Generate filename
        filename = f"{safe_log_type}_debug_{safe_session_id}.{safe_format_ext}"
        
        return os.path.join(logs_dir, filename)
    
    def _sanitize_filename_component(self, component: str) -> str:
        """Sanitize a filename component to prevent path traversal and invalid characters.
        
        Args:
            component: Raw filename component
            
        Returns:
            Sanitized filename component safe for filesystem use
        """
        if not component:
            return "unknown"
        
        # Remove path separators and dangerous characters
        import re
        # Keep only alphanumeric, underscore, hyphen, and dot
        sanitized = re.sub(r'[^a-zA-Z0-9_\-.]', '_', str(component))
        
        # Prevent empty result
        if not sanitized or sanitized == '.':
            sanitized = "unknown"
        
        # Limit length to prevent filesystem issues
        return sanitized[:50]
    
    def validate_log_path(self, path: str) -> bool:
        """Validate that a log path is safe and within the logs directory.
        
        Args:
            path: Path to validate
            
        Returns:
            True if path is valid and safe, False otherwise
        """
        try:
            # Resolve absolute paths
            abs_path = os.path.abspath(path)
            logs_dir = os.path.abspath(self.get_logs_directory())
            
            # Check if path is within logs directory
            return abs_path.startswith(logs_dir)
        except Exception:
            return False
    
    def cleanup_debug_files(self, session_id: str, log_types: Optional[List[str]] = None) -> None:
        """Clean up debug files for a specific session.
        
        Args:
            session_id: Session identifier to clean up
            log_types: Optional list of log types to clean up (default: all)
        """
        try:
            logs_dir = self.get_logs_directory()
            safe_session_id = self._sanitize_filename_component(session_id)
            
            # Build pattern to match debug files for this session
            if log_types:
                patterns = []
                for log_type in log_types:
                    safe_log_type = self._sanitize_filename_component(log_type)
                    patterns.append(f"{safe_log_type}_debug_{safe_session_id}.*")
            else:
                # Match all debug files for this session
                patterns = [f"*_debug_{safe_session_id}.*"]
            
            import glob
            files_removed = 0
            
            for pattern in patterns:
                full_pattern = os.path.join(logs_dir, pattern)
                matching_files = glob.glob(full_pattern)
                
                for file_path in matching_files:
                    try:
                        if os.path.exists(file_path):
                            os.remove(file_path)
                            files_removed += 1
                            logger.debug(f"Cleaned up debug file: {os.path.basename(file_path)}")
                    except Exception as e:
                        logger.debug(f"Could not remove debug file {file_path}: {e}")
            
            if files_removed > 0:
                logger.debug(f"Cleaned up {files_removed} debug files for session {session_id}")
                
        except Exception as e:
            logger.debug(f"Debug file cleanup failed for session {session_id}: {e}")


class VMAFFilterBuilder:
    """Builds properly formatted libvmaf filter parameters with cross-platform path handling."""
    
    def __init__(self, debug_log_manager: Optional[DebugLogManager] = None):
        self.debug_log_manager = debug_log_manager or DebugLogManager()
    
    def build_vmaf_filter(self, log_path: str, format_type: str = "json", 
                         n_threads: int = 4, model_path: Optional[str] = None) -> str:
        """Build a properly formatted libvmaf filter string.
        
        Args:
            log_path: Path to debug log file
            format_type: Output format ('json', 'xml', or 'text')
            n_threads: Number of threads for VMAF computation
            model_path: Optional path to VMAF model file
            
        Returns:
            Properly formatted libvmaf filter string
        """
        # Validate and escape the log path
        escaped_log_path = self.escape_file_path(log_path)
        
        # Build filter parameters
        params = []
        
        # Add log format if not text (text is default)
        if format_type.lower() in ['json', 'xml']:
            params.append(f"log_fmt={format_type.lower()}")
        
        # Add log path
        params.append(f"log_path={escaped_log_path}")
        
        # Add number of threads
        if n_threads > 0:
            params.append(f"n_threads={n_threads}")
        
        # Add model path if specified
        if model_path:
            escaped_model_path = self.escape_file_path(model_path)
            params.append(f"model_path={escaped_model_path}")
        
        # Join parameters with colons
        filter_string = f"libvmaf={':'.join(params)}"
        
        # Validate the filter string with detailed error reporting
        validation_result = self.validate_filter_syntax(filter_string)
        
        if not validation_result['is_valid']:
            logger.error(f"Generated VMAF filter has syntax errors: {validation_result['errors']}")
            
            # Try to use corrected version if available
            if validation_result.get('corrected_filter'):
                logger.info(f"Using auto-corrected VMAF filter: {validation_result['corrected_filter']}")
                return validation_result['corrected_filter']
            else:
                logger.warning(f"No auto-correction available, using original filter: {filter_string}")
        elif validation_result['warnings']:
            logger.warning(f"VMAF filter validation warnings: {validation_result['warnings']}")
            if validation_result['suggestions']:
                logger.info(f"VMAF filter suggestions: {validation_result['suggestions']}")
        
        return filter_string
    
    def escape_file_path(self, path: str) -> str:
        """Escape file path for use in FFmpeg filter parameters.
        
        Args:
            path: Raw file path
            
        Returns:
            Properly escaped file path for FFmpeg
        """
        if not path:
            return ""
        
        # Convert to absolute path to avoid relative path issues
        abs_path = os.path.abspath(path)
        
        # Handle Windows vs Unix path separators
        if os.name == 'nt':  # Windows
            # On Windows, use forward slashes and escape backslashes
            normalized_path = abs_path.replace('\\', '/')
        else:  # Unix-like systems
            normalized_path = abs_path
        
        # Escape special characters that could interfere with FFmpeg filter syntax
        # Characters that need escaping in FFmpeg filter context: : = , [ ] ' "
        special_chars = {
            ':': '\\:',
            '=': '\\=',
            ',': '\\,',
            '[': '\\[',
            ']': '\\]',
            "'": "\\'",
            '"': '\\"'
        }
        
        escaped_path = normalized_path
        for char, escaped_char in special_chars.items():
            escaped_path = escaped_path.replace(char, escaped_char)
        
        return escaped_path
    
    def validate_filter_syntax(self, filter_string: str) -> Dict[str, Any]:
        """Validate libvmaf filter string syntax with detailed error reporting.
        
        Args:
            filter_string: Filter string to validate
            
        Returns:
            Dictionary with validation results:
                - is_valid: bool
                - errors: List[str] (validation errors found)
                - warnings: List[str] (potential issues)
                - suggestions: List[str] (improvement suggestions)
                - corrected_filter: Optional[str] (auto-corrected version if possible)
        """
        validation_result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'corrected_filter': None
        }
        
        try:
            # Basic syntax checks
            if not filter_string:
                validation_result['is_valid'] = False
                validation_result['errors'].append("Filter string is empty")
                return validation_result
            
            if not filter_string.startswith('libvmaf='):
                validation_result['is_valid'] = False
                validation_result['errors'].append("Filter string must start with 'libvmaf='")
                validation_result['suggestions'].append("Ensure filter starts with 'libvmaf=' prefix")
                return validation_result
            
            # Extract parameters part
            params_part = filter_string[8:]  # Remove 'libvmaf='
            
            if not params_part:
                validation_result['is_valid'] = False
                validation_result['errors'].append("No parameters found after 'libvmaf='")
                validation_result['suggestions'].append("Add at least one parameter like 'log_path=/path/to/log'")
                return validation_result
            
            # Split parameters by unescaped colons
            params = self._split_filter_params(params_part)
            corrected_params = []
            
            for i, param in enumerate(params):
                param_validation = self._validate_single_parameter(param, i)
                
                if not param_validation['is_valid']:
                    validation_result['is_valid'] = False
                    validation_result['errors'].extend(param_validation['errors'])
                
                validation_result['warnings'].extend(param_validation['warnings'])
                validation_result['suggestions'].extend(param_validation['suggestions'])
                
                # Use corrected parameter if available
                corrected_param = param_validation.get('corrected_parameter', param)
                corrected_params.append(corrected_param)
            
            # Generate corrected filter if we have corrections
            if any(params[i] != corrected_params[i] for i in range(len(params))):
                validation_result['corrected_filter'] = f"libvmaf={':'.join(corrected_params)}"
                validation_result['suggestions'].append("Auto-corrected version available")
            
            # Additional semantic validation
            semantic_validation = self._validate_filter_semantics(params)
            validation_result['warnings'].extend(semantic_validation['warnings'])
            validation_result['suggestions'].extend(semantic_validation['suggestions'])
            
            # Log detailed validation results
            self._log_filter_validation_results(filter_string, validation_result)
            
            return validation_result
            
        except Exception as e:
            validation_result['is_valid'] = False
            validation_result['errors'].append(f"Filter syntax validation exception: {str(e)}")
            logger.error(f"Filter syntax validation error: {e}")
            return validation_result
    
    def _validate_single_parameter(self, param: str, param_index: int) -> Dict[str, Any]:
        """Validate a single filter parameter with detailed error reporting.
        
        Args:
            param: Single parameter string (e.g., "log_path=/path/to/file")
            param_index: Index of parameter in the parameter list
            
        Returns:
            Dictionary with parameter validation results
        """
        result = {
            'is_valid': True,
            'errors': [],
            'warnings': [],
            'suggestions': [],
            'corrected_parameter': param
        }
        
        if '=' not in param:
            result['is_valid'] = False
            result['errors'].append(f"Parameter {param_index + 1} missing '=' separator: '{param}'")
            result['suggestions'].append(f"Use format 'key=value' for parameter {param_index + 1}")
            return result
        
        key, value = param.split('=', 1)
        
        if not key.strip():
            result['is_valid'] = False
            result['errors'].append(f"Parameter {param_index + 1} has empty key")
            result['suggestions'].append("Provide a valid parameter name")
            return result
        
        if not value.strip():
            result['is_valid'] = False
            result['errors'].append(f"Parameter {param_index + 1} has empty value for key '{key}'")
            result['suggestions'].append(f"Provide a value for parameter '{key}'")
            return result
        
        # Validate specific parameter types
        key_lower = key.lower().strip()
        value_stripped = value.strip()
        
        if key_lower == 'log_path':
            path_validation = self._validate_log_path_parameter(value_stripped)
            result['warnings'].extend(path_validation['warnings'])
            result['suggestions'].extend(path_validation['suggestions'])
            if path_validation.get('corrected_path'):
                result['corrected_parameter'] = f"{key}={path_validation['corrected_path']}"
        
        elif key_lower == 'log_fmt':
            if value_stripped.lower() not in ['json', 'xml', 'csv']:
                result['warnings'].append(f"Unusual log format '{value_stripped}', expected 'json', 'xml', or 'csv'")
                result['suggestions'].append("Use 'json' format for most reliable parsing")
        
        elif key_lower == 'n_threads':
            try:
                thread_count = int(value_stripped)
                if thread_count <= 0:
                    result['warnings'].append(f"Thread count {thread_count} should be positive")
                    result['suggestions'].append("Use a positive number of threads (e.g., 4)")
                elif thread_count > 16:
                    result['warnings'].append(f"Thread count {thread_count} is very high, may cause resource issues")
                    result['suggestions'].append("Consider using 4-8 threads for optimal performance")
            except ValueError:
                result['is_valid'] = False
                result['errors'].append(f"Invalid thread count '{value_stripped}', must be a number")
                result['suggestions'].append("Use a numeric value for n_threads (e.g., 4)")
        
        elif key_lower == 'model_path':
            model_validation = self._validate_model_path_parameter(value_stripped)
            result['warnings'].extend(model_validation['warnings'])
            result['suggestions'].extend(model_validation['suggestions'])
        
        return result
    
    def _validate_log_path_parameter(self, log_path: str) -> Dict[str, Any]:
        """Validate log_path parameter with path-specific checks.
        
        Args:
            log_path: Log file path to validate
            
        Returns:
            Dictionary with path validation results
        """
        result = {
            'warnings': [],
            'suggestions': [],
            'corrected_path': None
        }
        
        # Check for common path issues
        if '\\' in log_path and os.name != 'nt':
            result['warnings'].append("Backslashes in path on non-Windows system")
            result['suggestions'].append("Use forward slashes for cross-platform compatibility")
            result['corrected_path'] = log_path.replace('\\', '/')
        
        # Check for unescaped special characters
        special_chars = [':', '=', ',', '[', ']']
        unescaped_chars = []
        for char in special_chars:
            if char in log_path and f'\\{char}' not in log_path:
                unescaped_chars.append(char)
        
        if unescaped_chars:
            result['warnings'].append(f"Unescaped special characters in path: {unescaped_chars}")
            result['suggestions'].append("Escape special characters with backslashes")
            
            # Auto-correct by escaping
            corrected = log_path
            for char in unescaped_chars:
                corrected = corrected.replace(char, f'\\{char}')
            result['corrected_path'] = corrected
        
        # Check if path looks like it's in logs directory
        if 'logs' not in log_path.lower():
            result['suggestions'].append("Consider placing log files in logs/ directory")
        
        return result
    
    def _validate_model_path_parameter(self, model_path: str) -> Dict[str, Any]:
        """Validate model_path parameter.
        
        Args:
            model_path: VMAF model file path to validate
            
        Returns:
            Dictionary with model path validation results
        """
        result = {
            'warnings': [],
            'suggestions': []
        }
        
        # Check for common model file extensions
        if not model_path.lower().endswith(('.pkl', '.json', '.model')):
            result['warnings'].append("Model path doesn't have expected extension (.pkl, .json, .model)")
            result['suggestions'].append("Verify model file format is correct")
        
        # Check for absolute vs relative path
        if not os.path.isabs(model_path):
            result['suggestions'].append("Consider using absolute path for model file")
        
        return result
    
    def _validate_filter_semantics(self, params: List[str]) -> Dict[str, Any]:
        """Validate semantic correctness of filter parameters.
        
        Args:
            params: List of parameter strings
            
        Returns:
            Dictionary with semantic validation results
        """
        result = {
            'warnings': [],
            'suggestions': []
        }
        
        param_keys = []
        for param in params:
            if '=' in param:
                key = param.split('=', 1)[0].strip().lower()
                param_keys.append(key)
        
        # Check for required parameters
        if 'log_path' not in param_keys:
            result['warnings'].append("No log_path parameter specified")
            result['suggestions'].append("Add log_path parameter for debugging and verification")
        
        # Check for duplicate parameters
        seen_keys = set()
        for key in param_keys:
            if key in seen_keys:
                result['warnings'].append(f"Duplicate parameter '{key}' found")
                result['suggestions'].append(f"Remove duplicate '{key}' parameters")
            seen_keys.add(key)
        
        # Check parameter combinations
        if 'log_fmt' in param_keys and 'log_path' not in param_keys:
            result['warnings'].append("log_fmt specified without log_path")
            result['suggestions'].append("Add log_path parameter when using log_fmt")
        
        return result
    
    def _log_filter_validation_results(self, filter_string: str, validation_result: Dict[str, Any]) -> None:
        """Log detailed filter validation results for debugging.
        
        Args:
            filter_string: Original filter string
            validation_result: Validation results dictionary
        """
        if not logger.isEnabledFor(logging.DEBUG):
            return
        
        logger.debug("=== VMAF FILTER VALIDATION RESULTS ===")
        logger.debug(f"Filter: {filter_string}")
        logger.debug(f"Valid: {validation_result['is_valid']}")
        
        if validation_result['errors']:
            logger.debug("ERRORS:")
            for error in validation_result['errors']:
                logger.debug(f"  - {error}")
        
        if validation_result['warnings']:
            logger.debug("WARNINGS:")
            for warning in validation_result['warnings']:
                logger.debug(f"  - {warning}")
        
        if validation_result['suggestions']:
            logger.debug("SUGGESTIONS:")
            for suggestion in validation_result['suggestions']:
                logger.debug(f"  - {suggestion}")
        
        if validation_result['corrected_filter']:
            logger.debug(f"CORRECTED: {validation_result['corrected_filter']}")
        
        logger.debug("=== END FILTER VALIDATION ===")
    
    def validate_and_correct_filter(self, filter_string: str) -> Tuple[bool, str, List[str]]:
        """Validate filter and return corrected version if possible.
        
        Args:
            filter_string: Filter string to validate and correct
            
        Returns:
            Tuple of (is_valid, corrected_filter_or_original, error_messages)
        """
        validation_result = self.validate_filter_syntax(filter_string)
        
        is_valid = validation_result['is_valid']
        corrected_filter = validation_result.get('corrected_filter', filter_string)
        error_messages = validation_result['errors']
        
        return is_valid, corrected_filter, error_messages
    
    def _split_filter_params(self, params_string: str) -> List[str]:
        """Split filter parameters by unescaped colons.
        
        Args:
            params_string: Parameter string to split
            
        Returns:
            List of individual parameters
        """
        params = []
        current_param = ""
        i = 0
        
        while i < len(params_string):
            char = params_string[i]
            
            if char == '\\' and i + 1 < len(params_string):
                # Escaped character - add both the backslash and next character
                current_param += char + params_string[i + 1]
                i += 2
            elif char == ':':
                # Unescaped colon - parameter separator
                if current_param:
                    params.append(current_param)
                    current_param = ""
                i += 1
            else:
                current_param += char
                i += 1
        
        # Add the last parameter
        if current_param:
            params.append(current_param)
        
        return params
    
    def get_safe_log_filename(self, base_name: str, session_id: str, format_type: str = "json") -> str:
        """Generate a safe log filename and return the full path.
        
        Args:
            base_name: Base name for the log file (e.g., 'vmaf')
            session_id: Unique session identifier
            format_type: Log format type ('json', 'xml', 'log')
            
        Returns:
            Full path to the safe log file
        """
        # Map format types to file extensions
        format_extensions = {
            'json': 'json',
            'xml': 'xml',
            'text': 'log',
            'log': 'log'
        }
        
        extension = format_extensions.get(format_type.lower(), 'log')
        
        return self.debug_log_manager.get_debug_log_path(base_name, session_id, extension)
    
    def build_complete_filter_complex(self, ref_path: str, dist_path: str, 
                                    log_path: str, format_type: str = "json",
                                    eval_height: Optional[int] = None,
                                    n_threads: int = 4) -> str:
        """Build complete filter_complex string for VMAF computation.
        
        Args:
            ref_path: Reference video path
            dist_path: Distorted video path  
            log_path: Debug log file path
            format_type: VMAF output format
            eval_height: Optional evaluation height for scaling
            n_threads: Number of threads for computation
            
        Returns:
            Complete filter_complex string ready for FFmpeg
        """
        # Build scaling filter if eval_height is specified
        if eval_height:
            scale_filter = f"scale=-2:{eval_height}:flags=bicubic,"
        else:
            scale_filter = ""
        
        # Build VMAF filter
        vmaf_filter = self.build_vmaf_filter(log_path, format_type, n_threads)
        
        # Construct complete filter complex
        filter_complex = (
            f"[0:v]{scale_filter}setpts=PTS-STARTPTS[dist];"
            f"[1:v]{scale_filter}setpts=PTS-STARTPTS[ref];"
            f"[dist][ref]{vmaf_filter}"
        )
        
        return filter_complex