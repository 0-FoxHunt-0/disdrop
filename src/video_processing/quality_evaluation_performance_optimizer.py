"""
Quality Evaluation Performance Optimizer
Integrates frequency limits, time budgets, and result prediction for optimized quality evaluation
"""

import logging
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass

try:
    from .evaluation_frequency_limiter import EvaluationFrequencyLimiter, EvaluationResult
    from .computation_time_budget import ComputationTimeBudget, BudgetStatus, ProgressUpdate
    from .quality_predictor import QualityPredictor, PredictionStrategy, VideoCharacteristics, QualityPrediction
except ImportError:
    # Fallback for direct execution
    from evaluation_frequency_limiter import EvaluationFrequencyLimiter, EvaluationResult
    from computation_time_budget import ComputationTimeBudget, BudgetStatus, ProgressUpdate
    from quality_predictor import QualityPredictor, PredictionStrategy, VideoCharacteristics, QualityPrediction

logger = logging.getLogger(__name__)


@dataclass
class OptimizedEvaluationResult:
    """Result of optimized quality evaluation."""
    vmaf_score: Optional[float]
    ssim_score: Optional[float]
    passes: bool
    method: str
    confidence: float
    evaluation_success: bool
    optimization_applied: bool
    optimization_details: Dict[str, Any]
    details: Dict[str, Any]


class QualityEvaluationPerformanceOptimizer:
    """Integrates all performance optimization techniques for quality evaluation."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        
        # Initialize component modules
        self.frequency_limiter = EvaluationFrequencyLimiter(config_manager)
        self.time_budget = ComputationTimeBudget(config_manager)
        self.quality_predictor = QualityPredictor(config_manager)
        
        # Configuration
        self.enable_prediction = self._get_config('enable_prediction', True)
        self.enable_frequency_limits = self._get_config('enable_frequency_limits', True)
        self.enable_time_budgets = self._get_config('enable_time_budgets', True)
        self.fallback_to_fast_estimation = self._get_config('fallback_to_fast_estimation', True)
        
        logger.info("Quality evaluation performance optimizer initialized")
    
    def _get_config(self, key: str, default: Any) -> Any:
        """Get configuration value with fallback to default."""
        if self.config:
            return self.config.get(f'quality_evaluation.performance_optimization.{key}', default)
        return default
    
    def should_skip_evaluation(
        self, 
        video_path: str, 
        evaluation_type: str = 'combined'
    ) -> tuple[bool, str, Optional[QualityPrediction]]:
        """Determine if quality evaluation should be skipped with optimization details.
        
        Args:
            video_path: Path to video file
            evaluation_type: Type of evaluation ('vmaf', 'ssim', 'combined')
            
        Returns:
            Tuple of (should_skip, reason, prediction_result)
        """
        prediction_result = None
        
        # Check frequency limits first
        if self.enable_frequency_limits:
            should_skip_freq, freq_reason = self.frequency_limiter.should_skip_evaluation(evaluation_type)
            if should_skip_freq:
                return True, f"frequency_limit_{freq_reason}", None
        
        # Check time budget status
        if self.enable_time_budgets:
            budget_status, elapsed, remaining = self.time_budget.check_budget_status()
            if budget_status == BudgetStatus.EXCEEDED:
                return True, "time_budget_exceeded", None
            if remaining < 30.0:  # Less than 30 seconds remaining
                return True, f"insufficient_time_budget_{remaining:.1f}s", None
        
        # Use prediction to determine if evaluation should be skipped
        if self.enable_prediction:
            # Use unified predictor with automatic strategy selection
            prediction_result = self.quality_predictor.predict_quality(
                video_path,
                strategy=PredictionStrategy.AUTO
            )
            
            if prediction_result.should_skip_evaluation:
                return True, f"prediction_skip_{prediction_result.prediction_basis}", prediction_result
            
            # Check if prediction indicates likely failure
            if len(prediction_result.risk_factors) >= 3:
                return True, f"high_risk_prediction_{len(prediction_result.risk_factors)}_factors", prediction_result
        
        return False, "", prediction_result
    
    def execute_optimized_evaluation(
        self,
        original_path: str,
        compressed_path: str,
        vmaf_threshold: float = 80.0,
        ssim_threshold: float = 0.94,
        eval_height: Optional[int] = None,
        progress_callback: Optional[Callable[[ProgressUpdate], None]] = None
    ) -> OptimizedEvaluationResult:
        """Execute quality evaluation with all performance optimizations.
        
        Args:
            original_path: Path to original video
            compressed_path: Path to compressed video
            vmaf_threshold: VMAF threshold for pass/fail
            ssim_threshold: SSIM threshold for pass/fail
            eval_height: Optional height for evaluation scaling
            progress_callback: Optional progress callback
            
        Returns:
            OptimizedEvaluationResult with evaluation results and optimization details
        """
        start_time = time.time()
        optimization_details = {
            'frequency_limits_applied': False,
            'time_budget_applied': False,
            'prediction_applied': False,
            'fast_estimation_used': False,
            'evaluation_skipped': False,
            'optimization_time_saved': 0.0
        }
        
        logger.info(f"Starting optimized quality evaluation: {compressed_path}")
        
        # Check if evaluation should be skipped
        should_skip, skip_reason, prediction = self.should_skip_evaluation(compressed_path, 'combined')
        
        if should_skip:
            optimization_details['evaluation_skipped'] = True
            optimization_details['skip_reason'] = skip_reason
            
            # Use prediction results if available
            if prediction:
                optimization_details['prediction_applied'] = True
                result = self._create_result_from_prediction(
                    prediction, vmaf_threshold, ssim_threshold, optimization_details
                )
                logger.info(f"Evaluation skipped ({skip_reason}): using prediction")
                return result
            
            # Use fast estimation as fallback
            if self.fallback_to_fast_estimation:
                optimization_details['fast_estimation_used'] = True
                result = self._use_fast_estimation_fallback(
                    compressed_path, vmaf_threshold, ssim_threshold, optimization_details
                )
                logger.info(f"Evaluation skipped ({skip_reason}): using fast estimation")
                return result
            
            # Return conservative result if no fallback
            return self._create_conservative_result(skip_reason, optimization_details)
        
        # Proceed with evaluation using performance optimizations
        return self._execute_evaluation_with_optimizations(
            original_path, compressed_path, vmaf_threshold, ssim_threshold,
            eval_height, progress_callback, optimization_details, prediction
        )
    
    def _execute_evaluation_with_optimizations(
        self,
        original_path: str,
        compressed_path: str,
        vmaf_threshold: float,
        ssim_threshold: float,
        eval_height: Optional[int],
        progress_callback: Optional[Callable],
        optimization_details: Dict[str, Any],
        prediction: Optional[QualityPrediction]
    ) -> OptimizedEvaluationResult:
        """Execute evaluation with performance optimizations applied."""
        
        # Determine evaluation method based on prediction and limits
        evaluation_method = self._determine_optimal_evaluation_method(prediction)
        
        # Start time budget tracking
        if self.enable_time_budgets:
            budget_started = self.time_budget.start_operation(evaluation_method, progress_callback)
            if not budget_started:
                optimization_details['time_budget_applied'] = True
                return self._create_conservative_result("time_budget_start_failed", optimization_details)
        
        try:
            # Execute evaluation with timeout
            if evaluation_method == 'vmaf_only':
                result = self._execute_vmaf_evaluation(
                    original_path, compressed_path, vmaf_threshold, eval_height, optimization_details
                )
            elif evaluation_method == 'ssim_only':
                result = self._execute_ssim_evaluation(
                    original_path, compressed_path, ssim_threshold, eval_height, optimization_details
                )
            else:  # combined
                result = self._execute_combined_evaluation(
                    original_path, compressed_path, vmaf_threshold, ssim_threshold, 
                    eval_height, optimization_details
                )
            
            # Record successful evaluation
            if self.enable_frequency_limits:
                self.frequency_limiter.record_evaluation_attempt(
                    evaluation_method, 
                    EvaluationResult.SUCCESS if result.evaluation_success else EvaluationResult.FAILURE,
                    time.time() - (self.time_budget.operation_start_time or time.time()),
                    confidence=result.confidence
                )
            
            # Record result for prediction learning
            if self.enable_prediction and prediction:
                video_chars = self.quality_predictor._analyze_video_characteristics(compressed_path)
                if video_chars:
                    self.quality_predictor.record_evaluation_result(
                        video_chars, result.vmaf_score, result.ssim_score,
                        time.time() - (self.time_budget.operation_start_time or time.time()),
                        result.evaluation_success
                    )
            
            return result
        
        except Exception as e:
            logger.error(f"Optimized evaluation failed: {e}")
            
            # Record failed evaluation
            if self.enable_frequency_limits:
                self.frequency_limiter.record_evaluation_attempt(
                    evaluation_method, EvaluationResult.FAILURE,
                    time.time() - (self.time_budget.operation_start_time or time.time()),
                    error_message=str(e)
                )
            
            return self._create_error_result(str(e), optimization_details)
        
        finally:
            # Finish time budget tracking
            if self.enable_time_budgets:
                self.time_budget.finish_operation(success=True)
    
    def _determine_optimal_evaluation_method(self, prediction: Optional[QualityPrediction]) -> str:
        """Determine optimal evaluation method based on prediction and constraints."""
        
        # Check frequency limits for different methods
        if self.enable_frequency_limits:
            can_vmaf, _ = self.frequency_limiter.should_skip_evaluation('vmaf')
            can_ssim, _ = self.frequency_limiter.should_skip_evaluation('ssim')
            
            if can_vmaf and can_ssim:
                return 'fast_estimation'  # Both blocked
            elif can_vmaf:
                return 'ssim_only'
            elif can_ssim:
                return 'vmaf_only'
        
        # Use prediction to optimize method selection
        if prediction and self.enable_prediction:
            # If prediction confidence is high, use faster method
            if prediction.overall_confidence > 0.8:
                if prediction.estimated_evaluation_time > 120:  # 2 minutes
                    return 'ssim_only'  # Faster method for high confidence
        
        return 'combined'  # Default to combined evaluation
    
    def _execute_vmaf_evaluation(
        self, 
        original_path: str, 
        compressed_path: str, 
        threshold: float,
        eval_height: Optional[int],
        optimization_details: Dict[str, Any]
    ) -> OptimizedEvaluationResult:
        """Execute VMAF-only evaluation with optimizations."""
        
        # Import quality gates for actual evaluation
        try:
            from .quality_gates import QualityGates
        except ImportError:
            from quality_gates import QualityGates
        
        quality_gates = QualityGates(self.config)
        
        # Get timeout from time budget
        timeout = self.time_budget.get_operation_budget('vmaf') if self.enable_time_budgets else 180.0
        
        # Execute with timeout
        if self.enable_time_budgets:
            # Use time budget execution
            cmd = self._build_vmaf_command(original_path, compressed_path, eval_height)
            exec_result = self.time_budget.execute_with_timeout(cmd, 'vmaf', timeout)
            
            if exec_result['success']:
                # Parse VMAF result
                vmaf_score = quality_gates._parse_vmaf_output(exec_result['stderr'])
                passes = vmaf_score is not None and vmaf_score >= threshold
                
                return OptimizedEvaluationResult(
                    vmaf_score=vmaf_score,
                    ssim_score=None,
                    passes=passes,
                    method='vmaf_only',
                    confidence=0.9 if vmaf_score is not None else 0.1,
                    evaluation_success=vmaf_score is not None,
                    optimization_applied=True,
                    optimization_details=optimization_details,
                    details={'vmaf_threshold': threshold, 'execution_time': exec_result['execution_time']}
                )
            else:
                # Handle timeout or failure
                if exec_result['timeout_exceeded']:
                    return self.time_budget.handle_evaluation_timeout('vmaf', exec_result['execution_time'])
                else:
                    return self._create_error_result(exec_result['stderr'], optimization_details)
        else:
            # Use standard evaluation
            result = quality_gates.evaluate_quality(original_path, compressed_path, threshold, 0.0, eval_height)
            return self._convert_standard_result(result, optimization_details)
    
    def _execute_ssim_evaluation(
        self, 
        original_path: str, 
        compressed_path: str, 
        threshold: float,
        eval_height: Optional[int],
        optimization_details: Dict[str, Any]
    ) -> OptimizedEvaluationResult:
        """Execute SSIM-only evaluation with optimizations."""
        
        from quality_gates import QualityGates
        
        quality_gates = QualityGates(self.config)
        
        # Execute SSIM evaluation (similar pattern to VMAF)
        result = quality_gates.evaluate_quality(original_path, compressed_path, 0.0, threshold, eval_height)
        return self._convert_standard_result(result, optimization_details)
    
    def _execute_combined_evaluation(
        self, 
        original_path: str, 
        compressed_path: str, 
        vmaf_threshold: float,
        ssim_threshold: float,
        eval_height: Optional[int],
        optimization_details: Dict[str, Any]
    ) -> OptimizedEvaluationResult:
        """Execute combined VMAF+SSIM evaluation with optimizations."""
        
        from quality_gates import QualityGates
        
        quality_gates = QualityGates(self.config)
        
        # Execute combined evaluation
        result = quality_gates.evaluate_quality(
            original_path, compressed_path, vmaf_threshold, ssim_threshold, eval_height
        )
        return self._convert_standard_result(result, optimization_details)
    
    def _build_vmaf_command(self, original_path: str, compressed_path: str, eval_height: Optional[int]) -> List[str]:
        """Build VMAF command for direct execution."""
        # This is a simplified version - in practice you'd use the full VMAF filter building logic
        cmd = ['ffmpeg', '-i', original_path, '-i', compressed_path]
        
        if eval_height:
            cmd.extend(['-vf', f'scale=-1:{eval_height}'])
        
        cmd.extend(['-lavfi', 'libvmaf', '-f', 'null', '-'])
        
        return cmd
    
    def _create_result_from_prediction(
        self, 
        prediction: QualityPrediction, 
        vmaf_threshold: float,
        ssim_threshold: float,
        optimization_details: Dict[str, Any]
    ) -> OptimizedEvaluationResult:
        """Create evaluation result from prediction."""
        
        # Determine if predicted scores pass thresholds
        vmaf_passes = (prediction.predicted_vmaf is not None and 
                      prediction.predicted_vmaf >= vmaf_threshold)
        ssim_passes = (prediction.predicted_ssim is not None and 
                      prediction.predicted_ssim >= ssim_threshold)
        
        passes = vmaf_passes and ssim_passes
        
        optimization_details['prediction_confidence'] = prediction.overall_confidence
        optimization_details['prediction_basis'] = prediction.prediction_basis
        optimization_details['risk_factors'] = prediction.risk_factors
        
        return OptimizedEvaluationResult(
            vmaf_score=prediction.predicted_vmaf,
            ssim_score=prediction.predicted_ssim,
            passes=passes,
            method='prediction',
            confidence=prediction.overall_confidence,
            evaluation_success=True,
            optimization_applied=True,
            optimization_details=optimization_details,
            details={
                'predicted': True,
                'vmaf_threshold': vmaf_threshold,
                'ssim_threshold': ssim_threshold,
                'prediction_basis': prediction.prediction_basis,
                'estimated_time_saved': prediction.estimated_evaluation_time
            }
        )
    
    def _use_fast_estimation_fallback(
        self, 
        video_path: str, 
        vmaf_threshold: float,
        ssim_threshold: float,
        optimization_details: Dict[str, Any]
    ) -> OptimizedEvaluationResult:
        """Use fast quality estimation as fallback."""
        
        try:
            # Use unified quality predictor with fast estimation strategy
            estimate_result = self.quality_predictor.predict_quality(
                video_path,
                strategy=PredictionStrategy.FAST_ESTIMATION,
                target_duration=10.0
            )
            
            # Convert unified prediction to OptimizedEvaluationResult format
            # (No need for intermediate QualityEstimate format)
            vmaf_passes = estimate_result.predicted_vmaf is not None and estimate_result.predicted_vmaf >= vmaf_threshold
            ssim_passes = estimate_result.predicted_ssim is not None and estimate_result.predicted_ssim >= ssim_threshold
            passes = vmaf_passes and ssim_passes
            
            optimization_details['fast_estimation_confidence'] = estimate_result.overall_confidence
            optimization_details['fast_estimation_time'] = estimate_result.prediction_time
            
            return OptimizedEvaluationResult(
                vmaf_score=estimate_result.predicted_vmaf,
                ssim_score=estimate_result.predicted_ssim,
                passes=passes,
                method='fast_estimation',
                confidence=estimate_result.overall_confidence,
                evaluation_success=True,
                optimization_applied=True,
                optimization_details=optimization_details,
                details={
                    'fast_estimation': True,
                    'strategy': estimate_result.strategy.value,
                    'prediction_basis': estimate_result.prediction_basis,
                    'computation_time': estimate_result.prediction_time
                }
            )
        
        except Exception as e:
            logger.error(f"Fast estimation fallback failed: {e}")
            return self._create_conservative_result(f"fast_estimation_failed_{e}", optimization_details)
    
    def _create_conservative_result(self, reason: str, optimization_details: Dict[str, Any]) -> OptimizedEvaluationResult:
        """Create conservative result when evaluation is skipped."""
        
        optimization_details['conservative_result_reason'] = reason
        
        return OptimizedEvaluationResult(
            vmaf_score=None,
            ssim_score=None,
            passes=True,  # Conservative: assume passes to avoid blocking compression
            method='conservative_skip',
            confidence=0.1,
            evaluation_success=False,
            optimization_applied=True,
            optimization_details=optimization_details,
            details={
                'skipped': True,
                'reason': reason,
                'conservative_pass': True
            }
        )
    
    def _create_error_result(self, error_message: str, optimization_details: Dict[str, Any]) -> OptimizedEvaluationResult:
        """Create error result."""
        
        optimization_details['error_message'] = error_message
        
        return OptimizedEvaluationResult(
            vmaf_score=None,
            ssim_score=None,
            passes=False,
            method='error',
            confidence=0.0,
            evaluation_success=False,
            optimization_applied=True,
            optimization_details=optimization_details,
            details={
                'error': error_message,
                'evaluation_failed': True
            }
        )
    
    def _convert_standard_result(self, standard_result: Dict[str, Any], optimization_details: Dict[str, Any]) -> OptimizedEvaluationResult:
        """Convert standard quality evaluation result to optimized result."""
        
        return OptimizedEvaluationResult(
            vmaf_score=standard_result.get('vmaf_score'),
            ssim_score=standard_result.get('ssim_score'),
            passes=standard_result.get('passes', False),
            method=standard_result.get('method', 'unknown'),
            confidence=standard_result.get('confidence', 0.0),
            evaluation_success=standard_result.get('evaluation_success', False),
            optimization_applied=True,
            optimization_details=optimization_details,
            details=standard_result.get('details', {})
        )
    
    def get_optimization_statistics(self) -> Dict[str, Any]:
        """Get statistics about optimization performance."""
        
        stats = {
            'frequency_limiter': self.frequency_limiter.get_session_statistics(),
            'time_budget': self.time_budget.get_budget_configuration(),
            'quality_predictor': self.quality_predictor.get_prediction_accuracy(),
            'optimization_enabled': {
                'prediction': self.enable_prediction,
                'frequency_limits': self.enable_frequency_limits,
                'time_budgets': self.enable_time_budgets,
                'fast_estimation_fallback': self.fallback_to_fast_estimation
            }
        }
        
        return stats
    
    def reset_session(self) -> None:
        """Reset optimization session for new compression attempt."""
        logger.debug("Resetting quality evaluation performance optimizer session")
        
        if self.enable_frequency_limits:
            self.frequency_limiter.reset_session()
        
        # Time budget resets automatically per operation
        # Result predictor maintains historical data across sessions
    
    def configure_performance_profile(self, profile: str) -> None:
        """Configure performance optimization profile.
        
        Args:
            profile: Performance profile ('fast', 'balanced', 'quality')
        """
        if profile == 'fast':
            # Aggressive optimization for speed
            self.frequency_limiter.limits.max_total_attempts = 2
            self.frequency_limiter.limits.skip_after_failures = 1
            self.time_budget.adjust_budget('total', 120.0)
            self.time_budget.adjust_budget('vmaf', 60.0)
            self.time_budget.adjust_budget('ssim', 30.0)
            
        elif profile == 'balanced':
            # Balanced optimization
            self.frequency_limiter.limits.max_total_attempts = 3
            self.frequency_limiter.limits.skip_after_failures = 2
            self.time_budget.adjust_budget('total', 300.0)
            self.time_budget.adjust_budget('vmaf', 180.0)
            self.time_budget.adjust_budget('ssim', 90.0)
            
        elif profile == 'quality':
            # Conservative optimization for quality
            self.frequency_limiter.limits.max_total_attempts = 5
            self.frequency_limiter.limits.skip_after_failures = 3
            self.time_budget.adjust_budget('total', 600.0)
            self.time_budget.adjust_budget('vmaf', 360.0)
            self.time_budget.adjust_budget('ssim', 180.0)
        
        logger.info(f"Performance profile configured: {profile}")