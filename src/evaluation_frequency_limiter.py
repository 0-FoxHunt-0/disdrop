"""
Evaluation Frequency Limiter
Implements limits on quality evaluation attempts per compression session
"""

import logging
import time
from typing import Dict, Any, Optional, List, NamedTuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class EvaluationResult(Enum):
    """Result of evaluation attempt."""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class EvaluationAttempt:
    """Information about a quality evaluation attempt."""
    timestamp: float
    evaluation_type: str  # 'vmaf', 'ssim', 'combined'
    result: EvaluationResult
    duration: float
    error_message: Optional[str] = None
    confidence: float = 0.0


@dataclass
class SessionLimits:
    """Evaluation limits for a compression session."""
    max_total_attempts: int
    max_consecutive_failures: int
    max_evaluation_time: float
    skip_after_failures: int
    timeout_seconds: float


class EvaluationFrequencyLimiter:
    """Manages evaluation frequency limits and skip logic for quality assessments."""
    
    def __init__(self, config_manager=None):
        self.config = config_manager
        
        # Session tracking
        self.session_attempts: List[EvaluationAttempt] = []
        self.session_start_time = time.time()
        self.consecutive_failures = 0
        self.total_evaluation_time = 0.0
        
        # Load configuration
        self.limits = self._load_session_limits()
        self.skip_expensive_evaluations = False
        
        logger.info(f"Evaluation frequency limiter initialized: "
                   f"max_attempts={self.limits.max_total_attempts}, "
                   f"max_consecutive_failures={self.limits.max_consecutive_failures}, "
                   f"timeout={self.limits.timeout_seconds}s")
    
    def _load_session_limits(self) -> SessionLimits:
        """Load evaluation limits from configuration."""
        if self.config:
            base_path = 'quality_evaluation.frequency_limits'
            return SessionLimits(
                max_total_attempts=self.config.get(f'{base_path}.max_total_attempts', 3),
                max_consecutive_failures=self.config.get(f'{base_path}.max_consecutive_failures', 2),
                max_evaluation_time=self.config.get(f'{base_path}.max_evaluation_time_seconds', 300.0),
                skip_after_failures=self.config.get(f'{base_path}.skip_after_failures', 2),
                timeout_seconds=self.config.get(f'{base_path}.timeout_seconds', 120.0)
            )
        
        # Default limits
        return SessionLimits(
            max_total_attempts=3,
            max_consecutive_failures=2,
            max_evaluation_time=300.0,
            skip_after_failures=2,
            timeout_seconds=120.0
        )
    
    def should_skip_evaluation(self, evaluation_type: str) -> tuple[bool, str]:
        """Check if evaluation should be skipped based on current limits.
        
        Args:
            evaluation_type: Type of evaluation ('vmaf', 'ssim', 'combined')
            
        Returns:
            Tuple of (should_skip, reason)
        """
        # Check if expensive evaluations are globally disabled
        if self.skip_expensive_evaluations:
            return True, "expensive_evaluations_disabled"
        
        # Check total attempts limit
        if len(self.session_attempts) >= self.limits.max_total_attempts:
            return True, f"max_attempts_reached_{self.limits.max_total_attempts}"
        
        # Check consecutive failures limit
        if self.consecutive_failures >= self.limits.skip_after_failures:
            return True, f"consecutive_failures_{self.consecutive_failures}"
        
        # Check total evaluation time limit
        if self.total_evaluation_time >= self.limits.max_evaluation_time:
            return True, f"time_budget_exceeded_{self.total_evaluation_time:.1f}s"
        
        # Check if we've had too many recent failures
        recent_failures = self._count_recent_failures(window_seconds=60.0)
        if recent_failures >= self.limits.max_consecutive_failures:
            return True, f"recent_failures_{recent_failures}"
        
        return False, ""
    
    def _count_recent_failures(self, window_seconds: float) -> int:
        """Count failures within the specified time window."""
        current_time = time.time()
        cutoff_time = current_time - window_seconds
        
        recent_failures = 0
        for attempt in reversed(self.session_attempts):
            if attempt.timestamp < cutoff_time:
                break
            if attempt.result == EvaluationResult.FAILURE:
                recent_failures += 1
        
        return recent_failures
    
    def record_evaluation_attempt(
        self, 
        evaluation_type: str, 
        result: EvaluationResult,
        duration: float,
        error_message: Optional[str] = None,
        confidence: float = 0.0
    ) -> None:
        """Record an evaluation attempt for tracking.
        
        Args:
            evaluation_type: Type of evaluation performed
            result: Result of the evaluation
            duration: Time taken for evaluation
            error_message: Error message if evaluation failed
            confidence: Confidence score of evaluation (0-1)
        """
        attempt = EvaluationAttempt(
            timestamp=time.time(),
            evaluation_type=evaluation_type,
            result=result,
            duration=duration,
            error_message=error_message,
            confidence=confidence
        )
        
        self.session_attempts.append(attempt)
        self.total_evaluation_time += duration
        
        # Update consecutive failure counter
        if result == EvaluationResult.FAILURE:
            self.consecutive_failures += 1
        else:
            self.consecutive_failures = 0
        
        # Log attempt
        logger.info(f"Evaluation attempt recorded: {evaluation_type} {result.value} "
                   f"({duration:.2f}s, confidence={confidence:.2f})")
        
        # Check if we should disable expensive evaluations
        if self.consecutive_failures >= self.limits.skip_after_failures:
            self.skip_expensive_evaluations = True
            logger.warning(f"Disabling expensive evaluations after {self.consecutive_failures} "
                          f"consecutive failures")
    
    def get_evaluation_timeout(self, evaluation_type: str) -> float:
        """Get timeout for specific evaluation type.
        
        Args:
            evaluation_type: Type of evaluation
            
        Returns:
            Timeout in seconds
        """
        # Calculate remaining time budget
        remaining_time = self.limits.max_evaluation_time - self.total_evaluation_time
        
        # Use configured timeout or remaining time, whichever is smaller
        base_timeout = self.limits.timeout_seconds
        
        if evaluation_type == 'vmaf':
            # VMAF typically takes longer
            base_timeout = min(base_timeout * 1.5, 180.0)
        elif evaluation_type == 'ssim':
            # SSIM is usually faster
            base_timeout = min(base_timeout * 0.8, 90.0)
        
        # Don't exceed remaining time budget
        return min(base_timeout, max(30.0, remaining_time))
    
    def handle_evaluation_timeout(self, evaluation_type: str, timeout_duration: float) -> Dict[str, Any]:
        """Handle evaluation timeout and return partial results.
        
        Args:
            evaluation_type: Type of evaluation that timed out
            timeout_duration: Duration of timeout
            
        Returns:
            Dictionary with partial results and timeout information
        """
        # Record timeout attempt
        self.record_evaluation_attempt(
            evaluation_type=evaluation_type,
            result=EvaluationResult.TIMEOUT,
            duration=timeout_duration,
            error_message=f"Evaluation timed out after {timeout_duration:.1f}s"
        )
        
        # Return partial result structure
        partial_result = {
            'vmaf_score': None,
            'ssim_score': None,
            'passes': False,  # Conservative default
            'method': 'timeout',
            'confidence': 0.1,  # Low confidence due to timeout
            'evaluation_success': False,
            'details': {
                'timeout': True,
                'timeout_duration': timeout_duration,
                'evaluation_type': evaluation_type,
                'remaining_attempts': max(0, self.limits.max_total_attempts - len(self.session_attempts)),
                'consecutive_failures': self.consecutive_failures,
                'total_evaluation_time': self.total_evaluation_time,
                'error': f"Quality evaluation timed out after {timeout_duration:.1f} seconds"
            }
        }
        
        logger.warning(f"Quality evaluation timeout: {evaluation_type} after {timeout_duration:.1f}s")
        
        return partial_result
    
    def get_session_statistics(self) -> Dict[str, Any]:
        """Get statistics about the current evaluation session.
        
        Returns:
            Dictionary with session statistics
        """
        total_attempts = len(self.session_attempts)
        successful_attempts = sum(1 for a in self.session_attempts if a.result == EvaluationResult.SUCCESS)
        failed_attempts = sum(1 for a in self.session_attempts if a.result == EvaluationResult.FAILURE)
        timeout_attempts = sum(1 for a in self.session_attempts if a.result == EvaluationResult.TIMEOUT)
        
        # Calculate average duration for successful attempts
        successful_durations = [a.duration for a in self.session_attempts if a.result == EvaluationResult.SUCCESS]
        avg_successful_duration = sum(successful_durations) / len(successful_durations) if successful_durations else 0.0
        
        # Calculate average confidence for successful attempts
        successful_confidences = [a.confidence for a in self.session_attempts if a.result == EvaluationResult.SUCCESS]
        avg_confidence = sum(successful_confidences) / len(successful_confidences) if successful_confidences else 0.0
        
        session_duration = time.time() - self.session_start_time
        
        return {
            'session_duration': session_duration,
            'total_attempts': total_attempts,
            'successful_attempts': successful_attempts,
            'failed_attempts': failed_attempts,
            'timeout_attempts': timeout_attempts,
            'consecutive_failures': self.consecutive_failures,
            'total_evaluation_time': self.total_evaluation_time,
            'avg_successful_duration': avg_successful_duration,
            'avg_confidence': avg_confidence,
            'expensive_evaluations_disabled': self.skip_expensive_evaluations,
            'remaining_attempts': max(0, self.limits.max_total_attempts - total_attempts),
            'remaining_time_budget': max(0.0, self.limits.max_evaluation_time - self.total_evaluation_time),
            'limits': {
                'max_total_attempts': self.limits.max_total_attempts,
                'max_consecutive_failures': self.limits.max_consecutive_failures,
                'max_evaluation_time': self.limits.max_evaluation_time,
                'skip_after_failures': self.limits.skip_after_failures,
                'timeout_seconds': self.limits.timeout_seconds
            }
        }
    
    def reset_session(self) -> None:
        """Reset the evaluation session for a new compression attempt."""
        logger.debug("Resetting evaluation frequency limiter session")
        
        self.session_attempts.clear()
        self.session_start_time = time.time()
        self.consecutive_failures = 0
        self.total_evaluation_time = 0.0
        self.skip_expensive_evaluations = False
    
    def disable_expensive_evaluations(self, reason: str = "manual") -> None:
        """Manually disable expensive evaluations.
        
        Args:
            reason: Reason for disabling evaluations
        """
        self.skip_expensive_evaluations = True
        logger.info(f"Expensive evaluations disabled: {reason}")
    
    def enable_expensive_evaluations(self) -> None:
        """Re-enable expensive evaluations."""
        self.skip_expensive_evaluations = False
        self.consecutive_failures = 0  # Reset failure counter
        logger.info("Expensive evaluations re-enabled")
    
    def get_skip_recommendation(self, evaluation_type: str) -> Dict[str, Any]:
        """Get detailed recommendation about whether to skip evaluation.
        
        Args:
            evaluation_type: Type of evaluation to check
            
        Returns:
            Dictionary with skip recommendation and reasoning
        """
        should_skip, reason = self.should_skip_evaluation(evaluation_type)
        
        stats = self.get_session_statistics()
        
        recommendation = {
            'should_skip': should_skip,
            'reason': reason,
            'evaluation_type': evaluation_type,
            'session_stats': stats,
            'timeout_seconds': self.get_evaluation_timeout(evaluation_type),
            'alternatives': []
        }
        
        # Suggest alternatives if skipping
        if should_skip:
            if evaluation_type == 'vmaf':
                recommendation['alternatives'] = ['ssim_only', 'fast_quality_estimation', 'skip_quality_check']
            elif evaluation_type == 'ssim':
                recommendation['alternatives'] = ['fast_quality_estimation', 'skip_quality_check']
            elif evaluation_type == 'combined':
                recommendation['alternatives'] = ['ssim_only', 'fast_quality_estimation', 'skip_quality_check']
        
        return recommendation