"""
Test script for quality evaluation performance optimization
"""

import logging
import time
import os
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_frequency_limiter():
    """Test evaluation frequency limiter."""
    logger.info("Testing Evaluation Frequency Limiter...")
    
    try:
        from evaluation_frequency_limiter import EvaluationFrequencyLimiter, EvaluationResult
        
        limiter = EvaluationFrequencyLimiter()
        
        # Test initial state
        should_skip, reason = limiter.should_skip_evaluation('vmaf')
        assert not should_skip, f"Should not skip initially, but got: {reason}"
        
        # Record some failures
        limiter.record_evaluation_attempt('vmaf', EvaluationResult.FAILURE, 30.0, "Test failure")
        limiter.record_evaluation_attempt('vmaf', EvaluationResult.FAILURE, 25.0, "Test failure 2")
        
        # Should skip after consecutive failures
        should_skip, reason = limiter.should_skip_evaluation('vmaf')
        assert should_skip, f"Should skip after failures, but didn't: {reason}"
        
        # Test statistics
        stats = limiter.get_session_statistics()
        assert stats['failed_attempts'] == 2
        assert stats['consecutive_failures'] == 2
        
        logger.info("✓ Frequency limiter tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Frequency limiter test failed: {e}")
        return False

def test_time_budget():
    """Test computation time budget."""
    logger.info("Testing Computation Time Budget...")
    
    try:
        from computation_time_budget import ComputationTimeBudget, BudgetStatus
        
        budget = ComputationTimeBudget()
        
        # Test budget configuration
        config = budget.get_budget_configuration()
        assert config['total_budget'] > 0
        assert config['vmaf_budget'] > 0
        assert config['ssim_budget'] > 0
        
        # Test operation start
        started = budget.start_operation('vmaf')
        assert started, "Should be able to start operation"
        
        # Test budget status
        status, elapsed, remaining = budget.check_budget_status()
        assert status == BudgetStatus.ACTIVE
        assert elapsed >= 0
        assert remaining > 0
        
        # Test operation finish
        summary = budget.finish_operation(success=True)
        assert summary['success'] == True
        assert summary['operation_type'] == 'vmaf'
        
        logger.info("✓ Time budget tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Time budget test failed: {e}")
        return False

def test_result_predictor():
    """Test evaluation result predictor."""
    logger.info("Testing Evaluation Result Predictor...")
    
    try:
        from evaluation_result_predictor import EvaluationResultPredictor, VideoCharacteristics
        
        predictor = EvaluationResultPredictor()
        
        # Create test video characteristics
        test_chars = VideoCharacteristics(
            duration=60.0,
            width=1920,
            height=1080,
            fps=30.0,
            bitrate=5000000,
            codec='h264',
            file_size_mb=50.0,
            complexity_score=0.5,
            motion_score=0.4,
            scene_changes=6
        )
        
        # Test prediction
        prediction = predictor.predict_quality_scores(test_chars)
        assert prediction.predicted_vmaf is not None
        assert prediction.predicted_ssim is not None
        assert 0 <= prediction.confidence_score <= 1
        assert prediction.estimated_evaluation_time > 0
        
        # Test historical result recording
        predictor.record_evaluation_result(
            test_chars, 85.0, 0.95, 45.0, True
        )
        
        # Test accuracy statistics
        accuracy = predictor.get_prediction_accuracy()
        assert accuracy['total_predictions'] == 1
        
        logger.info("✓ Result predictor tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Result predictor test failed: {e}")
        return False

def test_integrated_optimizer():
    """Test integrated performance optimizer."""
    logger.info("Testing Integrated Performance Optimizer...")
    
    try:
        from quality_evaluation_performance_optimizer import QualityEvaluationPerformanceOptimizer
        
        optimizer = QualityEvaluationPerformanceOptimizer()
        
        # Test skip evaluation logic (without actual video file)
        should_skip, reason, prediction = optimizer.should_skip_evaluation("nonexistent.mp4", "vmaf")
        # Should not skip for first attempt with nonexistent file
        
        # Test performance profile configuration
        optimizer.configure_performance_profile('fast')
        optimizer.configure_performance_profile('balanced')
        optimizer.configure_performance_profile('quality')
        
        # Test statistics
        stats = optimizer.get_optimization_statistics()
        assert 'frequency_limiter' in stats
        assert 'time_budget' in stats
        assert 'result_predictor' in stats
        
        # Test session reset
        optimizer.reset_session()
        
        logger.info("✓ Integrated optimizer tests passed")
        return True
        
    except Exception as e:
        logger.error(f"✗ Integrated optimizer test failed: {e}")
        return False

def main():
    """Run all performance optimization tests."""
    logger.info("Starting Quality Evaluation Performance Optimization Tests")
    
    tests = [
        test_frequency_limiter,
        test_time_budget,
        test_result_predictor,
        test_integrated_optimizer
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    logger.info(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All performance optimization tests passed!")
        return True
    else:
        logger.error(f"❌ {total - passed} tests failed")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)