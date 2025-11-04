#!/usr/bin/env python3
"""
Test script for Quality-Size Optimization Engine
Tests the implementation of task 3.3
"""

import sys
import os
import logging

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from quality_size_optimization_engine import (
    QualitySizeOptimizationEngine, OptimizationConstraints, 
    ParameterSpace, OptimizationPoint
)
from quality_improvement_engine import (
    QualityImprovementEngine, FailureAnalysis, CompressionParams,
    FailureType, RootCause
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def test_quality_size_optimization_engine():
    """Test the quality-size optimization engine implementation."""
    
    logger.info("Testing Quality-Size Optimization Engine")
    
    # Create test instances
    optimization_engine = QualitySizeOptimizationEngine()
    quality_engine = QualityImprovementEngine()
    
    # Create test constraints
    constraints = OptimizationConstraints(
        max_size_mb=10.0,
        min_quality_vmaf=70.0,
        min_quality_ssim=0.90,
        max_iterations=5,
        convergence_threshold=1.0,
        size_tolerance_mb=0.1,
        quality_improvement_threshold=2.0,
        time_budget_seconds=60
    )
    
    # Create test parameter space
    parameter_space = ParameterSpace(
        bitrate_range=(1000, 5000),
        crf_range=(18, 28),
        preset_options=['fast', 'medium', 'slow'],
        encoder_options=['libx264', 'libx265'],
        resolution_factors=[0.8, 0.9, 1.0],
        fps_factors=[0.9, 1.0]
    )
    
    # Create test initial point
    initial_point = OptimizationPoint(
        bitrate=2000,
        crf=23,
        preset='medium',
        encoder='libx264',
        resolution_factor=1.0,
        fps_factor=1.0
    )
    
    # Test optimization
    try:
        result = optimization_engine.optimize_quality_size_tradeoff(
            input_path="test_video.mp4",  # Dummy path for testing
            constraints=constraints,
            parameter_space=parameter_space,
            initial_point=initial_point,
            video_characteristics={'motion_level': 'medium', 'complexity': 'medium', 'duration': 60}
        )
        
        logger.info(f"Optimization completed successfully!")
        logger.info(f"Best point: bitrate={result.best_point.bitrate}, quality={result.best_point.quality_score}")
        logger.info(f"Convergence status: {result.convergence_status.value}")
        logger.info(f"Total iterations: {result.total_iterations}")
        logger.info(f"Fallback used: {result.fallback_used}")
        
        return True
        
    except Exception as e:
        logger.error(f"Optimization test failed: {e}")
        return False

def test_quality_improvement_integration():
    """Test the integration with quality improvement engine."""
    
    logger.info("Testing Quality Improvement Engine Integration")
    
    # Create test instances
    quality_engine = QualityImprovementEngine()
    
    # Create test failure analysis
    from quality_improvement_engine import ImprovementStrategy
    
    analysis = FailureAnalysis(
        failure_types=[FailureType.LOW_VMAF, FailureType.HIGH_BLOCKINESS],
        root_causes=[RootCause.INSUFFICIENT_BITRATE],
        vmaf_score=65.0,
        ssim_score=0.88,
        blockiness_score=0.15,
        banding_score=0.08,
        blur_score=None,
        noise_score=None,
        primary_issue=FailureType.LOW_VMAF,
        primary_root_cause=RootCause.INSUFFICIENT_BITRATE,
        severity=0.7,
        confidence=0.8,
        recommended_strategies=[ImprovementStrategy.INCREASE_BITRATE, ImprovementStrategy.BETTER_ENCODING],
        size_headroom_mb=2.0,
        can_improve=True,
        detailed_analysis={},
        improvement_potential=0.8
    )
    
    # Create test compression parameters
    current_params = CompressionParams(
        bitrate=1500,
        width=1920,
        height=1080,
        fps=30.0,
        encoder='libx264',
        preset='medium',
        crf=23,
        tune=None,
        gop_size=60,
        b_frames=3
    )
    
    # Test constraint-based optimization
    try:
        optimized_params = quality_engine.optimize_parameters_with_constraints(
            input_path="test_video.mp4",  # Dummy path for testing
            analysis=analysis,
            current_params=current_params,
            size_limit_mb=10.0,
            duration_seconds=60.0,
            video_characteristics={'motion_level': 'medium', 'complexity': 'medium'},
            use_multi_dimensional=True
        )
        
        if optimized_params:
            logger.info(f"Integration test successful!")
            logger.info(f"Original bitrate: {current_params.bitrate}k")
            logger.info(f"Optimized bitrate: {optimized_params.bitrate}k")
            logger.info(f"Encoder: {optimized_params.encoder}")
            logger.info(f"Preset: {optimized_params.preset}")
            return True
        else:
            logger.warning("Integration test returned no optimized parameters")
            return False
        
    except Exception as e:
        logger.error(f"Integration test failed: {e}")
        return False

def test_iterative_improvement():
    """Test iterative improvement with convergence detection."""
    
    logger.info("Testing Iterative Improvement with Convergence Detection")
    
    # Create a mock config that sets a lower threshold
    class MockConfig:
        def get(self, key, default):
            if key == 'quality_improvement.quality_improvement_threshold':
                return 1.0  # Lower threshold for testing
            return default
    
    quality_engine = QualityImprovementEngine(MockConfig())
    
    # Create test failure analysis
    from quality_improvement_engine import ImprovementStrategy
    
    analysis = FailureAnalysis(
        failure_types=[FailureType.LOW_VMAF],
        root_causes=[RootCause.INSUFFICIENT_BITRATE],
        vmaf_score=68.0,
        ssim_score=0.91,
        blockiness_score=0.10,
        banding_score=0.05,
        blur_score=None,
        noise_score=None,
        primary_issue=FailureType.LOW_VMAF,
        primary_root_cause=RootCause.INSUFFICIENT_BITRATE,
        severity=0.6,
        confidence=0.9,
        recommended_strategies=[ImprovementStrategy.INCREASE_BITRATE],
        size_headroom_mb=1.5,
        can_improve=True,
        detailed_analysis={},
        improvement_potential=0.7
    )
    
    current_params = CompressionParams(
        bitrate=1800,
        width=1920,
        height=1080,
        fps=30.0,
        encoder='libx264',
        preset='fast',
        crf=25,
        tune=None,
        gop_size=60,
        b_frames=2
    )
    
    try:
        improved_params = quality_engine._iterative_improvement_with_convergence(
            analysis=analysis,
            current_params=current_params,
            size_limit_mb=12.0,
            duration_seconds=60.0,
            max_iterations=5
        )
        
        if improved_params:
            logger.info(f"Iterative improvement successful!")
            logger.info(f"Original: {current_params.bitrate}k, {current_params.preset}")
            logger.info(f"Improved: {improved_params.bitrate}k, {improved_params.preset}")
            return True
        else:
            logger.warning("Iterative improvement returned no improvements")
            return False
            
    except Exception as e:
        logger.error(f"Iterative improvement test failed: {e}")
        return False

def main():
    """Run all tests."""
    
    logger.info("Starting Quality-Size Optimization Engine Tests")
    logger.info("=" * 60)
    
    tests = [
        ("Quality-Size Optimization Engine", test_quality_size_optimization_engine),
        ("Quality Improvement Integration", test_quality_improvement_integration),
        ("Iterative Improvement", test_iterative_improvement)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\nRunning test: {test_name}")
        logger.info("-" * 40)
        
        try:
            success = test_func()
            results.append((test_name, success))
            
            if success:
                logger.info(f"✓ {test_name} PASSED")
            else:
                logger.error(f"✗ {test_name} FAILED")
                
        except Exception as e:
            logger.error(f"✗ {test_name} ERROR: {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "PASSED" if success else "FAILED"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Task 3.3 implementation is working correctly.")
        return 0
    else:
        logger.error("❌ Some tests failed. Please check the implementation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())