# Disdrop Architecture Documentation

## Overview

This document describes the architecture of the Disdrop video compression system, including component responsibilities, usage guidelines, and integration patterns.

## Component Architecture

### Cache Layer

#### CacheManager (`src/cache_manager.py`)
**Primary unified interface for all caching functionality.**

**Responsibilities:**
- Cache management and optimization
- Performance monitoring and metrics collection
- Automatic cache cleanup and size management
- Cache warming strategies
- Video fingerprinting integration
- Quality result caching with prediction

**Key Methods:**
- `get_quality_result()` - Get quality result with cache lookup and prediction
- `cache_quality_result()` - Cache quality results with optimization
- `get_compression_result()` - Get cached compression results
- `find_similar_results()` - Find similar video results for prediction
- `get_cache_statistics()` - Get comprehensive cache statistics
- `cleanup_cache()` - Clean up expired entries
- `configure_cache_warming()` - Configure cache warming

**Usage:**
```python
from disdrop import CacheManager

# Create cache manager
cache_manager = CacheManager(cache_dir="/path/to/cache", max_cache_size_mb=500)

# Get quality result (with automatic prediction)
result, is_cache_hit, response_time = cache_manager.get_quality_result(
    video_path, compression_params
)

# Cache quality result
cache_manager.cache_quality_result(video_path, compression_params, quality_result)
```

**Migration from PerformanceCacheSystem:**
- `PerformanceCacheSystem` has been merged into `CacheManager`
- Use `CacheManager` instead of `PerformanceCacheSystem`
- `create_performance_cache_system()` replaced by `create_cache_manager()`

#### PerformanceCache (`src/performance_cache.py`)
**Core SQLite-based caching system.**

**Responsibilities:**
- Low-level cache storage and retrieval
- Database management
- Quality result prediction from similar videos
- Cache entry expiration and cleanup

**Note:** Typically accessed through `CacheManager`, not directly.

### Quality Prediction Layer

#### QualityPredictor (`src/quality_predictor.py`)
**Unified quality predictor with multiple strategies.**

**Responsibilities:**
- Statistical prediction from lightweight metrics
- Fast estimation via sampling
- Risk-based prediction from video characteristics
- Automatic strategy selection
- Learning from historical results

**Prediction Strategies:**
1. **STATISTICAL** - Predicts from lightweight quality metrics (blur, noise, complexity, motion)
2. **FAST_ESTIMATION** - Fast estimation via video sampling + lightweight metrics
3. **CHARACTERISTICS_BASED** - Predicts from video characteristics (resolution, bitrate, complexity)
4. **AUTO** - Automatically selects best strategy based on available data

**Key Methods:**
- `predict_quality()` - Main prediction method with strategy selection
- `record_evaluation_result()` - Record actual results for learning
- `get_prediction_accuracy()` - Get accuracy statistics

**Usage:**
```python
from disdrop import QualityPredictor, PredictionStrategy

# Create predictor
predictor = QualityPredictor(config_manager)

# Predict quality (auto-select strategy)
prediction = predictor.predict_quality(video_path, strategy=PredictionStrategy.AUTO)

# Or use specific strategy
prediction = predictor.predict_quality(
    video_path,
    strategy=PredictionStrategy.STATISTICAL,
    lightweight_result=lightweight_result
)

# Record actual results for learning
predictor.record_evaluation_result(
    video_chars, actual_vmaf, actual_ssim, evaluation_time, success
)
```

**Migration from Old Components:**
- `QualityPredictionModels` → Use `QualityPredictor` with `STATISTICAL` strategy
- `FastQualityEstimator` → Use `QualityPredictor` with `FAST_ESTIMATION` strategy
- `EvaluationResultPredictor` → Use `QualityPredictor` with `CHARACTERISTICS_BASED` strategy

### Performance Monitoring Layer

#### PerformanceMonitor (`src/performance_monitor.py`)
**Core metrics collection and session tracking.**

**Responsibilities:**
- Compression session tracking
- Operation metrics collection
- Cache hit/miss tracking
- Performance profile management
- Session metrics export

#### PerformanceAnalyzer (`src/performance_analyzer.py`)
**Advanced analysis of performance metrics.**

**Responsibilities:**
- Trend analysis
- Bottleneck identification
- Optimization recommendations
- Performance report generation

**Note:** Used internally by `PerformanceController`.

#### PerformanceController (`src/performance_controls.py`)
**User-facing performance features.**

**Responsibilities:**
- Performance profile management
- Progress reporting
- Time estimation
- User-friendly performance summaries
- Bottleneck reporting

**Usage:**
```python
from disdrop import PerformanceController

controller = PerformanceController(config_manager)

# Set performance profile
controller.set_performance_profile('fast')  # or 'balanced', 'quality'

# Get performance summary
summary = controller.get_performance_summary()

# Get bottleneck report
bottlenecks = controller.get_bottleneck_report()
```

### Quality Evaluation Layer

#### QualityGates (`src/quality_gates.py`)
**Objective quality measurement using VMAF and SSIM.**

**Responsibilities:**
- VMAF computation
- SSIM computation
- Quality validation against thresholds
- Resolution scaling for comparison

#### QualityEvaluationPerformanceOptimizer (`src/quality_evaluation_performance_optimizer.py`)
**Integrates performance optimization for quality evaluation.**

**Responsibilities:**
- Frequency limiting
- Time budget management
- Quality prediction integration
- Optimized evaluation execution

**Note:** Uses unified `QualityPredictor` internally.

### Video Compression Layer

#### DynamicVideoCompressor (`src/video_compressor.py`)
**Main orchestrator for video compression.**

**Responsibilities:**
- Compression workflow orchestration
- Component coordination
- Quality gates integration
- Artifact detection
- Compression strategy selection

## Component Responsibilities Summary

| Component | Responsibility | Replaces |
|-----------|---------------|----------|
| `CacheManager` | Unified cache interface | `PerformanceCacheSystem` |
| `PerformanceCache` | Core cache storage | - |
| `QualityPredictor` | Unified quality prediction | `QualityPredictionModels`, `FastQualityEstimator`, `EvaluationResultPredictor` |
| `PerformanceMonitor` | Metrics collection | - |
| `PerformanceAnalyzer` | Performance analysis | - |
| `PerformanceController` | User-facing controls | - |
| `QualityGates` | Quality evaluation | - |
| `QualityEvaluationPerformanceOptimizer` | Evaluation optimization | - |

## Integration Patterns

### Cache Usage Pattern
```python
# Recommended: Use CacheManager
cache_manager = CacheManager()

# Get quality result
result, is_hit, time = cache_manager.get_quality_result(video_path, params)

# Cache result
cache_manager.cache_quality_result(video_path, params, result)
```

### Quality Prediction Pattern
```python
# Recommended: Use QualityPredictor with auto strategy
predictor = QualityPredictor(config_manager)

# Predict with automatic strategy selection
prediction = predictor.predict_quality(video_path)

# Use prediction to skip evaluation
if prediction.should_skip_evaluation:
    use_prediction(prediction)
else:
    run_full_evaluation()
```

### Performance Monitoring Pattern
```python
# Use PerformanceController for user-facing features
controller = PerformanceController(config_manager)

# Set profile
controller.set_performance_profile('balanced')

# Monitor performance
summary = controller.get_performance_summary()
```

## Deprecated Components

The following components have been removed and their functionality merged:

- `PerformanceCacheSystem` → Use `CacheManager`
- `QualityPredictionModels` → Use `QualityPredictor` with `STATISTICAL` strategy
- `FastQualityEstimator` → Use `QualityPredictor` with `FAST_ESTIMATION` strategy
- `EvaluationResultPredictor` → Use `QualityPredictor` with `CHARACTERISTICS_BASED` strategy

## Component Dependencies

```
CacheManager
  ├── PerformanceCache (core storage)
  ├── VideoFingerprinter (fingerprinting)
  └── CacheKeyGenerator (key generation)

QualityPredictor
  ├── LightweightQualityMetrics (statistical prediction)
  ├── SamplingEngine (fast estimation)
  └── (internal models for characteristics-based)

PerformanceController
  ├── PerformanceMonitor (metrics)
  └── PerformanceAnalyzer (analysis)

QualityEvaluationPerformanceOptimizer
  ├── EvaluationFrequencyLimiter
  ├── ComputationTimeBudget
  └── QualityPredictor (unified)
```

## Best Practices

1. **Cache Management**: Always use `CacheManager` for cache operations, not `PerformanceCache` directly
2. **Quality Prediction**: Use `QualityPredictor` with `AUTO` strategy for best results
3. **Performance Monitoring**: Use `PerformanceController` for user-facing performance features
4. **Strategy Selection**: Let `QualityPredictor` auto-select strategy unless you have specific requirements













