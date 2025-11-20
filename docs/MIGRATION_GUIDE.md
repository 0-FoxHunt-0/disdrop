# Migration Guide: Component Consolidation

This guide helps you migrate from deprecated components to the new unified interfaces.

## Cache Layer Migration

### From PerformanceCacheSystem to CacheManager

**Before:**
```python
from disdrop import PerformanceCacheSystem, create_performance_cache_system

# Old way
cache_system = create_performance_cache_system(
    cache_dir="/path/to/cache",
    max_size_mb=500,
    enable_warming=True
)

result, is_hit, time = cache_system.get_quality_result(video_path, params)
cache_system.cache_quality_result(video_path, params, result)
```

**After:**
```python
from disdrop import CacheManager, create_cache_manager

# New way
cache_manager = create_cache_manager(
    cache_dir="/path/to/cache",
    max_size_mb=500,
    enable_warming=True
)

# Same API, different class name
result, is_hit, time = cache_manager.get_quality_result(video_path, params)
cache_manager.cache_quality_result(video_path, params, result)
```

**Changes:**
- `PerformanceCacheSystem` → `CacheManager`
- `create_performance_cache_system()` → `create_cache_manager()`
- All method names remain the same

## Quality Prediction Migration

### From Multiple Components to Unified QualityPredictor

**Before:**
```python
# Statistical prediction
from disdrop import QualityPredictionModels
models = QualityPredictionModels(config_manager)
prediction = models.predict_quality_scores(lightweight_result)

# Fast estimation
from disdrop import FastQualityEstimator
estimator = FastQualityEstimator(config_manager)
estimate = estimator.estimate_quality_fast(video_path)

# Characteristics-based prediction
from disdrop import EvaluationResultPredictor
predictor = EvaluationResultPredictor(config_manager)
video_chars = predictor.analyze_video_characteristics(video_path)
prediction = predictor.predict_quality_scores(video_chars)
```

**After:**
```python
from disdrop import QualityPredictor, PredictionStrategy

# Unified predictor
predictor = QualityPredictor(config_manager)

# Statistical prediction (same as QualityPredictionModels)
prediction = predictor.predict_quality(
    video_path,
    strategy=PredictionStrategy.STATISTICAL,
    lightweight_result=lightweight_result
)

# Fast estimation (same as FastQualityEstimator)
prediction = predictor.predict_quality(
    video_path,
    strategy=PredictionStrategy.FAST_ESTIMATION,
    target_duration=30.0
)

# Characteristics-based (same as EvaluationResultPredictor)
prediction = predictor.predict_quality(
    video_path,
    strategy=PredictionStrategy.CHARACTERISTICS_BASED,
    video_characteristics=video_chars
)

# Or auto-select strategy
prediction = predictor.predict_quality(
    video_path,
    strategy=PredictionStrategy.AUTO
)
```

**Changes:**
- `QualityPredictionModels.predict_quality_scores()` → `QualityPredictor.predict_quality()` with `STATISTICAL` strategy
- `FastQualityEstimator.estimate_quality_fast()` → `QualityPredictor.predict_quality()` with `FAST_ESTIMATION` strategy
- `EvaluationResultPredictor.predict_quality_scores()` → `QualityPredictor.predict_quality()` with `CHARACTERISTICS_BASED` strategy

## Data Structure Changes

### QualityPrediction

The unified `QualityPrediction` dataclass combines features from all three old components:

**Old QualityPrediction (from quality_prediction_models.py):**
```python
@dataclass
class QualityPrediction:
    predicted_vmaf: float
    predicted_ssim: float
    vmaf_confidence: float
    ssim_confidence: float
    overall_confidence: float
    model_version: str
    prediction_time: float
    feature_importance: Dict[str, float]
```

**New Unified QualityPrediction:**
```python
@dataclass
class QualityPrediction:
    predicted_vmaf: Optional[float]
    predicted_ssim: Optional[float]
    vmaf_confidence: float
    ssim_confidence: float
    overall_confidence: float
    confidence_level: PredictionConfidence  # NEW
    strategy: PredictionStrategy  # NEW
    prediction_time: float
    should_skip_evaluation: bool  # NEW
    prediction_basis: str  # NEW
    estimated_evaluation_time: float  # NEW
    risk_factors: List[str]  # NEW
    feature_importance: Dict[str, float]
    model_version: str = "unified_1.0"
```

**Migration:**
- All existing fields are preserved
- New fields provide additional information
- `predicted_vmaf` and `predicted_ssim` are now `Optional[float]` for better error handling

## Import Changes

### Before
```python
from disdrop import (
    PerformanceCacheSystem,
    create_performance_cache_system,
    QualityPredictionModels,
    FastQualityEstimator,
    EvaluationResultPredictor
)
```

### After
```python
from disdrop import (
    CacheManager,
    create_cache_manager,
    QualityPredictor,
    PredictionStrategy
)
```

## API Compatibility

### CacheManager API

All methods from `PerformanceCacheSystem` are available in `CacheManager`:

- ✅ `get_quality_result()` - Same signature
- ✅ `cache_quality_result()` - Same signature
- ✅ `get_compression_result()` - Same signature
- ✅ `cache_compression_result()` - Same signature
- ✅ `find_similar_results()` - Same signature
- ✅ `get_cache_statistics()` - Same signature
- ✅ `cleanup_cache()` - Same signature
- ✅ `clear_cache()` - Same signature
- ✅ `export_performance_report()` - Same signature
- ✅ `configure_cache_warming()` - Same signature
- ✅ `optimize_cache_performance()` - Same signature
- ✅ `get_video_fingerprint()` - Same signature
- ✅ `calculate_video_similarity()` - Same signature
- ✅ `shutdown()` - Same signature

### QualityPredictor API

Most functionality is available, but with unified interface:

- ✅ `predict_quality()` - Unified method replaces all three old methods
- ✅ `record_evaluation_result()` - Similar to old `EvaluationResultPredictor`
- ✅ `get_prediction_accuracy()` - Similar to old `EvaluationResultPredictor`

## Testing

Update your tests to use the new components:

```python
# Old test
from disdrop import PerformanceCacheSystem
cache = PerformanceCacheSystem()

# New test
from disdrop import CacheManager
cache = CacheManager()
```

```python
# Old test
from disdrop import FastQualityEstimator
estimator = FastQualityEstimator()
estimate = estimator.estimate_quality_fast(video_path)

# New test
from disdrop import QualityPredictor, PredictionStrategy
predictor = QualityPredictor()
prediction = predictor.predict_quality(
    video_path,
    strategy=PredictionStrategy.FAST_ESTIMATION
)
```

## Common Issues

### Issue: ImportError for old components

**Solution:** Update imports to use new unified components:
- `PerformanceCacheSystem` → `CacheManager`
- `QualityPredictionModels` → `QualityPredictor`
- `FastQualityEstimator` → `QualityPredictor`
- `EvaluationResultPredictor` → `QualityPredictor`

### Issue: Missing methods

**Solution:** All methods from old components are available in new unified components. Check the API documentation for exact method names.

### Issue: Different return types

**Solution:** The unified `QualityPrediction` dataclass includes all fields from old components plus new fields. Access fields as before, new fields are optional.

## Benefits of Migration

1. **Simplified API**: One component instead of three for quality prediction
2. **Better Strategy Selection**: Automatic strategy selection based on available data
3. **Unified Interface**: Consistent API across all prediction methods
4. **Better Performance**: Optimized implementations in unified components
5. **Easier Maintenance**: Single codebase instead of multiple overlapping components














