"""Disdrop package root.

Export primary classes and CLI for convenience when installed via pip.
"""

from .cli import main as cli_main  # noqa: F401
from .video_processing.video_compressor import DynamicVideoCompressor  # noqa: F401
from .gif_processing.gif_generator import GifGenerator  # noqa: F401
from .automated_workflow import AutomatedWorkflow  # noqa: F401
from .video_processing.bitrate_validator import BitrateValidator, ValidationResult, AdjustmentPlan, BitrateValidationError  # noqa: F401
from .video_processing.cache_manager import CacheManager, create_cache_manager  # noqa: F401
from .video_processing.performance_cache import PerformanceCache, QualityResult, CompressionResult  # noqa: F401
from .video_processing.video_fingerprinter import VideoFingerprinter, CompressionParams  # noqa: F401
from .video_processing.quality_predictor import QualityPredictor, PredictionStrategy, QualityPrediction, VideoCharacteristics  # noqa: F401
from .video_processing.content_analysis_engine import ContentAnalysisEngine, ContentAnalysis, ContentType  # noqa: F401
from .video_processing.content_aware_encoding_profiles import ContentAwareEncodingProfiles, EncodingProfile, BitrateAllocation  # noqa: F401
from .video_processing.quality_size_tradeoff_analyzer import QualitySizeTradeoffAnalyzer, TradeoffAnalysis, QualitySizeOption  # noqa: F401
from .video_processing.adaptive_quality_strategy import AdaptiveQualityStrategy  # noqa: F401