"""Disdrop package root.

Export primary classes and CLI for convenience when installed via pip.
"""

from .cli import main as cli_main  # noqa: F401
from .video_compressor import DynamicVideoCompressor  # noqa: F401
from .gif_generator import GifGenerator  # noqa: F401
from .automated_workflow import AutomatedWorkflow  # noqa: F401
from .bitrate_validator import BitrateValidator, ValidationResult, AdjustmentPlan, BitrateValidationError  # noqa: F401
from .performance_cache_integration import PerformanceCacheSystem, create_performance_cache_system  # noqa: F401
from .performance_cache import PerformanceCache, QualityResult, CompressionResult  # noqa: F401
from .video_fingerprinter import VideoFingerprinter, CompressionParams  # noqa: F401
from .content_analysis_engine import ContentAnalysisEngine, ContentAnalysis, ContentType  # noqa: F401
from .content_aware_encoding_profiles import ContentAwareEncodingProfiles, EncodingProfile, BitrateAllocation  # noqa: F401
from .quality_size_tradeoff_analyzer import QualitySizeTradeoffAnalyzer, TradeoffAnalysis, QualitySizeOption  # noqa: F401
from .adaptive_quality_strategy import AdaptiveQualityStrategy  # noqa: F401