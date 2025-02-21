from typing import Optional


class GIFOptimizationError(Exception):
    """Base class for GIF optimization errors."""

    def __init__(self, message: str, details: Optional[str] = None):
        self.message = message
        self.details = details
        super().__init__(self.message)


class ResourceExhaustionError(GIFOptimizationError):
    """Raised when system resources are exhausted."""
    pass


class OptimizationFailedError(GIFOptimizationError):
    """Raised when optimization fails to meet target size."""
    pass


class DimensionError(GIFOptimizationError):
    """Raised when there are issues with image dimensions."""
    pass


class QualityThresholdError(GIFOptimizationError):
    """Raised when quality would fall below minimum threshold."""
    pass
