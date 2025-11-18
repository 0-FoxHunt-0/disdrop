"""
Enhanced Error Handling Module
Provides centralized error categorization, logging, and recovery strategies for bitrate validation
and other processing failures in batch operations.
"""

import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of processing errors for better handling and reporting"""
    BITRATE_VALIDATION = "bitrate_validation"
    SEGMENTATION = "segmentation" 
    ENCODER = "encoder"
    TIMEOUT = "timeout"
    MEMORY = "memory"
    FILE_VALIDATION = "file_validation"
    NETWORK = "network"
    PERMISSION = "permission"
    GENERAL = "general"


@dataclass
class ProcessingError:
    """Structured representation of a processing error"""
    category: ErrorCategory
    message: str
    file_path: str
    exception_type: str
    severity: str  # 'warning', 'error', 'critical'
    suggestions: List[str]
    retryable: bool = True
    context: Optional[str] = None
    
    def get_short_description(self) -> str:
        """Get concise error description for logging"""
        return f"{self.category.value}: {self.message}"
    
    def get_detailed_description(self) -> str:
        """Get detailed error description with suggestions"""
        base = f"Error in {self.file_path}: {self.message}"
        if self.context:
            base += f" (Context: {self.context})"
        
        if self.suggestions:
            base += "\nSuggestions:\n" + "\n".join(f"  • {s}" for s in self.suggestions)
        
        return base


class ErrorHandler:
    """Centralized error handling and categorization for batch processing"""
    
    def __init__(self):
        self.error_counts = {category: 0 for category in ErrorCategory}
        self.processed_errors: List[ProcessingError] = []
        
    def categorize_error(self, exception: Exception, file_path: str, 
                        context: str = None) -> ProcessingError:
        """Categorize an exception into a structured ProcessingError"""
        error_msg = str(exception)
        exception_type = type(exception).__name__
        
        # Bitrate validation errors
        if (hasattr(exception, '__class__') and 
            'BitrateValidationError' in exception_type):
            return ProcessingError(
                category=ErrorCategory.BITRATE_VALIDATION,
                message=error_msg,
                file_path=file_path,
                exception_type=exception_type,
                severity='critical' if hasattr(exception, 'severity') and exception.severity == 'critical' else 'error',
                suggestions=[
                    "Enable segmentation: --enable-segmentation",
                    "Lower bitrate floor: --bitrate-floor 1", 
                    "Use hardware encoders: --force-hardware",
                    "Increase target size if possible"
                ],
                retryable=True,
                context=context
            )
        
        # Pattern-based categorization for other errors
        error_lower = error_msg.lower()
        
        if 'bitrate' in error_lower and ('minimum' in error_lower or 'floor' in error_lower):
            return ProcessingError(
                category=ErrorCategory.BITRATE_VALIDATION,
                message=error_msg,
                file_path=file_path,
                exception_type=exception_type,
                severity='error',
                suggestions=[
                    "Try --bitrate-floor to override minimum requirements",
                    "Enable segmentation for long videos",
                    "Increase target file size"
                ],
                retryable=True,
                context=context
            )
        
        elif 'segmentation' in error_lower:
            return ProcessingError(
                category=ErrorCategory.SEGMENTATION,
                message=error_msg,
                file_path=file_path,
                exception_type=exception_type,
                severity='error',
                suggestions=[
                    "Check video duration and complexity",
                    "Try smaller segment duration",
                    "Force single file: --force-single-file"
                ],
                retryable=True,
                context=context
            )
        
        elif 'encoder' in error_lower or 'ffmpeg' in error_lower:
            return ProcessingError(
                category=ErrorCategory.ENCODER,
                message=error_msg,
                file_path=file_path,
                exception_type=exception_type,
                severity='error',
                suggestions=[
                    "Try software encoding: --force-software",
                    "Check video file integrity",
                    "Update FFmpeg installation",
                    "Try different encoder settings"
                ],
                retryable=True,
                context=context
            )
        
        elif 'timeout' in error_lower or 'timed out' in error_lower:
            return ProcessingError(
                category=ErrorCategory.TIMEOUT,
                message=error_msg,
                file_path=file_path,
                exception_type=exception_type,
                severity='warning',
                suggestions=[
                    "Reduce video resolution or duration",
                    "Enable segmentation for large files",
                    "Increase timeout settings"
                ],
                retryable=True,
                context=context
            )
        
        elif 'memory' in error_lower or 'out of memory' in error_lower:
            return ProcessingError(
                category=ErrorCategory.MEMORY,
                message=error_msg,
                file_path=file_path,
                exception_type=exception_type,
                severity='error',
                suggestions=[
                    "Reduce video resolution",
                    "Enable segmentation",
                    "Close other applications",
                    "Process files individually"
                ],
                retryable=False,
                context=context
            )
        
        elif 'permission' in error_lower or 'access denied' in error_lower:
            return ProcessingError(
                category=ErrorCategory.PERMISSION,
                message=error_msg,
                file_path=file_path,
                exception_type=exception_type,
                severity='error',
                suggestions=[
                    "Check file permissions",
                    "Run with appropriate privileges",
                    "Ensure output directory is writable"
                ],
                retryable=False,
                context=context
            )
        
        elif 'network' in error_lower or 'connection' in error_lower:
            return ProcessingError(
                category=ErrorCategory.NETWORK,
                message=error_msg,
                file_path=file_path,
                exception_type=exception_type,
                severity='warning',
                suggestions=[
                    "Check network connectivity",
                    "Retry operation",
                    "Use local processing only"
                ],
                retryable=True,
                context=context
            )
        
        elif ('invalid' in error_lower or 'corrupt' in error_lower or 
              'format' in error_lower):
            return ProcessingError(
                category=ErrorCategory.FILE_VALIDATION,
                message=error_msg,
                file_path=file_path,
                exception_type=exception_type,
                severity='warning',
                suggestions=[
                    "Check file integrity",
                    "Try re-downloading the file",
                    "Convert to standard format first"
                ],
                retryable=False,
                context=context
            )
        
        else:
            return ProcessingError(
                category=ErrorCategory.GENERAL,
                message=error_msg,
                file_path=file_path,
                exception_type=exception_type,
                severity='error',
                suggestions=[
                    "Check system resources",
                    "Retry operation",
                    "Check logs for more details"
                ],
                retryable=True,
                context=context
            )
    
    def handle_error(self, exception: Exception, file_path: str, 
                    context: str = None, continue_processing: bool = True) -> ProcessingError:
        """Handle an error by categorizing it and logging appropriately"""
        error = self.categorize_error(exception, file_path, context)
        self.processed_errors.append(error)
        self.error_counts[error.category] += 1
        
        # Log based on severity
        if error.severity == 'critical':
            logger.error(f"CRITICAL ERROR: {error.get_short_description()}")
            logger.error(f"Details: {error.get_detailed_description()}")
        elif error.severity == 'error':
            logger.error(f"ERROR: {error.get_short_description()}")
            logger.info(f"Suggestions: {'; '.join(error.suggestions[:2])}")
        else:
            logger.warning(f"WARNING: {error.get_short_description()}")
        
        # Log continuation message for batch processing
        if continue_processing:
            logger.info(f"Continuing batch processing despite {error.category.value} error")
        
        return error
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get comprehensive error summary for batch processing"""
        total_errors = len(self.processed_errors)
        if total_errors == 0:
            return {'total_errors': 0, 'categories': {}, 'success_rate': 100.0}
        
        category_counts = {cat.value: count for cat, count in self.error_counts.items() if count > 0}
        
        # Calculate severity distribution
        severity_counts = {}
        for error in self.processed_errors:
            severity_counts[error.severity] = severity_counts.get(error.severity, 0) + 1
        retryable_count = sum(1 for error in self.processed_errors if error.retryable)
        non_retryable_count = total_errors - retryable_count
        
        return {
            'total_errors': total_errors,
            'categories': category_counts,
            'severity_distribution': severity_counts,
            'most_common_category': max(category_counts.items(), key=lambda x: x[1])[0] if category_counts else None,
            'critical_errors': severity_counts.get('critical', 0),
            'error_rate': total_errors,  # Will be calculated by caller with total files
            'retryable_errors': retryable_count,
            'non_retryable_errors': non_retryable_count
        }

    def get_top_failures(self, limit: int = 3) -> List[Dict[str, Any]]:
        """Return ranked failure categories with a representative message for reporting."""
        if limit <= 0 or not self.processed_errors:
            return []
        category_counts: Dict[str, int] = {}
        sample_messages: Dict[str, str] = {}
        for error in self.processed_errors:
            key = error.category.value
            category_counts[key] = category_counts.get(key, 0) + 1
            # Capture the latest message for the category
            sample_messages[key] = error.message
        sorted_categories = sorted(category_counts.items(), key=lambda item: item[1], reverse=True)[:limit]
        top_failures = []
        for category, count in sorted_categories:
            top_failures.append({
                'category': category,
                'count': count,
                'sample_message': sample_messages.get(category, '')
            })
        return top_failures
    
    def log_batch_summary(self, total_files: int, successful_files: int):
        """Log comprehensive batch processing summary with error analysis"""
        failed_files = total_files - successful_files
        success_rate = (successful_files / total_files * 100) if total_files > 0 else 0
        
        logger.info("=== BATCH PROCESSING ERROR ANALYSIS ===")
        logger.info(f"Total files: {total_files}, Successful: {successful_files}, Failed: {failed_files}")
        logger.info(f"Success rate: {success_rate:.1f}%")
        
        if failed_files == 0:
            logger.info("No errors encountered - perfect batch processing!")
            return
        
        # Error category breakdown
        logger.error("Error breakdown by category:")
        for category, count in self.error_counts.items():
            if count > 0:
                percentage = (count / failed_files) * 100
                logger.error(f"  • {category.value}: {count} files ({percentage:.1f}% of failures)")
        
        # Top recommendations based on error patterns
        logger.info("=== RECOMMENDED ACTIONS ===")
        
        bitrate_errors = self.error_counts[ErrorCategory.BITRATE_VALIDATION]
        if bitrate_errors > 0:
            logger.info(f"For {bitrate_errors} bitrate validation failures:")
            logger.info("  • Enable segmentation: --enable-segmentation")
            logger.info("  • Lower bitrate floor: --bitrate-floor 1")
        
        encoder_errors = self.error_counts[ErrorCategory.ENCODER]
        if encoder_errors > 0:
            logger.info(f"For {encoder_errors} encoder failures:")
            logger.info("  • Try software encoding: --force-software")
            logger.info("  • Check FFmpeg installation")
        
        memory_errors = self.error_counts[ErrorCategory.MEMORY]
        if memory_errors > 0:
            logger.info(f"For {memory_errors} memory errors:")
            logger.info("  • Reduce video resolution")
            logger.info("  • Enable segmentation")
        
        # Resilience message
        if successful_files > 0:
            logger.info(f"Batch processing resilience: {success_rate:.1f}% success rate maintained despite errors")
        else:
            logger.error("Batch processing failed completely - check system configuration")
    
    def get_category_suggestions(self, category: ErrorCategory) -> List[str]:
        """Get specific suggestions for an error category"""
        suggestions_map = {
            ErrorCategory.BITRATE_VALIDATION: [
                "Enable segmentation: --enable-segmentation",
                "Lower bitrate floor: --bitrate-floor 1",
                "Use hardware encoders: --force-hardware",
                "Increase target size"
            ],
            ErrorCategory.SEGMENTATION: [
                "Check video duration and complexity",
                "Try smaller segment duration",
                "Force single file: --force-single-file"
            ],
            ErrorCategory.ENCODER: [
                "Try software encoding: --force-software",
                "Check video file integrity",
                "Update FFmpeg installation"
            ],
            ErrorCategory.MEMORY: [
                "Reduce video resolution",
                "Enable segmentation",
                "Close other applications"
            ],
            ErrorCategory.TIMEOUT: [
                "Reduce video resolution or duration",
                "Enable segmentation for large files"
            ]
        }
        
        return suggestions_map.get(category, ["Check system configuration", "Retry operation"])
    
    def reset(self):
        """Reset error tracking for new batch processing session"""
        self.error_counts = {category: 0 for category in ErrorCategory}
        self.processed_errors.clear()