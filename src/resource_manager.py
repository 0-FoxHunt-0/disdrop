#!/usr/bin/env python3
import os
import signal
import logging
import atexit
import threading
import time
import datetime
from pathlib import Path
from typing import List, Dict, Set, Callable, Optional, Any, Union
from contextlib import contextmanager

# Import for performance monitoring
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False


class PerformanceMonitor:
    """
    Monitors and records performance metrics such as execution time, CPU usage, and memory usage.
    Can be used as a context manager or decorator to track performance of code blocks.

    Usage as context manager:
        with PerformanceMonitor("operation_name") as monitor:
            # code to monitor

    Usage as decorator:
        @PerformanceMonitor.track("operation_name")
        def function_to_monitor():
            # function code

    Manual usage:
        monitor = PerformanceMonitor("operation_name")
        monitor.start()
        # code to monitor
        metrics = monitor.stop()
    """

    # Class-level storage for metrics from all monitors
    _metrics_history = []
    _metrics_lock = threading.RLock()
    _logger = logging.getLogger("performance_monitor")

    def __init__(self, operation_name: str, log_level: str = "INFO",
                 auto_report: bool = True, track_resources: bool = True):
        """
        Initialize a performance monitor.

        Args:
            operation_name: Name of the operation being monitored
            log_level: Logging level for performance reports
            auto_report: Whether to automatically log results when monitoring stops
            track_resources: Whether to track CPU and memory usage (requires psutil)
        """
        self.operation_name = operation_name
        self.log_level = log_level.upper()
        self.auto_report = auto_report
        self.track_resources = track_resources and PSUTIL_AVAILABLE

        # Timing data
        self.start_time = None
        self.end_time = None
        self.elapsed_time = None

        # Resource usage data
        self.start_cpu_percent = None
        self.end_cpu_percent = None
        self.avg_cpu_percent = None
        self.peak_cpu_percent = None

        self.start_memory = None
        self.end_memory = None
        self.peak_memory = None
        self.memory_diff = None

        # Internal tracking
        self._is_running = False
        self._cpu_samples = []
        self._memory_samples = []
        self._sampling_thread = None
        self._keep_sampling = False

    def start(self) -> 'PerformanceMonitor':
        """Start monitoring performance."""
        if self._is_running:
            return self

        self.start_time = time.time()
        self._is_running = True

        # Get initial resource usage if tracking is enabled
        if self.track_resources:
            process = psutil.Process(os.getpid())
            self.start_cpu_percent = process.cpu_percent(interval=0.1)
            self.start_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Start background sampling thread for more accurate peak values
            self._keep_sampling = True
            self._cpu_samples = [self.start_cpu_percent]
            self._memory_samples = [self.start_memory]
            self._sampling_thread = threading.Thread(
                target=self._sample_resources, daemon=True)
            self._sampling_thread.start()

        return self

    def stop(self) -> Dict[str, Any]:
        """
        Stop monitoring and return performance metrics.

        Returns:
            Dict with performance metrics
        """
        if not self._is_running:
            return self.get_metrics()

        self.end_time = time.time()
        self.elapsed_time = self.end_time - self.start_time
        self._is_running = False

        # Stop resource sampling thread if it exists
        if self._sampling_thread is not None:
            self._keep_sampling = False
            self._sampling_thread.join(timeout=1.0)

        # Get final resource usage if tracking is enabled
        if self.track_resources:
            process = psutil.Process(os.getpid())
            self.end_cpu_percent = process.cpu_percent(interval=0.1)
            self.end_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Calculate derived statistics
            if len(self._cpu_samples) > 0:
                self.avg_cpu_percent = sum(
                    self._cpu_samples) / len(self._cpu_samples)
                self.peak_cpu_percent = max(self._cpu_samples)

            if len(self._memory_samples) > 0:
                self.peak_memory = max(self._memory_samples)
                self.memory_diff = self.end_memory - self.start_memory

        # Get metrics and store in history
        metrics = self.get_metrics()
        with PerformanceMonitor._metrics_lock:
            PerformanceMonitor._metrics_history.append(metrics)

        # Log report if auto_report is enabled
        if self.auto_report:
            self.report()

        return metrics

    def _sample_resources(self) -> None:
        """Sample resource usage in the background."""
        process = psutil.Process(os.getpid())

        while self._keep_sampling and self._is_running:
            try:
                # Sample CPU and memory usage
                cpu_percent = process.cpu_percent(interval=0.1)
                memory_mb = process.memory_info().rss / 1024 / 1024

                # Record samples
                self._cpu_samples.append(cpu_percent)
                self._memory_samples.append(memory_mb)

                # Sleep briefly to avoid excessive sampling
                time.sleep(0.5)
            except Exception as e:
                PerformanceMonitor._logger.warning(
                    f"Error sampling resources: {e}")
                break

    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current performance metrics.

        Returns:
            Dict containing performance metrics
        """
        metrics = {
            "operation": self.operation_name,
            "timestamp": datetime.datetime.now().isoformat(),
            "start_time": self.start_time,
            "end_time": self.end_time,
            "elapsed_time_seconds": self.elapsed_time
        }

        # Add resource metrics if tracked
        if self.track_resources:
            metrics.update({
                "cpu_usage_start_percent": self.start_cpu_percent,
                "cpu_usage_end_percent": self.end_cpu_percent,
                "cpu_usage_avg_percent": self.avg_cpu_percent,
                "cpu_usage_peak_percent": self.peak_cpu_percent,
                "memory_usage_start_mb": self.start_memory,
                "memory_usage_end_mb": self.end_memory,
                "memory_usage_peak_mb": self.peak_memory,
                "memory_usage_diff_mb": self.memory_diff
            })

        return metrics

    def report(self) -> None:
        """Log the performance metrics."""
        if not self.elapsed_time:
            PerformanceMonitor._logger.warning(
                f"Cannot report on operation '{self.operation_name}' - monitoring not completed")
            return

        # Format time as HH:MM:SS.mmm
        elapsed_str = str(datetime.timedelta(
            seconds=round(self.elapsed_time, 3)))

        # Basic message with elapsed time
        message = f"Performance: '{self.operation_name}' completed in {elapsed_str}"

        # Add resource usage if available
        if self.track_resources and self.avg_cpu_percent is not None:
            resource_info = (
                f" - CPU: {self.avg_cpu_percent:.1f}% avg, {self.peak_cpu_percent:.1f}% peak; "
                f"Memory: {self.peak_memory:.1f} MB peak, {self.memory_diff:+.1f} MB change"
            )
            message += resource_info

        # Log message at specified level
        log_method = getattr(PerformanceMonitor._logger,
                             self.log_level.lower())
        log_method(message)

    def __enter__(self) -> 'PerformanceMonitor':
        """Context manager entry point."""
        return self.start()

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit point."""
        self.stop()

    @classmethod
    def get_history(cls) -> List[Dict[str, Any]]:
        """
        Get history of all recorded metrics.

        Returns:
            List of metric dictionaries
        """
        with cls._metrics_lock:
            return cls._metrics_history.copy()

    @classmethod
    def clear_history(cls) -> None:
        """Clear the metrics history."""
        with cls._metrics_lock:
            cls._metrics_history = []

    @classmethod
    def get_summary(cls, operation_filter: Optional[str] = None) -> Dict[str, Any]:
        """
        Get a summary of metrics for a specific operation or all operations.

        Args:
            operation_filter: Optional operation name to filter by

        Returns:
            Dict containing summary metrics
        """
        with cls._metrics_lock:
            # Filter metrics if operation_filter is provided
            if operation_filter:
                metrics = [m for m in cls._metrics_history if m.get(
                    "operation") == operation_filter]
            else:
                metrics = cls._metrics_history.copy()

            if not metrics:
                return {"count": 0}

            # Calculate summary statistics
            elapsed_times = [m.get("elapsed_time_seconds", 0)
                             for m in metrics if m.get("elapsed_time_seconds")]

            summary = {
                "count": len(metrics),
                "total_time_seconds": sum(elapsed_times),
                "avg_time_seconds": sum(elapsed_times) / len(elapsed_times) if elapsed_times else 0,
                "min_time_seconds": min(elapsed_times) if elapsed_times else 0,
                "max_time_seconds": max(elapsed_times) if elapsed_times else 0
            }

            # Add resource metrics if available
            cpu_avgs = [m.get("cpu_usage_avg_percent", 0)
                        for m in metrics if m.get("cpu_usage_avg_percent")]
            if cpu_avgs:
                summary["avg_cpu_percent"] = sum(cpu_avgs) / len(cpu_avgs)

            memory_peaks = [m.get("memory_usage_peak_mb", 0)
                            for m in metrics if m.get("memory_usage_peak_mb")]
            if memory_peaks:
                summary["avg_peak_memory_mb"] = sum(
                    memory_peaks) / len(memory_peaks)
                summary["max_peak_memory_mb"] = max(memory_peaks)

            return summary

    @classmethod
    @contextmanager
    def temporary_level(cls, level: str):
        """
        Temporarily change the logging level for the performance monitor.

        Args:
            level: Logging level to use temporarily
        """
        original_level = cls._logger.level
        cls._logger.setLevel(level.upper())
        try:
            yield
        finally:
            cls._logger.setLevel(original_level)

    @staticmethod
    def track(operation_name: str, log_level: str = "INFO",
              auto_report: bool = True, track_resources: bool = True):
        """
        Decorator to track performance of a function.

        Args:
            operation_name: Name of the operation being monitored
            log_level: Logging level for performance reports
            auto_report: Whether to automatically log results when monitoring stops
            track_resources: Whether to track CPU and memory usage

        Returns:
            Decorator function
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                with PerformanceMonitor(
                    operation_name=operation_name,
                    log_level=log_level,
                    auto_report=auto_report,
                    track_resources=track_resources
                ):
                    return func(*args, **kwargs)
            return wrapper
        return decorator


class ResourceManager:
    """
    Manages system resources and handles interruption signals (CTRL+C).
    Provides functionality to register temporary resources that need to be cleaned up
    when the program exits normally or is interrupted.

    Features:
    - Signal handling for graceful termination (CTRL+C)
    - Registration system for temporary files and directories
    - Cleanup hooks for custom resource cleanup
    - Thread-safe operation
    - Performance monitoring for operations
    """

    def __init__(self, config: Dict = None):
        """
        Initialize the ResourceManager.

        Args:
            config: Configuration dictionary for the resource manager
        """
        self.logger = logging.getLogger(__name__)
        self.config = config or {}

        # Temp directories to monitor
        self.temp_dirs: List[Path] = []

        # Set of temporary files to clean up
        self.temp_files: Set[Path] = set()

        # Custom cleanup hooks
        self.cleanup_hooks: List[Callable] = []

        # Flag to track if shutdown is in progress
        self.shutdown_in_progress = False

        # Thread lock for synchronization
        self.lock = threading.RLock()

        # Setup signal handlers
        self._setup_signal_handlers()

        # Register atexit handler for normal program termination
        atexit.register(self.cleanup_resources)

        # Initialize performance monitor
        self.performance = PerformanceMonitor

        self.logger.info("ResourceManager initialized")

    def _setup_signal_handlers(self) -> None:
        """Set up signal handlers for interruption signals."""
        signal.signal(signal.SIGINT, self._signal_handler)  # CTRL+C
        # Termination signal
        signal.signal(signal.SIGTERM, self._signal_handler)

        # Handle Windows specific signals if on Windows
        try:
            # CTRL+BREAK on Windows
            signal.signal(signal.SIGBREAK, self._signal_handler)
        except AttributeError:
            # SIGBREAK not available on non-Windows platforms
            pass

        self.logger.debug("Signal handlers have been set up")

    def _signal_handler(self, sig, frame) -> None:
        """
        Handle interruption signals.

        Args:
            sig: Signal number
            frame: Current stack frame
        """
        signal_names = {
            signal.SIGINT: "SIGINT (CTRL+C)",
            signal.SIGTERM: "SIGTERM"
        }

        # Add Windows-specific signal name if available
        try:
            signal_names[signal.SIGBREAK] = "SIGBREAK (CTRL+BREAK)"
        except AttributeError:
            pass

        signal_name = signal_names.get(sig, f"Signal {sig}")
        self.logger.info(
            f"Received {signal_name}. Initiating graceful shutdown...")

        # Perform cleanup
        self.cleanup_resources()

        # Re-raise the signal with default behavior if needed
        if self.config.get('raise_after_cleanup', True):
            # Reset signal handler to default
            signal.signal(sig, signal.SIG_DFL)
            # Re-raise the signal
            os.kill(os.getpid(), sig)

    def register_temp_dir(self, dir_path: Path) -> None:
        """
        Register a temporary directory to monitor.

        Args:
            dir_path: Path to the temporary directory
        """
        with self.lock:
            if dir_path not in self.temp_dirs:
                self.temp_dirs.append(dir_path)
                self.logger.debug(f"Registered temp directory: {dir_path}")

    def register_temp_file(self, file_path: Path) -> None:
        """
        Register a specific temporary file for cleanup.

        Args:
            file_path: Path to the temporary file
        """
        with self.lock:
            self.temp_files.add(file_path)
            self.logger.debug(f"Registered temp file: {file_path}")

    def register_cleanup_hook(self, hook: Callable) -> None:
        """
        Register a custom cleanup hook function.

        Args:
            hook: Callable function to execute during cleanup
        """
        with self.lock:
            if hook not in self.cleanup_hooks:
                self.cleanup_hooks.append(hook)
                self.logger.debug(f"Registered cleanup hook: {hook.__name__}")

    def cleanup_resources(self) -> None:
        """Clean up all registered resources."""
        with self.lock:
            # Avoid multiple cleanup attempts
            if self.shutdown_in_progress:
                return

            self.shutdown_in_progress = True
            self.logger.info("Initiating resource cleanup...")

            # Execute custom cleanup hooks first
            for hook in self.cleanup_hooks:
                try:
                    hook()
                except Exception as e:
                    self.logger.error(
                        f"Error in cleanup hook {hook.__name__}: {e}")

            # Clean up registered temp files
            for file_path in self.temp_files:
                self._remove_file(file_path)

            # Clean up all files in registered temp directories
            for dir_path in self.temp_dirs:
                self._clean_temp_dir(dir_path)

            self.logger.info("Resource cleanup completed")

    def _clean_temp_dir(self, dir_path: Path) -> None:
        """
        Clean all files in a temporary directory.

        Args:
            dir_path: Path to the temporary directory
        """
        try:
            if not dir_path.exists() or not dir_path.is_dir():
                return

            # Delete all files in directory
            for file_path in dir_path.iterdir():
                if file_path.is_file():
                    self._remove_file(file_path)

            self.logger.debug(f"Cleaned temp directory: {dir_path}")
        except Exception as e:
            self.logger.error(f"Error cleaning temp directory {dir_path}: {e}")

    def _remove_file(self, file_path: Path) -> None:
        """
        Remove a specific file.

        Args:
            file_path: Path to the file to remove
        """
        try:
            if file_path.exists() and file_path.is_file():
                file_path.unlink()
                self.logger.debug(f"Removed temp file: {file_path}")
        except Exception as e:
            self.logger.error(f"Error removing temp file {file_path}: {e}")

    def get_shutdown_flag(self) -> bool:
        """
        Get the current shutdown flag status.

        Returns:
            bool: True if shutdown is in progress
        """
        with self.lock:
            return self.shutdown_in_progress

    @contextmanager
    def monitor_performance(self, operation_name: str, log_level: str = "INFO",
                            auto_report: bool = True, track_resources: bool = True):
        """
        Context manager to monitor performance of a code block.

        Args:
            operation_name: Name of the operation being monitored
            log_level: Logging level for performance reports
            auto_report: Whether to automatically log results when monitoring stops
            track_resources: Whether to track CPU and memory usage

        Returns:
            PerformanceMonitor instance
        """
        monitor = PerformanceMonitor(
            operation_name, log_level, auto_report, track_resources)
        monitor.start()
        try:
            yield monitor
        finally:
            monitor.stop()
