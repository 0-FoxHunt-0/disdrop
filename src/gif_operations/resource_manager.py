import os
import psutil
import time
import threading
from typing import Optional, Dict
import logging
from contextlib import contextmanager
from typing import Generator

logger = logging.getLogger(__name__)


class ResourceMonitor:
    """Monitor system resources during processing."""

    def __init__(self, memory_threshold: float = 85.0, cpu_threshold: float = 90.0):
        self.memory_threshold = memory_threshold
        self.cpu_threshold = cpu_threshold
        self._last_check = time.time()
        self._check_interval = 1.0
        self._lock = threading.Lock()
        self._stats: Dict[str, float] = {'cpu_avg': 0, 'memory_avg': 0}

    def get_safe_thread_count(self) -> int:
        """Get safe number of threads based on system resources."""
        try:
            cpu_count = psutil.cpu_count(logical=False) or 2
            memory_gb = psutil.virtual_memory().available / (1024 * 1024 * 1024)

            # Calculate based on available resources
            cpu_threads = max(1, int(cpu_count * 0.75))  # Use 75% of CPU cores
            # Assume 2GB per thread
            memory_threads = max(1, int(memory_gb / 2))

            # Take the minimum and cap at 8
            return min(cpu_threads, memory_threads, 8)
        except Exception as e:
            logger.error(f"Error calculating thread count: {e}")
            return 2  # Safe default

    def check_resources(self) -> bool:
        """Check if system resources are within acceptable limits."""
        try:
            with self._lock:
                current_time = time.time()
                if current_time - self._last_check < self._check_interval:
                    return True  # Use cached result if checked recently

                memory_percent = psutil.virtual_memory().percent
                cpu_percent = psutil.cpu_percent(interval=0.1)

                # Update moving averages
                self._stats['cpu_avg'] = (self._stats.get(
                    'cpu_avg', 0) * 0.7 + cpu_percent * 0.3)
                self._stats['memory_avg'] = (self._stats.get(
                    'memory_avg', 0) * 0.7 + memory_percent * 0.3)

                if self._stats['memory_avg'] > self.memory_threshold:
                    logger.warning(f"High memory usage: {memory_percent:.1f}%")
                    return False

                if self._stats['cpu_avg'] > self.cpu_threshold:
                    logger.warning(f"High CPU usage: {cpu_percent:.1f}%")
                    return False

                self._last_check = current_time
                return True

        except Exception as e:
            logger.error(f"Resource check failed: {e}")
            return False

    def cleanup(self):
        """Cleanup resources"""
        try:
            import gc
            gc.collect()

            # Try to release memory back to OS
            if hasattr(gc, 'malloc_trim'):  # Linux only
                gc.malloc_trim()
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


class ResourceGuard:
    """Guard against resource exhaustion."""

    def __init__(self, resource_monitor=None):
        self.monitor = resource_monitor if resource_monitor is not None else ResourceMonitor()

    @contextmanager
    def guarded_operation(self) -> Generator[None, None, None]:
        """Context manager for resource-guarded operations."""
        try:
            if not self.monitor.check_resources():
                raise ResourceWarning("Insufficient system resources")
            yield
        finally:
            # Force garbage collection
            import gc
            gc.collect()


def get_optimal_thread_count() -> int:
    """Calculate optimal thread count based on system resources."""
    try:
        cpu_count = psutil.cpu_count(logical=False) or 2
        memory_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)

        # Base thread count on CPU cores and available memory
        thread_count = min(
            cpu_count,
            max(2, int(memory_gb / 2))  # 2GB per thread
        )

        return max(1, min(thread_count, 8))  # Cap at 8 threads

    except Exception:
        return 2  # Safe default
