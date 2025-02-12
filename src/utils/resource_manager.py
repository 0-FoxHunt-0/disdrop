import os
import psutil
import time
import threading
from typing import Optional
import logging


class ResourceMonitor:
    """Monitor and manage system resources."""

    def __init__(self, cpu_limit: float = 80.0, memory_limit: float = 80.0):
        self.cpu_limit = cpu_limit
        self.memory_limit = memory_limit
        self._last_check = 0
        self._check_interval = 1  # seconds
        self._lock = threading.Lock()
        self.logger = logging.getLogger('developer')

    def get_resource_usage(self) -> tuple[float, float]:
        """Get current CPU and memory usage percentages."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            return cpu_percent, memory_percent
        except Exception as e:
            self.logger.error(f"Error getting resource usage: {e}")
            return 0.0, 0.0

    def should_throttle(self) -> bool:
        """Check if processing should be throttled."""
        with self._lock:
            current_time = time.time()
            if current_time - self._last_check < self._check_interval:
                return False

            self._last_check = current_time
            cpu_percent, memory_percent = self.get_resource_usage()

            if cpu_percent > self.cpu_limit or memory_percent > self.memory_limit:
                self.logger.warning(
                    f"Resource limits exceeded - CPU: {cpu_percent:.1f}%, Memory: {memory_percent:.1f}%")
                return True
            return False

    def get_safe_thread_count(self) -> int:
        """Calculate safe number of threads based on system resources."""
        cpu_count = os.cpu_count() or 1
        memory_available = psutil.virtual_memory().available / (1024 * 1024)  # MB

        # Limit based on CPU cores (max 80%)
        cpu_threads = max(1, int(cpu_count * 0.8))

        # Limit based on available memory (assume 500MB per thread)
        memory_threads = max(1, int((memory_available * 0.8) / 500))

        # Calculate optimal thread count
        thread_count = min(cpu_threads, memory_threads, 4)  # Max 4 threads

        self.logger.debug(
            f"Safe thread count: {thread_count} (CPU: {cpu_threads}, Memory: {memory_threads})")
        return thread_count

    def wait_for_resources(self, timeout: Optional[float] = None) -> bool:
        """Wait until resource usage is below limits or timeout occurs."""
        start_time = time.time()
        while self.should_throttle():
            if timeout and time.time() - start_time > timeout:
                self.logger.warning("Resource wait timeout exceeded")
                return False
            time.sleep(1)
        return True

    def check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        try:
            memory = psutil.virtual_memory()
            if memory.percent > self.memory_limit:
                self.logger.warning(
                    f"High memory usage: {memory.percent:.1f}%")
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error checking memory pressure: {e}")
            return True

    def get_available_memory_mb(self) -> float:
        """Get available memory in MB."""
        try:
            return psutil.virtual_memory().available / (1024 * 1024)
        except Exception:
            return 1000  # Safe default

    def cleanup(self) -> None:
        """Release any acquired resources."""
        import gc
        gc.collect()


class ResourceGuard:
    """Context manager for resource-aware operations."""

    def __init__(self, monitor: ResourceMonitor):
        self.monitor = monitor

    def __enter__(self):
        return self.monitor.wait_for_resources()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type is not None:
            self.monitor.cleanup()
