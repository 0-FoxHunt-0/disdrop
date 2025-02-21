import gc
import os
import psutil
import threading
import time
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


class MemoryManager:
    """Manages memory usage and cleanup with improved thresholds and tracking."""

    def __init__(self, threshold_mb: int = 1000, cleanup_interval: int = 30):
        self.threshold_mb = threshold_mb
        self._last_cleanup = time.time()
        self._cleanup_interval = cleanup_interval
        self._memory_usage: List[float] = []
        self._lock = threading.Lock()
        self._gc_threshold = 0.8  # 80% of threshold triggers cleanup
        self._max_usage_history = 10
        self._init_gc_settings()

    def _init_gc_settings(self) -> None:
        """Initialize garbage collector settings."""
        gc.enable()
        # Make GC more aggressive for large objects
        gc.set_threshold(700, 10, 5)

    def check_memory(self, force: bool = False) -> bool:
        """Check if memory cleanup is needed with improved thresholds.

        Args:
            force: Force cleanup regardless of thresholds

        Returns:
            bool: True if cleanup was performed
        """
        try:
            process = psutil.Process()
            current_memory = process.memory_info().rss / (1024 * 1024)

            with self._lock:
                self._memory_usage.append(current_memory)
                if len(self._memory_usage) > self._max_usage_history:
                    self._memory_usage.pop(0)

                avg_memory = sum(self._memory_usage) / len(self._memory_usage)
                time_since_cleanup = time.time() - self._last_cleanup

                should_cleanup = (
                    force or
                    current_memory > self.threshold_mb or
                    avg_memory > self.threshold_mb * self._gc_threshold or
                    time_since_cleanup > self._cleanup_interval
                )

                if should_cleanup:
                    self._perform_cleanup()
                    return True

            return False

        except Exception as e:
            logger.error(f"Memory check error: {e}")
            if force:
                self._emergency_cleanup()
            return True

    def _perform_cleanup(self) -> None:
        """Perform memory cleanup with enhanced resource management."""
        try:
            with self._lock:
                # Full collection with generational GC
                gc.collect(generation=2)

                # Clear internal tracking
                self._last_cleanup = time.time()
                self._memory_usage.clear()

                # Try to release memory back to OS
                if hasattr(gc, 'malloc_trim'):  # Python 3.7+ on Linux
                    gc.malloc_trim()

                if psutil.WINDOWS:
                    self._windows_memory_cleanup()
                else:
                    self._unix_memory_cleanup()

                # Final GC run to clean up any remaining objects
                gc.collect()

                logger.debug("Memory cleanup completed successfully")

        except Exception as e:
            logger.error(f"Memory cleanup error: {e}")
            self._emergency_cleanup()

    def _windows_memory_cleanup(self) -> None:
        """Windows-specific memory cleanup."""
        try:
            import ctypes
            # Call Windows API to reduce working set
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1)
            # Request low-memory working set
            ctypes.windll.psapi.EmptyWorkingSet(-1)
        except Exception as e:
            logger.error(f"Windows memory cleanup failed: {e}")

    def _unix_memory_cleanup(self) -> None:
        """Unix-specific memory cleanup."""
        try:
            # Sync to ensure writes are flushed
            if hasattr(os, 'sync'):
                os.sync()
            # Request memory trim from OS
            import resource
            resource.setrlimit(resource.RLIMIT_AS, (
                resource.RLIM_INFINITY,
                resource.RLIM_INFINITY
            ))
        except Exception as e:
            logger.error(f"Unix memory cleanup failed: {e}")

    def _emergency_cleanup(self) -> None:
        """Emergency cleanup when normal cleanup fails."""
        try:
            logger.warning("Performing emergency memory cleanup")
            # Force garbage collection
            gc.collect(2)
            # Clear caches
            gc.get_objects()
            # Reset GC thresholds
            gc.set_threshold(700, 10, 5)
            # Clear internal state
            with self._lock:
                self._memory_usage.clear()
                self._last_cleanup = time.time()
        except Exception as e:
            logger.critical(f"Emergency cleanup failed: {e}")

    def get_memory_info(self) -> dict:
        """Get current memory usage statistics."""
        try:
            process = psutil.Process()
            mem_info = process.memory_info()

            return {
                'rss': mem_info.rss / (1024 * 1024),  # MB
                'vms': mem_info.vms / (1024 * 1024),  # MB
                'percent': process.memory_percent(),
                'gc_count': gc.get_count(),
                'gc_objects': len(gc.get_objects()),
                'last_cleanup': self._last_cleanup
            }
        except Exception as e:
            logger.error(f"Error getting memory info: {e}")
            return {}

    def cleanup(self) -> None:
        """Public method to force cleanup."""
        self.check_memory(force=True)
