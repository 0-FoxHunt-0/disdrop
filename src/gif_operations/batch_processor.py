from asyncio import as_completed
from concurrent.futures import ThreadPoolExecutor
import logging
import os
import queue
import threading
from typing import Any, Callable, Generator, List
from src.gif_operations.memory import MemoryManager


class BatchProcessor:
    """Handles batch processing of files with memory management."""

    def __init__(self, max_batch_size: int = 5):
        self.max_batch_size = max_batch_size
        self.memory_manager = MemoryManager()
        self.queue = queue.Queue()
        self._results = []
        self._lock = threading.Lock()

    def process_batch(self, items: List[Any], processor_func: Callable) -> Generator:
        """Process items in batches with memory management."""
        for i in range(0, len(items), self.max_batch_size):
            batch = items[i:i + self.max_batch_size]
            futures = []

            with ThreadPoolExecutor(max_workers=min(len(batch), os.cpu_count() or 2)) as executor:
                for item in batch:
                    futures.append(executor.submit(processor_func, item))

                for future in as_completed(futures):
                    try:
                        result = future.result(timeout=120)
                        yield result
                    except Exception as e:
                        logging.error(f"Batch processing error: {str(e)}")
                        continue

            self.memory_manager.check_memory()
