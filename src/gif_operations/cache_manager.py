from contextlib import contextmanager
import threading
from cachetools import TTLCache

from src.gif_optimization import CACHE_SIZE


class CacheManager:
    """Manages caching for optimization operations"""

    def __init__(self):
        self.frame_cache = TTLCache(maxsize=CACHE_SIZE, ttl=300)
        self.palette_cache = TTLCache(maxsize=CACHE_SIZE, ttl=600)
        self.dimension_cache = TTLCache(maxsize=CACHE_SIZE, ttl=900)
        self._lock = threading.Lock()

    @contextmanager
    def cached_operation(self, cache_key: str, cache_type: str = 'frame'):
        cache = getattr(self, f'{cache_type}_cache')
        with self._lock:
            if cache_key in cache:
                yield cache[cache_key]
                return
        result = yield None
        with self._lock:
            cache[cache_key] = result
