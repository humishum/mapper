"""Simple standalone LRU cache"""

# TODO: Add eviction hooks
from collections import OrderedDict
from typing import Any, Optional
import logging

logger = logging.getLogger(__name__)


class LRUCache:
    """
    Simple LRU Cache

    When the cache is full, the least recently used item is removed.
    """

    def __init__(self, max_size: int = 5):
        self.max_size = max_size
        self.cache = OrderedDict()
        logger.info(f"Initialized LRU cache with max_size={max_size}")

    def get(self, key: str) -> Optional[Any]:
        """get item from cache, if it exists, otherwise return None"""
        if key not in self.cache:
            return None

        # Move to end to mark as recently used
        self.cache.move_to_end(key)
        logger.debug(f"Cache hit: {key}")
        return self.cache[key]

    def put(self, key: str, value: Any) -> None:
        """
        Put an item in the cache.

        Args:
            key: Cache key
            value: Value to cache
        """
        if key in self.cache:
            # Update existing item and mark as recently used
            self.cache.move_to_end(key)
            self.cache[key] = value
            logger.debug(f"Cache update: {key}")
        else:
            # Add new item
            self.cache[key] = value
            logger.info(f"Cache put: {key}")

            # Remove oldest item if cache is full
            if len(self.cache) > self.max_size:
                oldest_key = next(iter(self.cache))
                del self.cache[oldest_key]
                logger.info(f"Cache evicted (full): {oldest_key}")

    def clear(self) -> None:
        """Clear all items from the cache."""
        self.cache.clear()
        logger.info("Cache cleared")

    def size(self) -> int:
        """Get the current number of items in the cache."""
        return len(self.cache)

    def __contains__(self, key: str) -> bool:
        """Check if a key exists in the cache."""
        return key in self.cache
