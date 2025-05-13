"""
Multi-level caching system implementation for Phase 1.
Implements L1 (Memory), L2 (Disk), and L3 (Database) caching with intelligent
data movement between layers.
"""

import os
import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
import psutil
import logging
from .adaptive_cache import AdaptiveCache

logger = logging.getLogger(__name__)

class CacheLevel:
    L1_MEMORY = "l1_memory"
    L2_DISK = "l2_disk"
    L3_DATABASE = "l3_database"

class MultiLevelCache:
    """
    Coordinates caching across three levels:
    - L1: Memory-based LRU cache (fastest, limited size)
    - L2: Disk-based cache with compression (medium speed, larger size)
    - L3: Database cache for query results (slower, largest size)
    """

    def __init__(
        self,
        cache_dir: str = "cache",
        l1_max_size_mb: int = 512,
        l2_max_size_gb: int = 2,
        l3_ttl_hours: int = 24,
        enable_compression: bool = True
    ):
    """
      init  .
    
    Args:
        cache_dir: Description of cache_dir
        l1_max_size_mb: Description of l1_max_size_mb
        l2_max_size_gb: Description of l2_max_size_gb
        l3_ttl_hours: Description of l3_ttl_hours
        enable_compression: Description of enable_compression
    
    """

        self.cache_dir = Path(cache_dir)
        self.l2_cache_dir = self.cache_dir / "l2_disk"
        self.cache_dir.mkdir(exist_ok=True)
        self.l2_cache_dir.mkdir(exist_ok=True)

        # Initialize L1 Memory Cache
        self.l1_cache = AdaptiveCache(
            max_size=l1_max_size_mb * 1024 * 1024,  # Convert MB to bytes
            default_ttl=3600,  # 1 hour default TTL for L1
            eviction_policy='lru',
            adaptive_ttl=True
        )

        self.l2_max_size = l2_max_size_gb * 1024 * 1024 * 1024  # Convert GB to bytes
        self.l3_ttl = timedelta(hours=l3_ttl_hours)
        self.enable_compression = enable_compression

        self._lock = threading.RLock()
        self._metrics: Dict[str, Dict[str, int]] = {
            CacheLevel.L1_MEMORY: {"hits": 0, "misses": 0},
            CacheLevel.L2_DISK: {"hits": 0, "misses": 0},
            CacheLevel.L3_DATABASE: {"hits": 0, "misses": 0}
        }

        # Start monitoring thread
        self._start_monitoring()

    def _start_monitoring(self):
        """Start background monitoring of cache metrics and health."""
        def monitor():
    """
    Monitor.
    
    """

            while True:
                try:
                    self._check_cache_health()
                    time.sleep(60)  # Check every minute
                except Exception as e:
                    logger.error(f"Cache monitoring error: {e}")

        threading.Thread(target=monitor, daemon=True).start()

    def _check_cache_health(self):
        """Monitor cache health and perform maintenance."""
        # Check L1 memory usage
        l1_size = self.l1_cache._get_size_bytes()
        if l1_size > self.l1_cache.max_size * 0.9:  # 90% threshold
            logger.warning(f"L1 cache near capacity: {l1_size/1024/1024:.2f}MB")
            self._promote_to_l2()

        # Check L2 disk usage
        l2_size = sum(f.stat().st_size for f in self.l2_cache_dir.glob('**/*') if f.is_file())
        if l2_size > self.l2_max_size * 0.9:  # 90% threshold
            logger.warning(f"L2 cache near capacity: {l2_size/1024/1024/1024:.2f}GB")
            self._evict_l2_entries()

    def get(self, key: str, level: Optional[str] = None) -> Optional[Any]:
        """
        Retrieve item from cache, checking levels in order (L1 -> L2 -> L3).
        
        Args:
            key: Cache key
            level: Optional specific cache level to check
            
        Returns:
            Cached value if found, None otherwise
        """
        with self._lock:
            # Check L1 Memory Cache
            if level is None or level == CacheLevel.L1_MEMORY:
                value = self.l1_cache.get(key)
                if value is not None:
                    self._metrics[CacheLevel.L1_MEMORY]["hits"] += 1
                    return value
                self._metrics[CacheLevel.L1_MEMORY]["misses"] += 1

            # Check L2 Disk Cache
            if level is None or level == CacheLevel.L2_DISK:
                value = self._get_from_l2(key)
                if value is not None:
                    self._metrics[CacheLevel.L2_DISK]["hits"] += 1
                    # Promote to L1 if not specifically requesting L2
                    if level is None:
                        self.l1_cache.put(key, value)
                    return value
                self._metrics[CacheLevel.L2_DISK]["misses"] += 1

            # Check L3 Database Cache
            if level is None or level == CacheLevel.L3_DATABASE:
                value = self._get_from_l3(key)
                if value is not None:
                    self._metrics[CacheLevel.L3_DATABASE]["hits"] += 1
                    # Promote to L2 if not specifically requesting L3
                    if level is None:
                        self._store_in_l2(key, value)
                    return value
                self._metrics[CacheLevel.L3_DATABASE]["misses"] += 1

        return None

    def put(self, key: str, value: Any, level: str = CacheLevel.L1_MEMORY) -> None:
        """
        Store item in specified cache level.
        
        Args:
            key: Cache key
            value: Value to cache
            level: Cache level to store in
        """
        with self._lock:
            if level == CacheLevel.L1_MEMORY:
                self.l1_cache.put(key, value)
            elif level == CacheLevel.L2_DISK:
                self._store_in_l2(key, value)
            elif level == CacheLevel.L3_DATABASE:
                self._store_in_l3(key, value)
            else:
                raise ValueError(f"Invalid cache level: {level}")

    def _promote_to_l2(self) -> None:
        """Move least recently used items from L1 to L2."""
        items_to_move = self.l1_cache.get_lru_items(count=10)  # Move 10 items at a time
        for key, value in items_to_move:
            self._store_in_l2(key, value)
            self.l1_cache.remove(key)

    def _store_in_l2(self, key: str, value: Any) -> None:
        """Store item in L2 disk cache."""
        cache_file = self.l2_cache_dir / f"{key}.json"
        try:
            data = {
                'value': value,
                'timestamp': datetime.utcnow().isoformat()
            }
            with open(cache_file, 'w') as f:
                json.dump(data, f)
        except Exception as e:
            logger.error(f"Error writing to L2 cache: {e}")

    def _get_from_l2(self, key: str) -> Optional[Any]:
        """Retrieve item from L2 disk cache."""
        cache_file = self.l2_cache_dir / f"{key}.json"
        if not cache_file.exists():
            return None

        try:
            with open(cache_file, 'r') as f:
                data = json.load(f)
                timestamp = datetime.fromisoformat(data['timestamp'])
                if datetime.utcnow() - timestamp <= self.l3_ttl:
                    return data['value']
                # Remove expired file
                cache_file.unlink()
        except Exception as e:
            logger.error(f"Error reading from L2 cache: {e}")
        
        return None

    def _store_in_l3(self, key: str, value: Any) -> None:
        """Store item in L3 database cache."""
        # This should be implemented based on your specific database backend
        # For now, we'll just log that it would be stored
        logger.info(f"Would store in L3: {key}")
        pass

    def _get_from_l3(self, key: str) -> Optional[Any]:
        """Retrieve item from L3 database cache."""
        # This should be implemented based on your specific database backend
        return None

    def _evict_l2_entries(self) -> None:
        """Remove oldest entries from L2 cache until under size limit."""
        files = list(self.l2_cache_dir.glob('**/*'))
        files.sort(key=lambda x: x.stat().st_mtime)
        
        current_size = sum(f.stat().st_size for f in files)
        target_size = self.l2_max_size * 0.7  # Reduce to 70% of max

        for file in files:
            if current_size <= target_size:
                break
            try:
                size = file.stat().st_size
                file.unlink()
                current_size -= size
            except Exception as e:
                logger.error(f"Error removing L2 cache file: {e}")

    def get_metrics(self) -> Dict[str, Dict[str, Union[int, float]]]:
        """Get cache performance metrics."""
        metrics = {}
        for level, stats in self._metrics.items():
            total = stats["hits"] + stats["misses"]
            hit_rate = (stats["hits"] / total * 100) if total > 0 else 0
            metrics[level] = {
                "hits": stats["hits"],
                "misses": stats["misses"],
                "hit_rate": hit_rate
            }
        return metrics

    def clear(self, level: Optional[str] = None) -> None:
        """Clear specified cache level or all levels if none specified."""
        with self._lock:
            if level in (None, CacheLevel.L1_MEMORY):
                self.l1_cache.clear()
            
            if level in (None, CacheLevel.L2_DISK):
                for file in self.l2_cache_dir.glob('**/*'):
                    try:
                        file.unlink()
                    except Exception as e:
                        logger.error(f"Error clearing L2 cache: {e}")

            if level in (None, CacheLevel.L3_DATABASE):
                # Implement L3 cache clearing based on your database backend
                pass
