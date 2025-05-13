"""
Cache metrics implementation for tracking cache performance.
"""
import time
from typing import Dict, Any, Optional
from datetime import datetime


class CacheMetrics:
    """
    Tracks cache performance metrics including:
    - Hit/miss ratios
    - Cache size
    - Average operation times
    - Cache utilization
    """
    def __init__(self):
    """
      init  .
    
    """

        self.hits = 0
        self.misses = 0
        self.puts = 0
        self.invalidations = 0
        self.hit_times = []  # Lists for calculating average times
        self.miss_times = []
        self.put_times = []
        self.hit_bytes = 0  # Track size of hit data
        self.put_bytes = 0  # Track size of put data
        
        self.start_time = datetime.now()
        # Track hits by cache tier
        self.tier_hits = {
            "memory": 0,
            "disk": 0,
            "db": 0
        }
        
    def record_hit(self, tier: str, size_bytes: Optional[int] = None, operation_time: Optional[float] = None):
        """Record a cache hit with the tier that provided the data."""
        self.hits += 1
        
        if tier in self.tier_hits:
            self.tier_hits[tier] += 1
            
        if operation_time is not None:
            self.hit_times.append(operation_time)
            
        if size_bytes is not None:
            self.hit_bytes += size_bytes
            
    def record_miss(self, operation_time: Optional[float] = None):
        """Record a cache miss."""
        self.misses += 1
        
        if operation_time is not None:
            self.miss_times.append(operation_time)
            
    def record_put(self, size_bytes: int, operation_time: Optional[float] = None):
        """Record a cache put operation."""
        self.puts += 1
        self.put_bytes += size_bytes
        
        if operation_time is not None:
            self.put_times.append(operation_time)
            
    def record_invalidation(self, count: int = 1):
        """Record cache invalidation operations."""
        self.invalidations += count
        
    @property
    def total_operations(self) -> int:
        """Get the total number of cache operations."""
        return self.hits + self.misses + self.puts
        
    @property
    def hit_ratio(self) -> float:
        """Calculate the cache hit ratio."""
        if self.hits + self.misses == 0:
            return 0.0
        return self.hits / (self.hits + self.misses)
        
    @property
    def miss_ratio(self) -> float:
        """Calculate the cache miss ratio."""
        if self.hits + self.misses == 0:
            return 0.0
        return self.misses / (self.hits + self.misses)
        
    @property
    def avg_hit_time(self) -> float:
        """Calculate the average hit operation time."""
        if not self.hit_times:
            return 0.0
        return sum(self.hit_times) / len(self.hit_times)
        
    @property
    def avg_miss_time(self) -> float:
        """Calculate the average miss operation time."""
        if not self.miss_times:
            return 0.0
        return sum(self.miss_times) / len(self.miss_times)
        
    @property
    def avg_put_time(self) -> float:
        """Calculate the average put operation time."""
        if not self.put_times:
            return 0.0
        return sum(self.put_times) / len(self.put_times)
        
    @property
    def tier_hit_distribution(self) -> Dict[str, float]:
        """Calculate the distribution of hits across cache tiers."""
        if self.hits == 0:
            return {tier: 0.0 for tier in self.tier_hits}
            
        return {tier: hits / self.hits for tier, hits in self.tier_hits.items()}
        
    def get_metrics_dict(self) -> Dict[str, Any]:
        """Get all metrics as a dictionary."""
        uptime_seconds = (datetime.now() - self.start_time).total_seconds()
        
        return {
            "operations": {
                "hits": self.hits,
                "misses": self.misses,
                "puts": self.puts,
                "invalidations": self.invalidations,
                "total": self.total_operations
            },
            "ratios": {
                "hit_ratio": self.hit_ratio,
                "miss_ratio": self.miss_ratio,
                "tier_distribution": self.tier_hit_distribution
            },
            "performance": {
                "avg_hit_time_ms": self.avg_hit_time * 1000 if self.avg_hit_time else 0,
                "avg_miss_time_ms": self.avg_miss_time * 1000 if self.avg_miss_time else 0,
                "avg_put_time_ms": self.avg_put_time * 1000 if self.avg_put_time else 0
            },
            "data": {
                "hit_bytes": self.hit_bytes,
                "put_bytes": self.put_bytes,
                "hit_bytes_per_second": self.hit_bytes / uptime_seconds if uptime_seconds > 0 else 0,
                "put_bytes_per_second": self.put_bytes / uptime_seconds if uptime_seconds > 0 else 0
            },
            "uptime_seconds": uptime_seconds
        }
        
    def reset(self):
        """Reset all metrics."""
        self.__init__()
