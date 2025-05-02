\
"""
Implements adaptive caching strategies based on usage patterns.
"""

class AdaptiveCache:
    def __init__(self, max_size=1000):
        self.cache = {}
        self.usage_frequency = {}
        self.max_size = max_size
        print(f"Initialized AdaptiveCache with max size: {max_size}")

    def get(self, key):
        """Retrieves an item from the cache and updates usage frequency."""
        if key in self.cache:
            self.usage_frequency[key] = self.usage_frequency.get(key, 0) + 1
            print(f"Cache hit for key: {key}. New frequency: {self.usage_frequency[key]}")
            return self.cache[key]
        else:
            print(f"Cache miss for key: {key}")
            return None

    def put(self, key, value):
        """Adds an item to the cache, potentially evicting the least used item."""
        if len(self.cache) >= self.max_size:
            self._evict()

        self.cache[key] = value
        self.usage_frequency[key] = self.usage_frequency.get(key, 0) + 1
        print(f"Added/Updated key: {key} in cache. Frequency: {self.usage_frequency[key]}")


    def _evict(self):
        """Evicts the least frequently used item from the cache."""
        if not self.cache:
            return

        # Find the least frequently used item
        least_used_key = min(self.usage_frequency, key=self.usage_frequency.get)

        print(f"Evicting least used key: {least_used_key} with frequency: {self.usage_frequency[least_used_key]}")
        del self.cache[least_used_key]
        del self.usage_frequency[least_used_key]    def update_strategy(self, usage_data):
        """
        Adapts the caching strategy based on historical usage patterns.
        
        Args:
            usage_data: Dictionary with statistics about cache usage patterns
                        {'access_counts': {key: count}, 'hit_ratio': float,
                         'time_to_live': {key: seconds}, 'size_impact': float}
        """
        print(f"Updating caching strategy based on usage data")
        
        # Strategy 1: Adjust cache size based on hit ratio
        if 'hit_ratio' in usage_data:
            hit_ratio = usage_data['hit_ratio']
            if hit_ratio < 0.5:  # Low hit ratio suggests cache is too small
                new_size = min(int(self.max_size * 1.5), 10000)  # Increase by 50%, cap at 10000
                print(f"Low hit ratio ({hit_ratio:.2f}), increasing cache size from {self.max_size} to {new_size}")
                self.max_size = new_size
            elif hit_ratio > 0.9 and len(self.cache) < self.max_size * 0.7:
                # High hit ratio with unused capacity suggests we can reduce size
                new_size = max(int(self.max_size * 0.8), 100)  # Decrease by 20%, minimum 100
                print(f"High hit ratio ({hit_ratio:.2f}) with low usage, reducing cache size from {self.max_size} to {new_size}")
                self.max_size = new_size
                
        # Strategy 2: Switch eviction policy if needed
        if 'access_pattern' in usage_data:
            pattern = usage_data['access_pattern']
            if pattern == 'temporal_locality':
                # If data shows strong temporal locality, switch to LRU
                print("Detected strong temporal locality, switching to LRU eviction policy")
                self.eviction_policy = 'LRU'
                self.access_history = []  # Initialize LRU tracking
            elif pattern == 'frequency_based':
                # If data shows frequency-based access patterns, use LFU (default)
                print("Detected frequency-based access pattern, using LFU eviction policy")
                self.eviction_policy = 'LFU'
                
        # Strategy 3: Preload frequently accessed items
        if 'preload_candidates' in usage_data:
            preload_keys = usage_data['preload_candidates']
            print(f"Identified {len(preload_keys)} items for preloading")
            # Implementation would depend on how we fetch items to preload
            
        return {
            "max_size": self.max_size,
            "eviction_policy": getattr(self, 'eviction_policy', 'LFU')
        }

# Example Usage (can be removed or expanded)
if __name__ == '__main__':
    cache = AdaptiveCache(max_size=3)
    cache.put('a', 1)
    cache.put('b', 2)
    cache.put('c', 3)
    print(cache.get('a'))
    print(cache.get('b'))
    print(cache.get('a')) # Increase frequency of 'a'
    cache.put('d', 4) # Should evict 'c' (least frequent)
    print(cache.get('c')) # Should be a miss
    print(cache.cache)
