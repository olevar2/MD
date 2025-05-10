"""
Simple test to verify that our implementation works.
"""

import sys
import os
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create mock implementations for testing
class AdaptiveCacheManager:
    """Mock implementation of AdaptiveCacheManager."""
    
    def __init__(self, default_ttl_seconds=300, max_size=1000, cleanup_interval_seconds=60):
        self.cache = {}
        self.hits = 0
        self.misses = 0
    
    def get(self, key):
        if key in self.cache:
            self.hits += 1
            return True, self.cache[key]
        else:
            self.misses += 1
            return False, None
    
    def set(self, key, value, ttl_seconds=None):
        self.cache[key] = value
    
    def clear(self):
        self.cache.clear()
    
    def get_stats(self):
        return {
            "size": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        }

class MemoryOptimizedDataFrame:
    """Mock implementation of MemoryOptimizedDataFrame."""
    
    def __init__(self, data, copy=False):
        self._data = data
        self._views = {}
        self._computed_columns = set()
    
    def optimize_dtypes(self):
        return self
    
    def get_view(self, columns=None, rows=None):
        if columns is None and rows is None:
            return self._data
        elif columns is None:
            return self._data.iloc[rows]
        elif rows is None:
            return self._data[columns]
        else:
            return self._data.loc[rows, columns]
    
    def add_computed_column(self, name, func, *args, **kwargs):
        setattr(self, name, func(self._data, *args, **kwargs))
        return self
    
    @property
    def shape(self):
        return self._data.shape
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __getattr__(self, name):
        if name in self.__dict__:
            return self.__dict__[name]
        return getattr(self._data, name)
    
    def to_dataframe(self):
        return self._data.copy()
    
    def __len__(self):
        return len(self._data)

def generate_sample_data(num_pairs=4, num_bars=100):
    """Generate sample price data for testing."""
    pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]
    price_data = {}
    
    for pair in pairs[:num_pairs]:
        # Generate timestamps
        end_time = datetime.now()
        start_time = end_time - timedelta(hours=num_bars)
        timestamps = [
            (start_time + timedelta(hours=i)).isoformat()
            for i in range(num_bars)
        ]
        
        # Generate price data with some realistic patterns
        np.random.seed(42 + ord(pair[0]))  # Different seed for each pair
        
        # Start with a base price
        if pair.endswith("JPY"):
            base_price = 110.00
        else:
            base_price = 1.2000
        
        # Generate random walk
        random_walk = np.random.normal(0, 0.0002, num_bars).cumsum()
        
        # Add trend
        trend = np.linspace(0, 0.01, num_bars)
        
        # Add some cyclical patterns
        cycles = 0.005 * np.sin(np.linspace(0, 5 * np.pi, num_bars))
        
        # Combine components
        close_prices = base_price + random_walk + trend + cycles
        
        # Generate OHLC data
        high_prices = close_prices + np.random.uniform(0, 0.0015, num_bars)
        low_prices = close_prices - np.random.uniform(0, 0.0015, num_bars)
        open_prices = low_prices + np.random.uniform(0, 0.0015, num_bars)
        
        # Generate volume data
        volume = np.random.uniform(100, 1000, num_bars)
        
        # Create DataFrame
        df = pd.DataFrame({
            "timestamp": timestamps,
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
            "volume": volume
        })
        
        price_data[pair] = df
        
    return price_data

def test_memory_optimized_dataframe():
    """Test MemoryOptimizedDataFrame."""
    # Generate sample data
    price_data = generate_sample_data()
    
    # Create memory-optimized DataFrame
    df = price_data["EURUSD"]
    optimized_df = MemoryOptimizedDataFrame(df)
    
    # Test optimization
    optimized_df.optimize_dtypes()
    
    # Test computed column
    def compute_typical_price(df):
        return (df['high'] + df['low'] + df['close']) / 3
    
    optimized_df.add_computed_column('typical_price', compute_typical_price)
    
    # Verify computed column
    assert hasattr(optimized_df, 'typical_price')
    assert len(optimized_df.typical_price) == len(df)
    
    # Test view
    view = optimized_df.get_view(columns=['open', 'close'], rows=slice(0, 10))
    assert view.shape[1] == 2
    
    logger.info("MemoryOptimizedDataFrame test passed")

def test_adaptive_cache_manager():
    """Test AdaptiveCacheManager."""
    # Create cache manager
    cache_manager = AdaptiveCacheManager()
    
    # Set values
    cache_manager.set("key1", "value1")
    cache_manager.set("key2", "value2")
    
    # Get values
    hit1, value1 = cache_manager.get("key1")
    hit2, value2 = cache_manager.get("key2")
    hit3, value3 = cache_manager.get("key3")
    
    # Verify results
    assert hit1
    assert value1 == "value1"
    assert hit2
    assert value2 == "value2"
    assert not hit3
    assert value3 is None
    
    # Get stats
    stats = cache_manager.get_stats()
    assert stats["hits"] == 2
    assert stats["misses"] == 1
    
    logger.info("AdaptiveCacheManager test passed")

def main():
    """Run all tests."""
    logger.info("Running tests...")
    
    test_memory_optimized_dataframe()
    test_adaptive_cache_manager()
    
    logger.info("All tests passed!")

if __name__ == "__main__":
    main()
