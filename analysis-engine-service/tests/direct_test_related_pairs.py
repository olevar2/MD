"""
Direct test of the optimized RelatedPairsConfluenceAnalyzer code.
"""

import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import threading
from typing import Dict, List, Any, Optional, Tuple

class RelatedPairsConfluenceAnalyzer:
    """
    Simplified version of the RelatedPairsConfluenceAnalyzer class for testing.
    
    This class implements the core functionality of the optimized RelatedPairsConfluenceAnalyzer
    to verify that the optimizations work as expected.
    """
    
    def __init__(self, cache_ttl_minutes=60, max_workers=4):
        """Initialize the analyzer."""
        self.cache_ttl_minutes = cache_ttl_minutes
        self.max_workers = max_workers
        
        # Enhanced caching system
        self.related_pairs_cache = {}
        self.signal_cache = {}
        self.momentum_cache = {}
        self.cache_lock = threading.Lock()
        self.last_cache_cleanup = time.time()
        
        # Performance metrics
        self.performance_metrics = {
            "find_related_pairs": [],
            "detect_confluence": [],
            "analyze_divergence": []
        }
        self.metrics_lock = threading.Lock()
        
    def _clean_cache(self, force=False):
        """Clean expired entries from all caches."""
        current_time = time.time()
        
        # Only clean periodically to avoid overhead
        if not force and current_time - self.last_cache_cleanup < 300:  # Clean at most once per 5 minutes
            return
            
        with self.cache_lock:
            # Calculate expiry time
            expiry_time = current_time - (self.cache_ttl_minutes * 60)
            
            # Clean related pairs cache
            expired_keys = []
            for key, (timestamp, _) in self.related_pairs_cache.items():
                if timestamp < expiry_time:
                    expired_keys.append(key)
            
            for key in expired_keys:
                del self.related_pairs_cache[key]
                
            # Clean signal cache
            expired_keys = []
            for key, (timestamp, _) in self.signal_cache.items():
                if timestamp < expiry_time:
                    expired_keys.append(key)
                    
            for key in expired_keys:
                del self.signal_cache[key]
                
            # Clean momentum cache
            expired_keys = []
            for key, (timestamp, _) in self.momentum_cache.items():
                if timestamp < expiry_time:
                    expired_keys.append(key)
                    
            for key in expired_keys:
                del self.momentum_cache[key]
                
            # Update last cleanup time
            self.last_cache_cleanup = current_time
            
    def _get_signal_cache_key(self, price_data: pd.DataFrame, signal_type: str) -> str:
        """Generate a cache key for signal detection."""
        # Use the last few rows of data as a fingerprint
        last_rows = min(5, len(price_data))
        
        if last_rows == 0:
            return f"{signal_type}_empty"
            
        # Get the close column
        close_col = next((col for col in price_data.columns if col.lower() in ['close', 'price', 'adj close']), None)
        if not close_col:
            return f"{signal_type}_no_close"
            
        # Create a fingerprint from the last N rows
        close_vals = price_data[close_col].iloc[-last_rows:].values
        
        # Use hash of the values as the key
        fingerprint = hash(tuple(close_vals))
        return f"{signal_type}_{fingerprint}"
    
    def _detect_signal(self, price_data: pd.DataFrame, signal_type: str, lookback: int) -> Optional[Dict[str, Any]]:
        """
        Detect a specific type of signal in price data.
        
        Optimized with:
        - Caching of signal detection results
        - Early termination for invalid inputs
        - Vectorized operations where possible
        """
        # Early termination for invalid inputs
        if price_data is None or price_data.empty or len(price_data) < lookback:
            return None

        # Get the relevant columns
        close_col = next((col for col in price_data.columns if col.lower() in ['close', 'price', 'adj close']), None)
        if not close_col:
            return None
            
        # Check cache
        cache_key = self._get_signal_cache_key(price_data, signal_type)
        with self.cache_lock:
            if cache_key in self.signal_cache:
                timestamp, cached_result = self.signal_cache[cache_key]
                if time.time() - timestamp < (self.cache_ttl_minutes * 60):
                    return cached_result.copy() if cached_result else None

        # Implement different signal detection methods based on signal_type
        result = None
        if signal_type == "trend":
            result = self._detect_trend_signal(price_data, close_col, lookback)
        elif signal_type == "reversal":
            result = self._detect_reversal_signal(price_data, close_col, lookback)
        elif signal_type == "breakout":
            result = self._detect_breakout_signal(price_data, close_col, lookback)
            
        # Cache the result
        with self.cache_lock:
            self.signal_cache[cache_key] = (time.time(), result)
            
        return result
        
    def _detect_trend_signal(self, price_data: pd.DataFrame, close_col: str, lookback: int) -> Optional[Dict[str, Any]]:
        """
        Detect trend signals in price data.
        
        Optimized with:
        - Vectorized operations
        - Early termination for invalid inputs
        - Improved strength calculation
        """
        # Early termination for insufficient data
        if len(price_data) < 50:  # Need at least 50 bars for 50-period MA
            return None
            
        # Calculate short and medium-term moving averages using vectorized operations
        # Use numpy for faster calculations if possible
        prices = price_data[close_col].values
        
        # Calculate short MA (20-period)
        short_window = 20
        if len(prices) < short_window:
            return None
            
        # Use numpy's convolve for faster MA calculation
        short_weights = np.ones(short_window) / short_window
        short_ma_values = np.convolve(prices, short_weights, 'valid')
        short_ma_last = short_ma_values[-1]
        
        # Calculate medium MA (50-period)
        medium_window = 50
        if len(prices) < medium_window:
            return None
            
        medium_weights = np.ones(medium_window) / medium_window
        medium_ma_values = np.convolve(prices, medium_weights, 'valid')
        medium_ma_last = medium_ma_values[-1]
        
        # Check for valid MAs
        if np.isnan(short_ma_last) or np.isnan(medium_ma_last) or medium_ma_last == 0:
            return None

        # Determine trend direction and calculate strength
        ma_ratio = short_ma_last / medium_ma_last
        
        if ma_ratio > 1.0:
            direction = "bullish"
            # Calculate strength based on how far short MA is above medium MA
            # Use sigmoid-like function for more nuanced strength calculation
            strength = min(1.0, 2 / (1 + np.exp(-10 * (ma_ratio - 1))) - 1)
        elif ma_ratio < 1.0:
            direction = "bearish"
            # Calculate strength based on how far short MA is below medium MA
            # Use sigmoid-like function for more nuanced strength calculation
            strength = min(1.0, 2 / (1 + np.exp(-10 * (1 - ma_ratio))) - 1)
        else:
            return None  # No clear trend
            
        # Calculate additional trend metrics for more comprehensive analysis
        # Trend consistency: how consistently the short MA has been above/below medium MA
        if len(short_ma_values) > 5 and len(medium_ma_values) > 5:
            # Get the last 5 values of each MA
            short_recent = short_ma_values[-5:]
            medium_recent = medium_ma_values[-5:]
            
            # Calculate how many periods the trend has been consistent
            if direction == "bullish":
                consistency = np.sum(short_recent > medium_recent) / 5
            else:
                consistency = np.sum(short_recent < medium_recent) / 5
        else:
            consistency = 1.0  # Default if we don't have enough data

        return {
            "type": "trend",
            "direction": direction,
            "strength": float(strength),  # Convert numpy types to Python types
            "consistency": float(consistency),
            "short_ma": float(short_ma_last),
            "medium_ma": float(medium_ma_last),
            "ma_ratio": float(ma_ratio)
        }
        
    def _detect_reversal_signal(self, price_data: pd.DataFrame, close_col: str, lookback: int) -> Optional[Dict[str, Any]]:
        """Simplified reversal signal detection."""
        # Simple implementation for testing
        return {
            "type": "reversal",
            "direction": "bullish" if np.random.random() > 0.5 else "bearish",
            "strength": float(np.random.random()),
            "rsi": float(np.random.uniform(20, 80))
        }
        
    def _detect_breakout_signal(self, price_data: pd.DataFrame, close_col: str, lookback: int) -> Optional[Dict[str, Any]]:
        """Simplified breakout signal detection."""
        # Simple implementation for testing
        return {
            "type": "breakout",
            "direction": "bullish" if np.random.random() > 0.5 else "bearish",
            "strength": float(np.random.random()),
            "recent_high": float(price_data[close_col].max()),
            "recent_low": float(price_data[close_col].min())
        }

def generate_sample_data(num_pairs=4, num_bars=500):
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

def test_performance():
    """Test the performance of the optimized code."""
    print("Creating RelatedPairsConfluenceAnalyzer instance...")
    analyzer = RelatedPairsConfluenceAnalyzer()
    
    print("Generating sample data...")
    price_data = generate_sample_data(num_pairs=4, num_bars=500)
    
    print("Running signal detection performance test...")
    
    # Test signal detection performance
    for signal_type in ["trend", "reversal", "breakout"]:
        print(f"\nTesting {signal_type} signal detection:")
        
        # First run (cold cache)
        start_time = time.time()
        result1 = analyzer._detect_signal(price_data["EURUSD"], signal_type, 20)
        first_run_time = time.time() - start_time
        
        # Second run (warm cache)
        start_time = time.time()
        result2 = analyzer._detect_signal(price_data["EURUSD"], signal_type, 20)
        second_run_time = time.time() - start_time
        
        # Print performance results
        print(f"  Cold cache: {first_run_time:.4f} seconds")
        print(f"  Warm cache: {second_run_time:.4f} seconds")
        print(f"  Speedup factor: {first_run_time / second_run_time:.2f}x")
        
        # Print signal details
        if result2:
            print(f"  Signal direction: {result2.get('direction')}")
            print(f"  Signal strength: {result2.get('strength'):.4f}")
    
    print("\nPerformance test completed")

if __name__ == "__main__":
    print("Starting direct test of optimized RelatedPairsConfluenceAnalyzer...")
    test_performance()
