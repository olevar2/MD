"""
Tests for the optimized RelatedPairsConfluenceAnalyzer performance.

This module contains tests to verify the performance improvements in the
RelatedPairsConfluenceAnalyzer class.
"""

import unittest
import asyncio
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from analysis_engine.multi_asset.related_pairs_confluence_detector import RelatedPairsConfluenceAnalyzer


class TestRelatedPairsConfluenceAnalyzerPerformance(unittest.TestCase):
    """Test case for RelatedPairsConfluenceAnalyzer performance optimizations."""

    def setUp(self):
        """Set up test fixtures."""
        # Create mock services
        self.mock_correlation_service = MagicMock()
        self.mock_currency_strength_analyzer = MagicMock()
        
        # Initialize analyzer with mocks
        self.analyzer = RelatedPairsConfluenceAnalyzer(
            correlation_service=self.mock_correlation_service,
            currency_strength_analyzer=self.mock_currency_strength_analyzer,
            correlation_threshold=0.7,
            lookback_periods=20,
            cache_ttl_minutes=60,
            max_workers=4
        )
        
        # Generate sample price data
        self.sample_data = self._generate_sample_data()
        
        # Set up mock correlation data
        self.mock_correlations = {
            "EURUSD": {"GBPUSD": 0.85, "USDJPY": -0.75, "AUDUSD": 0.82},
            "GBPUSD": {"EURUSD": 0.85, "USDJPY": -0.65, "AUDUSD": 0.72},
            "USDJPY": {"EURUSD": -0.75, "GBPUSD": -0.65, "AUDUSD": -0.68},
            "AUDUSD": {"EURUSD": 0.82, "GBPUSD": 0.72, "USDJPY": -0.68}
        }
        
        # Configure mock to return correlations
        self.mock_correlation_service.get_all_correlations = MagicMock(
            return_value=self.mock_correlations
        )
        
    def _generate_sample_data(self, num_pairs=4, num_bars=500):
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
        
    async def test_find_related_pairs_performance(self):
        """Test the performance of find_related_pairs method."""
        # Run the method multiple times to test caching
        num_runs = 5
        total_time = 0
        
        for i in range(num_runs):
            start_time = time.time()
            result = await self.analyzer.find_related_pairs("EURUSD")
            end_time = time.time()
            
            # First run should be uncached, subsequent runs should use cache
            if i == 0:
                first_run_time = end_time - start_time
                print(f"First run (uncached): {first_run_time:.4f} seconds")
            else:
                total_time += (end_time - start_time)
                
        # Calculate average time for cached runs
        avg_cached_time = total_time / (num_runs - 1) if num_runs > 1 else 0
        print(f"Average cached run: {avg_cached_time:.4f} seconds")
        
        # Verify that caching provides significant speedup
        if num_runs > 1:
            self.assertLess(avg_cached_time, first_run_time * 0.5, 
                           "Caching should provide at least 50% speedup")
                           
        # Verify that the result is correct
        self.assertEqual(len(result), 3)  # Should find 3 related pairs
        self.assertIn("GBPUSD", result)
        self.assertIn("USDJPY", result)
        self.assertIn("AUDUSD", result)
        
    def test_detect_signal_performance(self):
        """Test the performance of signal detection methods."""
        # Get sample price data
        price_data = self.sample_data["EURUSD"]
        
        # Test trend signal detection
        start_time = time.time()
        trend_signal = self.analyzer._detect_signal(price_data, "trend", 20)
        trend_time = time.time() - start_time
        
        # Test reversal signal detection
        start_time = time.time()
        reversal_signal = self.analyzer._detect_signal(price_data, "reversal", 20)
        reversal_time = time.time() - start_time
        
        # Test breakout signal detection
        start_time = time.time()
        breakout_signal = self.analyzer._detect_signal(price_data, "breakout", 20)
        breakout_time = time.time() - start_time
        
        # Print performance results
        print(f"\nSignal detection performance:")
        print(f"  Trend signal: {trend_time:.4f} seconds")
        print(f"  Reversal signal: {reversal_time:.4f} seconds")
        print(f"  Breakout signal: {breakout_time:.4f} seconds")
        
        # Test caching performance
        start_time = time.time()
        cached_trend_signal = self.analyzer._detect_signal(price_data, "trend", 20)
        cached_trend_time = time.time() - start_time
        
        print(f"  Cached trend signal: {cached_trend_time:.4f} seconds")
        
        # Verify that caching provides significant speedup
        self.assertLess(cached_trend_time, trend_time * 0.5, 
                       "Caching should provide at least 50% speedup")
                       
        # Verify that the signals are detected correctly
        self.assertIsNotNone(trend_signal)
        self.assertIn("direction", trend_signal)
        self.assertIn("strength", trend_signal)
        
    def test_detect_confluence_performance(self):
        """Test the performance of detect_confluence method."""
        # Run the method and measure performance
        start_time = time.time()
        result = self.analyzer.detect_confluence(
            symbol="EURUSD",
            price_data=self.sample_data,
            signal_type="trend",
            signal_direction="bullish",
            related_pairs={"GBPUSD": 0.85, "USDJPY": -0.75, "AUDUSD": 0.82},
            use_currency_strength=True
        )
        end_time = time.time()
        
        # Print performance results
        execution_time = end_time - start_time
        print(f"\nDetect confluence: {execution_time:.4f} seconds")
        
        # Verify that the result is correct
        self.assertEqual(result["symbol"], "EURUSD")
        self.assertEqual(result["signal_type"], "trend")
        self.assertEqual(result["signal_direction"], "bullish")
        self.assertIn("confluence_score", result)
        
        # Verify reasonable performance
        self.assertLess(execution_time, 0.5, 
                       "Confluence detection should complete in under 0.5 seconds")


if __name__ == "__main__":
    unittest.main()
