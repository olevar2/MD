"""
Direct test runner for TimeseriesAggregator tests.

This script runs the test directly without relying on the package structure
to help diagnose and fix import issues.
"""
import os
import sys
import unittest
from datetime import datetime, timezone, timedelta
import pandas as pd

# Add necessary paths to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Import directly from the module files
from core.timeseries_aggregator import TimeseriesAggregator
from common_lib.schemas import OHLCVData, TimeframeEnum, AggregationMethodEnum


class TimeseriesAggregatorTest(unittest.TestCase):
    """Test cases for TimeseriesAggregator."""

    def test_simple(self):
        """A simple test that always passes."""
        self.assertTrue(True)

    def create_test_data(self):
        """Create test OHLCV data for testing aggregation."""
        # Use a fixed start time for deterministic tests
        # Start from a round hour to make aggregation predictable
        start_time = datetime(2025, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        data = []
        
        # Create 60 one-minute candles (covering 1 hour)
        for i in range(60):
            candle_time = start_time + timedelta(minutes=i)
            data.append(OHLCVData(
                timestamp=candle_time,
                open=100 + i * 0.1,
                high=100 + i * 0.1 + 0.05,
                low=100 + i * 0.1 - 0.05,
                close=100 + i * 0.1 + (0.02 if i % 2 == 0 else -0.02),
                volume=100 + i
            ))
        return data

    def test_basic_aggregation(self):
        """Test basic OHLCV aggregation from 1m to 5m."""
        aggregator = TimeseriesAggregator()
        data = self.create_test_data()
        
        # Aggregate from 1m to 5m
        result = aggregator.aggregate(
            data=data,
            source_timeframe=TimeframeEnum.ONE_MINUTE,
            target_timeframe=TimeframeEnum.FIVE_MINUTES,
            method=AggregationMethodEnum.OHLCV
        )
        
        # Should have 12 candles (60 minutes / 5)
        self.assertEqual(len(result), 12)
        
        # First 5m candle checks
        first_candle = result[0]
        self.assertEqual(first_candle.timestamp, data[0].timestamp)
        self.assertEqual(first_candle.open, data[0].open)
        self.assertEqual(first_candle.close, data[4].close)
        
        # Check high and low are correct
        expected_high = max(d.high for d in data[0:5])
        expected_low = min(d.low for d in data[0:5])
        expected_volume = sum(d.volume for d in data[0:5])
        
        self.assertEqual(first_candle.high, expected_high)
        self.assertEqual(first_candle.low, expected_low)
        self.assertEqual(first_candle.volume, expected_volume)

    def test_empty_data(self):
        """Test that empty data returns empty results."""
        aggregator = TimeseriesAggregator()
        result = aggregator.aggregate(
            data=[],
            source_timeframe=TimeframeEnum.ONE_MINUTE,
            target_timeframe=TimeframeEnum.FIVE_MINUTES
        )
        self.assertEqual(len(result), 0)


if __name__ == '__main__':
    unittest.main()
