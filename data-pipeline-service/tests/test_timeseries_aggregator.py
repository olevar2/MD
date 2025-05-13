"""
Tests for the TimeseriesAggregator service.

Contains unit tests for the TimeseriesAggregator class which handles
conversion of OHLCV data between different timeframes.
"""
import pytest
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

from core.timeseries_aggregator import TimeseriesAggregator
from common_lib.schemas import OHLCVData, TimeframeEnum, AggregationMethodEnum


def test_simple():
    """A simple test that always passes."""
    assert True


def create_test_data():
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


def test_basic_aggregation():
    """Test basic OHLCV aggregation from 1m to 5m."""
    aggregator = TimeseriesAggregator()
    data = create_test_data()
    
    # Aggregate from 1m to 5m
    result = aggregator.aggregate(
        data=data,
        source_timeframe=TimeframeEnum.ONE_MINUTE,
        target_timeframe=TimeframeEnum.FIVE_MINUTES,
        method=AggregationMethodEnum.OHLCV
    )
    
    # Should have 12 candles (60 minutes / 5)
    assert len(result) == 12
    
    # First 5m candle checks
    first_candle = result[0]
    assert first_candle.timestamp == data[0].timestamp
    assert first_candle.open == data[0].open
    assert first_candle.close == data[4].close
    
    # Check high and low are correct
    expected_high = max(d.high for d in data[0:5])
    expected_low = min(d.low for d in data[0:5])
    expected_volume = sum(d.volume for d in data[0:5])
    
    assert first_candle.high == expected_high
    assert first_candle.low == expected_low
    assert first_candle.volume == expected_volume


def test_vwap_aggregation():
    """Test VWAP aggregation."""
    aggregator = TimeseriesAggregator()
    data = create_test_data()
    
    # Aggregate using VWAP
    result = aggregator.aggregate(
        data=data,
        source_timeframe=TimeframeEnum.ONE_MINUTE,
        target_timeframe=TimeframeEnum.FIFTEEN_MINUTES,
        method=AggregationMethodEnum.VWAP
    )
    
    # Should have 4 candles (60 minutes / 15)
    assert len(result) == 4
    
    # Verify the first candle has correct VWAP calculation
    first_candle = result[0]
    
    # Calculate expected VWAP manually
    price_volume = sum(((d.high + d.low + d.close) / 3) * d.volume for d in data[0:15])
    total_volume = sum(d.volume for d in data[0:15])
    expected_vwap = price_volume / total_volume
    
    assert first_candle.close == pytest.approx(expected_vwap, rel=1e-10)
    assert first_candle.volume == total_volume


def test_empty_data():
    """Test that empty data returns empty results."""
    aggregator = TimeseriesAggregator()
    result = aggregator.aggregate(
        data=[],
        source_timeframe=TimeframeEnum.ONE_MINUTE,
        target_timeframe=TimeframeEnum.FIVE_MINUTES
    )
    assert len(result) == 0
