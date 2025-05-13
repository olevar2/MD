"""
Performance tests for Feature Store Service indicators.
"""

import pytest
import asyncio
import time
import random
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock, AsyncMock

from feature_store_service.indicators.technical import TechnicalIndicator
from feature_store_service.indicators.advanced_indicators_registrar import AdvancedIndicatorsRegistrar
from feature_store_service.database import Database
from feature_store_service.service_clients import ServiceClients


class TestFeatureStoreIndicatorsPerformance:
    """Performance tests for Feature Store Service indicators."""
    
    @pytest.fixture
    def mock_ohlcv_data(self):
        """Create mock OHLCV data."""
        data = []
        timestamp = time.time()
        
        for i in range(1000):
            data.append({
                "timestamp": timestamp + i * 60,
                "open": random.uniform(1.0, 1.1),
                "high": random.uniform(1.1, 1.2),
                "low": random.uniform(0.9, 1.0),
                "close": random.uniform(1.0, 1.1),
                "volume": random.randint(1000, 2000)
            })
        
        return data
    
    @pytest.mark.asyncio
    async def test_sma_performance(self, performance_metrics, time_it, mock_ohlcv_data):
        """Test SMA indicator performance."""
        # Create indicator
        indicator = TechnicalIndicator("sma")
        
        # Test performance
        @time_it("sma_calculation")
        async def calculate_sma():
    """
    Calculate sma.
    
    """

            return indicator.calculate(
                mock_ohlcv_data,
                {"period": 14, "price": "close"}
            )
        
        # Run multiple times to get better statistics
        for _ in range(10):
            result = await calculate_sma()
        
        # Get statistics
        stats = performance_metrics.get_stats("sma_calculation")
        
        # Print statistics
        print(f"SMA Calculation Performance:")
        print(f"  Min: {stats['min']:.6f} seconds")
        print(f"  Max: {stats['max']:.6f} seconds")
        print(f"  Mean: {stats['mean']:.6f} seconds")
        print(f"  Median: {stats['median']:.6f} seconds")
        print(f"  P95: {stats['p95']:.6f} seconds")
        print(f"  P99: {stats['p99']:.6f} seconds")
        print(f"  Count: {stats['count']}")
        
        # Verify result
        assert len(result) == len(mock_ohlcv_data) - 13
        
        # Verify performance
        assert stats["mean"] < 0.1, "SMA calculation is too slow"
    
    @pytest.mark.asyncio
    async def test_rsi_performance(self, performance_metrics, time_it, mock_ohlcv_data):
        """Test RSI indicator performance."""
        # Create indicator
        indicator = TechnicalIndicator("rsi")
        
        # Test performance
        @time_it("rsi_calculation")
        async def calculate_rsi():
    """
    Calculate rsi.
    
    """

            return indicator.calculate(
                mock_ohlcv_data,
                {"period": 14, "price": "close"}
            )
        
        # Run multiple times to get better statistics
        for _ in range(10):
            result = await calculate_rsi()
        
        # Get statistics
        stats = performance_metrics.get_stats("rsi_calculation")
        
        # Print statistics
        print(f"RSI Calculation Performance:")
        print(f"  Min: {stats['min']:.6f} seconds")
        print(f"  Max: {stats['max']:.6f} seconds")
        print(f"  Mean: {stats['mean']:.6f} seconds")
        print(f"  Median: {stats['median']:.6f} seconds")
        print(f"  P95: {stats['p95']:.6f} seconds")
        print(f"  P99: {stats['p99']:.6f} seconds")
        print(f"  Count: {stats['count']}")
        
        # Verify result
        assert len(result) == len(mock_ohlcv_data) - 14
        
        # Verify performance
        assert stats["mean"] < 0.1, "RSI calculation is too slow"
    
    @pytest.mark.asyncio
    async def test_macd_performance(self, performance_metrics, time_it, mock_ohlcv_data):
        """Test MACD indicator performance."""
        # Create indicator
        indicator = TechnicalIndicator("macd")
        
        # Test performance
        @time_it("macd_calculation")
        async def calculate_macd():
    """
    Calculate macd.
    
    """

            return indicator.calculate(
                mock_ohlcv_data,
                {"fast_period": 12, "slow_period": 26, "signal_period": 9, "price": "close"}
            )
        
        # Run multiple times to get better statistics
        for _ in range(10):
            result = await calculate_macd()
        
        # Get statistics
        stats = performance_metrics.get_stats("macd_calculation")
        
        # Print statistics
        print(f"MACD Calculation Performance:")
        print(f"  Min: {stats['min']:.6f} seconds")
        print(f"  Max: {stats['max']:.6f} seconds")
        print(f"  Mean: {stats['mean']:.6f} seconds")
        print(f"  Median: {stats['median']:.6f} seconds")
        print(f"  P95: {stats['p95']:.6f} seconds")
        print(f"  P99: {stats['p99']:.6f} seconds")
        print(f"  Count: {stats['count']}")
        
        # Verify result
        assert len(result) == len(mock_ohlcv_data) - 33
        
        # Verify performance
        assert stats["mean"] < 0.1, "MACD calculation is too slow"
    
    @pytest.mark.asyncio
    async def test_bollinger_bands_performance(self, performance_metrics, time_it, mock_ohlcv_data):
        """Test Bollinger Bands indicator performance."""
        # Create indicator
        indicator = TechnicalIndicator("bollinger_bands")
        
        # Test performance
        @time_it("bollinger_bands_calculation")
        async def calculate_bollinger_bands():
    """
    Calculate bollinger bands.
    
    """

            return indicator.calculate(
                mock_ohlcv_data,
                {"period": 20, "std_dev": 2, "price": "close"}
            )
        
        # Run multiple times to get better statistics
        for _ in range(10):
            result = await calculate_bollinger_bands()
        
        # Get statistics
        stats = performance_metrics.get_stats("bollinger_bands_calculation")
        
        # Print statistics
        print(f"Bollinger Bands Calculation Performance:")
        print(f"  Min: {stats['min']:.6f} seconds")
        print(f"  Max: {stats['max']:.6f} seconds")
        print(f"  Mean: {stats['mean']:.6f} seconds")
        print(f"  Median: {stats['median']:.6f} seconds")
        print(f"  P95: {stats['p95']:.6f} seconds")
        print(f"  P99: {stats['p99']:.6f} seconds")
        print(f"  Count: {stats['count']}")
        
        # Verify result
        assert len(result) == len(mock_ohlcv_data) - 19
        
        # Verify performance
        assert stats["mean"] < 0.1, "Bollinger Bands calculation is too slow"
    
    @pytest.mark.asyncio
    async def test_multiple_indicators_performance(self, performance_metrics, time_it, mock_ohlcv_data):
        """Test multiple indicators performance."""
        # Create indicators
        sma_indicator = TechnicalIndicator("sma")
        rsi_indicator = TechnicalIndicator("rsi")
        macd_indicator = TechnicalIndicator("macd")
        bb_indicator = TechnicalIndicator("bollinger_bands")
        
        # Test performance
        @time_it("multiple_indicators_calculation")
        async def calculate_multiple_indicators():
    """
    Calculate multiple indicators.
    
    """

            # Calculate indicators in parallel
            tasks = [
                sma_indicator.calculate(mock_ohlcv_data, {"period": 14, "price": "close"}),
                rsi_indicator.calculate(mock_ohlcv_data, {"period": 14, "price": "close"}),
                macd_indicator.calculate(mock_ohlcv_data, {"fast_period": 12, "slow_period": 26, "signal_period": 9, "price": "close"}),
                bb_indicator.calculate(mock_ohlcv_data, {"period": 20, "std_dev": 2, "price": "close"})
            ]
            
            return await asyncio.gather(*tasks)
        
        # Run multiple times to get better statistics
        for _ in range(10):
            results = await calculate_multiple_indicators()
        
        # Get statistics
        stats = performance_metrics.get_stats("multiple_indicators_calculation")
        
        # Print statistics
        print(f"Multiple Indicators Calculation Performance:")
        print(f"  Min: {stats['min']:.6f} seconds")
        print(f"  Max: {stats['max']:.6f} seconds")
        print(f"  Mean: {stats['mean']:.6f} seconds")
        print(f"  Median: {stats['median']:.6f} seconds")
        print(f"  P95: {stats['p95']:.6f} seconds")
        print(f"  P99: {stats['p99']:.6f} seconds")
        print(f"  Count: {stats['count']}")
        
        # Verify results
        assert len(results) == 4
        
        # Verify performance
        assert stats["mean"] < 0.2, "Multiple indicators calculation is too slow"
