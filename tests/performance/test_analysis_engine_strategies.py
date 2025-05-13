"""
Performance tests for Analysis Engine Service strategies.
"""

import pytest
import asyncio
import time
import random
from typing import Dict, Any, List, Optional
from unittest.mock import patch, MagicMock, AsyncMock

from analysis_engine.analysis.strategies.strategy_factory import StrategyFactory
from analysis_engine.analysis.strategies.base_strategy import BaseStrategy
from analysis_engine.database import Database
from analysis_engine.service_clients import ServiceClients


class TestAnalysisEngineStrategiesPerformance:
    """Performance tests for Analysis Engine Service strategies."""
    
    @pytest.fixture
    def mock_market_data(self):
        """Create mock market data."""
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
    
    @pytest.fixture
    def mock_indicators_data(self):
        """Create mock indicators data."""
        data = []
        timestamp = time.time()
        
        for i in range(1000):
            data.append({
                "timestamp": timestamp + i * 60,
                "sma_14": random.uniform(1.0, 1.1),
                "rsi_14": random.uniform(30, 70),
                "macd": random.uniform(-0.01, 0.01),
                "macd_signal": random.uniform(-0.01, 0.01),
                "macd_histogram": random.uniform(-0.01, 0.01),
                "bb_upper": random.uniform(1.1, 1.2),
                "bb_middle": random.uniform(1.0, 1.1),
                "bb_lower": random.uniform(0.9, 1.0)
            })
        
        return data
    
    @pytest.mark.asyncio
    async def test_trend_following_strategy_performance(self, performance_metrics, time_it, mock_market_data, mock_indicators_data):
        """Test trend following strategy performance."""
        # Create strategy
        strategy_factory = StrategyFactory()
        strategy = strategy_factory.create_strategy("trend_following")
        
        # Test performance
        @time_it("trend_following_strategy_execution")
        async def execute_strategy():
    """
    Execute strategy.
    
    """

            return await strategy.execute(
                market_data=mock_market_data,
                indicators_data=mock_indicators_data,
                parameters={
                    "fast_period": 12,
                    "slow_period": 26,
                    "signal_period": 9,
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30
                }
            )
        
        # Run multiple times to get better statistics
        for _ in range(10):
            result = await execute_strategy()
        
        # Get statistics
        stats = performance_metrics.get_stats("trend_following_strategy_execution")
        
        # Print statistics
        print(f"Trend Following Strategy Execution Performance:")
        print(f"  Min: {stats['min']:.6f} seconds")
        print(f"  Max: {stats['max']:.6f} seconds")
        print(f"  Mean: {stats['mean']:.6f} seconds")
        print(f"  Median: {stats['median']:.6f} seconds")
        print(f"  P95: {stats['p95']:.6f} seconds")
        print(f"  P99: {stats['p99']:.6f} seconds")
        print(f"  Count: {stats['count']}")
        
        # Verify result
        assert "signal" in result
        assert "confidence" in result
        assert "timestamp" in result
        
        # Verify performance
        assert stats["mean"] < 0.1, "Trend following strategy execution is too slow"
    
    @pytest.mark.asyncio
    async def test_mean_reversion_strategy_performance(self, performance_metrics, time_it, mock_market_data, mock_indicators_data):
        """Test mean reversion strategy performance."""
        # Create strategy
        strategy_factory = StrategyFactory()
        strategy = strategy_factory.create_strategy("mean_reversion")
        
        # Test performance
        @time_it("mean_reversion_strategy_execution")
        async def execute_strategy():
    """
    Execute strategy.
    
    """

            return await strategy.execute(
                market_data=mock_market_data,
                indicators_data=mock_indicators_data,
                parameters={
                    "bb_period": 20,
                    "bb_std_dev": 2,
                    "rsi_period": 14,
                    "rsi_overbought": 70,
                    "rsi_oversold": 30
                }
            )
        
        # Run multiple times to get better statistics
        for _ in range(10):
            result = await execute_strategy()
        
        # Get statistics
        stats = performance_metrics.get_stats("mean_reversion_strategy_execution")
        
        # Print statistics
        print(f"Mean Reversion Strategy Execution Performance:")
        print(f"  Min: {stats['min']:.6f} seconds")
        print(f"  Max: {stats['max']:.6f} seconds")
        print(f"  Mean: {stats['mean']:.6f} seconds")
        print(f"  Median: {stats['median']:.6f} seconds")
        print(f"  P95: {stats['p95']:.6f} seconds")
        print(f"  P99: {stats['p99']:.6f} seconds")
        print(f"  Count: {stats['count']}")
        
        # Verify result
        assert "signal" in result
        assert "confidence" in result
        assert "timestamp" in result
        
        # Verify performance
        assert stats["mean"] < 0.1, "Mean reversion strategy execution is too slow"
    
    @pytest.mark.asyncio
    async def test_breakout_strategy_performance(self, performance_metrics, time_it, mock_market_data, mock_indicators_data):
        """Test breakout strategy performance."""
        # Create strategy
        strategy_factory = StrategyFactory()
        strategy = strategy_factory.create_strategy("breakout")
        
        # Test performance
        @time_it("breakout_strategy_execution")
        async def execute_strategy():
    """
    Execute strategy.
    
    """

            return await strategy.execute(
                market_data=mock_market_data,
                indicators_data=mock_indicators_data,
                parameters={
                    "lookback_period": 20,
                    "volatility_factor": 1.5,
                    "confirmation_period": 3
                }
            )
        
        # Run multiple times to get better statistics
        for _ in range(10):
            result = await execute_strategy()
        
        # Get statistics
        stats = performance_metrics.get_stats("breakout_strategy_execution")
        
        # Print statistics
        print(f"Breakout Strategy Execution Performance:")
        print(f"  Min: {stats['min']:.6f} seconds")
        print(f"  Max: {stats['max']:.6f} seconds")
        print(f"  Mean: {stats['mean']:.6f} seconds")
        print(f"  Median: {stats['median']:.6f} seconds")
        print(f"  P95: {stats['p95']:.6f} seconds")
        print(f"  P99: {stats['p99']:.6f} seconds")
        print(f"  Count: {stats['count']}")
        
        # Verify result
        assert "signal" in result
        assert "confidence" in result
        assert "timestamp" in result
        
        # Verify performance
        assert stats["mean"] < 0.1, "Breakout strategy execution is too slow"
    
    @pytest.mark.asyncio
    async def test_multiple_strategies_performance(self, performance_metrics, time_it, mock_market_data, mock_indicators_data):
        """Test multiple strategies performance."""
        # Create strategies
        strategy_factory = StrategyFactory()
        trend_following_strategy = strategy_factory.create_strategy("trend_following")
        mean_reversion_strategy = strategy_factory.create_strategy("mean_reversion")
        breakout_strategy = strategy_factory.create_strategy("breakout")
        
        # Test performance
        @time_it("multiple_strategies_execution")
        async def execute_multiple_strategies():
    """
    Execute multiple strategies.
    
    """

            # Execute strategies in parallel
            tasks = [
                trend_following_strategy.execute(
                    market_data=mock_market_data,
                    indicators_data=mock_indicators_data,
                    parameters={
                        "fast_period": 12,
                        "slow_period": 26,
                        "signal_period": 9,
                        "rsi_period": 14,
                        "rsi_overbought": 70,
                        "rsi_oversold": 30
                    }
                ),
                mean_reversion_strategy.execute(
                    market_data=mock_market_data,
                    indicators_data=mock_indicators_data,
                    parameters={
                        "bb_period": 20,
                        "bb_std_dev": 2,
                        "rsi_period": 14,
                        "rsi_overbought": 70,
                        "rsi_oversold": 30
                    }
                ),
                breakout_strategy.execute(
                    market_data=mock_market_data,
                    indicators_data=mock_indicators_data,
                    parameters={
                        "lookback_period": 20,
                        "volatility_factor": 1.5,
                        "confirmation_period": 3
                    }
                )
            ]
            
            return await asyncio.gather(*tasks)
        
        # Run multiple times to get better statistics
        for _ in range(10):
            results = await execute_multiple_strategies()
        
        # Get statistics
        stats = performance_metrics.get_stats("multiple_strategies_execution")
        
        # Print statistics
        print(f"Multiple Strategies Execution Performance:")
        print(f"  Min: {stats['min']:.6f} seconds")
        print(f"  Max: {stats['max']:.6f} seconds")
        print(f"  Mean: {stats['mean']:.6f} seconds")
        print(f"  Median: {stats['median']:.6f} seconds")
        print(f"  P95: {stats['p95']:.6f} seconds")
        print(f"  P99: {stats['p99']:.6f} seconds")
        print(f"  Count: {stats['count']}")
        
        # Verify results
        assert len(results) == 3
        for result in results:
            assert "signal" in result
            assert "confidence" in result
            assert "timestamp" in result
        
        # Verify performance
        assert stats["mean"] < 0.2, "Multiple strategies execution is too slow"
