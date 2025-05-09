"""
Tests for error handling in the BacktestEngine.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
import json
import os
import pandas as pd
import numpy as np

from strategy_execution_engine.backtesting.backtest_engine import BacktestEngine
from strategy_execution_engine.error import (
    BacktestConfigError,
    BacktestDataError,
    BacktestExecutionError,
    BacktestReportError
)


@pytest.fixture
def sample_data():
    """Create sample price data for testing."""
    # Create date range
    dates = pd.date_range(start='2023-01-01', end='2023-01-10', freq='D')
    
    # Create price data
    data = pd.DataFrame({
        'open': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'high': [102, 103, 104, 105, 106, 107, 108, 109, 110, 111],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
        'close': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
    }, index=dates)
    
    return data


@pytest.fixture
def backtest_engine(sample_data):
    """Create a backtest engine with sample data."""
    engine = BacktestEngine(
        data=sample_data,
        initial_balance=10000.0,
        commission=0.001,
        slippage=0.0,
        spread=0.0,
        track_tool_effectiveness=True
    )
    return engine


@pytest.fixture
def mock_strategy_func():
    """Create a mock strategy function."""
    def strategy(data, engine, **params):
        # Open a position
        engine.open_position(
            timestamp=data.index[0],
            symbol="EUR/USD",
            direction="buy",
            size=1000,
            price=data['close'][0]
        )
        
        # Close the position
        engine.close_position(
            position_id=engine.positions[0]["id"],
            timestamp=data.index[-1],
            price=data['close'][-1]
        )
        
        return {"executed": True}
    
    return strategy


def test_run_strategy_no_data(backtest_engine):
    """Test run_strategy with no data."""
    backtest_engine.data = None
    
    with pytest.raises(BacktestDataError) as excinfo:
        backtest_engine.run_strategy(lambda data, engine: None)
    
    assert "No data provided for backtesting" in str(excinfo.value)


def test_run_strategy_invalid_strategy_func(backtest_engine):
    """Test run_strategy with invalid strategy function."""
    with pytest.raises(BacktestConfigError) as excinfo:
        backtest_engine.run_strategy("not_a_function")
    
    assert "Strategy function is not callable" in str(excinfo.value)


def test_run_strategy_execution_error(backtest_engine):
    """Test run_strategy with strategy execution error."""
    def failing_strategy(data, engine, **params):
        raise ValueError("Strategy execution failed")
    
    with pytest.raises(BacktestExecutionError) as excinfo:
        backtest_engine.run_strategy(failing_strategy)
    
    assert "Error running strategy" in str(excinfo.value)
    assert "Strategy execution failed" in str(excinfo.value.details.get("error"))


def test_open_position_empty_symbol(backtest_engine):
    """Test open_position with empty symbol."""
    with pytest.raises(BacktestExecutionError) as excinfo:
        backtest_engine.open_position(
            timestamp=datetime.now(),
            symbol="",
            direction="buy",
            size=1000,
            price=100
        )
    
    assert "Symbol cannot be empty" in str(excinfo.value)


def test_open_position_invalid_direction(backtest_engine):
    """Test open_position with invalid direction."""
    with pytest.raises(BacktestExecutionError) as excinfo:
        backtest_engine.open_position(
            timestamp=datetime.now(),
            symbol="EUR/USD",
            direction="invalid",
            size=1000,
            price=100
        )
    
    assert "Invalid direction" in str(excinfo.value)


def test_open_position_invalid_size(backtest_engine):
    """Test open_position with invalid size."""
    with pytest.raises(BacktestExecutionError) as excinfo:
        backtest_engine.open_position(
            timestamp=datetime.now(),
            symbol="EUR/USD",
            direction="buy",
            size=0,
            price=100
        )
    
    assert "Position size must be positive" in str(excinfo.value)


def test_close_position_empty_id(backtest_engine):
    """Test close_position with empty ID."""
    with pytest.raises(BacktestExecutionError) as excinfo:
        backtest_engine.close_position(
            position_id="",
            timestamp=datetime.now(),
            price=100
        )
    
    assert "Position ID cannot be empty" in str(excinfo.value)


def test_close_position_nonexistent_id(backtest_engine):
    """Test close_position with nonexistent ID."""
    result = backtest_engine.close_position(
        position_id="nonexistent",
        timestamp=datetime.now(),
        price=100
    )
    
    assert result is None


def test_register_tool_signal_empty_tool_name(backtest_engine):
    """Test register_tool_signal with empty tool name."""
    with pytest.raises(BacktestExecutionError) as excinfo:
        backtest_engine.register_tool_signal(
            tool_name="",
            signal_type="entry",
            direction="buy",
            strength=0.8,
            timestamp=datetime.now(),
            symbol="EUR/USD",
            timeframe="1h",
            price=100
        )
    
    assert "Tool name cannot be empty" in str(excinfo.value)


def test_register_tool_signal_empty_symbol(backtest_engine):
    """Test register_tool_signal with empty symbol."""
    with pytest.raises(BacktestExecutionError) as excinfo:
        backtest_engine.register_tool_signal(
            tool_name="RSI",
            signal_type="entry",
            direction="buy",
            strength=0.8,
            timestamp=datetime.now(),
            symbol="",
            timeframe="1h",
            price=100
        )
    
    assert "Symbol cannot be empty" in str(excinfo.value)


def test_register_tool_signal_invalid_direction(backtest_engine):
    """Test register_tool_signal with invalid direction."""
    with pytest.raises(BacktestExecutionError) as excinfo:
        backtest_engine.register_tool_signal(
            tool_name="RSI",
            signal_type="entry",
            direction="invalid",
            strength=0.8,
            timestamp=datetime.now(),
            symbol="EUR/USD",
            timeframe="1h",
            price=100
        )
    
    assert "Invalid direction" in str(excinfo.value)


def test_register_tool_signal_invalid_strength(backtest_engine):
    """Test register_tool_signal with invalid strength."""
    with pytest.raises(BacktestExecutionError) as excinfo:
        backtest_engine.register_tool_signal(
            tool_name="RSI",
            signal_type="entry",
            direction="buy",
            strength=1.5,
            timestamp=datetime.now(),
            symbol="EUR/USD",
            timeframe="1h",
            price=100
        )
    
    assert "Signal strength must be between 0.0 and 1.0" in str(excinfo.value)


def test_register_tool_outcome_no_ids(backtest_engine):
    """Test register_tool_outcome with no IDs."""
    with pytest.raises(BacktestExecutionError) as excinfo:
        backtest_engine.register_tool_outcome(
            signal_id="",
            outcome="success",
            exit_price=110,
            exit_timestamp=datetime.now(),
            internal_id=None
        )
    
    assert "Either signal_id or internal_id must be provided" in str(excinfo.value)


def test_register_tool_outcome_invalid_outcome(backtest_engine):
    """Test register_tool_outcome with invalid outcome."""
    with pytest.raises(BacktestExecutionError) as excinfo:
        backtest_engine.register_tool_outcome(
            signal_id="test_signal",
            outcome="invalid",
            exit_price=110,
            exit_timestamp=datetime.now()
        )
    
    assert "Invalid outcome" in str(excinfo.value)


def test_successful_strategy_execution(backtest_engine, mock_strategy_func):
    """Test successful strategy execution."""
    result = backtest_engine.run_strategy(mock_strategy_func)
    
    assert result["success"] is True
    assert "metrics" in result
    assert result["metrics"]["total_trades"] == 1
    assert result["metrics"]["winning_trades"] == 1


def test_calculate_metrics_no_trades(backtest_engine):
    """Test _calculate_metrics with no trades."""
    backtest_engine._calculate_metrics()
    
    assert backtest_engine.metrics["total_trades"] == 0
    assert backtest_engine.metrics["winning_trades"] == 0


def test_save_results_directory_creation(backtest_engine, tmp_path):
    """Test _save_results creates directory if it doesn't exist."""
    # Set output directory to a nonexistent path
    backtest_engine.output_dir = os.path.join(tmp_path, "nonexistent", "path")
    
    # Save results
    file_path = backtest_engine._save_results()
    
    # Check that the directory was created and the file exists
    assert os.path.exists(backtest_engine.output_dir)
    assert os.path.exists(file_path)
