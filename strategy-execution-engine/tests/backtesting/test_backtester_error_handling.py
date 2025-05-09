"""
Tests for error handling in the Backtester.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock
import json
import os
import pandas as pd
import numpy as np

from strategy_execution_engine.backtesting.backtester import Backtester, run_backtest
from strategy_execution_engine.error import (
    BacktestConfigError,
    BacktestDataError,
    BacktestExecutionError,
    BacktestReportError
)


@pytest.fixture
def mock_backtest_engine():
    """Create a mock backtest engine."""
    engine = MagicMock()
    engine.execute = AsyncMock(return_value={
        "trades": [
            {"pnl": 100},
            {"pnl": -50},
            {"pnl": 75}
        ],
        "equity_curve": [
            {"equity": 10000},
            {"equity": 10100},
            {"equity": 10050},
            {"equity": 10125}
        ]
    })
    return engine


@pytest.fixture
def mock_report_generator():
    """Create a mock report generator."""
    generator = MagicMock()
    generator.generate_report = AsyncMock(return_value="path/to/report.html")
    return generator


@pytest.fixture
def backtester(mock_backtest_engine, mock_report_generator):
    """Create a backtester with mock dependencies."""
    bt = Backtester()
    bt.backtest_engine = mock_backtest_engine
    bt.report_generator = mock_report_generator
    return bt


@pytest.fixture
def valid_config():
    """Create a valid strategy configuration."""
    return {
        "id": "test_strategy",
        "name": "Test Strategy",
        "version": "1.0",
        "type": "trend_following",
        "instruments": ["EUR/USD", "GBP/USD"]
    }


@pytest.mark.asyncio
async def test_run_backtest_empty_config_path(backtester):
    """Test run_backtest with empty config path."""
    with pytest.raises(BacktestConfigError) as excinfo:
        await backtester.run_backtest("")
    
    assert "Strategy configuration path cannot be empty" in str(excinfo.value)


@pytest.mark.asyncio
async def test_run_backtest_invalid_date_format(backtester):
    """Test run_backtest with invalid date format."""
    with pytest.raises(BacktestConfigError) as excinfo:
        await backtester.run_backtest("path/to/config.json", start_date="2023/01/01")
    
    assert "Invalid date format" in str(excinfo.value)
    assert "expected_format" in excinfo.value.details


@pytest.mark.asyncio
async def test_run_backtest_start_date_after_end_date(backtester):
    """Test run_backtest with start date after end date."""
    with pytest.raises(BacktestConfigError) as excinfo:
        await backtester.run_backtest(
            "path/to/config.json",
            start_date="2023-01-01",
            end_date="2022-01-01"
        )
    
    assert "Start date must be before end date" in str(excinfo.value)


@pytest.mark.asyncio
async def test_load_strategy_config_nonexistent_file(backtester):
    """Test _load_strategy_config with nonexistent file."""
    with pytest.raises(BacktestConfigError) as excinfo:
        backtester._load_strategy_config("nonexistent_file.json")
    
    assert "Strategy configuration file not found" in str(excinfo.value)


@pytest.mark.asyncio
async def test_load_strategy_config_invalid_json(backtester, tmp_path):
    """Test _load_strategy_config with invalid JSON."""
    # Create a file with invalid JSON
    config_path = tmp_path / "invalid.json"
    with open(config_path, 'w') as f:
        f.write("{invalid json")
    
    with pytest.raises(BacktestConfigError) as excinfo:
        backtester._load_strategy_config(str(config_path))
    
    assert "Invalid JSON in strategy config file" in str(excinfo.value)


@pytest.mark.asyncio
async def test_load_strategy_config_missing_required_fields(backtester, tmp_path):
    """Test _load_strategy_config with missing required fields."""
    # Create a file with missing required fields
    config_path = tmp_path / "incomplete.json"
    with open(config_path, 'w') as f:
        json.dump({"id": "test"}, f)
    
    with pytest.raises(BacktestConfigError) as excinfo:
        backtester._load_strategy_config(str(config_path))
    
    assert "Invalid strategy configuration format" in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_market_data_empty_assets(backtester):
    """Test _get_market_data with empty assets."""
    with pytest.raises(BacktestDataError) as excinfo:
        await backtester._get_market_data([], datetime.now(), datetime.now() + timedelta(days=1))
    
    assert "No assets specified for market data retrieval" in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_market_data_invalid_date_types(backtester):
    """Test _get_market_data with invalid date types."""
    with pytest.raises(BacktestDataError) as excinfo:
        await backtester._get_market_data(["EUR/USD"], "2023-01-01", datetime.now())
    
    assert "Invalid date types for market data retrieval" in str(excinfo.value)


@pytest.mark.asyncio
async def test_get_market_data_start_date_after_end_date(backtester):
    """Test _get_market_data with start date after end date."""
    with pytest.raises(BacktestDataError) as excinfo:
        await backtester._get_market_data(
            ["EUR/USD"],
            datetime.now() + timedelta(days=1),
            datetime.now()
        )
    
    assert "Start date must be before end date for market data retrieval" in str(excinfo.value)


@pytest.mark.asyncio
async def test_calculate_performance_metrics_empty_result(backtester):
    """Test _calculate_performance_metrics with empty result."""
    result = backtester._calculate_performance_metrics({})
    
    assert result.get("_placeholder") is True
    assert result.get("sharpe_ratio") == 0.0


@pytest.mark.asyncio
async def test_calculate_performance_metrics_no_trades(backtester):
    """Test _calculate_performance_metrics with no trades."""
    result = backtester._calculate_performance_metrics({"trades": []})
    
    assert result.get("_placeholder") is True
    assert result.get("total_trades") == 0


@pytest.mark.asyncio
async def test_calculate_performance_metrics_invalid_equity_curve(backtester):
    """Test _calculate_performance_metrics with invalid equity curve."""
    with pytest.raises(BacktestExecutionError) as excinfo:
        backtester._calculate_performance_metrics({
            "trades": [{"pnl": 100}],
            "equity_curve": [{"invalid": "data"}]
        })
    
    assert "Failed to process equity curve" in str(excinfo.value)


@pytest.mark.asyncio
async def test_backtest_execution_error(backtester, valid_config):
    """Test handling of backtest execution error."""
    # Mock the backtest engine to raise an exception
    backtester.backtest_engine.execute.side_effect = Exception("Execution failed")
    
    with pytest.raises(BacktestExecutionError) as excinfo:
        await backtester.run_backtest(valid_config)
    
    assert "Failed to execute backtest" in str(excinfo.value)
    assert "Execution failed" in str(excinfo.value.details.get("original_error"))


@pytest.mark.asyncio
async def test_report_generation_error(backtester, valid_config):
    """Test handling of report generation error."""
    # Configure backtester to generate reports
    backtester.config["generate_reports"] = True
    
    # Mock the report generator to raise an exception
    backtester.report_generator.generate_report.side_effect = Exception("Report generation failed")
    
    with pytest.raises(BacktestReportError) as excinfo:
        await backtester.run_backtest(valid_config)
    
    assert "Failed to generate backtest report" in str(excinfo.value)
    assert "Report generation failed" in str(excinfo.value.details.get("original_error"))


@pytest.mark.asyncio
async def test_convenience_function_error_handling():
    """Test error handling in the convenience function."""
    with pytest.raises(BacktestConfigError) as excinfo:
        await run_backtest("")
    
    assert "Strategy configuration path cannot be empty" in str(excinfo.value)


@pytest.mark.asyncio
async def test_successful_backtest(backtester, valid_config):
    """Test a successful backtest run."""
    result = await backtester.run_backtest(valid_config)
    
    assert result["success"] is True
    assert "metrics" in result
    assert "metadata" in result
    assert result["metadata"]["strategy_id"] == "test_strategy"
