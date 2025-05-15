import pytest
from unittest.mock import patch, AsyncMock
from datetime import datetime, timedelta, UTC

from analysis_coordinator_service.adapters.market_analysis_adapter import MarketAnalysisAdapter
from analysis_coordinator_service.adapters.causal_analysis_adapter import CausalAnalysisAdapter
from analysis_coordinator_service.adapters.backtesting_adapter import BacktestingAdapter

@pytest.mark.asyncio
async def test_market_analysis_adapter_analyze_market():
    # Arrange
    adapter = MarketAnalysisAdapter(base_url="http://market-analysis-service:8000")
    symbol = "EURUSD"
    timeframe = "1h"
    start_date = datetime.now(UTC)
    end_date = start_date + timedelta(days=1)
    parameters = {"patterns": ["head_and_shoulders"]}
    
    # Mock the _make_request method directly
    with patch.object(adapter, '_make_request', new=AsyncMock(return_value={"result": "success"})):
        # Act
        result = await adapter.analyze_market(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            parameters=parameters
        )
    
    # Assert
    assert result == {"result": "success"}

@pytest.mark.asyncio
async def test_market_analysis_adapter_error_handling():
    # Arrange
    adapter = MarketAnalysisAdapter(base_url="http://market-analysis-service:8000")
    symbol = "EURUSD"
    timeframe = "1h"
    start_date = datetime.now(UTC)
    
    # Mock the _make_request method to raise an exception
    with patch.object(adapter, '_make_request', new=AsyncMock(side_effect=Exception("Market analysis request failed: Internal Server Error"))):
        # Act & Assert
        with pytest.raises(Exception) as excinfo:
            await adapter.analyze_market(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date
            )
    
    assert "Market analysis request failed" in str(excinfo.value)

@pytest.mark.asyncio
async def test_causal_analysis_adapter_generate_causal_graph():
    # Arrange
    adapter = CausalAnalysisAdapter(base_url="http://causal-analysis-service:8000")
    symbol = "EURUSD"
    timeframe = "1h"
    start_date = datetime.now(UTC)
    end_date = start_date + timedelta(days=1)
    variables = ["price", "volume"]
    parameters = {"max_lag": 5}
    
    # Mock the _make_request method directly
    with patch.object(adapter, '_make_request', new=AsyncMock(return_value={"result": "success"})):
        # Act
        result = await adapter.generate_causal_graph(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            variables=variables,
            parameters=parameters
        )
    
    # Assert
    assert result == {"result": "success"}

@pytest.mark.asyncio
async def test_causal_analysis_adapter_error_handling():
    # Arrange
    adapter = CausalAnalysisAdapter(base_url="http://causal-analysis-service:8000")
    symbol = "EURUSD"
    timeframe = "1h"
    start_date = datetime.now(UTC)
    
    # Mock the _make_request method to raise an exception
    with patch.object(adapter, '_make_request', new=AsyncMock(side_effect=Exception("Causal graph request failed: Internal Server Error"))):
        # Act & Assert
        with pytest.raises(Exception) as excinfo:
            await adapter.generate_causal_graph(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date
            )
    
    assert "Causal graph request failed" in str(excinfo.value)

@pytest.mark.asyncio
async def test_backtesting_adapter_run_backtest():
    # Arrange
    adapter = BacktestingAdapter(base_url="http://backtesting-service:8000")
    symbol = "EURUSD"
    timeframe = "1h"
    start_date = datetime.now(UTC)
    end_date = start_date + timedelta(days=1)
    strategy_config = {"strategy": "moving_average_crossover", "fast_period": 10, "slow_period": 20}
    parameters = {"initial_capital": 10000}
    
    # Mock the _make_request method directly
    with patch.object(adapter, '_make_request', new=AsyncMock(return_value={"result": "success"})):
        # Act
        result = await adapter.run_backtest(
            symbol=symbol,
            timeframe=timeframe,
            start_date=start_date,
            end_date=end_date,
            strategy_config=strategy_config,
            parameters=parameters
        )
    
    # Assert
    assert result == {"result": "success"}

@pytest.mark.asyncio
async def test_backtesting_adapter_error_handling():
    # Arrange
    adapter = BacktestingAdapter(base_url="http://backtesting-service:8000")
    symbol = "EURUSD"
    timeframe = "1h"
    start_date = datetime.now(UTC)
    
    # Mock the _make_request method to raise an exception
    with patch.object(adapter, '_make_request', new=AsyncMock(side_effect=Exception("Backtest request failed: Internal Server Error"))):
        # Act & Assert
        with pytest.raises(Exception) as excinfo:
            await adapter.run_backtest(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date
            )
    
    assert "Backtest request failed" in str(excinfo.value)

@pytest.mark.asyncio
async def test_backtesting_adapter_get_backtest_result():
    # Arrange
    adapter = BacktestingAdapter(base_url="http://backtesting-service:8000")
    backtest_id = "test-backtest-id"
    
    # Mock the _make_request method directly
    with patch.object(adapter, '_make_request', new=AsyncMock(return_value={"result": "success"})):
        # Act
        result = await adapter.get_backtest_result(backtest_id)
    
    # Assert
    assert result == {"result": "success"}
