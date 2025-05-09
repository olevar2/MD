"""
Tests for error handling in the Market Data Service.
"""
import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch, AsyncMock

import pandas as pd
import numpy as np

from trading_gateway_service.services.market_data_service import MarketDataService
from trading_gateway_service.error import MarketDataError


@pytest.fixture
def mock_broker_adapter():
    """Create a mock broker adapter."""
    adapter = MagicMock()
    adapter.get_market_data = MagicMock(return_value={
        'bid': 1.0,
        'ask': 1.1,
        'volume': 1000
    })
    return adapter


@pytest.fixture
def market_data_service(mock_broker_adapter):
    """Create a market data service with mock dependencies."""
    broker_adapters = {'mock_broker': mock_broker_adapter}
    return MarketDataService(broker_adapters=broker_adapters)


@pytest.mark.asyncio
async def test_get_price_empty_instrument(market_data_service):
    """Test get_price with empty instrument."""
    with pytest.raises(MarketDataError) as excinfo:
        await market_data_service.get_price("")
    
    assert "Instrument cannot be empty" in str(excinfo.value)
    assert excinfo.value.details.get("symbol") == ""


@pytest.mark.asyncio
async def test_get_price_broker_error(market_data_service, mock_broker_adapter):
    """Test get_price when broker adapter raises an error."""
    # Make the broker adapter raise an exception
    mock_broker_adapter.get_market_data.side_effect = Exception("Broker error")
    
    with pytest.raises(MarketDataError) as excinfo:
        await market_data_service.get_price("EUR/USD")
    
    assert "Failed to get price for EUR/USD" in str(excinfo.value)
    assert "Broker error" in str(excinfo.value.details.get("error"))


@pytest.mark.asyncio
async def test_get_market_data_empty_instrument(market_data_service):
    """Test get_market_data with empty instrument."""
    with pytest.raises(MarketDataError) as excinfo:
        await market_data_service.get_market_data("")
    
    assert "Instrument cannot be empty" in str(excinfo.value)
    assert excinfo.value.details.get("symbol") == ""


@pytest.mark.asyncio
async def test_get_market_data_all_adapters_fail(market_data_service, mock_broker_adapter):
    """Test get_market_data when all broker adapters fail."""
    # Make the broker adapter raise an exception
    mock_broker_adapter.get_market_data.side_effect = Exception("Broker error")
    
    with pytest.raises(MarketDataError) as excinfo:
        await market_data_service.get_market_data("EUR/USD")
    
    assert "Failed to get market data for EUR/USD from all adapters" in str(excinfo.value)
    assert "adapter_errors" in excinfo.value.details


@pytest.mark.asyncio
async def test_get_historical_data_invalid_timeframe(market_data_service):
    """Test get_historical_data with invalid timeframe."""
    with pytest.raises(MarketDataError) as excinfo:
        await market_data_service.get_historical_data(
            instrument="EUR/USD",
            start_time=datetime.utcnow() - timedelta(days=1),
            end_time=datetime.utcnow(),
            timeframe="invalid"
        )
    
    assert "Invalid timeframe: invalid" in str(excinfo.value)
    assert "timeframe" in excinfo.value.details


@pytest.mark.asyncio
async def test_get_historical_data_invalid_time_range(market_data_service):
    """Test get_historical_data with invalid time range."""
    now = datetime.utcnow()
    
    with pytest.raises(MarketDataError) as excinfo:
        await market_data_service.get_historical_data(
            instrument="EUR/USD",
            start_time=now,
            end_time=now - timedelta(days=1),
            timeframe="1h"
        )
    
    assert "Start time must be before end time" in str(excinfo.value)
    assert "start_time" in excinfo.value.details
    assert "end_time" in excinfo.value.details


@pytest.mark.asyncio
async def test_get_volatility_empty_instrument(market_data_service):
    """Test get_volatility with empty instrument."""
    with pytest.raises(MarketDataError) as excinfo:
        await market_data_service.get_volatility("")
    
    assert "Instrument cannot be empty" in str(excinfo.value)
    assert excinfo.value.details.get("symbol") == ""


@pytest.mark.asyncio
async def test_get_avg_daily_volume_invalid_lookback(market_data_service):
    """Test get_avg_daily_volume with invalid lookback days."""
    with pytest.raises(MarketDataError) as excinfo:
        await market_data_service.get_historical_volume(
            instrument="EUR/USD",
            lookback_days=0
        )
    
    assert "Lookback days must be greater than 0" in str(excinfo.value)
    assert excinfo.value.details.get("lookback_days") == 0


@pytest.mark.asyncio
async def test_get_market_regime_empty_instrument(market_data_service):
    """Test get_market_regime with empty instrument."""
    with pytest.raises(MarketDataError) as excinfo:
        await market_data_service.get_market_regime("")
    
    assert "Instrument cannot be empty" in str(excinfo.value)
    assert excinfo.value.details.get("symbol") == ""


@pytest.mark.asyncio
async def test_get_predicted_volume_invalid_num_slices(market_data_service):
    """Test get_predicted_volume with invalid number of slices."""
    with pytest.raises(MarketDataError) as excinfo:
        await market_data_service.get_predicted_volume(
            instrument="EUR/USD",
            start_time=datetime.utcnow(),
            end_time=datetime.utcnow() + timedelta(hours=1),
            num_slices=0
        )
    
    assert "Number of slices must be greater than 0" in str(excinfo.value)
    assert excinfo.value.details.get("num_slices") == 0


@pytest.mark.asyncio
async def test_get_realtime_volume_invalid_lookback(market_data_service):
    """Test get_realtime_volume with invalid lookback minutes."""
    with pytest.raises(MarketDataError) as excinfo:
        await market_data_service.get_realtime_volume(
            instrument="EUR/USD",
            lookback_minutes=0
        )
    
    assert "Lookback minutes must be greater than 0" in str(excinfo.value)
    assert excinfo.value.details.get("lookback_minutes") == 0
