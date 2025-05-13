"""
Tests for the Market Data Service.
"""
import pytest
from unittest.mock import MagicMock, AsyncMock
import pandas as pd
import asyncio

from services.market_data_service import MarketDataService

@pytest.fixture
def mock_broker_adapter():
    """Provides a mock broker adapter."""
    return MagicMock()

@pytest.fixture
def market_data_service(mock_broker_adapter):
    """Provides an instance of the MarketDataService with mock dependencies."""
    return MarketDataService(broker_adapters={'mock': mock_broker_adapter}, config={})

@pytest.mark.asyncio
async def test_get_current_price_success(market_data_service, mock_broker_adapter):
    # Arrange
    mock_broker_adapter.get_market_data.return_value = {'bid': 1.1234, 'ask': 1.1236}
    # Act
    price = await market_data_service.get_price('EURUSD')
    # Assert
    assert price == pytest.approx((1.1234 + 1.1236) / 2)

@pytest.mark.asyncio
async def test_get_current_price_error(mock_broker_adapter):
    # Simulate broker error
    mock_broker_adapter.get_market_data.side_effect = Exception("Broker unavailable")
    service = MarketDataService(broker_adapters={'mock': mock_broker_adapter}, config={})
    price = await service.get_price('EURUSD')
    assert price is None

@pytest.mark.asyncio
async def test_check_cache_first(mock_broker_adapter):
    service = MarketDataService(broker_adapters={'mock': mock_broker_adapter}, config={})
    # Pre-populate cache with price and fresh timestamp
    service.price_cache['EURUSD'] = {'price': 1.2000}
    service.last_price_update['EURUSD'] = asyncio.get_event_loop().time()
    price = await service.get_price('EURUSD')
    assert price == 1.2000
    mock_broker_adapter.get_market_data.assert_not_called()

@pytest.mark.asyncio
async def test_get_historical_data_success():
    # Prepare sample DataFrame
    sample_df = pd.DataFrame([
        {'timestamp': '2025-04-30T12:00:00Z', 'open': 1.1, 'high': 1.2, 'low': 1.0, 'close': 1.15, 'volume': 1000}
    ])
    # Mock historical data service with AsyncMock
    mock_hist = AsyncMock()
    mock_hist.get_historical_data.return_value = sample_df
    service = MarketDataService(broker_adapters={}, config={'historical_data_service': mock_hist})
    # Act
    df = await service.get_historical_data(
        'EURUSD', '2025-04-30T12:00:00Z', '2025-04-30T13:00:00Z', timeframe='1H'
    )
    # Assert
    assert df.equals(sample_df)
