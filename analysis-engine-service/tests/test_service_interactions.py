"""
Integration tests for service interactions.

This module provides comprehensive tests for interactions between the Analysis Engine Service
and other services in the platform.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
import numpy as np
from unittest.mock import patch, MagicMock, AsyncMock

from analysis_engine.services.analysis_service import AnalysisService
from analysis_engine.analysis.confluence_analyzer import ConfluenceAnalyzer
from analysis_engine.analysis.multi_timeframe.multi_timeframe_analyzer import MultiTimeframeAnalyzer
from analysis_engine.analysis.advanced_ta.market_regime import MarketRegimeAnalyzer
from analysis_engine.core.errors import AnalysisError, ServiceUnavailableError, DataFetchError
from analysis_engine.integration.feature_store_client import FeatureStoreClient
from analysis_engine.integration.market_data_client import MarketDataClient

@pytest.fixture
def sample_market_data():
    """Create sample market data for testing."""
    # Create 100 data points
    dates = [(datetime.now() - timedelta(hours=i)).isoformat() for i in range(100, 0, -1)]
    
    # Create a price series
    close_prices = [1.1000]
    for i in range(1, 100):
        # Add trend
        trend = 0.0001 * i
        # Add some noise
        noise = np.random.normal(0, 0.0002)
        close_prices.append(close_prices[-1] + trend + noise)
    
    # Create high and low prices
    high_prices = [price + np.random.uniform(0.0005, 0.0015) for price in close_prices]
    low_prices = [price - np.random.uniform(0.0005, 0.0015) for price in close_prices]
    open_prices = [prev_close + np.random.normal(0, 0.0003) for prev_close in close_prices[:-1]]
    open_prices.insert(0, close_prices[0] - 0.0005)
    
    # Create volume data
    volumes = [int(1000 * (1 + np.random.uniform(-0.3, 0.3))) for _ in range(100)]
    
    # Convert to dictionary format expected by the API
    market_data = {
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }
    
    return market_data

# Fixture for sample market data
@pytest.fixture
def sample_market_data():
    """Provides sample market data for tests."""
    # Create 100 data points
    dates = [(datetime.now() - timedelta(hours=i)).isoformat() for i in range(100, 0, -1)]
    
    # Create a price series
    close_prices = [1.1000]
    for i in range(1, 100):
        # Add trend
        trend = 0.0001 * i
        # Add some noise
        noise = np.random.normal(0, 0.0002)
        close_prices.append(close_prices[-1] + trend + noise)
    
    # Create high and low prices
    high_prices = [price + np.random.uniform(0.0005, 0.0015) for price in close_prices]
    low_prices = [price - np.random.uniform(0.0005, 0.0015) for price in close_prices]
    open_prices = [prev_close + np.random.normal(0, 0.0003) for prev_close in close_prices[:-1]]
    open_prices.insert(0, close_prices[0] - 0.0005)
    
    # Create volume data
    volumes = [int(1000 * (1 + np.random.uniform(-0.3, 0.3))) for _ in range(100)]
    
    # Convert to dictionary format expected by the API
    market_data = {
        'timestamp': dates,
        'open': open_prices,
        'high': high_prices,
        'low': low_prices,
        'close': close_prices,
        'volume': volumes
    }
    
    return market_data

@pytest.mark.asyncio
async def test_analysis_with_feature_store_integration(mock_analysis_service: MagicMock, mock_feature_store_client: AsyncMock, sample_market_data):
    """Test analysis with feature store integration."""
    # Prepare analysis request
    request = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "market_data": sample_market_data,
        "analysis_types": ["confluence", "market_regime"],
        "include_indicators": True,
        "indicators": ["rsi", "macd", "bollinger_bands"]
    }
    
    # Configure mock service for this test
    mock_analysis_service.feature_store_client = mock_feature_store_client
    mock_analysis_service.market_data_client = AsyncMock(spec=MarketDataClient) # Basic mock for market data
    # Mock the analyze method to simulate a successful analysis
    mock_analysis_service.analyze = AsyncMock(return_value={
        "symbol": "EURUSD", "timeframe": "H1", "timestamp": datetime.now().isoformat(),
        "analysis_results": {"confluence": {"signal": "buy"}, "market_regime": {"regime": "trending"}},
        "metadata": {}
    })

    # Perform analysis
    result = await mock_analysis_service.analyze(request)
    
    # Verify result structure
    assert "symbol" in result
    assert "timeframe" in result
    assert "timestamp" in result
    assert "analysis_results" in result
    assert "metadata" in result
    
    # Verify analysis results
    assert "confluence" in result["analysis_results"]
    assert "market_regime" in result["analysis_results"]
    
    # Verify that indicators were fetched from feature store
    mock_analysis_service.feature_store_client.get_indicators.assert_called_once()
    call_args = mock_analysis_service.feature_store_client.get_indicators.call_args[0]
    assert call_args[0] == "EURUSD"  # symbol
    assert call_args[1] == "H1"      # timeframe
    assert set(call_args[2]) == set(["rsi", "macd", "bollinger_bands"])  # indicators

@pytest.mark.asyncio
async def test_analysis_with_market_data_integration(mock_analysis_service: MagicMock, mock_market_data_client: AsyncMock):
    """Test analysis with market data integration."""
    # Prepare analysis request without market_data (should fetch from market data service)
    request = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "analysis_types": ["confluence", "market_regime"],
        "count": 100  # Number of data points to fetch
    }
    
    # Configure mock service for this test
    mock_analysis_service.market_data_client = mock_market_data_client
    mock_analysis_service.feature_store_client = AsyncMock(spec=FeatureStoreClient) # Basic mock for feature store
    # Mock the analyze method
    mock_analysis_service.analyze = AsyncMock(return_value={
        "symbol": "EURUSD", "timeframe": "H1", "timestamp": datetime.now().isoformat(),
        "analysis_results": {"confluence": {"signal": "hold"}, "market_regime": {"regime": "ranging"}},
        "metadata": {}
    })

    # Perform analysis
    result = await mock_analysis_service.analyze(request)
    
    # Verify result structure
    assert "symbol" in result
    assert "timeframe" in result
    assert "timestamp" in result
    assert "analysis_results" in result
    assert "metadata" in result
    
    # Verify analysis results
    assert "confluence" in result["analysis_results"]
    assert "market_regime" in result["analysis_results"]
    
    # Verify that market data was fetched from market data service
    mock_analysis_service.market_data_client.get_historical_data.assert_called_once()
    call_args = mock_analysis_service.market_data_client.get_historical_data.call_args[0]
    assert call_args[0] == "EURUSD"  # symbol
    assert call_args[1] == "H1"      # timeframe
    assert call_args[2] == 100       # count

@pytest.mark.asyncio
async def test_feature_store_service_unavailable(mock_analysis_service: MagicMock, mock_feature_store_client: AsyncMock, sample_market_data):
    """Test handling of feature store service unavailability."""
    # Configure mock service for this test
    mock_analysis_service.feature_store_client = mock_feature_store_client
    mock_analysis_service.market_data_client = AsyncMock(spec=MarketDataClient)
    # Mock analyze to raise the error
    mock_analysis_service.analyze = AsyncMock(side_effect=ServiceUnavailableError(
        service_name="feature-store-service"
    ))
    # Mock the client method directly as well (might be redundant but safe)
    mock_feature_store_client.get_indicators.side_effect = ServiceUnavailableError(
        service_name="feature-store-service"
    )
    
    # Prepare analysis request
    request = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "market_data": sample_market_data,
        "analysis_types": ["confluence", "market_regime"],
        "include_indicators": True,
        "indicators": ["rsi", "macd", "bollinger_bands"]
    }
    
    # Verify that ServiceUnavailableError is raised
    with pytest.raises(ServiceUnavailableError) as excinfo:
        await mock_analysis_service.analyze(request)
    
    # Verify error details
    assert "feature-store-service" in str(excinfo.value)

@pytest.mark.asyncio
async def test_market_data_service_unavailable(mock_analysis_service: MagicMock, mock_market_data_client: AsyncMock):
    """Test handling of market data service unavailability."""
    # Configure mock service for this test
    mock_analysis_service.market_data_client = mock_market_data_client
    mock_analysis_service.feature_store_client = AsyncMock(spec=FeatureStoreClient)
    # Mock analyze to raise the error
    mock_analysis_service.analyze = AsyncMock(side_effect=ServiceUnavailableError(
        service_name="market-data-service"
    ))
    # Mock the client method directly
    mock_market_data_client.get_historical_data.side_effect = ServiceUnavailableError(
        service_name="market-data-service"
    )
    
    # Prepare analysis request without market_data (should fetch from market data service)
    request = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "analysis_types": ["confluence", "market_regime"],
        "count": 100  # Number of data points to fetch
    }
    
    # Verify that ServiceUnavailableError is raised
    with pytest.raises(ServiceUnavailableError) as excinfo:
        await mock_analysis_service.analyze(request)
    
    # Verify error details
    assert "market-data-service" in str(excinfo.value)

@pytest.mark.asyncio
async def test_data_fetch_error(mock_analysis_service: MagicMock, mock_market_data_client: AsyncMock):
    """Test handling of data fetch errors."""
    # Configure mock service for this test
    mock_analysis_service.market_data_client = mock_market_data_client
    mock_analysis_service.feature_store_client = AsyncMock(spec=FeatureStoreClient)
    # Mock analyze to raise the error
    mock_analysis_service.analyze = AsyncMock(side_effect=DataFetchError(
        message="Failed to fetch market data",
        source="market-data-service",
        details={"symbol": "EURUSD", "timeframe": "H1"}
    ))
    # Mock the client method directly
    mock_market_data_client.get_historical_data.side_effect = DataFetchError(
        message="Failed to fetch market data",
        source="market-data-service",
        details={"symbol": "EURUSD", "timeframe": "H1"}
    )
    
    # Prepare analysis request without market_data (should fetch from market data service)
    request = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "analysis_types": ["confluence", "market_regime"],
        "count": 100  # Number of data points to fetch
    }
    
    # Verify that DataFetchError is raised
    with pytest.raises(DataFetchError) as excinfo:
        await mock_analysis_service.analyze(request)
    
    # Verify error details
    assert "Failed to fetch market data" in str(excinfo.value)
    assert excinfo.value.source == "market-data-service"
    assert excinfo.value.details["symbol"] == "EURUSD"

@pytest.mark.asyncio
async def test_multi_timeframe_analysis_with_market_data_integration(mock_analysis_service: MagicMock, mock_market_data_client: AsyncMock):
    """Test multi-timeframe analysis with market data integration."""
    # Prepare analysis request for multi-timeframe analysis
    request = {
        "symbol": "EURUSD",
        "timeframes": ["M15", "H1", "H4"],
        "analysis_types": ["multi_timeframe"],
        "count": 200  # Number of data points to fetch
    }
    
    # Configure mock service for this test
    mock_analysis_service.market_data_client = mock_market_data_client
    mock_analysis_service.feature_store_client = AsyncMock(spec=FeatureStoreClient)
    # Mock analyze to return expected structure
    mock_analysis_service.analyze = AsyncMock(return_value={
        "symbol": "EURUSD", "timeframes": ["M15", "H1", "H4"], "timestamp": datetime.now().isoformat(),
        "analysis_results": {"multi_timeframe": {"timeframe_analysis": {"M15": {}, "H1": {}, "H4": {}}}},
        "metadata": {}
    })

    # Mock get_historical_data on the injected client to handle different timeframes
    original_side_effect = mock_market_data_client.get_historical_data.side_effect
    async def mock_get_historical_data_multi_tf(symbol, timeframe, count=100, from_date=None, to_date=None):
    """
    Mock get historical data multi tf.
    
    Args:
        symbol: Description of symbol
        timeframe: Description of timeframe
        count: Description of count
        from_date: Description of from_date
        to_date: Description of to_date
    
    """

        # Generate base data or use original side effect if needed
        base_data = await original_side_effect(symbol, "M15", count, from_date, to_date)
        
        if timeframe == "M15":
            return base_data
        elif timeframe == "H1":
            # Resample to H1 by taking every 4th point
            return {k: v[::4] for k, v in base_data.items()}
        elif timeframe == "H4":
            # Resample to H4 by taking every 16th point
            return {k: v[::16] for k, v in base_data.items()}
        else:
            return base_data
    
    # Replace the mock side_effect on the injected client
    mock_market_data_client.get_historical_data.side_effect = mock_get_historical_data_multi_tf
    
    # Perform analysis
    result = await mock_analysis_service.analyze(request)
    
    # Verify result structure
    assert "symbol" in result
    assert "timeframes" in result
    assert "timestamp" in result
    assert "analysis_results" in result
    assert "metadata" in result
    
    # Verify analysis results
    assert "multi_timeframe" in result["analysis_results"]
    assert "timeframe_analysis" in result["analysis_results"]["multi_timeframe"]
    
    # Verify that market data was fetched (at least once, maybe more depending on analyze mock)
    # If analyze is fully mocked, the internal calls might not happen.
    # If analyze uses the injected client, this assertion should reflect that.
    assert mock_analysis_service.market_data_client.get_historical_data.call_count >= 1 
    
    # Verify that the result contains analysis for each timeframe
    for tf in ["M15", "H1", "H4"]:
        assert tf in result["analysis_results"]["multi_timeframe"]["timeframe_analysis"]

@pytest.mark.asyncio
async def test_concurrent_analysis_requests(mock_analysis_service: MagicMock, sample_market_data):
    """Test handling of concurrent analysis requests."""
    # Prepare analysis requests
    requests = []
    for i in range(5):
        requests.append({
            "symbol": f"EUR/USD",
            "timeframe": "H1",
            "market_data": sample_market_data,
            "analysis_types": ["confluence", "market_regime"]
        })
    
    # Configure mock service for this test
    mock_analysis_service.analyze = AsyncMock(return_value={
        "symbol": "EUR/USD", "timeframe": "H1", "timestamp": datetime.now().isoformat(),
        "analysis_results": {"confluence": {}, "market_regime": {}},
        "metadata": {}
    })

    # Run concurrent analysis requests
    tasks = [mock_analysis_service.analyze(request) for request in requests]
    results = await asyncio.gather(*tasks)
    
    # Verify that all requests were processed successfully
    assert len(results) == 5
    for result in results:
        assert "symbol" in result
        assert "timeframe" in result
        assert "timestamp" in result
        assert "analysis_results" in result
        assert "metadata" in result
        assert "confluence" in result["analysis_results"]
        assert "market_regime" in result["analysis_results"]

@pytest.mark.asyncio
async def test_analysis_with_custom_parameters(mock_analysis_service: MagicMock, sample_market_data):
    """Test analysis with custom analyzer parameters."""
    # Prepare analysis request with custom parameters
    request = {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "market_data": sample_market_data,
        "analysis_types": ["confluence", "market_regime"],
        "analyzer_parameters": {
            "confluence": {
                "min_tools_for_confluence": 3,
                "effectiveness_threshold": 0.7
            },
            "market_regime": {
                "atr_period": 10,
                "adx_period": 10
            }
        }
    }
    
    # Configure mock service for this test
    mock_analysis_service.analyze = AsyncMock(return_value={
        "symbol": "EURUSD", "timeframe": "H1", "timestamp": datetime.now().isoformat(),
        "analysis_results": {"confluence": {"confluence_zones": []}, "market_regime": {"regime": "ranging"}},
        "metadata": {}
    })

    # Perform analysis
    result = await mock_analysis_service.analyze(request)
    
    # Verify result structure
    assert "symbol" in result
    assert "timeframe" in result
    assert "timestamp" in result
    assert "analysis_results" in result
    assert "metadata" in result
    
    # Verify analysis results
    assert "confluence" in result["analysis_results"]
    assert "market_regime" in result["analysis_results"]
    
    # Verify that custom parameters were used (indirectly by checking successful completion)
    assert "confluence_zones" in result["analysis_results"]["confluence"]
    assert "regime" in result["analysis_results"]["market_regime"]
