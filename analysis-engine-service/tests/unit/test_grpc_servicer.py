import pytest
import asyncio
import grpc
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timezone

from analysis_engine.core.grpc_server import AnalysisEngineServicer
from analysis_engine.core.container import ServiceContainer # Assuming this is the correct path
from generated_protos import analysis_engine_pb2
from generated_protos.analysis_engine_pb2 import OrderSide
from google.protobuf.timestamp_pb2 import Timestamp

# Mock common_lib models if they are complex or not easily instantiable
# For example, if ITradingGateway.get_market_data returns a Pydantic model
class MockMarketDataPydanticModel:
    def __init__(self, data):
        self.data = data # data is a list of dicts like [{"open": ..., "timestamp": ...}]

class MockAccountInfoPydanticModel:
    def __init__(self, balance):
        self.balance = balance

class MockFeatureInfoModel: # From common_lib.interfaces.feature_store IFeatureProvider
    def __init__(self, name, description=""):
        self.name = name
        self.description = description

class MockModelMetadataModel: # From common_lib.interfaces.ml_integration IMLModelRegistry
    def __init__(self, id, name, symbol, timeframe):
        self.id = id
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe


@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture
def mock_service_container():
    """Provides a mock ServiceContainer."""
    # In the actual implementation, AnalysisEngineServicer uses the global `service_dependencies`
    # So, we don't strictly need to pass a container to its constructor for these tests,
    # as long as we patch `analysis_engine.core.grpc_server.service_dependencies`.
    # However, if the servicer was refactored to use the container, this would be how to mock it.
    return MagicMock(spec=ServiceContainer)

@pytest.fixture
def servicer_instance(mock_service_container):
    """Provides an instance of the AnalysisEngineServicer.
    The service_container is passed but might not be used if direct import of service_dependencies is active.
    """
    return AnalysisEngineServicer(service_container=mock_service_container)

@pytest.fixture
def mock_grpc_context():
    """Provides a mock gRPC context with abort functionality."""
    context = MagicMock(spec=grpc.aio.ServicerContext)
    context.abort = AsyncMock() # Use AsyncMock for abort as it's called in async methods
    return context


@pytest.mark.asyncio
@patch('analysis_engine.core.grpc_server.service_dependencies') # Patch the imported singleton
async def test_get_market_overview_success(mock_sd, servicer_instance: AnalysisEngineServicer, mock_grpc_context):
    # Setup Mocks for service_dependencies
    mock_trading_gateway = AsyncMock()
    mock_sd.get_trading_gateway.return_value = mock_trading_gateway
    
    mock_feature_provider = AsyncMock()
    mock_sd.get_feature_provider.return_value = mock_feature_provider
    
    mock_ml_registry = AsyncMock()
    mock_sd.get_ml_model_registry.return_value = mock_ml_registry

    # Configure mock return values
    current_dt = datetime.now(timezone.utc)
    market_data_list = [
        {"open": 1.1, "high": 1.2, "low": 1.0, "close": 1.15, "volume": 1000, "timestamp": current_dt}
    ]
    mock_trading_gateway.get_market_data.return_value = MockMarketDataPydanticModel(data=market_data_list)
    
    mock_feature_provider.get_available_features.return_value = [
        MockFeatureInfoModel(name="SMA_50"), MockFeatureInfoModel(name="RSI_14")
    ]
    # Assuming get_feature_data would return more complex data; simplified for now
    # We'll rely on the placeholder values in the servicer for this test part for indicators.

    mock_ml_registry.list_models.return_value = [
        MockModelMetadataModel(id="model1", name="TrendModelAlpha", symbol="EURUSD", timeframe="H1")
    ]
    # Similar simplification for model predictions, relying on placeholders in servicer.

    request = analysis_engine_pb2.GetMarketOverviewRequest(symbol="EURUSD", timeframe="H1", lookback_days=7)
    response = await servicer_instance.GetMarketOverview(request, mock_grpc_context)

    # Assertions
    mock_trading_gateway.get_market_data.assert_called_once()
    assert response.market_data.open == 1.1
    assert response.market_data.close == 1.15
    assert response.market_data.volume == 1000
    expected_ts = Timestamp()
    expected_ts.FromDatetime(current_dt)
    assert response.market_data.timestamp == expected_ts

    # Based on current servicer logic which adds placeholders if features/models are found
    assert "SMA_50" in response.technical_indicators 
    assert "RSI_14_placeholder" in response.technical_indicators # from servicer placeholder
    assert "TrendModelAlpha" in response.model_predictions # from servicer placeholder logic
    assert response.risk_assessment.sentiment == "NEUTRAL_placeholder" # from servicer placeholder

@pytest.mark.asyncio
@patch('analysis_engine.core.grpc_server.service_dependencies')
async def test_get_market_overview_dependency_error(mock_sd, servicer_instance: AnalysisEngineServicer, mock_grpc_context):
    mock_trading_gateway = AsyncMock()
    mock_sd.get_trading_gateway.return_value = mock_trading_gateway
    mock_trading_gateway.get_market_data.side_effect = Exception("Trading Gateway unavailable")

    request = analysis_engine_pb2.GetMarketOverviewRequest(symbol="EURUSD", timeframe="H1", lookback_days=7)
    await servicer_instance.GetMarketOverview(request, mock_grpc_context)
    
    mock_grpc_context.abort.assert_called_once_with(grpc.StatusCode.INTERNAL, "Internal server error: Trading Gateway unavailable")


@pytest.mark.asyncio
@patch('analysis_engine.core.grpc_server.service_dependencies')
async def test_get_trading_opportunity_success(mock_sd, servicer_instance: AnalysisEngineServicer, mock_grpc_context):
    # Setup Mocks
    mock_trading_gateway = AsyncMock()
    mock_sd.get_trading_gateway.return_value = mock_trading_gateway
    
    mock_ml_registry = AsyncMock()
    mock_sd.get_ml_model_registry.return_value = mock_ml_registry
    
    mock_risk_manager = AsyncMock()
    mock_sd.get_risk_manager.return_value = mock_risk_manager

    # Configure mock return values
    mock_trading_gateway.get_account_info.return_value = MockAccountInfoPydanticModel(balance=50000.0)
    current_dt = datetime.now(timezone.utc)
    market_data_list = [
        {"open": 1.2000, "high": 1.2050, "low": 1.1980, "close": 1.2020, "volume": 1500, "timestamp": current_dt}
    ]
    mock_trading_gateway.get_market_data.return_value = MockMarketDataPydanticModel(data=market_data_list)
    mock_risk_manager.calculate_position_size.return_value = {
        "position_size": 0.5, "risk_amount": 500.0, "reward_amount": 1000.0, "risk_reward_ratio": 2.0
    }
    mock_ml_registry.list_models.return_value = [ # Simulate one bullish, one bearish
        MockModelMetadataModel(id="modelA1", name="AlphaTrend", symbol="GBPUSD", timeframe="M15"),
        MockModelMetadataModel(id="modelB2", name="BetaReversal", symbol="GBPUSD", timeframe="M15")
    ]
    # Servicer logic for model predictions is placeholder, will determine direction based on ID "1" or "2"

    request = analysis_engine_pb2.GetTradingOpportunityRequest(
        symbol="GBPUSD", timeframe="M15", account_id="acc_test", risk_percentage=1.0
    )
    response = await servicer_instance.GetTradingOpportunity(request, mock_grpc_context)

    mock_trading_gateway.get_account_info.assert_called_once_with("acc_test")
    mock_risk_manager.calculate_position_size.assert_called_once()
    
    assert response.account_balance == 50000.0
    assert response.entry_price == 1.2020
    assert response.position_size == 0.5
    assert response.risk_details.risk_reward_ratio == 2.0
    # Direction depends on dummy logic in servicer (modelA1 is bullish, modelB2 is bearish -> tie -> SELL)
    assert response.trade_direction == OrderSide.SELL # Default for tie/more bearish in current dummy logic
    assert "AlphaTrend" in response.model_predictions
    assert "BetaReversal" in response.model_predictions

@pytest.mark.asyncio
@patch('analysis_engine.core.grpc_server.service_dependencies')
async def test_get_trading_opportunity_risk_manager_error(mock_sd, servicer_instance: AnalysisEngineServicer, mock_grpc_context):
    mock_trading_gateway = AsyncMock()
    mock_sd.get_trading_gateway.return_value = mock_trading_gateway
    mock_ml_registry = AsyncMock() # Not strictly needed for this error path but good practice
    mock_sd.get_ml_model_registry.return_value = mock_ml_registry
    mock_risk_manager = AsyncMock()
    mock_sd.get_risk_manager.return_value = mock_risk_manager

    mock_trading_gateway.get_account_info.return_value = MockAccountInfoPydanticModel(balance=10000.0)
    current_dt = datetime.now(timezone.utc)
    market_data_list = [{"close": 1.1000, "timestamp": current_dt}]
    mock_trading_gateway.get_market_data.return_value = MockMarketDataPydanticModel(data=market_data_list)
    
    mock_risk_manager.calculate_position_size.side_effect = Exception("Risk service timeout")

    request = analysis_engine_pb2.GetTradingOpportunityRequest(
        symbol="EURCAD", timeframe="H1", account_id="acc_err", risk_percentage=2.0
    )
    await servicer_instance.GetTradingOpportunity(request, mock_grpc_context)

    mock_grpc_context.abort.assert_called_once_with(grpc.StatusCode.INTERNAL, "Internal server error: Risk service timeout")


# Tests for placeholder RPCs (GetIndicators, GetPatterns, PerformAnalysis)
# These tests remain similar to before as their internal logic hasn't changed.
@pytest.mark.asyncio
async def test_get_indicators(servicer_instance: AnalysisEngineServicer, mock_grpc_context):
    request = analysis_engine_pb2.GetIndicatorsRequest(
        symbol="AUDUSD",
        timeframe="D1",
        indicator_names=["MACD_12_26_9", "CCI_20"]
    )
    context_mock = MagicMock()
    
    response = await servicer_instance.GetIndicators(request, context_mock)
    
    assert isinstance(response, analysis_engine_pb2.GetIndicatorsResponse)
    assert "MACD_12_26_9" in response.indicators
    assert response.indicators["MACD_12_26_9"] == 0.0005
    assert "CCI_20" in response.indicators
    assert response.indicators["CCI_20"] == 110.0

@pytest.mark.asyncio
async def test_get_patterns(servicer_instance: AnalysisEngineServicer):
    request = analysis_engine_pb2.GetPatternsRequest(
        symbol="USDCAD",
        timeframe="H4",
        pattern_names=["HeadAndShoulders", "DoubleTop"]
    )
    context_mock = MagicMock()
    
    response = await servicer_instance.GetPatterns(request, context_mock)
    
    assert isinstance(response, analysis_engine_pb2.GetPatternsResponse)
    assert "HeadAndShoulders" in response.patterns
    assert response.patterns["HeadAndShoulders"] is False
    assert "DoubleTop" in response.patterns
    assert response.patterns["DoubleTop"] is True

@pytest.mark.asyncio
async def test_perform_analysis(servicer_instance: AnalysisEngineServicer):
    request = analysis_engine_pb2.PerformAnalysisRequest(
        symbol="NZDUSD",
        timeframe="W1",
        analysis_type="trend_volatility_combined"
    )
    context_mock = MagicMock()
    
    response = await servicer_instance.PerformAnalysis(request, context_mock)
    
    assert isinstance(response, analysis_engine_pb2.PerformAnalysisResponse)
    assert response.summary == "Analysis for NZDUSD (trend_volatility_combined) completed."
    assert "key_finding_1" in response.details
    assert response.details["key_finding_1"] == "Market is currently range-bound."
    assert "recommendation" in response.details
    assert response.details["recommendation"] == "Wait for breakout."
