import pytest
import asyncio
import grpc
import logging
from unittest.mock import patch, AsyncMock, MagicMock
from datetime import datetime, timezone

from analysis_engine.core.main import create_app
from analysis_engine.config import get_settings
from uvicorn import Server, Config as UvicornConfig

from generated_protos import analysis_engine_pb2, analysis_engine_pb2_grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc
from google.protobuf.timestamp_pb2 import Timestamp

# Import interfaces for mocking
from common_lib.interfaces.trading_gateway import ITradingGateway
from common_lib.interfaces.feature_store import IFeatureProvider
from common_lib.interfaces.ml_integration import IMLModelRegistry
from common_lib.interfaces.risk_management import IRiskManager

# Mock common_lib models (copied from unit tests for consistency)
class MockMarketDataPydanticModel:
    def __init__(self, data):
        self.data = data
class MockAccountInfoPydanticModel:
    def __init__(self, balance):
        self.balance = balance
class MockFeatureInfoModel:
    def __init__(self, name, description=""):
        self.name = name
        self.description = description
class MockModelMetadataModel:
    def __init__(self, id, name, symbol, timeframe):
        self.id = id
        self.name = name
        self.symbol = symbol
        self.timeframe = timeframe


logger = logging.getLogger(__name__)
server_task = None
app_instance = None

@pytest.fixture(scope="module")
def event_loop():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    loop.close()

@pytest.fixture(scope="module")
def mock_adapters():
    """Provides a dictionary of mock adapters."""
    mocks = {
        ITradingGateway: AsyncMock(spec=ITradingGateway),
        IFeatureProvider: AsyncMock(spec=IFeatureProvider),
        IMLModelRegistry: AsyncMock(spec=IMLModelRegistry),
        IRiskManager: AsyncMock(spec=IRiskManager),
    }
    # Reset mocks before each use if this fixture were function-scoped
    # For module scope, they are created once. Tests should ensure specific return_values.
    return mocks

async def start_server_with_mocks(mock_adapter_instances):
    global app_instance
    
    # This function will be patched into service_dependencies to return our mocks
    def mock_get_adapter(interface_type):
        logger.info(f"Mock factory's get_adapter called for {interface_type}")
        if interface_type in mock_adapter_instances:
            return mock_adapter_instances[interface_type]
        raise TypeError(f"Unknown interface type for mock factory: {interface_type}")

    mock_factory = MagicMock()
    mock_factory.get_adapter.side_effect = mock_get_adapter
    
    # Patch where get_common_adapter_factory is looked up by service_dependencies.py
    with patch('analysis_engine.core.service_dependencies.get_common_adapter_factory', return_value=mock_factory):
        # Import service_dependencies *after* patching, or re-import/reload if already imported at module level
        # For simplicity here, assuming it's fetched fresh during create_app() or ServiceDependencies init.
        # If service_dependencies.py has already initialized its `adapter_factory` at import time,
        # more complex patching of the `service_dependencies.adapter_factory` instance itself might be needed.
        
        # Re-initialize the singleton with the patched factory if possible, or patch the singleton's factory.
        # The current structure initializes `service_dependencies` singleton on import.
        # So, we need to patch the `adapter_factory` attribute of that specific instance.
        from analysis_engine.core.service_dependencies import service_dependencies
        original_factory = service_dependencies.adapter_factory
        service_dependencies.adapter_factory = mock_factory
        
        app_instance = create_app() # create_app will now use the patched factory via service_dependencies
        
        settings = get_settings()
        uvicorn_config = UvicornConfig(app=app_instance, host=settings.HOST, port=settings.PORT, log_level="warning")
        server = Server(uvicorn_config)
        logger.info(f"Starting Uvicorn server with MOCKED adapters for integration tests on {settings.HOST}:{settings.PORT}")
        
        try:
            await server.serve()
        finally:
            # Restore original factory after server stops
            service_dependencies.adapter_factory = original_factory


@pytest.fixture(scope="module", autouse=True)
async def manage_server(event_loop: asyncio.AbstractEventLoop, mock_adapters):
    global server_task
    logger.info("Setting up server with MOCKED adapters for module...")
    server_task = event_loop.create_task(start_server_with_mocks(mock_adapters))
    await asyncio.sleep(5) # Allow server to start
    logger.info("Server with MOCKED adapters setup complete.")
    yield
    logger.info("Tearing down server with MOCKED adapters...")
    if server_task and not server_task.done():
        server_task.cancel()
        try:
            await server_task
        except asyncio.CancelledError:
            logger.info("Mocked Server task cancelled successfully.")
        except Exception as e:
            logger.error(f"Error during mocked server task cancellation: {e}", exc_info=True)
    logger.info("Mocked Server teardown complete.")

@pytest.fixture(scope="module") # Keep grpc_channel module-scoped
async def grpc_channel():
    settings = get_settings()
    target_address = f"{settings.HOST}:{settings.GRPC_PORT}"
    logger.info(f"Creating gRPC channel to {target_address} for module tests.")
    channel = grpc.aio.insecure_channel(target_address)
    yield channel
    logger.info("Closing module-scoped gRPC channel.")
    await channel.close()

@pytest.fixture(autouse=True) # Reset mocks for each test function
async def reset_adapter_mocks(mock_adapters):
    for adapter_mock in mock_adapters.values():
        adapter_mock.reset_mock() # Resets call counts, side_effects, etc.
        # Re-establish async nature of methods if reset_mock clears it
        for name, _ in adapter_mock.method_calls: # This might not be needed if spec preserves it
            method = getattr(adapter_mock, name)
            if not isinstance(method, AsyncMock): # Or check if it's a MagicMock that needs to be an AsyncMock
                 setattr(adapter_mock, name, AsyncMock(name=name))


@pytest.mark.asyncio
async def test_grpc_health_check(grpc_channel: grpc.aio.Channel): # This test doesn't need mocks
    logger.info("Running test_grpc_health_check...")
    health_stub = health_pb2_grpc.HealthStub(grpc_channel)
    
    try:
        request_empty = health_pb2.HealthCheckRequest(service="")
        response_empty = await health_stub.Check(request_empty, timeout=5)
        assert response_empty.status == health_pb2.HealthCheckResponse.SERVING
        logger.info("Overall server health check PASSED.")

        service_name = analysis_engine_pb2.DESCRIPTOR.services_by_name['AnalysisEngine'].full_name
        request_service = health_pb2.HealthCheckRequest(service=service_name)
        response_service = await health_stub.Check(request_service, timeout=5)
        assert response_service.status == health_pb2.HealthCheckResponse.SERVING
        logger.info(f"Health check for '{service_name}' PASSED.")
    except grpc.aio.AioRpcError as e:
        pytest.fail(f"gRPC Health Check failed: {e.code()} - {e.details()}")

@pytest.mark.asyncio
async def test_get_market_overview_with_mocks(grpc_channel: grpc.aio.Channel, mock_adapters):
    logger.info("Running test_get_market_overview_with_mocks...")
    
    # Configure mock adapter returns
    current_dt = datetime.now(timezone.utc)
    mock_adapters[ITradingGateway].get_market_data.return_value = MockMarketDataPydanticModel(
        data=[{"open": 1.23, "high": 1.24, "low": 1.22, "close": 1.235, "volume": 1200, "timestamp": current_dt}]
    )
    mock_adapters[IFeatureProvider].get_available_features.return_value = [
        MockFeatureInfoModel(name="SMA_100"), MockFeatureInfoModel(name="EMA_50")
    ]
    mock_adapters[IMLModelRegistry].list_models.return_value = [
        MockModelMetadataModel(id="m1", name="TestModel1", symbol="EURUSD_INT", timeframe="H4")
    ]

    stub = analysis_engine_pb2_grpc.AnalysisEngineStub(grpc_channel)
    request = analysis_engine_pb2.GetMarketOverviewRequest(
        symbol="EURUSD_INT", timeframe="H4", lookback_days=10
    )
    
    response = await stub.GetMarketOverview(request, timeout=10)

    assert isinstance(response, analysis_engine_pb2.MarketOverviewResponse)
    assert response.market_data.open == 1.23
    assert response.market_data.close == 1.235
    expected_ts = Timestamp()
    expected_ts.FromDatetime(current_dt)
    assert response.market_data.timestamp == expected_ts
    
    # Assertions based on the actual logic in servicer that processes these mocks
    assert "SMA_100" in response.technical_indicators # From mocked get_available_features
    assert "RSI_14_placeholder" in response.technical_indicators # Default placeholder from servicer
    assert "TestModel1" in response.model_predictions # From mocked list_models

    mock_adapters[ITradingGateway].get_market_data.assert_called_once()
    mock_adapters[IFeatureProvider].get_available_features.assert_called_once()
    mock_adapters[IMLModelRegistry].list_models.assert_called_once()

@pytest.mark.asyncio
async def test_get_trading_opportunity_with_mocks(grpc_channel: grpc.aio.Channel, mock_adapters):
    logger.info("Running test_get_trading_opportunity_with_mocks...")

    # Configure mock adapter returns
    mock_adapters[ITradingGateway].get_account_info.return_value = MockAccountInfoPydanticModel(balance=75000.0)
    current_dt = datetime.now(timezone.utc)
    mock_adapters[ITradingGateway].get_market_data.return_value = MockMarketDataPydanticModel(
        data=[{"close": 1.5000, "timestamp": current_dt}]
    )
    mock_adapters[IRiskManager].calculate_position_size.return_value = {
        "position_size": 0.75, "risk_reward_ratio": 2.2
    }
    # For this test, let one model be "bullish" (ID ending with 1) and one "neutral/bearish"
    mock_adapters[IMLModelRegistry].list_models.return_value = [
        MockModelMetadataModel(id="oppModel1", name="OpportunitySeeker", symbol="AUDCAD_INT", timeframe="H1"),
        MockModelMetadataModel(id="oppModel2", name="RiskAvoider", symbol="AUDCAD_INT", timeframe="H1") 
    ]

    stub = analysis_engine_pb2_grpc.AnalysisEngineStub(grpc_channel)
    request = analysis_engine_pb2.GetTradingOpportunityRequest(
        symbol="AUDCAD_INT", timeframe="H1", account_id="int_test_acc", risk_percentage=1.0
    )

    response = await stub.GetTradingOpportunity(request, timeout=10)

    assert isinstance(response, analysis_engine_pb2.TradingOpportunityResponse)
    assert response.account_balance == 75000.0
    assert response.entry_price == 1.5000
    assert response.position_size == 0.75
    assert response.risk_details.risk_reward_ratio == 2.2
    # oppModel1 is bullish, oppModel2 is bearish (due to ID not ending in "1") -> tie -> SELL
    assert response.trade_direction == analysis_engine_pb2.OrderSide.SELL 
    assert "OpportunitySeeker" in response.model_predictions
    assert "RiskAvoider" in response.model_predictions
    
    mock_adapters[ITradingGateway].get_account_info.assert_called_once_with("int_test_acc")
    mock_adapters[IRiskManager].calculate_position_size.assert_called_once()
