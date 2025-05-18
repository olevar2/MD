import pytest
import grpc
import grpc.aio as aio
from unittest.mock import AsyncMock, MagicMock

# Import generated protobuf code
from common_lib.grpc.market_analysis import market_analysis_service_pb2
from common_lib.grpc.market_analysis import market_analysis_service_pb2_grpc

# Import the gRPC servicer and the service it depends on
from market_analysis_service.grpc_server.market_analysis_grpc_server import MarketAnalysisGrpcServicer
from market_analysis_service.services.market_analysis_service import MarketAnalysisService
from market_analysis_service.models.market_analysis_models import MarketAnalysisResponse, AnalysisResult, AnalysisType

# Mock the MarketAnalysisService
@pytest.fixture
def mock_market_analysis_service():
    return AsyncMock(spec=MarketAnalysisService)

# Fixture for the gRPC servicer
@pytest.fixture
def grpc_servicer(mock_market_analysis_service):
    return MarketAnalysisGrpcServicer(market_analysis_service=mock_market_analysis_service)

# Fixture for a mock gRPC context
@pytest.fixture
def mock_context():
    context = MagicMock()
    context.set_code = MagicMock()
    context.set_details = MagicMock()
    return context

@pytest.mark.asyncio
async def test_calculate_indicators_success(grpc_servicer, mock_market_analysis_service, mock_context):
    # Arrange
    grpc_request = market_analysis_service_pb2.CalculateIndicatorsRequest(
        symbol="AAPL",
        timeframe="1D",
        start_date="2023-01-01",
        end_date="2023-01-31"
    )

    # Prepare a mock response from the underlying service
    mock_indicators_data = {
        "SMA": [150.0, 151.0, 152.0],
        "RSI": [70.0, 65.0, 68.0]
    }
    mock_analysis_result = AnalysisResult(
        analysis_type=AnalysisType.TECHNICAL,
        result={"indicators": mock_indicators_data},
        confidence=1.0,
        execution_time_ms=50
    )
    mock_service_response = MarketAnalysisResponse(
        request_id="test-req-123",
        symbol="AAPL",
        timeframe="1D",
        start_date="2023-01-01",
        end_date="2023-01-31",
        analysis_results=[mock_analysis_result],
        total_execution_time_ms=50,
        timestamp="2023-01-31T12:00:00Z"
    )

    mock_market_analysis_service.analyze_market.return_value = mock_service_response

    # Act
    grpc_response = await grpc_servicer.CalculateIndicators(grpc_request, mock_context)

    # Assert
    mock_market_analysis_service.analyze_market.assert_called_once()
    assert grpc_response.request_id == "test-req-123"
    assert grpc_response.symbol == "AAPL"
    assert grpc_response.timeframe == "1D"
    assert grpc_response.start_date == "2023-01-01"
    assert grpc_response.end_date == "2023-01-31T12:00:00Z"
    assert grpc_response.indicators == mock_indicators_data
    assert grpc_response.execution_time_ms == 50
    assert grpc_response.timestamp == "2023-01-31T12:00:00Z"
    mock_context.set_code.assert_not_called()
    mock_context.set_details.assert_not_called()

@pytest.mark.asyncio
async def test_calculate_indicators_service_error(grpc_servicer, mock_market_analysis_service, mock_context):
    # Arrange
    grpc_request = market_analysis_service_pb2.CalculateIndicatorsRequest(
        symbol="AAPL",
        timeframe="1D",
        start_date="2023-01-01",
        end_date="2023-01-31"
    )

    # Simulate an exception from the underlying service
    mock_market_analysis_service.analyze_market.side_effect = Exception("Service failed")

    # Act
    grpc_response = await grpc_servicer.CalculateIndicators(grpc_request, mock_context)

    # Assert
    mock_market_analysis_service.analyze_market.assert_called_once()
    mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
    mock_context.set_details.assert_called_once_with("Internal server error: Service failed")
    # Ensure an empty response is returned on error
    assert isinstance(grpc_response, market_analysis_service_pb2.CalculateIndicatorsResponse)
    assert not grpc_response.indicators # Check if indicators are empty

@pytest.mark.asyncio
async def test_calculate_indicators_no_technical_results(grpc_servicer, mock_market_analysis_service, mock_context):
    # Arrange
    grpc_request = market_analysis_service_pb2.CalculateIndicatorsRequest(
        symbol="AAPL",
        timeframe="1D",
        start_date="2023-01-01",
        end_date="2023-01-31"
    )

    # Prepare a mock response from the underlying service *without* technical analysis results
    mock_analysis_result = AnalysisResult(
        analysis_type=AnalysisType.PATTERN,
        result={"patterns": []},
        confidence=1.0,
        execution_time_ms=50
    )
    mock_service_response = MarketAnalysisResponse(
        request_id="test-req-123",
        symbol="AAPL",
        timeframe="1D",
        start_date="2023-01-01",
        end_date="2023-01-31",
        analysis_results=[mock_analysis_result],
        total_execution_time_ms=50,
        timestamp="2023-01-31T12:00:00Z"
    )

    mock_market_analysis_service.analyze_market.return_value = mock_service_response

    # Act
    grpc_response = await grpc_servicer.CalculateIndicators(grpc_request, mock_context)

    # Assert
    mock_market_analysis_service.analyze_market.assert_called_once()
    mock_context.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)
    mock_context.set_details.assert_called_once_with("Technical analysis results not found")
    # Ensure an empty response is returned
    assert isinstance(grpc_response, market_analysis_service_pb2.CalculateIndicatorsResponse)
    assert not grpc_response.indicators # Check if indicators are empty

@pytest.mark.asyncio
async def test_calculate_indicators_empty_indicators_in_result(grpc_servicer, mock_market_analysis_service, mock_context):
    # Arrange
    grpc_request = market_analysis_service_pb2.CalculateIndicatorsRequest(
        symbol="AAPL",
        timeframe="1D",
        start_date="2023-01-01",
        end_date="2023-01-31"
    )

    # Prepare a mock response with technical analysis result but no 'indicators' key
    mock_analysis_result = AnalysisResult(
        analysis_type=AnalysisType.TECHNICAL,
        result={},
        confidence=1.0,
        execution_time_ms=50
    )
    mock_service_response = MarketAnalysisResponse(
        request_id="test-req-123",
        symbol="AAPL",
        timeframe="1D",
        start_date="2023-01-01",
        end_date="2023-01-31",
        analysis_results=[mock_analysis_result],
        total_execution_time_ms=50,
        timestamp="2023-01-31T12:00:00Z"
    )

    mock_market_analysis_service.analyze_market.return_value = mock_service_response

    # Act
    grpc_response = await grpc_servicer.CalculateIndicators(grpc_request, mock_context)

    # Assert
    mock_market_analysis_service.analyze_market.assert_called_once()
    mock_context.set_code.assert_called_once_with(grpc.StatusCode.NOT_FOUND)
    mock_context.set_details.assert_called_once_with("Technical analysis results not found")
    # Ensure an empty response is returned
    assert isinstance(grpc_response, market_analysis_service_pb2.CalculateIndicatorsResponse)
    assert not grpc_response.indicators # Check if indicators are empty

@pytest.mark.asyncio
async def test_detect_patterns_success(grpc_servicer, mock_market_analysis_service, mock_context):
    # Arrange
    grpc_request = market_analysis_service_pb2.DetectPatternsRequest(
        symbol="EURUSD",
        timeframe="1h",
        start_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 1).timestamp())),
        end_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 31).timestamp())),
        pattern_types=["HEAD_AND_SHOULDERS"],
        min_strength=0.7
    )

    # Prepare a mock response from the underlying service
    mock_patterns_data = {
        "HEAD_AND_SHOULDERS": [
            {"start_timestamp": "2023-01-10T00:00:00+00:00", "end_timestamp": "2023-01-15T00:00:00+00:00", "confidence": 0.8},
            {"start_timestamp": "2023-01-20T00:00:00+00:00", "end_timestamp": "2023-01-25T00:00:00+00:00", "confidence": 0.9}
        ]
    }
    mock_analysis_result = AnalysisResult(
        analysis_type=AnalysisType.PATTERN,
        result={'patterns': mock_patterns_data},
        confidence=1.0,
        execution_time_ms=60
    )
    mock_service_response = MarketAnalysisResponse(
        request_id="test-req-456",
        symbol="EURUSD",
        timeframe="1h",
        start_date="2023-01-01T00:00:00+00:00",
        end_date="2023-01-31T00:00:00+00:00",
        analysis_results=[mock_analysis_result],
        total_execution_time_ms=60,
        timestamp="2023-01-31T12:00:00Z"
    )

    mock_market_analysis_service.analyze_market.return_value = mock_service_response

    # Act
    grpc_response = await grpc_servicer.DetectPatterns(grpc_request, mock_context)

    # Assert
    mock_market_analysis_service.analyze_market.assert_called_once()
    assert grpc_response.patterns[0].name == "HEAD_AND_SHOULDERS"
    assert len(grpc_response.patterns[0].occurrences) == 2
    assert grpc_response.patterns[0].occurrences[0].confidence == 0.8
    mock_context.set_code.assert_not_called()
    mock_context.set_details.assert_not_called()

@pytest.mark.asyncio
async def test_detect_patterns_service_error(grpc_servicer, mock_market_analysis_service, mock_context):
    # Arrange
    grpc_request = market_analysis_service_pb2.DetectPatternsRequest(
        symbol="EURUSD",
        timeframe="1h",
        start_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 1).timestamp())),
        end_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 31).timestamp()))
    )

    # Simulate an exception from the underlying service
    mock_market_analysis_service.analyze_market.side_effect = Exception("Pattern service failed")

    # Act
    grpc_response = await grpc_servicer.DetectPatterns(grpc_request, mock_context)

    # Assert
    mock_market_analysis_service.analyze_market.assert_called_once()
    mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
    mock_context.set_details.assert_called_once_with("Internal server error: Pattern service failed")
    assert not grpc_response.patterns

@pytest.mark.asyncio
async def test_detect_support_resistance_success(grpc_servicer, mock_market_analysis_service, mock_context):
    # Arrange
    grpc_request = market_analysis_service_pb2.DetectSupportResistanceRequest(
        symbol="GBPUSD",
        timeframe="4h",
        start_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 1).timestamp())),
        end_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 31).timestamp())),
        method="PRICE_SWINGS",
        min_strength=0.6
    )

    # Prepare a mock response from the underlying service
    mock_sr_data = {
        "support": [
            {"value": 1.2000, "type": "SUPPORT", "strength": 0.8},
            {"value": 1.1950, "type": "SUPPORT", "strength": 0.7}
        ],
        "resistance": [
            {"value": 1.2100, "type": "RESISTANCE", "strength": 0.9}
        ]
    }
    mock_analysis_result = AnalysisResult(
        analysis_type=AnalysisType.SUPPORT_RESISTANCE,
        result={'support_resistance': mock_sr_data},
        confidence=1.0,
        execution_time_ms=70
    )
    mock_service_response = MarketAnalysisResponse(
        request_id="test-req-789",
        symbol="GBPUSD",
        timeframe="4h",
        start_date="2023-01-01T00:00:00+00:00",
        end_date="2023-01-31T00:00:00+00:00",
        analysis_results=[mock_analysis_result],
        total_execution_time_ms=70,
        timestamp="2023-01-31T12:00:00Z"
    )

    mock_market_analysis_service.analyze_market.return_value = mock_service_response

    # Act
    grpc_response = await grpc_servicer.DetectSupportResistance(grpc_request, mock_context)

    # Assert
    mock_market_analysis_service.analyze_market.assert_called_once()
    assert len(grpc_response.support_levels) == 2
    assert grpc_response.support_levels[0].value == 1.2000
    assert len(grpc_response.resistance_levels) == 1
    assert grpc_response.resistance_levels[0].value == 1.2100
    mock_context.set_code.assert_not_called()
    mock_context.set_details.assert_not_called()

@pytest.mark.asyncio
async def test_detect_support_resistance_service_error(grpc_servicer, mock_market_analysis_service, mock_context):
    # Arrange
    grpc_request = market_analysis_service_pb2.DetectSupportResistanceRequest(
        symbol="GBPUSD",
        timeframe="4h",
        start_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 1).timestamp())),
        end_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 31).timestamp()))
    )

    # Simulate an exception from the underlying service
    mock_market_analysis_service.analyze_market.side_effect = Exception("SR service failed")

    # Act
    grpc_response = await grpc_servicer.DetectSupportResistance(grpc_request, mock_context)

    # Assert
    mock_market_analysis_service.analyze_market.assert_called_once()
    mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
    mock_context.set_details.assert_called_once_with("Internal server error: SR service failed")
    assert not grpc_response.support_levels
    assert not grpc_response.resistance_levels

@pytest.mark.asyncio
async def test_detect_market_regime_success(grpc_servicer, mock_market_analysis_service, mock_context):
    # Arrange
    grpc_request = market_analysis_service_pb2.DetectMarketRegimeRequest(
        symbol="USDJPY",
        timeframe="1D",
        start_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 1).timestamp())),
        end_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 31).timestamp())),
        method="ATR"
    )

    # Prepare a mock response from the underlying service
    mock_regime_data = {
        "current_regime": "TRENDING",
        "regime_history": [
            {"regime": "RANGING", "start_timestamp": "2023-01-01T00:00:00+00:00", "end_timestamp": "2023-01-15T00:00:00+00:00"},
            {"regime": "TRENDING", "start_timestamp": "2023-01-16T00:00:00+00:00", "end_timestamp": "2023-01-31T00:00:00+00:00"}
        ]
    }
    mock_analysis_result = AnalysisResult(
        analysis_type=AnalysisType.MARKET_REGIME,
        result={'market_regime': mock_regime_data},
        confidence=1.0,
        execution_time_ms=80
    )
    mock_service_response = MarketAnalysisResponse(
        request_id="test-req-abc",
        symbol="USDJPY",
        timeframe="1D",
        start_date="2023-01-01T00:00:00+00:00",
        end_date="2023-01-31T00:00:00+00:00",
        analysis_results=[mock_analysis_result],
        total_execution_time_ms=80,
        timestamp="2023-01-31T12:00:00Z"
    )

    mock_market_analysis_service.analyze_market.return_value = mock_service_response

    # Act
    grpc_response = await grpc_servicer.DetectMarketRegime(grpc_request, mock_context)

    # Assert
    mock_market_analysis_service.analyze_market.assert_called_once()
    assert grpc_response.current_regime == common_types_pb2.MarketRegime.TRENDING
    assert len(grpc_response.regime_history) == 2
    assert grpc_response.regime_history[0].regime == common_types_pb2.MarketRegime.RANGING
    mock_context.set_code.assert_not_called()
    mock_context.set_details.assert_not_called()

@pytest.mark.asyncio
async def test_detect_market_regime_service_error(grpc_servicer, mock_market_analysis_service, mock_context):
    # Arrange
    grpc_request = market_analysis_service_pb2.DetectMarketRegimeRequest(
        symbol="USDJPY",
        timeframe="1D",
        start_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 1).timestamp())),
        end_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 31).timestamp()))
    )

    # Simulate an exception from the underlying service
    mock_market_analysis_service.analyze_market.side_effect = Exception("Regime service failed")

    # Act
    grpc_response = await grpc_servicer.DetectMarketRegime(grpc_request, mock_context)

    # Assert
    mock_market_analysis_service.analyze_market.assert_called_once()
    mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
    mock_context.set_details.assert_called_once_with("Internal server error: Regime service failed")
    assert grpc_response.current_regime == common_types_pb2.MarketRegime.UNKNOWN_MARKET_REGIME # Default value
    assert not grpc_response.regime_history

@pytest.mark.asyncio
async def test_perform_correlation_analysis_success(grpc_servicer, mock_market_analysis_service, mock_context):
    # Arrange
    grpc_request = market_analysis_service_pb2.PerformCorrelationAnalysisRequest(
        symbols=[common_types_pb2.Symbol(name="EURUSD"), common_types_pb2.Symbol(name="GBPUSD")],
        timeframe="1D",
        start_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 1).timestamp())),
        end_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 31).timestamp()))
    )

    # Prepare a mock response from the underlying service
    mock_correlation_data = {
        "EURUSD-GBPUSD": 0.85
    }
    mock_analysis_result = AnalysisResult(
        analysis_type=AnalysisType.CORRELATION,
        result={'correlations': mock_correlation_data},
        confidence=1.0,
        execution_time_ms=90
    )
    mock_service_response = MarketAnalysisResponse(
        request_id="test-req-def",
        symbol="EURUSD", # Assuming primary symbol is used here
        timeframe="1D",
        start_date="2023-01-01T00:00:00+00:00",
        end_date="2023-01-31T00:00:00+00:00",
        analysis_results=[mock_analysis_result],
        total_execution_time_ms=90,
        timestamp="2023-01-31T12:00:00Z"
    )

    mock_market_analysis_service.analyze_market.return_value = mock_service_response

    # Act
    grpc_response = await grpc_servicer.PerformCorrelationAnalysis(grpc_request, mock_context)

    # Assert
    mock_market_analysis_service.analyze_market.assert_called_once()
    assert "EURUSD-GBPUSD" in grpc_response.correlations
    assert grpc_response.correlations["EURUSD-GBPUSD"] == 0.85
    mock_context.set_code.assert_not_called()
    mock_context.set_details.assert_not_called()

@pytest.mark.asyncio
async def test_perform_correlation_analysis_service_error(grpc_servicer, mock_market_analysis_service, mock_context):
    # Arrange
    grpc_request = market_analysis_service_pb2.PerformCorrelationAnalysisRequest(
        symbols=[common_types_pb2.Symbol(name="EURUSD"), common_types_pb2.Symbol(name="GBPUSD")],
        timeframe="1D",
        start_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 1).timestamp())),
        end_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 31).timestamp()))
    )

    # Simulate an exception from the underlying service
    mock_market_analysis_service.analyze_market.side_effect = Exception("Correlation service failed")

    # Act
    grpc_response = await grpc_servicer.PerformCorrelationAnalysis(grpc_request, mock_context)

    # Assert
    mock_market_analysis_service.analyze_market.assert_called_once()
    mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
    mock_context.set_details.assert_called_once_with("Internal server error: Correlation service failed")
    assert not grpc_response.correlations

@pytest.mark.asyncio
async def test_perform_volatility_analysis_success(grpc_servicer, mock_market_analysis_service, mock_context):
    # Arrange
    grpc_request = market_analysis_service_pb2.PerformVolatilityAnalysisRequest(
        symbol="AUDCAD",
        timeframe="30m",
        start_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 1).timestamp())),
        end_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 31).timestamp()))
    )

    # Prepare a mock response from the underlying service
    mock_volatility_data = {
        "current_volatility": 0.005,
        "volatility_history": [
            {"timestamp": "2023-01-10T00:00:00+00:00", "value": 0.004},
            {"timestamp": "2023-01-20T00:00:00+00:00", "value": 0.006}
        ]
    }
    mock_analysis_result = AnalysisResult(
        analysis_type=AnalysisType.VOLATILITY,
        result={'volatility': mock_volatility_data},
        confidence=1.0,
        execution_time_ms=100
    )
    mock_service_response = MarketAnalysisResponse(
        request_id="test-req-ghi",
        symbol="AUDCAD",
        timeframe="30m",
        start_date="2023-01-01T00:00:00+00:00",
        end_date="2023-01-31T00:00:00+00:00",
        analysis_results=[mock_analysis_result],
        total_execution_time_ms=100,
        timestamp="2023-01-31T12:00:00Z"
    )

    mock_market_analysis_service.analyze_market.return_value = mock_service_response

    # Act
    grpc_response = await grpc_servicer.PerformVolatilityAnalysis(grpc_request, mock_context)

    # Assert
    mock_market_analysis_service.analyze_market.assert_called_once()
    assert grpc_response.current_volatility == 0.005
    assert len(grpc_response.volatility_history) == 2
    assert grpc_response.volatility_history[0].value == 0.004
    mock_context.set_code.assert_not_called()
    mock_context.set_details.assert_not_called()

@pytest.mark.asyncio
async def test_perform_volatility_analysis_service_error(grpc_servicer, mock_market_analysis_service, mock_context):
    # Arrange
    grpc_request = market_analysis_service_pb2.PerformVolatilityAnalysisRequest(
        symbol="AUDCAD",
        timeframe="30m",
        start_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 1).timestamp())),
        end_date=market_analysis_service_pb2.Timestamp(seconds=int(datetime(2023, 1, 31).timestamp()))
    )

    # Simulate an exception from the underlying service
    mock_market_analysis_service.analyze_market.side_effect = Exception("Volatility service failed")

    # Act
    grpc_response = await grpc_servicer.PerformVolatilityAnalysis(grpc_request, mock_context)

    # Assert
    mock_market_analysis_service.analyze_market.assert_called_once()
    mock_context.set_code.assert_called_once_with(grpc.StatusCode.INTERNAL)
    mock_context.set_details.assert_called_once_with("Internal server error: Volatility service failed")
    assert grpc_response.current_volatility == 0.0 # Default value
    assert not grpc_response.volatility_history