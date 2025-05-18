"""
Tests for the Market Analysis Service gRPC server.
"""

import pytest
import grpc
import asyncio
from unittest.mock import AsyncMock, MagicMock
from datetime import datetime, timezone

# Import generated protobuf stubs
from common_lib.grpc.market_analysis import market_analysis_service_pb2
from common_lib.grpc.market_analysis import market_analysis_service_pb2_grpc
from common_lib.grpc.common import common_types_pb2
from common_lib.grpc.common import error_types_pb2

# Import the servicer and core service models
from market_analysis_service.grpc_server.market_analysis_grpc_server import MarketAnalysisServiceServicer
from market_analysis_service.models.market_analysis_models import MarketAnalysisRequest, AnalysisType, MarketAnalysisResponse, ErrorModel

# Helper function to create a dummy ServicerContext
class DummyServicerContext:
    def __init__(self):
        self._code = grpc.StatusCode.OK
        self._details = None

    def set_code(self, code):
        self._code = code

    def set_details(self, details):
        self._details = details

    def abort(self, code, details):
        self.set_code(code)
        self.set_details(details)
        raise grpc.RpcError(code, details)

    async def abort_with_status(self, code, details):
        self.abort(code, details)

    def invocation_metadata(self):
        return []

    def peer(self):
        return 'test_peer'

    def time_remaining(self):
        return None # Or a suitable timeout value

@pytest.fixture
def mock_core_service():
    mock = AsyncMock()
    return mock

@pytest.fixture
def servicer(mock_core_service):
    return MarketAnalysisServiceServicer(mock_core_service)

# TODO: Add tests for CalculateIndicators
# TODO: Add tests for DetectPatterns
# TODO: Add tests for DetectSupportResistance
# TODO: Add tests for DetectMarketRegime
# TODO: Add tests for PerformCorrelationAnalysis
# TODO: Add tests for PerformVolatilityAnalysis

@pytest.mark.asyncio
async def test_perform_analysis_success(servicer, mock_core_service):
    # Mock the core service response
    mock_core_response = MarketAnalysisResponse(
        analysis_id="test-analysis-id",
        result={"key": "value"},
        timestamp=datetime.now(timezone.utc),
        error=None
    )
    mock_core_service.perform_analysis.return_value = mock_core_response

    # Create a dummy gRPC request
    grpc_request = market_analysis_service_pb2.MarketAnalysisRequest(
        symbol="AAPL",
        analysis_type=market_analysis_service_pb2.AnalysisType.TECHNICAL,
        parameters={
            "indicator": "RSI",
            "period": "14"
        }
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.PerformAnalysis(grpc_request, context)

    # Assertions
    mock_core_service.perform_analysis.assert_called_once()
    assert grpc_response.analysis_id == "test-analysis-id"
    assert grpc_response.result == '{"key": "value"}' # JSON string representation
    assert grpc_response.timestamp != "" # Check if timestamp is populated
    assert not grpc_response.error.code # Check if error code is not set (indicating success)
    assert not grpc_response.error.message # Check if error message is not set
    assert context._code == grpc.StatusCode.OK
    assert context._details is None

@pytest.mark.asyncio
async def test_perform_analysis_core_service_error(servicer, mock_core_service):
    # Mock the core service response with an error
    mock_core_response = MarketAnalysisResponse(
        analysis_id="",
        result=None,
        timestamp=None,
        error=ErrorModel(code=1001, message="Core service failed")
    )
    mock_core_service.perform_analysis.return_value = mock_core_response

    # Create a dummy gRPC request
    grpc_request = market_analysis_service_pb2.MarketAnalysisRequest(
        symbol="MSFT",
        analysis_type=market_analysis_service_pb2.AnalysisType.FUNDAMENTAL,
        parameters={}
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.PerformAnalysis(grpc_request, context)

    # Assertions
    mock_core_service.perform_analysis.assert_called_once()
    assert grpc_response.analysis_id == ""
    assert grpc_response.result == "" # Result should be empty on error
    assert grpc_response.timestamp == "" # Timestamp should be empty on error
    assert grpc_response.error.code == 1001
    assert grpc_response.error.message == "Core service failed"
    assert context._code == grpc.StatusCode.OK # Context code should still be OK if error is handled internally
    assert context._details is None

@pytest.mark.asyncio
async def test_perform_analysis_unexpected_exception(servicer, mock_core_service):
    # Mock the core service to raise an unexpected exception
    mock_core_service.perform_analysis.side_effect = Exception("Simulated unexpected error")

    # Create a dummy gRPC request
    grpc_request = market_analysis_service_pb2.MarketAnalysisRequest(
        symbol="GOOG",
        analysis_type=market_analysis_service_pb2.AnalysisType.SENTIMENT,
        parameters={}
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method and expect an RpcError
    grpc_response = await servicer.PerformAnalysis(grpc_request, context)

    # Assertions
    mock_core_service.perform_analysis.assert_called_once()
    # Check the response error details
    assert grpc_response.error.code == grpc.StatusCode.INTERNAL.value[0]
    assert "An internal error occurred: Simulated unexpected error" in grpc_response.error.message
    # Check the context details set by the servicer
    assert context._code == grpc.StatusCode.INTERNAL
    assert "An internal error occurred: Simulated unexpected error" in context._details


# TODO: Add tests for CalculateIndicators
# TODO: Add tests for DetectPatterns
# TODO: Add tests for DetectSupportResistance
# TODO: Add tests for DetectMarketRegime
# TODO: Add tests for PerformCorrelationAnalysis
# TODO: Add tests for PerformVolatilityAnalysis

@pytest.mark.asyncio
async def test_calculate_indicators_success(servicer, mock_core_service):
    # Mock the core service response for CalculateIndicators
    mock_core_response = MarketAnalysisResponse(
        analysis_results=[{'indicators': {'RSI': {'2023-01-01T00:00:00+00:00': 70.0}}}],
        error=None
    )
    mock_core_service.analyze_market.return_value = mock_core_response

    # Create a dummy gRPC request for CalculateIndicators
    grpc_request = market_analysis_service_pb2.CalculateIndicatorsRequest(
        symbol=common_types_pb2.Symbol(name="AAPL"),
        timeframe=common_types_pb2.Timeframe(name="1D"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
        indicators=["RSI"],
        parameters={
            "period": "14"
        }
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.CalculateIndicators(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert len(grpc_response.indicators) == 1
    assert grpc_response.indicators[0].name == "RSI"
    # Check timestamp conversion and value
    expected_timestamp = int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())
    assert expected_timestamp in grpc_response.indicators[0].values
    assert grpc_response.indicators[0].values[expected_timestamp] == 70.0
    assert not grpc_response.error.code
    assert not grpc_response.error.message
    assert context._code == grpc.StatusCode.OK
    assert context._details is None

@pytest.mark.asyncio
async def test_calculate_indicators_core_service_error(servicer, mock_core_service):
    # Mock the core service response with an error
    mock_core_response = MarketAnalysisResponse(
        analysis_results=None,
        error=ErrorModel(code=1002, message="Indicator calculation failed")
    )
    mock_core_service.analyze_market.return_value = mock_core_response

    # Create a dummy gRPC request
    grpc_request = market_analysis_service_pb2.CalculateIndicatorsRequest(
        symbol=common_types_pb2.Symbol(name="MSFT"),
        timeframe=common_types_pb2.Timeframe(name="1H"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
        indicators=["MACD"],
        parameters={}
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.CalculateIndicators(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert len(grpc_response.indicators) == 0 # No indicators on error
    assert grpc_response.error.code == error_types_pb2.ErrorCode.INTERNAL_ERROR # Mapped error code
    assert grpc_response.error.message == "Indicator calculation failed"
    assert context._code == grpc.StatusCode.INTERNAL # Context code should reflect the error
    assert context._details == "Indicator calculation failed"

@pytest.mark.asyncio
async def test_calculate_indicators_unexpected_exception(servicer, mock_core_service):
    # Mock the core service to raise an unexpected exception
    mock_core_service.analyze_market.side_effect = Exception("Simulated indicator error")

    # Create a dummy gRPC request
    grpc_request = market_analysis_service_pb2.CalculateIndicatorsRequest(
        symbol=common_types_pb2.Symbol(name="GOOG"),
        timeframe=common_types_pb2.Timeframe(name="4H"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
        indicators=["BBANDS"],
        parameters={}
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.CalculateIndicators(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert len(grpc_response.indicators) == 0 # No indicators on error
    assert grpc_response.error.code == error_types_pb2.ErrorCode.INTERNAL_ERROR
    assert "Internal server error: Simulated indicator error" in grpc_response.error.message
    assert context._code == grpc.StatusCode.INTERNAL
    assert "Internal server error: Simulated indicator error" in context._details


# TODO: Add tests for DetectPatterns
# TODO: Add tests for DetectSupportResistance
# TODO: Add tests for DetectMarketRegime
# TODO: Add tests for PerformCorrelationAnalysis
# TODO: Add tests for PerformVolatilityAnalysis

@pytest.mark.asyncio
async def test_detect_patterns_success(servicer, mock_core_service):
    # Mock the core service response for DetectPatterns
    mock_core_response = MarketAnalysisResponse(
        analysis_results=[{'patterns': {'HeadAndShoulders': [{'start_timestamp': '2023-01-01T00:00:00+00:00', 'end_timestamp': '2023-01-10T00:00:00+00:00', 'confidence': 0.8}]}}],
        error=None
    )
    mock_core_service.analyze_market.return_value = mock_core_response

    # Create a dummy gRPC request for DetectPatterns
    grpc_request = market_analysis_service_pb2.DetectPatternsRequest(
        symbol=common_types_pb2.Symbol(name="MSFT"),
        timeframe=common_types_pb2.Timeframe(name="4H"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
        patterns=["HeadAndShoulders"]
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.DetectPatterns(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert len(grpc_response.patterns) == 1
    assert grpc_response.patterns[0].name == "HeadAndShoulders"
    assert len(grpc_response.patterns[0].occurrences) == 1
    occurrence = grpc_response.patterns[0].occurrences[0]
    assert occurrence.start_timestamp == int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())
    assert occurrence.end_timestamp == int(datetime(2023, 1, 10, tzinfo=timezone.utc).timestamp())
    assert occurrence.confidence == 0.8
    assert not grpc_response.error.code
    assert not grpc_response.error.message
    assert context._code == grpc.StatusCode.OK
    assert context._details is None

@pytest.mark.asyncio
async def test_detect_patterns_core_service_error(servicer, mock_core_service):
    # Mock the core service response with an error
    mock_core_response = MarketAnalysisResponse(
        analysis_results=None,
        error=ErrorModel(code=1003, message="Pattern detection failed")
    )
    mock_core_service.analyze_market.return_value = mock_core_response

    # Create a dummy gRPC request
    grpc_request = market_analysis_service_pb2.DetectPatternsRequest(
        symbol=common_types_pb2.Symbol(name="GOOG"),
        timeframe=common_types_pb2.Timeframe(name="1D"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
        patterns=["DoubleTop"]
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.DetectPatterns(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert len(grpc_response.patterns) == 0 # No patterns on error
    assert grpc_response.error.code == error_types_pb2.ErrorCode.INTERNAL_ERROR # Mapped error code
    assert grpc_response.error.message == "Pattern detection failed"
    assert context._code == grpc.StatusCode.INTERNAL # Context code should reflect the error
    assert context._details == "Pattern detection failed"

@pytest.mark.asyncio
async def test_detect_patterns_unexpected_exception(servicer, mock_core_service):
    # Mock the core service to raise an unexpected exception
    mock_core_service.analyze_market.side_effect = Exception("Simulated pattern error")

    # Create a dummy gRPC request
    grpc_request = market_analysis_service_pb2.DetectPatternsRequest(
        symbol=common_types_pb2.Symbol(name="AAPL"),
        timeframe=common_types_pb2.Timeframe(name="1H"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
        patterns=["Flag"]
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.DetectPatterns(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert len(grpc_response.patterns) == 0 # No patterns on error
    assert grpc_response.error.code == error_types_pb2.ErrorCode.INTERNAL_ERROR
    assert "Internal server error: Simulated pattern error" in grpc_response.error.message
    assert context._code == grpc.StatusCode.INTERNAL
    assert "Internal server error: Simulated pattern error" in context._details

# TODO: Add tests for DetectSupportResistance
# TODO: Add tests for DetectMarketRegime
# TODO: Add tests for PerformCorrelationAnalysis
# TODO: Add tests for PerformVolatilityAnalysis

@pytest.mark.asyncio
async def test_detect_support_resistance_success(servicer, mock_core_service):
    # Mock the core service response for DetectSupportResistance
    mock_core_response = MarketAnalysisResponse(
        analysis_results=[{'support_resistance': {'support': [{'value': 100.0, 'type': common_types_pb2.LevelType.HORIZONTAL, 'strength': 0.9}], 'resistance': [{'value': 110.0, 'type': common_types_pb2.LevelType.DIAGONAL, 'strength': 0.7}]}}],
        error=None
    )
    mock_core_service.analyze_market.return_value = mock_core_response

    # Create a dummy gRPC request for DetectSupportResistance
    grpc_request = market_analysis_service_pb2.DetectSupportResistanceRequest(
        symbol=common_types_pb2.Symbol(name="AAPL"),
        timeframe=common_types_pb2.Timeframe(name="1D"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.DetectSupportResistance(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert len(grpc_response.support_levels) == 1
    assert grpc_response.support_levels[0].value == 100.0
    assert grpc_response.support_levels[0].type == common_types_pb2.LevelType.HORIZONTAL
    assert grpc_response.support_levels[0].strength == 0.9
    assert len(grpc_response.resistance_levels) == 1
    assert grpc_response.resistance_levels[0].value == 110.0
    assert grpc_response.resistance_levels[0].type == common_types_pb2.LevelType.DIAGONAL
    assert grpc_response.resistance_levels[0].strength == 0.7
    assert not grpc_response.error.code
    assert not grpc_response.error.message
    assert context._code == grpc.StatusCode.OK
    assert context._details is None

@pytest.mark.asyncio
async def test_detect_support_resistance_core_service_error(servicer, mock_core_service):
    # Mock the core service response with an error
    mock_core_response = MarketAnalysisResponse(
        analysis_results=None,
        error=ErrorModel(code=1004, message="Support/resistance detection failed")
    )
    mock_core_service.analyze_market.return_value = mock_core_response

    # Create a dummy gRPC request
    grpc_request = market_analysis_service_pb2.DetectSupportResistanceRequest(
        symbol=common_types_pb2.Symbol(name="MSFT"),
        timeframe=common_types_pb2.Timeframe(name="4H"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.DetectSupportResistance(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert len(grpc_response.support_levels) == 0
    assert len(grpc_response.resistance_levels) == 0
    assert grpc_response.error.code == error_types_pb2.ErrorCode.INTERNAL_ERROR
    assert grpc_response.error.message == "Support/resistance detection failed"
    assert context._code == grpc.StatusCode.INTERNAL
    assert context._details == "Support/resistance detection failed"

@pytest.mark.asyncio
async def test_detect_support_resistance_unexpected_exception(servicer, mock_core_service):
    # Mock the core service to raise an unexpected exception
    mock_core_service.analyze_market.side_effect = Exception("Simulated SR error")

    # Create a dummy gRPC request
    grpc_request = market_analysis_service_pb2.DetectSupportResistanceRequest(
        symbol=common_types_pb2.Symbol(name="GOOG"),
        timeframe=common_types_pb2.Timeframe(name="1H"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.DetectSupportResistance(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert len(grpc_response.support_levels) == 0
    assert len(grpc_response.resistance_levels) == 0
    assert grpc_response.error.code == error_types_pb2.ErrorCode.INTERNAL_ERROR
    assert "Internal server error: Simulated SR error" in grpc_response.error.message
    assert context._code == grpc.StatusCode.INTERNAL
    assert "Internal server error: Simulated SR error" in context._details

# TODO: Add tests for DetectMarketRegime
# TODO: Add tests for PerformCorrelationAnalysis
# TODO: Add tests for PerformVolatilityAnalysis

@pytest.mark.asyncio
async def test_detect_market_regime_success(servicer, mock_core_service):
    # Mock the core service response for DetectMarketRegime
    mock_core_response = MarketAnalysisResponse(
        analysis_results=[{'market_regime': {'regime': 'TRENDING', 'confidence': 0.95}}],
        error=None
    )
    mock_core_service.analyze_market.return_value = mock_core_response

    # Create a dummy gRPC request for DetectMarketRegime
    grpc_request = market_analysis_service_pb2.DetectMarketRegimeRequest(
        symbol=common_types_pb2.Symbol(name="GOOG"),
        timeframe=common_types_pb2.Timeframe(name="1D"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.DetectMarketRegime(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert grpc_response.regime == market_analysis_service_pb2.MarketRegimeType.TRENDING
    assert grpc_response.confidence == 0.95
    assert not grpc_response.error.code
    assert not grpc_response.error.message
    assert context._code == grpc.StatusCode.OK
    assert context._details is None

@pytest.mark.asyncio
async def test_detect_market_regime_core_service_error(servicer, mock_core_service):
    # Mock the core service response with an error
    mock_core_response = MarketAnalysisResponse(
        analysis_results=None,
        error=ErrorModel(code=1005, message="Market regime detection failed")
    )
    mock_core_service.analyze_market.return_value = mock_core_response

    # Create a dummy gRPC request
    grpc_request = market_analysis_service_pb2.DetectMarketRegimeRequest(
        symbol=common_types_pb2.Symbol(name="AAPL"),
        timeframe=common_types_pb2.Timeframe(name="4H"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.DetectMarketRegime(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert grpc_response.regime == market_analysis_service_pb2.MarketRegimeType.UNKNOWN_MARKET_REGIME
    assert grpc_response.confidence == 0.0
    assert grpc_response.error.code == error_types_pb2.ErrorCode.INTERNAL_ERROR
    assert grpc_response.error.message == "Market regime detection failed"
    assert context._code == grpc.StatusCode.INTERNAL
    assert context._details == "Market regime detection failed"

@pytest.mark.asyncio
async def test_detect_market_regime_unexpected_exception(servicer, mock_core_service):
    # Mock the core service to raise an unexpected exception
    mock_core_service.analyze_market.side_effect = Exception("Simulated regime error")

    # Create a dummy gRPC request
    grpc_request = market_analysis_service_pb2.DetectMarketRegimeRequest(
        symbol=common_types_pb2.Symbol(name="MSFT"),
        timeframe=common_types_pb2.Timeframe(name="1H"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.DetectMarketRegime(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert grpc_response.regime == market_analysis_service_pb2.MarketRegimeType.UNKNOWN_MARKET_REGIME
    assert grpc_response.confidence == 0.0
    assert grpc_response.error.code == error_types_pb2.ErrorCode.INTERNAL_ERROR
    assert "Internal server error: Simulated regime error" in grpc_response.error.message
    assert context._code == grpc.StatusCode.INTERNAL
    assert "Internal server error: Simulated regime error" in context._details


# TODO: Add tests for PerformCorrelationAnalysis
# TODO: Add tests for PerformVolatilityAnalysis

@pytest.mark.asyncio
async def test_perform_correlation_analysis_success(servicer, mock_core_service):
    # Mock the core service response for PerformCorrelationAnalysis
    mock_core_response = MarketAnalysisResponse(
        analysis_results=[{'correlation': {'AAPL-MSFT': 0.85}}],
        error=None
    )
    mock_core_service.analyze_market.return_value = mock_core_response

    # Create a dummy gRPC request for PerformCorrelationAnalysis
    grpc_request = market_analysis_service_pb2.PerformCorrelationAnalysisRequest(
        symbols=[common_types_pb2.Symbol(name="AAPL"), common_types_pb2.Symbol(name="MSFT")],
        timeframe=common_types_pb2.Timeframe(name="1D"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.PerformCorrelationAnalysis(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert len(grpc_response.correlations) == 1
    assert grpc_response.correlations[0].symbol_pair == "AAPL-MSFT"
    assert grpc_response.correlations[0].correlation_value == 0.85
    assert not grpc_response.error.code
    assert not grpc_response.error.message
    assert context._code == grpc.StatusCode.OK
    assert context._details is None

@pytest.mark.asyncio
async def test_perform_correlation_analysis_core_service_error(servicer, mock_core_service):
    # Mock the core service response with an error
    mock_core_response = MarketAnalysisResponse(
        analysis_results=None,
        error=ErrorModel(code=1006, message="Correlation analysis failed")
    )
    mock_core_service.analyze_market.return_value = mock_core_response

    # Create a dummy gRPC request
    grpc_request = market_analysis_service_pb2.PerformCorrelationAnalysisRequest(
        symbols=[common_types_pb2.Symbol(name="GOOG"), common_types_pb2.Symbol(name="MSFT")],
        timeframe=common_types_pb2.Timeframe(name="4H"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.PerformCorrelationAnalysis(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert len(grpc_response.correlations) == 0
    assert grpc_response.error.code == error_types_pb2.ErrorCode.INTERNAL_ERROR
    assert grpc_response.error.message == "Correlation analysis failed"
    assert context._code == grpc.StatusCode.INTERNAL
    assert context._details == "Correlation analysis failed"

@pytest.mark.asyncio
async def test_perform_correlation_analysis_unexpected_exception(servicer, mock_core_service):
    # Mock the core service to raise an unexpected exception
    mock_core_service.analyze_market.side_effect = Exception("Simulated correlation error")

    # Create a dummy gRPC request
    grpc_request = market_analysis_service_pb2.PerformCorrelationAnalysisRequest(
        symbols=[common_types_pb2.Symbol(name="AAPL"), common_types_pb2.Symbol(name="GOOG")],
        timeframe=common_types_pb2.Timeframe(name="1H"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.PerformCorrelationAnalysis(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert len(grpc_response.correlations) == 0
    assert grpc_response.error.code == error_types_pb2.ErrorCode.INTERNAL_ERROR
    assert "Internal server error: Simulated correlation error" in grpc_response.error.message
    assert context._code == grpc.StatusCode.INTERNAL
    assert "Internal server error: Simulated correlation error" in context._details


# TODO: Add tests for PerformVolatilityAnalysis

@pytest.mark.asyncio
async def test_perform_volatility_analysis_success(servicer, mock_core_service):
    # Mock the core service response for PerformVolatilityAnalysis
    mock_core_response = MarketAnalysisResponse(
        analysis_results=[{'volatility': {'AAPL': 0.015}}],
        error=None
    )
    mock_core_service.analyze_market.return_value = mock_core_response

    # Create a dummy gRPC request for PerformVolatilityAnalysis
    grpc_request = market_analysis_service_pb2.PerformVolatilityAnalysisRequest(
        symbol=common_types_pb2.Symbol(name="AAPL"),
        timeframe=common_types_pb2.Timeframe(name="1D"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.PerformVolatilityAnalysis(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert len(grpc_response.volatilities) == 1
    assert grpc_response.volatilities[0].symbol.name == "AAPL"
    assert grpc_response.volatilities[0].volatility_value == 0.015
    assert not grpc_response.error.code
    assert not grpc_response.error.message
    assert context._code == grpc.StatusCode.OK
    assert context._details is None

@pytest.mark.asyncio
async def test_perform_volatility_analysis_core_service_error(servicer, mock_core_service):
    # Mock the core service response with an error
    mock_core_response = MarketAnalysisResponse(
        analysis_results=None,
        error=ErrorModel(code=1007, message="Volatility analysis failed")
    )
    mock_core_service.analyze_market.return_value = mock_core_response

    # Create a dummy gRPC request
    grpc_request = market_analysis_service_pb2.PerformVolatilityAnalysisRequest(
        symbol=common_types_pb2.Symbol(name="MSFT"),
        timeframe=common_types_pb2.Timeframe(name="4H"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.PerformVolatilityAnalysis(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert len(grpc_response.volatilities) == 0
    assert grpc_response.error.code == error_types_pb2.ErrorCode.INTERNAL_ERROR
    assert grpc_response.error.message == "Volatility analysis failed"
    assert context._code == grpc.StatusCode.INTERNAL
    assert context._details == "Volatility analysis failed"

@pytest.mark.asyncio
async def test_perform_volatility_analysis_unexpected_exception(servicer, mock_core_service):
    # Mock the core service to raise an unexpected exception
    mock_core_service.analyze_market.side_effect = Exception("Simulated volatility error")

    # Create a dummy gRPC request
    grpc_request = market_analysis_service_pb2.PerformVolatilityAnalysisRequest(
        symbol=common_types_pb2.Symbol(name="GOOG"),
        timeframe=common_types_pb2.Timeframe(name="1H"),
        start_date=common_types_pb2.Timestamp(seconds=int(datetime(2023, 1, 1, tzinfo=timezone.utc).timestamp())),
    )

    # Create a dummy servicer context
    context = DummyServicerContext()

    # Call the servicer method
    grpc_response = await servicer.PerformVolatilityAnalysis(grpc_request, context)

    # Assertions
    mock_core_service.analyze_market.assert_called_once()
    assert len(grpc_response.volatilities) == 0
    assert grpc_response.error.code == error_types_pb2.ErrorCode.INTERNAL_ERROR
    assert "Internal server error: Simulated volatility error" in grpc_response.error.message
    assert context._code == grpc.StatusCode.INTERNAL
    assert "Internal server error: Simulated volatility error" in context._details