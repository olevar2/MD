"""
Tests for error handling in the analysis-engine-service.
"""
import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
import uuid
import json

from analysis_engine.core.exceptions_bridge import (
    AnalysisError,
    AnalyzerNotFoundError,
    InsufficientDataError,
    InvalidAnalysisParametersError,
    AnalysisTimeoutError,
    ServiceError,
    ConfigurationError,
    ServiceInitializationError,
    ServiceResolutionError,
    ServiceCleanupError,
    generate_correlation_id
)
from analysis_engine.core.service_container import ServiceContainer
from analysis_engine.services.analysis_service import AnalysisService
from analysis_engine.analysis.base_analyzer import BaseAnalyzer
from analysis_engine.models.analysis_result import AnalysisResult
from analysis_engine.models.market_data import MarketData


class TestServiceContainer:
    """Tests for the ServiceContainer error handling."""

    @pytest.fixture
    def service_container(self):
        """Create a service container for testing."""
        return ServiceContainer()

    @pytest.fixture
    def mock_service(self):
        """Create a mock service for testing."""
        service = MagicMock()
        service.cleanup = AsyncMock()
        return service

    @pytest.fixture
    def mock_factory(self):
        """Create a mock factory for testing."""
        async def factory(container):
            return MagicMock()
        return factory

    @pytest.fixture
    def failing_factory(self):
        """Create a factory that raises an exception."""
        async def factory(container):
            raise ValueError("Factory error")
        return factory

    async def test_register_factory_none_service_type(self, service_container, mock_factory):
        """Test register_factory with None service type."""
        with pytest.raises(ConfigurationError) as excinfo:
            service_container.register_factory(None, mock_factory)
        
        assert "Cannot register factory for None service type" in str(excinfo.value)
        assert excinfo.value.error_code == "INVALID_SERVICE_TYPE"

    async def test_register_factory_none_factory(self, service_container):
        """Test register_factory with None factory."""
        with pytest.raises(ConfigurationError) as excinfo:
            service_container.register_factory(MagicMock, None)
        
        assert "Factory for Mock must be a callable" in str(excinfo.value)
        assert excinfo.value.error_code == "INVALID_FACTORY"

    async def test_resolve_none_service_type(self, service_container):
        """Test resolve with None service type."""
        with pytest.raises(ServiceResolutionError) as excinfo:
            await service_container.resolve(None)
        
        assert "Cannot resolve None service type" in str(excinfo.value)
        assert excinfo.value.error_code == "INVALID_SERVICE_TYPE"

    async def test_resolve_unregistered_service(self, service_container):
        """Test resolve with unregistered service type."""
        with pytest.raises(ServiceResolutionError) as excinfo:
            await service_container.resolve(MagicMock)
        
        assert "No factory registered for Mock" in str(excinfo.value)
        assert excinfo.value.error_code == "FACTORY_NOT_FOUND"
        assert "available_services" in excinfo.value.details

    async def test_resolve_factory_error(self, service_container, failing_factory):
        """Test resolve with factory that raises an error."""
        service_container.register_factory(MagicMock, failing_factory)
        
        with pytest.raises(ServiceInitializationError) as excinfo:
            await service_container.resolve(MagicMock)
        
        assert "Failed to initialize service Mock" in str(excinfo.value)
        assert excinfo.value.error_code == "SERVICE_INITIALIZATION_FAILED"
        assert "Factory error" in str(excinfo.value.details.get("error", ""))

    async def test_resolve_optional_unregistered_service(self, service_container):
        """Test resolve_optional with unregistered service type."""
        result = await service_container.resolve_optional(MagicMock)
        assert result is None

    async def test_cleanup_service_error(self, service_container, mock_service):
        """Test cleanup with service that raises an error."""
        # Add a service that will raise an error during cleanup
        service_container._services[MagicMock] = mock_service
        mock_service.cleanup.side_effect = ValueError("Cleanup error")
        
        with pytest.raises(ServiceCleanupError) as excinfo:
            await service_container.cleanup()
        
        assert "Errors occurred during service cleanup" in str(excinfo.value)
        assert excinfo.value.error_code == "SERVICE_CLEANUP_FAILED"
        assert len(excinfo.value.details.get("cleanup_errors", [])) == 1


class MockAnalyzer(BaseAnalyzer):
    """Mock analyzer for testing."""
    
    def __init__(self, name="mock_analyzer", parameters=None):
        """Initialize the mock analyzer."""
        super().__init__(name, parameters)
    
    async def analyze(self, data):
        """Mock analyze method."""
        if getattr(data, "should_fail", False):
            raise ValueError("Analysis failed")
        
        if getattr(data, "should_timeout", False):
            await asyncio.sleep(10)  # This will trigger a timeout
            
        return AnalysisResult(
            analyzer_name=self.name,
            result_data={"result": "success"},
            is_valid=True,
            metadata={"symbol": getattr(data, "symbol", "unknown")}
        )


class TestBaseAnalyzer:
    """Tests for the BaseAnalyzer error handling."""
    
    @pytest.fixture
    def mock_analyzer(self):
        """Create a mock analyzer for testing."""
        return MockAnalyzer()
    
    @pytest.fixture
    def valid_data(self):
        """Create valid data for testing."""
        data = MagicMock()
        data.is_valid.return_value = True
        data.symbol = "EURUSD"
        data.timeframe = "H1"
        data.close = [1.0, 2.0, 3.0]
        return data
    
    @pytest.fixture
    def invalid_data(self):
        """Create invalid data for testing."""
        data = MagicMock()
        data.is_valid.return_value = False
        data.symbol = "EURUSD"
        data.timeframe = "H1"
        data.close = []
        return data
    
    @pytest.fixture
    def failing_data(self):
        """Create data that will cause the analyzer to fail."""
        data = MagicMock()
        data.is_valid.return_value = True
        data.symbol = "EURUSD"
        data.timeframe = "H1"
        data.close = [1.0, 2.0, 3.0]
        data.should_fail = True
        return data
    
    @pytest.fixture
    def timeout_data(self):
        """Create data that will cause the analyzer to timeout."""
        data = MagicMock()
        data.is_valid.return_value = True
        data.symbol = "EURUSD"
        data.timeframe = "H1"
        data.close = [1.0, 2.0, 3.0]
        data.should_timeout = True
        return data
    
    async def test_execute_null_data(self, mock_analyzer):
        """Test execute with null data."""
        result = await mock_analyzer.execute(None)
        
        assert result.is_valid is False
        assert "Null data provided for analysis" in result.result_data.get("error", "")
    
    async def test_execute_invalid_data(self, mock_analyzer, invalid_data):
        """Test execute with invalid data."""
        with patch('asyncio.create_task', new_callable=AsyncMock) as mock_create_task:
            with pytest.raises(InsufficientDataError) as excinfo:
                await mock_analyzer.execute(invalid_data)
            
            assert "Insufficient data for mock_analyzer analysis" in str(excinfo.value)
            assert excinfo.value.symbol == "EURUSD"
            assert excinfo.value.timeframe == "H1"
            assert excinfo.value.available_points == 0
    
    async def test_execute_analysis_error(self, mock_analyzer, failing_data):
        """Test execute with data that causes an analysis error."""
        result = await mock_analyzer.execute(failing_data)
        
        assert result.is_valid is False
        assert "Error in analyzer mock_analyzer" in result.result_data.get("error", "")
        assert result.result_data.get("error_code") == "UNEXPECTED_ANALYSIS_ERROR"
        assert "symbol" in result.metadata
        assert result.metadata.get("symbol") == "EURUSD"
        assert "timeframe" in result.metadata
        assert result.metadata.get("timeframe") == "H1"
        assert "error_details" in result.metadata
    
    @pytest.mark.asyncio
    async def test_execute_timeout(self, mock_analyzer, timeout_data):
        """Test execute with data that causes a timeout."""
        # Mock asyncio.wait_for to raise TimeoutError
        with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
            result = await mock_analyzer.execute(timeout_data)
            
            assert result.is_valid is False
            assert "Analysis timed out" in result.result_data.get("error", "")
            assert "symbol" in result.metadata
            assert result.metadata.get("symbol") == "EURUSD"
            assert "timeframe" in result.metadata
            assert result.metadata.get("timeframe") == "H1"


class TestAnalysisService:
    """Tests for the AnalysisService error handling."""
    
    @pytest.fixture
    def analysis_service(self):
        """Create an analysis service for testing."""
        service = AnalysisService()
        return service
    
    @pytest.fixture
    async def initialized_service(self, analysis_service):
        """Create an initialized analysis service for testing."""
        await analysis_service.initialize()
        return analysis_service
    
    @pytest.fixture
    def mock_analyzer(self):
        """Create a mock analyzer for testing."""
        analyzer = MockAnalyzer()
        return analyzer
    
    @pytest.fixture
    def valid_data(self):
        """Create valid data for testing."""
        data = MagicMock(spec=MarketData)
        data.is_valid.return_value = True
        data.symbol = "EURUSD"
        data.timeframe = "H1"
        data.close = [1.0, 2.0, 3.0]
        return data
    
    @pytest.fixture
    def invalid_data(self):
        """Create invalid data for testing."""
        data = MagicMock(spec=MarketData)
        data.is_valid.return_value = False
        data.symbol = "EURUSD"
        data.timeframe = "H1"
        data.close = []
        return data
    
    @pytest.fixture
    def multi_timeframe_data(self):
        """Create multi-timeframe data for testing."""
        h1_data = MagicMock(spec=MarketData)
        h1_data.is_valid.return_value = True
        h1_data.symbol = "EURUSD"
        h1_data.timeframe = "H1"
        h1_data.close = [1.0, 2.0, 3.0]
        
        h4_data = MagicMock(spec=MarketData)
        h4_data.is_valid.return_value = True
        h4_data.symbol = "EURUSD"
        h4_data.timeframe = "H4"
        h4_data.close = [1.0, 2.0, 3.0]
        
        return {"H1": h1_data, "H4": h4_data}
    
    @pytest.fixture
    def invalid_multi_timeframe_data(self):
        """Create invalid multi-timeframe data for testing."""
        h1_data = MagicMock(spec=MarketData)
        h1_data.is_valid.return_value = True
        h1_data.symbol = "EURUSD"
        h1_data.timeframe = "H1"
        h1_data.close = [1.0, 2.0, 3.0]
        
        h4_data = MagicMock(spec=MarketData)
        h4_data.is_valid.return_value = False
        h4_data.symbol = "EURUSD"
        h4_data.timeframe = "H4"
        h4_data.close = []
        
        return {"H1": h1_data, "H4": h4_data}
    
    async def test_run_analysis_empty_analyzer_name(self, initialized_service, valid_data):
        """Test run_analysis with empty analyzer name."""
        with pytest.raises(InvalidAnalysisParametersError) as excinfo:
            await initialized_service.run_analysis("", valid_data)
        
        assert "Analyzer name cannot be empty" in str(excinfo.value)
        assert excinfo.value.error_code == "EMPTY_ANALYZER_NAME"
    
    async def test_run_analysis_null_data(self, initialized_service):
        """Test run_analysis with null data."""
        with pytest.raises(InvalidAnalysisParametersError) as excinfo:
            await initialized_service.run_analysis("confluence", None)
        
        assert "Data cannot be null" in str(excinfo.value)
        assert excinfo.value.error_code == "NULL_DATA"
    
    async def test_run_analysis_invalid_data(self, initialized_service, invalid_data):
        """Test run_analysis with invalid data."""
        with pytest.raises(InsufficientDataError) as excinfo:
            await initialized_service.run_analysis("confluence", invalid_data)
        
        assert "Insufficient data for confluence analysis" in str(excinfo.value)
        assert excinfo.value.symbol == "EURUSD"
        assert excinfo.value.timeframe == "H1"
        assert excinfo.value.available_points == 0
    
    async def test_run_analysis_invalid_multi_timeframe_data(self, initialized_service, invalid_multi_timeframe_data):
        """Test run_analysis with invalid multi-timeframe data."""
        with patch.object(initialized_service, 'get_analyzer', return_value=MockAnalyzer()):
            with pytest.raises(InsufficientDataError) as excinfo:
                await initialized_service.run_analysis("multi_timeframe", invalid_multi_timeframe_data)
            
            assert "Insufficient data for multi_timeframe analysis in 1 timeframes" in str(excinfo.value)
            assert "timeframes" in excinfo.value.details
            assert len(excinfo.value.details["timeframes"]) == 1
            assert excinfo.value.details["timeframes"][0]["timeframe"] == "H4"
    
    async def test_run_analysis_empty_multi_timeframe_data(self, initialized_service):
        """Test run_analysis with empty multi-timeframe data."""
        with patch.object(initialized_service, 'get_analyzer', return_value=MockAnalyzer()):
            with pytest.raises(InvalidAnalysisParametersError) as excinfo:
                await initialized_service.run_analysis("multi_timeframe", {})
            
            assert "Empty timeframe dictionary provided for analysis" in str(excinfo.value)
            assert excinfo.value.error_code == "EMPTY_TIMEFRAME_DICT"
    
    async def test_run_analysis_analyzer_not_found(self, initialized_service, valid_data):
        """Test run_analysis with non-existent analyzer."""
        with pytest.raises(AnalyzerNotFoundError) as excinfo:
            await initialized_service.run_analysis("non_existent", valid_data)
        
        assert "Analyzer 'non_existent' not found" in str(excinfo.value)
        assert excinfo.value.analyzer_name == "non_existent"
        assert "available_analyzers" in excinfo.value.details
    
    async def test_run_analysis_timeout(self, initialized_service, valid_data):
        """Test run_analysis with timeout."""
        with patch.object(initialized_service, 'get_analyzer', return_value=MockAnalyzer()):
            with patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
                with pytest.raises(AnalysisTimeoutError) as excinfo:
                    await initialized_service.run_analysis("mock_analyzer", valid_data)
                
                assert "Analysis timed out for mock_analyzer" in str(excinfo.value)
                assert excinfo.value.analyzer_name == "mock_analyzer"
                assert excinfo.value.symbol == "EURUSD"
                assert excinfo.value.timeframe == "H1"
