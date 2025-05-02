"""
Integration tests for core functionality.

This module contains integration tests that verify the interaction between
different components of the analysis engine.
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock
from analysis_engine.main import create_app
from analysis_engine.core.container import ServiceContainer
from analysis_engine.core.monitoring.memory_monitor import MemoryMonitor
from analysis_engine.core.errors import AnalysisEngineError
from analysis_engine.services.analysis_service import AnalysisService
from analysis_engine.analysis.confluence.confluence_analyzer import ConfluenceAnalyzer
from analysis_engine.analysis.multi_timeframe.multi_timeframe_analyzer import MultiTimeframeAnalyzer

# Fixtures 'app', 'service_container', 'memory_monitor' are now imported from conftest.py

@pytest.fixture
def mock_market_data():
    """Create mock market data for testing."""
    return {
        "symbol": "EURUSD",
        "timeframe": "H1",
        "data": [
            {
                "timestamp": (datetime.now() - timedelta(hours=i)).isoformat(),
                "open": 1.1000 + (i * 0.001),
                "high": 1.1100 + (i * 0.001),
                "low": 1.0900 + (i * 0.001),
                "close": 1.1050 + (i * 0.001),
                "volume": 1000 + (i * 100)
            }
            for i in range(24)
        ]
    }

@pytest.mark.asyncio
async def test_service_container_initialization(service_container, memory_monitor):
    """Test service container initialization with memory monitoring."""
    # Register services
    analysis_service = AnalysisService()
    service_container.register_service("analysis", analysis_service)
    
    # Register analyzers
    confluence_analyzer = ConfluenceAnalyzer()
    multi_timeframe_analyzer = MultiTimeframeAnalyzer()
    service_container.register_analyzer("confluence", confluence_analyzer)
    service_container.register_analyzer("multi_timeframe", multi_timeframe_analyzer)
    
    # Initialize container
    await service_container.initialize()
    
    # Verify services and analyzers are initialized
    assert service_container._initialized
    assert "analysis" in service_container.list_services()
    assert "confluence" in service_container.list_analyzers()
    assert "multi_timeframe" in service_container.list_analyzers()

@pytest.mark.asyncio
async def test_analysis_service_integration(service_container, mock_market_data):
    """Test integration between analysis service and analyzers."""
    # Set up services
    analysis_service = AnalysisService()
    confluence_analyzer = ConfluenceAnalyzer()
    multi_timeframe_analyzer = MultiTimeframeAnalyzer()
    
    service_container.register_service("analysis", analysis_service)
    service_container.register_analyzer("confluence", confluence_analyzer)
    service_container.register_analyzer("multi_timeframe", multi_timeframe_analyzer)
    
    await service_container.initialize()
    
    # Perform analysis
    result = await analysis_service.analyze(mock_market_data)
    
    # Verify results
    assert "confluence" in result
    assert "multi_timeframe" in result
    assert "confidence" in result
    assert "timestamp" in result

@pytest.mark.asyncio
async def test_memory_monitoring_integration(service_container, memory_monitor):
    """Test integration of memory monitoring with service container."""
    # Start memory monitoring
    await memory_monitor.start_monitoring()
    
    # Register and initialize services
    analysis_service = AnalysisService()
    service_container.register_service("analysis", analysis_service)
    await service_container.initialize()
    
    # Verify memory monitoring is active
    assert memory_monitor._monitoring_task is not None
    assert not memory_monitor._monitoring_task.done()
    
    # Get memory stats
    stats = memory_monitor.get_memory_stats()
    assert "rss" in stats
    assert "vms" in stats
    assert "percent" in stats
    
    # Cleanup
    await service_container.cleanup()
    await memory_monitor.stop_monitoring()

@pytest.mark.asyncio
async def test_error_handling_integration(service_container, mock_market_data):
    """Test error handling across components."""
    # Set up services with error simulation
    analysis_service = AnalysisService()
    confluence_analyzer = ConfluenceAnalyzer()
    
    service_container.register_service("analysis", analysis_service)
    service_container.register_analyzer("confluence", confluence_analyzer)
    
    await service_container.initialize()
    
    # Simulate error in analyzer
    with patch.object(confluence_analyzer, 'analyze', side_effect=AnalysisEngineError("Test error")):
        with pytest.raises(AnalysisEngineError):
            await analysis_service.analyze(mock_market_data)
    
    # Verify service container is still functional
    assert service_container._initialized
    assert "analysis" in service_container.list_services()

@pytest.mark.asyncio
async def test_concurrent_analysis_requests(service_container, mock_market_data):
    """Test handling of concurrent analysis requests."""
    analysis_service = AnalysisService()
    service_container.register_service("analysis", analysis_service)
    await service_container.initialize()
    
    # Create multiple analysis requests
    async def run_analysis():
        return await analysis_service.analyze(mock_market_data)
    
    # Run multiple analyses concurrently
    tasks = [run_analysis() for _ in range(5)]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Verify all analyses completed
    assert len(results) == 5
    assert all(not isinstance(r, Exception) for r in results)

@pytest.mark.asyncio
async def test_resource_cleanup(service_container, memory_monitor):
    """Test proper cleanup of resources."""
    # Initialize services
    analysis_service = AnalysisService()
    service_container.register_service("analysis", analysis_service)
    await service_container.initialize()
    await memory_monitor.start_monitoring()
    
    # Perform cleanup
    await service_container.cleanup()
    await memory_monitor.stop_monitoring()
    
    # Verify cleanup
    assert not service_container._initialized
    assert memory_monitor._monitoring_task is None or memory_monitor._monitoring_task.done()

@pytest.mark.asyncio
async def test_analysis_pipeline_integration(service_container, mock_market_data):
    """Test the complete analysis pipeline integration."""
    # Set up all components
    analysis_service = AnalysisService()
    confluence_analyzer = ConfluenceAnalyzer()
    multi_timeframe_analyzer = MultiTimeframeAnalyzer()
    
    service_container.register_service("analysis", analysis_service)
    service_container.register_analyzer("confluence", confluence_analyzer)
    service_container.register_analyzer("multi_timeframe", multi_timeframe_analyzer)
    
    await service_container.initialize()
    
    # Run analysis pipeline
    result = await analysis_service.analyze(mock_market_data)
    
    # Verify pipeline results
    assert "confluence" in result
    assert "multi_timeframe" in result
    assert "confidence" in result
    assert "timestamp" in result
    
    # Verify memory usage
    stats = service_container.memory_monitor.get_memory_stats()
    assert stats["percent"] < 90  # Should not be in critical state
    
    # Cleanup
    await service_container.cleanup()