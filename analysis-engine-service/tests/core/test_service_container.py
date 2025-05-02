"""
Tests for the service container module.

This module contains tests for the ServiceContainer class and its functionality.
"""

import pytest
from unittest.mock import MagicMock, patch
from analysis_engine.core.container import ServiceContainer
from analysis_engine.core.errors import ServiceContainerError

@pytest.fixture
def service_container():
    """Create a service container instance for testing."""
    return ServiceContainer()

@pytest.fixture
def mock_analyzer():
    """Create a mock analyzer for testing."""
    analyzer = MagicMock()
    analyzer.initialize = MagicMock(return_value=None)
    analyzer.cleanup = MagicMock(return_value=None)
    return analyzer

@pytest.fixture
def mock_service():
    """Create a mock service for testing."""
    service = MagicMock()
    service.initialize = MagicMock(return_value=None)
    service.cleanup = MagicMock(return_value=None)
    return service

def test_service_container_initialization(service_container):
    """Test service container initialization."""
    assert service_container._services == {}
    assert service_container._analyzers == {}
    assert not service_container._initialized

def test_register_service(service_container, mock_service):
    """Test registering a service."""
    service_container.register_service("test_service", mock_service)
    assert "test_service" in service_container._services
    assert service_container._services["test_service"] == mock_service

def test_register_analyzer(service_container, mock_analyzer):
    """Test registering an analyzer."""
    service_container.register_analyzer("test_analyzer", mock_analyzer)
    assert "test_analyzer" in service_container._analyzers
    assert service_container._analyzers["test_analyzer"] == mock_analyzer

def test_register_duplicate_service(service_container, mock_service):
    """Test registering a duplicate service."""
    service_container.register_service("test_service", mock_service)
    with pytest.raises(ServiceContainerError):
        service_container.register_service("test_service", mock_service)

def test_register_duplicate_analyzer(service_container, mock_analyzer):
    """Test registering a duplicate analyzer."""
    service_container.register_analyzer("test_analyzer", mock_analyzer)
    with pytest.raises(ServiceContainerError):
        service_container.register_analyzer("test_analyzer", mock_analyzer)

@pytest.mark.asyncio
async def test_initialize(service_container, mock_service, mock_analyzer):
    """Test container initialization."""
    service_container.register_service("test_service", mock_service)
    service_container.register_analyzer("test_analyzer", mock_analyzer)
    
    await service_container.initialize()
    
    assert service_container._initialized
    mock_service.initialize.assert_called_once()
    mock_analyzer.initialize.assert_called_once()

@pytest.mark.asyncio
async def test_initialize_error(service_container, mock_service):
    """Test initialization error handling."""
    mock_service.initialize.side_effect = Exception("Test error")
    service_container.register_service("test_service", mock_service)
    
    with pytest.raises(ServiceContainerError):
        await service_container.initialize()
    
    assert not service_container._initialized

@pytest.mark.asyncio
async def test_cleanup(service_container, mock_service, mock_analyzer):
    """Test container cleanup."""
    service_container.register_service("test_service", mock_service)
    service_container.register_analyzer("test_analyzer", mock_analyzer)
    
    await service_container.initialize()
    await service_container.cleanup()
    
    mock_service.cleanup.assert_called_once()
    mock_analyzer.cleanup.assert_called_once()

@pytest.mark.asyncio
async def test_cleanup_error(service_container, mock_service):
    """Test cleanup error handling."""
    mock_service.cleanup.side_effect = Exception("Test error")
    service_container.register_service("test_service", mock_service)
    
    await service_container.initialize()
    with pytest.raises(ServiceContainerError):
        await service_container.cleanup()

def test_get_service(service_container, mock_service):
    """Test getting a registered service."""
    service_container.register_service("test_service", mock_service)
    service = service_container.get_service("test_service")
    assert service == mock_service

def test_get_nonexistent_service(service_container):
    """Test getting a nonexistent service."""
    with pytest.raises(ServiceContainerError):
        service_container.get_service("nonexistent")

def test_get_analyzer(service_container, mock_analyzer):
    """Test getting a registered analyzer."""
    service_container.register_analyzer("test_analyzer", mock_analyzer)
    analyzer = service_container.get_analyzer("test_analyzer")
    assert analyzer == mock_analyzer

def test_get_nonexistent_analyzer(service_container):
    """Test getting a nonexistent analyzer."""
    with pytest.raises(ServiceContainerError):
        service_container.get_analyzer("nonexistent")

def test_list_services(service_container, mock_service):
    """Test listing registered services."""
    service_container.register_service("test_service", mock_service)
    services = service_container.list_services()
    assert "test_service" in services

def test_list_analyzers(service_container, mock_analyzer):
    """Test listing registered analyzers."""
    service_container.register_analyzer("test_analyzer", mock_analyzer)
    analyzers = service_container.list_analyzers()
    assert "test_analyzer" in analyzers

@pytest.mark.asyncio
async def test_initialize_order(service_container, mock_service, mock_analyzer):
    """Test initialization order of services and analyzers."""
    init_order = []
    
    def mock_init():
        init_order.append("service")
        return None
    
    def mock_analyzer_init():
        init_order.append("analyzer")
        return None
    
    mock_service.initialize = MagicMock(side_effect=mock_init)
    mock_analyzer.initialize = MagicMock(side_effect=mock_analyzer_init)
    
    service_container.register_service("test_service", mock_service)
    service_container.register_analyzer("test_analyzer", mock_analyzer)
    
    await service_container.initialize()
    
    # Services should be initialized before analyzers
    assert init_order.index("service") < init_order.index("analyzer") 