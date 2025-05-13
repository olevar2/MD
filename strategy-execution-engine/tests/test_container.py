"""
Tests for the service container in the Strategy Execution Engine.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio

from services.container import ServiceContainer


class TestService:
    """Test service class for container tests."""
    def __init__(self, name):
        self.name = name


class AsyncTestService:
    """Async test service class for container tests."""
    def __init__(self, name):
    """
      init  .
    
    Args:
        name: Description of name
    
    """

        self.name = name
        self.initialized = False
        self.shutdown = False
    
    async def initialize(self):
        self.initialized = True
    
    async def close(self):
        self.shutdown = True


@pytest.fixture
def container():
    """Create a service container for testing."""
    return ServiceContainer()


def test_register_and_get(container):
    """Test registering and getting a service."""
    # Register a service
    service = TestService("test")
    container.register("test_service", service)
    
    # Get the service
    retrieved = container.get("test_service")
    
    # Verify
    assert retrieved is service
    assert retrieved.name == "test"


def test_get_typed(container):
    """Test getting a service with type checking."""
    # Register a service
    service = TestService("test")
    container.register("test_service", service)
    
    # Get the service with type checking
    retrieved = container.get_typed("test_service", TestService)
    
    # Verify
    assert retrieved is service
    assert retrieved.name == "test"


def test_get_typed_wrong_type(container):
    """Test getting a service with wrong type."""
    # Register a service
    service = TestService("test")
    container.register("test_service", service)
    
    # Try to get the service with wrong type
    with pytest.raises(TypeError):
        container.get_typed("test_service", AsyncTestService)


def test_get_nonexistent(container):
    """Test getting a non-existent service."""
    with pytest.raises(KeyError):
        container.get("nonexistent")


def test_has(container):
    """Test checking if a service exists."""
    # Register a service
    service = TestService("test")
    container.register("test_service", service)
    
    # Check if service exists
    assert container.has("test_service") is True
    assert container.has("nonexistent") is False


def test_remove(container):
    """Test removing a service."""
    # Register a service
    service = TestService("test")
    container.register("test_service", service)
    
    # Remove the service
    container.remove("test_service")
    
    # Verify
    assert container.has("test_service") is False
    
    # Try to remove non-existent service
    with pytest.raises(KeyError):
        container.remove("nonexistent")


@pytest.mark.asyncio
async def test_initialize(container):
    """Test initializing the container."""
    # Verify initial state
    assert container.is_initialized is False
    
    # Initialize
    await container.initialize()
    
    # Verify
    assert container.is_initialized is True


@pytest.mark.asyncio
async def test_shutdown(container):
    """Test shutting down the container."""
    # Initialize first
    await container.initialize()
    assert container.is_initialized is True
    
    # Register a service
    service = TestService("test")
    container.register("test_service", service)
    
    # Shutdown
    await container.shutdown()
    
    # Verify
    assert container.is_initialized is False
    assert not container.has("test_service")


@pytest.mark.asyncio
async def test_shutdown_not_initialized(container):
    """Test shutting down a non-initialized container."""
    # Verify initial state
    assert container.is_initialized is False
    
    # Shutdown
    await container.shutdown()
    
    # Verify still not initialized
    assert container.is_initialized is False
