import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
import time

from analysis_coordinator_service.utils.resilience import (
    with_retry,
    with_circuit_breaker,
    CircuitBreaker
)

@pytest.mark.asyncio
async def test_with_retry_success():
    # Arrange
    mock_func = AsyncMock()
    mock_func.return_value = "success"
    
    # Create a decorated function
    @with_retry(max_retries=3, backoff_factor=0.1)
    async def test_func():
        return await mock_func()
    
    # Act
    result = await test_func()
    
    # Assert
    assert result == "success"
    mock_func.assert_called_once()

@pytest.mark.asyncio
async def test_with_retry_failure_then_success():
    # Arrange
    mock_func = AsyncMock()
    mock_func.side_effect = [Exception("Error"), Exception("Error"), "success"]
    
    # Create a decorated function
    @with_retry(max_retries=3, backoff_factor=0.1)
    async def test_func():
        return await mock_func()
    
    # Act
    result = await test_func()
    
    # Assert
    assert result == "success"
    assert mock_func.call_count == 3

@pytest.mark.asyncio
async def test_with_retry_all_failures():
    # Arrange
    mock_func = AsyncMock()
    mock_func.side_effect = Exception("Error")
    
    # Create a decorated function
    @with_retry(max_retries=3, backoff_factor=0.1)
    async def test_func():
        return await mock_func()
    
    # Act & Assert
    with pytest.raises(Exception) as excinfo:
        await test_func()
    
    assert "Error" in str(excinfo.value)
    assert mock_func.call_count == 4  # Initial call + 3 retries

@pytest.mark.asyncio
async def test_circuit_breaker_closed():
    # Arrange
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
    
    # Act & Assert
    assert cb.can_execute() is True
    assert cb.state == "closed"
    
    # Record a success
    cb.record_success()
    assert cb.state == "closed"
    assert cb.failure_count == 0

@pytest.mark.asyncio
async def test_circuit_breaker_open():
    # Arrange
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=1)
    
    # Act
    for _ in range(3):
        cb.record_failure()
    
    # Assert
    assert cb.state == "open"
    assert cb.can_execute() is False

@pytest.mark.asyncio
async def test_circuit_breaker_half_open():
    # Arrange
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
    
    # Act
    for _ in range(3):
        cb.record_failure()
    
    assert cb.state == "open"
    
    # Wait for recovery timeout
    await asyncio.sleep(0.2)
    
    # Assert
    assert cb.can_execute() is True
    assert cb.state == "half-open"
    
    # Record a success
    cb.record_success()
    assert cb.state == "closed"
    assert cb.failure_count == 0

@pytest.mark.asyncio
async def test_circuit_breaker_half_open_failure():
    # Arrange
    cb = CircuitBreaker(failure_threshold=3, recovery_timeout=0.1)
    
    # Act
    for _ in range(3):
        cb.record_failure()
    
    assert cb.state == "open"
    
    # Wait for recovery timeout
    await asyncio.sleep(0.2)
    
    # Assert
    assert cb.can_execute() is True
    assert cb.state == "half-open"
    
    # Record a failure
    cb.record_failure()
    assert cb.state == "open"
    assert cb.can_execute() is False

@pytest.mark.asyncio
async def test_with_circuit_breaker_success():
    # Arrange
    mock_func = AsyncMock()
    mock_func.return_value = "success"
    
    # Create a decorated function
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=1)
    async def test_func():
        return await mock_func()
    
    # Act
    result = await test_func()
    
    # Assert
    assert result == "success"
    mock_func.assert_called_once()

@pytest.mark.asyncio
async def test_with_circuit_breaker_open():
    # Arrange
    mock_func = AsyncMock()
    mock_func.side_effect = Exception("Error")
    
    # Create a decorated function
    @with_circuit_breaker(failure_threshold=3, recovery_timeout=1)
    async def test_func():
        return await mock_func()
    
    # Act & Assert
    # Call the function until the circuit opens
    for _ in range(3):
        with pytest.raises(Exception):
            await test_func()
    
    # The circuit should now be open
    with pytest.raises(Exception) as excinfo:
        await test_func()
    
    assert "Circuit breaker open" in str(excinfo.value)
    assert mock_func.call_count == 3  # Only called until the circuit opened