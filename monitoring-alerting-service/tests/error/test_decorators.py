"""
Tests for error handling decorators in Monitoring & Alerting Service.
"""
import pytest
from unittest.mock import patch, MagicMock

from monitoring_alerting_service.error import (
    with_exception_handling,
    async_with_exception_handling,
    MonitoringAlertingError,
    AlertNotFoundError
)


def test_with_exception_handling_no_error():
    """Test with_exception_handling when no error occurs."""
    @with_exception_handling
    def test_function():
        return "success"
    
    result = test_function()
    assert result == "success"


def test_with_exception_handling_domain_error():
    """Test with_exception_handling with domain-specific error."""
    @with_exception_handling
    def test_function():
        raise AlertNotFoundError(
            message="Alert not found",
            alert_id="test-alert"
        )
    
    with pytest.raises(AlertNotFoundError) as excinfo:
        test_function()
    
    assert excinfo.value.message == "Alert not found"
    assert excinfo.value.error_code == "ALERT_NOT_FOUND_ERROR"
    assert excinfo.value.details == {"alert_id": "test-alert"}


def test_with_exception_handling_generic_error():
    """Test with_exception_handling with generic error."""
    @with_exception_handling
    def test_function():
        raise ValueError("Test error")
    
    with pytest.raises(MonitoringAlertingError) as excinfo:
        test_function()
    
    assert "Unexpected error: Test error" in excinfo.value.message
    assert excinfo.value.error_code == "UNEXPECTED_ERROR"
    assert "original_error" in excinfo.value.details


@pytest.mark.asyncio
async def test_async_with_exception_handling_no_error():
    """Test async_with_exception_handling when no error occurs."""
    @async_with_exception_handling
    async def test_function():
        return "success"
    
    result = await test_function()
    assert result == "success"


@pytest.mark.asyncio
async def test_async_with_exception_handling_domain_error():
    """Test async_with_exception_handling with domain-specific error."""
    @async_with_exception_handling
    async def test_function():
        raise AlertNotFoundError(
            message="Alert not found",
            alert_id="test-alert"
        )
    
    with pytest.raises(AlertNotFoundError) as excinfo:
        await test_function()
    
    assert excinfo.value.message == "Alert not found"
    assert excinfo.value.error_code == "ALERT_NOT_FOUND_ERROR"
    assert excinfo.value.details == {"alert_id": "test-alert"}


@pytest.mark.asyncio
async def test_async_with_exception_handling_generic_error():
    """Test async_with_exception_handling with generic error."""
    @async_with_exception_handling
    async def test_function():
        raise ValueError("Test error")
    
    with pytest.raises(MonitoringAlertingError) as excinfo:
        await test_function()
    
    assert "Unexpected error: Test error" in excinfo.value.message
    assert excinfo.value.error_code == "UNEXPECTED_ERROR"
    assert "original_error" in excinfo.value.details


@patch("monitoring_alerting_service.error.exceptions_bridge.logger")
def test_with_exception_handling_logging(mock_logger):
    """Test that with_exception_handling logs errors correctly."""
    @with_exception_handling
    def test_function():
        raise ValueError("Test error")
    
    with pytest.raises(MonitoringAlertingError):
        test_function()
    
    # Check that error was logged
    mock_logger.error.assert_called()
    args, _ = mock_logger.error.call_args
    assert "Unexpected error: Test error" in args[0]
