"""
Tests for error handling in the Strategy Execution Engine.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch
import asyncio
from fastapi import HTTPException

from core.error import (
    ForexTradingPlatformError,
    StrategyExecutionError,
    StrategyConfigurationError,
    StrategyLoadError,
    BacktestError,
    BacktestConfigError,
    BacktestDataError,
    BacktestExecutionError,
    BacktestReportError,
    with_error_handling,
    async_with_error_handling,
    map_exception_to_http
)


def test_forex_trading_platform_error():
    """Test ForexTradingPlatformError."""
    # Create error
    error = ForexTradingPlatformError("Test error", "TEST_ERROR", {"param": "value"})
    
    # Verify
    assert error.message == "Test error"
    assert error.code == "TEST_ERROR"
    assert error.details == {"param": "value"}
    assert str(error) == "Test error"
    
    # Test to_dict
    error_dict = error.to_dict()
    assert error_dict["error"] == "TEST_ERROR"
    assert error_dict["message"] == "Test error"
    assert error_dict["details"] == {"param": "value"}
    assert "timestamp" in error_dict
    assert error_dict["type"] == "ForexTradingPlatformError"


def test_strategy_execution_error():
    """Test StrategyExecutionError."""
    # Create error
    error = StrategyExecutionError("Test error", "TEST_ERROR", {"param": "value"})
    
    # Verify
    assert error.message == "Test error"
    assert error.code == "TEST_ERROR"
    assert error.details == {"param": "value"}
    assert isinstance(error, ForexTradingPlatformError)


def test_strategy_configuration_error():
    """Test StrategyConfigurationError."""
    # Create error
    error = StrategyConfigurationError("Test error", {"param": "value"})
    
    # Verify
    assert error.message == "Test error"
    assert error.code == "STRATEGY_CONFIGURATION_ERROR"
    assert error.details == {"param": "value"}
    assert isinstance(error, StrategyExecutionError)
    assert isinstance(error, ForexTradingPlatformError)


def test_with_error_handling_no_error():
    """Test with_error_handling decorator with no error."""
    # Define test function
    @with_error_handling
    def test_func(a, b):
        return a + b
    
    # Call function
    result = test_func(1, 2)
    
    # Verify
    assert result == 3


def test_with_error_handling_platform_error():
    """Test with_error_handling decorator with ForexTradingPlatformError."""
    # Define test function
    @with_error_handling
    def test_func():
        raise StrategyConfigurationError("Test error")
    
    # Call function and verify error is re-raised
    with pytest.raises(StrategyConfigurationError) as excinfo:
        test_func()
    
    assert str(excinfo.value) == "Test error"


def test_with_error_handling_generic_error():
    """Test with_error_handling decorator with generic error."""
    # Define test function
    @with_error_handling
    def test_func():
        raise ValueError("Test error")
    
    # Call function and verify error is wrapped
    with pytest.raises(StrategyExecutionError) as excinfo:
        test_func()
    
    assert "An unexpected error occurred" in str(excinfo.value)
    assert "Test error" in str(excinfo.value)


@pytest.mark.asyncio
async def test_async_with_error_handling_no_error():
    """Test async_with_error_handling decorator with no error."""
    # Define test function
    @async_with_error_handling
    async def test_func(a, b):
        return a + b
    
    # Call function
    result = await test_func(1, 2)
    
    # Verify
    assert result == 3


@pytest.mark.asyncio
async def test_async_with_error_handling_platform_error():
    """Test async_with_error_handling decorator with ForexTradingPlatformError."""
    # Define test function
    @async_with_error_handling
    async def test_func():
        raise StrategyConfigurationError("Test error")
    
    # Call function and verify error is re-raised
    with pytest.raises(StrategyConfigurationError) as excinfo:
        await test_func()
    
    assert str(excinfo.value) == "Test error"


@pytest.mark.asyncio
async def test_async_with_error_handling_generic_error():
    """Test async_with_error_handling decorator with generic error."""
    # Define test function
    @async_with_error_handling
    async def test_func():
        raise ValueError("Test error")
    
    # Call function and verify error is wrapped
    with pytest.raises(StrategyExecutionError) as excinfo:
        await test_func()
    
    assert "An unexpected error occurred" in str(excinfo.value)
    assert "Test error" in str(excinfo.value)


def test_map_exception_to_http_strategy_configuration_error():
    """Test mapping StrategyConfigurationError to HTTP exception."""
    # Create error
    error = StrategyConfigurationError("Test error")
    
    # Map to HTTP exception
    http_error = map_exception_to_http(error)
    
    # Verify
    assert isinstance(http_error, HTTPException)
    assert http_error.status_code == 400
    assert http_error.detail == "Test error"


def test_map_exception_to_http_strategy_load_error():
    """Test mapping StrategyLoadError to HTTP exception."""
    # Create error
    error = StrategyLoadError("Test error")
    
    # Map to HTTP exception
    http_error = map_exception_to_http(error)
    
    # Verify
    assert isinstance(http_error, HTTPException)
    assert http_error.status_code == 404
    assert http_error.detail == "Test error"


def test_map_exception_to_http_forex_platform_error():
    """Test mapping ForexTradingPlatformError to HTTP exception."""
    # Create error
    error = ForexTradingPlatformError("Test error")
    
    # Map to HTTP exception
    http_error = map_exception_to_http(error)
    
    # Verify
    assert isinstance(http_error, HTTPException)
    assert http_error.status_code == 500
    assert http_error.detail == "Test error"


def test_map_exception_to_http_generic_error():
    """Test mapping generic error to HTTP exception."""
    # Create error
    error = ValueError("Test error")
    
    # Map to HTTP exception
    http_error = map_exception_to_http(error)
    
    # Verify
    assert isinstance(http_error, HTTPException)
    assert http_error.status_code == 500
    assert http_error.detail == "An unexpected error occurred"
