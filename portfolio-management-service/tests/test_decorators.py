"""
Tests for error handling decorators in Portfolio Management Service.
"""
import pytest
import asyncio
from unittest.mock import patch, MagicMock

from portfolio_management_service.error import (
    with_exception_handling,
    async_with_exception_handling,
    PortfolioManagementError,
    PortfolioNotFoundError
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
        raise PortfolioNotFoundError(
            message="Portfolio not found",
            portfolio_id="test-portfolio"
        )
    
    with pytest.raises(PortfolioNotFoundError) as excinfo:
        test_function()
    
    assert excinfo.value.message == "Portfolio not found"
    assert excinfo.value.error_code == "PORTFOLIO_NOT_FOUND_ERROR"
    assert excinfo.value.details == {"portfolio_id": "test-portfolio"}


def test_with_exception_handling_generic_error():
    """Test with_exception_handling with generic error."""
    @with_exception_handling
    def test_function():
        raise ValueError("Test error")
    
    with pytest.raises(PortfolioManagementError) as excinfo:
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
        raise PortfolioNotFoundError(
            message="Portfolio not found",
            portfolio_id="test-portfolio"
        )
    
    with pytest.raises(PortfolioNotFoundError) as excinfo:
        await test_function()
    
    assert excinfo.value.message == "Portfolio not found"
    assert excinfo.value.error_code == "PORTFOLIO_NOT_FOUND_ERROR"
    assert excinfo.value.details == {"portfolio_id": "test-portfolio"}


@pytest.mark.asyncio
async def test_async_with_exception_handling_generic_error():
    """Test async_with_exception_handling with generic error."""
    @async_with_exception_handling
    async def test_function():
        raise ValueError("Test error")
    
    with pytest.raises(PortfolioManagementError) as excinfo:
        await test_function()
    
    assert "Unexpected error: Test error" in excinfo.value.message
    assert excinfo.value.error_code == "UNEXPECTED_ERROR"
    assert "original_error" in excinfo.value.details


@patch("portfolio_management_service.error.exceptions_bridge.logger")
def test_with_exception_handling_logging(mock_logger):
    """Test that with_exception_handling logs errors correctly."""
    @with_exception_handling
    def test_function():
        raise ValueError("Test error")
    
    with pytest.raises(PortfolioManagementError):
        test_function()
    
    # Check that error was logged
    mock_logger.error.assert_called()
    args, _ = mock_logger.error.call_args
    assert "Unexpected error: Test error" in args[0]


@patch("portfolio_management_service.error.exceptions_bridge.logger")
@pytest.mark.asyncio
async def test_async_with_exception_handling_logging(mock_logger):
    """Test that async_with_exception_handling logs errors correctly."""
    @async_with_exception_handling
    async def test_function():
        raise ValueError("Test error")
    
    with pytest.raises(PortfolioManagementError):
        await test_function()
    
    # Check that error was logged
    mock_logger.error.assert_called()
    args, _ = mock_logger.error.call_args
    assert "Unexpected error: Test error" in args[0]