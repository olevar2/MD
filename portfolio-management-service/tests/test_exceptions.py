"""
Tests for error handling in Portfolio Management Service.
"""
import pytest
from fastapi import HTTPException

from portfolio_management_service.error import (
    PortfolioManagementError,
    PortfolioNotFoundError,
    PositionNotFoundError,
    InsufficientBalanceError,
    PortfolioOperationError,
    AccountReconciliationError,
    TaxCalculationError,
    convert_to_http_exception
)


def test_portfolio_management_error():
    """Test PortfolioManagementError."""
    error = PortfolioManagementError(
        message="Test error",
        error_code="TEST_ERROR",
        details={"test": "value"}
    )
    
    assert error.message == "Test error"
    assert error.error_code == "TEST_ERROR"
    assert error.details == {"test": "value"}


def test_portfolio_not_found_error():
    """Test PortfolioNotFoundError."""
    error = PortfolioNotFoundError(
        message="Portfolio not found",
        portfolio_id="test-portfolio"
    )
    
    assert error.message == "Portfolio not found"
    assert error.error_code == "PORTFOLIO_NOT_FOUND_ERROR"
    assert error.details == {"portfolio_id": "test-portfolio"}


def test_position_not_found_error():
    """Test PositionNotFoundError."""
    error = PositionNotFoundError(
        message="Position not found",
        position_id="test-position"
    )
    
    assert error.message == "Position not found"
    assert error.error_code == "POSITION_NOT_FOUND_ERROR"
    assert error.details == {"position_id": "test-position"}


def test_insufficient_balance_error():
    """Test InsufficientBalanceError."""
    error = InsufficientBalanceError(
        message="Insufficient balance",
        required_amount=100.0,
        available_amount=50.0,
        currency="USD"
    )
    
    assert error.message == "Insufficient balance"
    assert error.error_code == "INSUFFICIENT_BALANCE_ERROR"
    assert error.details == {
        "required_amount": 100.0,
        "available_amount": 50.0,
        "currency": "USD"
    }


def test_portfolio_operation_error():
    """Test PortfolioOperationError."""
    error = PortfolioOperationError(
        message="Operation failed",
        operation="test-operation"
    )
    
    assert error.message == "Operation failed"
    assert error.error_code == "PORTFOLIO_OPERATION_ERROR"
    assert error.details == {"operation": "test-operation"}


def test_account_reconciliation_error():
    """Test AccountReconciliationError."""
    error = AccountReconciliationError(
        message="Reconciliation failed",
        account_id="test-account"
    )
    
    assert error.message == "Reconciliation failed"
    assert error.error_code == "ACCOUNT_RECONCILIATION_ERROR"
    assert error.details == {"account_id": "test-account"}


def test_tax_calculation_error():
    """Test TaxCalculationError."""
    error = TaxCalculationError(
        message="Tax calculation failed",
        details={"year": 2023}
    )
    
    assert error.message == "Tax calculation failed"
    assert error.error_code == "TAX_CALCULATION_ERROR"
    assert error.details == {"year": 2023}


def test_convert_to_http_exception():
    """Test convert_to_http_exception function."""
    # Test PortfolioNotFoundError (404)
    error = PortfolioNotFoundError(
        message="Portfolio not found",
        portfolio_id="test-portfolio"
    )
    http_exc = convert_to_http_exception(error)
    
    assert isinstance(http_exc, HTTPException)
    assert http_exc.status_code == 404
    assert http_exc.detail["error_code"] == "PORTFOLIO_NOT_FOUND_ERROR"
    assert http_exc.detail["message"] == "Portfolio not found"
    
    # Test InsufficientBalanceError (403)
    error = InsufficientBalanceError(
        message="Insufficient balance",
        required_amount=100.0,
        available_amount=50.0,
        currency="USD"
    )
    http_exc = convert_to_http_exception(error)
    
    assert isinstance(http_exc, HTTPException)
    assert http_exc.status_code == 403
    assert http_exc.detail["error_code"] == "INSUFFICIENT_BALANCE_ERROR"
    assert http_exc.detail["message"] == "Insufficient balance"
    
    # Test PortfolioOperationError (400)
    error = PortfolioOperationError(
        message="Operation failed",
        operation="test-operation"
    )
    http_exc = convert_to_http_exception(error)
    
    assert isinstance(http_exc, HTTPException)
    assert http_exc.status_code == 400
    assert http_exc.detail["error_code"] == "PORTFOLIO_OPERATION_ERROR"
    assert http_exc.detail["message"] == "Operation failed"
    
    # Test AccountReconciliationError (422)
    error = AccountReconciliationError(
        message="Reconciliation failed",
        account_id="test-account"
    )
    http_exc = convert_to_http_exception(error)
    
    assert isinstance(http_exc, HTTPException)
    assert http_exc.status_code == 422
    assert http_exc.detail["error_code"] == "ACCOUNT_RECONCILIATION_ERROR"
    assert http_exc.detail["message"] == "Reconciliation failed"
    
    # Test generic PortfolioManagementError (500)
    error = PortfolioManagementError(
        message="Generic error",
        error_code="GENERIC_ERROR"
    )
    http_exc = convert_to_http_exception(error)
    
    assert isinstance(http_exc, HTTPException)
    assert http_exc.status_code == 500
    assert http_exc.detail["error_code"] == "GENERIC_ERROR"
    assert http_exc.detail["message"] == "Generic error"