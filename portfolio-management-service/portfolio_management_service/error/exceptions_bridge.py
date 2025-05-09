"""
Exceptions Bridge Module

This module bridges the common-lib exceptions with portfolio-management-service specific exceptions.
It re-exports all common exceptions and adds service-specific ones.
"""
from typing import Dict, Any, Optional, Type, Callable, TypeVar, Union
import functools
import traceback
import logging
from fastapi import HTTPException, status

# Import common-lib exceptions
from common_lib.exceptions import (
    # Base exception
    ForexTradingPlatformError,
    
    # Configuration exceptions
    ConfigurationError,
    ConfigValidationError,
    
    # Data exceptions
    DataError,
    DataValidationError,
    DataFetchError,
    DataStorageError,
    DataTransformationError,
    
    # Service exceptions
    ServiceError,
    ServiceUnavailableError,
    ServiceTimeoutError,
    ServiceAuthenticationError,
    
    # Trading exceptions
    TradingError,
    OrderExecutionError,
    PositionError,
    
    # Model exceptions
    ModelError,
    
    # Security exceptions
    SecurityError,
    AuthenticationError,
    AuthorizationError,
    
    # Resilience exceptions
    ResilienceError,
    CircuitBreakerOpenError,
    RetryExhaustedError,
    TimeoutError,
    BulkheadFullError
)

from core_foundations.utils.logger import get_logger

logger = get_logger(__name__)

# Portfolio Management Service specific exceptions
class PortfolioManagementError(ForexTradingPlatformError):
    """Base exception for portfolio management service errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "PORTFOLIO_MANAGEMENT_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)


class PortfolioNotFoundError(PortfolioManagementError):
    """Exception raised when a portfolio is not found."""
    
    def __init__(
        self,
        message: str = "Portfolio not found",
        portfolio_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if portfolio_id:
            details["portfolio_id"] = portfolio_id
            
        super().__init__(message, "PORTFOLIO_NOT_FOUND_ERROR", details)


class PositionNotFoundError(PortfolioManagementError):
    """Exception raised when a position is not found."""
    
    def __init__(
        self,
        message: str = "Position not found",
        position_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if position_id:
            details["position_id"] = position_id
            
        super().__init__(message, "POSITION_NOT_FOUND_ERROR", details)


class InsufficientBalanceError(PortfolioManagementError):
    """Exception raised when there is insufficient balance for an operation."""
    
    def __init__(
        self,
        message: str = "Insufficient balance",
        required_amount: Optional[float] = None,
        available_amount: Optional[float] = None,
        currency: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if required_amount is not None:
            details["required_amount"] = required_amount
        if available_amount is not None:
            details["available_amount"] = available_amount
        if currency:
            details["currency"] = currency
            
        super().__init__(message, "INSUFFICIENT_BALANCE_ERROR", details)


class PortfolioOperationError(PortfolioManagementError):
    """Exception raised when a portfolio operation fails."""
    
    def __init__(
        self,
        message: str = "Portfolio operation failed",
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if operation:
            details["operation"] = operation
            
        super().__init__(message, "PORTFOLIO_OPERATION_ERROR", details)


class AccountReconciliationError(PortfolioManagementError):
    """Exception raised when account reconciliation fails."""
    
    def __init__(
        self,
        message: str = "Account reconciliation failed",
        account_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if account_id:
            details["account_id"] = account_id
            
        super().__init__(message, "ACCOUNT_RECONCILIATION_ERROR", details)


class TaxCalculationError(PortfolioManagementError):
    """Exception raised when tax calculation fails."""
    
    def __init__(
        self,
        message: str = "Tax calculation failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "TAX_CALCULATION_ERROR", details)


# Error conversion utilities
def convert_to_http_exception(exc: ForexTradingPlatformError) -> HTTPException:
    """
    Convert a ForexTradingPlatformError to an appropriate HTTPException.
    
    Args:
        exc: The exception to convert
        
    Returns:
        An HTTPException with appropriate status code and details
    """
    # Map exception types to status codes
    status_code_map = {
        # 400 Bad Request
        DataValidationError: status.HTTP_400_BAD_REQUEST,
        ConfigValidationError: status.HTTP_400_BAD_REQUEST,
        PortfolioOperationError: status.HTTP_400_BAD_REQUEST,
        
        # 401 Unauthorized
        AuthenticationError: status.HTTP_401_UNAUTHORIZED,
        ServiceAuthenticationError: status.HTTP_401_UNAUTHORIZED,
        
        # 403 Forbidden
        AuthorizationError: status.HTTP_403_FORBIDDEN,
        InsufficientBalanceError: status.HTTP_403_FORBIDDEN,
        
        # 404 Not Found
        PortfolioNotFoundError: status.HTTP_404_NOT_FOUND,
        PositionNotFoundError: status.HTTP_404_NOT_FOUND,
        
        # 408 Request Timeout
        TimeoutError: status.HTTP_408_REQUEST_TIMEOUT,
        
        # 422 Unprocessable Entity
        AccountReconciliationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
        TaxCalculationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
        
        # 429 Too Many Requests
        BulkheadFullError: status.HTTP_429_TOO_MANY_REQUESTS,
        
        # 503 Service Unavailable
        ServiceUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
        CircuitBreakerOpenError: status.HTTP_503_SERVICE_UNAVAILABLE,
        ServiceTimeoutError: status.HTTP_503_SERVICE_UNAVAILABLE,
    }
    
    # Get status code for exception type, default to 500
    for exc_type, code in status_code_map.items():
        if isinstance(exc, exc_type):
            status_code = code
            break
    else:
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    
    # Create HTTPException with details from original exception
    return HTTPException(
        status_code=status_code,
        detail={
            "error_code": exc.error_code,
            "message": exc.message,
            "details": exc.details
        }
    )


# Decorator for exception handling
F = TypeVar('F', bound=Callable)

def with_exception_handling(func: F) -> F:
    """
    Decorator to add standardized exception handling to functions.
    
    Args:
        func: The function to decorate
        
    Returns:
        Decorated function with exception handling
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ForexTradingPlatformError as e:
            logger.error(
                f"{e.__class__.__name__}: {e.message}",
                extra={
                    "error_code": e.error_code,
                    "details": e.details
                }
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error: {str(e)}",
                extra={
                    "traceback": traceback.format_exc()
                }
            )
            # Convert to ForexTradingPlatformError
            raise PortfolioManagementError(
                message=f"Unexpected error: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={"original_error": str(e)}
            )
    
    return wrapper


# Async decorator for exception handling
def async_with_exception_handling(func: F) -> F:
    """
    Decorator to add standardized exception handling to async functions.
    
    Args:
        func: The async function to decorate
        
    Returns:
        Decorated async function with exception handling
    """
    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ForexTradingPlatformError as e:
            logger.error(
                f"{e.__class__.__name__}: {e.message}",
                extra={
                    "error_code": e.error_code,
                    "details": e.details
                }
            )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error: {str(e)}",
                extra={
                    "traceback": traceback.format_exc()
                }
            )
            # Convert to ForexTradingPlatformError
            raise PortfolioManagementError(
                message=f"Unexpected error: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={"original_error": str(e)}
            )
    
    return wrapper