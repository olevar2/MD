"""
Exceptions Bridge Module

This module bridges the common-lib exceptions with risk-management-service specific exceptions.
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
    RiskLimitError,
    
    # Model exceptions
    ModelError,
    ModelTrainingError,
    ModelPredictionError,
    ModelLoadError,
    
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

# Risk Management Service specific exceptions
class RiskManagementError(ForexTradingPlatformError):
    """Base exception for risk management service errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "RISK_MANAGEMENT_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)


class RiskCalculationError(RiskManagementError):
    """Exception raised when risk calculations fail."""
    
    def __init__(
        self,
        message: str = "Risk calculation failed",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "RISK_CALCULATION_ERROR", details)


class RiskParameterError(RiskManagementError):
    """Exception raised when risk parameters are invalid."""
    
    def __init__(
        self,
        message: str = "Invalid risk parameters",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "RISK_PARAMETER_ERROR", details)


class CircuitBreakerError(RiskManagementError):
    """Exception raised when a circuit breaker is triggered."""
    
    def __init__(
        self,
        message: str = "Circuit breaker triggered",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, "CIRCUIT_BREAKER_ERROR", details)


class RiskLimitBreachError(RiskManagementError):
    """Exception raised when a risk limit is breached."""
    
    def __init__(
        self,
        message: str = "Risk limit breached",
        limit_type: Optional[str] = None,
        current_value: Optional[float] = None,
        limit_value: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        limit_type: Description of limit_type
        current_value: Description of current_value
        limit_value: Description of limit_value
        details: Description of details
        Any]]: Description of Any]]
    
    """

        details = details or {}
        if limit_type:
            details["limit_type"] = limit_type
        if current_value is not None:
            details["current_value"] = current_value
        if limit_value is not None:
            details["limit_value"] = limit_value
            
        super().__init__(message, "RISK_LIMIT_BREACH_ERROR", details)


class RiskProfileNotFoundError(RiskManagementError):
    """Exception raised when a risk profile is not found."""
    
    def __init__(
        self,
        message: str = "Risk profile not found",
        profile_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        profile_id: Description of profile_id
        details: Description of details
        Any]]: Description of Any]]
    
    """

        details = details or {}
        if profile_id:
            details["profile_id"] = profile_id
            
        super().__init__(message, "RISK_PROFILE_NOT_FOUND_ERROR", details)


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
        RiskParameterError: status.HTTP_400_BAD_REQUEST,
        
        # 401 Unauthorized
        AuthenticationError: status.HTTP_401_UNAUTHORIZED,
        ServiceAuthenticationError: status.HTTP_401_UNAUTHORIZED,
        
        # 403 Forbidden
        AuthorizationError: status.HTTP_403_FORBIDDEN,
        RiskLimitBreachError: status.HTTP_403_FORBIDDEN,
        CircuitBreakerError: status.HTTP_403_FORBIDDEN,
        
        # 404 Not Found
        RiskProfileNotFoundError: status.HTTP_404_NOT_FOUND,
        
        # 408 Request Timeout
        TimeoutError: status.HTTP_408_REQUEST_TIMEOUT,
        
        # 422 Unprocessable Entity
        RiskCalculationError: status.HTTP_422_UNPROCESSABLE_ENTITY,
        
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
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

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
            raise RiskManagementError(
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
    """
    Wrapper.
    
    Args:
        args: Description of args
        kwargs: Description of kwargs
    
    """

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
            raise RiskManagementError(
                message=f"Unexpected error: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={"original_error": str(e)}
            )
    
    return wrapper
