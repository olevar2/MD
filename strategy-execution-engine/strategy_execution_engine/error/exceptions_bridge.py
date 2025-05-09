"""
Exceptions Bridge Module

This module bridges the common-lib exceptions with strategy-execution-engine specific exceptions.
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
    
    # Trading exceptions
    TradingError,
    OrderExecutionError,
    PositionError,
    
    # Model exceptions
    ModelError,
    
    # Security exceptions
    AuthenticationError,
    AuthorizationError
)

logger = logging.getLogger(__name__)

# Strategy Execution Engine specific exceptions
class StrategyExecutionError(ForexTradingPlatformError):
    """Base exception for strategy execution errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "STRATEGY_EXECUTION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details=details)


class BacktestError(StrategyExecutionError):
    """Base exception for backtesting errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "BACKTEST_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details=details)


class BacktestConfigError(BacktestError):
    """Exception raised when backtest configuration is invalid."""
    
    def __init__(
        self,
        message: str = "Invalid backtest configuration",
        config_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if config_name:
            details["config_name"] = config_name
            
        super().__init__(message, "BACKTEST_CONFIG_ERROR", details)


class BacktestDataError(BacktestError):
    """Exception raised when backtest data is invalid or missing."""
    
    def __init__(
        self,
        message: str = "Invalid or missing backtest data",
        data_source: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if data_source:
            details["data_source"] = data_source
            
        super().__init__(message, "BACKTEST_DATA_ERROR", details)


class BacktestExecutionError(BacktestError):
    """Exception raised when backtest execution fails."""
    
    def __init__(
        self,
        message: str = "Backtest execution failed",
        backtest_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if backtest_id:
            details["backtest_id"] = backtest_id
            
        super().__init__(message, "BACKTEST_EXECUTION_ERROR", details)


class StrategyValidationError(StrategyExecutionError):
    """Exception raised when strategy validation fails."""
    
    def __init__(
        self,
        message: str = "Strategy validation failed",
        strategy_id: Optional[str] = None,
        validation_errors: Optional[Dict[str, Any]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if strategy_id:
            details["strategy_id"] = strategy_id
        if validation_errors:
            details["validation_errors"] = validation_errors
            
        super().__init__(message, "STRATEGY_VALIDATION_ERROR", details)


class StrategyExecutionTimeoutError(StrategyExecutionError):
    """Exception raised when strategy execution times out."""
    
    def __init__(
        self,
        message: str = "Strategy execution timed out",
        strategy_id: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if strategy_id:
            details["strategy_id"] = strategy_id
        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds
            
        super().__init__(message, "STRATEGY_EXECUTION_TIMEOUT", details)


class StrategyNotFoundError(StrategyExecutionError):
    """Exception raised when a strategy is not found."""
    
    def __init__(
        self,
        message: str = "Strategy not found",
        strategy_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if strategy_id:
            details["strategy_id"] = strategy_id
            
        super().__init__(message, "STRATEGY_NOT_FOUND", details)


class BacktestReportError(BacktestError):
    """Exception raised when generating a backtest report fails."""
    
    def __init__(
        self,
        message: str = "Failed to generate backtest report",
        backtest_id: Optional[str] = None,
        report_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if backtest_id:
            details["backtest_id"] = backtest_id
        if report_type:
            details["report_type"] = report_type
            
        super().__init__(message, "BACKTEST_REPORT_ERROR", details)


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
        StrategyValidationError: status.HTTP_400_BAD_REQUEST,
        BacktestConfigError: status.HTTP_400_BAD_REQUEST,
        
        # 401 Unauthorized
        AuthenticationError: status.HTTP_401_UNAUTHORIZED,
        
        # 403 Forbidden
        AuthorizationError: status.HTTP_403_FORBIDDEN,
        
        # 404 Not Found
        StrategyNotFoundError: status.HTTP_404_NOT_FOUND,
        
        # 408 Request Timeout
        StrategyExecutionTimeoutError: status.HTTP_408_REQUEST_TIMEOUT,
        
        # 422 Unprocessable Entity
        BacktestDataError: status.HTTP_422_UNPROCESSABLE_ENTITY,
        
        # 503 Service Unavailable
        ServiceUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
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
            raise StrategyExecutionError(
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
            raise StrategyExecutionError(
                message=f"Unexpected error: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={"original_error": str(e)}
            )
    
    return wrapper
