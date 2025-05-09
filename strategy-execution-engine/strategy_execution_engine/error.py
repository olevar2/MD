"""
Error Handling for Strategy Execution Engine

This module provides error handling for the Strategy Execution Engine.
"""

import logging
import functools
import traceback
from typing import Any, Callable, TypeVar, cast, Dict, Optional, Type, Union
from datetime import datetime

from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# Type variables for function decorators
F = TypeVar('F', bound=Callable[..., Any])
AsyncF = TypeVar('AsyncF', bound=Callable[..., Any])

class ForexTradingPlatformError(Exception):
    """Base exception for all Forex Trading Platform errors."""
    
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            code: Error code
            details: Additional error details
        """
        self.message = message
        self.code = code or "FOREX_PLATFORM_ERROR"
        self.details = details or {}
        self.timestamp = datetime.utcnow().isoformat()
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert exception to dictionary.
        
        Returns:
            Dict: Exception as dictionary
        """
        return {
            "error": self.code,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
            "type": self.__class__.__name__
        }

class StrategyExecutionError(ForexTradingPlatformError):
    """Base exception for strategy execution errors."""
    
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            code: Error code
            details: Additional error details
        """
        super().__init__(message, code or "STRATEGY_EXECUTION_ERROR", details)

class StrategyConfigurationError(StrategyExecutionError):
    """Exception for strategy configuration errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, "STRATEGY_CONFIGURATION_ERROR", details)

class StrategyLoadError(StrategyExecutionError):
    """Exception for strategy loading errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, "STRATEGY_LOAD_ERROR", details)

class StrategyExecutionTimeoutError(StrategyExecutionError):
    """Exception for strategy execution timeout errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, "STRATEGY_EXECUTION_TIMEOUT", details)

class BacktestError(ForexTradingPlatformError):
    """Base exception for backtest errors."""
    
    def __init__(self, message: str, code: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            code: Error code
            details: Additional error details
        """
        super().__init__(message, code or "BACKTEST_ERROR", details)

class BacktestConfigError(BacktestError):
    """Exception for backtest configuration errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, "BACKTEST_CONFIG_ERROR", details)

class BacktestDataError(BacktestError):
    """Exception for backtest data errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, "BACKTEST_DATA_ERROR", details)

class BacktestExecutionError(BacktestError):
    """Exception for backtest execution errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, "BACKTEST_EXECUTION_ERROR", details)

class BacktestReportError(BacktestError):
    """Exception for backtest report errors."""
    
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        """
        Initialize exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(message, "BACKTEST_REPORT_ERROR", details)

def with_error_handling(func: F) -> F:
    """
    Decorator for handling errors in synchronous functions.
    
    Args:
        func: Function to decorate
        
    Returns:
        F: Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return func(*args, **kwargs)
        except ForexTradingPlatformError as e:
            logger.error(f"{e.__class__.__name__}: {e.message}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unhandled exception in {func.__name__}: {str(e)}", exc_info=True)
            raise StrategyExecutionError(
                f"An unexpected error occurred: {str(e)}",
                details={"traceback": traceback.format_exc()}
            )
    
    return cast(F, wrapper)

def async_with_error_handling(func: AsyncF) -> AsyncF:
    """
    Decorator for handling errors in asynchronous functions.
    
    Args:
        func: Function to decorate
        
    Returns:
        AsyncF: Decorated function
    """
    @functools.wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return await func(*args, **kwargs)
        except ForexTradingPlatformError as e:
            logger.error(f"{e.__class__.__name__}: {e.message}", exc_info=True)
            raise
        except Exception as e:
            logger.error(f"Unhandled exception in {func.__name__}: {str(e)}", exc_info=True)
            raise StrategyExecutionError(
                f"An unexpected error occurred: {str(e)}",
                details={"traceback": traceback.format_exc()}
            )
    
    return cast(AsyncF, wrapper)

def map_exception_to_http(exc: Exception) -> HTTPException:
    """
    Map exception to HTTP exception.
    
    Args:
        exc: Exception to map
        
    Returns:
        HTTPException: HTTP exception
    """
    if isinstance(exc, StrategyConfigurationError) or isinstance(exc, BacktestConfigError):
        return HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc)
        )
    elif isinstance(exc, StrategyLoadError):
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(exc)
        )
    elif isinstance(exc, ForexTradingPlatformError):
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(exc)
        )
    else:
        return HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred"
        )
