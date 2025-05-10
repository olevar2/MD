"""
Exceptions Bridge Module

This module serves as a bridge between common-lib exceptions and service-specific exceptions.
It imports and re-exports common-lib exceptions, making it easier to manage exception imports
and future changes to the exception hierarchy.

Usage:
    from analysis_engine.core.exceptions_bridge import (
        ForexTradingPlatformError,
        DataValidationError,
        DataFetchError,
        ServiceUnavailableError,
        # etc.
    )
"""

import uuid
import logging
import traceback
from functools import wraps
from typing import Dict, Any, Optional, Type, List, Union, Callable
from fastapi import status

# Import common-lib exceptions
from common_lib.exceptions import (
    # Base exception
    ForexTradingPlatformError,

    # Configuration exceptions
    ConfigurationError,
    ConfigNotFoundError,
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

    # Authentication/Authorization exceptions
    AuthenticationError,
    AuthorizationError,

    # Model exceptions
    ModelError,
    ModelTrainingError,
    ModelPredictionError,

    # Utility function
    get_all_exception_classes
)

# Analysis-specific exceptions
class AnalysisError(ForexTradingPlatformError):
    """Base exception for analysis-related errors."""

    def __init__(
        self,
        message: str,
        error_code: str = "ANALYSIS_ERROR",
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        status_code: int = status.HTTP_400_BAD_REQUEST,
        *args,
        **kwargs
    ):
        super().__init__(message, error_code, details, *args, **kwargs)
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.status_code = status_code


class AnalyzerNotFoundError(AnalysisError):
    """Exception raised when a requested analyzer is not found."""

    def __init__(
        self,
        analyzer_name: str,
        available_analyzers: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
        *args,
        **kwargs
    ):
        details = {
            "analyzer_name": analyzer_name,
            "available_analyzers": available_analyzers or []
        }
        super().__init__(
            message=f"Analyzer '{analyzer_name}' not found",
            error_code="ANALYZER_NOT_FOUND",
            details=details,
            correlation_id=correlation_id,
            status_code=status.HTTP_404_NOT_FOUND,
            *args,
            **kwargs
        )


class InsufficientDataError(AnalysisError):
    """Exception raised when there is insufficient data for analysis."""

    def __init__(
        self,
        message: str = "Insufficient data for analysis",
        required_points: Optional[int] = None,
        available_points: Optional[int] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        correlation_id: Optional[str] = None,
        *args,
        **kwargs
    ):
        details = {
            "symbol": symbol,
            "timeframe": timeframe
        }

        if required_points is not None:
            details["required_points"] = required_points

        if available_points is not None:
            details["available_points"] = available_points

        super().__init__(
            message=message,
            error_code="INSUFFICIENT_DATA",
            details=details,
            correlation_id=correlation_id,
            status_code=status.HTTP_400_BAD_REQUEST,
            *args,
            **kwargs
        )


class InvalidAnalysisParametersError(AnalysisError):
    """Exception raised when invalid parameters are provided for analysis."""

    def __init__(
        self,
        message: str = "Invalid analysis parameters",
        parameters: Optional[Dict[str, Any]] = None,
        validation_errors: Optional[List[Dict[str, Any]]] = None,
        correlation_id: Optional[str] = None,
        *args,
        **kwargs
    ):
        details = {}

        if parameters is not None:
            details["parameters"] = parameters

        if validation_errors is not None:
            details["validation_errors"] = validation_errors

        super().__init__(
            message=message,
            error_code="INVALID_ANALYSIS_PARAMETERS",
            details=details,
            correlation_id=correlation_id,
            status_code=status.HTTP_400_BAD_REQUEST,
            *args,
            **kwargs
        )


class AnalysisTimeoutError(AnalysisError):
    """Exception raised when analysis takes too long to complete."""

    def __init__(
        self,
        analyzer_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        correlation_id: Optional[str] = None,
        *args,
        **kwargs
    ):
        details = {}

        if analyzer_name is not None:
            details["analyzer_name"] = analyzer_name

        if timeout_seconds is not None:
            details["timeout_seconds"] = timeout_seconds

        super().__init__(
            message=f"Analysis timed out after {timeout_seconds} seconds" if timeout_seconds else "Analysis timed out",
            error_code="ANALYSIS_TIMEOUT",
            details=details,
            correlation_id=correlation_id,
            status_code=status.HTTP_408_REQUEST_TIMEOUT,
            *args,
            **kwargs
        )


class MarketRegimeError(AnalysisError):
    """Exception raised when there is an error in market regime analysis."""

    def __init__(
        self,
        message: str = "Error in market regime analysis",
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        correlation_id: Optional[str] = None,
        *args,
        **kwargs
    ):
        details = {}

        if symbol is not None:
            details["symbol"] = symbol

        if timeframe is not None:
            details["timeframe"] = timeframe

        super().__init__(
            message=message,
            error_code="MARKET_REGIME_ERROR",
            details=details,
            correlation_id=correlation_id,
            status_code=status.HTTP_400_BAD_REQUEST,
            *args,
            **kwargs
        )


class SignalQualityError(AnalysisError):
    """Exception raised when there is an error in signal quality evaluation."""

    def __init__(
        self,
        message: str = "Error in signal quality evaluation",
        signal_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        *args,
        **kwargs
    ):
        details = {}

        if signal_id is not None:
            details["signal_id"] = signal_id

        super().__init__(
            message=message,
            error_code="SIGNAL_QUALITY_ERROR",
            details=details,
            correlation_id=correlation_id,
            status_code=status.HTTP_400_BAD_REQUEST,
            *args,
            **kwargs
        )


class ToolEffectivenessError(AnalysisError):
    """Exception raised when there is an error in tool effectiveness analysis."""

    def __init__(
        self,
        message: str = "Error in tool effectiveness analysis",
        tool_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        *args,
        **kwargs
    ):
        details = {}

        if tool_id is not None:
            details["tool_id"] = tool_id

        super().__init__(
            message=message,
            error_code="TOOL_EFFECTIVENESS_ERROR",
            details=details,
            correlation_id=correlation_id,
            status_code=status.HTTP_400_BAD_REQUEST,
            *args,
            **kwargs
        )


class NLPAnalysisError(AnalysisError):
    """Exception raised when there is an error in NLP analysis."""

    def __init__(
        self,
        message: str = "Error in NLP analysis",
        text: Optional[str] = None,
        correlation_id: Optional[str] = None,
        *args,
        **kwargs
    ):
        details = {}

        if text is not None:
            # Truncate text to avoid huge error messages
            details["text"] = text[:100] + "..." if len(text) > 100 else text

        super().__init__(
            message=message,
            error_code="NLP_ANALYSIS_ERROR",
            details=details,
            correlation_id=correlation_id,
            status_code=status.HTTP_400_BAD_REQUEST,
            *args,
            **kwargs
        )


class CorrelationAnalysisError(AnalysisError):
    """Exception raised when there is an error in correlation analysis."""

    def __init__(
        self,
        message: str = "Error in correlation analysis",
        symbols: Optional[List[str]] = None,
        correlation_id: Optional[str] = None,
        *args,
        **kwargs
    ):
        details = {}

        if symbols is not None:
            details["symbols"] = symbols

        super().__init__(
            message=message,
            error_code="CORRELATION_ANALYSIS_ERROR",
            details=details,
            correlation_id=correlation_id,
            status_code=status.HTTP_400_BAD_REQUEST,
            *args,
            **kwargs
        )


class ManipulationDetectionError(AnalysisError):
    """Exception raised when there is an error in manipulation detection."""

    def __init__(
        self,
        message: str = "Error in manipulation detection",
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        correlation_id: Optional[str] = None,
        *args,
        **kwargs
    ):
        details = {}

        if symbol is not None:
            details["symbol"] = symbol

        if timeframe is not None:
            details["timeframe"] = timeframe

        super().__init__(
            message=message,
            error_code="MANIPULATION_DETECTION_ERROR",
            details=details,
            correlation_id=correlation_id,
            status_code=status.HTTP_400_BAD_REQUEST,
            *args,
            **kwargs
        )


# Service-specific exceptions
class ServiceInitializationError(ServiceError):
    """Exception raised when a service fails to initialize."""

    def __init__(
        self,
        message: str = "Service initialization failed",
        service_name: Optional[str] = None,
        error_code: str = "SERVICE_INITIALIZATION_ERROR",
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        *args,
        **kwargs
    ):
        if service_name:
            message = f"{message} for service {service_name}"

        super().__init__(
            message=message,
            service_name=service_name or "unknown",
            error_code=error_code,
            details=details,
            *args,
            **kwargs
        )
        self.correlation_id = correlation_id or str(uuid.uuid4())


class ServiceResolutionError(ServiceError):
    """Exception raised when a service cannot be resolved from the container."""

    def __init__(
        self,
        message: str = "Service resolution failed",
        service_name: Optional[str] = None,
        error_code: str = "SERVICE_RESOLUTION_ERROR",
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        *args,
        **kwargs
    ):
        if service_name:
            message = f"{message} for service {service_name}"

        super().__init__(
            message=message,
            service_name=service_name or "unknown",
            error_code=error_code,
            details=details,
            *args,
            **kwargs
        )
        self.correlation_id = correlation_id or str(uuid.uuid4())


class ServiceCleanupError(ServiceError):
    """Exception raised when a service fails to clean up."""

    def __init__(
        self,
        message: str = "Service cleanup failed",
        service_name: Optional[str] = None,
        error_code: str = "SERVICE_CLEANUP_ERROR",
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        *args,
        **kwargs
    ):
        if service_name:
            message = f"{message} for service {service_name}"

        super().__init__(
            message=message,
            service_name=service_name or "unknown",
            error_code=error_code,
            details=details,
            *args,
            **kwargs
        )
        self.correlation_id = correlation_id or str(uuid.uuid4())


# Decorator for exception handling in synchronous functions
def with_exception_handling(func):
    """
    Decorator for handling exceptions in synchronous functions.

    This decorator catches exceptions, logs them, and re-raises them
    with appropriate context information.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except ForexTradingPlatformError:
            # Re-raise platform-specific exceptions
            raise
        except Exception as e:
            # Get function name and module for context
            func_name = func.__name__
            module_name = func.__module__

            # Get correlation ID from kwargs or generate a new one
            correlation_id = kwargs.get('correlation_id', str(uuid.uuid4()))

            # Log the error
            logging.error(
                f"Error in {module_name}.{func_name}: {str(e)}",
                extra={
                    "correlation_id": correlation_id,
                    "function": func_name,
                    "module": module_name,
                    "args": str(args),
                    "kwargs": str(kwargs)
                },
                exc_info=True
            )

            # Wrap the exception in a ForexTradingPlatformError
            raise ForexTradingPlatformError(
                message=f"Error in {func_name}: {str(e)}",
                error_code="FUNCTION_EXECUTION_ERROR",
                details={
                    "function": func_name,
                    "module": module_name,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            ) from e

    return wrapper


# Decorator for exception handling in asynchronous functions
def async_with_exception_handling(func):
    """
    Decorator for handling exceptions in asynchronous functions.

    This decorator catches exceptions, logs them, and re-raises them
    with appropriate context information.
    """
    @wraps(func)
    async def wrapper(*args, **kwargs):
        try:
            return await func(*args, **kwargs)
        except ForexTradingPlatformError:
            # Re-raise platform-specific exceptions
            raise
        except Exception as e:
            # Get function name and module for context
            func_name = func.__name__
            module_name = func.__module__

            # Get correlation ID from kwargs or generate a new one
            correlation_id = kwargs.get('correlation_id', str(uuid.uuid4()))

            # Log the error
            logging.error(
                f"Error in {module_name}.{func_name}: {str(e)}",
                extra={
                    "correlation_id": correlation_id,
                    "function": func_name,
                    "module": module_name,
                    "args": str(args),
                    "kwargs": str(kwargs)
                },
                exc_info=True
            )

            # Wrap the exception in a ForexTradingPlatformError
            raise ForexTradingPlatformError(
                message=f"Error in {func_name}: {str(e)}",
                error_code="FUNCTION_EXECUTION_ERROR",
                details={
                    "function": func_name,
                    "module": module_name,
                    "error": str(e),
                    "traceback": traceback.format_exc()
                }
            ) from e

    return wrapper


# Helper function to generate a correlation ID
def generate_correlation_id() -> str:
    """Generate a unique correlation ID for tracking errors."""
    return str(uuid.uuid4())


# Helper function to extract correlation ID from request headers
def get_correlation_id_from_request(request: Any) -> str:
    """Extract correlation ID from request headers or generate a new one."""
    if hasattr(request, "headers") and "X-Correlation-ID" in request.headers:
        return request.headers["X-Correlation-ID"]
    return generate_correlation_id()


# Re-export all imported exceptions and add analysis-specific exceptions
__all__ = [
    # Base exception
    "ForexTradingPlatformError",

    # Configuration exceptions
    "ConfigurationError",
    "ConfigNotFoundError",
    "ConfigValidationError",

    # Data exceptions
    "DataError",
    "DataValidationError",
    "DataFetchError",
    "DataStorageError",
    "DataTransformationError",

    # Service exceptions
    "ServiceError",
    "ServiceUnavailableError",
    "ServiceTimeoutError",

    # Authentication/Authorization exceptions
    "AuthenticationError",
    "AuthorizationError",

    # Model exceptions
    "ModelError",
    "ModelTrainingError",
    "ModelPredictionError",

    # Analysis-specific exceptions
    "AnalysisError",
    "AnalyzerNotFoundError",
    "InsufficientDataError",
    "InvalidAnalysisParametersError",
    "AnalysisTimeoutError",
    "MarketRegimeError",
    "SignalQualityError",
    "ToolEffectivenessError",
    "NLPAnalysisError",
    "CorrelationAnalysisError",
    "ManipulationDetectionError",

    # Service-specific exceptions
    "ServiceInitializationError",
    "ServiceResolutionError",
    "ServiceCleanupError",

    # Helper functions
    "generate_correlation_id",
    "get_correlation_id_from_request",
    "with_exception_handling",
    "async_with_exception_handling",

    # Utility function
    "get_all_exception_classes"
]

# Dictionary mapping common exception names to their classes
# This can be used for dynamic exception handling or mapping
EXCEPTION_MAP: Dict[str, Type[ForexTradingPlatformError]] = {
    name: cls for name, cls in get_all_exception_classes().items()
}
