"""
Error Handling Module

This module provides error handling functionality for the Analysis Engine Service.
It extends the common-lib exceptions to provide service-specific error handling.
"""

from typing import Any, Dict, Optional
import logging
import traceback
from fastapi import Request, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import ValidationError as PydanticValidationError
from analysis_engine.core.logging import get_logger
from analysis_engine.core.exceptions_bridge import (
    ForexTradingPlatformError,
    DataValidationError as CommonDataValidationError,
    DataFetchError as CommonDataFetchError,
    DataStorageError as CommonDataStorageError,
    DataTransformationError as CommonDataTransformationError,
    ConfigurationError as CommonConfigurationError,
    ServiceError,
    ServiceUnavailableError as CommonServiceUnavailableError,
    ServiceTimeoutError as CommonServiceTimeoutError,
    ModelError as CommonModelError,
    ModelTrainingError as CommonModelTrainingError,
    ModelPredictionError as CommonModelPredictionError
)

logger = get_logger(__name__)

class AnalysisEngineError(ForexTradingPlatformError):
    """
    Base exception for Analysis Engine Service.

    Extends the ForexTradingPlatformError from common-lib to maintain
    compatibility with the platform-wide error handling.
    """
    def __init__(
        self,
        message: str,
        error_code: str = "ANALYSIS_ENGINE_ERROR",
        details: Optional[Dict[str, Any]] = None,
        *args,
        **kwargs
    ):
        super().__init__(message, error_code, *args, **kwargs)
        # Store status_code for FastAPI response handling
        self.status_code = kwargs.get('status_code', status.HTTP_500_INTERNAL_SERVER_ERROR)

class ValidationError(CommonDataValidationError, AnalysisEngineError):
    """
    Validation error for Analysis Engine Service.

    Extends both CommonDataValidationError and AnalysisEngineError to maintain
    compatibility with both the platform-wide error handling and the service-specific
    error handling.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, *args, **kwargs):
        kwargs['status_code'] = status.HTTP_400_BAD_REQUEST
        super().__init__(message, details=details, *args, **kwargs)
        self.error_code = "VALIDATION_ERROR"

class DataFetchError(CommonDataFetchError, AnalysisEngineError):
    """
    Data fetch error for Analysis Engine Service.

    Extends both CommonDataFetchError and AnalysisEngineError to maintain
    compatibility with both the platform-wide error handling and the service-specific
    error handling.
    """
    def __init__(self, message: str, source: str = None, details: Optional[Dict[str, Any]] = None, *args, **kwargs):
        kwargs['status_code'] = status.HTTP_503_SERVICE_UNAVAILABLE
        super().__init__(message, source=source, *args, **kwargs)
        # Ensure details are properly set
        if details:
            self.details.update(details)

class AnalysisError(AnalysisEngineError):
    """
    Analysis error for Analysis Engine Service.

    Extends AnalysisEngineError to provide specific error handling for analysis operations.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, *args, **kwargs):
        kwargs['status_code'] = status.HTTP_500_INTERNAL_SERVER_ERROR
        super().__init__(message, error_code="ANALYSIS_ERROR", details=details, *args, **kwargs)

class ConfigurationError(CommonConfigurationError, AnalysisEngineError):
    """
    Configuration error for Analysis Engine Service.

    Extends both CommonConfigurationError and AnalysisEngineError to maintain
    compatibility with both the platform-wide error handling and the service-specific
    error handling.
    """
    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None, *args, **kwargs):
        kwargs['status_code'] = status.HTTP_500_INTERNAL_SERVER_ERROR
        super().__init__(message, *args, **kwargs)
        # Ensure details are properly set
        if details:
            self.details.update(details)

class ServiceUnavailableError(CommonServiceUnavailableError, AnalysisEngineError):
    """
    Service unavailable error for Analysis Engine Service.

    Extends both CommonServiceUnavailableError and AnalysisEngineError to maintain
    compatibility with both the platform-wide error handling and the service-specific
    error handling.
    """
    def __init__(self, service_name: str, details: Optional[Dict[str, Any]] = None, *args, **kwargs):
        kwargs['status_code'] = status.HTTP_503_SERVICE_UNAVAILABLE
        super().__init__(service_name, *args, **kwargs)
        # Ensure details are properly set
        if details:
            self.details.update(details)

# Removed custom create_error_response function in favor of using the standardized to_dict() method

async def forex_platform_exception_handler(request: Request, exc: ForexTradingPlatformError) -> JSONResponse:
    """
    Handle ForexTradingPlatformError exceptions.

    This handler processes all exceptions that inherit from ForexTradingPlatformError,
    including both common-lib exceptions and service-specific exceptions.
    """
    # Log the error with context
    logger.error(
        f"ForexTradingPlatformError: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Determine status code - use status_code attribute if available, otherwise use 500
    status_code = getattr(exc, "status_code", status.HTTP_500_INTERNAL_SERVER_ERROR)

    # Use the standardized to_dict() method for response content
    return JSONResponse(
        status_code=status_code,
        content=exc.to_dict(),
    )

async def analysis_engine_exception_handler(request: Request, exc: AnalysisEngineError) -> JSONResponse:
    """
    Handle AnalysisEngineError exceptions.

    This handler is maintained for backward compatibility with existing code.
    New code should use the forex_platform_exception_handler.
    """
    return forex_platform_exception_handler(request, exc)

async def validation_exception_handler(request: Request, exc: ValidationError) -> JSONResponse:
    """
    Handle ValidationError exceptions.

    This handler is maintained for backward compatibility with existing code.
    New code should use the forex_platform_exception_handler.
    """
    return forex_platform_exception_handler(request, exc)

async def data_fetch_exception_handler(request: Request, exc: DataFetchError) -> JSONResponse:
    """
    Handle DataFetchError exceptions.

    This handler is maintained for backward compatibility with existing code.
    New code should use the forex_platform_exception_handler.
    """
    return forex_platform_exception_handler(request, exc)

async def analysis_exception_handler(request: Request, exc: AnalysisError) -> JSONResponse:
    """
    Handle AnalysisError exceptions.

    This handler is maintained for backward compatibility with existing code.
    New code should use the forex_platform_exception_handler.
    """
    return forex_platform_exception_handler(request, exc)

async def configuration_exception_handler(request: Request, exc: ConfigurationError) -> JSONResponse:
    """
    Handle ConfigurationError exceptions.

    This handler is maintained for backward compatibility with existing code.
    New code should use the forex_platform_exception_handler.
    """
    return forex_platform_exception_handler(request, exc)

async def service_unavailable_exception_handler(request: Request, exc: ServiceUnavailableError) -> JSONResponse:
    """
    Handle ServiceUnavailableError exceptions.

    This handler is maintained for backward compatibility with existing code.
    New code should use the forex_platform_exception_handler.
    """
    return forex_platform_exception_handler(request, exc)

async def pydantic_validation_exception_handler(request: Request, exc: PydanticValidationError) -> JSONResponse:
    """
    Handle Pydantic ValidationError exceptions.

    Converts Pydantic validation errors to our custom ValidationError format.
    """
    # Extract errors from the exception
    errors = exc.errors() if hasattr(exc, "errors") else [{"msg": str(exc)}]

    # Log the validation error
    logger.warning(
        f"Validation error for {request.method} {request.url.path}",
        extra={
            "errors": errors,
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Convert Pydantic validation error to our custom ValidationError
    validation_error = ValidationError(
        message="Request validation failed",
        details={"errors": errors}
    )
    return forex_platform_exception_handler(request, validation_error)

async def fastapi_validation_exception_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    """
    Handle FastAPI RequestValidationError exceptions.

    Converts FastAPI validation errors to our custom ValidationError format.
    """
    # Extract errors from the exception
    errors = exc.errors() if hasattr(exc, "errors") else [{"msg": str(exc)}]

    # Log the validation error
    logger.warning(
        f"Validation error for {request.method} {request.url.path}",
        extra={
            "errors": errors,
            "path": request.url.path,
            "method": request.method,
        },
    )

    # Convert FastAPI validation error to our custom ValidationError
    validation_error = ValidationError(
        message="Request validation failed",
        details={"errors": errors}
    )
    return forex_platform_exception_handler(request, validation_error)

# Handlers for common-lib exceptions

async def data_validation_exception_handler(request: Request, exc: CommonDataValidationError) -> JSONResponse:
    """Handle DataValidationError exceptions from common-lib."""
    logger.warning(
        f"Data validation error: {exc.message}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "data": str(exc.data) if hasattr(exc, 'data') else None,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=exc.to_dict(),
    )

async def common_data_fetch_exception_handler(request: Request, exc: CommonDataFetchError) -> JSONResponse:
    """Handle DataFetchError exceptions from common-lib."""
    logger.error(
        f"Data fetch error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=exc.to_dict(),
    )

async def data_storage_exception_handler(request: Request, exc: CommonDataStorageError) -> JSONResponse:
    """Handle DataStorageError exceptions from common-lib."""
    logger.error(
        f"Data storage error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=exc.to_dict(),
    )

async def data_transformation_exception_handler(request: Request, exc: CommonDataTransformationError) -> JSONResponse:
    """Handle DataTransformationError exceptions from common-lib."""
    logger.error(
        f"Data transformation error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=exc.to_dict(),
    )

async def common_configuration_exception_handler(request: Request, exc: CommonConfigurationError) -> JSONResponse:
    """Handle ConfigurationError exceptions from common-lib."""
    logger.error(
        f"Configuration error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=exc.to_dict(),
    )

async def service_error_exception_handler(request: Request, exc: ServiceError) -> JSONResponse:
    """Handle ServiceError exceptions from common-lib."""
    logger.error(
        f"Service error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=exc.to_dict(),
    )

async def common_service_unavailable_exception_handler(request: Request, exc: CommonServiceUnavailableError) -> JSONResponse:
    """Handle ServiceUnavailableError exceptions from common-lib."""
    logger.error(
        f"Service unavailable: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=exc.to_dict(),
    )

async def service_timeout_exception_handler(request: Request, exc: CommonServiceTimeoutError) -> JSONResponse:
    """Handle ServiceTimeoutError exceptions from common-lib."""
    logger.error(
        f"Service timeout: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_504_GATEWAY_TIMEOUT,
        content=exc.to_dict(),
    )

async def model_error_exception_handler(request: Request, exc: CommonModelError) -> JSONResponse:
    """Handle ModelError exceptions from common-lib."""
    logger.error(
        f"Model error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=exc.to_dict(),
    )

async def model_training_exception_handler(request: Request, exc: CommonModelTrainingError) -> JSONResponse:
    """Handle ModelTrainingError exceptions from common-lib."""
    logger.error(
        f"Model training error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=exc.to_dict(),
    )

async def model_prediction_exception_handler(request: Request, exc: CommonModelPredictionError) -> JSONResponse:
    """Handle ModelPredictionError exceptions from common-lib."""
    logger.error(
        f"Model prediction error: {exc.message}",
        extra={
            "error_code": exc.error_code,
            "details": exc.details,
            "path": request.url.path,
            "method": request.method,
        },
    )

    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=exc.to_dict(),
    )

async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Handle generic exceptions.

    Wraps unhandled exceptions in an AnalysisEngineError for consistent error handling.
    """
    # Log the unhandled exception
    logger.error(
        f"Unhandled exception: {str(exc)}",
        extra={
            "path": request.url.path,
            "method": request.method,
            "traceback": traceback.format_exc(),
            "exception_type": exc.__class__.__name__
        },
    )

    # Wrap the exception in an AnalysisEngineError
    wrapped_exc = AnalysisEngineError(
        message="An unexpected error occurred",
        error_code="INTERNAL_SERVER_ERROR",
        details={
            "exception_type": exc.__class__.__name__,
            "exception_message": str(exc),
            # Only include traceback in debug mode
            "traceback": traceback.format_exc() if logger.level <= logging.DEBUG else None
        }
    )
    return forex_platform_exception_handler(request, wrapped_exc)