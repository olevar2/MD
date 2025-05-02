"""
Mock exceptions module for testing.

This module provides mock implementations of the common-lib exceptions
for testing purposes.
"""
from typing import Any, Dict, Optional


class ForexTradingPlatformError(Exception):
    """
    Base exception for all platform errors.

    All custom exceptions in the platform should inherit from this class.
    This allows for consistent error handling and logging across services.
    """

    def __init__(self, message: str = None, error_code: str = None, *args, **kwargs):
        self.message = message or "An error occurred in the Forex Trading Platform"
        self.error_code = error_code or "FOREX_PLATFORM_ERROR"
        super().__init__(self.message, *args)
        # Store additional details passed as keyword arguments
        self.details = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert the exception to a dictionary for JSON serialization."""
        result = {
            "error_type": self.__class__.__name__,
            "error_code": self.error_code,
            "message": self.message
        }

        if self.details:
            result["details"] = self.details

        return result


class ConfigurationError(ForexTradingPlatformError):
    """Exception raised for configuration-related errors."""

    def __init__(self, message: str = None, error_code: str = None, *args, **kwargs):
        message = message or "Configuration error"
        error_code = error_code or "CONFIG_ERROR"
        super().__init__(message, error_code, *args, **kwargs)


class DataError(ForexTradingPlatformError):
    """Base exception for data-related errors."""

    def __init__(self, message: str = None, error_code: str = None, *args, **kwargs):
        message = message or "Data error"
        error_code = error_code or "DATA_ERROR"
        super().__init__(message, error_code, *args, **kwargs)


class DataValidationError(DataError):
    """Exception raised for data validation failures."""

    def __init__(self, message: str = None, validation_errors: Dict[str, Any] = None, *args, **kwargs):
        message = message or "Data validation error"
        # Pass validation_errors to details via kwargs
        if validation_errors:
            kwargs['validation_errors'] = validation_errors
        super().__init__(message, "DATA_VALIDATION_ERROR", *args, **kwargs)


class DataFetchError(DataError):
    """Exception raised when data cannot be fetched from a source."""

    def __init__(self, message: str = None, source: str = None, status_code: int = None, *args, **kwargs):
        message = message or f"Failed to fetch data from {source or 'unknown source'}"
        # Pass source and status_code to details via kwargs
        kwargs['source'] = source
        if status_code is not None:
            kwargs['status_code'] = status_code
        super().__init__(message, "DATA_FETCH_ERROR", *args, **kwargs)


class ServiceError(ForexTradingPlatformError):
    """Base exception for service-related errors."""

    def __init__(self, message: str = None, service_name: str = None, error_code: str = None, *args, **kwargs):
        message = message or f"Error in service: {service_name or 'unknown'}"
        error_code = error_code or "SERVICE_ERROR"
        # Pass service_name to details via kwargs
        if service_name:
            kwargs['service_name'] = service_name
        super().__init__(message, error_code, *args, **kwargs)


class ServiceUnavailableError(ServiceError):
    """Exception raised when a dependent service is unavailable."""

    def __init__(self, service_name: str, *args, **kwargs):
        message = f"Service unavailable: {service_name}"
        # Pass service_name explicitly to parent and set specific error code
        super().__init__(message, service_name=service_name, error_code="SERVICE_UNAVAILABLE", *args, **kwargs)


class ModelError(ForexTradingPlatformError):
    """Base exception for ML model-related errors."""

    def __init__(self, message: str = None, model_id: str = None, error_code: str = None, *args, **kwargs):
        message = message or f"Model error for {model_id or 'unknown model'}"
        error_code = error_code or "MODEL_ERROR"
        # Pass model_id to details via kwargs
        kwargs['model_id'] = model_id
        super().__init__(message, error_code, *args, **kwargs)


class ModelTrainingError(ModelError):
    """Exception raised when model training fails."""

    def __init__(self, message: str = None, model_id: str = None, *args, **kwargs):
        message = message or f"Training failed for model {model_id or 'unknown'}"
        super().__init__(message, model_id=model_id, error_code="MODEL_TRAINING_ERROR", *args, **kwargs)


class ModelPredictionError(ModelError):
    """Exception raised when model prediction fails."""

    def __init__(self, message: str = None, model_id: str = None, *args, **kwargs):
        message = message or f"Prediction failed for model {model_id or 'unknown'}"
        super().__init__(message, model_id=model_id, error_code="MODEL_PREDICTION_ERROR", *args, **kwargs)


# Add a convenience function to get all exception classes
def get_all_exception_classes():
    """Return all exception classes defined in this module."""
    import sys
    import inspect

    return {
        name: cls for name, cls in inspect.getmembers(sys.modules[__name__], inspect.isclass)
        if issubclass(cls, ForexTradingPlatformError) and cls is not ForexTradingPlatformError
    }
