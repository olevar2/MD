"""
Base Exception Hierarchy for the Forex Trading Platform.

This module defines a comprehensive standardized exception hierarchy to be used across all services.
All platform exceptions should inherit from the appropriate base class to ensure consistent
error handling and reporting.
"""
from typing import Any, Dict, Optional


class DataValidationError(Exception):
    """Exception raised for data validation errors."""

    def __init__(self, message: str, data: Any = None):
        self.message = message
        self.data = data
        super().__init__(self.message)


class ForexTradingPlatformError(Exception):
    """
    Base exception for all platform errors.

    All custom exceptions in the platform should inherit from this class.
    This allows for consistent error handling and logging across services.
    """

    def __init__(self, message: str = None, error_code: str = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            message: Error message
            error_code: Error code for categorization
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

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
        """
        Initialize the exception.

        Args:
            message: Error message
            error_code: Error code for categorization
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = message or "Configuration error"
        error_code = error_code or "CONFIG_ERROR"
        super().__init__(message, error_code, *args, **kwargs)


class ConfigNotFoundError(ConfigurationError):
    """Exception raised when a configuration file or setting is not found."""

    def __init__(self, config_name: str = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            config_name: Name of the missing configuration
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = f"Configuration not found: {config_name}" if config_name else "Configuration not found"
        # Pass config_name to details via kwargs
        kwargs['config_name'] = config_name
        super().__init__(message, "CONFIG_NOT_FOUND", *args, **kwargs)


class ConfigValidationError(ConfigurationError):
    """Exception raised when configuration validation fails."""

    def __init__(self, errors: Dict[str, Any] = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            errors: Dictionary of validation errors
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = "Configuration validation failed"
        # Pass errors to details via kwargs
        if errors:
            kwargs['validation_errors'] = errors
        super().__init__(message, "CONFIG_VALIDATION_ERROR", *args, **kwargs)


class DataError(ForexTradingPlatformError):
    """Base exception for data-related errors."""

    def __init__(self, message: str = None, error_code: str = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            message: Error message
            error_code: Error code for categorization
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = message or "Data error"
        error_code = error_code or "DATA_ERROR"
        super().__init__(message, error_code, *args, **kwargs)


class DataValidationError(DataError):
    """Exception raised for data validation failures."""

    def __init__(self, message: str = None, validation_errors: Dict[str, Any] = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            message: Error message
            validation_errors: Dictionary of validation errors
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = message or "Data validation error"
        # Pass validation_errors to details via kwargs
        if validation_errors:
            kwargs['validation_errors'] = validation_errors
        super().__init__(message, "DATA_VALIDATION_ERROR", *args, **kwargs)


class DataFetchError(DataError):
    """Exception raised when data cannot be fetched from a source."""

    def __init__(self, message: str = None, source: str = None, status_code: int = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            message: Error message
            source: Data source that failed
            status_code: HTTP or other status code
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = message or f"Failed to fetch data from {source or 'unknown source'}"
        # Pass source and status_code to details via kwargs
        kwargs['source'] = source
        if status_code is not None:
            kwargs['status_code'] = status_code
        super().__init__(message, "DATA_FETCH_ERROR", *args, **kwargs)


class DataStorageError(DataError):
    """Exception raised when data cannot be stored."""

    def __init__(self, message: str = None, storage_type: str = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            message: Error message
            storage_type: Type of storage that failed
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = message or f"Failed to store data in {storage_type or 'storage'}"
        # Pass storage_type to details via kwargs
        if storage_type:
            kwargs['storage_type'] = storage_type
        super().__init__(message, "DATA_STORAGE_ERROR", *args, **kwargs)


class DataTransformationError(DataError):
    """Exception raised when data transformation fails."""

    def __init__(self, message: str = None, transformation: str = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            message: Error message
            transformation: Type of transformation that failed
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = message or f"Failed to transform data with {transformation or 'transformation'}"
        # Pass transformation to details via kwargs
        if transformation:
            kwargs['transformation'] = transformation
        super().__init__(message, "DATA_TRANSFORMATION_ERROR", *args, **kwargs)


class DataQualityError(DataError):
    """Exception raised for data quality issues."""

    def __init__(self, message: str = None, quality_check: str = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            message: Error message
            quality_check: Type of quality check that failed
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = message or f"Data quality check failed: {quality_check or 'unknown'}"
        # Pass quality_check to details via kwargs
        if quality_check:
            kwargs['quality_check'] = quality_check
        super().__init__(message, "DATA_QUALITY_ERROR", *args, **kwargs)


class ReconciliationError(DataError):
    """Exception raised when data reconciliation fails."""

    def __init__(self, message: str = None, details: Dict[str, Any] = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            message: Error message
            details: Dictionary with reconciliation details
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = message or "Data reconciliation failed"
        # Pass details to kwargs
        if details:
            kwargs.update(details)
        super().__init__(message, "RECONCILIATION_ERROR", *args, **kwargs)


class ServiceError(ForexTradingPlatformError):
    """Base exception for service-related errors."""

    def __init__(self, message: str = None, service_name: str = None, error_code: str = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            message: Error message
            service_name: Name of the service that failed
            error_code: Error code for categorization
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = message or f"Error in service: {service_name or 'unknown'}"
        error_code = error_code or "SERVICE_ERROR"
        # Pass service_name to details via kwargs
        if service_name:
            kwargs['service_name'] = service_name
        super().__init__(message, error_code, *args, **kwargs)


class ServiceUnavailableError(ServiceError):
    """Exception raised when a dependent service is unavailable."""

    def __init__(self, service_name: str, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            service_name: Name of the unavailable service
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = f"Service unavailable: {service_name}"
        # Pass service_name explicitly to parent and set specific error code
        super().__init__(message, service_name=service_name, error_code="SERVICE_UNAVAILABLE", *args, **kwargs)


class ServiceTimeoutError(ServiceError):
    """Exception raised when a service request times out."""

    def __init__(self, service_name: str, timeout_seconds: float = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            service_name: Name of the service that timed out
            timeout_seconds: Timeout duration in seconds
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = f"Service timeout: {service_name}"
        # Pass timeout_seconds to details via kwargs
        if timeout_seconds is not None:
            message += f" after {timeout_seconds} seconds"
            kwargs['timeout_seconds'] = timeout_seconds
        # Pass service_name explicitly to parent and set specific error code
        super().__init__(message, service_name=service_name, error_code="SERVICE_TIMEOUT", *args, **kwargs)


class AuthenticationError(ForexTradingPlatformError):
    """Exception raised for authentication failures."""

    def __init__(self, message: str = None, *args, **kwargs):
        message = message or "Authentication failed"
        super().__init__(message, "AUTHENTICATION_ERROR", *args, **kwargs)


class AuthorizationError(ForexTradingPlatformError):
    """Exception raised for authorization failures."""

    def __init__(self, message: str = None, resource: str = None, action: str = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            message: Error message
            resource: Resource that was being accessed
            action: Action that was attempted
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        if resource and action:
            message = message or f"Not authorized to {action} on {resource}"
            # Pass resource and action to details via kwargs
            kwargs.update({"resource": resource, "action": action})
        else:
            message = message or "Authorization failed"
        super().__init__(message, "AUTHORIZATION_ERROR", *args, **kwargs)


class TradingError(ForexTradingPlatformError):
    """Base exception for trading-related errors."""

    def __init__(self, message: str = None, error_code: str = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            message: Error message
            error_code: Error code for categorization
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = message or "Trading error"
        error_code = error_code or "TRADING_ERROR"
        super().__init__(message, error_code, *args, **kwargs)


class OrderExecutionError(TradingError):
    """Exception raised when an order fails to execute."""

    def __init__(self, message: str = None, order_id: str = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            message: Error message
            order_id: ID of the order that failed
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = message or f"Failed to execute order {order_id or 'unknown'}"
        # Pass order_id to details via kwargs
        kwargs['order_id'] = order_id
        super().__init__(message, "ORDER_EXECUTION_ERROR", *args, **kwargs)


class PositionError(TradingError):
    """Exception raised for position-related errors."""

    def __init__(self, message: str = None, position_id: str = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            message: Error message
            position_id: ID of the position with an error
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = message or f"Position error for {position_id or 'unknown'}"
        # Pass position_id to details via kwargs
        kwargs['position_id'] = position_id
        super().__init__(message, "POSITION_ERROR", *args, **kwargs)


class RiskLimitError(TradingError):
    """Exception raised when a risk limit is breached."""

    def __init__(self, message: str = None, limit_type: str = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            message: Error message
            limit_type: Type of risk limit that was breached
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = message or f"Risk limit breached: {limit_type or 'unknown'}"
        # Pass limit_type to details via kwargs
        kwargs['limit_type'] = limit_type
        super().__init__(message, "RISK_LIMIT_ERROR", *args, **kwargs)


class ModelError(ForexTradingPlatformError):
    """Base exception for ML model-related errors."""

    def __init__(self, message: str = None, model_id: str = None, error_code: str = None, *args, **kwargs):
        """
        Initialize the exception.

        Args:
            message: Error message
            model_id: ID of the model with an error
            error_code: Error code for categorization
            args: Additional positional arguments
            kwargs: Additional keyword arguments for details
        """

        message = message or f"Model error for {model_id or 'unknown model'}"
        error_code = error_code or "MODEL_ERROR"
        # Pass model_id to details via kwargs
        kwargs['model_id'] = model_id
        super().__init__(message, error_code, *args, **kwargs)


class ModelTrainingError(ModelError):
    """Exception raised when model training fails."""

    def __init__(self, message: str = None, model_id: str = None, *args, **kwargs):
    """
      init  .

    Args:
        message: Description of message
        model_id: Description of model_id
        args: Description of args
        kwargs: Description of kwargs

    """

        message = message or f"Training failed for model {model_id or 'unknown'}"
        super().__init__(message, model_id=model_id, error_code="MODEL_TRAINING_ERROR", *args, **kwargs)


class ModelPredictionError(ModelError):
    """Exception raised when model prediction fails."""

    def __init__(self, message: str = None, model_id: str = None, *args, **kwargs):
    """
      init  .

    Args:
        message: Description of message
        model_id: Description of model_id
        args: Description of args
        kwargs: Description of kwargs

    """

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