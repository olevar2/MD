"""
Exceptions Bridge Module

This module bridges the common-lib exceptions with monitoring-alerting-service specific exceptions.
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

logger = logging.getLogger(__name__)

# Monitoring & Alerting Service specific exceptions
class MonitoringAlertingError(ForexTradingPlatformError):
    """Base exception for monitoring and alerting service errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "MONITORING_ALERTING_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(message, error_code, details)


class AlertNotFoundError(MonitoringAlertingError):
    """Exception raised when an alert is not found."""
    
    def __init__(
        self,
        message: str = "Alert not found",
        alert_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        alert_id: Description of alert_id
        details: Description of details
        Any]]: Description of Any]]
    
    """

        details = details or {}
        if alert_id:
            details["alert_id"] = alert_id
            
        super().__init__(message, "ALERT_NOT_FOUND_ERROR", details)


class NotificationError(MonitoringAlertingError):
    """Exception raised when a notification fails to send."""
    
    def __init__(
        self,
        message: str = "Failed to send notification",
        channel: Optional[str] = None,
        alert_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        channel: Description of channel
        alert_id: Description of alert_id
        details: Description of details
        Any]]: Description of Any]]
    
    """

        details = details or {}
        if channel:
            details["channel"] = channel
        if alert_id:
            details["alert_id"] = alert_id
            
        super().__init__(message, "NOTIFICATION_ERROR", details)


class AlertStorageError(MonitoringAlertingError):
    """Exception raised when there's an error storing or retrieving alerts."""
    
    def __init__(
        self,
        message: str = "Alert storage error",
        operation: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        operation: Description of operation
        details: Description of details
        Any]]: Description of Any]]
    
    """

        details = details or {}
        if operation:
            details["operation"] = operation
            
        super().__init__(message, "ALERT_STORAGE_ERROR", details)


class MetricsExporterError(MonitoringAlertingError):
    """Exception raised when there's an error with a metrics exporter."""
    
    def __init__(
        self,
        message: str = "Metrics exporter error",
        exporter: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        exporter: Description of exporter
        details: Description of details
        Any]]: Description of Any]]
    
    """

        details = details or {}
        if exporter:
            details["exporter"] = exporter
            
        super().__init__(message, "METRICS_EXPORTER_ERROR", details)


class DashboardError(MonitoringAlertingError):
    """Exception raised when there's an error with a dashboard."""
    
    def __init__(
        self,
        message: str = "Dashboard error",
        dashboard: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        dashboard: Description of dashboard
        details: Description of details
        Any]]: Description of Any]]
    
    """

        details = details or {}
        if dashboard:
            details["dashboard"] = dashboard
            
        super().__init__(message, "DASHBOARD_ERROR", details)


class AlertRuleError(MonitoringAlertingError):
    """Exception raised when there's an error with an alert rule."""
    
    def __init__(
        self,
        message: str = "Alert rule error",
        rule: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        rule: Description of rule
        details: Description of details
        Any]]: Description of Any]]
    
    """

        details = details or {}
        if rule:
            details["rule"] = rule
            
        super().__init__(message, "ALERT_RULE_ERROR", details)


class ThresholdValidationError(MonitoringAlertingError):
    """Exception raised when a threshold validation fails."""
    
    def __init__(
        self,
        message: str = "Threshold validation error",
        threshold_type: Optional[str] = None,
        value: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
    """
      init  .
    
    Args:
        message: Description of message
        threshold_type: Description of threshold_type
        value: Description of value
        details: Description of details
        Any]]: Description of Any]]
    
    """

        details = details or {}
        if threshold_type:
            details["threshold_type"] = threshold_type
        if value is not None:
            details["value"] = value
            
        super().__init__(message, "THRESHOLD_VALIDATION_ERROR", details)


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
        ThresholdValidationError: status.HTTP_400_BAD_REQUEST,
        AlertRuleError: status.HTTP_400_BAD_REQUEST,
        
        # 401 Unauthorized
        AuthenticationError: status.HTTP_401_UNAUTHORIZED,
        ServiceAuthenticationError: status.HTTP_401_UNAUTHORIZED,
        
        # 403 Forbidden
        AuthorizationError: status.HTTP_403_FORBIDDEN,
        
        # 404 Not Found
        AlertNotFoundError: status.HTTP_404_NOT_FOUND,
        
        # 408 Request Timeout
        TimeoutError: status.HTTP_408_REQUEST_TIMEOUT,
        
        # 422 Unprocessable Entity
        AlertStorageError: status.HTTP_422_UNPROCESSABLE_ENTITY,
        
        # 429 Too Many Requests
        BulkheadFullError: status.HTTP_429_TOO_MANY_REQUESTS,
        
        # 503 Service Unavailable
        ServiceUnavailableError: status.HTTP_503_SERVICE_UNAVAILABLE,
        CircuitBreakerOpenError: status.HTTP_503_SERVICE_UNAVAILABLE,
        ServiceTimeoutError: status.HTTP_503_SERVICE_UNAVAILABLE,
        NotificationError: status.HTTP_503_SERVICE_UNAVAILABLE,
        MetricsExporterError: status.HTTP_503_SERVICE_UNAVAILABLE,
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
            raise MonitoringAlertingError(
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
            raise MonitoringAlertingError(
                message=f"Unexpected error: {str(e)}",
                error_code="UNEXPECTED_ERROR",
                details={"original_error": str(e)}
            )
    
    return wrapper