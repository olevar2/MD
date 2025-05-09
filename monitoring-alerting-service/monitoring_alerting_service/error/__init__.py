"""
Error Handling Package

This package provides error handling utilities for the Monitoring & Alerting Service.
"""

from .exceptions_bridge import (
    # Base exceptions
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
    BulkheadFullError,
    
    # Monitoring & Alerting specific exceptions
    MonitoringAlertingError,
    AlertNotFoundError,
    NotificationError,
    AlertStorageError,
    MetricsExporterError,
    DashboardError,
    AlertRuleError,
    ThresholdValidationError,
    
    # Utility functions
    convert_to_http_exception,
    with_exception_handling,
    async_with_exception_handling
)

from .error_handlers import (
    register_exception_handlers,
    format_error_response,
    get_correlation_id
)

__all__ = [
    # Base exceptions
    "ForexTradingPlatformError",
    
    # Configuration exceptions
    "ConfigurationError",
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
    "ServiceAuthenticationError",
    
    # Trading exceptions
    "TradingError",
    "OrderExecutionError",
    "PositionError",
    
    # Model exceptions
    "ModelError",
    
    # Security exceptions
    "SecurityError",
    "AuthenticationError",
    "AuthorizationError",
    
    # Resilience exceptions
    "ResilienceError",
    "CircuitBreakerOpenError",
    "RetryExhaustedError",
    "TimeoutError",
    "BulkheadFullError",
    
    # Monitoring & Alerting specific exceptions
    "MonitoringAlertingError",
    "AlertNotFoundError",
    "NotificationError",
    "AlertStorageError",
    "MetricsExporterError",
    "DashboardError",
    "AlertRuleError",
    "ThresholdValidationError",
    
    # Utility functions
    "convert_to_http_exception",
    "with_exception_handling",
    "async_with_exception_handling",
    "register_exception_handlers",
    "format_error_response",
    "get_correlation_id"
]