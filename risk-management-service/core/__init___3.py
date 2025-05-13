"""
Error Handling Package

This package provides standardized error handling for the risk-management-service.
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
    BulkheadFullError,
    
    # Risk Management specific exceptions
    RiskManagementError,
    RiskCalculationError,
    RiskParameterError,
    CircuitBreakerError,
    RiskLimitBreachError,
    RiskProfileNotFoundError,
    
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
    "RiskLimitError",
    
    # Model exceptions
    "ModelError",
    "ModelTrainingError",
    "ModelPredictionError",
    "ModelLoadError",
    
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
    
    # Risk Management specific exceptions
    "RiskManagementError",
    "RiskCalculationError",
    "RiskParameterError",
    "CircuitBreakerError",
    "RiskLimitBreachError",
    "RiskProfileNotFoundError",
    
    # Utility functions
    "convert_to_http_exception",
    "with_exception_handling",
    "async_with_exception_handling",
    "register_exception_handlers",
    "format_error_response",
    "get_correlation_id"
]
