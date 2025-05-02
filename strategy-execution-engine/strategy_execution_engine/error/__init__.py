"""
Error handling package for the Strategy Execution Engine.

This package provides error handling utilities and custom exceptions
for the Strategy Execution Engine.
"""

from .exceptions import (
    ForexTradingPlatformError,
    DataValidationError,
    DataFetchError,
    DataStorageError,
    DataTransformationError,
    ServiceError,
    ModelError,
    StrategyExecutionError,
    StrategyConfigurationError,
    StrategyLoadError,
    SignalGenerationError,
    OrderGenerationError,
    BacktestError,
    RiskManagementError
)

from .error_handler import (
    handle_error,
    with_error_handling,
    async_with_error_handling
)
