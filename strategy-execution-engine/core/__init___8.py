"""
Error handling package for the Strategy Execution Engine.

This package provides error handling utilities and custom exceptions
for the Strategy Execution Engine.
"""

from .exceptions import (
    # Base exceptions
    ForexTradingPlatformError,

    # Data exceptions
    DataValidationError,
    DataFetchError,
    DataStorageError,
    DataTransformationError,

    # Service exceptions
    ServiceError,
    ModelError,

    # Strategy exceptions
    StrategyExecutionError,
    StrategyConfigurationError,
    StrategyLoadError,
    SignalGenerationError,
    OrderGenerationError,

    # Backtest exceptions
    BacktestError,
    BacktestConfigError,
    BacktestDataError,
    BacktestExecutionError,
    BacktestReportError,

    # Risk exceptions
    RiskManagementError
)

from .error_handler import (
    handle_error,
    with_error_handling,
    async_with_error_handling
)

__all__ = [
    # Base exceptions
    'ForexTradingPlatformError',

    # Data exceptions
    'DataValidationError',
    'DataFetchError',
    'DataStorageError',
    'DataTransformationError',

    # Service exceptions
    'ServiceError',
    'ModelError',

    # Strategy exceptions
    'StrategyExecutionError',
    'StrategyConfigurationError',
    'StrategyLoadError',
    'SignalGenerationError',
    'OrderGenerationError',

    # Backtest exceptions
    'BacktestError',
    'BacktestConfigError',
    'BacktestDataError',
    'BacktestExecutionError',
    'BacktestReportError',

    # Risk exceptions
    'RiskManagementError',

    # Error handling utilities
    'handle_error',
    'with_error_handling',
    'async_with_error_handling'
]
