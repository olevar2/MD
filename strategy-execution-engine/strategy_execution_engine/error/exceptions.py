"""
Custom exceptions for the Strategy Execution Engine.

This module defines custom exceptions that align with the common-lib exceptions
used throughout the Forex Trading Platform.
"""

from typing import Any, Dict, Optional

# Import common-lib exceptions if available
try:
    from common_lib.exceptions import (
        ForexTradingPlatformError,
        DataValidationError,
        DataFetchError,
        DataStorageError,
        DataTransformationError,
        ServiceError,
        ModelError
    )
except ImportError:
    # Define base exception if common-lib is not available
    class ForexTradingPlatformError(Exception):
        """Base exception for all Forex Trading Platform errors."""
        
        def __init__(
            self,
            message: str,
            error_code: str = "FOREX_PLATFORM_ERROR",
            details: Optional[Dict[str, Any]] = None
        ):
            """
            Initialize the exception.
            
            Args:
                message: Human-readable error message
                error_code: Error code for categorization
                details: Additional error details
            """
            self.message = message
            self.error_code = error_code
            self.details = details or {}
            super().__init__(self.message)
        
        def to_dict(self) -> Dict[str, Any]:
            """
            Convert the exception to a dictionary.
            
            Returns:
                Dictionary representation of the exception
            """
            return {
                "error_type": self.__class__.__name__,
                "message": self.message,
                "error_code": self.error_code,
                "details": self.details
            }

    # Define other exceptions if common-lib is not available
    class DataValidationError(ForexTradingPlatformError):
        """Error related to data validation."""
        
        def __init__(
            self,
            message: str,
            data: Any = None,
            details: Optional[Dict[str, Any]] = None
        ):
            details = details or {}
            if data is not None:
                details["data"] = str(data)
            super().__init__(
                message=message,
                error_code="DATA_VALIDATION_ERROR",
                details=details
            )
            self.data = data

    class DataFetchError(ForexTradingPlatformError):
        """Error related to data fetching."""
        
        def __init__(
            self,
            message: str,
            details: Optional[Dict[str, Any]] = None
        ):
            super().__init__(
                message=message,
                error_code="DATA_FETCH_ERROR",
                details=details
            )

    class DataStorageError(ForexTradingPlatformError):
        """Error related to data storage."""
        
        def __init__(
            self,
            message: str,
            details: Optional[Dict[str, Any]] = None
        ):
            super().__init__(
                message=message,
                error_code="DATA_STORAGE_ERROR",
                details=details
            )

    class DataTransformationError(ForexTradingPlatformError):
        """Error related to data transformation."""
        
        def __init__(
            self,
            message: str,
            details: Optional[Dict[str, Any]] = None
        ):
            super().__init__(
                message=message,
                error_code="DATA_TRANSFORMATION_ERROR",
                details=details
            )

    class ServiceError(ForexTradingPlatformError):
        """Error related to service operation."""
        
        def __init__(
            self,
            message: str,
            details: Optional[Dict[str, Any]] = None
        ):
            super().__init__(
                message=message,
                error_code="SERVICE_ERROR",
                details=details
            )

    class ModelError(ForexTradingPlatformError):
        """Error related to model operation."""
        
        def __init__(
            self,
            message: str,
            details: Optional[Dict[str, Any]] = None
        ):
            super().__init__(
                message=message,
                error_code="MODEL_ERROR",
                details=details
            )


# Strategy Execution Engine specific exceptions
class StrategyExecutionError(ForexTradingPlatformError):
    """Error related to strategy execution."""
    
    def __init__(
        self,
        message: str,
        strategy_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if strategy_name:
            details["strategy_name"] = strategy_name
        super().__init__(
            message=message,
            error_code="STRATEGY_EXECUTION_ERROR",
            details=details
        )
        self.strategy_name = strategy_name


class StrategyConfigurationError(ForexTradingPlatformError):
    """Error related to strategy configuration."""
    
    def __init__(
        self,
        message: str,
        strategy_name: Optional[str] = None,
        config_key: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if strategy_name:
            details["strategy_name"] = strategy_name
        if config_key:
            details["config_key"] = config_key
        super().__init__(
            message=message,
            error_code="STRATEGY_CONFIGURATION_ERROR",
            details=details
        )
        self.strategy_name = strategy_name
        self.config_key = config_key


class StrategyLoadError(ForexTradingPlatformError):
    """Error related to strategy loading."""
    
    def __init__(
        self,
        message: str,
        strategy_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if strategy_name:
            details["strategy_name"] = strategy_name
        super().__init__(
            message=message,
            error_code="STRATEGY_LOAD_ERROR",
            details=details
        )
        self.strategy_name = strategy_name


class SignalGenerationError(ForexTradingPlatformError):
    """Error related to signal generation."""
    
    def __init__(
        self,
        message: str,
        strategy_name: Optional[str] = None,
        symbol: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if strategy_name:
            details["strategy_name"] = strategy_name
        if symbol:
            details["symbol"] = symbol
        super().__init__(
            message=message,
            error_code="SIGNAL_GENERATION_ERROR",
            details=details
        )
        self.strategy_name = strategy_name
        self.symbol = symbol


class OrderGenerationError(ForexTradingPlatformError):
    """Error related to order generation."""
    
    def __init__(
        self,
        message: str,
        strategy_name: Optional[str] = None,
        signal_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if strategy_name:
            details["strategy_name"] = strategy_name
        if signal_id:
            details["signal_id"] = signal_id
        super().__init__(
            message=message,
            error_code="ORDER_GENERATION_ERROR",
            details=details
        )
        self.strategy_name = strategy_name
        self.signal_id = signal_id


class BacktestError(ForexTradingPlatformError):
    """Error related to backtesting."""
    
    def __init__(
        self,
        message: str,
        strategy_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if strategy_name:
            details["strategy_name"] = strategy_name
        super().__init__(
            message=message,
            error_code="BACKTEST_ERROR",
            details=details
        )
        self.strategy_name = strategy_name


class RiskManagementError(ForexTradingPlatformError):
    """Error related to risk management."""
    
    def __init__(
        self,
        message: str,
        strategy_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        details = details or {}
        if strategy_name:
            details["strategy_name"] = strategy_name
        super().__init__(
            message=message,
            error_code="RISK_MANAGEMENT_ERROR",
            details=details
        )
        self.strategy_name = strategy_name
