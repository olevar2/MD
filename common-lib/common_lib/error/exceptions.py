"""
Exceptions Module

This module defines standardized exceptions for the Forex Trading Platform.
These exceptions are used across all services to ensure consistent error handling.

Key features:
1. Hierarchical exception structure
2. Standardized error codes
3. Detailed error information
4. Support for correlation IDs
"""

from typing import Dict, Any, Optional


class ForexTradingPlatformError(Exception):
    """
    Base exception for all Forex Trading Platform errors.
    
    All platform-specific exceptions should inherit from this class.
    """
    
    def __init__(
        self,
        message: str,
        error_code: str = "UNKNOWN_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the exception.
        
        Args:
            message: Human-readable error message
            error_code: Error code for programmatic handling
            details: Additional error details
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)
    
    def __str__(self) -> str:
        """
        Get string representation of the exception.
        
        Returns:
            String representation
        """
        return f"{self.__class__.__name__}[{self.error_code}]: {self.message}"


# Configuration Exceptions

class ConfigurationError(ForexTradingPlatformError):
    """Exception raised for configuration errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "CONFIGURATION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


# Data Exceptions

class DataError(ForexTradingPlatformError):
    """Base exception for data-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "DATA_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


class DataValidationError(DataError):
    """Exception raised for data validation errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "DATA_VALIDATION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


class DataFetchError(DataError):
    """Exception raised for errors when fetching data."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "DATA_FETCH_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


class DataStorageError(DataError):
    """Exception raised for errors when storing data."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "DATA_STORAGE_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


class DataTransformationError(DataError):
    """Exception raised for errors when transforming data."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "DATA_TRANSFORMATION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


# Service Exceptions

class ServiceError(ForexTradingPlatformError):
    """Base exception for service-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "SERVICE_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


class ServiceUnavailableError(ServiceError):
    """Exception raised when a service is unavailable."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "SERVICE_UNAVAILABLE",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


class ServiceTimeoutError(ServiceError):
    """Exception raised when a service request times out."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "SERVICE_TIMEOUT",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


# Authentication/Authorization Exceptions

class AuthenticationError(ForexTradingPlatformError):
    """Exception raised for authentication errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "AUTHENTICATION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


class AuthorizationError(ForexTradingPlatformError):
    """Exception raised for authorization errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "AUTHORIZATION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


# Network Exceptions

class NetworkError(ForexTradingPlatformError):
    """Exception raised for network-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "NETWORK_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


# Add more exception classes as needed for specific domains

# Trading Exceptions

class TradingError(ForexTradingPlatformError):
    """Base exception for trading-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "TRADING_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


class OrderExecutionError(TradingError):
    """Exception raised for order execution errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "ORDER_EXECUTION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


# Analysis Exceptions

class AnalysisError(ForexTradingPlatformError):
    """Base exception for analysis-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "ANALYSIS_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


class IndicatorCalculationError(AnalysisError):
    """Exception raised for indicator calculation errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "INDICATOR_CALCULATION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


# ML Exceptions

class MLError(ForexTradingPlatformError):
    """Base exception for ML-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "ML_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


class ModelTrainingError(MLError):
    """Exception raised for model training errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "MODEL_TRAINING_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)


class ModelPredictionError(MLError):
    """Exception raised for model prediction errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "MODEL_PREDICTION_ERROR",
        details: Optional[Dict[str, Any]] = None
    ):
        """Initialize the exception."""
        super().__init__(message, error_code, details)