"""
Custom exceptions for the forex trading platform.
"""

from typing import Dict, Any, Optional


class ForexBaseException(Exception):
    """Base exception for all forex platform exceptions."""
    
    def __init__(self, message: str, correlation_id: Optional[str] = None, details: Optional[Dict[str, Any]] = None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            correlation_id: Correlation ID for tracking the error
            details: Additional error details
        """
        self.message = message
        self.correlation_id = correlation_id
        self.details = details or {}
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the exception to a dictionary.
        
        Returns:
            Dictionary representation of the exception
        """
        return {
            'error': self.__class__.__name__,
            'message': self.message,
            'correlation_id': self.correlation_id,
            'details': self.details
        }


# Service exceptions
class ServiceException(ForexBaseException):
    """Exception raised for service-related errors."""
    pass


class ServiceUnavailableException(ServiceException):
    """Exception raised when a service is unavailable."""
    pass


class ServiceTimeoutException(ServiceException):
    """Exception raised when a service request times out."""
    pass


class ServiceAuthenticationException(ServiceException):
    """Exception raised when service authentication fails."""
    pass


class ServiceAuthorizationException(ServiceException):
    """Exception raised when service authorization fails."""
    pass


# Data exceptions
class DataException(ForexBaseException):
    """Exception raised for data-related errors."""
    pass


class DataValidationException(DataException):
    """Exception raised when data validation fails."""
    pass


class DataNotFoundException(DataException):
    """Exception raised when data is not found."""
    pass


class DataDuplicateException(DataException):
    """Exception raised when duplicate data is detected."""
    pass


class DataCorruptionException(DataException):
    """Exception raised when data corruption is detected."""
    pass


# Trading exceptions
class TradingException(ForexBaseException):
    """Exception raised for trading-related errors."""
    pass


class OrderExecutionException(TradingException):
    """Exception raised when order execution fails."""
    pass


class OrderValidationException(TradingException):
    """Exception raised when order validation fails."""
    pass


class InsufficientFundsException(TradingException):
    """Exception raised when there are insufficient funds for a trade."""
    pass


class MarketClosedException(TradingException):
    """Exception raised when the market is closed."""
    pass


# Configuration exceptions
class ConfigurationException(ForexBaseException):
    """Exception raised for configuration-related errors."""
    pass


class ConfigurationNotFoundException(ConfigurationException):
    """Exception raised when a configuration is not found."""
    pass


class ConfigurationValidationException(ConfigurationException):
    """Exception raised when configuration validation fails."""
    pass


# Security exceptions
class SecurityException(ForexBaseException):
    """Exception raised for security-related errors."""
    pass


class AuthenticationException(SecurityException):
    """Exception raised when authentication fails."""
    pass


class AuthorizationException(SecurityException):
    """Exception raised when authorization fails."""
    pass


class RateLimitException(SecurityException):
    """Exception raised when rate limit is exceeded."""
    pass


# Infrastructure exceptions
class InfrastructureException(ForexBaseException):
    """Exception raised for infrastructure-related errors."""
    pass


class DatabaseException(InfrastructureException):
    """Exception raised for database-related errors."""
    pass


class NetworkException(InfrastructureException):
    """Exception raised for network-related errors."""
    pass


class ResourceExhaustionException(InfrastructureException):
    """Exception raised when a resource is exhausted."""
    pass
