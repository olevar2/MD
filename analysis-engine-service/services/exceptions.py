"""
Common exceptions for the forex trading platform.

This module provides a set of custom exceptions used across multiple services.
"""

class BaseError(Exception):
    """Base class for all custom exceptions."""
    
    def __init__(self, message=None, code=None, details=None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            code: Error code
            details: Additional error details
        """
        self.message = message or "An error occurred"
        self.code = code or "INTERNAL_ERROR"
        self.details = details or {}
        
        super().__init__(self.message)


class ValidationError(BaseError):
    """Exception raised for validation errors."""
    
    def __init__(self, message=None, details=None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message or "Validation error",
            code="VALIDATION_ERROR",
            details=details
        )


class NotFoundError(BaseError):
    """Exception raised when a resource is not found."""
    
    def __init__(self, message=None, details=None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message or "Resource not found",
            code="NOT_FOUND",
            details=details
        )


class ServiceError(BaseError):
    """Exception raised for service-level errors."""
    
    def __init__(self, message=None, code=None, details=None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            code: Error code
            details: Additional error details
        """
        super().__init__(
            message=message or "Service error",
            code=code or "SERVICE_ERROR",
            details=details
        )


class DataError(BaseError):
    """Exception raised for data-related errors."""
    
    def __init__(self, message=None, details=None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message or "Data error",
            code="DATA_ERROR",
            details=details
        )


class ConfigurationError(BaseError):
    """Exception raised for configuration errors."""
    
    def __init__(self, message=None, details=None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message or "Configuration error",
            code="CONFIGURATION_ERROR",
            details=details
        )


class AuthenticationError(BaseError):
    """Exception raised for authentication errors."""
    
    def __init__(self, message=None, details=None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message or "Authentication error",
            code="AUTHENTICATION_ERROR",
            details=details
        )


class AuthorizationError(BaseError):
    """Exception raised for authorization errors."""
    
    def __init__(self, message=None, details=None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message or "Authorization error",
            code="AUTHORIZATION_ERROR",
            details=details
        )


class RateLimitError(BaseError):
    """Exception raised for rate limit errors."""
    
    def __init__(self, message=None, details=None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message or "Rate limit exceeded",
            code="RATE_LIMIT_ERROR",
            details=details
        )


class TimeoutError(BaseError):
    """Exception raised for timeout errors."""
    
    def __init__(self, message=None, details=None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            details: Additional error details
        """
        super().__init__(
            message=message or "Operation timed out",
            code="TIMEOUT_ERROR",
            details=details
        )


class ExternalServiceError(BaseError):
    """Exception raised for errors in external services."""
    
    def __init__(self, message=None, service=None, details=None):
        """
        Initialize the exception.
        
        Args:
            message: Error message
            service: Name of the external service
            details: Additional error details
        """
        details = details or {}
        if service:
            details["service"] = service
            
        super().__init__(
            message=message or f"Error in external service: {service}",
            code="EXTERNAL_SERVICE_ERROR",
            details=details
        )
