"""
Client Exceptions Module

This module defines standard exceptions for service clients.
These exceptions provide a consistent error handling approach across all services.
"""

from typing import Optional, Dict, Any

from common_lib.exceptions import (
    ServiceError,
    ServiceUnavailableError,
    ServiceTimeoutError,
    DataValidationError,
    AuthenticationError
)


class ClientError(ServiceError):
    """Base exception for client errors."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        status_code: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the client error.
        
        Args:
            message: Error message
            service_name: Name of the service that raised the error
            status_code: HTTP status code
            details: Additional error details
        """
        self.service_name = service_name
        self.status_code = status_code
        self.details = details or {}
        
        # Build a detailed message
        detailed_message = message
        if service_name:
            detailed_message = f"[{service_name}] {detailed_message}"
        if status_code:
            detailed_message = f"{detailed_message} (Status: {status_code})"
        
        super().__init__(detailed_message)


class ClientConnectionError(ServiceUnavailableError):
    """Exception raised when a client cannot connect to a service."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the client connection error.
        
        Args:
            message: Error message
            service_name: Name of the service that could not be connected to
            details: Additional error details
        """
        self.service_name = service_name
        self.details = details or {}
        
        # Build a detailed message
        detailed_message = message
        if service_name:
            detailed_message = f"[{service_name}] {detailed_message}"
        
        super().__init__(detailed_message)


class ClientTimeoutError(ServiceTimeoutError):
    """Exception raised when a client request times out."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        timeout_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the client timeout error.
        
        Args:
            message: Error message
            service_name: Name of the service that timed out
            timeout_seconds: Timeout value in seconds
            details: Additional error details
        """
        self.service_name = service_name
        self.timeout_seconds = timeout_seconds
        self.details = details or {}
        
        # Build a detailed message
        detailed_message = message
        if service_name:
            detailed_message = f"[{service_name}] {detailed_message}"
        if timeout_seconds:
            detailed_message = f"{detailed_message} (Timeout: {timeout_seconds}s)"
        
        super().__init__(detailed_message)


class ClientValidationError(DataValidationError):
    """Exception raised when client data validation fails."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        field_errors: Optional[Dict[str, str]] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the client validation error.
        
        Args:
            message: Error message
            service_name: Name of the service that raised the validation error
            field_errors: Dictionary of field-specific errors
            details: Additional error details
        """
        self.service_name = service_name
        self.field_errors = field_errors or {}
        self.details = details or {}
        
        # Build a detailed message
        detailed_message = message
        if service_name:
            detailed_message = f"[{service_name}] {detailed_message}"
        if field_errors:
            field_errors_str = ", ".join(f"{k}: {v}" for k, v in field_errors.items())
            detailed_message = f"{detailed_message} (Fields: {field_errors_str})"
        
        super().__init__(detailed_message)


class ClientAuthenticationError(AuthenticationError):
    """Exception raised when client authentication fails."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the client authentication error.
        
        Args:
            message: Error message
            service_name: Name of the service that raised the authentication error
            details: Additional error details
        """
        self.service_name = service_name
        self.details = details or {}
        
        # Build a detailed message
        detailed_message = message
        if service_name:
            detailed_message = f"[{service_name}] {detailed_message}"
        
        super().__init__(detailed_message)


class CircuitBreakerOpenError(ServiceUnavailableError):
    """Exception raised when a circuit breaker is open."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        reset_timeout_seconds: Optional[float] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the circuit breaker open error.
        
        Args:
            message: Error message
            service_name: Name of the service with the open circuit breaker
            reset_timeout_seconds: Time until the circuit breaker resets
            details: Additional error details
        """
        self.service_name = service_name
        self.reset_timeout_seconds = reset_timeout_seconds
        self.details = details or {}
        
        # Build a detailed message
        detailed_message = message
        if service_name:
            detailed_message = f"[{service_name}] {detailed_message}"
        if reset_timeout_seconds:
            detailed_message = f"{detailed_message} (Reset in: {reset_timeout_seconds}s)"
        
        super().__init__(detailed_message)


class BulkheadFullError(ServiceUnavailableError):
    """Exception raised when a bulkhead is full."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        max_concurrent: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the bulkhead full error.
        
        Args:
            message: Error message
            service_name: Name of the service with the full bulkhead
            max_concurrent: Maximum concurrent requests allowed
            details: Additional error details
        """
        self.service_name = service_name
        self.max_concurrent = max_concurrent
        self.details = details or {}
        
        # Build a detailed message
        detailed_message = message
        if service_name:
            detailed_message = f"[{service_name}] {detailed_message}"
        if max_concurrent:
            detailed_message = f"{detailed_message} (Max concurrent: {max_concurrent})"
        
        super().__init__(detailed_message)


class RetryExhaustedError(ServiceUnavailableError):
    """Exception raised when retries are exhausted."""
    
    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        max_attempts: Optional[int] = None,
        last_exception: Optional[Exception] = None,
        details: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the retry exhausted error.
        
        Args:
            message: Error message
            service_name: Name of the service that exhausted retries
            max_attempts: Maximum number of attempts
            last_exception: Last exception that caused the retry to fail
            details: Additional error details
        """
        self.service_name = service_name
        self.max_attempts = max_attempts
        self.last_exception = last_exception
        self.details = details or {}
        
        # Build a detailed message
        detailed_message = message
        if service_name:
            detailed_message = f"[{service_name}] {detailed_message}"
        if max_attempts:
            detailed_message = f"{detailed_message} (Max attempts: {max_attempts})"
        if last_exception:
            detailed_message = f"{detailed_message} (Last error: {str(last_exception)})"
        
        super().__init__(detailed_message)
