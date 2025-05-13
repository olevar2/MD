"""
Service Error Types.

This module defines error types related to service interactions in the forex trading platform.
"""

from typing import Any, Dict, Optional

from common_lib.errors.base import BaseError, ErrorCode, ErrorSeverity


class ServiceError(BaseError):
    """
    Base class for all service-related errors.
    
    This class is used for errors that occur during service interactions.
    """
    
    def __init__(
        self,
        message: str,
        service_name: str,
        code: ErrorCode = ErrorCode.SERVICE_INTERNAL_ERROR,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize a service error.
        
        Args:
            message: Error message
            service_name: Name of the service where the error occurred
            code: Error code
            severity: Error severity level
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        details = details or {}
        details["service_name"] = service_name
        
        super().__init__(
            message=message,
            code=code,
            severity=severity,
            details=details,
            correlation_id=correlation_id,
            cause=cause
        )
        
        self.service_name = service_name


class ServiceUnavailableError(ServiceError):
    """
    Error raised when a service is unavailable.
    
    This error is raised when a service cannot be reached or is not responding.
    """
    
    def __init__(
        self,
        service_name: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize a service unavailable error.
        
        Args:
            service_name: Name of the unavailable service
            message: Error message (if None, a default message is used)
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        message = message or f"Service '{service_name}' is unavailable"
        
        super().__init__(
            message=message,
            service_name=service_name,
            code=ErrorCode.SERVICE_UNAVAILABLE,
            severity=ErrorSeverity.ERROR,
            details=details,
            correlation_id=correlation_id,
            cause=cause
        )


class ServiceTimeoutError(ServiceError):
    """
    Error raised when a service request times out.
    
    This error is raised when a service request takes too long to complete.
    """
    
    def __init__(
        self,
        service_name: str,
        timeout: float,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize a service timeout error.
        
        Args:
            service_name: Name of the service where the timeout occurred
            timeout: Timeout value in seconds
            message: Error message (if None, a default message is used)
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        message = message or f"Request to service '{service_name}' timed out after {timeout} seconds"
        
        details = details or {}
        details["timeout"] = timeout
        
        super().__init__(
            message=message,
            service_name=service_name,
            code=ErrorCode.SERVICE_TIMEOUT,
            severity=ErrorSeverity.ERROR,
            details=details,
            correlation_id=correlation_id,
            cause=cause
        )


class ServiceAuthenticationError(ServiceError):
    """
    Error raised when authentication with a service fails.
    
    This error is raised when a service rejects the authentication credentials.
    """
    
    def __init__(
        self,
        service_name: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize a service authentication error.
        
        Args:
            service_name: Name of the service where authentication failed
            message: Error message (if None, a default message is used)
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        message = message or f"Authentication with service '{service_name}' failed"
        
        super().__init__(
            message=message,
            service_name=service_name,
            code=ErrorCode.SERVICE_AUTHENTICATION_FAILED,
            severity=ErrorSeverity.ERROR,
            details=details,
            correlation_id=correlation_id,
            cause=cause
        )


class ServiceAuthorizationError(ServiceError):
    """
    Error raised when authorization for a service operation fails.
    
    This error is raised when a service denies permission for an operation.
    """
    
    def __init__(
        self,
        service_name: str,
        operation: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize a service authorization error.
        
        Args:
            service_name: Name of the service where authorization failed
            operation: Name of the operation that was denied
            message: Error message (if None, a default message is used)
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        message = message or f"Authorization for operation '{operation}' on service '{service_name}' failed"
        
        details = details or {}
        details["operation"] = operation
        
        super().__init__(
            message=message,
            service_name=service_name,
            code=ErrorCode.SERVICE_AUTHORIZATION_FAILED,
            severity=ErrorSeverity.ERROR,
            details=details,
            correlation_id=correlation_id,
            cause=cause
        )


class ServiceValidationError(ServiceError):
    """
    Error raised when validation of service input fails.
    
    This error is raised when a service rejects input data as invalid.
    """
    
    def __init__(
        self,
        service_name: str,
        validation_errors: Dict[str, str],
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize a service validation error.
        
        Args:
            service_name: Name of the service where validation failed
            validation_errors: Dictionary mapping field names to error messages
            message: Error message (if None, a default message is used)
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        message = message or f"Validation failed for service '{service_name}'"
        
        details = details or {}
        details["validation_errors"] = validation_errors
        
        super().__init__(
            message=message,
            service_name=service_name,
            code=ErrorCode.SERVICE_VALIDATION_FAILED,
            severity=ErrorSeverity.ERROR,
            details=details,
            correlation_id=correlation_id,
            cause=cause
        )


class ServiceResourceNotFoundError(ServiceError):
    """
    Error raised when a service resource is not found.
    
    This error is raised when a service cannot find a requested resource.
    """
    
    def __init__(
        self,
        service_name: str,
        resource_type: str,
        resource_id: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize a service resource not found error.
        
        Args:
            service_name: Name of the service where the resource was not found
            resource_type: Type of the resource that was not found
            resource_id: ID of the resource that was not found
            message: Error message (if None, a default message is used)
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        message = message or f"Resource '{resource_type}' with ID '{resource_id}' not found in service '{service_name}'"
        
        details = details or {}
        details["resource_type"] = resource_type
        details["resource_id"] = resource_id
        
        super().__init__(
            message=message,
            service_name=service_name,
            code=ErrorCode.SERVICE_RESOURCE_NOT_FOUND,
            severity=ErrorSeverity.ERROR,
            details=details,
            correlation_id=correlation_id,
            cause=cause
        )


class ServiceResourceConflictError(ServiceError):
    """
    Error raised when a service resource conflict occurs.
    
    This error is raised when a service operation would result in a resource conflict.
    """
    
    def __init__(
        self,
        service_name: str,
        resource_type: str,
        resource_id: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize a service resource conflict error.
        
        Args:
            service_name: Name of the service where the resource conflict occurred
            resource_type: Type of the resource that caused the conflict
            resource_id: ID of the resource that caused the conflict
            message: Error message (if None, a default message is used)
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        message = message or f"Resource conflict for '{resource_type}' with ID '{resource_id}' in service '{service_name}'"
        
        details = details or {}
        details["resource_type"] = resource_type
        details["resource_id"] = resource_id
        
        super().__init__(
            message=message,
            service_name=service_name,
            code=ErrorCode.SERVICE_RESOURCE_CONFLICT,
            severity=ErrorSeverity.ERROR,
            details=details,
            correlation_id=correlation_id,
            cause=cause
        )


class ServiceInternalError(ServiceError):
    """
    Error raised when an internal service error occurs.
    
    This error is raised when a service encounters an unexpected internal error.
    """
    
    def __init__(
        self,
        service_name: str,
        message: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """
        Initialize a service internal error.
        
        Args:
            service_name: Name of the service where the internal error occurred
            message: Error message (if None, a default message is used)
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        message = message or f"Internal error in service '{service_name}'"
        
        super().__init__(
            message=message,
            service_name=service_name,
            code=ErrorCode.SERVICE_INTERNAL_ERROR,
            severity=ErrorSeverity.ERROR,
            details=details,
            correlation_id=correlation_id,
            cause=cause
        )