
"""
Base exceptions for the forex trading platform.

This module defines the base exception hierarchy for the platform,
ensuring consistent error handling across services.
"""

from enum import Enum
from typing import Dict, Any, Optional, List, Union
import uuid
import traceback


class ErrorCode(Enum):
    """Error codes for the platform."""
    # General errors (1-99)
    UNKNOWN_ERROR = 1
    VALIDATION_ERROR = 2
    CONFIGURATION_ERROR = 3
    DEPENDENCY_ERROR = 4
    TIMEOUT_ERROR = 5
    RESOURCE_NOT_FOUND = 6
    PERMISSION_DENIED = 7
    RATE_LIMIT_EXCEEDED = 8

    # Database errors (100-199)
    DATABASE_CONNECTION_ERROR = 100
    DATABASE_QUERY_ERROR = 101
    DATABASE_TRANSACTION_ERROR = 102
    DATABASE_CONSTRAINT_ERROR = 103
    DATABASE_TIMEOUT_ERROR = 104

    # API errors (200-299)
    API_REQUEST_ERROR = 200
    API_RESPONSE_ERROR = 201
    API_AUTHENTICATION_ERROR = 202
    API_AUTHORIZATION_ERROR = 203
    API_RATE_LIMIT_ERROR = 204
    API_TIMEOUT_ERROR = 205

    # Service errors (300-399)
    SERVICE_UNAVAILABLE = 300
    SERVICE_TIMEOUT = 301
    SERVICE_DEPENDENCY_ERROR = 302
    SERVICE_CIRCUIT_OPEN = 303

    # Data errors (400-499)
    DATA_VALIDATION_ERROR = 400
    DATA_INTEGRITY_ERROR = 401
    DATA_FORMAT_ERROR = 402
    DATA_MISSING_ERROR = 403
    DATA_INCONSISTENCY_ERROR = 404

    # Business logic errors (500-599)
    BUSINESS_RULE_VIOLATION = 500
    INSUFFICIENT_FUNDS = 501
    POSITION_LIMIT_EXCEEDED = 502
    INVALID_OPERATION = 503
    OPERATION_NOT_ALLOWED = 504

    # Security errors (600-699)
    SECURITY_VIOLATION = 600
    AUTHENTICATION_FAILED = 601
    AUTHORIZATION_FAILED = 602
    TOKEN_EXPIRED = 603
    INVALID_CREDENTIALS = 604


class BaseError(Exception):
    """Base exception for all platform errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.UNKNOWN_ERROR,
        details: Optional[Dict[str, Any]] = None,
        correlation_id: Optional[str] = None,
        cause: Optional[Exception] = None
    ):
        """Initialize the base error.

        Args:
            message: Human-readable error message
            error_code: Error code from the ErrorCode enum
            details: Additional error details
            correlation_id: Correlation ID for tracking the error across services
            cause: Original exception that caused this error
        """
        self.message = message
        self.error_code = error_code
        self.details = details or {}
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self.cause = cause
        self.timestamp = None  # Will be set when the error is logged

        # Add stack trace to details if a cause is provided
        if cause:
            self.details["cause"] = str(cause)
            self.details["traceback"] = "".join(traceback.format_exception(
                type(cause), cause, cause.__traceback__
            ))

        super().__init__(message)

    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary representation."""
        return {
            "message": self.message,
            "error_code": self.error_code.name,
            "error_code_value": self.error_code.value,
            "details": self.details,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp
        }


class ValidationError(BaseError):
    """Error raised when input validation fails."""

    def __init__(
        self,
        message: str,
        field: Optional[str] = None,
        value: Optional[Any] = None,
        constraint: Optional[str] = None,
        **kwargs
    ):
        """Initialize a validation error.

        Args:
            message: Human-readable error message
            field: Name of the field that failed validation
            value: Value that failed validation
            constraint: Constraint that was violated
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if field:
            details["field"] = field
        if value is not None:
            details["value"] = str(value)
        if constraint:
            details["constraint"] = constraint

        super().__init__(
            message=message,
            error_code=ErrorCode.VALIDATION_ERROR,
            details=details,
            **kwargs
        )


class DatabaseError(BaseError):
    """Base class for database-related errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.DATABASE_CONNECTION_ERROR,
        query: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize a database error.

        Args:
            message: Human-readable error message
            error_code: Specific database error code
            query: SQL query that caused the error
            parameters: Parameters for the query
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if query:
            # Sanitize the query to avoid logging sensitive information
            details["query"] = self._sanitize_query(query)
        if parameters:
            # Sanitize parameters to avoid logging sensitive information
            details["parameters"] = self._sanitize_parameters(parameters)

        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            **kwargs
        )

    @staticmethod
    def _sanitize_query(query: str) -> str:
        """Sanitize a SQL query to remove sensitive information."""
        # This is a simple implementation that could be enhanced
        return query

    @staticmethod
    def _sanitize_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize query parameters to remove sensitive information."""
        # This is a simple implementation that could be enhanced
        sanitized = {}
        sensitive_keys = ["password", "token", "secret", "key", "auth"]

        for key, value in parameters.items():
            if any(sensitive in key.lower() for sensitive in sensitive_keys):
                sanitized[key] = "********"
            else:
                sanitized[key] = value

        return sanitized


class APIError(BaseError):
    """Base class for API-related errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.API_REQUEST_ERROR,
        status_code: Optional[int] = None,
        endpoint: Optional[str] = None,
        method: Optional[str] = None,
        response: Optional[Dict[str, Any]] = None,
        **kwargs
    ):
        """Initialize an API error.

        Args:
            message: Human-readable error message
            error_code: Specific API error code
            status_code: HTTP status code
            endpoint: API endpoint that was called
            method: HTTP method used
            response: Response from the API
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if status_code:
            details["status_code"] = status_code
        if endpoint:
            details["endpoint"] = endpoint
        if method:
            details["method"] = method
        if response:
            details["response"] = response

        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            **kwargs
        )


class ServiceError(BaseError):
    """Base class for service-related errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.SERVICE_UNAVAILABLE,
        service_name: Optional[str] = None,
        operation: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        """Initialize a service error.

        Args:
            message: Human-readable error message
            error_code: Specific service error code
            service_name: Name of the service that failed
            operation: Operation that was attempted
            retry_after: Seconds to wait before retrying
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if service_name:
            details["service_name"] = service_name
        if operation:
            details["operation"] = operation
        if retry_after:
            details["retry_after"] = retry_after

        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            **kwargs
        )


class DataError(BaseError):
    """Base class for data-related errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.DATA_VALIDATION_ERROR,
        data_source: Optional[str] = None,
        data_type: Optional[str] = None,
        **kwargs
    ):
        """Initialize a data error.

        Args:
            message: Human-readable error message
            error_code: Specific data error code
            data_source: Source of the data
            data_type: Type of data
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if data_source:
            details["data_source"] = data_source
        if data_type:
            details["data_type"] = data_type

        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            **kwargs
        )


class BusinessError(BaseError):
    """Base class for business logic errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.BUSINESS_RULE_VIOLATION,
        rule: Optional[str] = None,
        entity: Optional[str] = None,
        entity_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize a business error.

        Args:
            message: Human-readable error message
            error_code: Specific business error code
            rule: Business rule that was violated
            entity: Entity involved in the error
            entity_id: ID of the entity
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if rule:
            details["rule"] = rule
        if entity:
            details["entity"] = entity
        if entity_id:
            details["entity_id"] = entity_id

        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            **kwargs
        )


class SecurityError(BaseError):
    """Base class for security-related errors."""

    def __init__(
        self,
        message: str,
        error_code: ErrorCode = ErrorCode.SECURITY_VIOLATION,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        **kwargs
    ):
        """Initialize a security error.

        Args:
            message: Human-readable error message
            error_code: Specific security error code
            user_id: ID of the user
            resource: Resource that was accessed
            action: Action that was attempted
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if user_id:
            details["user_id"] = user_id
        if resource:
            details["resource"] = resource
        if action:
            details["action"] = action

        super().__init__(
            message=message,
            error_code=error_code,
            details=details,
            **kwargs
        )


# Additional error classes needed for service client
class ForexTradingError(BaseError):
    """Base class for all forex trading platform errors."""
    pass


class ServiceUnavailableError(ServiceError):
    """Error raised when a service is unavailable."""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        retry_after: Optional[int] = None,
        **kwargs
    ):
        """Initialize a service unavailable error.

        Args:
            message: Human-readable error message
            service_name: Name of the service that is unavailable
            retry_after: Seconds to wait before retrying
            **kwargs: Additional arguments to pass to ServiceError
        """
        super().__init__(
            message=message,
            error_code=ErrorCode.SERVICE_UNAVAILABLE,
            service_name=service_name,
            retry_after=retry_after,
            **kwargs
        )


class ThirdPartyServiceError(ServiceError):
    """Error raised when a third-party service fails."""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        **kwargs
    ):
        """Initialize a third-party service error.

        Args:
            message: Human-readable error message
            service_name: Name of the third-party service
            **kwargs: Additional arguments to pass to ServiceError
        """
        super().__init__(
            message=message,
            error_code=ErrorCode.SERVICE_DEPENDENCY_ERROR,
            service_name=service_name,
            **kwargs
        )


class TimeoutError(ServiceError):
    """Error raised when a service request times out."""

    def __init__(
        self,
        message: str,
        service_name: Optional[str] = None,
        timeout: Optional[float] = None,
        **kwargs
    ):
        """Initialize a timeout error.

        Args:
            message: Human-readable error message
            service_name: Name of the service that timed out
            timeout: Timeout value in seconds
            **kwargs: Additional arguments to pass to ServiceError
        """
        details = kwargs.pop("details", {})
        if timeout:
            details["timeout"] = timeout

        super().__init__(
            message=message,
            error_code=ErrorCode.SERVICE_TIMEOUT,
            service_name=service_name,
            details=details,
            **kwargs
        )


class AuthenticationError(SecurityError):
    """Error raised when authentication fails."""

    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize an authentication error.

        Args:
            message: Human-readable error message
            user_id: ID of the user
            **kwargs: Additional arguments to pass to SecurityError
        """
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHENTICATION_FAILED,
            user_id=user_id,
            **kwargs
        )


class AuthorizationError(SecurityError):
    """Error raised when authorization fails."""

    def __init__(
        self,
        message: str,
        user_id: Optional[str] = None,
        resource: Optional[str] = None,
        action: Optional[str] = None,
        **kwargs
    ):
        """Initialize an authorization error.

        Args:
            message: Human-readable error message
            user_id: ID of the user
            resource: Resource that was accessed
            action: Action that was attempted
            **kwargs: Additional arguments to pass to SecurityError
        """
        super().__init__(
            message=message,
            error_code=ErrorCode.AUTHORIZATION_FAILED,
            user_id=user_id,
            resource=resource,
            action=action,
            **kwargs
        )


class NotFoundError(BaseError):
    """Error raised when a resource is not found."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        **kwargs
    ):
        """Initialize a not found error.

        Args:
            message: Human-readable error message
            resource_type: Type of resource that was not found
            resource_id: ID of the resource that was not found
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id

        super().__init__(
            message=message,
            error_code=ErrorCode.RESOURCE_NOT_FOUND,
            details=details,
            **kwargs
        )


class ConflictError(BaseError):
    """Error raised when there is a conflict with the current state of the resource."""

    def __init__(
        self,
        message: str,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        conflict_reason: Optional[str] = None,
        **kwargs
    ):
        """Initialize a conflict error.

        Args:
            message: Human-readable error message
            resource_type: Type of resource with the conflict
            resource_id: ID of the resource with the conflict
            conflict_reason: Reason for the conflict
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if resource_type:
            details["resource_type"] = resource_type
        if resource_id:
            details["resource_id"] = resource_id
        if conflict_reason:
            details["conflict_reason"] = conflict_reason

        super().__init__(
            message=message,
            error_code=ErrorCode.INVALID_OPERATION,
            details=details,
            **kwargs
        )


class RateLimitError(BaseError):
    """Error raised when a rate limit is exceeded."""

    def __init__(
        self,
        message: str,
        limit: Optional[int] = None,
        reset_after: Optional[int] = None,
        resource: Optional[str] = None,
        **kwargs
    ):
        """Initialize a rate limit error.

        Args:
            message: Human-readable error message
            limit: Rate limit that was exceeded
            reset_after: Seconds until the rate limit resets
            resource: Resource that was rate limited
            **kwargs: Additional arguments to pass to BaseError
        """
        details = kwargs.pop("details", {})
        if limit:
            details["limit"] = limit
        if reset_after:
            details["reset_after"] = reset_after
        if resource:
            details["resource"] = resource

        super().__init__(
            message=message,
            error_code=ErrorCode.RATE_LIMIT_EXCEEDED,
            details=details,
            **kwargs
        )
