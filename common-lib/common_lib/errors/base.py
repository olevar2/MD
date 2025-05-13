"""
Base Error Module

This module defines the base error classes for the platform.
"""

from typing import Dict, Any, Optional, List
from enum import Enum


class ErrorCode(Enum):
    """
    Error codes for the platform.
    """
    
    # General errors (1000-1999)
    UNKNOWN_ERROR = 1000
    INVALID_INPUT = 1001
    RESOURCE_NOT_FOUND = 1002
    RESOURCE_ALREADY_EXISTS = 1003
    OPERATION_NOT_SUPPORTED = 1004
    PERMISSION_DENIED = 1005
    AUTHENTICATION_FAILED = 1006
    RATE_LIMIT_EXCEEDED = 1007
    TIMEOUT = 1008
    
    # Service errors (2000-2999)
    SERVICE_UNAVAILABLE = 2000
    SERVICE_TIMEOUT = 2001
    SERVICE_AUTHENTICATION_FAILED = 2002
    SERVICE_AUTHORIZATION_FAILED = 2003
    SERVICE_VALIDATION_FAILED = 2004
    SERVICE_RESOURCE_NOT_FOUND = 2005
    SERVICE_RESOURCE_CONFLICT = 2006
    SERVICE_INTERNAL_ERROR = 2007
    
    # Data errors (3000-3999)
    DATA_VALIDATION_FAILED = 3000
    DATA_NOT_FOUND = 3001
    DATA_DUPLICATE = 3002
    DATA_CORRUPTION = 3003
    DATA_PROCESSING_FAILED = 3004
    
    # Market data errors (4000-4999)
    MARKET_DATA_NOT_AVAILABLE = 4000
    MARKET_DATA_DELAYED = 4001
    MARKET_DATA_INCOMPLETE = 4002
    MARKET_DATA_INVALID = 4003
    
    # Trading errors (5000-5999)
    TRADING_ORDER_REJECTED = 5000
    TRADING_ORDER_NOT_FOUND = 5001
    TRADING_POSITION_NOT_FOUND = 5002
    TRADING_INSUFFICIENT_FUNDS = 5003
    TRADING_MARKET_CLOSED = 5004
    TRADING_INVALID_PRICE = 5005
    TRADING_INVALID_QUANTITY = 5006
    
    # Analysis errors (6000-6999)
    ANALYSIS_FAILED = 6000
    ANALYSIS_INVALID_PARAMETERS = 6001
    ANALYSIS_INSUFFICIENT_DATA = 6002
    ANALYSIS_TIMEOUT = 6003
    
    # Feature store errors (7000-7999)
    FEATURE_NOT_FOUND = 7000
    FEATURE_CALCULATION_FAILED = 7001
    FEATURE_INVALID_PARAMETERS = 7002
    FEATURE_SET_NOT_FOUND = 7003
    FEATURE_SET_ALREADY_EXISTS = 7004


class BaseError(Exception):
    """
    Base error class for all errors in the platform.
    """
    
    def __init__(
        self,
        code: int,
        message: str,
        details: Optional[str] = None,
        correlation_id: Optional[str] = None,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the error.
        
        Args:
            code: Error code
            message: Error message
            details: Error details
            correlation_id: Correlation ID
            service: Service that raised the error
            operation: Operation that failed
            data: Additional data
        """
        self.code = code
        self.message = message
        self.details = details
        self.correlation_id = correlation_id
        self.service = service
        self.operation = operation
        self.data = data or {}
        
        super().__init__(message)
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the error to a dictionary.
        
        Returns:
            Dictionary representation of the error
        """
        result = {
            "code": self.code,
            "message": self.message
        }
        
        if self.details:
            result["details"] = self.details
        
        if self.correlation_id:
            result["correlation_id"] = self.correlation_id
        
        if self.service:
            result["service"] = self.service
        
        if self.operation:
            result["operation"] = self.operation
        
        if self.data:
            result["data"] = self.data
        
        return result
    
    def __str__(self) -> str:
        """
        Get a string representation of the error.
        
        Returns:
            String representation of the error
        """
        parts = [f"[{self.code}] {self.message}"]
        
        if self.details:
            parts.append(f"Details: {self.details}")
        
        if self.correlation_id:
            parts.append(f"Correlation ID: {self.correlation_id}")
        
        if self.service:
            parts.append(f"Service: {self.service}")
        
        if self.operation:
            parts.append(f"Operation: {self.operation}")
        
        return " | ".join(parts)


class ValidationError(BaseError):
    """
    Validation error.
    """
    
    def __init__(
        self,
        message: str,
        details: Optional[str] = None,
        correlation_id: Optional[str] = None,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
        field_errors: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize the validation error.
        
        Args:
            message: Error message
            details: Error details
            correlation_id: Correlation ID
            service: Service that raised the error
            operation: Operation that failed
            data: Additional data
            field_errors: Field-specific errors
        """
        super().__init__(
            code=ErrorCode.INVALID_INPUT.value,
            message=message,
            details=details,
            correlation_id=correlation_id,
            service=service,
            operation=operation,
            data=data or {}
        )
        
        self.field_errors = field_errors or {}
        
        if self.field_errors:
            self.data["field_errors"] = self.field_errors


class ResourceNotFoundError(BaseError):
    """
    Resource not found error.
    """
    
    def __init__(
        self,
        resource_type: str,
        resource_id: str,
        details: Optional[str] = None,
        correlation_id: Optional[str] = None,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the resource not found error.
        
        Args:
            resource_type: Type of the resource
            resource_id: ID of the resource
            details: Error details
            correlation_id: Correlation ID
            service: Service that raised the error
            operation: Operation that failed
            data: Additional data
        """
        super().__init__(
            code=ErrorCode.RESOURCE_NOT_FOUND.value,
            message=f"{resource_type} not found: {resource_id}",
            details=details,
            correlation_id=correlation_id,
            service=service,
            operation=operation,
            data=data or {}
        )
        
        self.resource_type = resource_type
        self.resource_id = resource_id
        
        self.data["resource_type"] = resource_type
        self.data["resource_id"] = resource_id


class ServiceError(BaseError):
    """
    Service error.
    """
    
    def __init__(
        self,
        code: int,
        message: str,
        details: Optional[str] = None,
        correlation_id: Optional[str] = None,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the service error.
        
        Args:
            code: Error code
            message: Error message
            details: Error details
            correlation_id: Correlation ID
            service: Service that raised the error
            operation: Operation that failed
            data: Additional data
        """
        super().__init__(
            code=code,
            message=message,
            details=details,
            correlation_id=correlation_id,
            service=service,
            operation=operation,
            data=data
        )


class DataError(BaseError):
    """
    Data error.
    """
    
    def __init__(
        self,
        code: int,
        message: str,
        details: Optional[str] = None,
        correlation_id: Optional[str] = None,
        service: Optional[str] = None,
        operation: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize the data error.
        
        Args:
            code: Error code
            message: Error message
            details: Error details
            correlation_id: Correlation ID
            service: Service that raised the error
            operation: Operation that failed
            data: Additional data
        """
        super().__init__(
            code=code,
            message=message,
            details=details,
            correlation_id=correlation_id,
            service=service,
            operation=operation,
            data=data
        )