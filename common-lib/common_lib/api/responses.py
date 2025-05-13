"""
API Response Utilities

This module provides standardized response formats for the Forex Trading Platform.
"""

from datetime import datetime
from typing import Any, Dict, Generic, List, Optional, TypeVar, Union

from pydantic import BaseModel, Field

T = TypeVar('T')


class StandardResponse(BaseModel, Generic[T]):
    """Standard response format for API endpoints."""
    
    # Status of the response
    status: str = "success"
    
    # Response data
    data: T
    
    # Metadata about the response
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    # Timestamp of the response
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standard error response format for API endpoints."""
    
    # Status of the response
    status: str = "error"
    
    # Error code
    code: str
    
    # Human-readable error message
    message: str
    
    # Detailed error information
    details: Optional[Dict[str, Any]] = None
    
    # Correlation ID for tracking the request
    correlation_id: Optional[str] = None
    
    # Timestamp of the error
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Service that generated the error
    service: Optional[str] = None


def format_response(
    data: Any,
    metadata: Optional[Dict[str, Any]] = None
) -> StandardResponse:
    """
    Format a successful response.
    
    Args:
        data: The response data
        metadata: Optional metadata about the response
        
    Returns:
        A StandardResponse object
    """
    return StandardResponse(
        status="success",
        data=data,
        metadata=metadata or {}
    )


def format_error(
    code: str,
    message: str,
    details: Optional[Dict[str, Any]] = None,
    correlation_id: Optional[str] = None,
    service: Optional[str] = None
) -> ErrorResponse:
    """
    Format an error response.
    
    Args:
        code: Error code
        message: Human-readable error message
        details: Detailed error information
        correlation_id: Correlation ID for tracking the request
        service: Service that generated the error
        
    Returns:
        An ErrorResponse object
    """
    return ErrorResponse(
        status="error",
        code=code,
        message=message,
        details=details,
        correlation_id=correlation_id,
        service=service
    )
