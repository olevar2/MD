"""
Standardized API Response Format

This module provides a standardized format for all API responses.
"""

from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, Field


class MetaData(BaseModel):
    """Metadata for API responses."""
    
    # Request correlation ID for tracking
    correlation_id: str
    
    # Request ID for tracking
    request_id: str
    
    # Timestamp of the response
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # API version
    version: str = "1.0"
    
    # Service that generated the response
    service: str = "api-gateway"


class Pagination(BaseModel):
    """Pagination information for list responses."""
    
    # Total number of items
    total: int
    
    # Number of items per page
    per_page: int
    
    # Current page number
    page: int
    
    # Total number of pages
    pages: int
    
    # URL for the next page (if available)
    next_page: Optional[str] = None
    
    # URL for the previous page (if available)
    prev_page: Optional[str] = None


class ErrorDetails(BaseModel):
    """Detailed error information."""
    
    # Error code
    code: str
    
    # Human-readable error message
    message: str
    
    # Detailed error information
    details: Optional[Dict[str, Any]] = None
    
    # Source of the error (service name)
    source: Optional[str] = None
    
    # Field that caused the error (for validation errors)
    field: Optional[str] = None


class StandardResponse(BaseModel):
    """Standardized API response format."""
    
    # Status of the response (success, error, warning)
    status: str
    
    # Response data
    data: Optional[Union[Dict[str, Any], List[Any]]] = None
    
    # Error information (if status is error)
    error: Optional[ErrorDetails] = None
    
    # Metadata for the response
    meta: MetaData
    
    # Pagination information (for list responses)
    pagination: Optional[Pagination] = None


def create_success_response(
    data: Union[Dict[str, Any], List[Any]],
    correlation_id: str,
    request_id: str,
    pagination: Optional[Pagination] = None,
    service: str = "api-gateway",
    version: str = "1.0"
) -> StandardResponse:
    """
    Create a success response.
    
    Args:
        data: Response data
        correlation_id: Correlation ID for tracking the request
        request_id: Request ID for tracking the request
        pagination: Pagination information (for list responses)
        service: Service that generated the response
        version: API version
        
    Returns:
        A StandardResponse object
    """
    return StandardResponse(
        status="success",
        data=data,
        meta=MetaData(
            correlation_id=correlation_id,
            request_id=request_id,
            service=service,
            version=version
        ),
        pagination=pagination
    )


def create_error_response(
    code: str,
    message: str,
    correlation_id: str,
    request_id: str,
    details: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    field: Optional[str] = None,
    service: str = "api-gateway",
    version: str = "1.0"
) -> StandardResponse:
    """
    Create an error response.
    
    Args:
        code: Error code
        message: Human-readable error message
        correlation_id: Correlation ID for tracking the request
        request_id: Request ID for tracking the request
        details: Detailed error information
        source: Source of the error (service name)
        field: Field that caused the error (for validation errors)
        service: Service that generated the response
        version: API version
        
    Returns:
        A StandardResponse object
    """
    return StandardResponse(
        status="error",
        error=ErrorDetails(
            code=code,
            message=message,
            details=details,
            source=source,
            field=field
        ),
        meta=MetaData(
            correlation_id=correlation_id,
            request_id=request_id,
            service=service,
            version=version
        )
    )


def create_warning_response(
    data: Union[Dict[str, Any], List[Any]],
    code: str,
    message: str,
    correlation_id: str,
    request_id: str,
    details: Optional[Dict[str, Any]] = None,
    source: Optional[str] = None,
    field: Optional[str] = None,
    pagination: Optional[Pagination] = None,
    service: str = "api-gateway",
    version: str = "1.0"
) -> StandardResponse:
    """
    Create a warning response.
    
    Args:
        data: Response data
        code: Warning code
        message: Human-readable warning message
        correlation_id: Correlation ID for tracking the request
        request_id: Request ID for tracking the request
        details: Detailed warning information
        source: Source of the warning (service name)
        field: Field that caused the warning (for validation warnings)
        pagination: Pagination information (for list responses)
        service: Service that generated the response
        version: API version
        
    Returns:
        A StandardResponse object
    """
    return StandardResponse(
        status="warning",
        data=data,
        error=ErrorDetails(
            code=code,
            message=message,
            details=details,
            source=source,
            field=field
        ),
        meta=MetaData(
            correlation_id=correlation_id,
            request_id=request_id,
            service=service,
            version=version
        ),
        pagination=pagination
    )