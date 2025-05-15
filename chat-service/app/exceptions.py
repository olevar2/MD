"""
Exception handling module for the chat service
"""
from fastapi import Request, status
from fastapi.responses import JSONResponse
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class ChatServiceError(Exception):
    """Base exception for chat service."""
    def __init__(self, message: str, status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
                 detail: Optional[str] = None):
        """Initialize chat service error.
        
        Args:
            message: Error message
            status_code: HTTP status code
            detail: Detailed error information
        """
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(self.message)

class ValidationError(ChatServiceError):
    """Exception for validation errors."""
    def __init__(self, message: str, detail: Optional[str] = None):
        """Initialize validation error.
        
        Args:
            message: Error message
            detail: Detailed error information
        """
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=detail
        )

class AuthenticationError(ChatServiceError):
    """Exception for authentication errors."""
    def __init__(self, message: str, detail: Optional[str] = None):
        """Initialize authentication error.
        
        Args:
            message: Error message
            detail: Detailed error information
        """
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=detail
        )

class NotFoundError(ChatServiceError):
    """Exception for resource not found errors."""
    def __init__(self, message: str, detail: Optional[str] = None):
        """Initialize not found error.
        
        Args:
            message: Error message
            detail: Detailed error information
        """
        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            detail=detail
        )

class DatabaseError(ChatServiceError):
    """Exception for database errors."""
    def __init__(self, message: str, detail: Optional[str] = None):
        """Initialize database error.
        
        Args:
            message: Error message
            detail: Detailed error information
        """
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )

class EventBusError(ChatServiceError):
    """Exception for event bus errors."""
    def __init__(self, message: str, detail: Optional[str] = None):
        """Initialize event bus error.
        
        Args:
            message: Error message
            detail: Detailed error information
        """
        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=detail
        )

async def chat_service_exception_handler(request: Request, exc: ChatServiceError) -> JSONResponse:
    """Handle chat service exceptions.
    
    Args:
        request: FastAPI request
        exc: ChatServiceError instance
        
    Returns:
        JSON response with error details
    """
    error_response: Dict[str, Any] = {
        "error": exc.message,
        "correlation_id": getattr(request.state, "correlation_id", None)
    }
    
    if exc.detail:
        error_response["detail"] = exc.detail
    
    # Log the error
    logger.error(
        f"Error handling request: {exc.message}",
        extra={
            "correlation_id": error_response["correlation_id"],
            "status_code": exc.status_code,
            "error_detail": exc.detail
        },
        exc_info=True
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response
    )

def setup_exception_handlers(app) -> None:
    """Setup exception handlers for the application.
    
    Args:
        app: FastAPI application instance
    """
    app.add_exception_handler(ChatServiceError, chat_service_exception_handler)