"""
Dependency injection module for the chat service
"""
from fastapi import Request, Depends, HTTPException, status
from typing import Optional, AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession
from .database import get_db
from .repositories.chat_repository import ChatRepository
from .events.event_bus import EventBus, get_event_bus
from .config.settings import get_settings

settings = get_settings()

def get_correlation_id(request: Request) -> str:
    """Get correlation ID from request state.
    
    Args:
        request: FastAPI request
        
    Returns:
        Correlation ID string
    """
    return getattr(request.state, 'correlation_id', None)

def get_user_id(request: Request) -> str:
    """Get user ID from request headers.
    
    Args:
        request: FastAPI request
        
    Returns:
        User ID string
        
    Raises:
        HTTPException: If user ID header is missing
    """
    user_id = request.headers.get('X-User-ID')
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User ID header is missing"
        )
    return user_id

async def get_chat_repository(db: AsyncSession = Depends(get_db)) -> AsyncGenerator[ChatRepository, None]:
    """Get chat repository instance.
    
    Args:
        db: Database session
        
    Yields:
        ChatRepository instance
    """
    try:
        repository = ChatRepository(db)
        yield repository
    finally:
        await db.close()

async def get_event_bus_instance() -> AsyncGenerator[EventBus, None]:
    """Get event bus instance.
    
    Yields:
        EventBus instance
    """
    event_bus = get_event_bus()
    try:
        await event_bus.start()
        yield event_bus
    finally:
        await event_bus.stop()

def validate_api_key(request: Request) -> None:
    """Validate API key from request headers.
    
    Args:
        request: FastAPI request
        
    Raises:
        HTTPException: If API key is invalid or missing
    """
    api_key = request.headers.get(settings.API_KEY_NAME)
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key is missing",
            headers={"WWW-Authenticate": settings.API_KEY_NAME}
        )
    
    if api_key != settings.SECRET_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": settings.API_KEY_NAME}
        )

def get_settings_instance():
    """Get settings instance.
    
    Returns:
        Settings instance
    """
    return get_settings()