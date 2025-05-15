"""
API Router for Chat Service
"""
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.security import APIKeyHeader
from typing import Dict, List, Any, Optional
from datetime import datetime
from common_lib.correlation import get_correlation_id
from ...services.chat_service import ChatService

# Create router
api_router = APIRouter()

# API key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    """Validate API key."""
    if api_key is None:
        raise HTTPException(status_code=401, detail="API key is missing")
    # TODO: Implement proper API key validation
    return api_key

async def get_user_id(api_key: str = Depends(get_api_key)):
    """Get user ID from API key."""
    # TODO: Implement proper user ID retrieval from API key
    return "demo_user"

@api_router.post("/chat/message")
async def send_message(
    message_data: Dict[str, Any] = Body(...),
    user_id: str = Depends(get_user_id),
    chat_service: ChatService = Depends()
):
    """Send a message to the chat service.
    
    Args:
        message_data: Message data including text and context
        user_id: User ID from API key
        chat_service: Chat service instance
    
    Returns:
        Response from the chat service
    """
    correlation_id = get_correlation_id()
    
    if "message" not in message_data:
        raise HTTPException(status_code=400, detail="Message text is required")
    
    message = message_data["message"]
    context = message_data.get("context", {})
    
    try:
        response = await chat_service.process_message(user_id, message, context)
        return response
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )

@api_router.get("/chat/history")
async def get_history(
    limit: int = Query(50, ge=1, le=100),
    before: Optional[str] = Query(None),
    user_id: str = Depends(get_user_id),
    chat_service: ChatService = Depends()
):
    """Get chat history.
    
    Args:
        limit: Maximum number of messages to return
        before: Get messages before this timestamp
        user_id: User ID from API key
        chat_service: Chat service instance
    
    Returns:
        List of chat messages
    """
    correlation_id = get_correlation_id()
    
    try:
        before_dt = datetime.fromisoformat(before) if before else None
        history = await chat_service.get_history(user_id, limit, before_dt)
        return history
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid timestamp format")
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching chat history: {str(e)}"
        )