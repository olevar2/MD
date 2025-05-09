"""
Chat API Endpoints

This module provides FastAPI endpoints for the chat interface.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.security import APIKeyHeader
from typing import Dict, List, Any, Optional
from datetime import datetime
import uuid

from .ChatBackendService import ChatBackendService

# Create router
router = APIRouter(prefix="/api/v1/chat", tags=["chat"])

# Initialize chat service
chat_service = ChatBackendService()

# API key security
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def get_api_key(api_key: str = Depends(api_key_header)):
    """Validate API key."""
    # In a real implementation, this would validate against a database or config
    if api_key is None:
        raise HTTPException(status_code=401, detail="API key is missing")
    # For demo purposes, accept any non-empty API key
    return api_key

async def get_user_id(api_key: str = Depends(get_api_key)):
    """Get user ID from API key."""
    # In a real implementation, this would look up the user ID from the API key
    # For demo purposes, use a fixed user ID
    return "demo_user"

@router.post("/message")
async def send_message(
    message_data: Dict[str, Any] = Body(...),
    user_id: str = Depends(get_user_id)
):
    """
    Send a message to the chat service.
    
    Args:
        message_data: Message data including text and context
        user_id: User ID
        
    Returns:
        Response from the chat service
    """
    if "message" not in message_data:
        raise HTTPException(status_code=400, detail="Message text is required")
    
    message = message_data["message"]
    context = message_data.get("context", {})
    
    try:
        response = await chat_service.process_message(user_id, message, context)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing message: {str(e)}")

@router.post("/execute-action")
async def execute_action(
    action_data: Dict[str, Any] = Body(...),
    user_id: str = Depends(get_user_id)
):
    """
    Execute a trading action.
    
    Args:
        action_data: Action data
        user_id: User ID
        
    Returns:
        Result of the action
    """
    if "action" not in action_data:
        raise HTTPException(status_code=400, detail="Action is required")
    
    action = action_data["action"]
    
    try:
        result = await chat_service.execute_trading_action(user_id, action)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing action: {str(e)}")

@router.get("/history")
async def get_history(
    limit: int = Query(50, ge=1, le=100),
    before: Optional[str] = Query(None),
    user_id: str = Depends(get_user_id)
):
    """
    Get chat history.
    
    Args:
        limit: Maximum number of messages to return
        before: Get messages before this timestamp
        user_id: User ID
        
    Returns:
        List of messages
    """
    before_dt = None
    if before:
        try:
            before_dt = datetime.fromisoformat(before)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format")
    
    try:
        messages = chat_service.get_chat_history(user_id, limit, before_dt)
        return {"messages": messages}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting chat history: {str(e)}")

@router.delete("/history")
async def clear_history(user_id: str = Depends(get_user_id)):
    """
    Clear chat history.
    
    Args:
        user_id: User ID
        
    Returns:
        Success status
    """
    try:
        success = chat_service.clear_chat_history(user_id)
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error clearing chat history: {str(e)}")

def setup_chat_routes(app):
    """
    Set up chat routes for the application.
    
    Args:
        app: FastAPI application
    """
    app.include_router(router)
