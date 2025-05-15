"""
Pydantic models for request/response validation
"""
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, Any, List
from datetime import datetime

class MessageBase(BaseModel):
    """Base model for chat messages."""
    message: str = Field(..., min_length=1, max_length=1000)
    context: Optional[Dict[str, Any]] = Field(default=None)

class MessageCreate(MessageBase):
    """Model for creating chat messages."""
    pass

class MessageResponse(MessageBase):
    """Model for chat message responses."""
    id: str
    user_id: str
    response: Optional[Dict[str, Any]] = None
    correlation_id: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class MessageList(BaseModel):
    """Model for list of chat messages."""
    messages: List[MessageResponse]
    total: int
    has_more: bool

class SessionBase(BaseModel):
    """Base model for chat sessions."""
    session_data: Optional[Dict[str, Any]] = None

class SessionCreate(SessionBase):
    """Model for creating chat sessions."""
    pass

class SessionResponse(SessionBase):
    """Model for chat session responses."""
    id: str
    user_id: str
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    
    class Config:
        orm_mode = True

class HistoryParams(BaseModel):
    """Model for chat history query parameters."""
    limit: Optional[int] = Field(default=50, ge=1, le=100)
    before: Optional[datetime] = None
    
    @validator('before')
    def validate_before(cls, v: Optional[datetime]) -> Optional[datetime]:
        """Validate before timestamp.
        
        Args:
            v: Timestamp value
            
        Returns:
            Validated timestamp
        """
        if v and v > datetime.utcnow():
            raise ValueError("'before' timestamp cannot be in the future")
        return v

class ErrorResponse(BaseModel):
    """Model for error responses."""
    error: str
    detail: Optional[str] = None
    correlation_id: Optional[str] = None