"""
Repository layer for chat-related database operations
"""
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import desc
from ..models.chat import ChatMessage, ChatSession

class ChatRepository:
    """Repository for chat-related database operations."""
    
    def __init__(self, db_session: Session):
        """Initialize with database session."""
        self.db = db_session
    
    async def create_message(self, user_id: str, message: str, context: Dict[str, Any], 
                           correlation_id: str) -> ChatMessage:
        """Create a new chat message.
        
        Args:
            user_id: User ID
            message: Message content
            context: Message context
            correlation_id: Correlation ID for request tracking
            
        Returns:
            Created ChatMessage instance
        """
        chat_message = ChatMessage(
            user_id=user_id,
            message=message,
            context=context,
            correlation_id=correlation_id
        )
        self.db.add(chat_message)
        await self.db.flush()
        return chat_message
    
    async def update_message_response(self, message_id: str, response: Dict[str, Any]) -> ChatMessage:
        """Update message response.
        
        Args:
            message_id: Message ID
            response: Response data
            
        Returns:
            Updated ChatMessage instance
        """
        message = await self.db.query(ChatMessage).filter(ChatMessage.id == message_id).first()
        if message:
            message.response = response
            message.updated_at = datetime.utcnow()
            await self.db.flush()
        return message
    
    async def get_message_history(self, user_id: str, limit: int, 
                                before: Optional[datetime] = None) -> List[ChatMessage]:
        """Get chat message history.
        
        Args:
            user_id: User ID
            limit: Maximum number of messages
            before: Get messages before this timestamp
            
        Returns:
            List of ChatMessage instances
        """
        query = self.db.query(ChatMessage).filter(ChatMessage.user_id == user_id)
        
        if before:
            query = query.filter(ChatMessage.created_at < before)
        
        return await query.order_by(desc(ChatMessage.created_at)).limit(limit).all()
    
    async def create_session(self, user_id: str, session_data: Optional[Dict[str, Any]] = None) -> ChatSession:
        """Create a new chat session.
        
        Args:
            user_id: User ID
            session_data: Optional session data
            
        Returns:
            Created ChatSession instance
        """
        # First, deactivate any existing active sessions
        await self.deactivate_user_sessions(user_id)
        
        session = ChatSession(
            user_id=user_id,
            session_data=session_data,
            is_active=1
        )
        self.db.add(session)
        await self.db.flush()
        return session
    
    async def get_active_session(self, user_id: str) -> Optional[ChatSession]:
        """Get user's active chat session.
        
        Args:
            user_id: User ID
            
        Returns:
            Active ChatSession instance if exists
        """
        return await self.db.query(ChatSession)\
            .filter(ChatSession.user_id == user_id, ChatSession.is_active == 1)\
            .first()
    
    async def deactivate_user_sessions(self, user_id: str) -> None:
        """Deactivate all active sessions for a user.
        
        Args:
            user_id: User ID
        """
        await self.db.query(ChatSession)\
            .filter(ChatSession.user_id == user_id, ChatSession.is_active == 1)\
            .update({
                ChatSession.is_active: 0,
                ChatSession.ended_at: datetime.utcnow()
            })
        await self.db.flush()
    
    async def update_session_data(self, session_id: str, session_data: Dict[str, Any]) -> ChatSession:
        """Update session data.
        
        Args:
            session_id: Session ID
            session_data: New session data
            
        Returns:
            Updated ChatSession instance
        """
        session = await self.db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            session.session_data = session_data
            session.updated_at = datetime.utcnow()
            await self.db.flush()
        return session