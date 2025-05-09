"""
Context Management Module

This module provides context management capabilities for multi-turn conversations,
tracking conversation history, user preferences, and contextual information.
"""

from typing import Dict, List, Any, Optional, Deque
import logging
from collections import deque
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)

class ContextManager:
    """
    Context management for multi-turn conversations.
    
    This class manages conversation context, including conversation history,
    user preferences, and contextual information for multi-turn conversations.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the context manager.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Maximum number of turns to keep in history
        self.max_history_turns = self.config.get("max_history_turns", 10)
        
        # Maximum age of context in seconds
        self.max_context_age = self.config.get("max_context_age", 3600)  # 1 hour
        
        # Initialize context store
        self.context_store = {}
    
    def get_context(self, user_id: str) -> Dict[str, Any]:
        """
        Get the current context for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            Current context
        """
        # Initialize context if not exists
        if user_id not in self.context_store:
            self._initialize_user_context(user_id)
        
        # Check if context is expired
        if self._is_context_expired(user_id):
            self._initialize_user_context(user_id)
        
        return self.context_store[user_id]
    
    def update_context(
        self, 
        user_id: str, 
        message: str, 
        intent: Dict[str, Any],
        entities: List[Dict[str, Any]],
        response: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Update the context with a new message and response.
        
        Args:
            user_id: User ID
            message: User message
            intent: Recognized intent
            entities: Extracted entities
            response: Generated response
            
        Returns:
            Updated context
        """
        # Get current context
        context = self.get_context(user_id)
        
        # Update last activity time
        context["last_activity"] = datetime.now().isoformat()
        
        # Update conversation history
        self._update_conversation_history(context, message, response)
        
        # Update current state
        self._update_current_state(context, intent, entities, response)
        
        # Update entity memory
        self._update_entity_memory(context, entities)
        
        # Update conversation flow
        self._update_conversation_flow(context, intent)
        
        # Save updated context
        self.context_store[user_id] = context
        
        return context
    
    def clear_context(self, user_id: str) -> None:
        """
        Clear the context for a user.
        
        Args:
            user_id: User ID
        """
        if user_id in self.context_store:
            self._initialize_user_context(user_id)
    
    def _initialize_user_context(self, user_id: str) -> None:
        """
        Initialize context for a new user.
        
        Args:
            user_id: User ID
        """
        self.context_store[user_id] = {
            "user_id": user_id,
            "created_at": datetime.now().isoformat(),
            "last_activity": datetime.now().isoformat(),
            "conversation_history": deque(maxlen=self.max_history_turns),
            "current_state": {
                "intent": None,
                "sub_intent": None,
                "entities": [],
                "topic": None
            },
            "entity_memory": {
                "currency_pair": None,
                "timeframe": None,
                "indicator": None,
                "price": None,
                "amount": None
            },
            "conversation_flow": None,
            "user_preferences": {}
        }
    
    def _is_context_expired(self, user_id: str) -> bool:
        """
        Check if the context is expired.
        
        Args:
            user_id: User ID
            
        Returns:
            True if context is expired, False otherwise
        """
        context = self.context_store[user_id]
        last_activity = datetime.fromisoformat(context["last_activity"])
        now = datetime.now()
        
        return (now - last_activity).total_seconds() > self.max_context_age
    
    def _update_conversation_history(
        self, 
        context: Dict[str, Any],
        message: str,
        response: Dict[str, Any]
    ) -> None:
        """
        Update the conversation history.
        
        Args:
            context: User context
            message: User message
            response: Generated response
        """
        history = context["conversation_history"]
        
        # Add new turn to history
        history.append({
            "timestamp": datetime.now().isoformat(),
            "user_message": message,
            "assistant_response": response
        })
    
    def _update_current_state(
        self, 
        context: Dict[str, Any],
        intent: Dict[str, Any],
        entities: List[Dict[str, Any]],
        response: Dict[str, Any]
    ) -> None:
        """
        Update the current state.
        
        Args:
            context: User context
            intent: Recognized intent
            entities: Extracted entities
            response: Generated response
        """
        current_state = context["current_state"]
        
        # Update intent
        if intent:
            current_state["intent"] = intent.get("primary", {}).get("intent")
            current_state["sub_intent"] = intent.get("sub_intent", {}).get("intent")
        
        # Update entities
        current_state["entities"] = entities
        
        # Determine topic based on intent and entities
        if current_state["intent"] == "trading":
            current_state["topic"] = "trading"
        elif current_state["intent"] == "analysis":
            current_state["topic"] = "analysis"
        elif current_state["intent"] == "chart":
            current_state["topic"] = "chart"
        elif current_state["intent"] == "information":
            # Determine specific information topic
            for entity in entities:
                if entity["label"] == "INDICATOR":
                    current_state["topic"] = f"information_indicator_{entity['value']}"
                    break
                elif entity["label"] == "TERM":
                    current_state["topic"] = f"information_term_{entity['value']}"
                    break
            else:
                current_state["topic"] = "information_general"
    
    def _update_entity_memory(
        self, 
        context: Dict[str, Any],
        entities: List[Dict[str, Any]]
    ) -> None:
        """
        Update the entity memory.
        
        Args:
            context: User context
            entities: Extracted entities
        """
        entity_memory = context["entity_memory"]
        
        # Update entity memory with new entities
        for entity in entities:
            if entity["label"] == "CURRENCY_PAIR":
                entity_memory["currency_pair"] = entity["value"]
            elif entity["label"] == "TIMEFRAME":
                entity_memory["timeframe"] = entity["value"]
            elif entity["label"] == "INDICATOR":
                entity_memory["indicator"] = entity["value"]
            elif entity["label"] == "PRICE":
                entity_memory["price"] = entity["value"]
            elif entity["label"] == "AMOUNT":
                entity_memory["amount"] = entity["value"]
    
    def _update_conversation_flow(
        self, 
        context: Dict[str, Any],
        intent: Dict[str, Any]
    ) -> None:
        """
        Update the conversation flow.
        
        Args:
            context: User context
            intent: Recognized intent
        """
        current_state = context["current_state"]
        previous_intent = context.get("previous_intent")
        
        # Determine conversation flow based on current and previous intent
        if previous_intent == "analysis" and current_state["intent"] == "trading":
            context["conversation_flow"] = "analysis_to_trading"
        elif previous_intent == "information" and current_state["intent"] == "chart":
            context["conversation_flow"] = "information_to_chart"
        elif previous_intent == "chart" and current_state["intent"] == "analysis":
            context["conversation_flow"] = "chart_to_analysis"
        else:
            context["conversation_flow"] = None
        
        # Update previous intent
        context["previous_intent"] = current_state["intent"]
    
    def get_entity_from_context(
        self, 
        user_id: str, 
        entity_type: str
    ) -> Optional[Any]:
        """
        Get an entity value from context.
        
        Args:
            user_id: User ID
            entity_type: Entity type to retrieve
            
        Returns:
            Entity value or None if not found
        """
        context = self.get_context(user_id)
        entity_memory = context["entity_memory"]
        
        return entity_memory.get(entity_type)
    
    def set_user_preference(
        self, 
        user_id: str, 
        preference_key: str, 
        preference_value: Any
    ) -> None:
        """
        Set a user preference.
        
        Args:
            user_id: User ID
            preference_key: Preference key
            preference_value: Preference value
        """
        context = self.get_context(user_id)
        context["user_preferences"][preference_key] = preference_value
    
    def get_user_preference(
        self, 
        user_id: str, 
        preference_key: str, 
        default_value: Any = None
    ) -> Any:
        """
        Get a user preference.
        
        Args:
            user_id: User ID
            preference_key: Preference key
            default_value: Default value if preference not found
            
        Returns:
            Preference value or default value if not found
        """
        context = self.get_context(user_id)
        return context["user_preferences"].get(preference_key, default_value)
    
    def get_conversation_summary(self, user_id: str) -> Dict[str, Any]:
        """
        Get a summary of the conversation.
        
        Args:
            user_id: User ID
            
        Returns:
            Conversation summary
        """
        context = self.get_context(user_id)
        
        # Extract relevant information for summary
        summary = {
            "turn_count": len(context["conversation_history"]),
            "current_topic": context["current_state"]["topic"],
            "current_intent": context["current_state"]["intent"],
            "current_currency_pair": context["entity_memory"]["currency_pair"],
            "current_timeframe": context["entity_memory"]["timeframe"],
            "conversation_flow": context["conversation_flow"],
            "last_activity": context["last_activity"]
        }
        
        return summary
