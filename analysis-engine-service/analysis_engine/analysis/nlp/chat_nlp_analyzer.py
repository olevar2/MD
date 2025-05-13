"""
Chat NLP Analyzer Module

This module provides NLP capabilities specifically for the chat interface,
including intent recognition, entity extraction, and context management.
"""
from typing import Dict, List, Any, Optional, Union
import logging
from datetime import datetime
from analysis_engine.analysis.nlp.base_nlp_analyzer import BaseNLPAnalyzer
from analysis_engine.analysis.nlp.intent_recognition import IntentRecognizer
from analysis_engine.analysis.nlp.entity_extraction import EntityExtractor
from analysis_engine.analysis.nlp.context_management import ContextManager
logger = logging.getLogger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class ChatNLPAnalyzer(BaseNLPAnalyzer):
    """
    NLP analyzer specialized for chat interface interactions.

    This analyzer extends the BaseNLPAnalyzer with chat-specific functionality
    for intent recognition, entity extraction, and context management.
    """

    def __init__(self, parameters: Dict[str, Any]=None):
        """
        Initialize the chat NLP analyzer

        Args:
            parameters: Configuration parameters for the analyzer
        """
        super().__init__('chat_nlp_analyzer', parameters)
        self.intent_recognizer = IntentRecognizer(parameters)
        self.entity_extractor = EntityExtractor(parameters)
        self.context_manager = ContextManager(parameters)

    @with_resilience('process_message')
    def process_message(self, user_id: str, message: str, context: Dict[str,
        Any]=None) ->Dict[str, Any]:
        """
        Process a user message with NLP analysis.

        Args:
            user_id: User ID
            message: User message
            context: Optional context information

        Returns:
            Dictionary with NLP analysis results
        """
        user_context = self.context_manager.get_context(user_id)
        entities = self.entity_extractor.extract_entities(message)
        intent = self.intent_recognizer.recognize_intent(message, entities,
            user_context)
        result = {'intent': intent, 'entities': entities, 'context':
            user_context}
        return result

    @with_resilience('update_context')
    def update_context(self, user_id: str, message: str, intent: Dict[str,
        Any], entities: List[Dict[str, Any]], response: Dict[str, Any]) ->Dict[
        str, Any]:
        """
        Update the conversation context.

        Args:
            user_id: User ID
            message: User message
            intent: Recognized intent
            entities: Extracted entities
            response: Generated response

        Returns:
            Updated context
        """
        return self.context_manager.update_context(user_id, message, intent,
            entities, response)

    @with_resilience('get_primary_intent')
    def get_primary_intent(self, message: str, entities: List[Dict[str, Any
        ]]=None, context: Dict[str, Any]=None) ->str:
        """
        Get the primary intent of a message.

        Args:
            message: User message
            entities: Optional list of extracted entities
            context: Optional context information

        Returns:
            Primary intent name
        """
        intent_info = self.intent_recognizer.recognize_intent(message,
            entities, context)
        return intent_info['primary']['intent']

    def extract_custom_entities(self, message: str) ->List[Dict[str, Any]]:
        """
        Extract domain-specific entities from message.

        Args:
            message: User message

        Returns:
            List of extracted entities
        """
        return self.entity_extractor.extract_entities(message)

    @with_resilience('get_entity_from_context')
    def get_entity_from_context(self, user_id: str, entity_type: str
        ) ->Optional[Any]:
        """
        Get an entity value from context.

        Args:
            user_id: User ID
            entity_type: Entity type to retrieve

        Returns:
            Entity value or None if not found
        """
        return self.context_manager.get_entity_from_context(user_id,
            entity_type)

    def set_user_preference(self, user_id: str, preference_key: str,
        preference_value: Any) ->None:
        """
        Set a user preference.

        Args:
            user_id: User ID
            preference_key: Preference key
            preference_value: Preference value
        """
        self.context_manager.set_user_preference(user_id, preference_key,
            preference_value)

    @with_resilience('get_user_preference')
    def get_user_preference(self, user_id: str, preference_key: str,
        default_value: Any=None) ->Any:
        """
        Get a user preference.

        Args:
            user_id: User ID
            preference_key: Preference key
            default_value: Default value if preference not found

        Returns:
            Preference value or default value if not found
        """
        return self.context_manager.get_user_preference(user_id,
            preference_key, default_value)

    @with_resilience('get_conversation_summary')
    def get_conversation_summary(self, user_id: str) ->Dict[str, Any]:
        """
        Get a summary of the conversation.

        Args:
            user_id: User ID

        Returns:
            Conversation summary
        """
        return self.context_manager.get_conversation_summary(user_id)

    def clear_context(self, user_id: str) ->None:
        """
        Clear the context for a user.

        Args:
            user_id: User ID
        """
        self.context_manager.clear_context(user_id)
