"""
Chat Backend Service for Forex Trading Platform

This module provides the backend implementation for the chat interface,
handling message processing, NLP, and integration with other services.
"""

import logging
import json
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import asyncio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("chat-backend-service")

class ChatBackendService:
    """Backend service for the chat interface."""
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the chat backend service.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.message_history = {}  # User ID -> List of messages
        
        # Initialize NLP components if available
        self.nlp_processor = None
        try:
            from analysis_engine.analysis.nlp import BaseNLPAnalyzer
            self.nlp_processor = BaseNLPAnalyzer()
            logger.info("NLP processor initialized")
        except ImportError:
            logger.warning("NLP processor not available")
        
        # Initialize ML client if available
        self.ml_client = None
        try:
            from ml_integration_service.clients import get_ml_workbench_client
            self.ml_client = get_ml_workbench_client()
            logger.info("ML client initialized")
        except ImportError:
            logger.warning("ML client not available")
        
        # Initialize trading client if available
        self.trading_client = None
        try:
            from trading_gateway_service.clients import get_trading_client
            self.trading_client = get_trading_client()
            logger.info("Trading client initialized")
        except ImportError:
            logger.warning("Trading client not available")
    
    async def process_message(
        self, 
        user_id: str, 
        message: str, 
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Process a user message and generate a response.
        
        Args:
            user_id: User ID
            message: User message
            context: Message context (e.g., current symbol, timeframe)
            
        Returns:
            Response message
        """
        logger.info(f"Processing message from user {user_id}: {message}")
        
        # Store user message in history
        if user_id not in self.message_history:
            self.message_history[user_id] = []
        
        user_message = {
            "id": str(uuid.uuid4()),
            "text": message,
            "sender": "user",
            "timestamp": datetime.now().isoformat()
        }
        self.message_history[user_id].append(user_message)
        
        # Process message with NLP if available
        intent = None
        entities = []
        sentiment = None
        
        if self.nlp_processor:
            try:
                # Extract entities
                entities = self.nlp_processor.extract_entities(message)
                
                # Analyze sentiment
                sentiment = self.nlp_processor.analyze_sentiment(message)
                
                # Determine intent (simplified)
                intent = self._determine_intent(message, entities)
                
                logger.info(f"NLP results - Intent: {intent}, Entities: {len(entities)}, Sentiment: {sentiment['compound']}")
            except Exception as e:
                logger.error(f"Error processing message with NLP: {str(e)}")
        else:
            # Simple intent detection if NLP not available
            intent = self._simple_intent_detection(message)
        
        # Generate response based on intent
        response = await self._generate_response(user_id, message, intent, entities, sentiment, context)
        
        # Store assistant message in history
        assistant_message = {
            "id": str(uuid.uuid4()),
            "text": response["text"],
            "sender": "assistant",
            "timestamp": datetime.now().isoformat(),
            "tradingAction": response.get("tradingAction"),
            "chartData": response.get("chartData"),
            "attachments": response.get("attachments")
        }
        self.message_history[user_id].append(assistant_message)
        
        return response
    
    def _determine_intent(self, message: str, entities: List[Dict[str, Any]]) -> str:
        """
        Determine the intent of a message using NLP.
        
        Args:
            message: User message
            entities: Extracted entities
            
        Returns:
            Intent string
        """
        # This is a simplified implementation
        # In a real implementation, this would use a more sophisticated approach
        
        message_lower = message.lower()
        
        # Trading intents
        if any(term in message_lower for term in ["buy", "sell", "trade", "order", "position"]):
            return "trading"
        
        # Analysis intents
        if any(term in message_lower for term in ["analyze", "analysis", "predict", "forecast"]):
            return "analysis"
        
        # Chart intents
        if any(term in message_lower for term in ["chart", "graph", "plot", "visualization"]):
            return "chart"
        
        # Information intents
        if any(term in message_lower for term in ["what", "how", "explain", "tell me", "show me"]):
            return "information"
        
        # Default intent
        return "general"
    
    def _simple_intent_detection(self, message: str) -> str:
        """
        Simple intent detection without NLP.
        
        Args:
            message: User message
            
        Returns:
            Intent string
        """
        message_lower = message.lower()
        
        # Trading intents
        if any(term in message_lower for term in ["buy", "sell", "trade", "order", "position"]):
            return "trading"
        
        # Analysis intents
        if any(term in message_lower for term in ["analyze", "analysis", "predict", "forecast"]):
            return "analysis"
        
        # Chart intents
        if any(term in message_lower for term in ["chart", "graph", "plot", "visualization"]):
            return "chart"
        
        # Information intents
        if any(term in message_lower for term in ["what", "how", "explain", "tell me", "show me"]):
            return "information"
        
        # Default intent
        return "general"
    
    async def _generate_response(
        self, 
        user_id: str, 
        message: str, 
        intent: str, 
        entities: List[Dict[str, Any]], 
        sentiment: Optional[Dict[str, float]], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Generate a response based on intent and entities.
        
        Args:
            user_id: User ID
            message: User message
            intent: Detected intent
            entities: Extracted entities
            sentiment: Sentiment analysis results
            context: Message context
            
        Returns:
            Response message
        """
        # Default response
        response = {
            "text": "I'm your Forex Trading Assistant. I can help you analyze markets, execute trades, and monitor your portfolio. What would you like to do today?",
        }
        
        # Generate response based on intent
        if intent == "trading":
            response = await self._handle_trading_intent(message, entities, context)
        elif intent == "analysis":
            response = await self._handle_analysis_intent(message, entities, context)
        elif intent == "chart":
            response = await self._handle_chart_intent(message, entities, context)
        elif intent == "information":
            response = await self._handle_information_intent(message, entities, context)
        
        return response
    
    async def _handle_trading_intent(
        self, 
        message: str, 
        entities: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Handle trading intent.
        
        Args:
            message: User message
            entities: Extracted entities
            context: Message context
            
        Returns:
            Response message
        """
        # Extract trading parameters
        symbol = self._extract_symbol(message, entities, context)
        action_type = "buy" if "buy" in message.lower() else "sell"
        
        # Create trading action
        trading_action = {
            "type": action_type,
            "symbol": symbol
        }
        
        return {
            "text": f"I can help you {action_type} {symbol}. Would you like me to execute this trade for you?",
            "tradingAction": trading_action
        }
    
    async def _handle_analysis_intent(
        self, 
        message: str, 
        entities: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Handle analysis intent.
        
        Args:
            message: User message
            entities: Extracted entities
            context: Message context
            
        Returns:
            Response message
        """
        # Extract analysis parameters
        symbol = self._extract_symbol(message, entities, context)
        timeframe = self._extract_timeframe(message, entities, context)
        
        # Get analysis from ML client if available
        analysis_text = f"Based on my analysis of recent market data, {symbol} is showing a bullish trend on the {timeframe} timeframe. The RSI indicator is at 65, suggesting moderate bullish momentum, while the MACD is showing a recent crossover. Key resistance levels are at 1.0850 and 1.0900."
        
        if self.ml_client:
            try:
                # This would be replaced with actual ML client call
                # analysis = await self.ml_client.get_analysis(symbol, timeframe)
                pass
            except Exception as e:
                logger.error(f"Error getting analysis from ML client: {str(e)}")
        
        return {
            "text": analysis_text
        }
    
    async def _handle_chart_intent(
        self, 
        message: str, 
        entities: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Handle chart intent.
        
        Args:
            message: User message
            entities: Extracted entities
            context: Message context
            
        Returns:
            Response message
        """
        # Extract chart parameters
        symbol = self._extract_symbol(message, entities, context)
        timeframe = self._extract_timeframe(message, entities, context)
        
        # Create chart data (placeholder)
        chart_data = {
            "symbol": symbol,
            "timeframe": timeframe,
            "data": {}  # This would be actual chart data
        }
        
        return {
            "text": f"Here's the {timeframe} chart for {symbol}. I've highlighted some key support and resistance levels.",
            "chartData": chart_data
        }
    
    async def _handle_information_intent(
        self, 
        message: str, 
        entities: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Handle information intent.
        
        Args:
            message: User message
            entities: Extracted entities
            context: Message context
            
        Returns:
            Response message
        """
        # Simple information response
        return {
            "text": "I can provide information about forex trading, technical analysis, and market conditions. What specific information are you looking for?"
        }
    
    def _extract_symbol(
        self, 
        message: str, 
        entities: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Extract symbol from message, entities, or context.
        
        Args:
            message: User message
            entities: Extracted entities
            context: Message context
            
        Returns:
            Symbol string
        """
        # Check entities first
        for entity in entities:
            if entity["label"] == "CURRENCY_PAIR":
                return entity["text"]
        
        # Check message for common pairs
        common_pairs = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"]
        for pair in common_pairs:
            if pair in message.upper():
                return pair
        
        # Check context
        if context and "currentSymbol" in context:
            return context["currentSymbol"]
        
        # Default
        return "EURUSD"
    
    def _extract_timeframe(
        self, 
        message: str, 
        entities: List[Dict[str, Any]], 
        context: Optional[Dict[str, Any]]
    ) -> str:
        """
        Extract timeframe from message, entities, or context.
        
        Args:
            message: User message
            entities: Extracted entities
            context: Message context
            
        Returns:
            Timeframe string
        """
        # Check entities first
        for entity in entities:
            if entity["label"] == "TIMEFRAME":
                return entity["text"]
        
        # Check message for common timeframes
        timeframes = {
            "1m": ["1m", "1 minute", "1min"],
            "5m": ["5m", "5 minute", "5min"],
            "15m": ["15m", "15 minute", "15min"],
            "30m": ["30m", "30 minute", "30min"],
            "1h": ["1h", "1 hour", "hourly"],
            "4h": ["4h", "4 hour"],
            "1d": ["1d", "daily", "day"],
            "1w": ["1w", "weekly", "week"]
        }
        
        message_lower = message.lower()
        for tf, aliases in timeframes.items():
            if any(alias in message_lower for alias in aliases):
                return tf
        
        # Check context
        if context and "currentTimeframe" in context:
            return context["currentTimeframe"]
        
        # Default
        return "1h"
    
    async def execute_trading_action(
        self, 
        user_id: str, 
        action: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Execute a trading action.
        
        Args:
            user_id: User ID
            action: Trading action
            
        Returns:
            Result of the action
        """
        logger.info(f"Executing trading action for user {user_id}: {action}")
        
        if not self.trading_client:
            logger.warning("Trading client not available")
            return {
                "success": False,
                "message": "Trading functionality is not available"
            }
        
        try:
            # This would be replaced with actual trading client call
            # result = await self.trading_client.execute_order(action)
            
            # Simulate successful execution
            result = {
                "success": True,
                "orderId": str(uuid.uuid4()),
                "message": f"Successfully executed {action['type']} order for {action['symbol']}",
                "timestamp": datetime.now().isoformat()
            }
            
            return result
        except Exception as e:
            logger.error(f"Error executing trading action: {str(e)}")
            return {
                "success": False,
                "message": f"Error executing trading action: {str(e)}"
            }
    
    def get_chat_history(
        self, 
        user_id: str, 
        limit: int = 50, 
        before: Optional[datetime] = None
    ) -> List[Dict[str, Any]]:
        """
        Get chat history for a user.
        
        Args:
            user_id: User ID
            limit: Maximum number of messages to return
            before: Get messages before this timestamp
            
        Returns:
            List of messages
        """
        if user_id not in self.message_history:
            return []
        
        messages = self.message_history[user_id]
        
        # Filter by timestamp if before is provided
        if before:
            before_str = before.isoformat()
            messages = [msg for msg in messages if msg["timestamp"] < before_str]
        
        # Sort by timestamp (newest first) and limit
        messages = sorted(messages, key=lambda msg: msg["timestamp"], reverse=True)[:limit]
        
        # Sort by timestamp (oldest first) for return
        return sorted(messages, key=lambda msg: msg["timestamp"])
    
    def clear_chat_history(self, user_id: str) -> bool:
        """
        Clear chat history for a user.
        
        Args:
            user_id: User ID
            
        Returns:
            True if successful, False otherwise
        """
        if user_id in self.message_history:
            self.message_history[user_id] = []
            return True
        return False
