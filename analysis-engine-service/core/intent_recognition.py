"""
Intent Recognition Module

This module provides advanced intent recognition capabilities for the chat interface,
using a combination of pattern matching, keyword analysis, and context awareness.
"""

from typing import Dict, List, Any, Optional, Tuple
import logging
import re
from collections import defaultdict

logger = logging.getLogger(__name__)

class IntentRecognizer:
    """
    Advanced intent recognition for chat messages.
    
    This class provides sophisticated intent recognition capabilities,
    combining pattern matching, keyword analysis, and context awareness.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the intent recognizer.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        
        # Define intent categories and their patterns
        self.intent_categories = {
            "trading": {
                "patterns": [
                    r'\b(buy|sell|trade|order|position|entry|exit|stop loss|take profit)\b',
                    r'\b(long|short|market order|limit order|open|close)\b',
                    r'\b(execute|place|submit|cancel|modify)\b',
                ],
                "keywords": [
                    "buy", "sell", "trade", "order", "position", "entry", "exit", 
                    "stop", "loss", "take", "profit", "long", "short", "market", 
                    "limit", "execute", "place", "submit", "cancel", "modify"
                ],
                "boost_entities": ["CURRENCY_PAIR", "PRICE", "AMOUNT"]
            },
            "analysis": {
                "patterns": [
                    r'\b(analyze|analysis|predict|forecast|trend|outlook|sentiment)\b',
                    r'\b(technical|fundamental|indicator|oscillator|moving average|rsi|macd)\b',
                    r'\b(strength|weakness|momentum|volatility|volume|pressure)\b',
                ],
                "keywords": [
                    "analyze", "analysis", "predict", "forecast", "trend", "outlook", 
                    "sentiment", "technical", "fundamental", "indicator", "oscillator", 
                    "moving", "average", "rsi", "macd", "strength", "weakness", 
                    "momentum", "volatility", "volume", "pressure"
                ],
                "boost_entities": ["CURRENCY_PAIR", "TIMEFRAME", "INDICATOR"]
            },
            "chart": {
                "patterns": [
                    r'\b(chart|graph|plot|visualization|candlestick|line chart|bar chart)\b',
                    r'\b(timeframe|period|interval|daily|hourly|weekly|monthly)\b',
                    r'\b(show|display|view|see|look at)\b',
                ],
                "keywords": [
                    "chart", "graph", "plot", "visualization", "candlestick", "line", 
                    "bar", "timeframe", "period", "interval", "daily", "hourly", 
                    "weekly", "monthly", "show", "display", "view", "see", "look"
                ],
                "boost_entities": ["CURRENCY_PAIR", "TIMEFRAME", "INDICATOR"]
            },
            "information": {
                "patterns": [
                    r'\b(what|how|explain|tell me|show me|info|information|details)\b',
                    r'\b(why|when|where|who|which|describe|definition|meaning)\b',
                    r'\b(learn|understand|know|clarify|elaborate)\b',
                ],
                "keywords": [
                    "what", "how", "explain", "tell", "show", "info", "information", 
                    "details", "why", "when", "where", "who", "which", "describe", 
                    "definition", "meaning", "learn", "understand", "know", "clarify", 
                    "elaborate"
                ],
                "boost_entities": ["INDICATOR", "TERM"]
            },
            "account": {
                "patterns": [
                    r'\b(account|balance|equity|margin|profit|loss|performance)\b',
                    r'\b(history|statement|report|summary|statistics)\b',
                    r'\b(deposit|withdraw|transfer|fund)\b',
                ],
                "keywords": [
                    "account", "balance", "equity", "margin", "profit", "loss", 
                    "performance", "history", "statement", "report", "summary", 
                    "statistics", "deposit", "withdraw", "transfer", "fund"
                ],
                "boost_entities": ["AMOUNT", "DATE"]
            },
            "settings": {
                "patterns": [
                    r'\b(settings|preferences|configure|setup|options|parameters)\b',
                    r'\b(change|update|modify|set|adjust)\b',
                    r'\b(default|profile|theme|notification|alert)\b',
                ],
                "keywords": [
                    "settings", "preferences", "configure", "setup", "options", 
                    "parameters", "change", "update", "modify", "set", "adjust", 
                    "default", "profile", "theme", "notification", "alert"
                ],
                "boost_entities": []
            },
        }
        
        # Add sub-intents for more specific recognition
        self.sub_intents = {
            "trading": {
                "buy": {
                    "patterns": [r'\b(buy|long|purchase|acquire)\b'],
                    "keywords": ["buy", "long", "purchase", "acquire"]
                },
                "sell": {
                    "patterns": [r'\b(sell|short|dispose|offload)\b'],
                    "keywords": ["sell", "short", "dispose", "offload"]
                },
                "modify": {
                    "patterns": [r'\b(modify|change|update|adjust|edit)\b'],
                    "keywords": ["modify", "change", "update", "adjust", "edit"]
                },
                "close": {
                    "patterns": [r'\b(close|exit|terminate|end)\b'],
                    "keywords": ["close", "exit", "terminate", "end"]
                }
            },
            "analysis": {
                "technical": {
                    "patterns": [r'\b(technical|indicator|oscillator|chart pattern|price action)\b'],
                    "keywords": ["technical", "indicator", "oscillator", "chart", "pattern", "price", "action"]
                },
                "fundamental": {
                    "patterns": [r'\b(fundamental|news|economic|report|announcement)\b'],
                    "keywords": ["fundamental", "news", "economic", "report", "announcement"]
                },
                "sentiment": {
                    "patterns": [r'\b(sentiment|market sentiment|mood|feeling|positioning)\b'],
                    "keywords": ["sentiment", "market", "mood", "feeling", "positioning"]
                }
            },
            "chart": {
                "view": {
                    "patterns": [r'\b(view|show|display|see|look at)\b'],
                    "keywords": ["view", "show", "display", "see", "look"]
                },
                "customize": {
                    "patterns": [r'\b(customize|configure|setup|adjust|change)\b'],
                    "keywords": ["customize", "configure", "setup", "adjust", "change"]
                }
            }
        }
        
        # Initialize context weights
        self.context_weights = {
            "previous_intent": 0.3,  # Weight for previous intent
            "conversation_flow": 0.2,  # Weight for conversation flow
            "user_preferences": 0.1,  # Weight for user preferences
        }
    
    def recognize_intent(
        self, 
        message: str, 
        entities: List[Dict[str, Any]] = None,
        context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Recognize the intent of a message.
        
        Args:
            message: User message
            entities: Extracted entities
            context: Conversation context
            
        Returns:
            Dictionary with intent information
        """
        entities = entities or []
        context = context or {}
        
        # Initialize scores for each intent category
        intent_scores = {intent: 0.0 for intent in self.intent_categories.keys()}
        intent_scores["general"] = 0.1  # Default intent has a small base score
        
        # Calculate base scores from patterns and keywords
        message_lower = message.lower()
        
        # Check patterns for each intent
        for intent, config in self.intent_categories.items():
            # Check patterns
            for pattern in config["patterns"]:
                if re.search(pattern, message_lower, re.IGNORECASE):
                    intent_scores[intent] += 0.3
            
            # Check keywords
            for keyword in config["keywords"]:
                if keyword in message_lower:
                    intent_scores[intent] += 0.1
            
            # Boost score based on entities
            for entity in entities:
                if entity["label"] in config["boost_entities"]:
                    intent_scores[intent] += 0.1
        
        # Apply context-based adjustments
        if context:
            self._apply_context_adjustments(intent_scores, context)
        
        # Determine primary and secondary intents
        sorted_intents = sorted(intent_scores.items(), key=lambda x: x[1], reverse=True)
        primary_intent = sorted_intents[0][0]
        primary_score = sorted_intents[0][1]
        
        # Determine sub-intent if applicable
        sub_intent = None
        sub_intent_score = 0.0
        
        if primary_intent in self.sub_intents:
            sub_intent_scores = {}
            for sub, sub_config in self.sub_intents[primary_intent].items():
                score = 0.0
                
                # Check patterns
                for pattern in sub_config["patterns"]:
                    if re.search(pattern, message_lower, re.IGNORECASE):
                        score += 0.3
                
                # Check keywords
                for keyword in sub_config["keywords"]:
                    if keyword in message_lower:
                        score += 0.1
                
                sub_intent_scores[sub] = score
            
            # Get highest scoring sub-intent
            if sub_intent_scores:
                top_sub_intent = max(sub_intent_scores.items(), key=lambda x: x[1])
                if top_sub_intent[1] > 0:
                    sub_intent = top_sub_intent[0]
                    sub_intent_score = top_sub_intent[1]
        
        # Prepare result
        result = {
            "primary": {
                "intent": primary_intent,
                "confidence": primary_score
            },
            "secondary": {
                "intent": sorted_intents[1][0] if len(sorted_intents) > 1 else None,
                "confidence": sorted_intents[1][1] if len(sorted_intents) > 1 else 0.0
            },
            "sub_intent": {
                "intent": sub_intent,
                "confidence": sub_intent_score
            },
            "all_scores": intent_scores
        }
        
        return result
    
    def _apply_context_adjustments(
        self, 
        intent_scores: Dict[str, float],
        context: Dict[str, Any]
    ) -> None:
        """
        Apply context-based adjustments to intent scores.
        
        Args:
            intent_scores: Intent scores to adjust
            context: Conversation context
        """
        # Adjust based on previous intent
        if "previous_intent" in context:
            prev_intent = context["previous_intent"]
            if prev_intent in intent_scores:
                # Boost the previous intent slightly for continuity
                intent_scores[prev_intent] += self.context_weights["previous_intent"]
        
        # Adjust based on conversation flow
        if "conversation_flow" in context:
            flow = context["conversation_flow"]
            if flow == "analysis_to_trading":
                # If the flow is from analysis to trading, boost trading intent
                intent_scores["trading"] += self.context_weights["conversation_flow"]
            elif flow == "information_to_chart":
                # If the flow is from information to chart, boost chart intent
                intent_scores["chart"] += self.context_weights["conversation_flow"]
        
        # Adjust based on user preferences
        if "user_preferences" in context:
            preferences = context["user_preferences"]
            if "preferred_analysis_type" in preferences:
                # Boost analysis intent if user prefers analysis
                if preferences["preferred_analysis_type"] == "technical":
                    intent_scores["analysis"] += self.context_weights["user_preferences"]
                    intent_scores["chart"] += self.context_weights["user_preferences"] / 2
            
            if "trading_frequency" in preferences and preferences["trading_frequency"] == "high":
                # Boost trading intent if user trades frequently
                intent_scores["trading"] += self.context_weights["user_preferences"]
