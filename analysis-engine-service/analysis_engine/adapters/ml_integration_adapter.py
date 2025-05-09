"""
ML Integration Adapter Module

This module provides adapters for ML integration functionality,
helping to break circular dependencies between services.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime, timedelta
import os
import httpx
import asyncio
import json

from common_lib.ml.interfaces import (
    IMLModelConnector,
    IExplanationGenerator,
    IUserPreferenceManager,
    ModelConfiguration,
    ModelPrediction
)

logger = logging.getLogger(__name__)


class MLModelConnectorAdapter(IMLModelConnector):
    """
    Adapter for ML model connector functionality.
    
    This adapter can either use a direct connection to the ML integration service
    or provide standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Get ML integration service URL from config or environment
        ml_integration_base_url = self.config.get(
            "ml_integration_base_url", 
            os.environ.get("ML_INTEGRATION_BASE_URL", "http://ml-integration-service:8000")
        )
        
        # Set up the client with resolved URL
        self.client = httpx.AsyncClient(
            base_url=f"{ml_integration_base_url.rstrip('/')}/api/v1",
            timeout=30.0
        )
        
        # Cache for predictions
        self.prediction_cache = {}
        self.cache_ttl = self.config.get("cache_ttl_minutes", 15)  # Cache TTL in minutes
    
    async def get_market_analysis(
        self,
        symbol: str,
        timeframe: str,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get market analysis for a symbol."""
        try:
            # Check cache first
            cache_key = f"market_analysis_{symbol}_{timeframe}"
            if cache_key in self.prediction_cache:
                cache_entry = self.prediction_cache[cache_key]
                if datetime.now() < cache_entry["expiry"]:
                    return cache_entry["data"]
            
            # Prepare request parameters
            params = {
                "symbol": symbol,
                "timeframe": timeframe
            }
            
            # Add user preferences if provided
            if user_preferences:
                params["user_preferences"] = json.dumps(user_preferences)
            
            # Make API request
            response = await self.client.get("/ml/market-analysis", params=params)
            response.raise_for_status()
            
            # Parse response
            analysis_data = response.json()
            
            # Cache the result
            self.prediction_cache[cache_key] = {
                "data": analysis_data,
                "expiry": datetime.now() + timedelta(minutes=self.cache_ttl)
            }
            
            return analysis_data
            
        except Exception as e:
            self.logger.error(f"Error getting market analysis: {str(e)}")
            
            # Return fallback analysis
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "trend": {
                    "direction": "neutral",
                    "strength": 0.5,
                    "confidence": 0.5
                },
                "support_resistance": {
                    "support": [],
                    "resistance": []
                },
                "indicators": {},
                "regime": "unknown"
            }
    
    async def get_price_prediction(
        self,
        symbol: str,
        timeframe: str,
        horizon: str = "short_term",
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get price prediction for a symbol."""
        try:
            # Check cache first
            cache_key = f"price_prediction_{symbol}_{timeframe}_{horizon}"
            if cache_key in self.prediction_cache:
                cache_entry = self.prediction_cache[cache_key]
                if datetime.now() < cache_entry["expiry"]:
                    return cache_entry["data"]
            
            # Prepare request parameters
            params = {
                "symbol": symbol,
                "timeframe": timeframe,
                "horizon": horizon
            }
            
            # Add user preferences if provided
            if user_preferences:
                params["user_preferences"] = json.dumps(user_preferences)
            
            # Make API request
            response = await self.client.get("/ml/price-prediction", params=params)
            response.raise_for_status()
            
            # Parse response
            prediction_data = response.json()
            
            # Cache the result
            self.prediction_cache[cache_key] = {
                "data": prediction_data,
                "expiry": datetime.now() + timedelta(minutes=self.cache_ttl)
            }
            
            return prediction_data
            
        except Exception as e:
            self.logger.error(f"Error getting price prediction: {str(e)}")
            
            # Return fallback prediction
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "horizon": horizon,
                "timestamp": datetime.now().isoformat(),
                "direction": "neutral",
                "target_price": None,
                "confidence": 0.5,
                "prediction_window": "24h"
            }
    
    async def get_trading_recommendation(
        self,
        symbol: str,
        timeframe: str,
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get trading recommendation for a symbol."""
        try:
            # Check cache first
            cache_key = f"trading_recommendation_{symbol}_{timeframe}"
            if cache_key in self.prediction_cache:
                cache_entry = self.prediction_cache[cache_key]
                if datetime.now() < cache_entry["expiry"]:
                    return cache_entry["data"]
            
            # Prepare request parameters
            params = {
                "symbol": symbol,
                "timeframe": timeframe
            }
            
            # Add user preferences if provided
            if user_preferences:
                params["user_preferences"] = json.dumps(user_preferences)
            
            # Make API request
            response = await self.client.get("/ml/trading-recommendation", params=params)
            response.raise_for_status()
            
            # Parse response
            recommendation_data = response.json()
            
            # Cache the result
            self.prediction_cache[cache_key] = {
                "data": recommendation_data,
                "expiry": datetime.now() + timedelta(minutes=self.cache_ttl)
            }
            
            return recommendation_data
            
        except Exception as e:
            self.logger.error(f"Error getting trading recommendation: {str(e)}")
            
            # Return fallback recommendation
            return {
                "symbol": symbol,
                "timeframe": timeframe,
                "timestamp": datetime.now().isoformat(),
                "action": "hold",
                "confidence": 0.5,
                "risk_level": "medium",
                "entry_price": None,
                "stop_loss": None,
                "take_profit": None
            }
    
    async def get_sentiment_analysis(
        self,
        symbol: str,
        lookback_days: int = 7
    ) -> Dict[str, Any]:
        """Get sentiment analysis for a symbol."""
        try:
            # Check cache first
            cache_key = f"sentiment_analysis_{symbol}_{lookback_days}"
            if cache_key in self.prediction_cache:
                cache_entry = self.prediction_cache[cache_key]
                if datetime.now() < cache_entry["expiry"]:
                    return cache_entry["data"]
            
            # Prepare request parameters
            params = {
                "symbol": symbol,
                "lookback_days": lookback_days
            }
            
            # Make API request
            response = await self.client.get("/ml/sentiment-analysis", params=params)
            response.raise_for_status()
            
            # Parse response
            sentiment_data = response.json()
            
            # Cache the result
            self.prediction_cache[cache_key] = {
                "data": sentiment_data,
                "expiry": datetime.now() + timedelta(minutes=self.cache_ttl)
            }
            
            return sentiment_data
            
        except Exception as e:
            self.logger.error(f"Error getting sentiment analysis: {str(e)}")
            
            # Return fallback sentiment
            return {
                "symbol": symbol,
                "timestamp": datetime.now().isoformat(),
                "overall_sentiment": "neutral",
                "sentiment_score": 0.5,
                "confidence": 0.5,
                "sources": [],
                "lookback_days": lookback_days
            }


class ExplanationGeneratorAdapter(IExplanationGenerator):
    """
    Adapter for explanation generator functionality.
    
    This adapter can either use a direct connection to the ML integration service
    or provide standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Get ML integration service URL from config or environment
        ml_integration_base_url = self.config.get(
            "ml_integration_base_url", 
            os.environ.get("ML_INTEGRATION_BASE_URL", "http://ml-integration-service:8000")
        )
        
        # Set up the client with resolved URL
        self.client = httpx.AsyncClient(
            base_url=f"{ml_integration_base_url.rstrip('/')}/api/v1",
            timeout=30.0
        )
    
    async def generate_explanation(
        self,
        model_type: str,
        prediction: Dict[str, Any],
        inputs: Dict[str, Any],
        user_preferences: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate explanation for a model prediction."""
        try:
            # Prepare request data
            request_data = {
                "model_type": model_type,
                "prediction": prediction,
                "inputs": inputs
            }
            
            # Add user preferences if provided
            if user_preferences:
                request_data["user_preferences"] = user_preferences
            
            # Make API request
            response = await self.client.post("/ml/explanation", json=request_data)
            response.raise_for_status()
            
            # Parse response
            explanation_data = response.json()
            
            return explanation_data
            
        except Exception as e:
            self.logger.error(f"Error generating explanation: {str(e)}")
            
            # Return fallback explanation
            return {
                "explanation_text": "No explanation available.",
                "feature_importance": {},
                "confidence": prediction.get("confidence", 0.5) if isinstance(prediction, dict) else 0.5,
                "model_type": model_type
            }
    
    async def get_feature_importance(
        self,
        model_type: str,
        model_id: str,
        prediction: Dict[str, Any],
        inputs: Dict[str, Any]
    ) -> Dict[str, float]:
        """Get feature importance for a model prediction."""
        try:
            # Prepare request data
            request_data = {
                "model_type": model_type,
                "model_id": model_id,
                "prediction": prediction,
                "inputs": inputs
            }
            
            # Make API request
            response = await self.client.post("/ml/feature-importance", json=request_data)
            response.raise_for_status()
            
            # Parse response
            importance_data = response.json()
            
            return importance_data
            
        except Exception as e:
            self.logger.error(f"Error getting feature importance: {str(e)}")
            
            # Return fallback feature importance
            if isinstance(inputs, dict):
                # Generate random importance for input features
                import random
                features = list(inputs.keys())
                importance = {feature: random.random() for feature in features}
                # Normalize to sum to 1
                total = sum(importance.values())
                if total > 0:
                    importance = {k: v / total for k, v in importance.items()}
                return importance
            else:
                return {}


class UserPreferenceManagerAdapter(IUserPreferenceManager):
    """
    Adapter for user preference manager functionality.
    
    This adapter can either use a direct connection to the ML integration service
    or provide standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Get ML integration service URL from config or environment
        ml_integration_base_url = self.config.get(
            "ml_integration_base_url", 
            os.environ.get("ML_INTEGRATION_BASE_URL", "http://ml-integration-service:8000")
        )
        
        # Set up the client with resolved URL
        self.client = httpx.AsyncClient(
            base_url=f"{ml_integration_base_url.rstrip('/')}/api/v1",
            timeout=30.0
        )
        
        # Local cache for user preferences
        self.preferences_cache = {}
        self.cache_ttl = self.config.get("cache_ttl_minutes", 15)  # Cache TTL in minutes
    
    async def get_user_preferences(
        self,
        user_id: str
    ) -> Dict[str, Any]:
        """Get user preferences."""
        try:
            # Check cache first
            if user_id in self.preferences_cache:
                cache_entry = self.preferences_cache[user_id]
                if datetime.now() < cache_entry["expiry"]:
                    return cache_entry["preferences"]
            
            # Make API request
            response = await self.client.get(f"/users/{user_id}/preferences")
            response.raise_for_status()
            
            # Parse response
            preferences = response.json()
            
            # Cache the result
            self.preferences_cache[user_id] = {
                "preferences": preferences,
                "expiry": datetime.now() + timedelta(minutes=self.cache_ttl)
            }
            
            return preferences
            
        except Exception as e:
            self.logger.error(f"Error getting user preferences: {str(e)}")
            
            # Return default preferences
            return {
                "risk_profile": "moderate",
                "preferred_timeframes": ["1h", "4h", "1d"],
                "preferred_indicators": ["RSI", "MACD", "EMA"],
                "notification_settings": {
                    "email": False,
                    "push": False,
                    "in_app": True
                },
                "chart_settings": {
                    "theme": "light",
                    "default_timeframe": "1h",
                    "show_indicators": True
                }
            }
    
    async def update_user_preferences(
        self,
        user_id: str,
        preferences: Dict[str, Any]
    ) -> bool:
        """Update user preferences."""
        try:
            # Make API request
            response = await self.client.put(
                f"/users/{user_id}/preferences",
                json=preferences
            )
            response.raise_for_status()
            
            # Update cache
            if user_id in self.preferences_cache:
                self.preferences_cache[user_id]["preferences"].update(preferences)
                self.preferences_cache[user_id]["expiry"] = datetime.now() + timedelta(minutes=self.cache_ttl)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating user preferences: {str(e)}")
            return False
    
    async def detect_preferences_from_message(
        self,
        user_id: str,
        message: str
    ) -> Dict[str, Any]:
        """Detect user preferences from a message."""
        try:
            # Prepare request data
            request_data = {
                "user_id": user_id,
                "message": message
            }
            
            # Make API request
            response = await self.client.post("/users/detect-preferences", json=request_data)
            response.raise_for_status()
            
            # Parse response
            detected_preferences = response.json()
            
            # Update cache if preferences were detected
            if detected_preferences and user_id in self.preferences_cache:
                self.preferences_cache[user_id]["preferences"].update(detected_preferences)
            
            return detected_preferences
            
        except Exception as e:
            self.logger.error(f"Error detecting preferences from message: {str(e)}")
            return {}
