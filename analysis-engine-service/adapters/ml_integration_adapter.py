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
from common_lib.ml.interfaces import IMLModelConnector, IExplanationGenerator, IUserPreferenceManager, ModelConfiguration, ModelPrediction
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class MLModelConnectorAdapter(IMLModelConnector):
    """
    Adapter for ML model connector functionality.
    
    This adapter can either use a direct connection to the ML integration service
    or provide standalone functionality to avoid circular dependencies.
    """

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        ml_integration_base_url = self.config.get('ml_integration_base_url',
            os.environ.get('ML_INTEGRATION_BASE_URL',
            'http://ml-integration-service:8000'))
        self.client = httpx.AsyncClient(base_url=
            f"{ml_integration_base_url.rstrip('/')}/api/v1", timeout=30.0)
        self.prediction_cache = {}
        self.cache_ttl = self.config_manager.get('cache_ttl_minutes', 15)

    @with_resilience('get_market_analysis')
    @async_with_exception_handling
    async def get_market_analysis(self, symbol: str, timeframe: str,
        user_preferences: Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """Get market analysis for a symbol."""
        try:
            cache_key = f'market_analysis_{symbol}_{timeframe}'
            if cache_key in self.prediction_cache:
                cache_entry = self.prediction_cache[cache_key]
                if datetime.now() < cache_entry['expiry']:
                    return cache_entry['data']
            params = {'symbol': symbol, 'timeframe': timeframe}
            if user_preferences:
                params['user_preferences'] = json.dumps(user_preferences)
            response = await self.client.get('/ml/market-analysis', params=
                params)
            response.raise_for_status()
            analysis_data = response.json()
            self.prediction_cache[cache_key] = {'data': analysis_data,
                'expiry': datetime.now() + timedelta(minutes=self.cache_ttl)}
            return analysis_data
        except Exception as e:
            self.logger.error(f'Error getting market analysis: {str(e)}')
            return {'symbol': symbol, 'timeframe': timeframe, 'timestamp':
                datetime.now().isoformat(), 'trend': {'direction':
                'neutral', 'strength': 0.5, 'confidence': 0.5},
                'support_resistance': {'support': [], 'resistance': []},
                'indicators': {}, 'regime': 'unknown'}

    @with_analysis_resilience('get_price_prediction')
    @async_with_exception_handling
    async def get_price_prediction(self, symbol: str, timeframe: str,
        horizon: str='short_term', user_preferences: Optional[Dict[str, Any
        ]]=None) ->Dict[str, Any]:
        """Get price prediction for a symbol."""
        try:
            cache_key = f'price_prediction_{symbol}_{timeframe}_{horizon}'
            if cache_key in self.prediction_cache:
                cache_entry = self.prediction_cache[cache_key]
                if datetime.now() < cache_entry['expiry']:
                    return cache_entry['data']
            params = {'symbol': symbol, 'timeframe': timeframe, 'horizon':
                horizon}
            if user_preferences:
                params['user_preferences'] = json.dumps(user_preferences)
            response = await self.client.get('/ml/price-prediction', params
                =params)
            response.raise_for_status()
            prediction_data = response.json()
            self.prediction_cache[cache_key] = {'data': prediction_data,
                'expiry': datetime.now() + timedelta(minutes=self.cache_ttl)}
            return prediction_data
        except Exception as e:
            self.logger.error(f'Error getting price prediction: {str(e)}')
            return {'symbol': symbol, 'timeframe': timeframe, 'horizon':
                horizon, 'timestamp': datetime.now().isoformat(),
                'direction': 'neutral', 'target_price': None, 'confidence':
                0.5, 'prediction_window': '24h'}

    @with_resilience('get_trading_recommendation')
    @async_with_exception_handling
    async def get_trading_recommendation(self, symbol: str, timeframe: str,
        user_preferences: Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """Get trading recommendation for a symbol."""
        try:
            cache_key = f'trading_recommendation_{symbol}_{timeframe}'
            if cache_key in self.prediction_cache:
                cache_entry = self.prediction_cache[cache_key]
                if datetime.now() < cache_entry['expiry']:
                    return cache_entry['data']
            params = {'symbol': symbol, 'timeframe': timeframe}
            if user_preferences:
                params['user_preferences'] = json.dumps(user_preferences)
            response = await self.client.get('/ml/trading-recommendation',
                params=params)
            response.raise_for_status()
            recommendation_data = response.json()
            self.prediction_cache[cache_key] = {'data': recommendation_data,
                'expiry': datetime.now() + timedelta(minutes=self.cache_ttl)}
            return recommendation_data
        except Exception as e:
            self.logger.error(f'Error getting trading recommendation: {str(e)}'
                )
            return {'symbol': symbol, 'timeframe': timeframe, 'timestamp':
                datetime.now().isoformat(), 'action': 'hold', 'confidence':
                0.5, 'risk_level': 'medium', 'entry_price': None,
                'stop_loss': None, 'take_profit': None}

    @with_resilience('get_sentiment_analysis')
    @async_with_exception_handling
    async def get_sentiment_analysis(self, symbol: str, lookback_days: int=7
        ) ->Dict[str, Any]:
        """Get sentiment analysis for a symbol."""
        try:
            cache_key = f'sentiment_analysis_{symbol}_{lookback_days}'
            if cache_key in self.prediction_cache:
                cache_entry = self.prediction_cache[cache_key]
                if datetime.now() < cache_entry['expiry']:
                    return cache_entry['data']
            params = {'symbol': symbol, 'lookback_days': lookback_days}
            response = await self.client.get('/ml/sentiment-analysis',
                params=params)
            response.raise_for_status()
            sentiment_data = response.json()
            self.prediction_cache[cache_key] = {'data': sentiment_data,
                'expiry': datetime.now() + timedelta(minutes=self.cache_ttl)}
            return sentiment_data
        except Exception as e:
            self.logger.error(f'Error getting sentiment analysis: {str(e)}')
            return {'symbol': symbol, 'timestamp': datetime.now().isoformat
                (), 'overall_sentiment': 'neutral', 'sentiment_score': 0.5,
                'confidence': 0.5, 'sources': [], 'lookback_days':
                lookback_days}


class ExplanationGeneratorAdapter(IExplanationGenerator):
    """
    Adapter for explanation generator functionality.
    
    This adapter can either use a direct connection to the ML integration service
    or provide standalone functionality to avoid circular dependencies.
    """

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        ml_integration_base_url = self.config.get('ml_integration_base_url',
            os.environ.get('ML_INTEGRATION_BASE_URL',
            'http://ml-integration-service:8000'))
        self.client = httpx.AsyncClient(base_url=
            f"{ml_integration_base_url.rstrip('/')}/api/v1", timeout=30.0)

    @async_with_exception_handling
    async def generate_explanation(self, model_type: str, prediction: Dict[
        str, Any], inputs: Dict[str, Any], user_preferences: Optional[Dict[
        str, Any]]=None) ->Dict[str, Any]:
        """Generate explanation for a model prediction."""
        try:
            request_data = {'model_type': model_type, 'prediction':
                prediction, 'inputs': inputs}
            if user_preferences:
                request_data['user_preferences'] = user_preferences
            response = await self.client.post('/ml/explanation', json=
                request_data)
            response.raise_for_status()
            explanation_data = response.json()
            return explanation_data
        except Exception as e:
            self.logger.error(f'Error generating explanation: {str(e)}')
            return {'explanation_text': 'No explanation available.',
                'feature_importance': {}, 'confidence': prediction.get(
                'confidence', 0.5) if isinstance(prediction, dict) else 0.5,
                'model_type': model_type}

    @with_resilience('get_feature_importance')
    @async_with_exception_handling
    async def get_feature_importance(self, model_type: str, model_id: str,
        prediction: Dict[str, Any], inputs: Dict[str, Any]) ->Dict[str, float]:
        """Get feature importance for a model prediction."""
        try:
            request_data = {'model_type': model_type, 'model_id': model_id,
                'prediction': prediction, 'inputs': inputs}
            response = await self.client.post('/ml/feature-importance',
                json=request_data)
            response.raise_for_status()
            importance_data = response.json()
            return importance_data
        except Exception as e:
            self.logger.error(f'Error getting feature importance: {str(e)}')
            if isinstance(inputs, dict):
                import random
                features = list(inputs.keys())
                importance = {feature: random.random() for feature in features}
                total = sum(importance.values())
                if total > 0:
                    importance = {k: (v / total) for k, v in importance.items()
                        }
                return importance
            else:
                return {}


class UserPreferenceManagerAdapter(IUserPreferenceManager):
    """
    Adapter for user preference manager functionality.
    
    This adapter can either use a direct connection to the ML integration service
    or provide standalone functionality to avoid circular dependencies.
    """

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the adapter.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        ml_integration_base_url = self.config.get('ml_integration_base_url',
            os.environ.get('ML_INTEGRATION_BASE_URL',
            'http://ml-integration-service:8000'))
        self.client = httpx.AsyncClient(base_url=
            f"{ml_integration_base_url.rstrip('/')}/api/v1", timeout=30.0)
        self.preferences_cache = {}
        self.cache_ttl = self.config_manager.get('cache_ttl_minutes', 15)

    @with_resilience('get_user_preferences')
    @async_with_exception_handling
    async def get_user_preferences(self, user_id: str) ->Dict[str, Any]:
        """Get user preferences."""
        try:
            if user_id in self.preferences_cache:
                cache_entry = self.preferences_cache[user_id]
                if datetime.now() < cache_entry['expiry']:
                    return cache_entry['preferences']
            response = await self.client.get(f'/users/{user_id}/preferences')
            response.raise_for_status()
            preferences = response.json()
            self.preferences_cache[user_id] = {'preferences': preferences,
                'expiry': datetime.now() + timedelta(minutes=self.cache_ttl)}
            return preferences
        except Exception as e:
            self.logger.error(f'Error getting user preferences: {str(e)}')
            return {'risk_profile': 'moderate', 'preferred_timeframes': [
                '1h', '4h', '1d'], 'preferred_indicators': ['RSI', 'MACD',
                'EMA'], 'notification_settings': {'email': False, 'push': 
                False, 'in_app': True}, 'chart_settings': {'theme': 'light',
                'default_timeframe': '1h', 'show_indicators': True}}

    @with_resilience('update_user_preferences')
    @async_with_exception_handling
    async def update_user_preferences(self, user_id: str, preferences: Dict
        [str, Any]) ->bool:
        """Update user preferences."""
        try:
            response = await self.client.put(f'/users/{user_id}/preferences',
                json=preferences)
            response.raise_for_status()
            if user_id in self.preferences_cache:
                self.preferences_cache[user_id]['preferences'].update(
                    preferences)
                self.preferences_cache[user_id]['expiry'] = datetime.now(
                    ) + timedelta(minutes=self.cache_ttl)
            return True
        except Exception as e:
            self.logger.error(f'Error updating user preferences: {str(e)}')
            return False

    @async_with_exception_handling
    async def detect_preferences_from_message(self, user_id: str, message: str
        ) ->Dict[str, Any]:
        """Detect user preferences from a message."""
        try:
            request_data = {'user_id': user_id, 'message': message}
            response = await self.client.post('/users/detect-preferences',
                json=request_data)
            response.raise_for_status()
            detected_preferences = response.json()
            if detected_preferences and user_id in self.preferences_cache:
                self.preferences_cache[user_id]['preferences'].update(
                    detected_preferences)
            return detected_preferences
        except Exception as e:
            self.logger.error(
                f'Error detecting preferences from message: {str(e)}')
            return {}
