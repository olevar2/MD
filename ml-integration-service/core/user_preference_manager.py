"""
User Preference Manager for Chat Interface

This module manages user preferences for the chat interface,
enabling personalized responses and ML model selection.
"""
from typing import Dict, List, Any, Optional, Union
import logging
import json
import os
from datetime import datetime
import asyncio
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('chat-user-preference-manager')


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class UserPreferenceManager:
    """Manager for user preferences in the chat interface."""

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the user preference manager.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.preferences_dir = self.config.get('preferences_dir',
            './user_preferences')
        self.default_preferences = {'technical_knowledge': 'intermediate',
            'preferred_indicators': ['rsi', 'macd', 'ema'],
            'chart_timeframe': '4h', 'risk_profile': 'moderate',
            'preferred_trend_model': 'trend_classifier_v2',
            'preferred_price_model': 'price_movement_predictor_v1',
            'preferred_volatility_model': 'volatility_predictor_v1',
            'preferred_sr_model': 'sr_predictor_v2',
            'preferred_forecast_model': 'price_forecaster_v1',
            'preferred_recommendation_model': 'trading_advisor_v1',
            'notification_preferences': {'price_alerts': True,
            'trade_signals': True, 'market_news': False}, 'language_style':
            'standard', 'created_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat()}
        os.makedirs(self.preferences_dir, exist_ok=True)
        logger.info(
            f'User preference manager initialized with preferences dir: {self.preferences_dir}'
            )

    @async_with_exception_handling
    async def get_user_preferences(self, user_id: str) ->Dict[str, Any]:
        """
        Get preferences for a user.

        Args:
            user_id: User ID

        Returns:
            User preferences dictionary
        """
        preferences_file = os.path.join(self.preferences_dir, f'{user_id}.json'
            )
        if os.path.exists(preferences_file):
            try:
                with open(preferences_file, 'r') as f:
                    preferences = json.load(f)
                logger.info(f'Loaded preferences for user {user_id}')
                return preferences
            except Exception as e:
                logger.error(
                    f'Error loading preferences for user {user_id}: {str(e)}')
                return self.default_preferences.copy()
        else:
            logger.info(
                f'No preferences found for user {user_id}, using defaults')
            return self.default_preferences.copy()

    @async_with_exception_handling
    async def update_user_preferences(self, user_id: str, preferences: Dict
        [str, Any]) ->Dict[str, Any]:
        """
        Update preferences for a user.

        Args:
            user_id: User ID
            preferences: New preferences to update

        Returns:
            Updated user preferences
        """
        current_preferences = await self.get_user_preferences(user_id)
        for key, value in preferences.items():
            if isinstance(value, dict
                ) and key in current_preferences and isinstance(
                current_preferences[key], dict):
                current_preferences[key].update(value)
            else:
                current_preferences[key] = value
        current_preferences['updated_at'] = datetime.now().isoformat()
        preferences_file = os.path.join(self.preferences_dir, f'{user_id}.json'
            )
        try:
            with open(preferences_file, 'w') as f:
                json.dump(current_preferences, f, indent=2)
            logger.info(f'Updated preferences for user {user_id}')
        except Exception as e:
            logger.error(
                f'Error saving preferences for user {user_id}: {str(e)}')
        return current_preferences

    async def update_preference(self, user_id: str, preference_key: str,
        preference_value: Any) ->Dict[str, Any]:
        """
        Update a single preference for a user.

        Args:
            user_id: User ID
            preference_key: Preference key to update
            preference_value: New preference value

        Returns:
            Updated user preferences
        """
        update = {preference_key: preference_value}
        return await self.update_user_preferences(user_id, update)

    async def detect_preferences_from_message(self, user_id: str, message: str
        ) ->Optional[Dict[str, Any]]:
        """
        Detect user preferences from a message.

        Args:
            user_id: User ID
            message: User message

        Returns:
            Detected preferences or None if no preferences detected
        """
        detected_preferences = {}
        message_lower = message.lower()
        if any(term in message_lower for term in ['beginner',
            'new to trading', 'learning', 'novice', 'simple', 'basics']):
            detected_preferences['technical_knowledge'] = 'beginner'
        elif any(term in message_lower for term in ['advanced',
            'experienced', 'professional', 'expert', 'detailed', 'technical']):
            detected_preferences['technical_knowledge'] = 'advanced'
        indicators = []
        indicator_terms = {'rsi': ['rsi', 'relative strength',
            'relative strength index'], 'macd': ['macd',
            'moving average convergence', 'convergence divergence'],
            'bollinger': ['bollinger', 'bollinger bands', 'bands'], 'ema':
            ['ema', 'moving average', 'exponential moving'], 'sma': ['sma',
            'simple moving', 'simple moving average'], 'stochastic': [
            'stochastic', 'stoch'], 'atr': ['atr', 'average true range'],
            'adx': ['adx', 'directional', 'directional movement'],
            'ichimoku': ['ichimoku', 'ichimoku cloud', 'cloud'],
            'fibonacci': ['fibonacci', 'fib', 'retracement']}
        for indicator, terms in indicator_terms.items():
            if any(term in message_lower for term in terms):
                indicators.append(indicator)
        if indicators:
            detected_preferences['preferred_indicators'] = indicators
        timeframes = {'1m': ['1 minute', '1m', 'one minute', '1 min'], '5m':
            ['5 minute', '5m', 'five minute', '5 min'], '15m': ['15 minute',
            '15m', 'fifteen minute', '15 min'], '30m': ['30 minute', '30m',
            'thirty minute', '30 min'], '1h': ['1 hour', '1h', 'one hour',
            'hourly', '1 hr'], '4h': ['4 hour', '4h', 'four hour', '4 hr'],
            '1d': ['daily', '1d', 'one day', 'day', 'daily chart'], '1w': [
            'weekly', '1w', 'one week', 'week', 'weekly chart']}
        for timeframe, terms in timeframes.items():
            if any(term in message_lower for term in terms):
                detected_preferences['chart_timeframe'] = timeframe
                break
        if any(term in message_lower for term in ['conservative',
            'cautious', 'low risk', 'safe', 'careful']):
            detected_preferences['risk_profile'] = 'conservative'
        elif any(term in message_lower for term in ['moderate', 'balanced',
            'medium risk']):
            detected_preferences['risk_profile'] = 'moderate'
        elif any(term in message_lower for term in ['aggressive',
            'high risk', 'risky', 'speculative']):
            detected_preferences['risk_profile'] = 'aggressive'
        if any(term in message_lower for term in ['simple',
            'explain simply', 'easy to understand', 'beginner friendly']):
            detected_preferences['language_style'] = 'simple'
        elif any(term in message_lower for term in ['standard', 'normal',
            'regular']):
            detected_preferences['language_style'] = 'standard'
        elif any(term in message_lower for term in ['technical', 'detailed',
            'advanced explanation', 'in-depth']):
            detected_preferences['language_style'] = 'technical'
        if 'trend' in message_lower and 'model' in message_lower:
            if 'advanced' in message_lower or 'latest' in message_lower:
                detected_preferences['preferred_trend_model'
                    ] = 'trend_classifier_v3'
            elif 'simple' in message_lower or 'basic' in message_lower:
                detected_preferences['preferred_trend_model'
                    ] = 'trend_classifier_v1'
        if 'price' in message_lower and 'model' in message_lower:
            if 'advanced' in message_lower or 'latest' in message_lower:
                detected_preferences['preferred_price_model'
                    ] = 'price_movement_predictor_v2'
        preference_keywords = {'prefer': ['prefer', 'preference', 'like',
            'want'], 'use': ['use', 'using', 'utilize', 'with'], 'show': [
            'show', 'display', 'see', 'view'], 'set': ['set', 'configure',
            'change', 'update']}
        preference_actions = []
        for action, keywords in preference_keywords.items():
            if any(keyword in message_lower for keyword in keywords):
                preference_actions.append(action)
        if preference_actions and ('settings' in message_lower or 
            'preferences' in message_lower or 'profile' in message_lower):
            if 'notification' in message_lower or 'alert' in message_lower:
                if 'price' in message_lower and 'alert' in message_lower:
                    enable = not ('disable' in message_lower or 'turn off' in
                        message_lower)
                    detected_preferences['notification_preferences'] = {
                        'price_alerts': enable}
                if 'trade' in message_lower and 'signal' in message_lower:
                    enable = not ('disable' in message_lower or 'turn off' in
                        message_lower)
                    if 'notification_preferences' not in detected_preferences:
                        detected_preferences['notification_preferences'] = {}
                    detected_preferences['notification_preferences'][
                        'trade_signals'] = enable
                if 'news' in message_lower:
                    enable = not ('disable' in message_lower or 'turn off' in
                        message_lower)
                    if 'notification_preferences' not in detected_preferences:
                        detected_preferences['notification_preferences'] = {}
                    detected_preferences['notification_preferences'][
                        'market_news'] = enable
        if detected_preferences:
            logger.info(
                f'Detected preferences for user {user_id}: {detected_preferences}'
                )
            await self.update_user_preferences(user_id, detected_preferences)
            return detected_preferences
        return None

    async def get_preference(self, user_id: str, preference_key: str,
        default_value: Any=None) ->Any:
        """
        Get a specific preference for a user.

        Args:
            user_id: User ID
            preference_key: Preference key
            default_value: Default value if preference not found

        Returns:
            Preference value or default value
        """
        preferences = await self.get_user_preferences(user_id)
        if '.' in preference_key:
            keys = preference_key.split('.')
            value = preferences
            for key in keys:
                if isinstance(value, dict) and key in value:
                    value = value[key]
                else:
                    return default_value
            return value
        return preferences.get(preference_key, default_value)

    @async_with_exception_handling
    async def reset_user_preferences(self, user_id: str) ->Dict[str, Any]:
        """
        Reset preferences for a user to defaults.

        Args:
            user_id: User ID

        Returns:
            Default preferences
        """
        preferences_file = os.path.join(self.preferences_dir, f'{user_id}.json'
            )
        if os.path.exists(preferences_file):
            try:
                os.remove(preferences_file)
                logger.info(f'Reset preferences for user {user_id}')
            except Exception as e:
                logger.error(
                    f'Error resetting preferences for user {user_id}: {str(e)}'
                    )
        return self.default_preferences.copy()

    @async_with_exception_handling
    async def get_all_users(self) ->List[str]:
        """
        Get list of all users with saved preferences.

        Returns:
            List of user IDs
        """
        users = []
        try:
            for filename in os.listdir(self.preferences_dir):
                if filename.endswith('.json'):
                    user_id = filename[:-5]
                    users.append(user_id)
        except Exception as e:
            logger.error(f'Error getting user list: {str(e)}')
        return users
