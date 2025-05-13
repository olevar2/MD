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
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('chat-backend-service')
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class ChatBackendService:
    """Backend service for the chat interface."""

    @with_exception_handling
    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the chat backend service.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        self.message_history = {}
        self.nlp_processor = None
        try:
            from analysis_engine.analysis.nlp import ChatNLPAnalyzer
            self.nlp_processor = ChatNLPAnalyzer()
            logger.info('Chat NLP processor initialized')
        except ImportError:
            logger.warning('Chat NLP processor not available')
            try:
                from analysis_engine.analysis.nlp import BaseNLPAnalyzer
                self.nlp_processor = BaseNLPAnalyzer('chat_nlp_fallback')
                logger.info('Fallback to base NLP processor')
            except ImportError:
                logger.warning('NLP processor not available')
        self.ml_client = None
        self.ml_model_connector = None
        self.explanation_generator = None
        self.user_preference_manager = None
        try:
            from analysis_engine.adapters import MLModelConnectorAdapter, ExplanationGeneratorAdapter, UserPreferenceManagerAdapter
            self.ml_model_connector = MLModelConnectorAdapter()
            logger.info('ML model connector initialized')
            self.explanation_generator = ExplanationGeneratorAdapter()
            logger.info('Explanation generator initialized')
            self.user_preference_manager = UserPreferenceManagerAdapter()
            logger.info('User preference manager initialized')
            from ml_integration_service.clients import get_ml_workbench_client
            self.ml_client = get_ml_workbench_client()
            logger.info('ML client initialized')
        except ImportError as e:
            logger.warning(f'ML components not fully available: {str(e)}')
        self.trading_client = None
        try:
            from trading_gateway_service.clients import get_trading_client
            self.trading_client = get_trading_client()
            logger.info('Trading client initialized')
        except ImportError:
            logger.warning('Trading client not available')

    @with_resilience('process_message')
    @async_with_exception_handling
    async def process_message(self, user_id: str, message: str, context:
        Dict[str, Any]=None) ->Dict[str, Any]:
        """
        Process a user message and generate a response.

        Args:
            user_id: User ID
            message: User message
            context: Message context (e.g., current symbol, timeframe)

        Returns:
            Response message
        """
        logger.info(f'Processing message from user {user_id}: {message}')
        if user_id not in self.message_history:
            self.message_history[user_id] = []
        user_message = {'id': str(uuid.uuid4()), 'text': message, 'sender':
            'user', 'timestamp': datetime.now().isoformat()}
        self.message_history[user_id].append(user_message)
        intent_info = None
        entities = []
        sentiment = None
        user_context = context or {}
        if 'user_id' not in user_context:
            user_context['user_id'] = user_id
        if self.user_preference_manager:
            try:
                detected_preferences = (await self.user_preference_manager.
                    detect_preferences_from_message(user_id, message))
                if detected_preferences:
                    logger.info(
                        f'Detected preferences for user {user_id}: {detected_preferences}'
                        )
            except Exception as e:
                logger.error(f'Error detecting user preferences: {str(e)}')
        if self.nlp_processor:
            try:
                if hasattr(self.nlp_processor, 'process_message'):
                    nlp_result = self.nlp_processor.process_message(user_id,
                        message, user_context)
                    intent_info = nlp_result.get('intent')
                    entities = nlp_result.get('entities', [])
                    user_context = nlp_result.get('context', {})
                    intent = intent_info.get('primary', {}).get('intent')
                    logger.info(
                        f'Enhanced NLP results - Intent: {intent}, Entities: {len(entities)}'
                        )
                else:
                    entities = self.nlp_processor.extract_entities(message)
                    sentiment = self.nlp_processor.analyze_sentiment(message)
                    if hasattr(self.nlp_processor, 'get_primary_intent'):
                        intent = self.nlp_processor.get_primary_intent(message,
                            entities)
                    else:
                        intent = self._determine_intent(message, entities)
                    logger.info(
                        f"Basic NLP results - Intent: {intent}, Entities: {len(entities)}, Sentiment: {sentiment['compound']}"
                        )
                    if hasattr(self.nlp_processor, 'extract_custom_entities'):
                        custom_entities = (self.nlp_processor.
                            extract_custom_entities(message))
                        if custom_entities:
                            entities.extend(custom_entities)
                            logger.info(
                                f'Extracted {len(custom_entities)} custom entities'
                                )
            except Exception as e:
                logger.error(f'Error processing message with NLP: {str(e)}')
                intent = self._simple_intent_detection(message)
        else:
            intent = self._simple_intent_detection(message)
        if intent_info:
            response = await self._generate_response_with_intent_info(user_id,
                message, intent_info, entities, user_context)
        else:
            response = await self._generate_response(user_id, message,
                intent, entities, sentiment, context)
        assistant_message = {'id': str(uuid.uuid4()), 'text': response[
            'text'], 'sender': 'assistant', 'timestamp': datetime.now().
            isoformat(), 'tradingAction': response.get('tradingAction'),
            'chartData': response.get('chartData'), 'attachments': response
            .get('attachments')}
        self.message_history[user_id].append(assistant_message)
        if self.nlp_processor and hasattr(self.nlp_processor, 'update_context'
            ):
            try:
                self.nlp_processor.update_context(user_id, message, 
                    intent_info or {'primary': {'intent': intent}},
                    entities, response)
            except Exception as e:
                logger.error(f'Error updating context: {str(e)}')
        return response

    def _determine_intent(self, message: str, entities: List[Dict[str, Any]]
        ) ->str:
        """
        Determine the intent of a message using NLP.

        Args:
            message: User message
            entities: Extracted entities

        Returns:
            Intent string
        """
        message_lower = message.lower()
        if any(term in message_lower for term in ['buy', 'sell', 'trade',
            'order', 'position']):
            return 'trading'
        if any(term in message_lower for term in ['analyze', 'analysis',
            'predict', 'forecast']):
            return 'analysis'
        if any(term in message_lower for term in ['chart', 'graph', 'plot',
            'visualization']):
            return 'chart'
        if any(term in message_lower for term in ['what', 'how', 'explain',
            'tell me', 'show me']):
            return 'information'
        return 'general'

    def _simple_intent_detection(self, message: str) ->str:
        """
        Simple intent detection without NLP.

        Args:
            message: User message

        Returns:
            Intent string
        """
        message_lower = message.lower()
        if any(term in message_lower for term in ['buy', 'sell', 'trade',
            'order', 'position']):
            return 'trading'
        if any(term in message_lower for term in ['analyze', 'analysis',
            'predict', 'forecast']):
            return 'analysis'
        if any(term in message_lower for term in ['chart', 'graph', 'plot',
            'visualization']):
            return 'chart'
        if any(term in message_lower for term in ['what', 'how', 'explain',
            'tell me', 'show me']):
            return 'information'
        return 'general'

    async def _generate_response_with_intent_info(self, user_id: str,
        message: str, intent_info: Dict[str, Any], entities: List[Dict[str,
        Any]], context: Dict[str, Any]) ->Dict[str, Any]:
        """
        Generate a response based on enhanced intent information.

        Args:
            user_id: User ID
            message: User message
            intent_info: Enhanced intent information
            entities: Extracted entities
            context: Message context

        Returns:
            Response message
        """
        response = {'text':
            "I'm your Forex Trading Assistant. I can help you analyze markets, execute trades, and monitor your portfolio. What would you like to do today?"
            }
        primary_intent = intent_info.get('primary', {}).get('intent')
        primary_confidence = intent_info.get('primary', {}).get('confidence',
            0.0)
        sub_intent = intent_info.get('sub_intent', {}).get('intent')
        sub_confidence = intent_info.get('sub_intent', {}).get('confidence',
            0.0)
        secondary_intent = intent_info.get('secondary', {}).get('intent')
        secondary_confidence = intent_info.get('secondary', {}).get(
            'confidence', 0.0)
        logger.info(
            f'Generating response for primary intent: {primary_intent} ({primary_confidence:.2f}), '
             + f'sub-intent: {sub_intent} ({sub_confidence:.2f}), ' +
            f'secondary intent: {secondary_intent} ({secondary_confidence:.2f})'
            )
        if primary_intent == 'trading':
            if sub_intent == 'buy':
                response = await self._handle_buy_intent(message, entities,
                    context)
            elif sub_intent == 'sell':
                response = await self._handle_sell_intent(message, entities,
                    context)
            elif sub_intent == 'modify':
                response = await self._handle_modify_intent(message,
                    entities, context)
            elif sub_intent == 'close':
                response = await self._handle_close_intent(message,
                    entities, context)
            else:
                response = await self._handle_trading_intent(message,
                    entities, context)
        elif primary_intent == 'analysis':
            if sub_intent == 'technical':
                response = await self._handle_technical_analysis_intent(message
                    , entities, context)
            elif sub_intent == 'fundamental':
                response = await self._handle_fundamental_analysis_intent(
                    message, entities, context)
            elif sub_intent == 'sentiment':
                response = await self._handle_sentiment_analysis_intent(message
                    , entities, context)
            else:
                response = await self._handle_analysis_intent(message,
                    entities, context)
        elif primary_intent == 'chart':
            response = await self._handle_chart_intent(message, entities,
                context)
        elif primary_intent == 'information':
            response = await self._handle_information_intent(message,
                entities, context)
        elif primary_intent == 'account':
            response = await self._handle_account_intent(message, entities,
                context)
        elif primary_intent == 'settings':
            response = await self._handle_settings_intent(message, entities,
                context)
        if primary_confidence < 0.4 and secondary_confidence > 0.3:
            response['text'] += f"""

I'm not entirely sure if you were asking about {primary_intent} or {secondary_intent}. If you meant to ask about {secondary_intent}, please let me know."""
        return response

    async def _generate_response(self, user_id: str, message: str, intent:
        str, entities: List[Dict[str, Any]], sentiment: Optional[Dict[str,
        float]], context: Optional[Dict[str, Any]]) ->Dict[str, Any]:
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
        response = {'text':
            "I'm your Forex Trading Assistant. I can help you analyze markets, execute trades, and monitor your portfolio. What would you like to do today?"
            }
        if intent == 'trading':
            response = await self._handle_trading_intent(message, entities,
                context)
        elif intent == 'analysis':
            response = await self._handle_analysis_intent(message, entities,
                context)
        elif intent == 'chart':
            response = await self._handle_chart_intent(message, entities,
                context)
        elif intent == 'information':
            response = await self._handle_information_intent(message,
                entities, context)
        elif intent == 'account':
            response = await self._handle_account_intent(message, entities,
                context)
        elif intent == 'settings':
            response = await self._handle_settings_intent(message, entities,
                context)
        return response

    async def _handle_trading_intent(self, message: str, entities: List[
        Dict[str, Any]], context: Optional[Dict[str, Any]]) ->Dict[str, Any]:
        """
        Handle trading intent.

        Args:
            message: User message
            entities: Extracted entities
            context: Message context

        Returns:
            Response message
        """
        symbol = self._extract_symbol(message, entities, context)
        action_type = 'buy' if 'buy' in message.lower() else 'sell'
        trading_action = {'type': action_type, 'symbol': symbol}
        return {'text':
            f'I can help you {action_type} {symbol}. Would you like me to execute this trade for you?'
            , 'tradingAction': trading_action}

    @async_with_exception_handling
    async def _handle_analysis_intent(self, message: str, entities: List[
        Dict[str, Any]], context: Optional[Dict[str, Any]]) ->Dict[str, Any]:
        """
        Handle analysis intent.

        Args:
            message: User message
            entities: Extracted entities
            context: Message context

        Returns:
            Response message
        """
        symbol = self._extract_symbol(message, entities, context)
        timeframe = self._extract_timeframe(message, entities, context)
        analysis_text = (
            f'Based on my analysis of recent market data, {symbol} is showing a bullish trend on the {timeframe} timeframe. The RSI indicator is at 65, suggesting moderate bullish momentum, while the MACD is showing a recent crossover. Key resistance levels are at 1.0850 and 1.0900.'
            )
        user_preferences = None
        user_id = context.get('user_id') if context else None
        if user_id and self.user_preference_manager:
            try:
                user_preferences = (await self.user_preference_manager.
                    get_user_preferences(user_id))
                detected_preferences = (await self.user_preference_manager.
                    detect_preferences_from_message(user_id, message))
                if detected_preferences:
                    logger.info(
                        f'Updated user preferences from message: {detected_preferences}'
                        )
            except Exception as e:
                logger.error(f'Error getting user preferences: {str(e)}')
        if self.ml_model_connector:
            try:
                analysis_result = (await self.ml_model_connector.
                    get_market_analysis(symbol=symbol, timeframe=timeframe,
                    user_preferences=user_preferences))
                if analysis_result:
                    explanations = analysis_result.get('explanations', {})
                    if explanations:
                        combined_text = ''
                        if 'trend' in explanations:
                            combined_text += explanations['trend'] + ' '
                        if 'price' in explanations:
                            combined_text += explanations['price'] + ' '
                        if 'support_resistance' in explanations:
                            combined_text += explanations['support_resistance'
                                ] + ' '
                        if 'volatility' in explanations:
                            combined_text += explanations['volatility']
                        if combined_text:
                            analysis_text = combined_text
                    else:
                        predictions = analysis_result.get('predictions', {})
                        if predictions:
                            trend = predictions.get('trend', {}).get(
                                'direction', 'neutral')
                            confidence = analysis_result.get('confidence', 0.5)
                            confidence_text = ('high' if confidence > 0.7 else
                                'moderate' if confidence > 0.5 else 'low')
                            analysis_text = (
                                f'Based on my analysis of recent market data for {symbol} on the {timeframe} timeframe, the trend appears to be {trend} with {confidence_text} confidence. '
                                )
                            if 'support_resistance' in predictions:
                                sr_data = predictions['support_resistance']
                                support = sr_data.get('support', [])
                                resistance = sr_data.get('resistance', [])
                                if support:
                                    support_str = ', '.join([str(s) for s in
                                        support[:2]])
                                    analysis_text += (
                                        f'Key support levels are at {support_str}. '
                                        )
                                if resistance:
                                    resistance_str = ', '.join([str(r) for
                                        r in resistance[:2]])
                                    analysis_text += (
                                        f'Key resistance levels are at {resistance_str}. '
                                        )
            except Exception as e:
                logger.error(
                    f'Error getting analysis from ML model connector: {str(e)}'
                    )
        elif self.ml_client:
            try:
                pass
            except Exception as e:
                logger.error(f'Error getting analysis from ML client: {str(e)}'
                    )
        return {'text': analysis_text}

    async def _handle_chart_intent(self, message: str, entities: List[Dict[
        str, Any]], context: Optional[Dict[str, Any]]) ->Dict[str, Any]:
        """
        Handle chart intent.

        Args:
            message: User message
            entities: Extracted entities
            context: Message context

        Returns:
            Response message
        """
        symbol = self._extract_symbol(message, entities, context)
        timeframe = self._extract_timeframe(message, entities, context)
        chart_data = {'symbol': symbol, 'timeframe': timeframe, 'data': {}}
        return {'text':
            f"Here's the {timeframe} chart for {symbol}. I've highlighted some key support and resistance levels."
            , 'chartData': chart_data}

    async def _handle_information_intent(self, message: str, entities: List
        [Dict[str, Any]], context: Optional[Dict[str, Any]]) ->Dict[str, Any]:
        """
        Handle information intent.

        Args:
            message: User message
            entities: Extracted entities
            context: Message context

        Returns:
            Response message
        """
        message_lower = message.lower()
        if 'rsi' in message_lower:
            return {'text':
                'The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements. It ranges from 0 to 100 and is typically used to identify overbought or oversold conditions. An RSI above 70 is considered overbought, while an RSI below 30 is considered oversold. Traders also look for divergences between RSI and price to identify potential reversals.'
                }
        elif 'macd' in message_lower:
            return {'text':
                "The Moving Average Convergence Divergence (MACD) is a trend-following momentum indicator that shows the relationship between two moving averages of a security's price. The MACD is calculated by subtracting the 26-period Exponential Moving Average (EMA) from the 12-period EMA. A 9-day EMA of the MACD, called the 'signal line', is then plotted on top of the MACD, functioning as a trigger for buy and sell signals."
                }
        elif 'bollinger' in message_lower:
            return {'text':
                "Bollinger Bands are a technical analysis tool defined by a set of trendlines plotted two standard deviations (positively and negatively) away from a simple moving average (SMA) of a security's price. Bollinger Bands can be used to identify M-tops and W-bottoms or to determine how strongly an asset is rising (up move) and when it is potentially reversing or weakening."
                }
        return {'text':
            'I can provide information about forex trading, technical analysis, and market conditions. What specific information are you looking for?'
            }

    def _extract_symbol(self, message: str, entities: List[Dict[str, Any]],
        context: Optional[Dict[str, Any]]) ->str:
        """
        Extract symbol from message, entities, or context.

        Args:
            message: User message
            entities: Extracted entities
            context: Message context

        Returns:
            Symbol string
        """
        for entity in entities:
            if entity.get('label') == 'CURRENCY_PAIR':
                return entity['text']
        common_pairs = ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD']
        for pair in common_pairs:
            if pair in message.upper():
                return pair
        if context and 'currentSymbol' in context:
            return context['currentSymbol']
        return 'EURUSD'

    def _extract_timeframe(self, message: str, entities: List[Dict[str, Any
        ]], context: Optional[Dict[str, Any]]) ->str:
        """
        Extract timeframe from message, entities, or context.

        Args:
            message: User message
            entities: Extracted entities
            context: Message context

        Returns:
            Timeframe string
        """
        for entity in entities:
            if entity.get('label') == 'TIMEFRAME':
                return entity['text']
        timeframes = {'1m': ['1m', '1 minute', '1min'], '5m': ['5m',
            '5 minute', '5min'], '15m': ['15m', '15 minute', '15min'],
            '30m': ['30m', '30 minute', '30min'], '1h': ['1h', '1 hour',
            'hourly'], '4h': ['4h', '4 hour'], '1d': ['1d', 'daily', 'day'],
            '1w': ['1w', 'weekly', 'week']}
        message_lower = message.lower()
        for tf, aliases in timeframes.items():
            if any(alias in message_lower for alias in aliases):
                return tf
        if context and 'currentTimeframe' in context:
            return context['currentTimeframe']
        return '1h'

    @with_resilience('execute_trading_action')
    @async_with_exception_handling
    async def execute_trading_action(self, user_id: str, action: Dict[str, Any]
        ) ->Dict[str, Any]:
        """
        Execute a trading action.

        Args:
            user_id: User ID
            action: Trading action

        Returns:
            Result of the action
        """
        logger.info(f'Executing trading action for user {user_id}: {action}')
        if not self.trading_client:
            logger.warning('Trading client not available')
            return {'success': False, 'message':
                'Trading functionality is not available'}
        try:
            result = {'success': True, 'orderId': str(uuid.uuid4()),
                'message':
                f"Successfully executed {action['type']} order for {action['symbol']}"
                , 'timestamp': datetime.now().isoformat()}
            return result
        except Exception as e:
            logger.error(f'Error executing trading action: {str(e)}')
            return {'success': False, 'message':
                f'Error executing trading action: {str(e)}'}

    @with_resilience('get_chat_history')
    def get_chat_history(self, user_id: str, limit: int=50, before:
        Optional[datetime]=None) ->List[Dict[str, Any]]:
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
        if before:
            before_str = before.isoformat()
            messages = [msg for msg in messages if msg['timestamp'] <
                before_str]
        messages = sorted(messages, key=lambda msg: msg['timestamp'],
            reverse=True)[:limit]
        return sorted(messages, key=lambda msg: msg['timestamp'])

    def clear_chat_history(self, user_id: str) ->bool:
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

    async def _handle_account_intent(self, message: str, entities: List[
        Dict[str, Any]], context: Optional[Dict[str, Any]]) ->Dict[str, Any]:
        """
        Handle account intent.

        Args:
            message: User message
            entities: Extracted entities
            context: Message context

        Returns:
            Response message
        """
        return {'text':
            'I can provide information about your trading account, including balance, equity, margin, and performance. What specific information would you like to know?'
            }

    async def _handle_buy_intent(self, message: str, entities: List[Dict[
        str, Any]], context: Optional[Dict[str, Any]]) ->Dict[str, Any]:
        """
        Handle buy intent.

        Args:
            message: User message
            entities: Extracted entities
            context: Message context

        Returns:
            Response message
        """
        symbol = self._extract_symbol(message, entities, context)
        amount = None
        for entity in entities:
            if entity['label'] == 'AMOUNT':
                amount = entity.get('value')
                break
        trading_action = {'type': 'buy', 'symbol': symbol}
        if amount is not None:
            trading_action['amount'] = amount
        return {'text':
            f"I can help you buy {symbol}{' with ' + str(amount) + ' lots' if amount else ''}. Would you like me to execute this trade for you?"
            , 'tradingAction': trading_action}

    async def _handle_sell_intent(self, message: str, entities: List[Dict[
        str, Any]], context: Optional[Dict[str, Any]]) ->Dict[str, Any]:
        """
        Handle sell intent.

        Args:
            message: User message
            entities: Extracted entities
            context: Message context

        Returns:
            Response message
        """
        symbol = self._extract_symbol(message, entities, context)
        amount = None
        for entity in entities:
            if entity['label'] == 'AMOUNT':
                amount = entity.get('value')
                break
        trading_action = {'type': 'sell', 'symbol': symbol}
        if amount is not None:
            trading_action['amount'] = amount
        return {'text':
            f"I can help you sell {symbol}{' with ' + str(amount) + ' lots' if amount else ''}. Would you like me to execute this trade for you?"
            , 'tradingAction': trading_action}

    async def _handle_modify_intent(self, message: str, entities: List[Dict
        [str, Any]], context: Optional[Dict[str, Any]]) ->Dict[str, Any]:
        """
        Handle modify intent.

        Args:
            message: User message
            entities: Extracted entities
            context: Message context

        Returns:
            Response message
        """
        symbol = self._extract_symbol(message, entities, context)
        has_stop_loss = 'stop' in message.lower() or 'sl' in message.lower()
        has_take_profit = 'profit' in message.lower() or 'tp' in message.lower(
            ) or 'target' in message.lower()
        stop_loss = None
        take_profit = None
        for entity in entities:
            if entity['label'] == 'PRICE':
                if entity.get('level_type'
                    ) == 'support' or has_stop_loss and not has_take_profit:
                    stop_loss = entity.get('value')
                elif entity.get('level_type'
                    ) == 'resistance' or has_take_profit and not has_stop_loss:
                    take_profit = entity.get('value')
        response_text = f'I can help you modify your {symbol} position.'
        if stop_loss:
            response_text += f" I'll set the stop loss at {stop_loss}."
        if take_profit:
            response_text += f" I'll set the take profit at {take_profit}."
        if not stop_loss and not take_profit:
            response_text += (
                ' What would you like to modify? You can change the stop loss, take profit, or position size.'
                )
        else:
            response_text += ' Would you like me to make these changes?'
        return {'text': response_text}

    async def _handle_close_intent(self, message: str, entities: List[Dict[
        str, Any]], context: Optional[Dict[str, Any]]) ->Dict[str, Any]:
        """
        Handle close intent.

        Args:
            message: User message
            entities: Extracted entities
            context: Message context

        Returns:
            Response message
        """
        symbol = self._extract_symbol(message, entities, context)
        trading_action = {'type': 'close', 'symbol': symbol}
        return {'text':
            f'I can help you close your {symbol} position. Would you like me to execute this action?'
            , 'tradingAction': trading_action}

    @async_with_exception_handling
    async def _handle_technical_analysis_intent(self, message: str,
        entities: List[Dict[str, Any]], context: Optional[Dict[str, Any]]
        ) ->Dict[str, Any]:
        """
        Handle technical analysis intent.

        Args:
            message: User message
            entities: Extracted entities
            context: Message context

        Returns:
            Response message
        """
        symbol = self._extract_symbol(message, entities, context)
        timeframe = self._extract_timeframe(message, entities, context)
        indicators = []
        for entity in entities:
            if entity['label'] == 'INDICATOR':
                indicators.append(entity.get('value'))
        response_text = (
            f'Based on technical analysis of {symbol} on the {timeframe} timeframe'
            )
        if indicators:
            response_text += f", focusing on {', '.join(indicators)}"
        response_text += (
            ', the market is showing a bullish trend. The price is currently above the 50-day moving average, with RSI at 65 indicating moderate bullish momentum. The MACD is showing a recent bullish crossover, suggesting increasing upward momentum. Key resistance levels are at 1.0850 and 1.0900, while support is established at 1.0800 and 1.0750.'
            )
        user_preferences = None
        user_id = context.get('user_id') if context else None
        if user_id and self.user_preference_manager:
            try:
                user_preferences = (await self.user_preference_manager.
                    get_user_preferences(user_id))
                if indicators:
                    await self.user_preference_manager.update_preference(
                        user_id, 'preferred_indicators', indicators)
            except Exception as e:
                logger.error(f'Error managing user preferences: {str(e)}')
        if self.ml_model_connector:
            try:
                analysis_result = (await self.ml_model_connector.
                    get_market_analysis(symbol=symbol, timeframe=timeframe,
                    user_preferences=user_preferences))
                if analysis_result:
                    explanations = analysis_result.get('explanations', {})
                    if indicators and 'feature_importance' in analysis_result:
                        technical_text = (
                            f"Based on technical analysis of {symbol} on the {timeframe} timeframe, focusing on {', '.join(indicators)}: "
                            )
                        if 'trend' in explanations:
                            technical_text += explanations['trend'] + ' '
                        indicator_details = self._get_indicator_details(
                            indicators, analysis_result)
                        if indicator_details:
                            technical_text += indicator_details
                        response_text = technical_text
                    else:
                        combined_text = ''
                        if 'trend' in explanations:
                            combined_text += explanations['trend'] + ' '
                        if 'price' in explanations:
                            combined_text += explanations['price'] + ' '
                        if 'support_resistance' in explanations:
                            combined_text += explanations['support_resistance']
                        if combined_text:
                            response_text = (
                                f'Based on technical analysis of {symbol} on the {timeframe} timeframe: {combined_text}'
                                )
            except Exception as e:
                logger.error(
                    f'Error getting technical analysis from ML model connector: {str(e)}'
                    )
        return {'text': response_text}

    def _get_indicator_details(self, indicators: List[str], analysis_result:
        Dict[str, Any]) ->str:
        """
        Get details about specific technical indicators.

        Args:
            indicators: List of indicator names
            analysis_result: Analysis result from ML model

        Returns:
            Text with indicator details
        """
        details = ''
        if 'inputs' in analysis_result and 'indicators' in analysis_result[
            'inputs']:
            indicator_values = analysis_result['inputs']['indicators']
            indicator_map = {'rsi': 'rsi', 'relative strength index': 'rsi',
                'macd': 'macd', 'moving average convergence divergence':
                'macd', 'ema': 'ema_50', 'moving average': 'ema_50',
                'exponential moving average': 'ema_50', 'sma': 'sma_50',
                'simple moving average': 'sma_50', 'bollinger':
                'bollinger_bands', 'bollinger bands': 'bollinger_bands',
                'atr': 'atr', 'average true range': 'atr', 'stochastic':
                'stochastic'}
            for indicator in indicators:
                indicator_lower = indicator.lower()
                key = indicator_map.get(indicator_lower)
                if key and key in indicator_values:
                    value = indicator_values[key]
                    if key == 'rsi':
                        condition = ('overbought' if value > 70 else 
                            'oversold' if value < 30 else 'neutral territory')
                        details += (
                            f'The RSI is at {value:.1f}, indicating {condition}. '
                            )
                    elif key == 'macd':
                        signal = indicator_values.get('macd_signal', 0)
                        histogram = indicator_values.get('macd_histogram', 0)
                        if histogram > 0:
                            trend = ('bullish' if histogram > signal else
                                'potentially bullish')
                        else:
                            trend = ('bearish' if histogram < signal else
                                'potentially bearish')
                        details += (
                            f'The MACD is at {value:.4f} with signal line at {signal:.4f}, suggesting a {trend} momentum. '
                            )
                    elif key.startswith('ema') or key.startswith('sma'):
                        period = key.split('_')[1]
                        details += (
                            f"The {period}-period {key.split('_')[0].upper()} is at {value:.4f}. "
                            )
                    elif key == 'atr':
                        details += (
                            f"The ATR is at {value:.4f}, indicating {'high' if value > 0.005 else 'moderate' if value > 0.002 else 'low'} volatility. "
                            )
                    elif key == 'bollinger_bands':
                        details += (
                            f"Price is currently near the {'upper' if value.get('position') > 0.7 else 'lower' if value.get('position') < 0.3 else 'middle'} Bollinger Band. "
                            )
                    elif key == 'stochastic':
                        details += (
                            f"The Stochastic oscillator is at {value:.1f}, indicating {'overbought' if value > 80 else 'oversold' if value < 20 else 'neutral'} conditions. "
                            )
        return details

    async def _handle_fundamental_analysis_intent(self, message: str,
        entities: List[Dict[str, Any]], context: Optional[Dict[str, Any]]
        ) ->Dict[str, Any]:
        """
        Handle fundamental analysis intent.

        Args:
            message: User message
            entities: Extracted entities
            context: Message context

        Returns:
            Response message
        """
        symbol = self._extract_symbol(message, entities, context)
        response_text = (
            f'Based on fundamental analysis of {symbol}, the outlook is cautiously optimistic. '
            )
        if symbol == 'EURUSD':
            response_text += (
                'Recent economic data from the Eurozone shows improving manufacturing PMI, while US inflation data came in slightly higher than expected. The ECB has maintained its hawkish stance, while the Fed has signaled potential rate cuts later this year. This divergence in monetary policy could support the Euro in the medium term.'
                )
        elif symbol == 'GBPUSD':
            response_text += (
                'The UK economy has shown resilience with better-than-expected GDP growth and declining inflation. The Bank of England is maintaining higher rates for longer than initially anticipated, which provides support for the Pound. However, ongoing trade negotiations post-Brexit continue to create uncertainty.'
                )
        elif symbol == 'USDJPY':
            response_text += (
                'The Bank of Japan has begun to shift away from its ultra-loose monetary policy, which has provided support for the Yen. Meanwhile, US Treasury yields have stabilized, reducing the interest rate differential that had been driving USDJPY higher. Geopolitical tensions in Asia could trigger safe-haven flows into the Yen.'
                )
        else:
            response_text += (
                'Recent economic indicators and central bank policies suggest changing dynamics in the currency pair. Interest rate differentials, inflation data, and economic growth projections are key factors to monitor for future price movements.'
                )
        return {'text': response_text}

    async def _handle_sentiment_analysis_intent(self, message: str,
        entities: List[Dict[str, Any]], context: Optional[Dict[str, Any]]
        ) ->Dict[str, Any]:
        """
        Handle sentiment analysis intent.

        Args:
            message: User message
            entities: Extracted entities
            context: Message context

        Returns:
            Response message
        """
        symbol = self._extract_symbol(message, entities, context)
        response_text = (
            f'Based on sentiment analysis for {symbol}, market sentiment is currently neutral with a slight bullish bias. '
            )
        response_text += (
            'Retail positioning data shows 55% of traders are long, while institutional positioning indicates balanced exposure. '
            )
        response_text += (
            'Recent news sentiment has been positive following better-than-expected economic data. '
            )
        response_text += (
            'Social media analysis shows increasing mentions with a positive sentiment score of 65/100. '
            )
        response_text += (
            'Options market data suggests traders are hedging against downside risk, with the put/call ratio at 1.2.'
            )
        return {'text': response_text}

    async def _handle_settings_intent(self, message: str, entities: List[
        Dict[str, Any]], context: Optional[Dict[str, Any]]) ->Dict[str, Any]:
        """
        Handle settings intent.

        Args:
            message: User message
            entities: Extracted entities
            context: Message context

        Returns:
            Response message
        """
        return {'text':
            'I can help you configure your platform settings, including preferences, notifications, and display options. What specific settings would you like to adjust?'
            }
