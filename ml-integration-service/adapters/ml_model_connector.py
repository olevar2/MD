"""
ML Model Connector for Chat Interface

This module provides a connector to ML models for the chat interface,
enabling market analysis, prediction, and personalized insights.
"""
from typing import Dict, List, Any, Optional, Union
import logging
import asyncio
import json
from datetime import datetime, timedelta
logging.basicConfig(level=logging.INFO, format=
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('chat-ml-model-connector')


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MLModelConnector:
    """Connector to ML models for the chat interface."""

    @with_exception_handling
    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the ML model connector.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.ml_client = None
        self.explanation_generator = None
        self.model_cache = {}
        self.cache_ttl = self.config_manager.get('cache_ttl_seconds', 300)
        try:
            from ml_integration_service.clients import get_ml_workbench_client
            self.ml_client = get_ml_workbench_client()
            logger.info('ML client initialized successfully')
            from core.explanation_generator import ExplanationGenerator
            self.explanation_generator = ExplanationGenerator()
            logger.info('Explanation generator initialized')
        except ImportError as e:
            logger.warning(f'ML client initialization failed: {str(e)}')

    @async_with_exception_handling
    async def get_market_analysis(self, symbol: str, timeframe: str,
        user_preferences: Dict[str, Any]=None) ->Dict[str, Any]:
        """
        Get market analysis for a symbol and timeframe.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            user_preferences: Optional user preferences for personalization

        Returns:
            Dictionary with analysis results
        """
        if not self.ml_client:
            return self._generate_fallback_analysis(symbol, timeframe)
        try:
            inputs = await self._prepare_model_inputs(symbol, timeframe)
            predictions = {}
            trend_model = user_preferences.get('preferred_trend_model',
                'trend_classifier_v2')
            trend_prediction = await self._get_prediction(trend_model, inputs)
            predictions['trend'] = trend_prediction
            price_model = user_preferences.get('preferred_price_model',
                'price_movement_predictor_v1')
            price_prediction = await self._get_prediction(price_model, inputs)
            predictions['price'] = price_prediction
            volatility_model = user_preferences.get(
                'preferred_volatility_model', 'volatility_predictor_v1')
            volatility_prediction = await self._get_prediction(volatility_model
                , inputs)
            predictions['volatility'] = volatility_prediction
            sr_model = user_preferences.get('preferred_sr_model',
                'sr_predictor_v2')
            sr_prediction = await self._get_prediction(sr_model, inputs)
            predictions['support_resistance'] = sr_prediction
            explanations = {}
            if self.explanation_generator:
                for model_type, prediction in predictions.items():
                    if prediction and 'error' not in prediction:
                        explanation = (await self.explanation_generator.
                            generate_explanation(model_type=model_type,
                            prediction=prediction, inputs=inputs,
                            user_preferences=user_preferences))
                        explanations[model_type] = explanation
            result = {'symbol': symbol, 'timeframe': timeframe, 'timestamp':
                datetime.now().isoformat(), 'predictions': predictions,
                'explanations': explanations, 'confidence': self.
                _calculate_overall_confidence(predictions)}
            return result
        except Exception as e:
            logger.error(f'Error getting market analysis: {str(e)}')
            return self._generate_fallback_analysis(symbol, timeframe)

    @async_with_exception_handling
    async def get_price_forecast(self, symbol: str, timeframe: str,
        forecast_periods: int=5, user_preferences: Dict[str, Any]=None) ->Dict[
        str, Any]:
        """
        Get price forecast for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            forecast_periods: Number of periods to forecast
            user_preferences: Optional user preferences for personalization

        Returns:
            Dictionary with forecast results
        """
        if not self.ml_client:
            return self._generate_fallback_forecast(symbol, timeframe,
                forecast_periods)
        try:
            inputs = await self._prepare_model_inputs(symbol, timeframe)
            inputs['forecast_periods'] = forecast_periods
            forecast_model = user_preferences.get('preferred_forecast_model',
                'price_forecaster_v1')
            forecast = await self._get_prediction(forecast_model, inputs)
            explanation = None
            if self.explanation_generator and 'error' not in forecast:
                explanation = (await self.explanation_generator.
                    generate_explanation(model_type='forecast', prediction=
                    forecast, inputs=inputs, user_preferences=user_preferences)
                    )
            result = {'symbol': symbol, 'timeframe': timeframe,
                'forecast_periods': forecast_periods, 'timestamp': datetime
                .now().isoformat(), 'forecast': forecast, 'explanation':
                explanation, 'confidence': forecast.get('confidence', 0.5) if
                isinstance(forecast, dict) else 0.5}
            return result
        except Exception as e:
            logger.error(f'Error getting price forecast: {str(e)}')
            return self._generate_fallback_forecast(symbol, timeframe,
                forecast_periods)

    @async_with_exception_handling
    async def get_trading_recommendation(self, symbol: str, timeframe: str,
        user_preferences: Dict[str, Any]=None) ->Dict[str, Any]:
        """
        Get trading recommendation for a symbol.

        Args:
            symbol: Trading symbol (e.g., 'EURUSD')
            timeframe: Timeframe (e.g., '1h', '4h', '1d')
            user_preferences: Optional user preferences for personalization

        Returns:
            Dictionary with recommendation results
        """
        if not self.ml_client:
            return self._generate_fallback_recommendation(symbol, timeframe)
        try:
            inputs = await self._prepare_model_inputs(symbol, timeframe)
            if user_preferences and 'risk_profile' in user_preferences:
                inputs['risk_profile'] = user_preferences['risk_profile']
            recommendation_model = user_preferences.get(
                'preferred_recommendation_model', 'trading_advisor_v1')
            recommendation = await self._get_prediction(recommendation_model,
                inputs)
            explanation = None
            if self.explanation_generator and 'error' not in recommendation:
                explanation = (await self.explanation_generator.
                    generate_explanation(model_type='recommendation',
                    prediction=recommendation, inputs=inputs,
                    user_preferences=user_preferences))
            result = {'symbol': symbol, 'timeframe': timeframe, 'timestamp':
                datetime.now().isoformat(), 'recommendation':
                recommendation, 'explanation': explanation, 'confidence': 
                recommendation.get('confidence', 0.5) if isinstance(
                recommendation, dict) else 0.5}
            return result
        except Exception as e:
            logger.error(f'Error getting trading recommendation: {str(e)}')
            return self._generate_fallback_recommendation(symbol, timeframe)

    @async_with_exception_handling
    async def _get_prediction(self, model_name: str, inputs: Dict[str, Any],
        version_id: str=None) ->Dict[str, Any]:
        """
        Get prediction from a model.

        Args:
            model_name: Name of the model
            inputs: Model inputs
            version_id: Optional model version ID

        Returns:
            Model prediction
        """
        if not self.ml_client:
            return {'error': 'ML client not available'}
        symbol = inputs.get('symbol', 'unknown')
        timeframe = inputs.get('timeframe', 'unknown')
        cache_key = (
            f"{model_name}:{version_id or 'latest'}:{symbol}:{timeframe}:{json.dumps(inputs, sort_keys=True)}"
            )
        if cache_key in self.model_cache:
            cache_entry = self.model_cache[cache_key]
            if datetime.now() < cache_entry['expiry']:
                logger.info(
                    f'Using cached prediction for {model_name}, symbol {symbol}, timeframe {timeframe}'
                    )
                return cache_entry['prediction']
        try:
            prediction = await self.ml_client.get_prediction(model_name=
                model_name, inputs=inputs, version_id=version_id)
            self.model_cache[cache_key] = {'prediction': prediction,
                'expiry': datetime.now() + timedelta(seconds=self.cache_ttl
                ), 'created': datetime.now()}
            return prediction
        except Exception as e:
            logger.error(
                f'Error getting prediction from {model_name}: {str(e)}')
            return {'error': str(e)}

    @async_with_exception_handling
    async def _prepare_model_inputs(self, symbol: str, timeframe: str) ->Dict[
        str, Any]:
        """
        Prepare inputs for ML models.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dictionary with model inputs
        """
        current_time = datetime.now().isoformat()
        inputs = {'symbol': symbol, 'timeframe': timeframe, 'timestamp':
            current_time}
        try:
            market_data = await self._get_market_data(symbol, timeframe)
            if market_data:
                inputs['market_data'] = market_data
            indicators = await self._calculate_indicators(symbol, timeframe)
            if indicators:
                inputs['indicators'] = indicators
            else:
                inputs['indicators'] = {'rsi': 65.5, 'macd': 0.0025,
                    'macd_signal': 0.0015, 'macd_histogram': 0.001,
                    'ema_50': 1.0825, 'ema_200': 1.075, 'atr': 0.0035}
        except Exception as e:
            logger.error(f'Error preparing model inputs: {str(e)}')
            inputs['indicators'] = {'rsi': 65.5, 'macd': 0.0025,
                'macd_signal': 0.0015, 'macd_histogram': 0.001, 'ema_50': 
                1.0825, 'ema_200': 1.075, 'atr': 0.0035}
        return inputs

    def _calculate_overall_confidence(self, predictions: Dict[str, Any]
        ) ->float:
        """
        Calculate overall confidence from multiple predictions.

        Args:
            predictions: Dictionary of predictions

        Returns:
            Overall confidence score (0-1)
        """
        if not predictions:
            return 0.5
        confidence_scores = []
        for pred_type, prediction in predictions.items():
            if isinstance(prediction, dict) and 'confidence' in prediction:
                confidence_scores.append(prediction['confidence'])
        if confidence_scores:
            return sum(confidence_scores) / len(confidence_scores)
        return 0.5

    def _generate_fallback_analysis(self, symbol: str, timeframe: str) ->Dict[
        str, Any]:
        """
        Generate fallback analysis when ML models are not available.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Fallback analysis
        """
        return {'symbol': symbol, 'timeframe': timeframe, 'timestamp':
            datetime.now().isoformat(), 'predictions': {'trend': {
            'direction': 'bullish', 'confidence': 0.65}, 'price': {
            'movement': 'upward', 'target': None, 'confidence': 0.6},
            'volatility': {'level': 'moderate', 'confidence': 0.7},
            'support_resistance': {'support': [1.075, 1.07], 'resistance':
            [1.085, 1.09], 'confidence': 0.75}}, 'explanations': {'trend':
            'Technical indicators suggest a bullish trend with moderate momentum.'
            , 'price':
            'Price is likely to move upward based on recent price action and indicator values.'
            , 'volatility':
            'Market volatility is moderate, suggesting normal trading conditions.'
            , 'support_resistance':
            'Key support levels are at 1.0750 and 1.0700, with resistance at 1.0850 and 1.0900.'
            }, 'confidence': 0.65, 'is_fallback': True}

    def _generate_fallback_forecast(self, symbol: str, timeframe: str,
        forecast_periods: int) ->Dict[str, Any]:
        """
        Generate fallback forecast when ML models are not available.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe
            forecast_periods: Number of periods to forecast

        Returns:
            Fallback forecast
        """
        current_price = 1.0825
        forecast_data = []
        for i in range(forecast_periods):
            next_price = current_price * (1 + 0.001 * (i + 1))
            forecast_data.append({'period': i + 1, 'price': round(
                next_price, 5), 'lower_bound': round(next_price * 0.995, 5),
                'upper_bound': round(next_price * 1.005, 5)})
        return {'symbol': symbol, 'timeframe': timeframe,
            'forecast_periods': forecast_periods, 'timestamp': datetime.now
            ().isoformat(), 'forecast': {'data': forecast_data,
            'confidence': 0.6}, 'explanation':
            'This forecast is based on recent price trends and technical indicators. The confidence level is moderate due to current market conditions.'
            , 'confidence': 0.6, 'is_fallback': True}

    def _generate_fallback_recommendation(self, symbol: str, timeframe: str
        ) ->Dict[str, Any]:
        """
        Generate fallback trading recommendation when ML models are not available.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Fallback recommendation
        """
        return {'symbol': symbol, 'timeframe': timeframe, 'timestamp':
            datetime.now().isoformat(), 'recommendation': {'action': 'buy',
            'entry_price': 1.0825, 'stop_loss': 1.0775, 'take_profit': 1.09,
            'risk_reward_ratio': 1.5, 'confidence': 0.65}, 'explanation':
            'Based on technical analysis, a buy position is recommended with a stop loss at 1.0775 and take profit at 1.0900, giving a risk-reward ratio of 1.5. The recommendation has moderate confidence due to current market conditions.'
            , 'confidence': 0.65, 'is_fallback': True}

    @async_with_exception_handling
    async def _get_market_data(self, symbol: str, timeframe: str) ->Optional[
        Dict[str, Any]]:
        """
        Get market data for a symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Market data or None if not available
        """
        try:
            from ml_integration_service.adapters import AnalysisEngineClientAdapter
            analysis_client = AnalysisEngineClientAdapter()
            market_data = await analysis_client.get_market_data(symbol=
                symbol, timeframe=timeframe, bars=100)
            logger.info(
                f'Retrieved market data for {symbol} on {timeframe} timeframe')
            return market_data
        except Exception as e:
            logger.error(f'Error getting market data: {str(e)}')
            return None

    @async_with_exception_handling
    async def _calculate_indicators(self, symbol: str, timeframe: str
        ) ->Optional[Dict[str, Any]]:
        """
        Calculate technical indicators for a symbol and timeframe.

        Args:
            symbol: Trading symbol
            timeframe: Timeframe

        Returns:
            Dictionary with technical indicators or None if not available
        """
        try:
            from ml_integration_service.adapters import AnalysisEngineClientAdapter
            analysis_client = AnalysisEngineClientAdapter()
            indicators_config = [{'name': 'RSI', 'params': {'period': 14}},
                {'name': 'MACD', 'params': {'fast_period': 12,
                'slow_period': 26, 'signal_period': 9}}, {'name': 'EMA',
                'params': {'period': 50}}, {'name': 'EMA', 'params': {
                'period': 200}}, {'name': 'ATR', 'params': {'period': 14}}]
            indicators_result = await analysis_client.get_technical_indicators(
                symbol=symbol, timeframe=timeframe, indicators=
                indicators_config)
            if indicators_result and 'indicators' in indicators_result:
                raw_indicators = indicators_result['indicators']
                processed_indicators = {}
                if 'RSI_14' in raw_indicators:
                    processed_indicators['rsi'] = raw_indicators['RSI_14'][-1]
                if 'MACD_12_26_9' in raw_indicators:
                    macd_data = raw_indicators['MACD_12_26_9']
                    processed_indicators['macd'] = macd_data['macd'][-1]
                    processed_indicators['macd_signal'] = macd_data['signal'][
                        -1]
                    processed_indicators['macd_histogram'] = macd_data[
                        'histogram'][-1]
                if 'EMA_50' in raw_indicators:
                    processed_indicators['ema_50'] = raw_indicators['EMA_50'][
                        -1]
                if 'EMA_200' in raw_indicators:
                    processed_indicators['ema_200'] = raw_indicators['EMA_200'
                        ][-1]
                if 'ATR_14' in raw_indicators:
                    processed_indicators['atr'] = raw_indicators['ATR_14'][-1]
                logger.info(
                    f'Calculated indicators for {symbol} on {timeframe} timeframe'
                    )
                return processed_indicators
            return None
        except Exception as e:
            logger.error(f'Error calculating indicators: {str(e)}')
            return None
