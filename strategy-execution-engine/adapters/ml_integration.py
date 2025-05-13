"""
ML Backtester Integration Module.

This module enhances the existing backtester with the ability to
incorporate machine learning predictions into trading strategies.
"""
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional, Callable, Union
from datetime import datetime, timedelta
from core_foundations.utils.logger import get_logger
from core.backtest_engine import BacktestEngine
from adapters.ml_prediction_adapter import MLPredictionServiceAdapter
logger = get_logger('ml_backtester')


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MLBacktesterIntegration:
    """
    Integration layer between the backtester and ML prediction service.

    This class adds ML prediction capabilities to the standard backtester,
    allowing strategies to incorporate ML signals into their decision-making.
    """

    def __init__(self, backtest_engine: BacktestEngine, ml_client: Optional
        [MLPredictionServiceAdapter]=None, api_base_url: str=
        'http://localhost:8002'):
        """
        Initialize the ML backtester integration.

        Args:
            backtest_engine: Existing backtest engine instance
            ml_client: Optional ML prediction client (creates one if None)
            api_base_url: Base URL for the ML prediction API
        """
        self.backtest_engine = backtest_engine
        self.ml_client = ml_client or MLPredictionServiceAdapter(config={
            'ml_integration_base_url': api_base_url})
        self._prediction_cache = {}
        self._patch_backtest_engine()
        logger.info('ML Backtester Integration initialized')

    def _patch_backtest_engine(self):
        """
        Patch the backtest engine with ML prediction capabilities.
        This extends the original engine without modifying its source code.
        """
        self._original_init_backtest = self.backtest_engine.init_backtest
        self._original_process_bar = self.backtest_engine.process_bar
        self.backtest_engine.init_backtest = self._patched_init_backtest
        self.backtest_engine.process_bar = self._patched_process_bar
        self.backtest_engine.get_ml_prediction = self.get_prediction
        self.backtest_engine.get_ml_forecast = self.get_forecast
        logger.info('Backtest engine patched with ML prediction capabilities')

    def _patched_init_backtest(self, *args, **kwargs):
        """
        Patched initialization method that adds ML-specific setup.
        """
        result = self._original_init_backtest(*args, **kwargs)
        self._prediction_cache = {}
        logger.info('Backtest initialized with ML prediction capabilities')
        return result

    def _patched_process_bar(self, *args, **kwargs):
        """
        Patched bar processing method that clears prediction cache
        for the current bar to ensure fresh predictions.
        """
        current_bar_time = self.backtest_engine.current_bar_time
        if current_bar_time:
            cache_keys = list(self._prediction_cache.keys())
            for key in cache_keys:
                if key.startswith(f'{current_bar_time}:'):
                    del self._prediction_cache[key]
        return self._original_process_bar(*args, **kwargs)

    @async_with_exception_handling
    async def get_prediction(self, model_name: str, inputs: Dict[str, Any]=
        None, version_id: Optional[str]=None, use_cache: bool=True) ->Dict[
        str, Any]:
        """
        Get a prediction from an ML model.

        Args:
            model_name: Name of the model to use
            inputs: Input data for prediction (if None, uses current context)
            version_id: Optional specific model version to use
            use_cache: Whether to use cached predictions

        Returns:
            Dictionary containing prediction results

        Raises:
            Exception: If the prediction fails
        """
        if inputs is None:
            inputs = self._get_current_market_context()
        current_time = self.backtest_engine.current_bar_time
        cache_key = f"{current_time}:{model_name}:{version_id or 'default'}"
        if use_cache and cache_key in self._prediction_cache:
            return self._prediction_cache[cache_key]
        try:
            prediction_result = await self.ml_client.get_prediction(model_id
                =model_name, inputs=inputs, version_id=version_id)
            result = {'prediction': prediction_result.prediction,
                'confidence': prediction_result.confidence, 'model_id':
                prediction_result.model_id, 'version_id': prediction_result
                .version_id, 'timestamp': prediction_result.timestamp.
                isoformat(), 'explanation': prediction_result.explanation,
                'metadata': prediction_result.metadata or {}}
            if use_cache:
                self._prediction_cache[cache_key] = result
            return result
        except Exception as e:
            logger.error(f'Error getting ML prediction: {str(e)}')
            return {'prediction': None, 'error': str(e), 'metadata': {
                'model_name': model_name, 'version_id': version_id,
                'status': 'error'}}

    @async_with_exception_handling
    async def get_forecast(self, model_name: str, horizon: int=1,
        return_direction: bool=False, threshold: float=0.0, version_id:
        Optional[str]=None) ->Union[float, bool]:
        """
        Get a simplified forecast from an ML model.

        This is a convenience method that returns just the forecasted value
        or direction, suitable for direct use in strategy logic.

        Args:
            model_name: Name of the model to use
            horizon: How many steps ahead to forecast (1-indexed)
            return_direction: If True, returns directional signal (bool) instead of value
            threshold: Threshold for directional signal (if return_direction=True)
            version_id: Optional specific model version to use

        Returns:
            Forecasted value (float) or direction (bool)

        Raises:
            Exception: If the forecast fails
        """
        try:
            result = await self.get_prediction(model_name=model_name,
                version_id=version_id)
            if result.get('prediction') is None:
                raise ValueError(result.get('error',
                    'Unknown prediction error'))
            prediction_data = result['prediction']
            if isinstance(prediction_data, dict
                ) and 'predictions' in prediction_data:
                forecasts = prediction_data['predictions']
                if isinstance(forecasts, list) and horizon <= len(forecasts):
                    forecast_value = forecasts[horizon - 1]
                else:
                    forecast_value = forecasts
            elif isinstance(prediction_data, list) and horizon <= len(
                prediction_data):
                forecast_value = prediction_data[horizon - 1]
            else:
                forecast_value = prediction_data
            if return_direction:
                current_price = self.backtest_engine.get_current_price()
                return forecast_value - current_price > threshold
            else:
                return forecast_value
        except Exception as e:
            logger.error(f'Error getting forecast: {str(e)}')
            return 0.0 if not return_direction else False

    def _get_current_market_context(self) ->Dict[str, Any]:
        """
        Get the current market context from the backtest engine.

        Returns:
            Dictionary with market data suitable for ML model input
        """
        engine = self.backtest_engine
        symbol = engine.current_symbol
        timestamp = engine.current_bar_time
        lookback = 60
        historical_data = engine.get_historical_bars(lookback)
        indicators = {}
        if hasattr(engine, 'get_indicators'):
            indicators = engine.get_indicators()
        context = {'symbol': symbol, 'timestamp': timestamp.isoformat() if
            timestamp else None, 'historical_data': historical_data.to_dict
            (orient='records') if isinstance(historical_data, pd.DataFrame)
             else [], 'indicators': indicators}
        return context
