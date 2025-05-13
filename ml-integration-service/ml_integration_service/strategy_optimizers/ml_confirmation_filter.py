"""
ML Confirmation Filter

This module implements a machine learning-based confirmation filter for trading signals.
It integrates ML model predictions with strategy signals to improve entry and exit decisions.

Part of Phase 4 implementation to enhance the adaptive trading capabilities.
"""
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from ml_integration_service.clients.ml_workbench_client import MLWorkbenchClient
from ml_integration_service.models.prediction_request import PredictionRequest
from ml_integration_service.models.filter_config import FilterConfig
from ml_integration_service.utils.performance_metrics import calculate_filter_effectiveness


from ml_integration_service.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MLConfirmationFilter:
    """
    Machine learning-based confirmation filter that enhances signal quality
    by leveraging predictions from trained models to confirm or reject signals.
    """

    def __init__(self, config: Optional[Dict[str, Any]]=None):
        """
        Initialize the ML confirmation filter with configuration.
        
        Args:
            config: Configuration parameters for the filter
        """
        self.logger = logging.getLogger(__name__)
        self.ml_client = MLWorkbenchClient()
        default_config = {'min_confirmation_threshold': 0.65,
            'rejection_threshold': 0.35, 'models': {'trend_direction':
            'trend_classifier_v2', 'reversal_detection':
            'reversal_detector_v3', 'volatility_forecast':
            'volatility_predictor_v1', 'support_resistance':
            'sr_predictor_v2'}, 'use_ensemble': True, 'ensemble_weights': {
            'trend_direction': 0.4, 'reversal_detection': 0.3,
            'volatility_forecast': 0.1, 'support_resistance': 0.2},
            'cache_predictions': True, 'cache_duration_minutes': 60,
            'adapting_thresholds': True, 'model_fallback': True,
            'performance_tracking': True, 'inference_timeout_seconds': 5}
        self.config = default_config.copy()
        if config:
            self.config.update(config)
        self.prediction_cache = {}
        self.performance_history = {'total_signals': 0, 'confirmed_signals':
            0, 'rejected_signals': 0, 'confirmed_correct': 0,
            'confirmed_incorrect': 0, 'rejected_correct': 0,
            'rejected_incorrect': 0, 'by_model': {}}
        self.logger.info('ML Confirmation Filter initialized')

    @with_exception_handling
    def filter_signal(self, signal: Dict[str, Any], features: Dict[str, Any
        ], price_data: pd.DataFrame) ->Dict[str, Any]:
        """
        Filter a trading signal using ML model predictions.
        
        Args:
            signal: Original trading signal
            features: Calculated features for ML models
            price_data: Recent price data for context
            
        Returns:
            Enhanced signal with ML confirmation data
        """
        try:
            self.performance_history['total_signals'] += 1
            enhanced_signal = signal.copy()
            enhanced_signal['ml_confirmation'] = {'confirmed': False,
                'confidence': 0.0, 'model_predictions': {},
                'ensemble_score': 0.0, 'confirmation_time': datetime.now().
                isoformat()}
            predictions = self._get_model_predictions(signal, features,
                price_data)
            if not predictions:
                self.logger.warning(
                    'No model predictions available for confirmation')
                return enhanced_signal
            enhanced_signal['ml_confirmation']['model_predictions'
                ] = predictions
            if self.config['use_ensemble'] and len(predictions) > 1:
                ensemble_score = self._calculate_ensemble_score(predictions,
                    signal['direction'])
                enhanced_signal['ml_confirmation']['ensemble_score'
                    ] = ensemble_score
                confidence = ensemble_score
            else:
                main_model = self._get_primary_model_for_signal(signal)
                confidence = predictions.get(main_model, {}).get('confidence',
                    0.0)
            enhanced_signal['ml_confirmation']['confidence'] = confidence
            if confidence >= self.config['min_confirmation_threshold']:
                enhanced_signal['ml_confirmation']['confirmed'] = True
                enhanced_signal['confidence'] = min(1.0, signal.get(
                    'confidence', 0.5) * (1 + confidence / 2))
                self.performance_history['confirmed_signals'] += 1
            elif confidence <= self.config['rejection_threshold']:
                enhanced_signal['ml_confirmation']['confirmed'] = False
                enhanced_signal['confidence'] = max(0.1, signal.get(
                    'confidence', 0.5) * confidence)
                enhanced_signal['rejected'] = True
                enhanced_signal['rejection_reason'] = 'ml_confirmation_failed'
                self.performance_history['rejected_signals'] += 1
            else:
                enhanced_signal['confidence'] = signal.get('confidence', 0.5
                    ) * (0.8 + confidence / 2)
            if enhanced_signal['ml_confirmation']['confirmed']:
                self._enhance_signal_with_ml_insights(enhanced_signal,
                    predictions)
            return enhanced_signal
        except Exception as e:
            self.logger.error(
                f'Error filtering signal with ML confirmation: {str(e)}',
                exc_info=True)
            return signal

    @with_exception_handling
    def _get_model_predictions(self, signal: Dict[str, Any], features: Dict
        [str, Any], price_data: pd.DataFrame) ->Dict[str, Dict[str, Any]]:
        """Get predictions from all relevant models for this signal."""
        predictions = {}
        models_to_use = self._get_relevant_models_for_signal(signal)
        for model_type, model_name in models_to_use.items():
            cache_key = (
                f"{model_name}_{signal['symbol']}_{signal['signal_type']}")
            if self.config['cache_predictions'
                ] and cache_key in self.prediction_cache:
                cached_pred = self.prediction_cache[cache_key]
                cache_age = (datetime.now() - cached_pred['timestamp']
                    ).total_seconds() / 60
                if cache_age <= self.config['cache_duration_minutes']:
                    predictions[model_type] = cached_pred['prediction']
                    continue
            request = PredictionRequest(model_name=model_name, symbol=
                signal['symbol'], features=features, signal_type=signal[
                'signal_type'], direction=signal['direction'], timestamp=
                datetime.now().isoformat())
            try:
                prediction = self.ml_client.get_prediction(request, timeout
                    =self.config['inference_timeout_seconds'])
                if prediction and 'error' not in prediction:
                    predictions[model_type] = prediction
                    if self.config['cache_predictions']:
                        self.prediction_cache[cache_key] = {'prediction':
                            prediction, 'timestamp': datetime.now()}
                elif self.config['model_fallback']:
                    fallback_model = self._get_fallback_model(model_type)
                    if fallback_model and fallback_model != model_name:
                        self.logger.info(
                            f'Using fallback model {fallback_model} for {model_type}'
                            )
                        request.model_name = fallback_model
                        fallback_prediction = self.ml_client.get_prediction(
                            request, timeout=self.config[
                            'inference_timeout_seconds'])
                        if (fallback_prediction and 'error' not in
                            fallback_prediction):
                            fallback_prediction['is_fallback'] = True
                            predictions[model_type] = fallback_prediction
            except Exception as e:
                self.logger.error(
                    f'Error getting prediction from {model_name}: {str(e)}')
        return predictions

    def _calculate_ensemble_score(self, predictions: Dict[str, Dict[str,
        Any]], direction: str) ->float:
        """Calculate an ensemble score based on multiple model predictions."""
        total_weight = 0.0
        weighted_sum = 0.0
        for model_type, prediction in predictions.items():
            weight = self.config['ensemble_weights'].get(model_type, 0.0)
            confidence = prediction.get('confidence', 0.5)
            if direction == 'buy' and prediction.get('direction') == 'sell':
                confidence = 1.0 - confidence
            elif direction == 'sell' and prediction.get('direction') == 'buy':
                confidence = 1.0 - confidence
            weighted_sum += weight * confidence
            total_weight += weight
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return 0.5

    def _get_relevant_models_for_signal(self, signal: Dict[str, Any]) ->Dict[
        str, str]:
        """Determine which models to use based on signal type."""
        signal_type = signal.get('signal_type', '').lower()
        models = self.config['models'].copy()
        if 'breakout' in signal_type:
            return {'trend_direction': models['trend_direction'],
                'volatility_forecast': models['volatility_forecast']}
        elif 'reversal' in signal_type or 'exhaustion' in signal_type:
            return {'reversal_detection': models['reversal_detection'],
                'support_resistance': models['support_resistance']}
        elif 'harmonic' in signal_type or 'pattern' in signal_type:
            return models
        elif 'trend' in signal_type or 'momentum' in signal_type:
            return {'trend_direction': models['trend_direction'],
                'volatility_forecast': models['volatility_forecast']}
        else:
            return models

    def _get_primary_model_for_signal(self, signal: Dict[str, Any]) ->str:
        """Get the name of the primary model to use for this signal type."""
        signal_type = signal.get('signal_type', '').lower()
        if 'breakout' in signal_type:
            return 'trend_direction'
        elif 'reversal' in signal_type:
            return 'reversal_detection'
        elif 'harmonic' in signal_type or 'pattern' in signal_type:
            return 'support_resistance'
        elif 'elliott' in signal_type:
            return 'trend_direction'
        elif 'trend' in signal_type or 'momentum' in signal_type:
            return 'trend_direction'
        else:
            return 'trend_direction'

    def _get_fallback_model(self, model_type: str) ->Optional[str]:
        """Get fallback model name for a given model type."""
        fallbacks = {'trend_direction': 'legacy_trend_classifier_v1',
            'reversal_detection': 'legacy_reversal_detector_v2',
            'volatility_forecast': 'naive_volatility_predictor',
            'support_resistance': 'price_action_sr_detector'}
        return fallbacks.get(model_type)

    def _enhance_signal_with_ml_insights(self, enhanced_signal: Dict[str,
        Any], predictions: Dict[str, Dict[str, Any]]) ->None:
        """Add additional ML insights to confirmed signals."""
        if 'volatility_forecast' in predictions:
            vol_pred = predictions['volatility_forecast']
            if 'volatility_forecast_percent' in vol_pred:
                forecast_volatility = vol_pred['volatility_forecast_percent']
                current_tp = enhanced_signal.get('take_profit')
                current_sl = enhanced_signal.get('stop_loss')
                entry_price = enhanced_signal.get('entry_price')
                if current_tp and current_sl and entry_price:
                    if enhanced_signal['direction'] == 'buy':
                        risk = entry_price - current_sl
                        reward = current_tp - entry_price
                    else:
                        risk = current_sl - entry_price
                        reward = entry_price - current_tp
                    if forecast_volatility > 0 and risk > 0:
                        adjusted_reward = risk * max(1.0, 
                            forecast_volatility / 0.01)
                        if enhanced_signal['direction'] == 'buy':
                            adjusted_tp = entry_price + adjusted_reward
                        else:
                            adjusted_tp = entry_price - adjusted_reward
                        enhanced_signal['original_take_profit'] = current_tp
                        enhanced_signal['take_profit'] = adjusted_tp
                        enhanced_signal['ml_confirmation'][
                            'tp_adjustment_reason'] = 'volatility_forecast'
        if 'trend_direction' in predictions:
            trend_pred = predictions['trend_direction']
            if 'time_to_target_bars' in trend_pred:
                enhanced_signal['ml_confirmation']['estimated_bars_to_target'
                    ] = trend_pred['time_to_target_bars']
        if 'support_resistance' in predictions:
            sr_pred = predictions['support_resistance']
            if 'key_levels' in sr_pred:
                enhanced_signal['ml_confirmation']['key_levels'] = sr_pred[
                    'key_levels']
                take_profit = enhanced_signal.get('take_profit')
                if take_profit:
                    for level in sr_pred['key_levels']:
                        level_price = level.get('price')
                        level_strength = level.get('strength', 0.5)
                        if level_price and abs(level_price - take_profit
                            ) / take_profit < 0.002:
                            enhanced_signal['ml_confirmation'][
                                'target_at_key_level'] = True
                            enhanced_signal['ml_confirmation'][
                                'key_level_strength'] = level_strength
                            break

    @with_exception_handling
    def update_performance(self, signal_id: str, outcome: Dict[str, Any]
        ) ->None:
        """
        Update performance metrics based on trade outcomes.
        
        Args:
            signal_id: ID of the signal that was filtered
            outcome: Trade outcome data
        """
        if not self.config['performance_tracking']:
            return
        try:
            was_profitable = outcome.get('profitable', False)
            was_confirmed = outcome.get('ml_confirmation', {}).get('confirmed',
                False)
            if was_confirmed:
                if was_profitable:
                    self.performance_history['confirmed_correct'] += 1
                else:
                    self.performance_history['confirmed_incorrect'] += 1
            elif was_profitable:
                self.performance_history['rejected_incorrect'] += 1
            else:
                self.performance_history['rejected_correct'] += 1
            model_predictions = outcome.get('ml_confirmation', {}).get(
                'model_predictions', {})
            for model_type, prediction in model_predictions.items():
                if model_type not in self.performance_history['by_model']:
                    self.performance_history['by_model'][model_type] = {'total'
                        : 0, 'correct': 0, 'incorrect': 0}
                self.performance_history['by_model'][model_type]['total'] += 1
                model_direction = prediction.get('direction')
                signal_direction = outcome.get('direction')
                is_correct = (model_direction == signal_direction and
                    was_profitable or model_direction != signal_direction and
                    not was_profitable)
                if is_correct:
                    self.performance_history['by_model'][model_type]['correct'
                        ] += 1
                else:
                    self.performance_history['by_model'][model_type][
                        'incorrect'] += 1
            if self.config['adapting_thresholds'] and self.performance_history[
                'confirmed_signals'] + self.performance_history[
                'rejected_signals'] >= 50:
                self._adapt_thresholds()
        except Exception as e:
            self.logger.error(
                f'Error updating ML confirmation filter performance: {str(e)}',
                exc_info=True)

    @with_exception_handling
    def _adapt_thresholds(self) ->None:
        """Adapt confirmation thresholds based on historical performance."""
        try:
            confirmed_total = self.performance_history['confirmed_correct'
                ] + self.performance_history['confirmed_incorrect']
            rejected_total = self.performance_history['rejected_correct'
                ] + self.performance_history['rejected_incorrect']
            if confirmed_total == 0 or rejected_total == 0:
                return
            confirmed_accuracy = self.performance_history['confirmed_correct'
                ] / confirmed_total
            rejection_accuracy = self.performance_history['rejected_correct'
                ] / rejected_total
            if confirmed_accuracy < 0.5:
                new_threshold = min(0.9, self.config[
                    'min_confirmation_threshold'] + 0.05)
                self.logger.info(
                    f"Increasing confirmation threshold from {self.config['min_confirmation_threshold']} to {new_threshold} due to low accuracy ({confirmed_accuracy:.2%})"
                    )
                self.config['min_confirmation_threshold'] = new_threshold
            elif confirmed_accuracy > 0.75 and self.config[
                'min_confirmation_threshold'] > 0.6:
                new_threshold = max(0.6, self.config[
                    'min_confirmation_threshold'] - 0.03)
                self.logger.info(
                    f"Decreasing confirmation threshold from {self.config['min_confirmation_threshold']} to {new_threshold} due to high accuracy ({confirmed_accuracy:.2%})"
                    )
                self.config['min_confirmation_threshold'] = new_threshold
            if rejection_accuracy < 0.5:
                new_threshold = max(0.1, self.config['rejection_threshold'] -
                    0.05)
                self.logger.info(
                    f"Decreasing rejection threshold from {self.config['rejection_threshold']} to {new_threshold} due to low rejection accuracy ({rejection_accuracy:.2%})"
                    )
                self.config['rejection_threshold'] = new_threshold
            elif rejection_accuracy > 0.75 and self.config[
                'rejection_threshold'] < 0.5:
                new_threshold = min(0.5, self.config['rejection_threshold'] +
                    0.03)
                self.logger.info(
                    f"Increasing rejection threshold from {self.config['rejection_threshold']} to {new_threshold} due to high rejection accuracy ({rejection_accuracy:.2%})"
                    )
                self.config['rejection_threshold'] = new_threshold
        except Exception as e:
            self.logger.error(
                f'Error adapting ML confirmation thresholds: {str(e)}',
                exc_info=True)

    def get_performance_metrics(self) ->Dict[str, Any]:
        """Get performance metrics for the ML confirmation filter."""
        metrics = {'total_signals': self.performance_history[
            'total_signals'], 'confirmed_signals': self.performance_history
            ['confirmed_signals'], 'rejected_signals': self.
            performance_history['rejected_signals'], 'confirmation_metrics':
            {}, 'model_metrics': {}}
        confirmed_total = self.performance_history['confirmed_correct'
            ] + self.performance_history['confirmed_incorrect']
        rejected_total = self.performance_history['rejected_correct'
            ] + self.performance_history['rejected_incorrect']
        if confirmed_total > 0:
            metrics['confirmation_metrics']['confirmation_accuracy'
                ] = self.performance_history['confirmed_correct'
                ] / confirmed_total
        if rejected_total > 0:
            metrics['confirmation_metrics']['rejection_accuracy'
                ] = self.performance_history['rejected_correct'
                ] / rejected_total
        all_correct = self.performance_history['confirmed_correct'
            ] + self.performance_history['rejected_correct']
        all_total = confirmed_total + rejected_total
        if all_total > 0:
            metrics['confirmation_metrics']['overall_accuracy'
                ] = all_correct / all_total
        for model_type, model_stats in self.performance_history['by_model'
            ].items():
            if model_stats['total'] > 0:
                accuracy = model_stats['correct'] / model_stats['total']
                metrics['model_metrics'][model_type] = {'accuracy':
                    accuracy, 'total_predictions': model_stats['total']}
        return metrics
