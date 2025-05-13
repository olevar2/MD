"""
Feedback Loop Integration Module

Implements a feedback loop system to continuously improve causal inference
models based on trading outcomes and performance metrics.
"""
import logging
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from dataclasses import dataclass
from ..prediction.causal_predictor import CausalPredictor, CausalEnsemblePredictor
from ..integration.system_integrator import CausalSystemIntegrator
from ..graph.causal_graph_generator import CausalGraphGenerator
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

@dataclass
class FeedbackMetrics:
    """Stores feedback metrics for model performance."""
    timestamp: datetime
    prediction_accuracy: float
    causal_confidence: float
    trading_pnl: float
    model_version: str
    insights_used: List[str]
    market_conditions: Dict[str, Any]


class FeedbackLoopManager:
    """
    Manages the feedback loop for continuous improvement of causal models.
    """

    def __init__(self, feedback_window: timedelta=timedelta(days=30),
        min_feedback_samples: int=100):
        self.feedback_window = feedback_window
        self.min_feedback_samples = min_feedback_samples
        self.integrator = CausalSystemIntegrator()
        self.graph_generator = CausalGraphGenerator()
        self.feedback_history: List[FeedbackMetrics] = []
        self.model_registry = {}
        self.current_model_version = '0.1.0'

    def record_feedback(self, predictions: Dict[str, Any], outcomes: Dict[
        str, Any], market_data: pd.DataFrame) ->None:
        """
        Records feedback metrics for model improvement.
        
        Args:
            predictions: Dictionary containing model predictions and metadata
            outcomes: Dictionary containing actual trading outcomes
            market_data: Market data at the time of prediction
        """
        pred_values = np.array(predictions.get('values', []))
        actual_values = np.array(outcomes.get('values', []))
        if len(pred_values) > 0 and len(actual_values) > 0:
            accuracy = np.mean(np.abs(pred_values - actual_values))
        else:
            accuracy = 0.0
        insights_used = predictions.get('causal_insights', [])
        feedback = FeedbackMetrics(timestamp=datetime.now(),
            prediction_accuracy=accuracy, causal_confidence=predictions.get
            ('confidence', 0.0), trading_pnl=outcomes.get('pnl', 0.0),
            model_version=self.current_model_version, insights_used=[i[
            'factor'] for i in insights_used], market_conditions=self.
            _extract_market_conditions(market_data))
        self.feedback_history.append(feedback)
        if self._should_update_model():
            self._update_models()

    def _extract_market_conditions(self, market_data: pd.DataFrame) ->Dict[
        str, Any]:
        """Extracts relevant market conditions from data."""
        conditions = {'volatility': market_data.std().mean(), 'trend': self
            ._calculate_trend_strength(market_data), 'correlation_regime':
            self._identify_correlation_regime(market_data)}
        return conditions

    def _calculate_trend_strength(self, data: pd.DataFrame, window: int=20
        ) ->float:
        """Calculates the overall trend strength in the market."""
        trends = []
        for col in data.columns:
            if data[col].dtype in [np.float64, np.float32, np.int64, np.int32]:
                x = np.arange(len(data[col][-window:]))
                y = data[col][-window:].values
                slope = np.polyfit(x, y, 1)[0]
                trends.append(abs(slope))
        return np.mean(trends) if trends else 0.0

    def _identify_correlation_regime(self, data: pd.DataFrame, threshold:
        float=0.6) ->str:
        """Identifies the current correlation regime in the market."""
        if len(data.columns) < 2:
            return 'insufficient_data'
        corr_matrix = data.corr()
        avg_corr = np.mean(np.abs(corr_matrix.values[np.triu_indices_from(
            corr_matrix, k=1)]))
        if avg_corr > threshold:
            return 'high_correlation'
        elif avg_corr < -threshold:
            return 'negative_correlation'
        else:
            return 'low_correlation'

    def _should_update_model(self) ->bool:
        """Determines if model update is needed based on feedback."""
        if len(self.feedback_history) < self.min_feedback_samples:
            return False
        recent_feedback = [f for f in self.feedback_history if datetime.now
            () - f.timestamp <= self.feedback_window]
        if len(recent_feedback) < self.min_feedback_samples:
            return False
        recent_accuracy = np.mean([f.prediction_accuracy for f in
            recent_feedback[-50:]])
        overall_accuracy = np.mean([f.prediction_accuracy for f in
            recent_feedback])
        return recent_accuracy < 0.9 * overall_accuracy

    @with_exception_handling
    def _update_models(self) ->None:
        """Updates causal models based on feedback insights."""
        try:
            recent_feedback = [f for f in self.feedback_history if datetime
                .now() - f.timestamp <= self.feedback_window]
            if len(recent_feedback) < self.min_feedback_samples:
                logger.info('Insufficient feedback samples to update models')
                return
            logger.info(
                f'Updating causal models based on {len(recent_feedback)} feedback samples'
                )
            factor_performance = self._analyze_factor_performance(
                recent_feedback)
            self._update_causal_weights(factor_performance)
            if self._needs_retraining(recent_feedback):
                self._retrain_models(factor_performance)
            version_parts = self.current_model_version.split('.')
            self.current_model_version = (
                f'{version_parts[0]}.{version_parts[1]}.{int(version_parts[2]) + 1}'
                )
        except Exception as e:
            logger.error(f'Error during model update: {str(e)}', exc_info=True)

    @with_exception_handling
    def _analyze_factor_performance(self, feedback_samples: List[
        FeedbackMetrics]) ->Dict[str, Dict[str, Any]]:
        """
        Analyzes performance of different causal factors based on feedback.
        
        Args:
            feedback_samples: List of feedback metrics
            
        Returns:
            Dictionary mapping causal factors to performance metrics
        """
        all_factors = set()
        for feedback in feedback_samples:
            all_factors.update(feedback.insights_used)
        factor_performance = {}
        for factor in all_factors:
            factor_samples = [f for f in feedback_samples if factor in f.
                insights_used]
            if not factor_samples:
                continue
            accuracy = np.mean([f.prediction_accuracy for f in factor_samples])
            profit = np.mean([f.trading_pnl for f in factor_samples])
            confidence = np.mean([f.causal_confidence for f in factor_samples])
            market_conditions = {}
            for condition_type in ['volatility', 'trend', 'correlation_regime'
                ]:
                try:
                    condition_values = [f.market_conditions.get(
                        condition_type) for f in factor_samples]
                    condition_values = [v for v in condition_values if v is not
                        None]
                    if not condition_values:
                        continue
                    if isinstance(condition_values[0], (float, int)):
                        condition_arr = np.array(condition_values)
                        pnl_arr = np.array([f.trading_pnl for f in
                            factor_samples])
                        corr = np.corrcoef(condition_arr, pnl_arr)[0, 1]
                        market_conditions[condition_type] = {'correlation':
                            float(corr) if not np.isnan(corr) else 0.0,
                            'avg_value': float(np.mean(condition_values))}
                    elif isinstance(condition_values[0], str):
                        condition_groups = {}
                        for idx, cond in enumerate(condition_values):
                            if cond not in condition_groups:
                                condition_groups[cond] = []
                            condition_groups[cond].append(factor_samples[
                                idx].trading_pnl)
                        market_conditions[condition_type] = {regime: np.
                            mean(values) for regime, values in
                            condition_groups.items()}
                except Exception as e:
                    logger.warning(
                        f'Error analyzing market conditions for {condition_type}: {str(e)}'
                        )
            factor_performance[factor] = {'accuracy': float(accuracy),
                'profit': float(profit), 'confidence': float(confidence),
                'sample_size': len(factor_samples), 'market_conditions':
                market_conditions}
        return factor_performance

    @with_exception_handling
    def _update_causal_weights(self, factor_performance: Dict[str, Dict[str,
        Any]]) ->None:
        """
        Updates weights in the causal model based on feedback.
        
        Args:
            factor_performance: Dictionary mapping factors to performance metrics
        """
        weights = {}
        for factor, metrics in factor_performance.items():
            perf_weight = metrics['accuracy'] * 0.4 + metrics['profit'
                ] / 10.0 * 0.6
            if 'market_conditions' in metrics:
                for condition_type, condition_data in metrics[
                    'market_conditions'].items():
                    if isinstance(condition_data, dict
                        ) and 'correlation' in condition_data:
                        corr = condition_data['correlation']
                        perf_weight *= 1.0 + abs(corr) * 0.2
            weights[factor] = max(0.1, min(1.0, perf_weight))
        try:
            self.integrator.update_relationship_weights(weights)
            logger.info(f'Updated causal weights for {len(weights)} factors')
        except Exception as e:
            logger.error(f'Failed to update causal weights: {str(e)}')

    def _needs_retraining(self, feedback_samples: List[FeedbackMetrics]
        ) ->bool:
        """
        Determines if models need retraining based on performance degradation.
        
        Args:
            feedback_samples: List of recent feedback samples
            
        Returns:
            Boolean indicating if retraining is needed
        """
        if len(feedback_samples) < 50:
            return False
        timestamps = [f.timestamp for f in feedback_samples]
        accuracies = [f.prediction_accuracy for f in feedback_samples]
        profits = [f.trading_pnl for f in feedback_samples]
        sorted_data = sorted(zip(timestamps, accuracies, profits), key=lambda
            x: x[0])
        sorted_accuracies = [a for _, a, _ in sorted_data]
        sorted_profits = [p for _, _, p in sorted_data]
        n_halves = 2
        half_size = len(sorted_data) // n_halves
        accuracy_degradation = np.mean(sorted_accuracies[-half_size:]
            ) < 0.9 * np.mean(sorted_accuracies[:half_size])
        profit_degradation = np.mean(sorted_profits[-half_size:]
            ) < 0.7 * np.mean(sorted_profits[:half_size])
        return accuracy_degradation or profit_degradation

    @with_exception_handling
    def _retrain_models(self, factor_performance: Dict[str, Dict[str, Any]]
        ) ->None:
        """
        Retrains causal models based on feedback performance.
        
        Args:
            factor_performance: Dictionary mapping factors to performance metrics
        """
        logger.info('Retraining causal models with updated parameters')
        try:
            high_perf_factors = []
            low_perf_factors = []
            for factor, metrics in factor_performance.items():
                if metrics['accuracy'] > 0.7 and metrics['profit'] > 0:
                    high_perf_factors.append((factor, metrics))
                elif metrics['accuracy'] < 0.5 or metrics['profit'] < 0:
                    low_perf_factors.append((factor, metrics))
            self._backup_current_model()
            logger.info(
                f'Prioritizing {len(high_perf_factors)} high-performing factors'
                )
            logger.info(
                f'De-emphasizing {len(low_perf_factors)} low-performing factors'
                )
            self.model_registry[self.current_model_version] = {'timestamp':
                datetime.now(), 'high_performing_factors': [f[0] for f in
                high_perf_factors], 'low_performing_factors': [f[0] for f in
                low_perf_factors], 'performance_metrics': {f: metrics for f,
                metrics in factor_performance.items()}}
        except Exception as e:
            logger.error(f'Error during model retraining: {str(e)}',
                exc_info=True)

    def _backup_current_model(self) ->None:
        """Creates a backup of the current model."""
        if self.current_model_version in self.model_registry:
            backup_version = (
                f"{self.current_model_version}_backup_{datetime.now().strftime('%Y%m%d%H%M%S')}"
                )
            self.model_registry[backup_version] = self.model_registry[self.
                current_model_version].copy()
            logger.info(
                f'Created backup of model {self.current_model_version} as {backup_version}'
                )

    @with_resilience('get_performance_metrics')
    def get_performance_metrics(self, time_window: Optional[timedelta]=None,
        factors: Optional[List[str]]=None) ->Dict[str, Any]:
        """
        Retrieves performance metrics for causal factors.
        
        Args:
            time_window: Optional time window for analysis
            factors: Optional list of causal factors to analyze
            
        Returns:
            Dictionary containing performance metrics
        """
        if time_window is None:
            time_window = self.feedback_window
        recent_feedback = [f for f in self.feedback_history if datetime.now
            () - f.timestamp <= time_window]
        if not recent_feedback:
            return {'error': 'No feedback data available for analysis'}
        overall_metrics = {'average_accuracy': float(np.mean([f.
            prediction_accuracy for f in recent_feedback])), 'average_pnl':
            float(np.mean([f.trading_pnl for f in recent_feedback])),
            'sample_size': len(recent_feedback), 'time_period': str(
            time_window)}
        factor_analysis = self._analyze_factor_performance(recent_feedback)
        if factors:
            factor_analysis = {k: v for k, v in factor_analysis.items() if 
                k in factors}
        return {'overall': overall_metrics, 'factors': factor_analysis,
            'current_model': self.current_model_version}
