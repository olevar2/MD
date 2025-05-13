"""
Feedback Categorization System

This module implements the feedback categorization system that automatically classifies
trading feedback based on various criteria including statistical significance,
performance thresholds, and market conditions.
"""
from typing import Dict, Any, List, Optional
import logging
import numpy as np
from datetime import datetime, timedelta
from core_foundations.models.feedback import TradeFeedback, FeedbackCategory, FeedbackTag
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class FeedbackCategorizer:
    """
    The FeedbackCategorizer classifies trading feedback into appropriate categories
    based on statistical analysis, performance thresholds, and other criteria.
    
    Key capabilities:
    - Automatically categorize feedback based on predefined rules
    - Validate statistical significance of feedback
    - Apply appropriate tags based on feedback characteristics
    - Group related feedback for aggregate analysis
    """

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the FeedbackCategorizer.
        
        Args:
            config: Configuration parameters for categorization thresholds
        """
        self.config = config or {}
        self._set_default_config()
        self.historical_data = {'strategy': {}, 'model': {}, 'instrument': {}}
        logger.info('FeedbackCategorizer initialized')

    def _set_default_config(self):
        """Set default configuration parameters if not provided."""
        defaults = {'profit_threshold': 0.0, 'win_rate_threshold': 0.5,
            'min_sample_size': 10, 'confidence_level': 0.95,
            'high_impact_threshold': 0.03, 'time_window': 24 * 60 * 60,
            'anomaly_z_score': 2.0, 'parameter_impact_threshold': 0.1}
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value

    def categorize(self, feedback: TradeFeedback) ->TradeFeedback:
        """
        Categorize a feedback instance based on predefined rules and statistical analysis.
        
        Args:
            feedback: The feedback to categorize
            
        Returns:
            TradeFeedback: Categorized feedback with updated category and tags
        """
        self._store_historical_data(feedback)
        feedback = self._apply_basic_categorization(feedback)
        feedback = self._apply_statistical_validation(feedback)
        feedback = self._apply_trend_detection(feedback)
        feedback = self._apply_anomaly_detection(feedback)
        feedback.status = 'categorized'
        logger.debug(
            f'Categorized feedback (ID: {feedback.feedback_id}) as {feedback.category}'
            )
        return feedback

    def _store_historical_data(self, feedback: TradeFeedback):
        """
        Store feedback in historical data for statistical analysis.
        
        Args:
            feedback: Feedback to store
        """
        if feedback.strategy_id:
            if feedback.strategy_id not in self.historical_data['strategy']:
                self.historical_data['strategy'][feedback.strategy_id] = []
            self.historical_data['strategy'][feedback.strategy_id].append(
                feedback)
        if feedback.model_id:
            if feedback.model_id not in self.historical_data['model']:
                self.historical_data['model'][feedback.model_id] = []
            self.historical_data['model'][feedback.model_id].append(feedback)
        if feedback.instrument:
            if feedback.instrument not in self.historical_data['instrument']:
                self.historical_data['instrument'][feedback.instrument] = []
            self.historical_data['instrument'][feedback.instrument].append(
                feedback)
        max_history = self.config_manager.get('max_history_size', 1000)
        for category in self.historical_data.values():
            for key, items in category.items():
                if len(items) > max_history:
                    category[key] = sorted(items, key=lambda x: x.timestamp if
                        hasattr(x, 'timestamp') else datetime.min)[-
                        max_history:]

    def _apply_basic_categorization(self, feedback: TradeFeedback
        ) ->TradeFeedback:
        """
        Apply basic categorization rules based on outcome metrics.
        
        Args:
            feedback: Feedback to categorize
            
        Returns:
            TradeFeedback: Categorized feedback
        """
        metrics = feedback.outcome_metrics
        if feedback.source == 'strategy_execution':
            profit = metrics.get('profit_loss', metrics.get('profit', 0))
            if profit > self.config['profit_threshold']:
                feedback.category = FeedbackCategory.SUCCESS
            elif profit < self.config['profit_threshold']:
                feedback.category = FeedbackCategory.FAILURE
            else:
                feedback.category = FeedbackCategory.NEUTRAL
            if 'account_balance' in metrics and 'previous_balance' in metrics:
                pct_change = abs((metrics['account_balance'] - metrics[
                    'previous_balance']) / metrics['previous_balance'])
                if pct_change > self.config['high_impact_threshold']:
                    feedback.tags.append(FeedbackTag.HIGH_IMPACT)
        elif feedback.source == 'model_prediction':
            error = metrics.get('error', metrics.get('prediction_error', 0))
            error_threshold = metrics.get('error_threshold', self.config.
                get('prediction_error_threshold', 0.01))
            if abs(error) < error_threshold:
                feedback.category = FeedbackCategory.SUCCESS
            else:
                feedback.category = FeedbackCategory.FAILURE
            if abs(error) > error_threshold * 3:
                feedback.tags.append(FeedbackTag.REQUIRES_ATTENTION)
        elif feedback.source == 'risk_management':
            risk_breach = metrics.get('risk_breach', False)
            if risk_breach:
                feedback.category = FeedbackCategory.WARNING
                feedback.tags.append(FeedbackTag.REQUIRES_ATTENTION)
            else:
                feedback.category = FeedbackCategory.INFORMATION
        return feedback

    def _apply_statistical_validation(self, feedback: TradeFeedback
        ) ->TradeFeedback:
        """
        Apply statistical significance validation to the feedback.
        
        Args:
            feedback: Feedback to validate
            
        Returns:
            TradeFeedback: Validated feedback
        """
        if not feedback.strategy_id and not feedback.model_id:
            return feedback
        historical_data = []
        if (feedback.strategy_id and feedback.strategy_id in self.
            historical_data['strategy']):
            historical_data = self.historical_data['strategy'][feedback.
                strategy_id]
        elif feedback.model_id and feedback.model_id in self.historical_data[
            'model']:
            historical_data = self.historical_data['model'][feedback.model_id]
        if len(historical_data) < self.config['min_sample_size']:
            return feedback
        if feedback.source == 'strategy_execution':
            wins = sum(1 for f in historical_data if f.outcome_metrics.get(
                'profit_loss', f.outcome_metrics.get('profit', 0)) > 0)
            win_rate = wins / len(historical_data)
            if len(historical_data) >= self.config['min_sample_size']:
                p_null = 0.5
                z_score = (win_rate - p_null) / np.sqrt(p_null * (1 -
                    p_null) / len(historical_data))
                p_value = 2 * (1 - self._standard_normal_cdf(abs(z_score)))
                if p_value < 1 - self.config['confidence_level']:
                    feedback.tags.append(FeedbackTag.VALIDATED)
        elif feedback.source == 'model_prediction' and len(historical_data
            ) >= self.config['min_sample_size']:
            errors = [f.outcome_metrics.get('error', 0) for f in
                historical_data if 'error' in f.outcome_metrics]
            if errors:
                mean_error = np.mean(errors)
                std_error = np.std(errors) if len(errors) > 1 else 1.0
                current_error = feedback.outcome_metrics.get('error', 0)
                z_score = abs(current_error - mean_error) / (std_error if 
                    std_error > 0 else 1.0)
                if z_score > self._z_score_for_confidence(self.config[
                    'confidence_level']):
                    feedback.tags.append(FeedbackTag.VALIDATED)
        return feedback

    def _apply_trend_detection(self, feedback: TradeFeedback) ->TradeFeedback:
        """
        Apply trend detection to the feedback.
        
        Args:
            feedback: Feedback to analyze
            
        Returns:
            TradeFeedback: Analyzed feedback
        """
        if not feedback.strategy_id and not feedback.model_id:
            return feedback
        historical_data = []
        if (feedback.strategy_id and feedback.strategy_id in self.
            historical_data['strategy']):
            historical_data = self.historical_data['strategy'][feedback.
                strategy_id]
        elif feedback.model_id and feedback.model_id in self.historical_data[
            'model']:
            historical_data = self.historical_data['model'][feedback.model_id]
        if len(historical_data) < self.config['min_sample_size']:
            return feedback
        now = datetime.utcnow()
        time_threshold = now - timedelta(seconds=self.config['time_window'])
        recent_data = [f for f in historical_data if f.timestamp >
            time_threshold]
        if len(recent_data) < self.config['min_sample_size'] / 2:
            return feedback
        if feedback.source == 'strategy_execution':
            recent_wins = sum(1 for f in recent_data if f.outcome_metrics.
                get('profit_loss', f.outcome_metrics.get('profit', 0)) > 0)
            recent_win_rate = recent_wins / len(recent_data
                ) if recent_data else 0
            overall_wins = sum(1 for f in historical_data if f.
                outcome_metrics.get('profit_loss', f.outcome_metrics.get(
                'profit', 0)) > 0)
            overall_win_rate = overall_wins / len(historical_data
                ) if historical_data else 0
            if abs(recent_win_rate - overall_win_rate) > 0.15:
                feedback.tags.append(FeedbackTag.TRENDING)
        elif feedback.source == 'model_prediction':
            recent_errors = [abs(f.outcome_metrics.get('error', 0)) for f in
                recent_data if 'error' in f.outcome_metrics]
            overall_errors = [abs(f.outcome_metrics.get('error', 0)) for f in
                historical_data if 'error' in f.outcome_metrics]
            if recent_errors and overall_errors:
                recent_mean_error = np.mean(recent_errors)
                overall_mean_error = np.mean(overall_errors)
                if recent_mean_error > overall_mean_error * 1.5:
                    feedback.tags.append(FeedbackTag.TRENDING)
                    feedback.tags.append(FeedbackTag.REQUIRES_ATTENTION)
        return feedback

    def _apply_anomaly_detection(self, feedback: TradeFeedback
        ) ->TradeFeedback:
        """
        Apply anomaly detection to the feedback.
        
        Args:
            feedback: Feedback to analyze
            
        Returns:
            TradeFeedback: Analyzed feedback
        """
        if (not feedback.strategy_id and not feedback.model_id and not
            feedback.instrument):
            return feedback
        key_type = ('strategy' if feedback.strategy_id else 'model' if
            feedback.model_id else 'instrument')
        key_value = (feedback.strategy_id or feedback.model_id or feedback.
            instrument)
        if key_value not in self.historical_data[key_type]:
            return feedback
        historical_data = self.historical_data[key_type][key_value]
        if len(historical_data) < self.config['min_sample_size']:
            return feedback
        if feedback.source == 'strategy_execution':
            profits = [f.outcome_metrics.get('profit_loss', f.
                outcome_metrics.get('profit', 0)) for f in historical_data]
            if profits:
                mean_profit = np.mean(profits)
                std_profit = np.std(profits) if len(profits) > 1 else 1.0
                current_profit = feedback.outcome_metrics.get('profit_loss',
                    feedback.outcome_metrics.get('profit', 0))
                z_score = abs(current_profit - mean_profit) / (std_profit if
                    std_profit > 0 else 1.0)
                if z_score > self.config['anomaly_z_score']:
                    feedback.tags.append(FeedbackTag.ANOMALY)
                    if z_score > self.config['anomaly_z_score'] * 2:
                        feedback.tags.append(FeedbackTag.HIGH_IMPACT)
        elif feedback.source == 'model_prediction':
            errors = [abs(f.outcome_metrics.get('error', 0)) for f in
                historical_data if 'error' in f.outcome_metrics]
            if errors:
                mean_error = np.mean(errors)
                std_error = np.std(errors) if len(errors) > 1 else 1.0
                current_error = abs(feedback.outcome_metrics.get('error', 0))
                z_score = (current_error - mean_error) / (std_error if 
                    std_error > 0 else 1.0)
                if z_score > self.config['anomaly_z_score']:
                    feedback.tags.append(FeedbackTag.ANOMALY)
        return feedback

    def _standard_normal_cdf(self, x):
        """
        Compute standard normal cumulative distribution function.
        
        Args:
            x: Value to compute CDF for
            
        Returns:
            float: CDF value
        """
        return 0.5 * (1 + np.math.erf(x / np.sqrt(2)))

    def _z_score_for_confidence(self, confidence_level):
        """
        Convert a confidence level to a z-score.
        
        Args:
            confidence_level: The confidence level (e.g., 0.95)
            
        Returns:
            float: Corresponding z-score
        """
        if confidence_level >= 0.99:
            return 2.576
        elif confidence_level >= 0.975:
            return 2.326
        elif confidence_level >= 0.95:
            return 1.96
        elif confidence_level >= 0.9:
            return 1.645
        else:
            return np.sqrt(2) * np.math.erfinv(2 * confidence_level - 1)

    @with_resilience('get_historical_statistics')
    def get_historical_statistics(self, strategy_id: Optional[str]=None,
        model_id: Optional[str]=None, instrument: Optional[str]=None) ->Dict[
        str, Any]:
        """
        Get statistical summaries of historical feedback.
        
        Args:
            strategy_id: Optional filter by strategy ID
            model_id: Optional filter by model ID
            instrument: Optional filter by instrument
            
        Returns:
            Dict[str, Any]: Statistical summary
        """
        result = {'sample_count': 0, 'success_rate': 0.0, 'avg_profit': 0.0,
            'win_loss_ratio': 0.0, 'statistically_significant': False}
        historical_data = []
        if strategy_id and strategy_id in self.historical_data['strategy']:
            historical_data = self.historical_data['strategy'][strategy_id]
        elif model_id and model_id in self.historical_data['model']:
            historical_data = self.historical_data['model'][model_id]
        elif instrument and instrument in self.historical_data['instrument']:
            historical_data = self.historical_data['instrument'][instrument]
        if not historical_data:
            return result
        result['sample_count'] = len(historical_data)
        strategy_data = [f for f in historical_data if f.source ==
            'strategy_execution']
        if strategy_data:
            wins = sum(1 for f in strategy_data if f.outcome_metrics.get(
                'profit_loss', f.outcome_metrics.get('profit', 0)) > 0)
            result['success_rate'] = wins / len(strategy_data)
            profits = [f.outcome_metrics.get('profit_loss', f.
                outcome_metrics.get('profit', 0)) for f in strategy_data]
            result['avg_profit'] = sum(profits) / len(profits
                ) if profits else 0
            win_amount = sum(max(0, p) for p in profits)
            loss_amount = sum(abs(min(0, p)) for p in profits)
            result['win_loss_ratio'
                ] = win_amount / loss_amount if loss_amount > 0 else float(
                'inf')
            if len(strategy_data) >= self.config['min_sample_size']:
                p_null = 0.5
                z_score = (result['success_rate'] - p_null) / np.sqrt(
                    p_null * (1 - p_null) / len(strategy_data))
                p_value = 2 * (1 - self._standard_normal_cdf(abs(z_score)))
                result['statistically_significant'
                    ] = p_value < 1 - self.config['confidence_level']
        model_data = [f for f in historical_data if f.source ==
            'model_prediction']
        if model_data:
            result['model_stats'] = {'sample_count': len(model_data),
                'avg_error': 0.0, 'error_std': 0.0}
            errors = [f.outcome_metrics.get('error', 0) for f in model_data if
                'error' in f.outcome_metrics]
            if errors:
                result['model_stats']['avg_error'] = sum(errors) / len(errors)
                if len(errors) > 1:
                    variance = sum((e - result['model_stats']['avg_error']) **
                        2 for e in errors) / (len(errors) - 1)
                    result['model_stats']['error_std'] = np.sqrt(variance)
        return result
