"""
Statistical Validator Adapter Module

This module provides adapter implementations for statistical validation interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import asyncio
import json
import copy
from scipy import stats
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)


from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class StatisticalValidatorAdapter:
    """
    Adapter for statistical validation that implements common validation methods.
    
    This adapter can either wrap an actual service instance or provide
    standalone functionality to avoid circular dependencies.
    """

    def __init__(self, service_instance=None):
        """
        Initialize the adapter.
        
        Args:
            service_instance: Optional actual service instance to wrap
        """
        self.service = service_instance
        self.logger = logger

    @with_exception_handling
    def validate_parameter_adjustment(self, before_performance: List[Dict[
        str, Any]], after_performance: List[Dict[str, Any]], metric_name:
        str, confidence_level: float=0.95) ->Dict[str, Any]:
        """
        Validate whether a parameter adjustment led to statistically significant improvement.
        
        Args:
            before_performance: List of performance metrics before adjustment
            after_performance: List of performance metrics after adjustment
            metric_name: Name of the metric to validate
            confidence_level: Statistical confidence level
            
        Returns:
            Dictionary with validation results
        """
        if self.service:
            try:
                return self.service.validate_parameter_adjustment(
                    before_performance=before_performance,
                    after_performance=after_performance, metric_name=
                    metric_name, confidence_level=confidence_level)
            except Exception as e:
                logger.error(f'Error validating parameter adjustment: {str(e)}'
                    )
        try:
            before_values = [p.get(metric_name, 0) for p in
                before_performance if metric_name in p]
            after_values = [p.get(metric_name, 0) for p in
                after_performance if metric_name in p]
            if not before_values or not after_values:
                return {'is_valid': False, 'reason': 'Insufficient data',
                    'p_value': None, 'confidence_level': confidence_level}
            before_mean = np.mean(before_values)
            after_mean = np.mean(after_values)
            improvement = after_mean > before_mean
            t_stat, p_value = stats.ttest_ind(after_values, before_values,
                equal_var=False)
            is_significant = p_value < 1 - confidence_level
            is_valid = is_significant and improvement
            return {'is_valid': is_valid, 'is_significant': is_significant,
                'improvement': improvement, 'before_mean': before_mean,
                'after_mean': after_mean, 'change': after_mean -
                before_mean, 'pct_change': (after_mean - before_mean) /
                before_mean if before_mean != 0 else float('inf'),
                'p_value': p_value, 't_statistic': t_stat,
                'confidence_level': confidence_level, 'sample_sizes': {
                'before': len(before_values), 'after': len(after_values)}}
        except Exception as e:
            logger.error(f'Error in fallback validation: {str(e)}')
            return {'is_valid': False, 'reason':
                f'Error in validation: {str(e)}', 'p_value': None,
                'confidence_level': confidence_level}

    @with_exception_handling
    def validate_strategy_performance(self, performance_metrics: Dict[str,
        Any], baseline_metrics: Optional[Dict[str, Any]]=None, min_trades:
        int=30, confidence_level: float=0.95) ->Dict[str, Any]:
        """
        Validate whether a strategy's performance is statistically valid.
        
        Args:
            performance_metrics: Strategy performance metrics
            baseline_metrics: Optional baseline metrics for comparison
            min_trades: Minimum number of trades required for validation
            confidence_level: Statistical confidence level
            
        Returns:
            Dictionary with validation results
        """
        if self.service:
            try:
                return self.service.validate_strategy_performance(
                    performance_metrics=performance_metrics,
                    baseline_metrics=baseline_metrics, min_trades=
                    min_trades, confidence_level=confidence_level)
            except Exception as e:
                logger.error(f'Error validating strategy performance: {str(e)}'
                    )
        try:
            trade_count = performance_metrics.get('trade_count', 0)
            if trade_count < min_trades:
                return {'is_valid': False, 'reason':
                    f'Insufficient trades ({trade_count} < {min_trades})',
                    'confidence_level': confidence_level}
            win_rate = performance_metrics.get('win_rate', 0)
            profit_factor = performance_metrics.get('profit_factor', 0)
            if baseline_metrics:
                baseline_win_rate = baseline_metrics.get('win_rate', 0.5)
                baseline_profit_factor = baseline_metrics.get('profit_factor',
                    1.0)
                wins = int(win_rate * trade_count)
                p_value_win_rate = stats.binom_test(wins, trade_count,
                    baseline_win_rate)
                is_win_rate_valid = (win_rate > baseline_win_rate and 
                    p_value_win_rate < 1 - confidence_level)
                is_profit_factor_valid = profit_factor > baseline_profit_factor
                is_valid = is_win_rate_valid and is_profit_factor_valid
                return {'is_valid': is_valid, 'win_rate': {'value':
                    win_rate, 'baseline': baseline_win_rate, 'p_value':
                    p_value_win_rate, 'is_valid': is_win_rate_valid},
                    'profit_factor': {'value': profit_factor, 'baseline':
                    baseline_profit_factor, 'is_valid':
                    is_profit_factor_valid}, 'trade_count': trade_count,
                    'confidence_level': confidence_level}
            else:
                is_win_rate_valid = win_rate > 0.5
                is_profit_factor_valid = profit_factor > 1.0
                wins = int(win_rate * trade_count)
                p_value_win_rate = stats.binom_test(wins, trade_count, 0.5)
                is_valid = is_win_rate_valid and is_profit_factor_valid
                return {'is_valid': is_valid, 'win_rate': {'value':
                    win_rate, 'threshold': 0.5, 'p_value': p_value_win_rate,
                    'is_valid': is_win_rate_valid and p_value_win_rate < 1 -
                    confidence_level}, 'profit_factor': {'value':
                    profit_factor, 'threshold': 1.0, 'is_valid':
                    is_profit_factor_valid}, 'trade_count': trade_count,
                    'confidence_level': confidence_level}
        except Exception as e:
            logger.error(f'Error in fallback validation: {str(e)}')
            return {'is_valid': False, 'reason':
                f'Error in validation: {str(e)}', 'confidence_level':
                confidence_level}

    @with_exception_handling
    def validate_signal_quality(self, signals: List[Dict[str, Any]],
        outcomes: List[Dict[str, Any]], min_signals: int=20) ->Dict[str, Any]:
        """
        Validate the quality of trading signals.
        
        Args:
            signals: List of trading signals
            outcomes: List of signal outcomes
            min_signals: Minimum number of signals required for validation
            
        Returns:
            Dictionary with validation results
        """
        if self.service:
            try:
                return self.service.validate_signal_quality(signals=signals,
                    outcomes=outcomes, min_signals=min_signals)
            except Exception as e:
                logger.error(f'Error validating signal quality: {str(e)}')
        try:
            if len(signals) < min_signals or len(outcomes) < min_signals:
                return {'is_valid': False, 'reason':
                    f'Insufficient signals ({len(signals)} < {min_signals})',
                    'metrics': {}}
            matched_pairs = []
            for signal in signals:
                signal_id = signal.get('id')
                if not signal_id:
                    continue
                for outcome in outcomes:
                    if outcome.get('signal_id') == signal_id:
                        matched_pairs.append((signal, outcome))
                        break
            if len(matched_pairs) < min_signals:
                return {'is_valid': False, 'reason':
                    f'Insufficient matched signals ({len(matched_pairs)} < {min_signals})'
                    , 'metrics': {}}
            correct_count = sum(1 for s, o in matched_pairs if o.get(
                'is_correct', False))
            accuracy = correct_count / len(matched_pairs)
            buy_signals = [(s, o) for s, o in matched_pairs if s.get(
                'direction') == 'buy']
            sell_signals = [(s, o) for s, o in matched_pairs if s.get(
                'direction') == 'sell']
            buy_correct = sum(1 for s, o in buy_signals if o.get(
                'is_correct', False))
            sell_correct = sum(1 for s, o in sell_signals if o.get(
                'is_correct', False))
            buy_precision = buy_correct / len(buy_signals
                ) if buy_signals else 0
            sell_precision = sell_correct / len(sell_signals
                ) if sell_signals else 0
            profits = [o.get('profit', 0) for s, o in matched_pairs]
            avg_profit = sum(profits) / len(profits) if profits else 0
            is_valid = accuracy > 0.5 and avg_profit > 0
            return {'is_valid': is_valid, 'metrics': {'accuracy': accuracy,
                'buy_precision': buy_precision, 'sell_precision':
                sell_precision, 'avg_profit': avg_profit, 'signal_count':
                len(matched_pairs), 'buy_count': len(buy_signals),
                'sell_count': len(sell_signals)}}
        except Exception as e:
            logger.error(f'Error in fallback validation: {str(e)}')
            return {'is_valid': False, 'reason':
                f'Error in validation: {str(e)}', 'metrics': {}}
