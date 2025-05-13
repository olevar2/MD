"""
Parameter Tracking Service

This module implements the tracking service for strategy parameters, providing 
statistical validation and analysis of parameter effectiveness over time.
It forms a critical component of the bidirectional feedback loop between 
strategy execution and adaptive layer.
"""
import logging
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime, timedelta
import asyncio
import json
import uuid
import numpy as np
from scipy import stats
from dataclasses import dataclass, field
from core_foundations.utils.logger import get_logger
from core_foundations.models.feedback import TradeFeedback
from core_foundations.events.event_publisher import EventPublisher
logger = get_logger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

@dataclass
class ParameterChangeRecord:
    """Record of a parameter change for tracking purposes"""
    parameter_id: str = ''
    strategy_id: str = ''
    parameter_name: str = ''
    old_value: Any = None
    new_value: Any = None
    change_reason: str = ''
    source_component: str = ''
    timestamp: str = ''
    market_conditions: Dict[str, Any] = field(default_factory=dict)
    effectiveness_metrics: Dict[str, Any] = field(default_factory=dict)
    confidence_level: float = 0.0


class ParameterTrackingService:
    """
    Tracks parameter changes and their effectiveness over time,
    providing statistical validation of parameter adjustments
    """

    def __init__(self, event_publisher: Optional[EventPublisher]=None,
        config: Dict[str, Any]=None):
        """
        Initialize the parameter tracking service
        
        Args:
            event_publisher: For publishing parameter tracking events
            config: Configuration options for the service
        """
        self.event_publisher = event_publisher
        self.config = config or self._get_default_config()
        self.parameter_changes: List[ParameterChangeRecord] = []
        self.parameter_history: Dict[str, List[Dict[str, Any]]] = {}
        self.parameter_stats: Dict[str, Dict[str, Any]] = {}
        self.recent_performance: Dict[str, List[Dict[str, Any]]] = {}
        logger.info('Parameter Tracking Service initialized')

    def _get_default_config(self) ->Dict[str, Any]:
        """Get default configuration settings"""
        return {'min_sample_size': 10, 'significance_level': 0.05,
            'sensitivity_threshold': 0.15, 'confidence_threshold': 0.7,
            'performance_window_days': 14, 'correlation_threshold': 0.3,
            'publish_events': True}

    @async_with_exception_handling
    async def record_parameter_change(self, strategy_id: str,
        parameter_name: str, old_value: Any, new_value: Any, change_reason:
        str, source_component: str, market_conditions: Optional[Dict[str,
        Any]]=None, effectiveness_metrics: Optional[Dict[str, Any]]=None,
        confidence_level: float=0.5) ->str:
        """
        Record a parameter change for tracking and analysis
        
        Args:
            strategy_id: ID of the strategy owning the parameter
            parameter_name: Name of the parameter being changed
            old_value: Previous parameter value
            new_value: New parameter value
            change_reason: Reason for the parameter change
            source_component: Component that initiated the change
            market_conditions: Current market conditions
            effectiveness_metrics: Current effectiveness metrics
            confidence_level: Confidence level for this parameter change
            
        Returns:
            str: ID of the recorded parameter change
        """
        parameter_id = str(uuid.uuid4())
        record = ParameterChangeRecord(parameter_id=parameter_id,
            strategy_id=strategy_id, parameter_name=parameter_name,
            old_value=old_value, new_value=new_value, change_reason=
            change_reason, source_component=source_component, timestamp=
            datetime.utcnow().isoformat(), market_conditions=
            market_conditions or {}, effectiveness_metrics=
            effectiveness_metrics or {}, confidence_level=confidence_level)
        self.parameter_changes.append(record)
        param_key = f'{strategy_id}:{parameter_name}'
        if param_key not in self.parameter_history:
            self.parameter_history[param_key] = []
        self.parameter_history[param_key].append({'parameter_id':
            parameter_id, 'timestamp': record.timestamp, 'old_value':
            old_value, 'new_value': new_value, 'change_reason':
            change_reason, 'source_component': source_component,
            'confidence_level': confidence_level})
        if self.event_publisher and self.config_manager.get('publish_events', True):
            try:
                await self.event_publisher.publish('parameter.changed', {
                    'parameter_id': parameter_id, 'strategy_id':
                    strategy_id, 'parameter_name': parameter_name,
                    'old_value': str(old_value), 'new_value': str(new_value
                    ), 'change_reason': change_reason, 'source_component':
                    source_component, 'timestamp': record.timestamp,
                    'confidence_level': confidence_level})
            except Exception as e:
                logger.error(
                    f'Failed to publish parameter change event: {str(e)}')
        logger.debug(
            f'Parameter change recorded: {parameter_name} from {old_value} to {new_value} for strategy {strategy_id} (confidence: {confidence_level:.2f})'
            )
        return parameter_id

    @async_with_exception_handling
    async def record_parameter_performance(self, parameter_id: str,
        strategy_id: str, parameter_name: str, performance_metrics: Dict[
        str, Any], timestamp: Optional[str]=None) ->bool:
        """
        Record performance outcome for a parameter change
        
        Args:
            parameter_id: ID of the parameter change
            strategy_id: ID of the strategy
            parameter_name: Name of the parameter
            performance_metrics: Key performance metrics
            timestamp: Timestamp of the performance measurement
            
        Returns:
            bool: Whether the performance was recorded successfully
        """
        param_key = f'{strategy_id}:{parameter_name}'
        if param_key not in self.recent_performance:
            self.recent_performance[param_key] = []
        performance_record = {'parameter_id': parameter_id, 'timestamp': 
            timestamp or datetime.utcnow().isoformat(), 'metrics':
            performance_metrics}
        self.recent_performance[param_key].append(performance_record)
        max_history = 1000
        if len(self.recent_performance[param_key]) > max_history:
            self.recent_performance[param_key] = self.recent_performance[
                param_key][-max_history:]
        if self.event_publisher and self.config_manager.get('publish_events', True):
            try:
                await self.event_publisher.publish('parameter.performance',
                    {'parameter_id': parameter_id, 'strategy_id':
                    strategy_id, 'parameter_name': parameter_name,
                    'performance_metrics': performance_metrics, 'timestamp':
                    performance_record['timestamp']})
            except Exception as e:
                logger.error(
                    f'Failed to publish parameter performance event: {str(e)}')
        return True

    @with_resilience('get_parameter_history')
    def get_parameter_history(self, strategy_id: str, parameter_name: str,
        start_date: Optional[datetime]=None, end_date: Optional[datetime]=None
        ) ->List[Dict[str, Any]]:
        """
        Get history of parameter changes
        
        Args:
            strategy_id: ID of the strategy
            parameter_name: Name of the parameter
            start_date: Optional start date filter
            end_date: Optional end date filter
            
        Returns:
            List of parameter change records
        """
        param_key = f'{strategy_id}:{parameter_name}'
        if param_key not in self.parameter_history:
            return []
        history = self.parameter_history[param_key]
        if start_date or end_date:
            filtered_history = []
            for record in history:
                record_date = datetime.fromisoformat(record['timestamp'])
                if start_date and record_date < start_date:
                    continue
                if end_date and record_date > end_date:
                    continue
                filtered_history.append(record)
            return filtered_history
        return history

    @with_analysis_resilience('calculate_parameter_effectiveness')
    async def calculate_parameter_effectiveness(self, strategy_id: str,
        parameter_name: str, lookback_days: int=0) ->Dict[str, Any]:
        """
        Calculate the effectiveness statistics for a parameter
        
        Args:
            strategy_id: ID of the strategy
            parameter_name: Name of the parameter
            lookback_days: Number of days to look back (0 = all time)
            
        Returns:
            Dictionary with effectiveness metrics
        """
        param_key = f'{strategy_id}:{parameter_name}'
        history = self.parameter_history.get(param_key, [])
        performance = self.recent_performance.get(param_key, [])
        if not history or not performance:
            return {'parameter': parameter_name, 'strategy_id': strategy_id,
                'sample_size': 0, 'effectiveness_score': 0.0,
                'statistical_significance': False, 'confidence': 0.0,
                'evaluation_timestamp': datetime.utcnow().isoformat()}
        if lookback_days > 0:
            cutoff_date = datetime.utcnow() - timedelta(days=lookback_days)
            history = [record for record in history if datetime.
                fromisoformat(record['timestamp']) >= cutoff_date]
            performance = [record for record in performance if datetime.
                fromisoformat(record['timestamp']) >= cutoff_date]
        effectiveness_score = self._calculate_effectiveness_score(history,
            performance)
        statistical_significance = self._calculate_statistical_significance(
            history, performance)
        confidence = self._calculate_confidence_level(history, performance)
        results = {'parameter': parameter_name, 'strategy_id': strategy_id,
            'sample_size': len(performance), 'effectiveness_score':
            effectiveness_score, 'statistical_significance':
            statistical_significance, 'confidence': confidence,
            'evaluation_timestamp': datetime.utcnow().isoformat()}
        self.parameter_stats[param_key] = {**results, 'last_updated':
            datetime.utcnow().isoformat()}
        return results

    def _calculate_effectiveness_score(self, history: List[Dict[str, Any]],
        performance: List[Dict[str, Any]]) ->float:
        """
        Calculate the effectiveness score for a parameter
        
        Args:
            history: Parameter change history
            performance: Performance metrics
            
        Returns:
            float: Effectiveness score (0.0 to 1.0)
        """
        if not performance:
            return 0.0
        win_rates = []
        profit_factors = []
        for record in performance:
            metrics = record.get('metrics', {})
            if 'win_rate' in metrics:
                win_rates.append(metrics['win_rate'])
            if 'profit_factor' in metrics:
                profit_factors.append(metrics['profit_factor'])
        if not win_rates and not profit_factors:
            return 0.0
        avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else 0.0
        avg_profit_factor = sum(profit_factors) / len(profit_factors
            ) if profit_factors else 0.0
        norm_win_rate = min(1.0, max(0.0, avg_win_rate / 100.0 if 
            avg_win_rate > 1.0 else avg_win_rate))
        norm_profit_factor = min(1.0, max(0.0, avg_profit_factor / 2.0))
        if win_rates and profit_factors:
            return 0.6 * norm_win_rate + 0.4 * norm_profit_factor
        elif win_rates:
            return norm_win_rate
        else:
            return norm_profit_factor

    @with_exception_handling
    def _calculate_statistical_significance(self, history: List[Dict[str,
        Any]], performance: List[Dict[str, Any]]) ->bool:
        """
        Determine if parameter changes have statistically significant effects
        
        Args:
            history: Parameter change history
            performance: Performance metrics
            
        Returns:
            bool: Whether parameter changes are statistically significant
        """
        if len(performance) < self.config['min_sample_size']:
            return False
        performance_by_value = {}
        for change in history:
            param_id = change['parameter_id']
            new_value = change['new_value']
            for perf in performance:
                if perf['parameter_id'] == param_id:
                    if new_value not in performance_by_value:
                        performance_by_value[new_value] = []
                    performance_by_value[new_value].append(perf)
        if len(performance_by_value) < 2:
            return False
        valid_groups = [values for values in performance_by_value.values() if
            len(values) >= 5]
        if len(valid_groups) < 2:
            return False
        profits_by_group = []
        for group in valid_groups:
            profits = [g['metrics'].get('profit', 0) for g in group if 
                'profit' in g['metrics']]
            if profits:
                profits_by_group.append(profits)
        if len(profits_by_group) < 2:
            return False
        try:
            f_stat, p_value = stats.f_oneway(*profits_by_group)
            return p_value < self.config['significance_level']
        except Exception as e:
            logger.warning(
                f'Error in statistical significance calculation: {str(e)}')
            return False

    def _calculate_confidence_level(self, history: List[Dict[str, Any]],
        performance: List[Dict[str, Any]]) ->float:
        """
        Calculate the confidence level for parameter effectiveness
        
        Args:
            history: Parameter change history
            performance: Performance metrics
            
        Returns:
            float: Confidence level (0.0 to 1.0)
        """
        sample_size = len(performance)
        if sample_size == 0:
            return 0.0
        if sample_size < 5:
            base_confidence = 0.1 * sample_size
        elif sample_size < 10:
            base_confidence = 0.4 + 0.04 * (sample_size - 5)
        elif sample_size < 30:
            base_confidence = 0.6 + 0.01 * (sample_size - 10)
        else:
            base_confidence = 0.8 + min(0.2, 0.005 * (sample_size - 30))
        consistency_factor = self._calculate_consistency_factor(performance)
        confidence = base_confidence * consistency_factor
        return min(1.0, confidence)

    def _calculate_consistency_factor(self, performance: List[Dict[str, Any]]
        ) ->float:
        """
        Calculate consistency factor for confidence adjustment
        
        Args:
            performance: List of performance records
            
        Returns:
            float: Consistency factor (0.0 to 1.0)
        """
        if len(performance) < 2:
            return 1.0
        profits = []
        for record in performance:
            metrics = record.get('metrics', {})
            if 'profit' in metrics:
                profits.append(metrics['profit'])
        if len(profits) < 2:
            return 1.0
        mean = np.mean(profits)
        std_dev = np.std(profits)
        if mean == 0:
            return 0.7
        cv = abs(std_dev / mean)
        if cv <= 0.5:
            return 1.0
        elif cv <= 1.0:
            return 0.9
        elif cv <= 1.5:
            return 0.8
        elif cv <= 2.0:
            return 0.7
        else:
            return 0.6

    @with_resilience('get_parameter_sensitivity_analysis')
    def get_parameter_sensitivity_analysis(self, strategy_id: str,
        parameter_name: str) ->Dict[str, Any]:
        """
        Perform sensitivity analysis for a parameter
        
        Args:
            strategy_id: Strategy ID
            parameter_name: Parameter name
            
        Returns:
            Dict with sensitivity analysis results
        """
        param_key = f'{strategy_id}:{parameter_name}'
        history = self.parameter_history.get(param_key, [])
        performance = self.recent_performance.get(param_key, [])
        if not history or not performance:
            return {'parameter': parameter_name, 'strategy_id': strategy_id,
                'sensitivity': 0.0, 'optimal_value': None,
                'value_performance_map': {}}
        value_performance = {}
        for change in history:
            param_id = change['parameter_id']
            param_value = change['new_value']
            param_value_str = str(param_value)
            if param_value_str not in value_performance:
                value_performance[param_value_str] = {'value': param_value,
                    'win_rate': 0.0, 'profit_factor': 0.0, 'avg_profit': 
                    0.0, 'count': 0}
            for perf in performance:
                if perf['parameter_id'] == param_id:
                    metrics = perf.get('metrics', {})
                    vp = value_performance[param_value_str]
                    vp['win_rate'] += metrics.get('win_rate', 0.0)
                    vp['profit_factor'] += metrics.get('profit_factor', 0.0)
                    vp['avg_profit'] += metrics.get('profit', 0.0)
                    vp['count'] += 1
        for value_str, vp in value_performance.items():
            if vp['count'] > 0:
                vp['win_rate'] /= vp['count']
                vp['profit_factor'] /= vp['count']
                vp['avg_profit'] /= vp['count']
        sensitivity, optimal_value = self._calculate_sensitivity(
            value_performance)
        return {'parameter': parameter_name, 'strategy_id': strategy_id,
            'sensitivity': sensitivity, 'optimal_value': optimal_value,
            'value_performance_map': value_performance}

    @with_exception_handling
    def _calculate_sensitivity(self, value_performance: Dict[str, Dict[str,
        Any]]) ->Tuple[float, Any]:
        """
        Calculate parameter sensitivity and optimal value
        
        Args:
            value_performance: Map of parameter values to performance metrics
            
        Returns:
            Tuple of (sensitivity score, optimal value)
        """
        if not value_performance:
            return 0.0, None
        values = []
        profits = []
        for value_str, vp in value_performance.items():
            if vp['count'] > 0:
                try:
                    value = self._convert_value(vp['value'])
                    values.append(value)
                    profits.append(vp['avg_profit'])
                except ValueError:
                    continue
        if not values or len(values) < 2:
            return 0.0, None
        best_idx = np.argmax(profits)
        optimal_value = values[best_idx]
        profit_std = np.std(profits)
        profit_mean = np.mean(profits)
        if profit_mean == 0:
            return 0.0, optimal_value
        sensitivity = min(1.0, abs(profit_std / profit_mean))
        return sensitivity, optimal_value

    @with_exception_handling
    def _convert_value(self, value: Any) ->Any:
        """Convert string value back to appropriate type"""
        if isinstance(value, (int, float, bool)):
            return value
        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower == 'true':
                return True
            elif value_lower == 'false':
                return False
            try:
                return int(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return value
        return value
