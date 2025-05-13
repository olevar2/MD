"""
Feedback Collector for gathering execution data to improve strategy performance.

This component collects feedback on trade executions, tracking metrics like execution
quality, slippage, and P&L in various market conditions for strategy improvement.
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union
import json
from core_foundations.models.position import Position
from core_foundations.utils.logger import get_logger


from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class FeedbackCollector:
    """
    Collects feedback on trade executions for strategy improvement.
    
    This component gathers execution data to provide feedback to the adaptive layer
    and machine learning models, helping to improve strategy performance over time.
    """

    def __init__(self, config: Optional[Dict[str, Any]]=None):
        """
        Initialize the FeedbackCollector.
        
        Args:
            config: Configuration parameters
        """
        self.logger = get_logger(self.__class__.__name__)
        self.config = config or {}
        self.feedback_storage = self.config_manager.get('feedback_storage', 'memory')
        self._feedback_data = []
        self._max_memory_items = self.config_manager.get('max_memory_items', 1000)
        self.tracked_metrics = self.config.get('tracked_metrics', [
            'execution_time', 'slippage', 'partial_fills',
            'price_improvement', 'market_impact', 'pnl', 'holding_time'])
        self.ml_client = self.config_manager.get('ml_client', None)
        self.adaptive_layer_client = self.config.get('adaptive_layer_client',
            None)
        self._aggregated_metrics = {'market_regime': {}, 'instrument': {},
            'session': {}, 'strategy': {}, 'signal_type': {}}
        self.logger.info('FeedbackCollector initialized')

    @async_with_exception_handling
    async def collect_execution_feedback(self, position: Position,
        original_position: Optional[Position]=None, pnl: Optional[float]=
        None, market_conditions: Optional[Dict[str, Any]]=None) ->str:
        """
        Collect feedback on a trade execution.
        
        Args:
            position: The position that was executed
            original_position: Original position for comparison (e.g., for slippage calculation)
            pnl: Profit/loss from the trade
            market_conditions: Market conditions at execution time
            
        Returns:
            ID of the feedback record
        """
        feedback_id = str(uuid.uuid4())
        feedback = {'id': feedback_id, 'timestamp': datetime.now().
            isoformat(), 'instrument': position.symbol, 'direction':
            position.direction.value, 'size': position.size, 'entry_price':
            position.entry_price, 'exit_price': position.close_price,
            'entry_time': position.open_time.isoformat() if position.
            open_time else None, 'exit_time': position.close_time.isoformat
            () if position.close_time else None, 'pnl': pnl, 'metadata':
            position.metadata}
        if position.metadata:
            if 'strategy' in position.metadata:
                feedback['strategy'] = position.metadata['strategy']
            if 'signal_type' in position.metadata:
                feedback['signal_type'] = position.metadata['signal_type']
            if 'signal_confidence' in position.metadata:
                feedback['signal_confidence'] = position.metadata[
                    'signal_confidence']
        metrics = await self._calculate_execution_metrics(position,
            original_position)
        feedback['execution_metrics'] = metrics
        if market_conditions:
            feedback['market_conditions'] = market_conditions
        await self._store_feedback(feedback)
        await self._update_aggregated_metrics(feedback)
        if self.ml_client and hasattr(self.ml_client,
            'update_with_execution_feedback'):
            try:
                await self.ml_client.update_with_execution_feedback(feedback)
            except Exception as e:
                self.logger.warning(
                    f'Failed to update ML client with feedback: {str(e)}')
        if self.adaptive_layer_client and hasattr(self.
            adaptive_layer_client, 'update_with_execution_feedback'):
            try:
                await self.adaptive_layer_client.update_with_execution_feedback(
                    feedback)
            except Exception as e:
                self.logger.warning(
                    f'Failed to update adaptive layer with feedback: {str(e)}')
        self.logger.debug(
            f'Collected execution feedback for {position.symbol}: ID={feedback_id}'
            )
        return feedback_id

    async def get_execution_metrics(self, filter_params: Optional[Dict[str,
        Any]]=None, group_by: Optional[List[str]]=None) ->Dict[str, Any]:
        """
        Get aggregated execution metrics based on filter parameters.
        
        Args:
            filter_params: Parameters to filter the metrics by
            group_by: Fields to group the metrics by
            
        Returns:
            Dictionary with aggregated metrics
        """
        if not group_by:
            return self._get_filtered_aggregated_metrics(filter_params)
        filtered_data = await self._get_filtered_feedback_data(filter_params)
        return self._aggregate_feedback_data(filtered_data, group_by)

    async def get_raw_feedback(self, filter_params: Optional[Dict[str, Any]
        ]=None, limit: int=100, offset: int=0) ->List[Dict[str, Any]]:
        """
        Get raw feedback records based on filter parameters.
        
        Args:
            filter_params: Parameters to filter the feedback by
            limit: Maximum number of records to return
            offset: Offset for pagination
            
        Returns:
            List of feedback records
        """
        filtered_data = await self._get_filtered_feedback_data(filter_params)
        paginated_data = filtered_data[offset:offset + limit]
        return paginated_data

    async def clear_feedback_data(self) ->None:
        """Clear all feedback data."""
        self._feedback_data = []
        self._aggregated_metrics = {'market_regime': {}, 'instrument': {},
            'session': {}, 'strategy': {}, 'signal_type': {}}
        self.logger.info('Cleared all feedback data')

    async def _calculate_execution_metrics(self, position: Position,
        original_position: Optional[Position]) ->Dict[str, Any]:
        """
        Calculate execution metrics for a position.
        
        Args:
            position: The executed position
            original_position: Original position for comparison
            
        Returns:
            Dictionary with execution metrics
        """
        metrics = {}
        if position.open_time and position.close_time:
            holding_time_seconds = (position.close_time - position.open_time
                ).total_seconds()
            metrics['holding_time_seconds'] = holding_time_seconds
            metrics['holding_time_minutes'] = holding_time_seconds / 60
            metrics['holding_time_hours'] = holding_time_seconds / 3600
        if (original_position and original_position.entry_price and
            position.entry_price):
            slippage = abs(position.entry_price - original_position.entry_price
                )
            slippage_pips = slippage * 10000
            metrics['slippage_pips'] = slippage_pips
            slippage_percentage = (slippage / original_position.entry_price *
                100)
            metrics['slippage_percentage'] = slippage_percentage
            if position.direction.value == 'buy':
                is_positive = (position.entry_price < original_position.
                    entry_price)
            else:
                is_positive = (position.entry_price > original_position.
                    entry_price)
            metrics['price_improvement'] = is_positive
            if is_positive:
                metrics['price_improvement_pips'] = slippage_pips
            else:
                metrics['price_deterioration_pips'] = slippage_pips
        if position.metadata and 'execution_time_ms' in position.metadata:
            metrics['execution_time_ms'] = position.metadata[
                'execution_time_ms']
        if position.metadata and 'partial_fills' in position.metadata:
            metrics['partial_fills'] = position.metadata['partial_fills']
            metrics['partial_fill_count'] = len(position.metadata[
                'partial_fills'])
        return metrics

    @async_with_exception_handling
    async def _store_feedback(self, feedback: Dict[str, Any]) ->None:
        """
        Store feedback data.
        
        Args:
            feedback: Feedback data to store
        """
        if self.feedback_storage == 'memory':
            self._feedback_data.append(feedback)
            if len(self._feedback_data) > self._max_memory_items:
                self._feedback_data = self._feedback_data[-self.
                    _max_memory_items:]
        elif self.feedback_storage == 'file':
            file_path = self.config.get('feedback_file_path',
                'execution_feedback.jsonl')
            try:
                with open(file_path, 'a') as f:
                    f.write(json.dumps(feedback) + '\n')
            except Exception as e:
                self.logger.error(f'Failed to write feedback to file: {str(e)}'
                    )
                self._feedback_data.append(feedback)
        elif self.feedback_storage == 'database':
            db_client = self.config_manager.get('db_client', None)
            if db_client and hasattr(db_client, 'store_execution_feedback'):
                try:
                    await db_client.store_execution_feedback(feedback)
                except Exception as e:
                    self.logger.error(
                        f'Failed to write feedback to database: {str(e)}')
                    self._feedback_data.append(feedback)
            else:
                self._feedback_data.append(feedback)

    async def _get_filtered_feedback_data(self, filter_params: Optional[
        Dict[str, Any]]=None) ->List[Dict[str, Any]]:
        """
        Get feedback data filtered by parameters.
        
        Args:
            filter_params: Parameters to filter by
            
        Returns:
            Filtered feedback data
        """
        if not filter_params:
            return self._feedback_data.copy()
        filtered_data = []
        for feedback in self._feedback_data:
            matches = True
            for key, value in filter_params.items():
                if '.' in key:
                    parts = key.split('.')
                    current = feedback
                    for part in parts:
                        if part in current:
                            current = current[part]
                        else:
                            matches = False
                            break
                    if matches and current != value:
                        matches = False
                elif key in feedback:
                    if feedback[key] != value:
                        matches = False
                else:
                    matches = False
            if matches:
                filtered_data.append(feedback)
        return filtered_data

    async def _update_aggregated_metrics(self, feedback: Dict[str, Any]
        ) ->None:
        """
        Update aggregated metrics with new feedback.
        
        Args:
            feedback: New feedback data
        """
        instrument = feedback.get('instrument', 'unknown')
        strategy = feedback.get('strategy', 'unknown')
        signal_type = feedback.get('signal_type', 'unknown')
        market_regime = feedback.get('market_conditions', {}).get('regime',
            'unknown')
        session = feedback.get('market_conditions', {}).get('session',
            'unknown')
        self._update_dimension_metrics(self._aggregated_metrics[
            'instrument'], instrument, feedback)
        self._update_dimension_metrics(self._aggregated_metrics['strategy'],
            strategy, feedback)
        self._update_dimension_metrics(self._aggregated_metrics[
            'signal_type'], signal_type, feedback)
        self._update_dimension_metrics(self._aggregated_metrics[
            'market_regime'], market_regime, feedback)
        self._update_dimension_metrics(self._aggregated_metrics['session'],
            session, feedback)

    def _update_dimension_metrics(self, dimension_metrics: Dict[str, Any],
        dimension_value: str, feedback: Dict[str, Any]) ->None:
        """
        Update metrics for a specific dimension.
        
        Args:
            dimension_metrics: Metrics dictionary for the dimension
            dimension_value: Value of the dimension to update
            feedback: Feedback data
        """
        if dimension_value not in dimension_metrics:
            dimension_metrics[dimension_value] = {'count': 0, 'win_count': 
                0, 'loss_count': 0, 'total_pnl': 0, 'avg_pnl': 0,
                'avg_holding_time_minutes': 0, 'avg_slippage_pips': 0,
                'price_improvement_count': 0}
        metrics = dimension_metrics[dimension_value]
        metrics['count'] += 1
        pnl = feedback.get('pnl', 0)
        if pnl is not None:
            metrics['total_pnl'] += pnl
            metrics['avg_pnl'] = metrics['total_pnl'] / metrics['count']
            if pnl > 0:
                metrics['win_count'] += 1
            else:
                metrics['loss_count'] += 1
        holding_time = feedback.get('execution_metrics', {}).get(
            'holding_time_minutes')
        if holding_time is not None:
            prev_total = metrics['avg_holding_time_minutes'] * (metrics[
                'count'] - 1)
            metrics['avg_holding_time_minutes'] = (prev_total + holding_time
                ) / metrics['count']
        slippage = feedback.get('execution_metrics', {}).get('slippage_pips')
        if slippage is not None:
            prev_total = metrics['avg_slippage_pips'] * (metrics['count'] - 1)
            metrics['avg_slippage_pips'] = (prev_total + slippage) / metrics[
                'count']
        price_improvement = feedback.get('execution_metrics', {}).get(
            'price_improvement')
        if price_improvement:
            metrics['price_improvement_count'] += 1

    def _get_filtered_aggregated_metrics(self, filter_params: Optional[Dict
        [str, Any]]=None) ->Dict[str, Any]:
        """
        Get pre-aggregated metrics based on filter parameters.
        
        Args:
            filter_params: Parameters to filter by
            
        Returns:
            Filtered aggregated metrics
        """
        if not filter_params:
            return self._aggregated_metrics
        result = {}
        dimension = filter_params.get('dimension')
        if dimension and dimension in self._aggregated_metrics:
            dimension_metrics = self._aggregated_metrics[dimension]
            value = filter_params.get('value')
            if value and value in dimension_metrics:
                return {dimension: {value: dimension_metrics[value]}}
            else:
                return {dimension: dimension_metrics}
        return self._aggregated_metrics

    def _aggregate_feedback_data(self, data: List[Dict[str, Any]], group_by:
        List[str]) ->Dict[str, Any]:
        """
        Aggregate feedback data based on grouping fields.
        
        Args:
            data: Feedback data to aggregate
            group_by: Fields to group by
            
        Returns:
            Aggregated metrics
        """
        result = {}
        groups = {}
        for feedback in data:
            group_values = []
            for field in group_by:
                if '.' in field:
                    parts = field.split('.')
                    current = feedback
                    value = None
                    for part in parts:
                        if isinstance(current, dict) and part in current:
                            current = current[part]
                            value = current
                        else:
                            value = None
                            break
                    group_values.append(str(value))
                else:
                    group_values.append(str(feedback.get(field, 'unknown')))
            group_key = ':'.join(group_values)
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(feedback)
        for group_key, group_data in groups.items():
            group_metrics = {'count': len(group_data), 'win_count': 0,
                'loss_count': 0, 'total_pnl': 0, 'avg_pnl': 0,
                'avg_holding_time_minutes': 0, 'avg_slippage_pips': 0,
                'price_improvement_count': 0}
            total_holding_time = 0
            total_slippage = 0
            holding_time_count = 0
            slippage_count = 0
            for feedback in group_data:
                pnl = feedback.get('pnl', 0)
                if pnl is not None:
                    group_metrics['total_pnl'] += pnl
                    if pnl > 0:
                        group_metrics['win_count'] += 1
                    else:
                        group_metrics['loss_count'] += 1
                holding_time = feedback.get('execution_metrics', {}).get(
                    'holding_time_minutes')
                if holding_time is not None:
                    total_holding_time += holding_time
                    holding_time_count += 1
                slippage = feedback.get('execution_metrics', {}).get(
                    'slippage_pips')
                if slippage is not None:
                    total_slippage += slippage
                    slippage_count += 1
                price_improvement = feedback.get('execution_metrics', {}).get(
                    'price_improvement')
                if price_improvement:
                    group_metrics['price_improvement_count'] += 1
            if group_metrics['count'] > 0:
                group_metrics['avg_pnl'] = group_metrics['total_pnl'
                    ] / group_metrics['count']
            if holding_time_count > 0:
                group_metrics['avg_holding_time_minutes'
                    ] = total_holding_time / holding_time_count
            if slippage_count > 0:
                group_metrics['avg_slippage_pips'
                    ] = total_slippage / slippage_count
            if group_metrics['count'] > 0:
                group_metrics['win_rate'] = group_metrics['win_count'
                    ] / group_metrics['count']
                group_metrics['price_improvement_rate'] = group_metrics[
                    'price_improvement_count'] / group_metrics['count']
            keys = group_key.split(':')
            current_level = result
            for i, (field, key) in enumerate(zip(group_by, keys)):
                if i == len(group_by) - 1:
                    if field not in current_level:
                        current_level[field] = {}
                    current_level[field][key] = group_metrics
                else:
                    field_name = field.split('.')[-1]
                    if field_name not in current_level:
                        current_level[field_name] = {}
                    if key not in current_level[field_name]:
                        current_level[field_name][key] = {}
                    current_level = current_level[field_name][key]
        return result

    async def store_execution_quality_metrics(self, instrument: str,
        strategy_id: str, metrics: Dict[str, Any], feedback_id: Optional[
        str]=None) ->str:
        """
        Store execution quality metrics for analysis.
        
        Args:
            instrument: The trading instrument
            strategy_id: The strategy ID
            metrics: Execution quality metrics
            feedback_id: Optional feedback ID for linking related records
            
        Returns:
            ID of the feedback record
        """
        if not feedback_id:
            feedback_id = str(uuid.uuid4())
        feedback = {'id': feedback_id, 'timestamp': datetime.now().
            isoformat(), 'instrument': instrument, 'strategy_id':
            strategy_id, 'feedback_type': 'execution_quality',
            'execution_metrics': metrics}
        await self._store_feedback(feedback)
        self.logger.info(
            f'Stored execution quality metrics for {instrument} (strategy: {strategy_id}): ID={feedback_id}, metrics={list(metrics.keys())}'
            )
        return feedback_id
