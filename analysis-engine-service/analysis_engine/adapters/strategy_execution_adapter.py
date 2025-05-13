"""
Strategy Execution Adapter Module

This module provides adapter implementations for strategy execution interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union, Tuple
from datetime import datetime
import logging
import asyncio
import json
import copy
from common_lib.strategy.interfaces import IStrategyExecutor, ISignalAggregator, IStrategyEvaluator, SignalDirection, SignalTimeframe, SignalSource, MarketRegimeType
from core_foundations.utils.logger import get_logger
logger = get_logger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class StrategyExecutorAdapter(IStrategyExecutor):
    """
    Adapter for strategy execution that implements the common interface.
    
    This adapter can either wrap an actual executor instance or provide
    standalone functionality to avoid circular dependencies.
    """

    def __init__(self, executor_instance=None):
        """
        Initialize the adapter.
        
        Args:
            executor_instance: Optional actual executor instance to wrap
        """
        self.executor = executor_instance
        self.execution_history = []
        self.backtest_history = []

    @with_resilience('execute_strategy')
    @async_with_exception_handling
    async def execute_strategy(self, strategy_id: str, symbol: str,
        timeframe: str, parameters: Dict[str, Any]=None) ->Dict[str, Any]:
        """
        Execute a trading strategy.
        
        Args:
            strategy_id: The ID of the strategy to execute
            symbol: The trading symbol
            timeframe: The timeframe to use
            parameters: Optional strategy parameters
            
        Returns:
            Dictionary with execution results
        """
        if self.executor:
            try:
                return await self.executor.execute_strategy(strategy_id=
                    strategy_id, symbol=symbol, timeframe=timeframe,
                    parameters=parameters)
            except Exception as e:
                logger.warning(f'Error executing strategy: {str(e)}')
        execution_result = {'strategy_id': strategy_id, 'symbol': symbol,
            'timeframe': timeframe, 'parameters': parameters or {},
            'timestamp': datetime.now().isoformat(), 'signals': [],
            'execution_status': 'simulated'}
        self.execution_history.append(execution_result)
        return execution_result

    @async_with_exception_handling
    async def backtest_strategy(self, strategy_id: str, symbol: str,
        timeframe: str, start_date: datetime, end_date: datetime,
        parameters: Dict[str, Any]=None) ->Dict[str, Any]:
        """
        Backtest a trading strategy.
        
        Args:
            strategy_id: The ID of the strategy to backtest
            symbol: The trading symbol
            timeframe: The timeframe to use
            start_date: Start date for backtesting
            end_date: End date for backtesting
            parameters: Optional strategy parameters
            
        Returns:
            Dictionary with backtest results
        """
        if self.executor:
            try:
                return await self.executor.backtest_strategy(strategy_id=
                    strategy_id, symbol=symbol, timeframe=timeframe,
                    start_date=start_date, end_date=end_date, parameters=
                    parameters)
            except Exception as e:
                logger.warning(f'Error backtesting strategy: {str(e)}')
        backtest_result = {'strategy_id': strategy_id, 'symbol': symbol,
            'timeframe': timeframe, 'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(), 'parameters': parameters or {
            }, 'trades': [], 'metrics': {'total_trades': 0, 'win_rate': 0.0,
            'profit_factor': 0.0, 'sharpe_ratio': 0.0, 'max_drawdown': 0.0,
            'net_profit': 0.0}, 'execution_status': 'simulated'}
        self.backtest_history.append(backtest_result)
        return backtest_result

    @with_resilience('get_strategy_signals')
    @async_with_exception_handling
    async def get_strategy_signals(self, strategy_id: str, symbol: str,
        timeframe: str, parameters: Dict[str, Any]=None) ->List[Dict[str, Any]
        ]:
        """
        Get signals from a trading strategy.
        
        Args:
            strategy_id: The ID of the strategy
            symbol: The trading symbol
            timeframe: The timeframe to use
            parameters: Optional strategy parameters
            
        Returns:
            List of signal dictionaries
        """
        if self.executor:
            try:
                return await self.executor.get_strategy_signals(strategy_id
                    =strategy_id, symbol=symbol, timeframe=timeframe,
                    parameters=parameters)
            except Exception as e:
                logger.warning(f'Error getting strategy signals: {str(e)}')
        signals = [{'source_id': f'{strategy_id}_signal', 'source_type':
            'strategy', 'direction': 'neutral', 'symbol': symbol,
            'timeframe': timeframe, 'strength': 0.5, 'timestamp': datetime.
            now().isoformat(), 'metadata': {'strategy_id': strategy_id,
            'parameters': parameters or {}}}]
        return signals

    @async_with_exception_handling
    async def optimize_strategy(self, strategy_id: str, symbol: str,
        timeframe: str, start_date: datetime, end_date: datetime,
        parameters_range: Dict[str, Any]=None) ->Dict[str, Any]:
        """
        Optimize a trading strategy.
        
        Args:
            strategy_id: The ID of the strategy to optimize
            symbol: The trading symbol
            timeframe: The timeframe to use
            start_date: Start date for optimization
            end_date: End date for optimization
            parameters_range: Range of parameters to optimize
            
        Returns:
            Dictionary with optimization results
        """
        if self.executor:
            try:
                return await self.executor.optimize_strategy(strategy_id=
                    strategy_id, symbol=symbol, timeframe=timeframe,
                    start_date=start_date, end_date=end_date,
                    parameters_range=parameters_range)
            except Exception as e:
                logger.warning(f'Error optimizing strategy: {str(e)}')
        optimization_result = {'strategy_id': strategy_id, 'symbol': symbol,
            'timeframe': timeframe, 'start_date': start_date.isoformat(),
            'end_date': end_date.isoformat(), 'parameters_range': 
            parameters_range or {}, 'optimal_parameters': {},
            'optimization_metrics': {'iterations': 0, 'best_sharpe': 0.0,
            'best_profit': 0.0, 'best_drawdown': 0.0}, 'execution_status':
            'simulated'}
        return optimization_result


class SignalAggregatorAdapter(ISignalAggregator):
    """
    Adapter for signal aggregation that implements the common interface.
    
    This adapter can either wrap an actual aggregator instance or provide
    standalone functionality to avoid circular dependencies.
    """

    def __init__(self, aggregator_instance=None):
        """
        Initialize the adapter.
        
        Args:
            aggregator_instance: Optional actual aggregator instance to wrap
        """
        self.aggregator = aggregator_instance
        self.aggregation_history = []
        self.effectiveness_cache = {}

    @async_with_exception_handling
    async def aggregate_signals(self, signals: List[Dict[str, Any]], symbol:
        str, timeframe: str, market_regime: str=None) ->Dict[str, Any]:
        """
        Aggregate multiple trading signals.
        
        Args:
            signals: List of signal dictionaries
            symbol: The trading symbol
            timeframe: The timeframe of the signals
            market_regime: Optional market regime
            
        Returns:
            Dictionary with aggregated signal
        """
        if self.aggregator:
            try:
                return await self.aggregator.aggregate_signals(signals=
                    signals, symbol=symbol, timeframe=timeframe,
                    market_regime=market_regime)
            except Exception as e:
                logger.warning(f'Error aggregating signals: {str(e)}')
        if not signals:
            return {'direction': 'neutral', 'strength': 0.0, 'confidence': 
                0.0, 'timestamp': datetime.now().isoformat(),
                'component_signals': []}
        buy_count = sum(1 for s in signals if s.get('direction') == 'buy')
        sell_count = sum(1 for s in signals if s.get('direction') == 'sell')
        neutral_count = sum(1 for s in signals if s.get('direction') ==
            'neutral')
        strengths = [s.get('strength', 0.5) for s in signals if 'strength' in s
            ]
        avg_strength = sum(strengths) / len(strengths) if strengths else 0.5
        if buy_count > sell_count and buy_count > neutral_count:
            direction = 'buy'
            confidence = buy_count / len(signals)
        elif sell_count > buy_count and sell_count > neutral_count:
            direction = 'sell'
            confidence = sell_count / len(signals)
        else:
            direction = 'neutral'
            confidence = neutral_count / len(signals
                ) if neutral_count > 0 else 0.5
        aggregated_signal = {'direction': direction, 'strength':
            avg_strength, 'confidence': confidence, 'timestamp': datetime.
            now().isoformat(), 'component_signals': copy.deepcopy(signals),
            'symbol': symbol, 'timeframe': timeframe, 'market_regime': 
            market_regime or 'unknown'}
        self.aggregation_history.append(aggregated_signal)
        return aggregated_signal

    @with_resilience('get_signal_effectiveness')
    @async_with_exception_handling
    async def get_signal_effectiveness(self, source_id: str, market_regime:
        str=None, timeframe: str=None) ->Dict[str, float]:
        """
        Get effectiveness metrics for a signal source.
        
        Args:
            source_id: The source identifier
            market_regime: Optional market regime
            timeframe: Optional timeframe
            
        Returns:
            Dictionary with effectiveness metrics
        """
        if self.aggregator:
            try:
                return await self.aggregator.get_signal_effectiveness(source_id
                    =source_id, market_regime=market_regime, timeframe=
                    timeframe)
            except Exception as e:
                logger.warning(f'Error getting signal effectiveness: {str(e)}')
        cache_key = f'{source_id}_{market_regime}_{timeframe}'
        if cache_key in self.effectiveness_cache:
            return self.effectiveness_cache[cache_key]
        effectiveness = {'accuracy': 0.5, 'profit_factor': 1.0, 'win_rate':
            0.5, 'average_profit': 0.0, 'average_loss': 0.0, 'sample_size': 0}
        self.effectiveness_cache[cache_key] = effectiveness
        return effectiveness


class StrategyEvaluatorAdapter(IStrategyEvaluator):
    """
    Adapter for strategy evaluation that implements the common interface.
    
    This adapter can either wrap an actual evaluator instance or provide
    standalone functionality to avoid circular dependencies.
    """

    def __init__(self, evaluator_instance=None):
        """
        Initialize the adapter.
        
        Args:
            evaluator_instance: Optional actual evaluator instance to wrap
        """
        self.evaluator = evaluator_instance
        self.evaluation_history = []

    @async_with_exception_handling
    async def evaluate_strategy(self, strategy_id: str, backtest_results:
        Dict[str, Any]) ->Dict[str, Any]:
        """
        Evaluate a strategy based on backtest results.
        
        Args:
            strategy_id: The ID of the strategy
            backtest_results: Results from backtesting
            
        Returns:
            Dictionary with evaluation metrics
        """
        if self.evaluator:
            try:
                return await self.evaluator.evaluate_strategy(strategy_id=
                    strategy_id, backtest_results=backtest_results)
            except Exception as e:
                logger.warning(f'Error evaluating strategy: {str(e)}')
        evaluation = {'strategy_id': strategy_id, 'evaluation_timestamp':
            datetime.now().isoformat(), 'metrics': {'sharpe_ratio':
            backtest_results.get('metrics', {}).get('sharpe_ratio', 0.0),
            'sortino_ratio': 0.0, 'calmar_ratio': 0.0, 'profit_factor':
            backtest_results.get('metrics', {}).get('profit_factor', 0.0),
            'win_rate': backtest_results.get('metrics', {}).get('win_rate',
            0.0), 'max_drawdown': backtest_results.get('metrics', {}).get(
            'max_drawdown', 0.0), 'recovery_factor': 0.0, 'expectancy': 0.0
            }, 'rating': 'neutral', 'strengths': [], 'weaknesses': []}
        self.evaluation_history.append(evaluation)
        return evaluation

    @async_with_exception_handling
    async def compare_strategies(self, strategy_results: Dict[str, Dict[str,
        Any]]) ->Dict[str, Any]:
        """
        Compare multiple strategies.
        
        Args:
            strategy_results: Dictionary mapping strategy IDs to their results
            
        Returns:
            Dictionary with comparison results
        """
        if self.evaluator:
            try:
                return await self.evaluator.compare_strategies(strategy_results
                    =strategy_results)
            except Exception as e:
                logger.warning(f'Error comparing strategies: {str(e)}')
        comparison = {'comparison_timestamp': datetime.now().isoformat(),
            'strategies': list(strategy_results.keys()), 'metrics': {},
            'rankings': {}, 'best_overall': None}
        metrics = ['sharpe_ratio', 'profit_factor', 'win_rate', 'max_drawdown']
        for metric in metrics:
            comparison['metrics'][metric] = {}
            metric_values = {}
            for strategy_id, results in strategy_results.items():
                value = results.get('metrics', {}).get(metric, 0.0)
                metric_values[strategy_id] = value
                comparison['metrics'][metric][strategy_id] = value
            reverse = metric != 'max_drawdown'
            ranked = sorted(metric_values.items(), key=lambda x: x[1],
                reverse=reverse)
            comparison['rankings'][metric] = [item[0] for item in ranked]
        scores = {strategy_id: (0) for strategy_id in strategy_results.keys()}
        for metric, ranking in comparison['rankings'].items():
            for i, strategy_id in enumerate(ranking):
                if metric == 'max_drawdown':
                    scores[strategy_id] += len(ranking) - i
                else:
                    scores[strategy_id] += i + 1
        if scores:
            comparison['best_overall'] = max(scores.items(), key=lambda x: x[1]
                )[0]
        return comparison

    @with_resilience('get_strategy_performance')
    @async_with_exception_handling
    async def get_strategy_performance(self, strategy_id: str, start_date:
        datetime=None, end_date: datetime=None) ->Dict[str, Any]:
        """
        Get performance metrics for a strategy.
        
        Args:
            strategy_id: The ID of the strategy
            start_date: Optional start date for the period
            end_date: Optional end date for the period
            
        Returns:
            Dictionary with performance metrics
        """
        if self.evaluator:
            try:
                return await self.evaluator.get_strategy_performance(
                    strategy_id=strategy_id, start_date=start_date,
                    end_date=end_date)
            except Exception as e:
                logger.warning(f'Error getting strategy performance: {str(e)}')
        performance = {'strategy_id': strategy_id, 'period_start': 
            start_date.isoformat() if start_date else None, 'period_end': 
            end_date.isoformat() if end_date else None, 'metrics': {
            'total_trades': 0, 'win_rate': 0.0, 'profit_factor': 0.0,
            'sharpe_ratio': 0.0, 'sortino_ratio': 0.0, 'max_drawdown': 0.0,
            'net_profit': 0.0, 'annualized_return': 0.0}, 'monthly_returns':
            {}, 'drawdown_periods': [], 'equity_curve': []}
        return performance
