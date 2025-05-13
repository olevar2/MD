"""
Enhanced Tool Effectiveness Adapter Module

This module provides adapters for enhanced tool effectiveness functionality,
helping to break circular dependencies between services.
"""
import logging
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import uuid
import os
import httpx
import asyncio
import json
from common_lib.effectiveness.interfaces import MarketRegimeEnum, TimeFrameEnum, SignalEvent, SignalOutcome
from common_lib.effectiveness.enhanced_interfaces import IEnhancedToolEffectivenessTracker, IAdaptiveLayerService
from common_lib.adaptive.interfaces import AdaptationLevelEnum, IAdaptiveStrategyService
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

class EnhancedToolEffectivenessTrackerAdapter(IEnhancedToolEffectivenessTracker
    ):
    """
    Adapter for enhanced tool effectiveness tracking that implements the common interface.

    This adapter can either use a direct API connection to the analysis engine service
    or provide standalone functionality to avoid circular dependencies.
    """

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the adapter.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        analysis_engine_base_url = self.config.get('analysis_engine_base_url',
            os.environ.get('ANALYSIS_ENGINE_BASE_URL',
            'http://analysis-engine-service:8000'))
        self.client = httpx.AsyncClient(base_url=
            f"{analysis_engine_base_url.rstrip('/')}/api/v1", timeout=30.0)
        self.effectiveness_cache = {}
        self.cache_ttl = self.config_manager.get('cache_ttl_minutes', 15)
        self.signals = []
        self.outcomes = []

    @with_exception_handling
    def record_signal(self, signal: SignalEvent) ->str:
        """Record a new trading signal."""
        try:
            signal_id = getattr(signal, 'id', str(uuid.uuid4()))
            self.signals.append({'id': signal_id, 'tool_id': signal.tool_id,
                'symbol': signal.symbol, 'timeframe': signal.timeframe,
                'direction': signal.direction, 'market_regime': signal.
                market_regime, 'timestamp': signal.timestamp.isoformat() if
                isinstance(signal.timestamp, datetime) else signal.
                timestamp, 'metadata': signal.metadata})
            asyncio.create_task(self._send_signal_to_service(signal))
            return signal_id
        except Exception as e:
            logger.error(f'Error recording signal: {str(e)}')
            return str(uuid.uuid4())

    @with_exception_handling
    def record_outcome(self, outcome: SignalOutcome) ->bool:
        """Record the outcome of a signal."""
        try:
            self.outcomes.append({'signal_id': outcome.signal_id, 'result':
                outcome.result, 'pnl': outcome.pnl, 'exit_price': outcome.
                exit_price, 'exit_time': outcome.exit_time.isoformat() if
                isinstance(outcome.exit_time, datetime) else outcome.
                exit_time, 'metadata': outcome.metadata})
            asyncio.create_task(self._send_outcome_to_service(outcome))
            return True
        except Exception as e:
            logger.error(f'Error recording outcome: {str(e)}')
            return False

    @with_exception_handling
    def get_tool_effectiveness(self, tool_id: str, market_regime: Optional[
        str]=None, timeframe: Optional[str]=None, symbol: Optional[str]=
        None, start_date: Optional[datetime]=None, end_date: Optional[
        datetime]=None) ->Dict[str, Any]:
        """Get effectiveness metrics for a tool."""
        try:
            cache_key = f'{tool_id}_{market_regime}_{timeframe}_{symbol}'
            if cache_key in self.effectiveness_cache:
                cache_entry = self.effectiveness_cache[cache_key]
                cache_age = datetime.now() - cache_entry['timestamp']
                if cache_age.total_seconds() < self.cache_ttl * 60:
                    return cache_entry['data']
            effectiveness_data = asyncio.run(self.
                _get_effectiveness_from_service(tool_id, market_regime,
                timeframe, symbol, start_date, end_date))
            self.effectiveness_cache[cache_key] = {'timestamp': datetime.
                now(), 'data': effectiveness_data}
            return effectiveness_data
        except Exception as e:
            logger.error(f'Error getting tool effectiveness: {str(e)}')
            return self._calculate_effectiveness_locally(tool_id,
                market_regime, timeframe, symbol, start_date, end_date)

    @with_exception_handling
    def get_best_tools(self, market_regime: str, timeframe: Optional[str]=
        None, symbol: Optional[str]=None, min_signals: int=10, top_n:
        Optional[int]=None) ->List[Dict[str, Any]]:
        """Get best performing tools for a market regime."""
        try:
            best_tools = asyncio.run(self._get_best_tools_from_service(
                market_regime, timeframe, symbol, min_signals, top_n))
            return best_tools
        except Exception as e:
            logger.error(f'Error getting best tools: {str(e)}')
            return self._calculate_best_tools_locally(market_regime,
                timeframe, symbol, min_signals, top_n)

    @with_exception_handling
    def get_tool_performance_history(self, tool_id: str, market_regime:
        Optional[str]=None, timeframe: Optional[str]=None, symbol: Optional
        [str]=None, start_date: Optional[datetime]=None, end_date: Optional
        [datetime]=None) ->List[Dict[str, Any]]:
        """Get historical performance data for a tool."""
        try:
            performance_history = asyncio.run(self.
                _get_performance_history_from_service(tool_id,
                market_regime, timeframe, symbol, start_date, end_date))
            return performance_history
        except Exception as e:
            logger.error(f'Error getting tool performance history: {str(e)}')
            return []

    @with_exception_handling
    def get_regime_transition_performance(self, tool_id: str, from_regime:
        str, to_regime: str, timeframe: Optional[str]=None, symbol:
        Optional[str]=None) ->Dict[str, Any]:
        """Get tool performance during market regime transitions."""
        try:
            transition_performance = asyncio.run(self.
                _get_transition_performance_from_service(tool_id,
                from_regime, to_regime, timeframe, symbol))
            return transition_performance
        except Exception as e:
            logger.error(
                f'Error getting regime transition performance: {str(e)}')
            return {'tool_id': tool_id, 'from_regime': from_regime,
                'to_regime': to_regime, 'timeframe': timeframe, 'symbol':
                symbol, 'transition_count': 0, 'success_rate': 0.0,
                'average_pnl': 0.0, 'is_fallback': True}

    @with_exception_handling
    def get_tool_correlation_matrix(self, tool_ids: List[str],
        market_regime: Optional[str]=None, timeframe: Optional[str]=None,
        symbol: Optional[str]=None) ->Dict[str, Dict[str, float]]:
        """Get correlation matrix between different tools."""
        try:
            correlation_matrix = asyncio.run(self.
                _get_correlation_matrix_from_service(tool_ids,
                market_regime, timeframe, symbol))
            return correlation_matrix
        except Exception as e:
            logger.error(f'Error getting tool correlation matrix: {str(e)}')
            matrix = {}
            for tool_id in tool_ids:
                matrix[tool_id] = {other_id: (0.0) for other_id in tool_ids}
                matrix[tool_id][tool_id] = 1.0
            return matrix

    @with_exception_handling
    def get_tool_effectiveness_confidence(self, tool_id: str, market_regime:
        Optional[str]=None, timeframe: Optional[str]=None, symbol: Optional
        [str]=None) ->Dict[str, Any]:
        """Get confidence metrics for tool effectiveness."""
        try:
            confidence_metrics = asyncio.run(self.
                _get_effectiveness_confidence_from_service(tool_id,
                market_regime, timeframe, symbol))
            return confidence_metrics
        except Exception as e:
            logger.error(
                f'Error getting tool effectiveness confidence: {str(e)}')
            return {'tool_id': tool_id, 'market_regime': market_regime,
                'timeframe': timeframe, 'symbol': symbol, 'sample_size': 0,
                'confidence_level': 0.95, 'win_rate_ci': [0.0, 0.0],
                'pnl_ci': [0.0, 0.0], 'is_fallback': True}

    @with_exception_handling
    def get_optimal_tool_combination(self, market_regime: str, timeframe:
        Optional[str]=None, symbol: Optional[str]=None, max_tools: int=5
        ) ->List[Dict[str, Any]]:
        """Get optimal combination of tools for a market regime."""
        try:
            optimal_combination = asyncio.run(self.
                _get_optimal_combination_from_service(market_regime,
                timeframe, symbol, max_tools))
            return optimal_combination
        except Exception as e:
            logger.error(f'Error getting optimal tool combination: {str(e)}')
            return []

    @async_with_exception_handling
    async def _send_signal_to_service(self, signal: SignalEvent) ->bool:
        """Send a signal to the analysis engine service."""
        try:
            signal_dict = {'id': getattr(signal, 'id', str(uuid.uuid4())),
                'tool_id': signal.tool_id, 'symbol': signal.symbol,
                'timeframe': signal.timeframe, 'direction': signal.
                direction, 'market_regime': signal.market_regime,
                'timestamp': signal.timestamp.isoformat() if isinstance(
                signal.timestamp, datetime) else signal.timestamp,
                'metadata': signal.metadata}
            response = await self.client.post('/tool-effectiveness/signals',
                json=signal_dict)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f'Error sending signal to service: {str(e)}')
            return False

    @async_with_exception_handling
    async def _send_outcome_to_service(self, outcome: SignalOutcome) ->bool:
        """Send an outcome to the analysis engine service."""
        try:
            outcome_dict = {'signal_id': outcome.signal_id, 'result':
                outcome.result, 'pnl': outcome.pnl, 'exit_price': outcome.
                exit_price, 'exit_time': outcome.exit_time.isoformat() if
                isinstance(outcome.exit_time, datetime) else outcome.
                exit_time, 'metadata': outcome.metadata}
            response = await self.client.post('/tool-effectiveness/outcomes',
                json=outcome_dict)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f'Error sending outcome to service: {str(e)}')
            return False

    @async_with_exception_handling
    async def _get_effectiveness_from_service(self, tool_id: str,
        market_regime: Optional[str]=None, timeframe: Optional[str]=None,
        symbol: Optional[str]=None, start_date: Optional[datetime]=None,
        end_date: Optional[datetime]=None) ->Dict[str, Any]:
        """Get effectiveness metrics from the analysis engine service."""
        try:
            params = {'tool_id': tool_id}
            if market_regime:
                params['market_regime'] = market_regime
            if timeframe:
                params['timeframe'] = timeframe
            if symbol:
                params['symbol'] = symbol
            if start_date:
                params['start_date'] = start_date.isoformat()
            if end_date:
                params['end_date'] = end_date.isoformat()
            response = await self.client.get('/tool-effectiveness/metrics',
                params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error getting effectiveness from service: {str(e)}')
            raise

    @async_with_exception_handling
    async def _get_best_tools_from_service(self, market_regime: str,
        timeframe: Optional[str]=None, symbol: Optional[str]=None,
        min_signals: int=10, top_n: Optional[int]=None) ->List[Dict[str, Any]]:
        """Get best tools from the analysis engine service."""
        try:
            params = {'market_regime': market_regime, 'min_signals':
                min_signals}
            if timeframe:
                params['timeframe'] = timeframe
            if symbol:
                params['symbol'] = symbol
            if top_n:
                params['top_n'] = top_n
            response = await self.client.get('/tool-effectiveness/best-tools',
                params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f'Error getting best tools from service: {str(e)}')
            raise

    @async_with_exception_handling
    async def _get_performance_history_from_service(self, tool_id: str,
        market_regime: Optional[str]=None, timeframe: Optional[str]=None,
        symbol: Optional[str]=None, start_date: Optional[datetime]=None,
        end_date: Optional[datetime]=None) ->List[Dict[str, Any]]:
        """Get performance history from the analysis engine service."""
        try:
            params = {'tool_id': tool_id}
            if market_regime:
                params['market_regime'] = market_regime
            if timeframe:
                params['timeframe'] = timeframe
            if symbol:
                params['symbol'] = symbol
            if start_date:
                params['start_date'] = start_date.isoformat()
            if end_date:
                params['end_date'] = end_date.isoformat()
            response = await self.client.get(
                '/tool-effectiveness/performance-history', params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(
                f'Error getting performance history from service: {str(e)}')
            raise

    @async_with_exception_handling
    async def _get_transition_performance_from_service(self, tool_id: str,
        from_regime: str, to_regime: str, timeframe: Optional[str]=None,
        symbol: Optional[str]=None) ->Dict[str, Any]:
        """Get transition performance from the analysis engine service."""
        try:
            params = {'tool_id': tool_id, 'from_regime': from_regime,
                'to_regime': to_regime}
            if timeframe:
                params['timeframe'] = timeframe
            if symbol:
                params['symbol'] = symbol
            response = await self.client.get(
                '/tool-effectiveness/regime-transitions', params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(
                f'Error getting transition performance from service: {str(e)}')
            raise

    @async_with_exception_handling
    async def _get_correlation_matrix_from_service(self, tool_ids: List[str
        ], market_regime: Optional[str]=None, timeframe: Optional[str]=None,
        symbol: Optional[str]=None) ->Dict[str, Dict[str, float]]:
        """Get correlation matrix from the analysis engine service."""
        try:
            params = {'tool_ids': ','.join(tool_ids)}
            if market_regime:
                params['market_regime'] = market_regime
            if timeframe:
                params['timeframe'] = timeframe
            if symbol:
                params['symbol'] = symbol
            response = await self.client.get(
                '/tool-effectiveness/correlation-matrix', params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(
                f'Error getting correlation matrix from service: {str(e)}')
            raise

    @async_with_exception_handling
    async def _get_effectiveness_confidence_from_service(self, tool_id: str,
        market_regime: Optional[str]=None, timeframe: Optional[str]=None,
        symbol: Optional[str]=None) ->Dict[str, Any]:
        """Get effectiveness confidence from the analysis engine service."""
        try:
            params = {'tool_id': tool_id}
            if market_regime:
                params['market_regime'] = market_regime
            if timeframe:
                params['timeframe'] = timeframe
            if symbol:
                params['symbol'] = symbol
            response = await self.client.get('/tool-effectiveness/confidence',
                params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(
                f'Error getting effectiveness confidence from service: {str(e)}'
                )
            raise

    @async_with_exception_handling
    async def _get_optimal_combination_from_service(self, market_regime:
        str, timeframe: Optional[str]=None, symbol: Optional[str]=None,
        max_tools: int=5) ->List[Dict[str, Any]]:
        """Get optimal tool combination from the analysis engine service."""
        try:
            params = {'market_regime': market_regime, 'max_tools': max_tools}
            if timeframe:
                params['timeframe'] = timeframe
            if symbol:
                params['symbol'] = symbol
            response = await self.client.get(
                '/tool-effectiveness/optimal-combination', params=params)
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(
                f'Error getting optimal combination from service: {str(e)}')
            raise

    def _calculate_effectiveness_locally(self, tool_id: str, market_regime:
        Optional[str]=None, timeframe: Optional[str]=None, symbol: Optional
        [str]=None, start_date: Optional[datetime]=None, end_date: Optional
        [datetime]=None) ->Dict[str, Any]:
        """Calculate effectiveness metrics locally as a fallback."""
        filtered_signals = [s for s in self.signals if s['tool_id'] == tool_id]
        if market_regime:
            filtered_signals = [s for s in filtered_signals if s.get(
                'market_regime') == market_regime]
        if timeframe:
            filtered_signals = [s for s in filtered_signals if s.get(
                'timeframe') == timeframe]
        if symbol:
            filtered_signals = [s for s in filtered_signals if s.get(
                'symbol') == symbol]
        if start_date:
            filtered_signals = [s for s in filtered_signals if self.
                _parse_datetime(s.get('timestamp')) >= start_date]
        if end_date:
            filtered_signals = [s for s in filtered_signals if self.
                _parse_datetime(s.get('timestamp')) <= end_date]
        signal_ids = [s['id'] for s in filtered_signals]
        filtered_outcomes = [o for o in self.outcomes if o['signal_id'] in
            signal_ids]
        total_signals = len(filtered_signals)
        total_outcomes = len(filtered_outcomes)
        if total_outcomes == 0:
            return {'tool_id': tool_id, 'market_regime': market_regime,
                'timeframe': timeframe, 'symbol': symbol, 'total_signals':
                total_signals, 'total_outcomes': total_outcomes, 'win_rate':
                0.0, 'average_pnl': 0.0, 'is_fallback': True}
        wins = sum(1 for o in filtered_outcomes if o['result'] == 'win')
        win_rate = wins / total_outcomes if total_outcomes > 0 else 0.0
        total_pnl = sum(o.get('pnl', 0.0) for o in filtered_outcomes)
        average_pnl = total_pnl / total_outcomes if total_outcomes > 0 else 0.0
        return {'tool_id': tool_id, 'market_regime': market_regime,
            'timeframe': timeframe, 'symbol': symbol, 'total_signals':
            total_signals, 'total_outcomes': total_outcomes, 'win_rate':
            win_rate, 'average_pnl': average_pnl, 'is_fallback': True}

    def _calculate_best_tools_locally(self, market_regime: str, timeframe:
        Optional[str]=None, symbol: Optional[str]=None, min_signals: int=10,
        top_n: Optional[int]=None) ->List[Dict[str, Any]]:
        """Calculate best tools locally as a fallback."""
        tool_ids = set(s['tool_id'] for s in self.signals)
        tool_metrics = []
        for tool_id in tool_ids:
            metrics = self._calculate_effectiveness_locally(tool_id,
                market_regime, timeframe, symbol)
            if metrics['total_signals'] >= min_signals:
                tool_metrics.append(metrics)
        sorted_metrics = sorted(tool_metrics, key=lambda x: (x['win_rate'],
            x['average_pnl']), reverse=True)
        if top_n:
            sorted_metrics = sorted_metrics[:top_n]
        return sorted_metrics

    @with_exception_handling
    def _parse_datetime(self, dt_str: str) ->datetime:
        """Parse datetime string to datetime object."""
        if not dt_str:
            return datetime.min
        try:
            return datetime.fromisoformat(dt_str.replace('Z', '+00:00'))
        except (ValueError, TypeError):
            return datetime.min


class AdaptiveLayerServiceAdapter(IAdaptiveLayerService,
    IAdaptiveStrategyService):
    """
    Adapter for adaptive layer service that implements the common interfaces.

    This adapter can either use a direct API connection to the analysis engine service
    or provide standalone functionality to avoid circular dependencies.

    Implements both IAdaptiveLayerService and IAdaptiveStrategyService interfaces
    to break circular dependencies between services.
    """

    def __init__(self, config: Dict[str, Any]=None):
        """
        Initialize the adapter.

        Args:
            config: Configuration parameters
        """
        self.config = config or {}
        analysis_engine_base_url = self.config.get('analysis_engine_base_url',
            os.environ.get('ANALYSIS_ENGINE_BASE_URL',
            'http://analysis-engine-service:8000'))
        self.client = httpx.AsyncClient(base_url=
            f"{analysis_engine_base_url.rstrip('/')}/api/v1", timeout=30.0)
        self.parameters_cache = {}
        self.cache_ttl = self.config_manager.get('cache_ttl_minutes', 5)

    @async_with_exception_handling
    async def get_adaptive_parameters(self, symbol: str, timeframe: str,
        strategy_id: Optional[str]=None, context: Optional[Dict[str, Any]]=None
        ) ->Dict[str, Any]:
        """Get adaptive parameters for a trading strategy."""
        try:
            cache_key = f'{symbol}_{timeframe}_{strategy_id}'
            if cache_key in self.parameters_cache:
                cache_entry = self.parameters_cache[cache_key]
                cache_age = datetime.now() - cache_entry['timestamp']
                if cache_age.total_seconds() < self.cache_ttl * 60:
                    return cache_entry['data']
            params = {'symbol': symbol, 'timeframe': timeframe}
            if strategy_id:
                params['strategy_id'] = strategy_id
            request_data = {'context': context} if context else {}
            response = await self.client.post('/adaptive-layer/parameters',
                params=params, json=request_data)
            response.raise_for_status()
            adaptive_parameters = response.json()
            self.parameters_cache[cache_key] = {'timestamp': datetime.now(),
                'data': adaptive_parameters}
            return adaptive_parameters
        except Exception as e:
            logger.error(f'Error getting adaptive parameters: {str(e)}')
            return self._get_default_adaptive_parameters(symbol, timeframe,
                strategy_id)

    @async_with_exception_handling
    async def record_strategy_performance(self, strategy_id: str, symbol:
        str, timeframe: str, performance_metrics: Dict[str, Any],
        parameters_used: Dict[str, Any]) ->bool:
        """Record strategy performance for adaptive learning."""
        try:
            request_data = {'strategy_id': strategy_id, 'symbol': symbol,
                'timeframe': timeframe, 'performance_metrics':
                performance_metrics, 'parameters_used': parameters_used,
                'timestamp': datetime.now().isoformat()}
            response = await self.client.post('/adaptive-layer/performance',
                json=request_data)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f'Error recording strategy performance: {str(e)}')
            return False

    @async_with_exception_handling
    async def get_tool_signal_weights(self, market_regime: str, tools: List
        [str], timeframe: Optional[str]=None, symbol: Optional[str]=None
        ) ->Dict[str, float]:
        """
        Get signal weights for tools based on their effectiveness.

        Args:
            market_regime: Current market regime
            tools: List of tools to get weights for
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter

        Returns:
            Dictionary mapping tool IDs to weights
        """
        try:
            params = {'market_regime': market_regime}
            if timeframe:
                params['timeframe'] = timeframe
            if symbol:
                params['symbol'] = symbol
            request_data = {'tools': tools}
            response = await self.client.post('/adaptive-layer/tool-weights',
                params=params, json=request_data)
            response.raise_for_status()
            weights = response.json()
            return weights
        except Exception as e:
            logger.error(f'Error getting tool signal weights: {str(e)}')
            weight = 1.0 / len(tools) if tools else 0.0
            return {tool_id: weight for tool_id in tools}

    @async_with_exception_handling
    async def run_adaptation_cycle(self, market_regime: str, timeframe:
        Optional[str]=None, symbol: Optional[str]=None, lookback_hours: int=24
        ) ->Dict[str, Any]:
        """
        Run an adaptation cycle to adjust parameters based on current conditions.

        Args:
            market_regime: Current market regime
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter
            lookback_hours: Hours of data to consider

        Returns:
            Dictionary with adaptation results
        """
        try:
            params = {'market_regime': market_regime, 'lookback_hours':
                lookback_hours}
            if timeframe:
                params['timeframe'] = timeframe
            if symbol:
                params['symbol'] = symbol
            response = await self.client.post(
                '/adaptive-layer/adaptation-cycle', params=params)
            response.raise_for_status()
            adaptation_results = response.json()
            return adaptation_results
        except Exception as e:
            logger.error(f'Error running adaptation cycle: {str(e)}')
            return {'market_regime': market_regime, 'timeframe': timeframe,
                'symbol': symbol, 'adaptations_made': 0, 'status': 'error',
                'error': str(e), 'is_fallback': True}

    @async_with_exception_handling
    async def get_adaptation_recommendations(self, symbol: str, timeframe:
        str, current_market_data: Dict[str, Any], strategy_id: Optional[str
        ]=None) ->Dict[str, Any]:
        """Get recommendations for strategy adaptation."""
        try:
            params = {'symbol': symbol, 'timeframe': timeframe}
            if strategy_id:
                params['strategy_id'] = strategy_id
            request_data = {'current_market_data': current_market_data}
            response = await self.client.post('/adaptive-layer/recommendations'
                , params=params, json=request_data)
            response.raise_for_status()
            recommendations = response.json()
            return recommendations
        except Exception as e:
            logger.error(f'Error getting adaptation recommendations: {str(e)}')
            return self._get_default_adaptation_recommendations(symbol,
                timeframe, strategy_id)

    def _get_default_adaptive_parameters(self, symbol: str, timeframe: str,
        strategy_id: Optional[str]=None) ->Dict[str, Any]:
        """Get default adaptive parameters as a fallback."""
        return {'symbol': symbol, 'timeframe': timeframe, 'strategy_id':
            strategy_id, 'signal_weights': {'technical': 0.5, 'pattern': 
            0.3, 'sentiment': 0.2}, 'timeframe_weights': {'current': 0.6,
            'higher': 0.3, 'lower': 0.1}, 'risk_adjustment_factor': 1.0,
            'preferred_model': 'default', 'parameter_adjustments': {},
            'timestamp': datetime.now().isoformat(), 'is_fallback': True}

    def _get_default_adaptation_recommendations(self, symbol: str,
        timeframe: str, strategy_id: Optional[str]=None) ->Dict[str, Any]:
        """Get default adaptation recommendations as a fallback."""
        return {'symbol': symbol, 'timeframe': timeframe, 'strategy_id':
            strategy_id, 'market_regime': 'unknown', 'confidence': 0.5,
            'recommended_tools': [], 'parameter_adjustments': {},
            'risk_recommendation': 'neutral', 'timestamp': datetime.now().
            isoformat(), 'is_fallback': True}

    @with_exception_handling
    def set_adaptation_level(self, level: Union[str, AdaptationLevelEnum]
        ) ->None:
        """
        Set the adaptation aggressiveness level.

        Args:
            level: Adaptation level
        """
        try:
            level_str = level.value if isinstance(level, AdaptationLevelEnum
                ) else str(level)
            self.config['adaptation_level'] = level_str
            asyncio.create_task(self._set_adaptation_level_on_service(
                level_str))
        except Exception as e:
            logger.error(f'Error setting adaptation level: {str(e)}')

    @with_exception_handling
    def get_adaptation_level(self) ->str:
        """
        Get the current adaptation aggressiveness level.

        Returns:
            Current adaptation level
        """
        try:
            if 'adaptation_level' in self.config:
                return self.config['adaptation_level']
            return AdaptationLevelEnum.MODERATE.value
        except Exception as e:
            logger.error(f'Error getting adaptation level: {str(e)}')
            return AdaptationLevelEnum.MODERATE.value

    @async_with_exception_handling
    async def _set_adaptation_level_on_service(self, level: str) ->bool:
        """Send adaptation level to the analysis engine service."""
        try:
            request_data = {'level': level}
            response = await self.client.post(
                '/adaptive-layer/adaptation-level', json=request_data)
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f'Error setting adaptation level on service: {str(e)}'
                )
            return False
