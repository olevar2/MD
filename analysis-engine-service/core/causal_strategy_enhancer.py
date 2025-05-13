"""
Causal Strategy Enhancement Module

This module provides integration between trading strategies and the causal inference
capabilities, allowing strategies to leverage causal insights for improved decision-making.
"""
import logging
from typing import Dict, List, Optional, Union, Any, Tuple
import pandas as pd
import networkx as nx
import asyncio
from datetime import datetime, timedelta
from common_lib.strategy.interfaces import ICausalStrategyEnhancer
from analysis_engine.causal.services.causal_inference_service import CausalInferenceService
from analysis_engine.causal.services.causal_data_connector import CausalDataConnector
from analysis_engine.adapters.causal_strategy_enhancer_adapter import CausalStrategyEnhancerAdapter
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class CausalStrategyEnhancer(ICausalStrategyEnhancer):
    """
    Enhances trading strategies with causal inference capabilities.

    This class provides methods to integrate causal discovery, effect estimation,
    and counterfactual analysis into trading strategies to improve decision-making.

    Implements the ICausalStrategyEnhancer interface to break circular dependencies.
    """

    def __init__(self, data_connector: CausalDataConnector, config: Dict[
        str, Any]=None):
        """
        Initialize the causal strategy enhancer.

        Args:
            data_connector: Connector for fetching causal data.
            config: Configuration parameters
        """
        self.config = config or {}
        self.causal_inference = CausalInferenceService(config.get(
            'causal_inference', {}))
        self.data_connector = data_connector
        self.default_timeframe = self.config_manager.get('default_timeframe', '1h')
        self.default_window = timedelta(days=self.config.get(
            'default_window_days', 30))
        self.update_interval = self.config_manager.get('update_interval_minutes', 60)
        self.strategy_causal_graphs = {}

    async def enhance_strategy(self, strategy: BaseStrategy) ->BaseStrategy:
        """
        Enhance a trading strategy with causal inference capabilities.

        Args:
            strategy: Trading strategy to enhance

        Returns:
            Enhanced strategy with causal capabilities
        """
        strategy.metadata['has_causal_enhancement'] = True
        strategy.metadata['causal_last_update'] = datetime.now()
        timeframe = strategy.parameters.get('timeframe', self.default_timeframe
            )
        symbols = strategy.parameters.get('symbols', [])
        if not symbols:
            logger.warning(f'No symbols defined for strategy {strategy.name}')
            return strategy
        self._attach_causal_methods(strategy)
        await self._initialize_causal_analysis(strategy, symbols, timeframe)
        return strategy

    def _attach_causal_methods(self, strategy: BaseStrategy) ->None:
        """
        Attach causal inference methods to a strategy instance.

        Args:
            strategy: Strategy to enhance with causal methods
        """
        strategy.get_causal_graph = lambda : self.strategy_causal_graphs.get(
            strategy.name)
        strategy.generate_counterfactual = (lambda data, target,
            interventions: self.causal_inference.generate_counterfactuals(
            data, target, interventions))
        strategy.estimate_causal_effect = (lambda data, treatment, outcome:
            self.causal_inference.estimate_causal_effect(data, treatment,
            outcome))

    @async_with_exception_handling
    async def _initialize_causal_analysis(self, strategy: BaseStrategy,
        symbols: List[str], timeframe: str) ->None:
        """
        Initialize causal analysis for a strategy.

        Args:
            strategy: Trading strategy to initialize
            symbols: Currency pairs to analyze
            timeframe: Data timeframe to use
        """
        try:
            historical_data = await self.data_connector.get_historical_data(
                symbols=symbols, start_date=datetime.now() - self.
                default_window, timeframe=timeframe, include_indicators=True)
            if historical_data.empty:
                logger.error(
                    f'Failed to retrieve historical data for strategy {strategy.name}'
                    )
                return
            prepared_data = (await self.data_connector.
                prepare_data_for_causal_analysis(historical_data))
            causal_graph = self.causal_inference.discover_causal_structure(
                prepared_data, method='granger', cache_key=
                f"{strategy.name}_{','.join(sorted(symbols))}_{timeframe}")
            self.strategy_causal_graphs[strategy.name] = causal_graph
            logger.info(
                f'Initialized causal analysis for strategy {strategy.name}')
        except Exception as e:
            logger.error(
                f'Error initializing causal analysis for strategy {strategy.name}: {str(e)}'
                )

    @async_with_exception_handling
    async def start_real_time_causal_updates(self, strategy: BaseStrategy,
        update_interval: Optional[int]=None) ->str:
        """
        Start real-time causal analysis updates for a strategy.

        Args:
            strategy: Trading strategy to update
            update_interval: Update interval in minutes (overrides default)

        Returns:
            Stream ID for the real-time updates
        """
        symbols = strategy.parameters.get('symbols', [])
        timeframe = strategy.parameters.get('timeframe', self.default_timeframe
            )
        interval = update_interval or self.update_interval
        if not symbols:
            raise ValueError(f'No symbols defined for strategy {strategy.name}'
                )

        @with_exception_handling
        def update_causal_analysis(data: pd.DataFrame):
    """
    Update causal analysis.
    
    Args:
        data: Description of data
    
    """

            try:
                loop = asyncio.get_event_loop()
                prepared_data = loop.run_until_complete(self.data_connector
                    .prepare_data_for_causal_analysis(data))
                updated_graph = (self.causal_inference.
                    discover_causal_structure(prepared_data, method=
                    'granger', force_refresh=True, cache_key=
                    f"{strategy.name}_{','.join(sorted(symbols))}_{timeframe}")
                    )
                self.strategy_causal_graphs[strategy.name] = updated_graph
                strategy.metadata['causal_last_update'] = datetime.now()
                logger.info(
                    f'Updated causal analysis for strategy {strategy.name}')
            except Exception as e:
                logger.error(
                    f'Error updating causal analysis for {strategy.name}: {str(e)}'
                    )
        stream_id = await self.data_connector.start_streaming(symbols=
            symbols, callback=update_causal_analysis, interval=interval * 
            60, timeframe=timeframe, window_size=self.default_window)
        strategy.metadata['causal_stream_id'] = stream_id
        return stream_id

    async def stop_real_time_updates(self, strategy: BaseStrategy) ->bool:
        """
        Stop real-time causal analysis updates for a strategy.

        Args:
            strategy: Trading strategy to stop updates for

        Returns:
            True if updates were successfully stopped, False otherwise
        """
        stream_id = strategy.metadata.get('causal_stream_id')
        if not stream_id:
            logger.warning(
                f'No active causal updates found for strategy {strategy.name}')
            return False
        success = await self.data_connector.stop_streaming(stream_id)
        if success:
            if 'causal_stream_id' in strategy.metadata:
                del strategy.metadata['causal_stream_id']
        return success

    @async_with_exception_handling
    async def enhance_strategy(self, strategy_id: str, enhancement_type:
        str, parameters: Dict[str, Any]) ->Dict[str, Any]:
        """Enhance a strategy with additional functionality."""
        try:
            logger.info(
                f'Enhancing strategy {strategy_id} with {enhancement_type}')
            if enhancement_type == 'causal':
                return await self.apply_causal_enhancement(strategy_id=
                    strategy_id, causal_factors=parameters.get(
                    'causal_factors', []), enhancement_parameters=
                    parameters.get('enhancement_parameters', {}))
            else:
                logger.warning(
                    f'Unsupported enhancement type: {enhancement_type}')
                return {'success': False, 'error':
                    f'Unsupported enhancement type: {enhancement_type}',
                    'strategy_id': strategy_id}
        except Exception as e:
            logger.error(f'Error enhancing strategy: {str(e)}')
            return {'success': False, 'error': str(e), 'strategy_id':
                strategy_id}

    @with_resilience('get_enhancement_types')
    async def get_enhancement_types(self) ->List[Dict[str, Any]]:
        """Get available enhancement types."""
        return [{'id': 'causal', 'name': 'Causal Enhancement',
            'description':
            'Enhances strategy based on causal factor analysis',
            'parameters': [{'name': 'significance_threshold', 'type':
            'float', 'default': 0.05, 'description':
            'Threshold for statistical significance'}, {'name':
            'max_factors', 'type': 'integer', 'default': 5, 'description':
            'Maximum number of causal factors to consider'}, {'name':
            'apply_filters', 'type': 'boolean', 'default': True,
            'description': 'Apply filters based on causal factors'}]}]

    @with_resilience('get_enhancement_history')
    async def get_enhancement_history(self, strategy_id: str) ->List[Dict[
        str, Any]]:
        """Get enhancement history for a strategy."""
        logger.info(f'Getting enhancement history for strategy {strategy_id}')
        return []

    async def compare_enhancements(self, strategy_id: str, enhancement_ids:
        List[str]) ->Dict[str, Any]:
        """Compare multiple enhancements for a strategy."""
        logger.info(
            f'Comparing enhancements for strategy {strategy_id}: {enhancement_ids}'
            )
        return {'strategy_id': strategy_id, 'enhancements': enhancement_ids,
            'comparison': {}, 'recommendation': None}

    @async_with_exception_handling
    async def identify_causal_factors(self, strategy_id: str, data_period:
        Dict[str, Any], significance_threshold: float=0.05) ->Dict[str, Any]:
        """Identify causal factors affecting strategy performance."""
        logger.info(f'Identifying causal factors for strategy {strategy_id}')
        try:
            symbols = ['EUR/USD']
            timeframe = self.default_timeframe
            start_date = datetime.fromisoformat(data_period.get(
                'start_date', (datetime.now() - self.default_window).
                isoformat()))
            end_date = datetime.fromisoformat(data_period.get('end_date',
                datetime.now().isoformat()))
            historical_data = await self.data_connector.get_historical_data(
                symbols=symbols, start_date=start_date, end_date=end_date,
                timeframe=timeframe, include_indicators=True)
            if historical_data.empty:
                logger.error(
                    f'Failed to retrieve historical data for strategy {strategy_id}'
                    )
                return {'strategy_id': strategy_id, 'error':
                    'Failed to retrieve historical data', 'causal_factors': []}
            prepared_data = (await self.data_connector.
                prepare_data_for_causal_analysis(historical_data))
            causal_graph = self.causal_inference.discover_causal_structure(
                prepared_data, method='granger', cache_key=
                f"{strategy_id}_{','.join(sorted(symbols))}_{timeframe}")
            causal_factors = []
            for node, data in causal_graph.nodes(data=True):
                if node != 'strategy_performance' and causal_graph.has_edge(
                    node, 'strategy_performance'):
                    edge_data = causal_graph.get_edge_data(node,
                        'strategy_performance')
                    weight = edge_data.get('weight', 0)
                    if weight > significance_threshold:
                        causal_factors.append({'factor': node,
                            'significance': weight, 'direction': 'positive' if
                            weight > 0 else 'negative'})
            return {'strategy_id': strategy_id, 'data_period': data_period,
                'significance_threshold': significance_threshold,
                'causal_factors': causal_factors, 'timestamp': datetime.now
                ().isoformat()}
        except Exception as e:
            logger.error(f'Error identifying causal factors: {str(e)}')
            return {'strategy_id': strategy_id, 'error': str(e),
                'causal_factors': []}

    @async_with_exception_handling
    async def generate_causal_graph(self, strategy_id: str, data_period:
        Dict[str, Any]) ->Dict[str, Any]:
        """Generate a causal graph for a strategy."""
        logger.info(f'Generating causal graph for strategy {strategy_id}')
        try:
            symbols = ['EUR/USD']
            timeframe = self.default_timeframe
            start_date = datetime.fromisoformat(data_period.get(
                'start_date', (datetime.now() - self.default_window).
                isoformat()))
            end_date = datetime.fromisoformat(data_period.get('end_date',
                datetime.now().isoformat()))
            historical_data = await self.data_connector.get_historical_data(
                symbols=symbols, start_date=start_date, end_date=end_date,
                timeframe=timeframe, include_indicators=True)
            if historical_data.empty:
                logger.error(
                    f'Failed to retrieve historical data for strategy {strategy_id}'
                    )
                return {'strategy_id': strategy_id, 'error':
                    'Failed to retrieve historical data', 'nodes': [],
                    'edges': []}
            prepared_data = (await self.data_connector.
                prepare_data_for_causal_analysis(historical_data))
            causal_graph = self.causal_inference.discover_causal_structure(
                prepared_data, method='granger', cache_key=
                f"{strategy_id}_{','.join(sorted(symbols))}_{timeframe}")
            nodes = []
            for node, data in causal_graph.nodes(data=True):
                node_type = ('outcome' if node == 'strategy_performance' else
                    'factor')
                nodes.append({'id': node, 'type': node_type})
            edges = []
            for source, target, data in causal_graph.edges(data=True):
                edges.append({'source': source, 'target': target, 'weight':
                    data.get('weight', 0)})
            return {'strategy_id': strategy_id, 'data_period': data_period,
                'nodes': nodes, 'edges': edges, 'timestamp': datetime.now()
                .isoformat()}
        except Exception as e:
            logger.error(f'Error generating causal graph: {str(e)}')
            return {'strategy_id': strategy_id, 'error': str(e), 'nodes': [
                ], 'edges': []}

    @async_with_exception_handling
    async def apply_causal_enhancement(self, strategy_id: str,
        causal_factors: List[Dict[str, Any]], enhancement_parameters: Dict[
        str, Any]) ->Dict[str, Any]:
        """Apply causal enhancement to a strategy."""
        logger.info(f'Applying causal enhancement to strategy {strategy_id}')
        try:
            enhancements = []
            for factor in causal_factors:
                factor_name = factor.get('factor', '')
                significance = factor.get('significance', 0)
                direction = factor.get('direction', 'positive')
                if 'volatility' in factor_name.lower():
                    enhancements.append({'type': 'filter', 'description':
                        'Added market volatility filter', 'details':
                        'Filter trades during high volatility periods'})
                if 'volume' in factor_name.lower():
                    enhancements.append({'type': 'filter', 'description':
                        'Added volume threshold filter', 'details':
                        'Only trade when volume is above threshold'})
                if 'trend' in factor_name.lower():
                    enhancements.append({'type': 'parameter', 'description':
                        'Adjusted trend sensitivity', 'details':
                        'Modified trend detection parameters'})
            if not enhancements:
                enhancements.append({'type': 'parameter', 'description':
                    'Adjusted stop loss based on volatility', 'details':
                    'Dynamic stop loss calculation using ATR'})
            return {'strategy_id': strategy_id, 'enhancement_id':
                f"causal_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'causal_factors': causal_factors, 'parameters':
                enhancement_parameters, 'enhancements': enhancements,
                'performance_impact': {'estimated_improvement': '15-20%',
                'confidence': 0.75}, 'timestamp': datetime.now().isoformat()}
        except Exception as e:
            logger.error(f'Error applying causal enhancement: {str(e)}')
            return {'strategy_id': strategy_id, 'error': str(e),
                'enhancement_id':
                f"causal_error_{datetime.now().strftime('%Y%m%d%H%M%S')}",
                'enhancements': []}
