"""
Causal Enhanced Strategy

This strategy leverages causal inference to identify and exploit temporal causal relationships
between currency pairs and market factors for more informed trading decisions.
"""
import networkx as nx
import httpx
import os
from typing import Dict, List, Any, Optional, Tuple
import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import json
from core.base_strategy import BaseStrategy
from adapters.causal_strategy_enhancer_adapter import CausalStrategyEnhancerAdapter
logger = logging.getLogger(__name__)


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

def dataframe_to_json_list(df: pd.DataFrame) ->List[Dict[str, Any]]:
    """
    Convert DataFrame to a JSON-serializable list of records.

    Args:
        df: DataFrame to convert

    Returns:
        List of dictionaries representing the DataFrame rows
    """
    if df.empty:
        return []
    df_reset = df.reset_index()
    for col in df_reset.columns:
        if pd.api.types.is_datetime64_any_dtype(df_reset[col]):
            df_reset[col] = df_reset[col].dt.strftime('%Y-%m-%dT%H:%M:%S')
    df_reset = df_reset.replace([np.inf, -np.inf], None)
    records = df_reset.to_dict(orient='records')
    for record in records:
        for key, value in record.items():
            if pd.isna(value):
                record[key] = None
    return records


def json_to_networkx(graph_data: Dict[str, Any]) ->nx.DiGraph:
    """
    Convert a node-link JSON representation to a NetworkX DiGraph.

    Args:
        graph_data: Node-link formatted graph data

    Returns:
        NetworkX DiGraph
    """
    graph = nx.DiGraph()
    for node in graph_data.get('nodes', []):
        node_id = node.get('id')
        if node_id:
            graph.add_node(node_id, **{k: v for k, v in node.items() if k !=
                'id'})
    for edge in graph_data.get('links', []):
        source = edge.get('source')
        target = edge.get('target')
        if source and target:
            graph.add_edge(source, target, **{k: v for k, v in edge.items() if
                k not in ('source', 'target')})
    return graph


class CausalEnhancedStrategy(BaseStrategy):
    """
    Trading strategy that uses causal inference to identify and exploit causal relationships
    between currency pairs and other market factors.

    This strategy:
    1. Discovers causal structures in forex markets using Granger causality
    2. Estimates the strength of causal effects between pairs
    3. Generates counterfactual scenarios to anticipate market movements
    4. Makes trading decisions based on causal insights
    """

    def __init__(self, name: str, parameters: Dict[str, Any]=None,
        config_manager: Optional[Any]=None):
        """
        Initialize the causal enhanced strategy.

        Args:
            name: Strategy name
            parameters: Strategy parameters
            config_manager: Optional ConfigurationManager instance.
        """
        super().__init__(name, parameters)
        if not self.parameters:
            self.parameters = {'symbols': ['EURUSD', 'GBPUSD', 'USDJPY',
                'AUDUSD'], 'timeframe': '1h', 'window_size': 30,
                'position_size_pct': 2.0, 'max_positions': 3,
                'stop_loss_pct': 1.0, 'take_profit_pct': 2.0,
                'causal_min_confidence': 0.7, 'lookback_periods': 500,
                'counterfactual_scenarios': 3, 'effect_threshold': 0.15}
        config = {}
        if config_manager:
            resolved_base_url = config_manager.get(
                'services.analysis_engine.base_url')
            if resolved_base_url:
                config['analysis_engine_base_url'] = resolved_base_url
        logger.info(f'Initializing CausalStrategyEnhancerAdapter')
        self.causal_enhancer = CausalStrategyEnhancerAdapter(config=config)
        resolved_base_url = config_manager.get('analysis_engine_base_url')
        if resolved_base_url is None:
            resolved_base_url = os.environ.get('ANALYSIS_ENGINE_BASE_URL',
                'http://analysis-engine-service:8000')
        analysis_engine_base_url = resolved_base_url.rstrip('/')
        logger.info(
            f'Connecting to Analysis Engine at: {analysis_engine_base_url}')
        self.analysis_engine_client = httpx.AsyncClient(base_url=
            analysis_engine_base_url, timeout=60.0)
        self.current_positions = {}
        self.causal_graph = None
        self.last_analysis_time = None
        self.discovered_relationships = {}
        self.effect_estimates = {}

    @async_with_exception_handling
    async def initialize(self) ->None:
        """Initialize the strategy with causal analysis."""
        logger.info(f'Initializing {self.name} strategy')
        try:
            await self._perform_causal_analysis()
            self.metadata['causal_enabled'] = True
            self.metadata['initialization_time'] = datetime.now()
            self.is_active = True
            logger.info(f'Strategy {self.name} initialized successfully')
        except Exception as e:
            logger.error(f'Error initializing strategy {self.name}: {str(e)}')
            self.is_active = False

    @async_with_exception_handling
    async def execute(self, market_data: pd.DataFrame) ->List[Dict[str, Any]]:
        """
        Execute the strategy based on current market data and causal insights.

        Args:
            market_data: Current market data

        Returns:
            List of trading signals/orders
        """
        if not self.is_active:
            logger.warning(
                f'Strategy {self.name} is not active, skipping execution')
            return []
        signals = []
        try:
            current_time = datetime.now()
            hours_since_analysis = 0
            if self.last_analysis_time:
                time_diff = current_time - self.last_analysis_time
                hours_since_analysis = time_diff.total_seconds() / 3600
            if not self.last_analysis_time or hours_since_analysis >= 12:
                await self._perform_causal_analysis(market_data)
            signals = await self._generate_signals(market_data)
            self._update_performance_metrics(market_data)
            logger.info(
                f'Strategy {self.name} execution completed: {len(signals)} signals generated'
                )
        except Exception as e:
            logger.error(f'Error executing strategy {self.name}: {str(e)}')
        return signals

    @async_with_exception_handling
    async def _perform_causal_analysis(self, data: pd.DataFrame=None) ->None:
        """
        Perform causal analysis on the provided market data by calling the analysis-engine API.

        Args:
            data: DataFrame containing historical market data with indicators.
        """
        if data is None or data.empty:
            logger.error('No data provided for causal analysis')
            return
        try:
            start_date = data.index[0].isoformat() if hasattr(data.index[0],
                'isoformat') else str(data.index[0])
            end_date = data.index[-1].isoformat() if hasattr(data.index[-1],
                'isoformat') else str(data.index[-1])
            data_period = {'start_date': start_date, 'end_date': end_date}
            graph_result = await self.causal_enhancer.generate_causal_graph(
                strategy_id=self.name, data_period=data_period)
            if 'error' in graph_result:
                logger.error(
                    f"Failed to generate causal graph: {graph_result['error']}"
                    )
                return
            self.causal_graph = self._convert_graph_result_to_networkx(
                graph_result)
            if not self.causal_graph or not self.causal_graph.edges():
                logger.error('Failed to discover causal structure via adapter')
                return
            logger.info(
                f'Causal graph updated via adapter with {self.causal_graph.number_of_edges()} edges'
                )
            self._extract_trading_relationships(self.causal_graph, data)
            factors_result = (await self.causal_enhancer.
                identify_causal_factors(strategy_id=self.name, data_period=
                data_period, significance_threshold=self.parameters.get(
                'effect_threshold', 0.15)))
            if 'error' in factors_result:
                logger.error(
                    f"Failed to identify causal factors: {factors_result['error']}"
                    )
            elif 'causal_factors' in factors_result:
                causal_factors = factors_result['causal_factors']
                self.effect_estimates = {}
                for factor in causal_factors:
                    source = factor.get('factor', '')
                    target = 'strategy_performance'
                    significance = factor.get('significance', 0)
                    self.effect_estimates[source, target] = significance
                logger.info(
                    f'Causal factors identified via adapter: {len(causal_factors)} factors'
                    )
            self.last_analysis_time = datetime.now()
            self.metadata['last_causal_analysis'] = self.last_analysis_time
            logger.info(
                f'Causal analysis completed for {self.name}: found {len(self.discovered_relationships)} causal relationships'
                )
        except Exception as e:
            logger.error(f'Error during causal analysis via adapter: {str(e)}',
                exc_info=True)
            self._reset_causal_state()

    def _convert_graph_result_to_networkx(self, graph_result: Dict[str, Any]
        ) ->nx.DiGraph:
        """
        Convert a graph result from the adapter to a NetworkX DiGraph.

        Args:
            graph_result: Graph result from the adapter

        Returns:
            NetworkX DiGraph
        """
        graph = nx.DiGraph()
        for node in graph_result.get('nodes', []):
            node_id = node.get('id')
            if node_id:
                node_type = node.get('type', 'factor')
                graph.add_node(node_id, type=node_type)
        for edge in graph_result.get('edges', []):
            source = edge.get('source')
            target = edge.get('target')
            weight = edge.get('weight', 0)
            if source and target:
                graph.add_edge(source, target, weight=weight)
        return graph

    def _reset_causal_state(self):
        """Reset causal analysis state on error."""
        self.causal_graph = None
        self.effect_estimates = {}
        self.discovered_relationships = {}
        logger.warning('Reset causal analysis state due to error')

    def _extract_trading_relationships(self, causal_graph: nx.DiGraph, data:
        pd.DataFrame) ->None:
        """
        Extract actionable trading relationships from the causal graph.

        Args:
            causal_graph: Discovered causal graph
            data: Prepared market data
        """
        self.discovered_relationships = {}
        price_columns = [col for col in data.columns if 'close' in col or 
            'price' in col]
        for source, target in causal_graph.edges():
            if any(price in source for price in price_columns) and any(
                price in target for price in price_columns):
                effect_size = self._calculate_effect_size(source, target, data)
                if effect_size >= self.parameters['effect_threshold']:
                    relationship_key = f'{source}->{target}'
                    self.discovered_relationships[relationship_key] = {'source'
                        : source, 'target': target, 'effect_size':
                        effect_size, 'confidence': self.
                        _calculate_confidence(source, target, data)}
                    self.effect_estimates[source, target] = effect_size

    @with_exception_handling
    def _calculate_effect_size(self, source: str, target: str, data: pd.
        DataFrame) ->float:
        """
        Calculate the effect size of source variable on target variable.

        Args:
            source: Source variable name
            target: Target variable name
            data: Market data

        Returns:
            Effect size as a float between 0 and 1
        """
        try:
            if source not in data.columns or target not in data.columns:
                return 0.0
            correlation = data[source].corr(data[target])
            effect_size = abs(correlation)
            return effect_size
        except Exception as e:
            logger.error(f'Error calculating effect size: {str(e)}')
            return 0.0

    @with_exception_handling
    def _calculate_confidence(self, source: str, target: str, data: pd.
        DataFrame) ->float:
        """
        Calculate confidence in the causal relationship.

        Args:
            source: Source variable name
            target: Target variable name
            data: Market data

        Returns:
            Confidence as a float between 0 and 1
        """
        try:
            if source not in data.columns or target not in data.columns:
                return 0.0
            from statsmodels.tsa.stattools import grangercausalitytests
            test_data = data[[source, target]].dropna()
            test_results = grangercausalitytests(test_data, maxlag=5,
                verbose=False)
            p_value = test_results[1][0]['ssr_ftest'][1]
            confidence = max(0, min(1, 1 - p_value))
            return confidence
        except Exception as e:
            logger.error(f'Error calculating causal confidence: {str(e)}')
            return 0.0

    async def _generate_signals(self, market_data: pd.DataFrame) ->List[Dict
        [str, Any]]:
        """
        Generate trading signals based on causal insights and counterfactual analysis.

        Args:
            market_data: Current market data

        Returns:
            List of trading signals/orders
        """
        signals = []
        if not self.causal_graph or not self.discovered_relationships:
            logger.warning(
                'No causal relationships discovered yet, cannot generate signals'
                )
            return signals
        trading_opportunities = await self._identify_trading_opportunities(
            market_data)
        for opportunity in trading_opportunities[:self.parameters[
            'max_positions']]:
            symbol = opportunity['symbol']
            direction = opportunity['direction']
            confidence = opportunity['confidence']
            if confidence < self.parameters['causal_min_confidence']:
                continue
            position_size = self.parameters['position_size_pct'] * confidence
            current_price = market_data[f'{symbol}_close'].iloc[-1]
            stop_loss_pct = self.parameters['stop_loss_pct']
            take_profit_pct = self.parameters['take_profit_pct']
            if direction == 'buy':
                stop_loss = current_price * (1 - stop_loss_pct / 100)
                take_profit = current_price * (1 + take_profit_pct / 100)
            else:
                stop_loss = current_price * (1 + stop_loss_pct / 100)
                take_profit = current_price * (1 - take_profit_pct / 100)
            signal = {'symbol': symbol, 'action': direction,
                'position_size': position_size, 'stop_loss': stop_loss,
                'take_profit': take_profit, 'confidence': confidence,
                'reason': opportunity['reason'], 'timestamp': datetime.now()}
            signals.append(signal)
            logger.info(
                f'Generated {direction} signal for {symbol} with confidence {confidence:.2f}'
                )
        return signals

    async def _identify_trading_opportunities(self, market_data: pd.DataFrame
        ) ->List[Dict[str, Any]]:
        """
        Identify trading opportunities based on causal relationships and counterfactuals.

        Args:
            market_data: Current market data

        Returns:
            List of trading opportunities sorted by confidence
        """
        opportunities = []
        counterfactual_scenarios = (await self.
            _generate_counterfactual_scenarios(market_data))
        for scenario_name, scenario_data in counterfactual_scenarios.items():
            parts = scenario_name.split('_')
            if len(parts) < 2:
                continue
            symbol = parts[1]
            current_price = market_data.get(f'{symbol}_close', None)
            if current_price is None:
                continue
            current_price_value = current_price.iloc[-1]
            counterfactual_price = scenario_data.get(
                f'counterfactual_{symbol}_close', None)
            if counterfactual_price is None:
                continue
            counterfactual_price_value = counterfactual_price.iloc[-1]
            price_diff_pct = (counterfactual_price_value - current_price_value
                ) / current_price_value * 100
            if abs(price_diff_pct) < 0.1:
                continue
            direction = 'buy' if price_diff_pct > 0 else 'sell'
            edge_keys = [key for key in self.discovered_relationships.keys(
                ) if symbol in key.split('->')[1]]
            edge_confidences = [rel['confidence'] for key, rel in self.
                discovered_relationships.items() if key in edge_keys]
            causal_confidence = max(edge_confidences
                ) if edge_confidences else 0
            signal_confidence = 0.7 * causal_confidence + 0.3 * min(1.0, 
                abs(price_diff_pct) / 3)
            opportunity = {'symbol': symbol, 'direction': direction,
                'confidence': signal_confidence, 'expected_move_pct':
                price_diff_pct, 'reason':
                f'Causal inference and counterfactual analysis from {scenario_name} scenario'
                , 'scenario': scenario_name}
            opportunities.append(opportunity)
        opportunities.sort(key=lambda x: x['confidence'], reverse=True)
        return opportunities

    @async_with_exception_handling
    async def _generate_counterfactual_scenarios(self, market_data: pd.
        DataFrame) ->Dict[str, pd.DataFrame]:
        """
        Generate counterfactual scenarios for trading decisions via the causal strategy enhancer adapter.

        Args:
            market_data: Current market data

        Returns:
            Dictionary of counterfactual scenarios
        """
        counterfactuals = {}
        if not self.causal_graph or market_data.empty:
            logger.warning(
                'Cannot generate counterfactuals: missing causal graph or market data'
                )
            return counterfactuals
        start_date = market_data.index[0].isoformat() if hasattr(market_data
            .index[0], 'isoformat') else str(market_data.index[0])
        end_date = market_data.index[-1].isoformat() if hasattr(market_data
            .index[-1], 'isoformat') else str(market_data.index[-1])
        data_period = {'start_date': start_date, 'end_date': end_date}
        symbols = self.parameters['symbols']
        for symbol in symbols:
            target_var = f'{symbol}_close'
            if target_var not in market_data.columns:
                continue
            parents = []
            for source, target in self.causal_graph.edges():
                if target_var in target:
                    parents.append(source)
            if not parents:
                continue
            sorted_parents = []
            for parent in parents:
                effect_size = self.effect_estimates.get((parent, target_var), 0
                    )
                sorted_parents.append((parent, effect_size))
            sorted_parents.sort(key=lambda x: x[1], reverse=True)
            top_parents = [p for p, _ in sorted_parents[:3]]
            intervention = {}
            for parent in top_parents:
                if parent in market_data.columns:
                    intervention[parent] = float(market_data[parent].
                        quantile(0.9))
            if not intervention:
                continue
            causal_factors = []
            for parent, effect_size in sorted_parents:
                if parent in top_parents:
                    causal_factors.append({'factor': parent, 'significance':
                        float(effect_size), 'direction': 'positive' if 
                        effect_size > 0 else 'negative'})
            try:
                enhancement_result = (await self.causal_enhancer.
                    apply_causal_enhancement(strategy_id=self.name,
                    causal_factors=causal_factors, enhancement_parameters={
                    'target_variable': target_var, 'interventions':
                    intervention, 'data_period': data_period}))
                if 'error' in enhancement_result:
                    logger.error(
                        f"Error applying causal enhancement: {enhancement_result['error']}"
                        )
                    cf_result = await self._call_generate_counterfactuals(data
                        =market_data, target_var=target_var, interventions=
                        intervention)
                else:
                    cf_result = self._create_synthetic_counterfactual(
                        market_data, target_var, intervention)
            except Exception as e:
                logger.error(f'Error using causal enhancer adapter: {str(e)}')
                cf_result = await self._call_generate_counterfactuals(data=
                    market_data, target_var=target_var, interventions=
                    intervention)
            if cf_result is not None and not cf_result.empty:
                scenario_name = f'scenario_{symbol}'
                counterfactuals[scenario_name] = cf_result
                logger.info(
                    f'Generated counterfactual scenario for {symbol} with {len(cf_result)} data points'
                    )
                if target_var in cf_result.columns:
                    counterfactual_col_name = f'counterfactual_{target_var}'
                    counterfactuals[scenario_name][counterfactual_col_name
                        ] = cf_result[target_var]
            else:
                logger.warning(
                    f'Failed to generate counterfactual scenario for {symbol}')
        return counterfactuals

    @with_exception_handling
    def _create_synthetic_counterfactual(self, data: pd.DataFrame,
        target_var: str, interventions: Dict[str, float]) ->pd.DataFrame:
        """
        Create a synthetic counterfactual DataFrame based on interventions.

        This is a fallback method when the API call fails.

        Args:
            data: Original data
            target_var: Target variable to predict
            interventions: Dictionary of interventions

        Returns:
            DataFrame with counterfactual predictions
        """
        try:
            cf_data = data.copy()
            for var, value in interventions.items():
                if var in cf_data.columns:
                    cf_data[var] = value
            if target_var in cf_data.columns:
                effects = {}
                for (source, target
                    ), effect_size in self.effect_estimates.items():
                    if target == target_var and source in interventions:
                        effects[source] = effect_size
                if effects:
                    original_value = data[target_var].iloc[-1]
                    cf_value = original_value
                    for source, effect_size in effects.items():
                        if source in data.columns and source in interventions:
                            original_source_value = data[source].iloc[-1]
                            new_source_value = interventions[source]
                            percent_change = (new_source_value -
                                original_source_value) / original_source_value
                            cf_value += (original_value * percent_change *
                                effect_size)
                    cf_data[target_var] = cf_value
                    cf_data[f'counterfactual_{target_var}'] = cf_value
            return cf_data
        except Exception as e:
            logger.error(f'Error creating synthetic counterfactual: {str(e)}')
            return None

    @async_with_exception_handling
    async def _fetch_enhanced_data(self, symbols: List[str], timeframe: str,
        start_date: Optional[datetime]=None, end_date: Optional[datetime]=None
        ) ->pd.DataFrame:
        """Fetch enhanced data (including indicators) from the analysis-engine-service API."""
        request_payload = {'symbols': symbols, 'timeframe': timeframe,
            'include_indicators': True}
        if start_date:
            request_payload['start_date'] = start_date.isoformat()
        if end_date:
            request_payload['end_date'] = end_date.isoformat()
        try:
            response = await self.analysis_engine_client.post(
                '/api/v1/causal-visualization/enhanced-data', json=
                request_payload)
            response.raise_for_status()
            data = response.json()
            if not data or not data.get('data'):
                logger.warning(
                    'Received empty data from analysis-engine-service')
                return pd.DataFrame()
            df = pd.DataFrame(data['data'])
            if df.empty:
                return df
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            logger.info(
                f'Successfully fetched {len(df)} data points from analysis-engine-service'
                )
            return df
        except httpx.HTTPStatusError as e:
            logger.error(
                f'HTTP error fetching data from analysis-engine: {e.response.status_code} - {e.response.text}'
                )
            return pd.DataFrame()
        except httpx.RequestError as e:
            logger.error(
                f'Request error fetching data from analysis-engine: {str(e)}')
            return pd.DataFrame()
        except Exception as e:
            logger.error(
                f'Unexpected error processing data from analysis-engine: {str(e)}'
                , exc_info=True)
            return pd.DataFrame()

    async def _update_causal_analysis(self):
        """Fetch latest data and trigger the causal analysis."""
        logger.info('Updating causal analysis...')
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.parameters['window_size'])
        data = await self._fetch_enhanced_data(symbols=self.parameters[
            'symbols'], timeframe=self.parameters['timeframe'], start_date=
            start_date, end_date=end_date)
        if data.empty:
            logger.warning('No data available to update causal analysis.')
            return
        await self._perform_causal_analysis(data)

    async def generate_signals(self) ->Dict[str, str]:
        """Generate trading signals based on the latest causal analysis."""
        signals = {}
        if not self.last_analysis_time or datetime.now(
            ) - self.last_analysis_time > timedelta(minutes=60):
            await self._update_causal_analysis()
        if not self.causal_graph or not self.effect_estimates:
            logger.warning(
                'Causal analysis data not available, cannot generate signals.')
            return {}
        for (cause_node, effect_node
            ), effect_data in self.effect_estimates.items():
            effect_strength = effect_data.get('average_effect', 0)
            if cause_node.startswith('EURUSD') and effect_node.startswith(
                'GBPUSD') and effect_strength > self.parameters.get(
                'effect_threshold', 0.15):
                logger.info(
                    f'Potential BUY signal for GBPUSD based on causal link from EURUSD (Effect: {effect_strength:.3f})'
                    )
                signals['GBPUSD'] = 'BUY'
            elif cause_node.startswith('USDJPY') and effect_node.startswith(
                'AUDUSD') and effect_strength < -self.parameters.get(
                'effect_threshold', 0.15):
                logger.info(
                    f'Potential SELL signal for AUDUSD based on causal link from USDJPY (Effect: {effect_strength:.3f})'
                    )
                signals['AUDUSD'] = 'SELL'
        limited_signals = dict(list(signals.items())[:self.parameters.get(
            'max_positions', 3)])
        logger.info(f'Generated signals: {limited_signals}')
        return limited_signals

    async def execute_trades(self, signals: Dict[str, str]) ->None:
        """Execute trades based on generated signals."""
        account_balance = 100000
        risk_per_trade = self.parameters.get('position_size_pct', 2.0) / 100.0
        for symbol, signal in signals.items():
            if symbol not in self.current_positions:
                position_size = account_balance * risk_per_trade
                logger.info(
                    f'Executing {signal} for {symbol}, Size: {position_size:.2f}'
                    )
                self.current_positions[symbol] = {'signal': signal, 'size':
                    position_size, 'entry_time': datetime.now()}
            else:
                logger.info(
                    f'Already have a position for {symbol}, skipping new signal.'
                    )
        positions_to_close = []
        for symbol, position_data in self.current_positions.items():
            if symbol not in signals:
                logger.info(f'Closing position for {symbol}')
                positions_to_close.append(symbol)
        for symbol in positions_to_close:
            del self.current_positions[symbol]

    async def run_iteration(self) ->None:
        """Run a single iteration of the strategy."""
        logger.info(f'Running iteration for strategy: {self.name}')
        signals = await self.generate_signals()
        await self.execute_trades(signals)
        logger.info(f'Iteration complete for strategy: {self.name}')

    async def close(self) ->None:
        """Clean up resources when the strategy stops."""
        await self.analysis_engine_client.aclose()
        logger.info(f'Strategy {self.name} closed.')

    @async_with_exception_handling
    async def _call_discover_causal_structure(self, data: pd.DataFrame,
        method: str='granger') ->Optional[nx.DiGraph]:
        """
        Call the analysis-engine-service API to discover causal structure.

        Args:
            data: DataFrame containing market data
            method: Causal discovery method (e.g., 'granger', 'pc', etc.)

        Returns:
            NetworkX DiGraph representing causal structure, or None if API call fails
        """
        logger.info(
            f'Calling analysis-engine API to discover causal structure using {method} method'
            )
        try:
            data_list = dataframe_to_json_list(data)
            payload = {'data': data_list, 'method': method, 'alpha': 0.05}
            response = await self.analysis_engine_client.post(
                '/api/v1/causal/discover-structure', json=payload)
            response.raise_for_status()
            result = response.json()
            if not result or 'graph' not in result:
                logger.warning('No valid graph returned from discovery API')
                return None
            graph = json_to_networkx(result['graph'])
            logger.info(
                f'Successfully received causal graph with {graph.number_of_edges()} edges'
                )
            return graph
        except httpx.HTTPStatusError as e:
            logger.error(
                f'HTTP error during causal structure discovery: {e.response.status_code} - {e.response.text}'
                )
            return None
        except httpx.RequestError as e:
            logger.error(
                f'Request error during causal structure discovery: {str(e)}')
            return None
        except Exception as e:
            logger.error(
                f'Error processing causal structure API response: {str(e)}',
                exc_info=True)
            return None

    @async_with_exception_handling
    async def _call_estimate_effects(self, data: pd.DataFrame, graph: nx.
        DiGraph) ->Dict[Tuple[str, str], float]:
        """
        Call the analysis-engine-service API to estimate causal effects.

        Args:
            data: DataFrame containing market data
            graph: Causal graph structure

        Returns:
            Dictionary mapping (cause, effect) tuples to effect strengths
        """
        logger.info('Calling analysis-engine API to estimate causal effects')
        try:
            data_list = dataframe_to_json_list(data)
            graph_data = nx.node_link_data(graph)
            payload = {'data': data_list, 'graph': graph_data}
            response = await self.analysis_engine_client.post(
                '/api/v1/causal/estimate-effects', json=payload)
            response.raise_for_status()
            result = response.json()
            if not result or 'effects' not in result:
                logger.warning('No valid effects returned from API')
                return {}
            effects_dict = {}
            for effect_data in result['effects']:
                cause = effect_data.get('cause')
                effect = effect_data.get('effect')
                strength = effect_data.get('strength', 0.0)
                if cause and effect:
                    effects_dict[cause, effect] = strength
            logger.info(
                f'Successfully received {len(effects_dict)} causal effect estimates'
                )
            return effects_dict
        except httpx.HTTPStatusError as e:
            logger.error(
                f'HTTP error during causal effect estimation: {e.response.status_code} - {e.response.text}'
                )
            return {}
        except httpx.RequestError as e:
            logger.error(
                f'Request error during causal effect estimation: {str(e)}')
            return {}
        except Exception as e:
            logger.error(
                f'Error processing causal effects API response: {str(e)}',
                exc_info=True)
            return {}

    @async_with_exception_handling
    async def _call_generate_counterfactuals(self, data: pd.DataFrame,
        target_var: str, interventions: Dict[str, float]) ->Optional[pd.
        DataFrame]:
        """
        Call the analysis-engine-service API to generate counterfactual scenarios.

        Args:
            data: DataFrame containing market data
            target_var: The target variable to predict
            interventions: Dictionary mapping variable names to intervention values

        Returns:
            DataFrame with counterfactual predictions, or None if API call fails
        """
        logger.info(
            f'Calling analysis-engine API to generate counterfactuals for {target_var}'
            )
        try:
            data_list = dataframe_to_json_list(data)
            scenario_name = f"scenario_{target_var.split('_')[0]}"
            payload = {'data': data_list, 'target': target_var,
                'interventions': {scenario_name: interventions}, 'features':
                None}
            response = await self.analysis_engine_client.post(
                '/api/v1/causal/generate-counterfactuals', json=payload)
            response.raise_for_status()
            result = response.json()
            if not result or 'counterfactuals' not in result or not result[
                'counterfactuals']:
                logger.warning('No valid counterfactual data returned from API'
                    )
                return None
            scenario_data = result['counterfactuals'].get(scenario_name)
            if not scenario_data:
                logger.warning(
                    f'No counterfactual data for scenario {scenario_name}')
                return None
            cf_data = pd.DataFrame(scenario_data)
            if 'timestamp' in cf_data.columns:
                cf_data['timestamp'] = pd.to_datetime(cf_data['timestamp'])
                cf_data.set_index('timestamp', inplace=True)
            logger.info(
                f'Successfully received counterfactual data with {len(cf_data)} rows'
                )
            return cf_data
        except httpx.HTTPStatusError as e:
            logger.error(
                f'HTTP error during counterfactual generation: {e.response.status_code} - {e.response.text}'
                )
            return None
        except httpx.RequestError as e:
            logger.error(
                f'Request error during counterfactual generation: {str(e)}')
            return None
        except KeyError as e:
            logger.error(
                f'Key error processing counterfactual API response: {str(e)}')
            return None
        except ValueError as e:
            logger.error(
                f'Value error processing counterfactual API response: {str(e)}'
                )
            return None
        except Exception as e:
            logger.error(
                f'Error processing counterfactual API response: {str(e)}',
                exc_info=True)
            return None
