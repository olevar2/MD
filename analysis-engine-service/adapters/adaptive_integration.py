"""
Adaptive Layer Integration Service

This module provides integration between the Adaptive Layer and other services,
particularly the strategy execution engine. It handles applying adaptive parameters
to trading strategies and facilitates communication between components.
"""
import logging
import requests
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from analysis_engine.repositories.tool_effectiveness_repository import ToolEffectivenessRepository
from analysis_engine.services.market_regime_detector import MarketRegimeService
from analysis_engine.services.adaptive_layer import AdaptiveLayer, AdaptationStrategy
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class AdaptiveLayerIntegrationService:
    """
    Service for integrating the Adaptive Layer with other components
    of the trading platform, particularly the strategy execution engine.
    Handles parameter updates and application to strategies.
    """

    def __init__(self, repository: ToolEffectivenessRepository,
        market_regime_service: MarketRegimeService,
        strategy_execution_api_url: Optional[str]=None):
        """
        Initialize the Adaptive Layer Integration Service
        
        Args:
            repository: Repository for tool effectiveness data
            market_regime_service: Service for market regime detection
            strategy_execution_api_url: URL for the strategy execution engine API
        """
        self.repository = repository
        self.market_regime_service = market_regime_service
        self.strategy_execution_api_url = strategy_execution_api_url
        self.logger = logging.getLogger(__name__)

    @with_resilience('update_strategy_parameters')
    @with_exception_handling
    def update_strategy_parameters(self, strategy_id: str, symbol: str,
        timeframe: str, price_data: pd.DataFrame, available_tools: List[str
        ], adaptation_strategy: str='moderate') ->Dict[str, Any]:
        """
        Update strategy parameters based on market conditions and tool effectiveness,
        and apply them to the strategy execution engine.
        
        Args:
            strategy_id: ID of the strategy to update
            symbol: Trading symbol
            timeframe: Chart timeframe
            price_data: OHLCV price data
            available_tools: List of tool IDs available for the strategy
            adaptation_strategy: Strategy for parameter adaptation
            
        Returns:
            Dictionary with update status and parameters
        """
        try:
            self.logger.info(
                f'Updating strategy parameters for {strategy_id} on {symbol} {timeframe}'
                )
            try:
                adaptation_mode = AdaptationStrategy(adaptation_strategy)
            except ValueError:
                adaptation_mode = AdaptationStrategy.MODERATE
                self.logger.warning(
                    f'Invalid adaptation strategy: {adaptation_strategy}. Using MODERATE instead.'
                    )
            adaptive_layer = AdaptiveLayer(tool_effectiveness_repository=
                self.repository, market_regime_service=self.
                market_regime_service, adaptation_strategy=adaptation_mode)
            adaptive_params = adaptive_layer.generate_adaptive_parameters(
                symbol=symbol, timeframe=timeframe, price_data=price_data,
                available_tools=available_tools)
            if self.strategy_execution_api_url:
                self.logger.info(
                    f'Applying parameters to strategy execution engine for {strategy_id}'
                    )
                update_result = self._apply_parameters_to_strategy(strategy_id
                    =strategy_id, parameters=adaptive_params)
                return {'strategy_id': strategy_id, 'parameters':
                    adaptive_params, 'applied': True, 'application_result':
                    update_result, 'timestamp': datetime.now().isoformat()}
            else:
                self.logger.info(
                    f'Strategy execution API URL not provided. Parameters generated but not applied.'
                    )
                return {'strategy_id': strategy_id, 'parameters':
                    adaptive_params, 'applied': False, 'application_result':
                    'Strategy execution API URL not provided', 'timestamp':
                    datetime.now().isoformat()}
        except Exception as e:
            self.logger.error(f'Error updating strategy parameters: {str(e)}',
                exc_info=True)
            return {'strategy_id': strategy_id, 'error': str(e), 'applied':
                False, 'timestamp': datetime.now().isoformat()}

    @with_exception_handling
    def _apply_parameters_to_strategy(self, strategy_id: str, parameters:
        Dict[str, Any]) ->Dict[str, Any]:
        """
        Apply parameters to a strategy in the execution engine
        
        Args:
            strategy_id: ID of the strategy
            parameters: Parameters to apply
            
        Returns:
            Result of the update operation
        """
        try:
            payload = {'strategy_id': strategy_id, 'parameters': parameters,
                'source': 'adaptive_layer', 'timestamp': datetime.now().
                isoformat()}
            endpoint = (
                f'{self.strategy_execution_api_url}/strategies/{strategy_id}/parameters'
                )
            response = requests.post(endpoint, json=payload)
            if response.status_code == 200:
                return response.json()
            else:
                self.logger.error(
                    f'Error applying parameters to strategy: {response.text}')
                return {'success': False, 'error':
                    f'API error: {response.status_code} - {response.text}'}
        except Exception as e:
            self.logger.error(
                f'Error applying parameters to strategy: {str(e)}',
                exc_info=True)
            return {'success': False, 'error': str(e)}


class AdaptiveStrategyOptimizer:
    """
    Service for optimizing trading strategies based on effectiveness metrics
    and providing recommendations for tool selection and configuration.
    """

    def __init__(self, repository: ToolEffectivenessRepository,
        market_regime_service: MarketRegimeService):
        """
        Initialize the Adaptive Strategy Optimizer
        
        Args:
            repository: Repository for tool effectiveness data
            market_regime_service: Service for market regime detection
        """
        self.repository = repository
        self.market_regime_service = market_regime_service
        self.logger = logging.getLogger(__name__)

    @with_exception_handling
    def generate_strategy_recommendations(self, strategy_id: str, symbol:
        str, timeframe: str, price_data: pd.DataFrame, current_tools: List[
        str], all_available_tools: List[str]) ->Dict[str, Any]:
        """
        Generate recommendations for optimizing a strategy based on effectiveness metrics
        
        Args:
            strategy_id: ID of the strategy
            symbol: Trading symbol
            timeframe: Chart timeframe
            price_data: OHLCV price data
            current_tools: Tools currently used in the strategy
            all_available_tools: All available tools that could be used
            
        Returns:
            Dictionary with recommendations for tool selection and configuration
        """
        try:
            self.logger.info(
                f'Generating recommendations for strategy {strategy_id} on {symbol} {timeframe}'
                )
            regime_result = self.market_regime_service.detect_market_regime(
                price_data=price_data, symbol=symbol, timeframe=timeframe)
            current_regime = regime_result['regime']
            regime_confidence = regime_result['confidence']
            self.logger.info(
                f'Detected market regime: {current_regime} with confidence {regime_confidence:.2f}'
                )
            tool_metrics = self._get_tool_effectiveness(tools=
                all_available_tools, symbol=symbol, timeframe=timeframe,
                regime=current_regime)
            tools_by_effectiveness = sorted(tool_metrics, key=lambda x: x.
                get('win_rate', 0) * x.get('profit_factor', 0), reverse=True)
            tools_to_add = [t for t in tools_by_effectiveness[:5] if t[
                'tool_id'] not in current_tools and t.get('sample_size', 0) >=
                20 and t.get('win_rate', 0) >= 55 and t.get('profit_factor',
                0) >= 1.2]
            tools_to_remove = [t for t in tools_by_effectiveness if t[
                'tool_id'] in current_tools and (t.get('sample_size', 0) >=
                20 and (t.get('win_rate', 0) < 45 or t.get('profit_factor',
                0) < 1.0))]
            parameter_optimizations = self._generate_parameter_optimizations(
                tools=current_tools, symbol=symbol, timeframe=timeframe,
                regime=current_regime)
            result = {'strategy_id': strategy_id, 'symbol': symbol,
                'timeframe': timeframe, 'current_regime': current_regime,
                'regime_confidence': regime_confidence,
                'analysis_timestamp': datetime.now().isoformat(),
                'tool_recommendations': tools_by_effectiveness,
                'tools_to_add': tools_to_add, 'tools_to_remove':
                tools_to_remove, 'parameter_optimizations':
                parameter_optimizations}
            return result
        except Exception as e:
            self.logger.error(
                f'Error generating strategy recommendations: {str(e)}',
                exc_info=True)
            return {'strategy_id': strategy_id, 'error': str(e),
                'timestamp': datetime.now().isoformat()}

    @with_analysis_resilience('analyze_strategy_effectiveness_trend')
    @with_exception_handling
    def analyze_strategy_effectiveness_trend(self, strategy_id: str, symbol:
        str, timeframe: str, period_days: int=30, look_back_periods: int=6
        ) ->Dict[str, Any]:
        """
        Analyze how a strategy's effectiveness has changed over time
        
        Args:
            strategy_id: ID of the strategy
            symbol: Trading symbol
            timeframe: Chart timeframe
            period_days: Number of days in each analysis period
            look_back_periods: Number of periods to look back
            
        Returns:
            Dictionary with effectiveness trend analysis
        """
        try:
            self.logger.info(
                f'Analyzing effectiveness trend for strategy {strategy_id} on {symbol} {timeframe}'
                )
            end_date = datetime.now()
            periods = []
            for i in range(look_back_periods):
                period_end = end_date - timedelta(days=i * period_days)
                period_start = period_end - timedelta(days=period_days)
                periods.append({'start_date': period_start, 'end_date':
                    period_end, 'period': f'Period {look_back_periods - i}'})
            period_metrics = []
            for period in periods:
                metrics = self.repository.get_tool_effectiveness_for_strategy(
                    strategy_id=strategy_id, symbol=symbol, timeframe=
                    timeframe, from_date=period['start_date'], to_date=
                    period['end_date'])
                period_metrics.append({'period': period['period'],
                    'start_date': period['start_date'].isoformat(),
                    'end_date': period['end_date'].isoformat(), 'metrics':
                    metrics})
            trends = self._analyze_effectiveness_trends(period_metrics)
            result = {'strategy_id': strategy_id, 'symbol': symbol,
                'timeframe': timeframe, 'period_days': period_days,
                'look_back_periods': look_back_periods,
                'analysis_timestamp': datetime.now().isoformat(),
                'period_metrics': period_metrics, 'trends': trends}
            return result
        except Exception as e:
            self.logger.error(
                f'Error analyzing strategy effectiveness trend: {str(e)}',
                exc_info=True)
            return {'strategy_id': strategy_id, 'error': str(e),
                'timestamp': datetime.now().isoformat()}

    @with_exception_handling
    def _get_tool_effectiveness(self, tools: List[str], symbol: str,
        timeframe: str, regime: str) ->List[Dict[str, Any]]:
        """
        Get effectiveness metrics for a list of tools in a specific market regime
        
        Args:
            tools: List of tool IDs
            symbol: Trading symbol
            timeframe: Chart timeframe
            regime: Market regime
            
        Returns:
            List of tool effectiveness metrics
        """
        result = []
        for tool_id in tools:
            try:
                metrics = self.repository.get_tool_effectiveness_by_regime(
                    tool_id=tool_id, symbol=symbol, timeframe=timeframe,
                    regime=regime)
                metrics['tool_id'] = tool_id
                result.append(metrics)
            except Exception as e:
                self.logger.error(
                    f'Error getting effectiveness for tool {tool_id}: {str(e)}'
                    )
                result.append({'tool_id': tool_id, 'error': str(e)})
        return result

    @with_exception_handling
    def _generate_parameter_optimizations(self, tools: List[str], symbol:
        str, timeframe: str, regime: str) ->Dict[str, Any]:
        """
        Generate parameter optimization recommendations for tools
        
        Args:
            tools: List of tool IDs
            symbol: Trading symbol
            timeframe: Chart timeframe
            regime: Market regime
            
        Returns:
            Dictionary with parameter optimization recommendations
        """
        optimizations = {}
        for tool_id in tools:
            try:
                optimal_params = self.repository.get_optimal_parameters(tool_id
                    =tool_id, symbol=symbol, timeframe=timeframe, regime=regime
                    )
                if optimal_params:
                    optimizations[tool_id] = optimal_params
            except Exception as e:
                self.logger.error(
                    f'Error getting optimal parameters for tool {tool_id}: {str(e)}'
                    )
                optimizations[tool_id] = {'error': str(e)}
        return optimizations

    def _analyze_effectiveness_trends(self, period_metrics: List[Dict[str,
        Any]]) ->Dict[str, Any]:
        """
        Analyze effectiveness trends across periods
        
        Args:
            period_metrics: List of metrics for each period
            
        Returns:
            Dictionary with trend analysis
        """
        win_rate_trend = self._calculate_metric_trend([p['metrics'].get(
            'win_rate', 0) for p in period_metrics])
        profit_factor_trend = self._calculate_metric_trend([p['metrics'].
            get('profit_factor', 0) for p in period_metrics])
        import numpy as np
        win_rate_stability = np.std([p['metrics'].get('win_rate', 0) for p in
            period_metrics]) if len(period_metrics) > 1 else 0
        profit_factor_stability = np.std([p['metrics'].get('profit_factor',
            0) for p in period_metrics]) if len(period_metrics) > 1 else 0
        return {'win_rate_trend': win_rate_trend, 'profit_factor_trend':
            profit_factor_trend, 'win_rate_stability': win_rate_stability,
            'profit_factor_stability': profit_factor_stability}

    def _calculate_metric_trend(self, values: List[float]) ->str:
        """
        Calculate the trend direction for a metric
        
        Args:
            values: List of metric values across periods
            
        Returns:
            String describing the trend (improving, stable, declining)
        """
        if len(values) < 2:
            return 'insufficient_data'
        differences = [(values[i] - values[i - 1]) for i in range(1, len(
            values))]
        avg_change = sum(differences) / len(differences)
        if avg_change > 0.02:
            return 'improving'
        elif avg_change < -0.02:
            return 'declining'
        else:
            return 'stable'
