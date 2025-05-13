"""
Adaptive Layer Service

This module provides the core functionality for the Adaptive Layer,
allowing strategies to dynamically adjust to changing market conditions
based on effectiveness metrics and market regime detection.
"""
import logging
from typing import Dict, List, Any, Optional, Union, Callable, Tuple, Set
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from pydantic import BaseModel
from enum import Enum
import asyncio
import json
from common_lib.effectiveness.interfaces import MarketRegimeEnum as MarketRegime, TimeFrameEnum as TimeFrame
from common_lib.adaptive.interfaces import AdaptationLevelEnum, IAdaptiveStrategyService
from adapters.tool_effectiveness_adapter import ToolEffectivenessTrackerAdapter
from adapters.enhanced_tool_effectiveness_adapter import EnhancedToolEffectivenessTrackerAdapter, AdaptiveLayerServiceAdapter
from adapters.adaptive_strategy_adapter import AdaptiveStrategyServiceAdapter


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class AdaptationLevel(Enum):
    """Defines how aggressively the system should adapt to changing conditions"""
    NONE = 0
    CONSERVATIVE = 1
    MODERATE = 2
    AGGRESSIVE = 3
    EXPERIMENTAL = 4


class AdaptationContext(BaseModel):
    """Context information for adaptation decisions"""
    current_market_regime: str
    previous_market_regime: Optional[str]
    regime_change_detected: bool
    current_volatility_percentile: float
    current_liquidity_score: float
    trading_session: str
    time_in_regime: int
    effectiveness_trends: Dict[str, List[float]]
    sample_size_sufficient: bool
    confidence_level: float


class ParameterAdjustment(BaseModel):
    """Represents an adjustment to a strategy parameter"""
    parameter_name: str
    previous_value: Any
    new_value: Any
    adjustment_reason: str
    confidence_level: float
    reversion_threshold: Optional[float] = None
    is_experimental: bool = False


class AdaptationResult(BaseModel):
    """Result of an adaptation cycle"""
    adaptation_id: str
    timestamp: datetime
    market_regime: str
    adaptations_applied: List[ParameterAdjustment]
    expected_impact: Dict[str, float]
    tools_affected: List[str]
    is_reversion: bool = False
    metadata: Dict[str, Any] = {}


class AdaptiveLayerService(IAdaptiveStrategyService):
    """
    Core service for the Adaptive Layer that enables strategies to
    dynamically adjust to changing market conditions.

    Implements the IAdaptiveStrategyService interface to break circular dependencies.
    """

    def __init__(self, initial_adaptation_level: AdaptationLevel=
        AdaptationLevel.MODERATE):
        """
        Initialize the Adaptive Layer service

        Args:
            initial_adaptation_level: Initial adaptation aggressiveness level
        """
        self.adaptation_level = initial_adaptation_level
        self.logger = logging.getLogger(__name__)
        self.enhanced_tracker = EnhancedToolEffectivenessTrackerAdapter()
        self.adaptive_layer_service = AdaptiveLayerServiceAdapter()
        self.adaptation_level_map = {AdaptationLevel.NONE:
            AdaptationLevelEnum.NONE, AdaptationLevel.CONSERVATIVE:
            AdaptationLevelEnum.CONSERVATIVE, AdaptationLevel.MODERATE:
            AdaptationLevelEnum.MODERATE, AdaptationLevel.AGGRESSIVE:
            AdaptationLevelEnum.AGGRESSIVE, AdaptationLevel.EXPERIMENTAL:
            AdaptationLevelEnum.EXPERIMENTAL}
        self.adaptation_handlers: Dict[str, Callable] = {}
        self.adaptation_history: List[AdaptationResult] = []
        self.current_parameters: Dict[str, Dict[str, Any]] = {}
        self.parameter_constraints: Dict[str, Dict[str, Any]] = {}
        self.regime_optimal_parameters: Dict[str, Dict[str, Dict[str, Any]]
            ] = {}
        self._initialize_decline_detection()
        self.signal_weights_cache = {}
        self.signal_weights_cache_expiry = {}
        self.signal_weights_cache_duration = timedelta(minutes=15)
        self.effectiveness_history = {}
        self.weight_adjustment_factors = {}
        self.recent_performance_window = timedelta(days=3)

    async def get_tool_signal_weights(self, market_regime: MarketRegime,
        tools: List[str], timeframe: Optional[TimeFrame]=None, symbol:
        Optional[str]=None) ->Dict[str, float]:
        """
        Calculate signal weights based on tool effectiveness metrics for the SignalAggregator

        Args:
            market_regime: Current market regime
            tools: List of tools/indicators for which to calculate weights
            timeframe: Trading timeframe
            symbol: Trading instrument symbol

        Returns:
            Dictionary mapping tool IDs to weights based on their effectiveness
        """
        cache_key = (
            f"{market_regime.value}:{','.join(sorted(tools))}:{timeframe.value if timeframe else 'all'}:{symbol or 'all'}"
            )
        now = datetime.now()
        if (cache_key in self.signal_weights_cache and cache_key in self.
            signal_weights_cache_expiry and now < self.
            signal_weights_cache_expiry[cache_key]):
            self.logger.debug(f'Using cached signal weights for {cache_key}')
            return self.signal_weights_cache[cache_key]
        weights = {}
        total_weight = 0.0
        effectiveness_metrics = {}
        for tool_id in tools:
            metrics = (await self.effectiveness_repository.
                get_tool_effectiveness_metrics_async(tool_id=tool_id,
                timeframe=timeframe.value if timeframe else None,
                instrument=symbol, market_regime=market_regime))
            if metrics:
                effectiveness_metrics[tool_id] = metrics
        for tool_id, metrics in effectiveness_metrics.items():
            win_rate = metrics.get('win_rate', 0.5)
            profit_factor = metrics.get('profit_factor', 1.0)
            expected_payoff = metrics.get('expected_payoff', 0.0)
            regime_metrics = metrics.get('regime_metrics', {}).get(
                market_regime.value, {})
            regime_win_rate = regime_metrics.get('win_rate', win_rate)
            regime_profit_factor = regime_metrics.get('profit_factor',
                profit_factor)
            effectiveness_score = (await self.
                _calculate_comprehensive_effectiveness_score(tool_id,
                metrics, market_regime, timeframe, symbol))
            recent_factor = self._get_recent_performance_factor(tool_id,
                market_regime)
            base_weight = effectiveness_score * recent_factor
            sample_size = metrics.get('signal_count', 0)
            sample_size_factor = min(sample_size / 100, 1.0)
            weight = base_weight * (0.5 + 0.5 * sample_size_factor)
            adaptation_factor = {AdaptationLevel.NONE: 0.0, AdaptationLevel
                .CONSERVATIVE: 0.7, AdaptationLevel.MODERATE: 1.0,
                AdaptationLevel.AGGRESSIVE: 1.3, AdaptationLevel.
                EXPERIMENTAL: 1.5}
            weight *= adaptation_factor.get(self.adaptation_level, 1.0)
            weights[tool_id] = max(0.1, min(weight, 1.0))
            total_weight += weights[tool_id]
        if total_weight > 0:
            weights = {k: (v / total_weight) for k, v in weights.items()}
        else:
            equal_weight = 1.0 / len(tools) if tools else 0.0
            weights = {tool_id: equal_weight for tool_id in tools}
        self.signal_weights_cache[cache_key] = weights
        self.signal_weights_cache_expiry[cache_key
            ] = now + self.signal_weights_cache_duration
        self.logger.info(
            f'Calculated signal weights based on tool effectiveness: {weights}'
            )
        return weights

    async def _calculate_comprehensive_effectiveness_score(self, tool_id:
        str, metrics: Dict[str, Any], market_regime: MarketRegime,
        timeframe: Optional[TimeFrame]=None, symbol: Optional[str]=None
        ) ->float:
        """
        Calculate a comprehensive effectiveness score for a tool based on all available metrics

        Args:
            tool_id: Tool identifier
            metrics: Tool effectiveness metrics dictionary
            market_regime: Current market regime
            timeframe: Trading timeframe
            symbol: Trading instrument symbol

        Returns:
            Effectiveness score between 0.0 and 1.0
        """
        win_rate = metrics.get('win_rate', 0.5)
        profit_factor = metrics.get('profit_factor', 1.0)
        expected_payoff = metrics.get('expected_payoff', 0.0)
        max_drawdown = metrics.get('max_drawdown', 0.0)
        recovery_factor = metrics.get('recovery_factor', 0.0)
        regime_metrics = metrics.get('regime_metrics', {}).get(market_regime
            .value, {})
        regime_win_rate = regime_metrics.get('win_rate', win_rate)
        regime_profit_factor = regime_metrics.get('profit_factor',
            profit_factor)
        regime_expected_payoff = regime_metrics.get('expected_payoff',
            expected_payoff)
        trend_factor = await self._calculate_effectiveness_trend_factor(tool_id
            , market_regime)
        win_rate_score = regime_win_rate
        pf_score = min(regime_profit_factor / 3.0, 1.0)
        ep_score = min(max(regime_expected_payoff, 0.0) / 0.01, 1.0)
        dd_penalty = max(0.0, min(abs(max_drawdown) / 0.2, 1.0))
        rf_bonus = min(recovery_factor / 3.0, 0.15)
        effectiveness_score = (0.35 * win_rate_score + 0.3 * pf_score + 0.2 *
            ep_score - 0.1 * dd_penalty + rf_bonus)
        effectiveness_score *= trend_factor
        return max(0.1, min(effectiveness_score, 1.0))

    async def _calculate_effectiveness_trend_factor(self, tool_id: str,
        market_regime: MarketRegime) ->float:
        """
        Calculate a factor based on the trend of a tool's effectiveness

        Args:
            tool_id: Tool identifier
            market_regime: Current market regime

        Returns:
            Trend factor between 0.8 and 1.2
        """
        if tool_id not in self.effectiveness_history:
            return 1.0
        history = self.effectiveness_history[tool_id]
        if len(history) < 3:
            return 1.0
        regime_history = [entry for entry in history if entry.get(
            'market_regime') == market_regime.value]
        if len(regime_history) < 3:
            regime_history = history
        sorted_history = sorted(regime_history, key=lambda x: x.get(
            'timestamp', datetime.min))
        win_rates = [entry.get('win_rate', 0.5) for entry in sorted_history
            [-5:]]
        if len(win_rates) >= 3:
            n = len(win_rates)
            x = list(range(n))
            slope = (n * sum(i * j for i, j in zip(x, win_rates)) - sum(x) *
                sum(win_rates)) / (n * sum(i * i for i in x) - sum(x) ** 2)
            trend_factor = 1.0 + slope * 5.0
            return max(0.8, min(trend_factor, 1.2))
        return 1.0

    def _get_recent_performance_factor(self, tool_id: str, market_regime:
        MarketRegime) ->float:
        """
        Get a factor based on very recent performance (last few days)

        Args:
            tool_id: Tool identifier
            market_regime: Current market regime

        Returns:
            Recent performance factor between 0.8 and 1.2
        """
        if tool_id not in self.weight_adjustment_factors:
            return 1.0
        regime_factors = self.weight_adjustment_factors.get(tool_id, {})
        return regime_factors.get(market_regime.value, 1.0)

    @async_with_exception_handling
    async def get_adaptive_parameters(self, symbol: str, timeframe: str,
        strategy_id: Optional[str]=None, context: Optional[Dict[str, Any]]=None
        ) ->Dict[str, Any]:
        """
        Get adaptive parameters for a trading strategy.

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            strategy_id: Optional strategy identifier
            context: Optional contextual information

        Returns:
            Dictionary with adaptive parameters
        """
        try:
            tf = timeframe
            if isinstance(timeframe, str):
                try:
                    tf = TimeFrame[timeframe.upper()]
                except (KeyError, AttributeError):
                    tf = timeframe
            market_regime = None
            if context and 'market_regime' in context:
                regime_str = context['market_regime']
                try:
                    market_regime = MarketRegime[regime_str.upper()]
                except (KeyError, AttributeError):
                    market_regime = None
            available_tools = context.get('available_tools', []
                ) if context else []
            signal_weights = {}
            if market_regime and available_tools:
                signal_weights = await self.get_tool_signal_weights(
                    market_regime=market_regime, tools=available_tools,
                    timeframe=tf, symbol=symbol)
            return {'symbol': symbol, 'timeframe': timeframe, 'strategy_id':
                strategy_id, 'adaptation_level': self.get_adaptation_level(
                ), 'market_regime': market_regime.value if market_regime else
                'unknown', 'signal_weights': signal_weights, 'parameters':
                {'stop_loss_pips': 20, 'take_profit_pips': 40, 'max_trades':
                3, 'risk_per_trade': 0.02}}
        except Exception as e:
            self.logger.error(f'Error getting adaptive parameters: {str(e)}')
            return {'symbol': symbol, 'timeframe': timeframe, 'strategy_id':
                strategy_id, 'adaptation_level': self.get_adaptation_level(
                ), 'parameters': {'stop_loss_pips': 20, 'take_profit_pips':
                40, 'max_trades': 3, 'risk_per_trade': 0.02},
                'signal_weights': {'technical_analysis': 0.4,
                'pattern_recognition': 0.3, 'machine_learning': 0.3}}

    @async_with_exception_handling
    async def record_strategy_performance(self, strategy_id: str, symbol:
        str, timeframe: str, performance_metrics: Dict[str, Any],
        parameters_used: Dict[str, Any]) ->bool:
        """
        Record strategy performance for adaptive learning.

        Args:
            strategy_id: Strategy identifier
            symbol: Trading symbol
            timeframe: Chart timeframe
            performance_metrics: Performance metrics
            parameters_used: Parameters used for the strategy

        Returns:
            Success flag
        """
        try:
            self.logger.info(
                f'Recording strategy performance: {strategy_id}, {symbol}, {timeframe}, metrics: {performance_metrics}'
                )
            return True
        except Exception as e:
            self.logger.error(f'Error recording strategy performance: {str(e)}'
                )
            return False

    @async_with_exception_handling
    async def get_adaptation_recommendations(self, symbol: str, timeframe:
        str, current_market_data: Dict[str, Any], strategy_id: Optional[str
        ]=None) ->Dict[str, Any]:
        """
        Get recommendations for strategy adaptation.

        Args:
            symbol: Trading symbol
            timeframe: Chart timeframe
            current_market_data: Current market data
            strategy_id: Optional strategy identifier

        Returns:
            Dictionary with adaptation recommendations
        """
        try:
            self.logger.info(
                f'Getting adaptation recommendations: {symbol}, {timeframe}, strategy_id: {strategy_id}'
                )
            return {'symbol': symbol, 'timeframe': timeframe, 'strategy_id':
                strategy_id, 'recommendations': {'stop_loss': 'no_change',
                'take_profit': 'no_change', 'entry_criteria': 'no_change'},
                'confidence': 0.5, 'status': 'not_implemented'}
        except Exception as e:
            self.logger.error(
                f'Error getting adaptation recommendations: {str(e)}')
            return {'symbol': symbol, 'timeframe': timeframe, 'strategy_id':
                strategy_id, 'recommendations': {'stop_loss': 'no_change',
                'take_profit': 'no_change', 'entry_criteria': 'no_change'},
                'confidence': 0.5, 'status': 'error', 'error': str(e)}

    @with_exception_handling
    def set_adaptation_level(self, level: Union[str, AdaptationLevelEnum]
        ) ->None:
        """
        Set the adaptation aggressiveness level.

        Args:
            level: Adaptation level
        """
        if isinstance(level, str):
            try:
                enum_level = AdaptationLevelEnum(level.lower())
                for internal_level, enum_value in self.adaptation_level_map.items(
                    ):
                    if enum_value == enum_level:
                        self.adaptation_level = internal_level
                        break
                else:
                    self.logger.warning(
                        f'Invalid adaptation level: {level}, using MODERATE')
                    self.adaptation_level = AdaptationLevel.MODERATE
            except ValueError:
                self.logger.warning(
                    f'Invalid adaptation level: {level}, using MODERATE')
                self.adaptation_level = AdaptationLevel.MODERATE
        else:
            for internal_level, enum_value in self.adaptation_level_map.items(
                ):
                if enum_value == level:
                    self.adaptation_level = internal_level
                    break
            else:
                self.logger.warning(
                    f'Invalid adaptation level enum, using MODERATE')
                self.adaptation_level = AdaptationLevel.MODERATE
        self.logger.info(
            f'Adaptation level set to: {self.adaptation_level.name}')

    def get_adaptation_level(self) ->str:
        """
        Get the current adaptation aggressiveness level.

        Returns:
            Current adaptation level
        """
        enum_level = self.adaptation_level_map.get(self.adaptation_level,
            AdaptationLevelEnum.MODERATE)
        return enum_level.value

    async def update_weight_adjustment_factor(self, tool_id: str,
        market_regime: MarketRegime, success: bool, impact: float=0.05) ->None:
        """
        Update weight adjustment factor based on very recent tool performance

        Args:
            tool_id: Tool identifier
            market_regime: Current market regime
            success: Whether the tool was successful
            impact: How much to adjust the factor by
        """
        if tool_id not in self.weight_adjustment_factors:
            self.weight_adjustment_factors[tool_id] = {}
        current_factor = self.weight_adjustment_factors[tool_id].get(
            market_regime.value, 1.0)
        if success:
            new_factor = min(current_factor + impact, 1.2)
        else:
            new_factor = max(current_factor - impact, 0.8)
        self.weight_adjustment_factors[tool_id][market_regime.value
            ] = new_factor
        self.logger.debug(
            f'Updated weight adjustment factor for {tool_id} in {market_regime.value}: {current_factor:.2f} -> {new_factor:.2f}'
            )

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
            regime = market_regime
            if isinstance(market_regime, str):
                try:
                    regime = MarketRegime[market_regime.upper()]
                except (KeyError, AttributeError):
                    regime = MarketRegime.UNKNOWN
            tf = timeframe
            if isinstance(timeframe, str):
                try:
                    tf = TimeFrame[timeframe.upper()]
                except (KeyError, AttributeError):
                    tf = None
            active_tools = await self._get_active_tools(lookback_hours)
            context = await self._build_adaptation_context(market_regime=
                regime, timeframe=tf, symbol=symbol, lookback_hours=
                lookback_hours)
            results = {}
            for tool_id in active_tools:
                try:
                    current_params = self._get_current_parameters(tool_id)
                    adjustments = self._generate_parameter_adjustments(tool_id
                        =tool_id, market_regime=regime, context=context)
                    constrained_adjustments = (self.
                        _apply_parameter_constraints(tool_id=tool_id,
                        adjustments=adjustments))
                    expected_impact = await self._calculate_expected_impact(
                        tool_id=tool_id, adjustments=
                        constrained_adjustments, context=context)
                    result = self._create_adaptation_result(tool_id=tool_id,
                        market_regime=regime.value, adjustments=
                        constrained_adjustments, expected_impact=
                        expected_impact)
                    results[tool_id] = result
                    self._update_current_parameters(tool_id=tool_id,
                        adjustments=constrained_adjustments)
                    self.adaptation_history.append(result)
                except Exception as e:
                    self.logger.error(
                        f'Error adapting tool {tool_id}: {str(e)}')
            return {'market_regime': market_regime, 'timeframe': timeframe,
                'symbol': symbol, 'adaptations_made': len(results),
                'results': results}
        except Exception as e:
            self.logger.error(f'Error running adaptation cycle: {str(e)}')
            return {'market_regime': market_regime, 'timeframe': timeframe,
                'symbol': symbol, 'adaptations_made': 0, 'status': 'error',
                'error': str(e)}

    async def get_aggregator_weights(self, market_regime: MarketRegime,
        timeframe: Optional[TimeFrame]=None, symbol: Optional[str]=None
        ) ->Dict[str, float]:
        """
        Get weights for the SignalAggregator categories based on current market regime
        and tool effectiveness metrics

        Args:
            market_regime: Current market regime
            timeframe: Trading timeframe
            symbol: Trading instrument symbol

        Returns:
            Dictionary of weights for different signal categories
        """
        regime_base_weights = {MarketRegime.TRENDING: {'technical_analysis':
            0.55, 'machine_learning': 0.25, 'market_regime': 0.15,
            'correlation': 0.05}, MarketRegime.RANGING: {
            'technical_analysis': 0.5, 'machine_learning': 0.2,
            'market_regime': 0.1, 'correlation': 0.2}, MarketRegime.
            VOLATILE: {'technical_analysis': 0.4, 'machine_learning': 0.3,
            'market_regime': 0.25, 'correlation': 0.05}, MarketRegime.
            BREAKOUT: {'technical_analysis': 0.6, 'machine_learning': 0.2,
            'market_regime': 0.15, 'correlation': 0.05}}
        base_weights = regime_base_weights.get(market_regime, {
            'technical_analysis': 0.5, 'machine_learning': 0.3,
            'market_regime': 0.1, 'correlation': 0.1})
        categories = list(base_weights.keys())
        category_metrics = {}
        for category in categories:
            category_tools = (await self.effectiveness_repository.
                get_tools_by_category_async(category))
            if not category_tools:
                continue
            effectiveness_sum = 0
            tool_count = 0
            for tool_id in category_tools:
                metrics = (await self.effectiveness_repository.
                    get_tool_effectiveness_metrics_async(tool_id=tool_id,
                    timeframe=timeframe.value if timeframe else None,
                    instrument=symbol, market_regime=market_regime))
                if metrics:
                    win_rate = metrics.get('win_rate', 0.5)
                    effectiveness_sum += win_rate
                    tool_count += 1
            if tool_count > 0:
                category_metrics[category] = effectiveness_sum / tool_count
        adjusted_weights = {}
        total_adjustment = 0
        for category, base_weight in base_weights.items():
            if category in category_metrics:
                effectiveness = category_metrics[category]
                adjustment_factor = 1.0 + (effectiveness - 0.5) * 0.5
                adjusted_weight = base_weight * adjustment_factor
            else:
                adjusted_weight = base_weight
            adjusted_weights[category] = adjusted_weight
            total_adjustment += adjusted_weight
        if total_adjustment > 0:
            normalized_weights = {k: (v / total_adjustment) for k, v in
                adjusted_weights.items()}
        else:
            normalized_weights = base_weights
        self.logger.info(
            f'Calculated aggregator weights for {market_regime.value}: {normalized_weights}'
            )
        return normalized_weights

    @async_with_exception_handling
    async def run_adaptation_cycle(self, market_regime: MarketRegime,
        timeframe: Optional[TimeFrame]=None, symbol: Optional[str]=None,
        lookback_hours: int=24) ->Dict[str, AdaptationResult]:
        """
        Run an adaptation cycle to adjust parameters based on
        current market conditions and effectiveness metrics

        Args:
            market_regime: Current market regime
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter
            lookback_hours: Hours of data to consider

        Returns:
            Dictionary mapping tool IDs to adaptation results
        """
        try:
            if self.adaptation_level == AdaptationLevel.NONE:
                self.logger.info('Adaptation is disabled (level NONE)')
                return {}
            active_tool_ids = await self._get_active_tools(lookback_hours)
            if not active_tool_ids:
                self.logger.info(
                    'No active tools found with sufficient data for adaptation'
                    )
                return {}
            context = await self._build_adaptation_context(market_regime=
                market_regime, timeframe=timeframe, symbol=symbol,
                lookback_hours=lookback_hours)
            results = {}
            for tool_id in active_tool_ids:
                if tool_id in self.adaptation_handlers:
                    current_params = self.current_parameters.get(tool_id,
                        self._get_tool_parameters(tool_id))
                    handler = self.adaptation_handlers[tool_id]
                    adjustments = handler(context, current_params)
                    adjustments = self._apply_parameter_constraints(tool_id,
                        adjustments)
                    if adjustments:
                        result = await self._create_adaptation_result(tool_id
                            =tool_id, market_regime=market_regime.value,
                            adjustments=adjustments, context=context)
                        self._apply_parameter_adjustments(tool_id, adjustments)
                        self.adaptation_history.append(result)
                        results[tool_id] = result
                        self.logger.info(
                            f'Applied {len(adjustments)} adaptations to {tool_id} in {market_regime.value} regime'
                            )
                    else:
                        self.logger.info(f'No adaptations needed for {tool_id}'
                            )
            return results
        except Exception as e:
            self.logger.error(f'Error in adaptation cycle: {str(e)}',
                exc_info=True)
            return {}

    @async_with_exception_handling
    async def _get_active_tools(self, lookback_hours: int) ->List[str]:
        """Get active tools with sufficient data for adaptation"""
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(hours=lookback_hours)
            active_tools = (await self.adaptive_layer_service.
                get_adaptation_recommendations(start_date=start_date,
                end_date=end_date, min_sample_count=self.decline_thresholds
                [self.adaptation_level]['minimum_sample_size']))
            return [t.tool_id for t in active_tools]
        except Exception as e:
            self.logger.error(f'Error getting active tools: {str(e)}')
            return []

    async def _build_adaptation_context(self, market_regime: MarketRegime,
        timeframe: Optional[TimeFrame]=None, symbol: Optional[str]=None,
        lookback_hours: int=24) ->AdaptationContext:
        """Build context for adaptation decisions"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=lookback_hours)
        previous_regime_data = (await self.effectiveness_repository.
            get_previous_regime_async(current_regime=market_regime.value,
            before_date=end_date, max_lookback_hours=lookback_hours * 2))
        previous_regime = previous_regime_data.get('regime'
            ) if previous_regime_data else None
        trading_session = self._determine_trading_session(datetime.utcnow())
        time_in_regime_minutes = previous_regime_data.get(
            'minutes_since_change', 0) if previous_regime_data else 0
        effectiveness_trends = {}
        active_tool_ids = await self._get_active_tools(lookback_hours)
        for tool_id in active_tool_ids:
            metrics = (await self.effectiveness_repository.
                get_tool_metrics_async(tool_id=tool_id, metric_type=
                'composite_score', start_date=start_date, end_date=end_date,
                order_by_date=True))
            trend = [m.value for m in metrics if m.value is not None]
            effectiveness_trends[tool_id] = trend
        market_metrics = (await self.effectiveness_repository.
            get_market_metrics_async(start_date=start_date, end_date=
            end_date, symbol=symbol))
        volatility_percentile = market_metrics.get('volatility_percentile', 0.5
            )
        liquidity_score = market_metrics.get('liquidity_score', 0.5)
        sample_size_sufficient = True
        min_samples = self.decline_thresholds[self.adaptation_level][
            'minimum_sample_size']
        for tool_id in active_tool_ids:
            count = (await self.effectiveness_repository.
                count_tool_outcomes_async(tool_id=tool_id, start_date=
                start_date, end_date=end_date))
            if count < min_samples:
                sample_size_sufficient = False
                break
        confidence_base = {AdaptationLevel.CONSERVATIVE: 0.7,
            AdaptationLevel.MODERATE: 0.8, AdaptationLevel.AGGRESSIVE: 0.85,
            AdaptationLevel.EXPERIMENTAL: 0.6}
        confidence_level = confidence_base[self.adaptation_level]
        if not sample_size_sufficient:
            confidence_level *= 0.7
        context = AdaptationContext(current_market_regime=market_regime.
            value, previous_market_regime=previous_regime,
            regime_change_detected=previous_regime != market_regime.value,
            current_volatility_percentile=volatility_percentile,
            current_liquidity_score=liquidity_score, trading_session=
            trading_session, time_in_regime=time_in_regime_minutes,
            effectiveness_trends=effectiveness_trends,
            sample_size_sufficient=sample_size_sufficient, confidence_level
            =confidence_level)
        return context

    def _determine_trading_session(self, timestamp: datetime) ->str:
        """Determine the current trading session based on UTC time"""
        hour = timestamp.hour
        if 0 <= hour < 8:
            return 'asian'
        elif 8 <= hour < 12:
            return 'asian_european_overlap'
        elif 12 <= hour < 16:
            return 'european'
        elif 16 <= hour < 20:
            return 'european_american_overlap'
        else:
            return 'american'

    def _get_tool_parameters(self, tool_id: str) ->Dict[str, Any]:
        """Get current parameters for a tool (fetching from database)"""
        return self.current_parameters.get(tool_id, {})

    def _apply_parameter_constraints(self, tool_id: str, adjustments: List[
        ParameterAdjustment]) ->List[ParameterAdjustment]:
        """Apply constraints to parameter adjustments"""
        if tool_id not in self.parameter_constraints:
            return adjustments
        constraints = self.parameter_constraints[tool_id]
        valid_adjustments = []
        for adj in adjustments:
            param_name = adj.parameter_name
            if param_name in constraints:
                constraint = constraints[param_name]
                new_value = adj.new_value
                if 'min' in constraint and new_value < constraint['min']:
                    new_value = constraint['min']
                if 'max' in constraint and new_value > constraint['max']:
                    new_value = constraint['max']
                if 'step' in constraint and constraint['step'] > 0:
                    step = constraint['step']
                    new_value = round(new_value / step) * step
                if 'values' in constraint and new_value not in constraint[
                    'values']:
                    allowed_values = constraint['values']
                    new_value = min(allowed_values, key=lambda x: abs(x -
                        new_value))
                if new_value != adj.new_value:
                    adj.new_value = new_value
                    adj.adjustment_reason += ' (constrained)'
            valid_adjustments.append(adj)
        return valid_adjustments

    def _apply_parameter_adjustments(self, tool_id: str, adjustments: List[
        ParameterAdjustment]) ->None:
        """Apply parameter adjustments to tool"""
        if tool_id not in self.current_parameters:
            self.current_parameters[tool_id] = {}
        for adj in adjustments:
            param_name = adj.parameter_name
            new_value = adj.new_value
            self.current_parameters[tool_id][param_name] = new_value

    async def _create_adaptation_result(self, tool_id: str, market_regime:
        str, adjustments: List[ParameterAdjustment], context: AdaptationContext
        ) ->AdaptationResult:
        """Create result object for an adaptation"""
        adaptation_id = (
            f'{tool_id}_{market_regime}_{datetime.utcnow().isoformat()}')
        is_reversion = any('revert' in adj.adjustment_reason.lower() for
            adj in adjustments)
        expected_impact = await self._calculate_expected_impact(tool_id,
            adjustments, context)
        return AdaptationResult(adaptation_id=adaptation_id, timestamp=
            datetime.utcnow(), market_regime=market_regime,
            adaptations_applied=adjustments, expected_impact=
            expected_impact, tools_affected=[tool_id], is_reversion=
            is_reversion, metadata={'adaptation_level': self.
            adaptation_level.name, 'context': context.dict()})

    async def _calculate_expected_impact(self, tool_id: str, adjustments:
        List[ParameterAdjustment], context: AdaptationContext) ->Dict[str,
        float]:
        """Calculate expected impact of adaptations on key metrics"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(hours=24)
        baseline_metrics = self.enhanced_tracker.get_tool_effectiveness(tool_id
            =tool_id, start_date=start_date, end_date=end_date)
        expected_impact = {}
        total_adjustment_magnitude = sum(abs((adj.new_value - adj.
            previous_value) / max(adj.previous_value, 1)) for adj in
            adjustments if isinstance(adj.previous_value, (int, float)) and
            isinstance(adj.new_value, (int, float)))
        improvement_factor = {AdaptationLevel.CONSERVATIVE: 0.02,
            AdaptationLevel.MODERATE: 0.05, AdaptationLevel.AGGRESSIVE: 0.1,
            AdaptationLevel.EXPERIMENTAL: 0.15}[self.adaptation_level]
        improvement_factor *= context.confidence_level
        if 'win_rate' in baseline_metrics:
            base_win_rate = baseline_metrics['win_rate']
            max_possible_improvement = min(0.95 - base_win_rate, 0.2)
            expected_impact['win_rate'] = min(improvement_factor *
                total_adjustment_magnitude, max_possible_improvement)
        if 'profit_factor' in baseline_metrics:
            base_pf = baseline_metrics['profit_factor']
            expected_impact['profit_factor'
                ] = base_pf * improvement_factor * total_adjustment_magnitude
        if 'expected_payoff' in baseline_metrics:
            base_payoff = baseline_metrics['expected_payoff']
            expected_impact['expected_payoff'
                ] = base_payoff * improvement_factor * total_adjustment_magnitude
        return expected_impact

    @async_with_exception_handling
    async def detect_effectiveness_decline(self, tool_id: str,
        lookback_hours: int=24, timeframe: Optional[TimeFrame]=None, symbol:
        Optional[str]=None) ->Dict[str, Any]:
        """
        Detect if a tool's effectiveness is declining

        Args:
            tool_id: ID of tool to check
            lookback_hours: Hours to look back for comparison
            timeframe: Optional timeframe filter
            symbol: Optional symbol filter

        Returns:
            Dictionary with decline detection results
        """
        try:
            end_date = datetime.utcnow()
            start_date = end_date - timedelta(hours=lookback_hours)
            thresholds = self.decline_thresholds[self.adaptation_level]
            recent_metrics = (await self.effectiveness_repository.
                get_tool_metrics_async(tool_id=tool_id, start_date=
                start_date, end_date=end_date, order_by_date=True))
            if len(recent_metrics) < thresholds['lookback_periods'] + 1:
                return {'tool_id': tool_id, 'decline_detected': False,
                    'reason': 'Insufficient data for comparison'}
            metrics_by_type = {}
            for metric in recent_metrics:
                if metric.metric_type not in metrics_by_type:
                    metrics_by_type[metric.metric_type] = []
                metrics_by_type[metric.metric_type].append(metric)
            decline_detected = False
            decline_details = {}
            if 'win_rate' in metrics_by_type:
                win_rate_metrics = metrics_by_type['win_rate']
                if len(win_rate_metrics) >= 2:
                    current = win_rate_metrics[-1].value
                    previous_avg = sum(m.value for m in win_rate_metrics[-1 -
                        thresholds['lookback_periods']:-1]) / thresholds[
                        'lookback_periods']
                    if current < previous_avg * (1 - thresholds[
                        'win_rate_decline']):
                        decline_detected = True
                        decline_details['win_rate'] = {'current': current,
                            'previous_avg': previous_avg, 'decline_pct': (
                            previous_avg - current) / previous_avg}
            if 'profit_factor' in metrics_by_type:
                pf_metrics = metrics_by_type['profit_factor']
                if len(pf_metrics) >= 2:
                    current = pf_metrics[-1].value
                    previous_avg = sum(m.value for m in pf_metrics[-1 -
                        thresholds['lookback_periods']:-1]) / thresholds[
                        'lookback_periods']
                    if current < previous_avg * (1 - thresholds[
                        'profit_factor_decline']):
                        decline_detected = True
                        decline_details['profit_factor'] = {'current':
                            current, 'previous_avg': previous_avg,
                            'decline_pct': (previous_avg - current) /
                            previous_avg}
            return {'tool_id': tool_id, 'decline_detected':
                decline_detected, 'details': decline_details,
                'adaptation_level': self.adaptation_level.name,
                'thresholds': thresholds}
        except Exception as e:
            self.logger.error(
                f'Error detecting effectiveness decline: {str(e)}')
            return {'tool_id': tool_id, 'decline_detected': False, 'error':
                str(e)}

    async def get_optimal_parameters(self, tool_id: str, market_regime:
        MarketRegime, context: Optional[Dict[str, Any]]=None) ->Dict[str, Any]:
        """
        Get optimal parameters for a tool in the given market regime

        Args:
            tool_id: ID of the tool
            market_regime: Current market regime
            context: Additional context (volatility, etc.)

        Returns:
            Dictionary of optimal parameters
        """
        regime_key = market_regime.value
        if (regime_key in self.regime_optimal_parameters and tool_id in
            self.regime_optimal_parameters[regime_key]):
            return self.regime_optimal_parameters[regime_key][tool_id]
        return self.current_parameters.get(tool_id, {})

    async def record_adaptation_feedback(self, adaptation_id: str, metrics:
        Dict[str, float], success: bool, notes: str='') ->None:
        """
        Record feedback on an adaptation's actual impact

        Args:
            adaptation_id: ID of the adaptation
            metrics: Actual metrics observed after adaptation
            success: Whether the adaptation was successful
            notes: Additional notes
        """
        for adaptation in self.adaptation_history:
            if adaptation.adaptation_id == adaptation_id:
                adaptation.metadata['feedback'] = {'actual_metrics':
                    metrics, 'success': success, 'notes': notes,
                    'timestamp': datetime.utcnow().isoformat()}
                if success and adaptation.metadata.get('context', {}).get(
                    'confidence_level', 0) > 0.7:
                    tool_id = adaptation.tools_affected[0]
                    regime = adaptation.market_regime
                    parameter_updates = {adj.parameter_name: adj.new_value for
                        adj in adaptation.adaptations_applied}
                    if regime not in self.regime_optimal_parameters:
                        self.regime_optimal_parameters[regime] = {}
                    if tool_id not in self.regime_optimal_parameters[regime]:
                        self.regime_optimal_parameters[regime][tool_id] = {}
                    self.regime_optimal_parameters[regime][tool_id].update(
                        parameter_updates)
                    self.logger.info(
                        f'Updated optimal parameters for {tool_id} in {regime} regime based on successful adaptation feedback'
                        )
                self.logger.info(
                    f'Recorded feedback for adaptation {adaptation_id}')
                break
