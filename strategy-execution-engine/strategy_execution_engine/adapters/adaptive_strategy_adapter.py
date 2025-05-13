"""
Adaptive Strategy Adapter Module

This module provides adapter implementations for adaptive strategy interfaces,
helping to break circular dependencies between services.
"""
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import logging
import asyncio
import json
import copy
from common_lib.adaptive.interfaces import IAdaptiveStrategyService, AdaptationLevelEnum
from core_foundations.utils.logger import get_logger
from strategy_execution_engine.adaptive_layer.adaptive_service import AdaptiveService
from strategy_execution_engine.models.market_regime import MarketRegime
from strategy_execution_engine.models.timeframe import TimeFrame
logger = get_logger(__name__)


from strategy_execution_engine.error.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class AdaptiveStrategyServiceAdapter(IAdaptiveStrategyService):
    """
    Adapter for adaptive strategy service that implements the common interface.
    
    This adapter can either wrap an actual service instance or provide
    standalone functionality to avoid circular dependencies.
    """

    def __init__(self, service_instance=None):
        """
        Initialize the adapter.
        
        Args:
            service_instance: Optional actual service instance to wrap
        """
        self.service = service_instance or AdaptiveService()
        self.adaptation_level = AdaptationLevelEnum.MODERATE
        self.logger = logger

    @async_with_exception_handling
    async def get_adaptive_parameters(self, symbol: str, timeframe: str,
        strategy_id: Optional[str]=None, context: Optional[Dict[str, Any]]=None
        ) ->Dict[str, Any]:
        """Get adaptive parameters for a trading strategy."""
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
                signal_weights = await self.service.get_tool_signal_weights(
                    market_regime=market_regime, tools=available_tools,
                    timeframe=tf, symbol=symbol)
            return {'symbol': symbol, 'timeframe': timeframe, 'strategy_id':
                strategy_id, 'adaptation_level': self.adaptation_level.
                value, 'market_regime': market_regime.value if
                market_regime else 'unknown', 'signal_weights':
                signal_weights, 'parameters': {'stop_loss_pips': 20,
                'take_profit_pips': 40, 'max_trades': 3, 'risk_per_trade': 
                0.02}}
        except Exception as e:
            logger.error(f'Error getting adaptive parameters: {str(e)}')
            return {'symbol': symbol, 'timeframe': timeframe, 'strategy_id':
                strategy_id, 'adaptation_level': self.adaptation_level.
                value, 'parameters': {'stop_loss_pips': 20,
                'take_profit_pips': 40, 'max_trades': 3, 'risk_per_trade': 
                0.02}, 'signal_weights': {'technical_analysis': 0.4,
                'pattern_recognition': 0.3, 'machine_learning': 0.3}}

    @async_with_exception_handling
    async def record_strategy_performance(self, strategy_id: str, symbol:
        str, timeframe: str, performance_metrics: Dict[str, Any],
        parameters_used: Dict[str, Any]) ->bool:
        """Record strategy performance for adaptive learning."""
        try:
            logger.info(
                f'Strategy performance recorded: {strategy_id}, {symbol}, {timeframe}, {performance_metrics}'
                )
            return True
        except Exception as e:
            logger.error(f'Error recording strategy performance: {str(e)}')
            return False

    @async_with_exception_handling
    async def get_tool_signal_weights(self, market_regime: str, tools: List
        [str], timeframe: Optional[str]=None, symbol: Optional[str]=None
        ) ->Dict[str, float]:
        """Get signal weights for tools based on their effectiveness."""
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
            return await self.service.get_tool_signal_weights(market_regime
                =regime, tools=tools, timeframe=tf, symbol=symbol)
        except Exception as e:
            logger.error(f'Error getting tool signal weights: {str(e)}')
            weight = 1.0 / len(tools) if tools else 0.0
            return {tool_id: weight for tool_id in tools}

    @async_with_exception_handling
    async def run_adaptation_cycle(self, market_regime: str, timeframe:
        Optional[str]=None, symbol: Optional[str]=None, lookback_hours: int=24
        ) ->Dict[str, Any]:
        """Run an adaptation cycle to adjust parameters based on current conditions."""
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
            return await self.service.run_adaptation_cycle(market_regime=
                regime, timeframe=tf, symbol=symbol, lookback_hours=
                lookback_hours)
        except Exception as e:
            logger.error(f'Error running adaptation cycle: {str(e)}')
            return {'market_regime': market_regime, 'timeframe': timeframe,
                'symbol': symbol, 'adaptations_made': 0, 'status': 'error',
                'error': str(e)}

    @async_with_exception_handling
    async def get_adaptation_recommendations(self, symbol: str, timeframe:
        str, current_market_data: Dict[str, Any], strategy_id: Optional[str
        ]=None) ->Dict[str, Any]:
        """Get recommendations for strategy adaptation."""
        try:
            return {'symbol': symbol, 'timeframe': timeframe, 'strategy_id':
                strategy_id, 'recommendations': {'stop_loss': 'no_change',
                'take_profit': 'no_change', 'entry_criteria': 'no_change'},
                'confidence': 0.5, 'status': 'not_implemented'}
        except Exception as e:
            logger.error(f'Error getting adaptation recommendations: {str(e)}')
            return {'symbol': symbol, 'timeframe': timeframe, 'strategy_id':
                strategy_id, 'recommendations': {'stop_loss': 'no_change',
                'take_profit': 'no_change', 'entry_criteria': 'no_change'},
                'confidence': 0.5, 'status': 'error', 'error': str(e)}

    @with_exception_handling
    def set_adaptation_level(self, level: Union[str, AdaptationLevelEnum]
        ) ->None:
        """Set the adaptation aggressiveness level."""
        if isinstance(level, str):
            try:
                self.adaptation_level = AdaptationLevelEnum(level.lower())
            except ValueError:
                logger.warning(
                    f'Invalid adaptation level: {level}, using MODERATE')
                self.adaptation_level = AdaptationLevelEnum.MODERATE
        else:
            self.adaptation_level = level
        if hasattr(self.service, 'adaptation_level'):
            try:
                from strategy_execution_engine.adaptive_layer.adaptive_service import AdaptationLevel
                service_level = AdaptationLevel[self.adaptation_level.name]
                self.service.adaptation_level = service_level
            except Exception as e:
                logger.error(f'Error setting adaptation level: {str(e)}')

    @with_exception_handling
    def get_adaptation_level(self) ->str:
        """Get the current adaptation aggressiveness level."""
        if hasattr(self.service, 'adaptation_level'):
            try:
                return self.service.adaptation_level.name.lower()
            except Exception as e:
                logger.error(f'Error getting adaptation level: {str(e)}')
        return self.adaptation_level.value
