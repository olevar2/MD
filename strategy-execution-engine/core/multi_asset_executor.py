"""
Multi-Asset Strategy Executor

This module implements the executor for multi-asset strategies,
providing a unified interface for running strategies across different asset classes.
"""
import logging
import asyncio
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from core.base_strategy import BaseStrategy
from core.asset_strategy_factory import AssetStrategyFactory, AssetClass
from services.analysis_integration_service import AnalysisIntegrationService
from core.signal_aggregator import SignalAggregator
from strategy_execution_engine.signals.signal_models import SignalStrength
from strategy_execution_engine.signals.signal_models import TradeDirection
from services.adaptive_service import AdaptiveLayerService


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class MultiAssetStrategyExecutor:
    """
    Executor for multi-asset strategies.
    
    This class is responsible for executing strategies across different asset classes,
    handling the specific requirements and adaptations needed for each asset type.
    """

    def __init__(self, strategy_factory: AssetStrategyFactory,
        analysis_service: AnalysisIntegrationService, adaptive_layer:
        AdaptiveLayerService, config: Dict[str, Any]=None):
        """
        Initialize the multi-asset strategy executor.
        
        Args:
            strategy_factory: Factory for creating asset-specific strategies
            analysis_service: Service for integrated analysis
            adaptive_layer: Service for adaptive parameters
            config: Configuration for the executor
        """
        self.strategy_factory = strategy_factory
        self.analysis_service = analysis_service
        self.adaptive_layer = adaptive_layer
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        self.active_strategies = {}

    @async_with_exception_handling
    async def execute_strategy(self, symbol: str, asset_class: str,
        timeframes: List[str]=None, strategy_name: Optional[str]=None) ->Dict[
        str, Any]:
        """
        Execute a strategy for the specified symbol and asset class.
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD", "BTCUSD", "AAPL")
            asset_class: Asset class (forex, crypto, stock, etc.)
            timeframes: List of timeframes to analyze (defaults to strategy default)
            strategy_name: Optional strategy name override
            
        Returns:
            Strategy execution results including signals, orders, and explanations
        """
        try:
            normalized_asset_class = asset_class.lower()
            asset_enum = AssetClass(normalized_asset_class)
        except ValueError:
            self.logger.warning(
                f'Unknown asset class: {asset_class}, defaulting to forex')
            normalized_asset_class = 'forex'
            asset_enum = AssetClass.FOREX
        strategy_key = (
            f"{symbol}_{normalized_asset_class}_{strategy_name or 'default'}")
        if strategy_key not in self.active_strategies:
            strategy = self.strategy_factory.get_strategy_for_asset(symbol=
                symbol, asset_class=normalized_asset_class, strategy_name=
                strategy_name)
            self.active_strategies[strategy_key] = strategy
        else:
            strategy = self.active_strategies[strategy_key]
        if not timeframes:
            timeframes = strategy.get_default_timeframes()
        timeframe_signals = {}
        timeframe_analysis = {}
        for tf in timeframes:
            analysis_results = (await self.analysis_service.
                get_integrated_signals(symbol=symbol, timeframe=tf,
                lookback_bars=strategy.get_required_history_bars()))
            timeframe_signals[tf] = analysis_results.get('aggregated_signal',
                {})
            timeframe_analysis[tf] = analysis_results.get('components', {})
        market_regime = timeframe_analysis.get(timeframes[0], {}).get(
            'market_regime', {}).get('regime', 'unknown')
        await self._apply_adaptive_parameters(strategy, symbol, market_regime)
        strategy_result = await strategy.execute(signals=timeframe_signals,
            analysis_data=timeframe_analysis)
        self.logger.info(
            f"Strategy {strategy_name or 'default'} for {symbol} ({asset_class}) executed: {strategy_result.get('decision', 'NEUTRAL')}"
            )
        enhanced_result = self._enhance_result_with_asset_info(strategy_result,
            symbol, normalized_asset_class)
        self._update_adaptive_layer(enhanced_result, symbol,
            normalized_asset_class, market_regime)
        return enhanced_result

    @async_with_exception_handling
    async def _apply_adaptive_parameters(self, strategy: BaseStrategy,
        symbol: str, market_regime: str) ->None:
        """
        Apply adaptive parameters to strategy from the adaptive layer.
        
        Args:
            strategy: The strategy to update
            symbol: Trading symbol
            market_regime: Current market regime
        """
        try:
            adaptive_params = (await self.adaptive_layer.
                get_adaptive_parameters(symbol=symbol, strategy_name=
                strategy.get_name(), market_regime=market_regime))
            if adaptive_params:
                strategy.apply_adaptive_parameters(adaptive_params)
                self.logger.debug(
                    f'Applied adaptive parameters to {symbol}: {adaptive_params}'
                    )
        except Exception as e:
            self.logger.error(f'Error applying adaptive parameters: {str(e)}')

    def _enhance_result_with_asset_info(self, result: Dict[str, Any],
        symbol: str, asset_class: str) ->Dict[str, Any]:
        """
        Enhance strategy result with asset-specific information.
        
        Args:
            result: Strategy execution result
            symbol: Trading symbol
            asset_class: Asset class
            
        Returns:
            Enhanced result with asset-specific details
        """
        enhanced = result.copy()
        enhanced['asset_class'] = asset_class
        enhanced['execution_timestamp'] = datetime.now().isoformat()
        if 'order' in enhanced and enhanced['order']:
            strategy = self.active_strategies.get(
                f'{symbol}_{asset_class}_default')
            if strategy:
                position_info = strategy.calculate_position_size(direction=
                    enhanced['order'].get('direction', 'NEUTRAL'),
                    risk_percent=self.config.get('default_risk_percent', 
                    1.0), entry_price=enhanced['order'].get('entry_price'))
                enhanced['order']['position_size'] = position_info.get(
                    'position_size')
                enhanced['order']['risk_amount'] = position_info.get(
                    'risk_amount')
                enhanced['order']['position_value'] = position_info.get(
                    'position_value')
                enhanced['order']['asset_specific'] = position_info.get(
                    'asset_specific', {})
                enhanced['position_explanation'] = position_info.get(
                    'explanation', '')
        return enhanced

    @with_exception_handling
    def _update_adaptive_layer(self, result: Dict[str, Any], symbol: str,
        asset_class: str, market_regime: str) ->None:
        """
        Update adaptive layer with execution feedback for learning.
        
        Args:
            result: Strategy execution result
            symbol: Trading symbol
            asset_class: Asset class
            market_regime: Current market regime
        """
        try:
            feedback = {'symbol': symbol, 'asset_class': asset_class,
                'market_regime': market_regime, 'timestamp': datetime.now()
                .isoformat(), 'strategy_name': result.get('strategy_name',
                'unknown'), 'decision': result.get('decision', 'NEUTRAL'),
                'confidence': result.get('confidence', 0.0), 'signals_used':
                result.get('signals_used', []), 'executed': 'order' in
                result and result['order'] is not None}
            asyncio.create_task(self.adaptive_layer.
                record_execution_feedback(feedback))
        except Exception as e:
            self.logger.error(f'Error updating adaptive layer: {str(e)}')

    @async_with_exception_handling
    async def execute_batch(self, symbols: Dict[str, str]) ->Dict[str, Dict
        [str, Any]]:
        """
        Execute strategies for multiple symbols in batch.
        
        Args:
            symbols: Dictionary mapping symbols to their asset classes
            
        Returns:
            Dictionary of execution results by symbol
        """
        tasks = {}
        for symbol, asset_class in symbols.items():
            tasks[symbol] = self.execute_strategy(symbol=symbol,
                asset_class=asset_class)
        results = {}
        for symbol, task in tasks.items():
            try:
                results[symbol] = await task
            except Exception as e:
                self.logger.error(
                    f'Error executing strategy for {symbol}: {str(e)}')
                results[symbol] = {'error': str(e), 'symbol': symbol}
        return results

    async def get_strategy_performance(self, symbol: str, asset_class: str,
        strategy_name: Optional[str]=None) ->Dict[str, Any]:
        """
        Get performance metrics for a specific strategy.
        
        Args:
            symbol: Trading symbol
            asset_class: Asset class
            strategy_name: Optional strategy name
            
        Returns:
            Performance metrics for the strategy
        """
        strategy_key = (
            f"{symbol}_{asset_class.lower()}_{strategy_name or 'default'}")
        if strategy_key not in self.active_strategies:
            self.logger.warning(f'Strategy {strategy_key} not found')
            return {'error': 'Strategy not found'}
        strategy = self.active_strategies[strategy_key]
        return await strategy.get_performance_metrics()
