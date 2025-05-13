"""
Asset Strategy Factory Module

This module provides a factory for creating asset-specific strategy implementations.
It allows for different trading strategies to be customized based on asset class.
"""
import logging
from typing import Dict, Any, Optional, Type, List
import json
import os
from enum import Enum
from core.base_strategy import BaseStrategy
from core.multi_timeframe_confluence_strategy import MultiTimeframeConfluenceStrategy


from core.exceptions_bridge import (
    with_exception_handling,
    async_with_exception_handling,
    ForexTradingPlatformError,
    ServiceError,
    DataError,
    ValidationError
)

class AssetClass(Enum):
    """Supported asset classes"""
    FOREX = 'forex'
    CRYPTO = 'crypto'
    STOCK = 'stock'
    COMMODITY = 'commodity'
    INDEX = 'index'
    BOND = 'bond'
    ETF = 'etf'


class AssetStrategyFactory:
    """
    Factory for creating asset-specific strategy implementations.
    
    This factory manages the creation and configuration of strategies
    tailored to different asset classes.
    """

    def __init__(self, config_path: Optional[str]=None):
        """
        Initialize the asset strategy factory.
        
        Args:
            config_path: Optional path to strategy configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.strategies = {}
        self.asset_configs = {}
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        else:
            self._load_default_configs()

    @with_exception_handling
    def _load_config(self, config_path: str) ->None:
        """Load strategy configuration from file"""
        try:
            with open(config_path, 'r') as f:
                self.asset_configs = json.load(f)
                self.logger.info(
                    f'Loaded asset strategy configurations from {config_path}')
        except Exception as e:
            self.logger.error(
                f'Failed to load strategy config from {config_path}: {str(e)}')
            self._load_default_configs()

    def _load_default_configs(self) ->None:
        """Load default strategy configurations"""
        self.asset_configs = {AssetClass.FOREX.value: {'default_strategy':
            'MultiTimeframeConfluenceStrategy', 'parameters': {
            'confirmation_threshold': 2, 'min_confluence_score': 0.7,
            'volatility_multiplier': 1.0, 'use_atr_for_stops': True,
            'atr_stop_multiplier': 1.5, 'timeframe_weights': {'M15': 0.7,
            'H1': 1.0, 'H4': 1.2, 'D1': 1.3}}}, AssetClass.CRYPTO.value: {
            'default_strategy': 'MultiTimeframeConfluenceStrategy',
            'parameters': {'confirmation_threshold': 3,
            'min_confluence_score': 0.8, 'volatility_multiplier': 1.5,
            'use_atr_for_stops': True, 'atr_stop_multiplier': 2.0,
            'timeframe_weights': {'M15': 0.6, 'H1': 0.8, 'H4': 1.0, 'D1': 
            1.2}}}, AssetClass.STOCK.value: {'default_strategy':
            'MultiTimeframeConfluenceStrategy', 'parameters': {
            'confirmation_threshold': 2, 'min_confluence_score': 0.65,
            'volatility_multiplier': 0.8, 'use_atr_for_stops': True,
            'atr_stop_multiplier': 1.3, 'timeframe_weights': {'H1': 0.7,
            'H4': 0.9, 'D1': 1.2, 'W1': 1.5}}}, AssetClass.COMMODITY.value:
            {'default_strategy': 'MultiTimeframeConfluenceStrategy',
            'parameters': {'confirmation_threshold': 2,
            'min_confluence_score': 0.7, 'volatility_multiplier': 1.2,
            'use_atr_for_stops': True, 'atr_stop_multiplier': 1.6,
            'timeframe_weights': {'H1': 0.8, 'H4': 1.0, 'D1': 1.3, 'W1': 
            1.4}}}, AssetClass.INDEX.value: {'default_strategy':
            'MultiTimeframeConfluenceStrategy', 'parameters': {
            'confirmation_threshold': 2, 'min_confluence_score': 0.65,
            'volatility_multiplier': 0.9, 'use_atr_for_stops': True,
            'atr_stop_multiplier': 1.4, 'timeframe_weights': {'H1': 0.8,
            'H4': 1.0, 'D1': 1.3, 'W1': 1.4}}}}
        self.logger.info('Loaded default asset strategy configurations')

    def get_strategy_for_asset(self, symbol: str, asset_class: str,
        strategy_name: Optional[str]=None) ->BaseStrategy:
        """
        Get a strategy implementation tailored for the specific asset.
        
        Args:
            symbol: Trading symbol
            asset_class: Asset class (forex, crypto, etc.)
            strategy_name: Optional strategy name to use, otherwise uses default for asset class
            
        Returns:
            An asset-specific strategy implementation
        """
        asset_class = asset_class.lower()
        if asset_class not in [ac.value for ac in AssetClass]:
            self.logger.warning(
                f'Unknown asset class: {asset_class}, defaulting to forex')
            asset_class = AssetClass.FOREX.value
        asset_config = self.asset_configs.get(asset_class, self.
            asset_configs.get(AssetClass.FOREX.value))
        if not strategy_name:
            strategy_name = asset_config.get('default_strategy',
                'MultiTimeframeConfluenceStrategy')
        params = asset_config_manager.get('parameters', {})
        if strategy_name == 'MultiTimeframeConfluenceStrategy':
            strategy = MultiTimeframeConfluenceStrategy(symbol=symbol,
                config=self._create_strategy_config(params))
            strategy.set_asset_class(asset_class)
            self._customize_strategy_for_asset(strategy, asset_class, params)
            return strategy
        else:
            self.logger.warning(
                f'Unknown strategy: {strategy_name}, using MultiTimeframeConfluenceStrategy'
                )
            strategy = MultiTimeframeConfluenceStrategy(symbol=symbol,
                config=self._create_strategy_config(params))
            strategy.set_asset_class(asset_class)
            self._customize_strategy_for_asset(strategy, asset_class, params)
            return strategy

    def _create_strategy_config(self, params: Dict[str, Any]) ->Dict[str, Any]:
        """Create a strategy configuration from parameters"""
        config = {'timeframes': ['M15', 'H1', 'H4', 'D1'], 'indicators': {
            'primary': ['ema', 'rsi', 'macd'], 'secondary': ['bollinger',
            'atr', 'ichimoku']}, 'confirmation_threshold': params.get(
            'confirmation_threshold', 2), 'min_confluence_score': params.
            get('min_confluence_score', 0.7), 'volatility_multiplier':
            params.get('volatility_multiplier', 1.0)}
        if params.get('use_atr_for_stops', True):
            config['stop_loss'] = {'method': 'atr', 'multiplier': params.
                get('atr_stop_multiplier', 1.5)}
        else:
            config['stop_loss'] = {'method': 'swing', 'buffer_pips': 10}
        if 'timeframe_weights' in params:
            config['timeframe_weights'] = params['timeframe_weights']
        return config

    def _customize_strategy_for_asset(self, strategy: BaseStrategy,
        asset_class: str, params: Dict[str, Any]) ->None:
        """Apply asset-specific customizations to the strategy"""
        if asset_class == AssetClass.FOREX.value:
            self._customize_forex_strategy(strategy, params)
        elif asset_class == AssetClass.CRYPTO.value:
            self._customize_crypto_strategy(strategy, params)
        elif asset_class == AssetClass.STOCK.value:
            self._customize_stock_strategy(strategy, params)
        elif asset_class == AssetClass.COMMODITY.value:
            self._customize_commodity_strategy(strategy, params)
        elif asset_class == AssetClass.INDEX.value:
            self._customize_index_strategy(strategy, params)

    def _customize_forex_strategy(self, strategy: BaseStrategy, params:
        Dict[str, Any]) ->None:
        """Customize strategy for forex assets"""
        if hasattr(strategy, 'set_price_precision'):
            strategy.set_price_precision(5)
        if hasattr(strategy, 'set_session_importance'):
            strategy.set_session_importance({'asian': 0.8, 'london': 1.0,
                'new_york': 1.0, 'overlap': 1.2})

    def _customize_crypto_strategy(self, strategy: BaseStrategy, params:
        Dict[str, Any]) ->None:
        """Customize strategy for crypto assets"""
        if hasattr(strategy, 'set_price_precision'):
            strategy.set_price_precision(8)
        if hasattr(strategy, 'set_volatility_handling'):
            strategy.set_volatility_handling({
                'increase_filter_strength_in_high_volatility': True,
                'dynamic_stop_adjustment': True,
                'volatility_threshold_multiplier': 1.5})

    def _customize_stock_strategy(self, strategy: BaseStrategy, params:
        Dict[str, Any]) ->None:
        """Customize strategy for stock assets"""
        if hasattr(strategy, 'set_session_importance'):
            strategy.set_session_importance({'pre_market': 0.7,
                'regular_hours': 1.0, 'post_market': 0.6})
        if hasattr(strategy, 'set_additional_filters'):
            strategy.set_additional_filters(['earnings_calendar',
                'sector_correlation', 'index_relative_strength'])

    def _customize_commodity_strategy(self, strategy: BaseStrategy, params:
        Dict[str, Any]) ->None:
        """Customize strategy for commodity assets"""
        if hasattr(strategy, 'set_session_importance'):
            strategy.set_session_importance({'asian': 0.8, 'london': 0.9,
                'new_york': 1.0})
        if hasattr(strategy, 'set_seasonal_analysis'):
            strategy.set_seasonal_analysis(True)

    def _customize_index_strategy(self, strategy: BaseStrategy, params:
        Dict[str, Any]) ->None:
        """Customize strategy for index assets"""
        if hasattr(strategy, 'set_session_importance'):
            strategy.set_session_importance({'pre_market': 0.6,
                'regular_hours': 1.0, 'post_market': 0.5})
        if hasattr(strategy, 'set_correlation_assets'):
            strategy.set_correlation_assets(['major_stocks', 'sector_etfs',
                'vix'])
