"""
Asset Strategy Framework

This module provides the framework for implementing and managing asset-specific
trading strategies that integrate with all analysis components.
"""
import logging
from typing import Dict, List, Any, Optional, Union
from abc import ABC, abstractmethod
from enum import Enum
import asyncio
from datetime import datetime
from analysis_engine.multi_asset.asset_registry import AssetClass, AssetRegistry
from analysis_engine.integration.analysis_integration_service import AnalysisIntegrationService
from analysis_engine.models.market_data import MarketData
logger = logging.getLogger(__name__)
from analysis_engine.core.exceptions_bridge import with_exception_handling, async_with_exception_handling, ForexTradingPlatformError, ServiceError, DataError, ValidationError


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class AssetStrategyType(Enum):
    """Enum representing different strategy types for various asset classes"""
    FOREX_TREND = 'forex_trend'
    FOREX_RANGE = 'forex_range'
    FOREX_BREAKOUT = 'forex_breakout'
    CRYPTO_MOMENTUM = 'crypto_momentum'
    CRYPTO_MEAN_REVERSION = 'crypto_mean_reversion'
    CRYPTO_VOLATILITY = 'crypto_volatility'
    STOCK_MOMENTUM = 'stock_momentum'
    STOCK_VALUE = 'stock_value'
    STOCK_EARNINGS = 'stock_earnings'
    COMMODITY_SEASONAL = 'commodity_seasonal'
    COMMODITY_FUNDAMENTAL = 'commodity_fundamental'
    INDEX_MACRO = 'index_macro'
    MULTI_ASSET_ROTATION = 'multi_asset_rotation'


class BaseAssetStrategy(ABC):
    """
    Base class for asset-specific strategies
    
    This abstract class defines the interface that all asset-specific 
    strategies must implement.
    """

    def __init__(self, strategy_type: AssetStrategyType, asset_class:
        AssetClass, analysis_service: AnalysisIntegrationService=None,
        config: Dict[str, Any]=None):
        """
        Initialize a base asset strategy
        
        Args:
            strategy_type: Type of strategy
            asset_class: Asset class the strategy is designed for
            analysis_service: Analysis integration service
            config: Strategy configuration
        """
        self.strategy_type = strategy_type
        self.asset_class = asset_class
        self.analysis_service = analysis_service or AnalysisIntegrationService(
            )
        self.config = config or {}
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}'
            )

    @abstractmethod
    async def analyze(self, symbol: str, market_data: Dict[str, MarketData]
        ) ->Dict[str, Any]:
        """
        Analyze market data and generate trading signals
        
        Args:
            symbol: Asset symbol
            market_data: Dictionary of market data by timeframe
            
        Returns:
            Dictionary with analysis results and signals
        """
        pass

    @with_resilience('get_strategy_parameters')
    @abstractmethod
    def get_strategy_parameters(self, market_regime: str) ->Dict[str, Any]:
        """
        Get strategy parameters based on market regime
        
        Args:
            market_regime: Current market regime
            
        Returns:
            Dictionary with strategy parameters
        """
        pass

    @abstractmethod
    def adjust_parameters(self, params: Dict[str, Any], market_context:
        Dict[str, Any]) ->Dict[str, Any]:
        """
        Adjust strategy parameters based on market context
        
        Args:
            params: Current strategy parameters
            market_context: Market context information
            
        Returns:
            Adjusted parameters
        """
        pass

    @with_resilience('get_position_sizing')
    @abstractmethod
    def get_position_sizing(self, signal_strength: float, confidence: float
        ) ->float:
        """
        Calculate position sizing based on signal strength and confidence
        
        Args:
            signal_strength: Strength of the trading signal
            confidence: Confidence in the signal
            
        Returns:
            Position size as a percentage of available capital
        """
        pass

    @with_resilience('validate_asset')
    def validate_asset(self, symbol: str) ->bool:
        """
        Validate that this strategy can be applied to the given asset
        
        Args:
            symbol: Asset symbol
            
        Returns:
            True if the strategy can be applied, False otherwise
        """
        registry = AssetRegistry()
        asset_info = registry.get_asset(symbol)
        if not asset_info:
            return False
        return asset_info.get('asset_class') == self.asset_class

    @with_resilience('get_required_components')
    def get_required_components(self) ->List[str]:
        """
        Get list of required analysis components for this strategy
        
        Returns:
            List of component names
        """
        return ['technical', 'pattern', 'multi_timeframe', 'ml_prediction',
            'sentiment', 'market_regime']


class AssetStrategyFactory:
    """
    Factory for creating asset-specific strategies
    """

    def __init__(self, analysis_service: AnalysisIntegrationService=None):
        """
        Initialize the strategy factory
        
        Args:
            analysis_service: Analysis integration service to pass to strategies
        """
        self.analysis_service = analysis_service or AnalysisIntegrationService(
            )
        self._strategies = {}
        self._register_strategies()

    def _register_strategies(self):
        """Register all available strategy implementations"""
        pass

    @with_resilience('get_strategy')
    @with_exception_handling
    def get_strategy(self, strategy_type: Union[str, AssetStrategyType],
        config: Dict[str, Any]=None) ->Optional[BaseAssetStrategy]:
        """
        Get a strategy instance by type
        
        Args:
            strategy_type: Strategy type identifier
            config: Strategy configuration
            
        Returns:
            Strategy instance or None if not found
        """
        if isinstance(strategy_type, str):
            try:
                strategy_type = AssetStrategyType(strategy_type)
            except ValueError:
                logger.error(f'Invalid strategy type: {strategy_type}')
                return None
        strategy_class = self._strategies.get(strategy_type)
        if not strategy_class:
            logger.error(
                f'No implementation registered for strategy type: {strategy_type}'
                )
            return None
        return strategy_class(self.analysis_service, config or {})

    @with_resilience('get_strategy_for_asset')
    def get_strategy_for_asset(self, symbol: str, preferred_strategy_type:
        Optional[Union[str, AssetStrategyType]]=None, config: Dict[str, Any
        ]=None) ->Optional[BaseAssetStrategy]:
        """
        Get an appropriate strategy for a given asset
        
        Args:
            symbol: Asset symbol
            preferred_strategy_type: Preferred strategy type
            config: Strategy configuration
            
        Returns:
            Strategy instance or None if no suitable strategy found
        """
        registry = AssetRegistry()
        asset_info = registry.get_asset(symbol)
        if not asset_info:
            logger.error(f'Asset not found: {symbol}')
            return None
        asset_class = asset_info.get('asset_class')
        if preferred_strategy_type:
            strategy = self.get_strategy(preferred_strategy_type, config)
            if strategy and strategy.asset_class == asset_class:
                return strategy
        if asset_class == AssetClass.FOREX:
            return self.get_strategy(AssetStrategyType.FOREX_TREND, config)
        elif asset_class == AssetClass.CRYPTO:
            return self.get_strategy(AssetStrategyType.CRYPTO_MOMENTUM, config)
        elif asset_class == AssetClass.STOCKS:
            return self.get_strategy(AssetStrategyType.STOCK_MOMENTUM, config)
        elif asset_class == AssetClass.COMMODITIES:
            return self.get_strategy(AssetStrategyType.COMMODITY_SEASONAL,
                config)
        else:
            logger.error(f'No default strategy for asset class: {asset_class}')
            return None
