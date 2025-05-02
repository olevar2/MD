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

# Import base strategy classes
from strategy_execution_engine.strategies.base_strategy import BaseStrategy
from strategy_execution_engine.strategies.multi_timeframe_confluence_strategy import MultiTimeframeConfluenceStrategy


class AssetClass(Enum):
    """Supported asset classes"""
    FOREX = "forex"
    CRYPTO = "crypto"
    STOCK = "stock"
    COMMODITY = "commodity"
    INDEX = "index"
    BOND = "bond"
    ETF = "etf"


class AssetStrategyFactory:
    """
    Factory for creating asset-specific strategy implementations.
    
    This factory manages the creation and configuration of strategies
    tailored to different asset classes.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the asset strategy factory.
        
        Args:
            config_path: Optional path to strategy configuration file
        """
        self.logger = logging.getLogger(__name__)
        self.strategies = {}
        self.asset_configs = {}
        
        # Load default strategy configurations if available
        if config_path and os.path.exists(config_path):
            self._load_config(config_path)
        else:
            self._load_default_configs()
    
    def _load_config(self, config_path: str) -> None:
        """Load strategy configuration from file"""
        try:
            with open(config_path, 'r') as f:
                self.asset_configs = json.load(f)
                self.logger.info(f"Loaded asset strategy configurations from {config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load strategy config from {config_path}: {str(e)}")
            self._load_default_configs()
    
    def _load_default_configs(self) -> None:
        """Load default strategy configurations"""
        self.asset_configs = {
            AssetClass.FOREX.value: {
                "default_strategy": "MultiTimeframeConfluenceStrategy",
                "parameters": {
                    "confirmation_threshold": 2,
                    "min_confluence_score": 0.7,
                    "volatility_multiplier": 1.0,
                    "use_atr_for_stops": True,
                    "atr_stop_multiplier": 1.5,
                    "timeframe_weights": {
                        "M15": 0.7,
                        "H1": 1.0,
                        "H4": 1.2,
                        "D1": 1.3
                    }
                }
            },
            AssetClass.CRYPTO.value: {
                "default_strategy": "MultiTimeframeConfluenceStrategy",
                "parameters": {
                    "confirmation_threshold": 3,  # More confirmations for volatile crypto
                    "min_confluence_score": 0.8,  # Higher confidence required
                    "volatility_multiplier": 1.5,  # Higher volatility in crypto
                    "use_atr_for_stops": True,
                    "atr_stop_multiplier": 2.0,   # Wider stops for crypto volatility
                    "timeframe_weights": {
                        "M15": 0.6,
                        "H1": 0.8,
                        "H4": 1.0,
                        "D1": 1.2
                    }
                }
            },
            AssetClass.STOCK.value: {
                "default_strategy": "MultiTimeframeConfluenceStrategy",
                "parameters": {
                    "confirmation_threshold": 2,
                    "min_confluence_score": 0.65,
                    "volatility_multiplier": 0.8,  # Lower volatility than forex
                    "use_atr_for_stops": True,
                    "atr_stop_multiplier": 1.3,
                    "timeframe_weights": {
                        "H1": 0.7,
                        "H4": 0.9,
                        "D1": 1.2,
                        "W1": 1.5  # Stocks often have stronger weekly patterns
                    }
                }
            },
            AssetClass.COMMODITY.value: {
                "default_strategy": "MultiTimeframeConfluenceStrategy",
                "parameters": {
                    "confirmation_threshold": 2,
                    "min_confluence_score": 0.7,
                    "volatility_multiplier": 1.2,
                    "use_atr_for_stops": True,
                    "atr_stop_multiplier": 1.6,
                    "timeframe_weights": {
                        "H1": 0.8,
                        "H4": 1.0,
                        "D1": 1.3,
                        "W1": 1.4
                    }
                }
            },
            AssetClass.INDEX.value: {
                "default_strategy": "MultiTimeframeConfluenceStrategy",
                "parameters": {
                    "confirmation_threshold": 2,
                    "min_confluence_score": 0.65,
                    "volatility_multiplier": 0.9,
                    "use_atr_for_stops": True,
                    "atr_stop_multiplier": 1.4,
                    "timeframe_weights": {
                        "H1": 0.8,
                        "H4": 1.0,
                        "D1": 1.3,
                        "W1": 1.4
                    }
                }
            }
        }
        
        self.logger.info("Loaded default asset strategy configurations")
    
    def get_strategy_for_asset(
        self, 
        symbol: str, 
        asset_class: str, 
        strategy_name: Optional[str] = None
    ) -> BaseStrategy:
        """
        Get a strategy implementation tailored for the specific asset.
        
        Args:
            symbol: Trading symbol
            asset_class: Asset class (forex, crypto, etc.)
            strategy_name: Optional strategy name to use, otherwise uses default for asset class
            
        Returns:
            An asset-specific strategy implementation
        """
        # Normalize asset class
        asset_class = asset_class.lower()
        if asset_class not in [ac.value for ac in AssetClass]:
            self.logger.warning(f"Unknown asset class: {asset_class}, defaulting to forex")
            asset_class = AssetClass.FOREX.value
        
        # Get config for this asset class
        asset_config = self.asset_configs.get(asset_class, self.asset_configs.get(AssetClass.FOREX.value))
        
        # Determine which strategy to use
        if not strategy_name:
            strategy_name = asset_config.get("default_strategy", "MultiTimeframeConfluenceStrategy")
            
        # Get parameters for this asset class and strategy
        params = asset_config.get("parameters", {})
        
        # Create the appropriate strategy instance
        if strategy_name == "MultiTimeframeConfluenceStrategy":
            strategy = MultiTimeframeConfluenceStrategy(
                symbol=symbol,
                config=self._create_strategy_config(params)
            )
            
            # Apply asset-specific customizations
            strategy.set_asset_class(asset_class)
            self._customize_strategy_for_asset(strategy, asset_class, params)
            
            return strategy
        else:
            self.logger.warning(f"Unknown strategy: {strategy_name}, using MultiTimeframeConfluenceStrategy")
            strategy = MultiTimeframeConfluenceStrategy(
                symbol=symbol,
                config=self._create_strategy_config(params)
            )
            
            # Apply asset-specific customizations
            strategy.set_asset_class(asset_class)
            self._customize_strategy_for_asset(strategy, asset_class, params)
            
            return strategy
    
    def _create_strategy_config(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Create a strategy configuration from parameters"""
        # Base configuration
        config = {
            "timeframes": ["M15", "H1", "H4", "D1"],
            "indicators": {
                "primary": ["ema", "rsi", "macd"],
                "secondary": ["bollinger", "atr", "ichimoku"]
            },
            "confirmation_threshold": params.get("confirmation_threshold", 2),
            "min_confluence_score": params.get("min_confluence_score", 0.7),
            "volatility_multiplier": params.get("volatility_multiplier", 1.0),
        }
        
        # Add stop loss config
        if params.get("use_atr_for_stops", True):
            config["stop_loss"] = {
                "method": "atr",
                "multiplier": params.get("atr_stop_multiplier", 1.5)
            }
        else:
            config["stop_loss"] = {
                "method": "swing",
                "buffer_pips": 10
            }
        
        # Add timeframe weights if specified
        if "timeframe_weights" in params:
            config["timeframe_weights"] = params["timeframe_weights"]
        
        return config
    
    def _customize_strategy_for_asset(
        self, 
        strategy: BaseStrategy, 
        asset_class: str, 
        params: Dict[str, Any]
    ) -> None:
        """Apply asset-specific customizations to the strategy"""
        # Apply asset-specific customizations based on asset class
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
    
    def _customize_forex_strategy(self, strategy: BaseStrategy, params: Dict[str, Any]) -> None:
        """Customize strategy for forex assets"""
        # Forex-specific customizations
        if hasattr(strategy, 'set_price_precision'):
            strategy.set_price_precision(5)  # Most forex pairs have 5 decimal places
        
        if hasattr(strategy, 'set_session_importance'):
            # Set importance weights for different forex sessions
            strategy.set_session_importance({
                "asian": 0.8,
                "london": 1.0,
                "new_york": 1.0,
                "overlap": 1.2  # London/NY overlap often has highest volatility
            })
    
    def _customize_crypto_strategy(self, strategy: BaseStrategy, params: Dict[str, Any]) -> None:
        """Customize strategy for crypto assets"""
        # Crypto-specific customizations
        if hasattr(strategy, 'set_price_precision'):
            strategy.set_price_precision(8)  # Crypto often needs high precision
            
        if hasattr(strategy, 'set_volatility_handling'):
            # More aggressive volatility handling for crypto
            strategy.set_volatility_handling({
                "increase_filter_strength_in_high_volatility": True,
                "dynamic_stop_adjustment": True,
                "volatility_threshold_multiplier": 1.5
            })
    
    def _customize_stock_strategy(self, strategy: BaseStrategy, params: Dict[str, Any]) -> None:
        """Customize strategy for stock assets"""
        # Stock-specific customizations
        if hasattr(strategy, 'set_session_importance'):
            # Stocks typically follow their primary exchange hours
            strategy.set_session_importance({
                "pre_market": 0.7,
                "regular_hours": 1.0,
                "post_market": 0.6
            })
            
        if hasattr(strategy, 'set_additional_filters'):
            # Add stock-specific filters
            strategy.set_additional_filters([
                "earnings_calendar",
                "sector_correlation",
                "index_relative_strength"
            ])
    
    def _customize_commodity_strategy(self, strategy: BaseStrategy, params: Dict[str, Any]) -> None:
        """Customize strategy for commodity assets"""
        # Commodity-specific customizations
        if hasattr(strategy, 'set_session_importance'):
            # Different commodities have different active sessions
            strategy.set_session_importance({
                "asian": 0.8,
                "london": 0.9,
                "new_york": 1.0
            })
            
        if hasattr(strategy, 'set_seasonal_analysis'):
            # Enable seasonal analysis for commodities
            strategy.set_seasonal_analysis(True)
    
    def _customize_index_strategy(self, strategy: BaseStrategy, params: Dict[str, Any]) -> None:
        """Customize strategy for index assets"""
        # Index-specific customizations
        if hasattr(strategy, 'set_session_importance'):
            # Indices follow their exchange hours
            strategy.set_session_importance({
                "pre_market": 0.6,
                "regular_hours": 1.0,
                "post_market": 0.5
            })
            
        if hasattr(strategy, 'set_correlation_assets'):
            # Set correlated assets to watch
            strategy.set_correlation_assets([
                "major_stocks",
                "sector_etfs",
                "vix"
            ])
