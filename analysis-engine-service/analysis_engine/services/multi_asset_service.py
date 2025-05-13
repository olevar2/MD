"""
Multi-Asset Service Module

This module provides services for working with different asset classes
and markets across the trading platform.
"""
import logging
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import pandas as pd
from analysis_engine.multi_asset.asset_registry import AssetRegistry
from analysis_engine.multi_asset.asset_adapter import AssetAdapterFactory, BaseAssetAdapter
from common_lib.multi_asset.interfaces import AssetClassEnum as AssetClass, IMultiAssetService


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class MultiAssetService(IMultiAssetService):
    """
    Service for working with multiple asset classes and markets.

    This service provides methods to:
    1. Access asset information
    2. Get appropriate adapters for different assets
    3. Normalize data across different asset types
    4. Provide asset-specific analysis parameters

    Implements the IMultiAssetService interface from common-lib.
    """

    def __init__(self, config_path: Optional[str]=None):
        """
        Initialize the multi-asset service

        Args:
            config_path: Path to asset configuration file (JSON)
        """
        self.logger = logging.getLogger(__name__)
        if not config_path:
            config_path = os.path.join(os.path.dirname(__file__),
                'multi_asset', 'asset_config.json')
        self.asset_registry = AssetRegistry(config_path)
        self.adapter_factory = AssetAdapterFactory(self.asset_registry)

    @with_resilience('get_adapter')
    def get_adapter(self, symbol: str) ->BaseAssetAdapter:
        """Get the appropriate adapter for a symbol"""
        return self.adapter_factory.get_adapter(symbol)

    def normalize_data(self, data: pd.DataFrame, symbol: str) ->pd.DataFrame:
        """Normalize data for the specified symbol"""
        adapter = self.get_adapter(symbol)
        return adapter.normalize_data(data, symbol)

    @with_analysis_resilience('calculate_volatility')
    def calculate_volatility(self, data: pd.DataFrame, symbol: str, window:
        int=14) ->pd.Series:
        """Calculate symbol-specific volatility"""
        adapter = self.get_adapter(symbol)
        return adapter.calculate_volatility(data, symbol, window)

    @with_resilience('get_price_levels')
    def get_price_levels(self, data: pd.DataFrame, symbol: str) ->Dict[str,
        float]:
        """Get significant price levels for the symbol"""
        adapter = self.get_adapter(symbol)
        return adapter.get_price_levels(data, symbol)

    @with_resilience('get_correlated_symbols')
    def get_correlated_symbols(self, symbol: str, threshold: float=0.7) ->List[
        str]:
        """Get correlated symbols above the threshold"""
        correlated_assets = self.asset_registry.get_correlated_assets(symbol,
            threshold)
        return [asset['symbol'] for asset in correlated_assets]

    @with_resilience('get_asset_info')
    def get_asset_info(self, symbol: str) ->Dict[str, Any]:
        """Get detailed information about an asset"""
        asset = self.asset_registry.get_asset(symbol)
        if not asset:
            return {}
        return {'symbol': asset.symbol, 'display_name': asset.display_name,
            'asset_class': asset.asset_class, 'market_type': asset.
            market_type, 'base_currency': asset.base_currency,
            'quote_currency': asset.quote_currency, 'trading_parameters':
            self.asset_registry.get_trading_parameters(symbol),
            'available_timeframes': asset.available_timeframes, 'metadata':
            asset.metadata or {}}

    def list_assets_by_class(self, asset_class: AssetClass) ->List[str]:
        """List all assets of a specific class"""
        assets = self.asset_registry.list_assets(asset_class=asset_class)
        return [asset.symbol for asset in assets]

    @with_resilience('get_asset_group')
    def get_asset_group(self, group_name: str) ->List[str]:
        """Get all symbols in a named group"""
        assets = self.asset_registry.get_asset_group(group_name)
        return [asset.symbol for asset in assets]

    @with_resilience('get_analysis_parameters')
    def get_analysis_parameters(self, symbol: str) ->Dict[str, Any]:
        """Get asset-specific analysis parameters"""
        asset = self.asset_registry.get_asset(symbol)
        if not asset:
            return {}
        adapter = self.get_adapter(symbol)
        params = {'volatility_window': 14, 'pip_value': asset.config.
            pip_value, 'typical_spread': adapter.get_typical_spreads(symbol
            ).get('typical', 0), 'position_sizing_factors': adapter.
            get_position_sizing_factors(symbol)}
        if asset.asset_class == AssetClass.FOREX:
            params['pattern_precision'] = 5
            params['default_stop_atr_multiple'] = 1.5
        elif asset.asset_class == AssetClass.CRYPTO:
            params['pattern_precision'] = 2
            params['default_stop_atr_multiple'] = 2.0
            params['volatility_window'] = 20
        elif asset.asset_class == AssetClass.STOCKS:
            params['pattern_precision'] = 2
            params['default_stop_atr_multiple'] = 1.0
        return params
