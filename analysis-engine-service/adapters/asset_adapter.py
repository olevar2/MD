"""
Asset Adapter Module

This module provides adapters for different asset classes to ensure
consistent analysis across multiple markets and instruments.
"""
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import numpy as np
from analysis_engine.multi_asset.asset_registry import AssetClass, AssetRegistry, AssetDefinition
logger = logging.getLogger(__name__)


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class BaseAssetAdapter(ABC):
    """Base class for asset-specific adapters"""

    def __init__(self, asset_registry: AssetRegistry):
    """
      init  .
    
    Args:
        asset_registry: Description of asset_registry
    
    """

        self.asset_registry = asset_registry
        self.logger = logging.getLogger(f'{__name__}.{self.__class__.__name__}'
            )

    @abstractmethod
    def normalize_data(self, data: pd.DataFrame, symbol: str) ->pd.DataFrame:
        """Normalize price data for consistent analysis"""
        pass

    @with_analysis_resilience('calculate_volatility')
    @abstractmethod
    def calculate_volatility(self, data: pd.DataFrame, symbol: str, window:
        int=14) ->pd.Series:
        """Calculate asset-specific volatility"""
        pass

    @with_resilience('get_price_levels')
    @abstractmethod
    def get_price_levels(self, data: pd.DataFrame, symbol: str) ->Dict[str,
        float]:
        """Get significant price levels for the asset"""
        pass

    @with_resilience('get_typical_spreads')
    @abstractmethod
    def get_typical_spreads(self, symbol: str) ->Dict[str, float]:
        """Get typical spread information for the asset"""
        pass

    @with_resilience('get_position_sizing_factors')
    def get_position_sizing_factors(self, symbol: str) ->Dict[str, float]:
        """Get position sizing factors specific to the asset type"""
        asset = self.asset_registry.get_asset(symbol)
        if not asset:
            self.logger.warning(f'Asset not found in registry: {symbol}')
            return {}
        factors = {'pip_value': asset.config.pip_value, 'min_quantity':
            asset.config.min_quantity}
        if asset.config.lot_size:
            factors['lot_size'] = asset.config.lot_size
        if asset.config.margin_rate:
            factors['margin_rate'] = asset.config.margin_rate
        return factors


class ForexAssetAdapter(BaseAssetAdapter):
    """Adapter for forex assets"""

    def normalize_data(self, data: pd.DataFrame, symbol: str) ->pd.DataFrame:
        """Normalize forex data for consistent analysis"""
        asset = self.asset_registry.get_asset(symbol)
        if not asset:
            self.logger.warning(f'Forex asset not found in registry: {symbol}')
            return data
        pip_location = asset.config.pip_location
        pip_factor = 10 ** abs(pip_location)
        normalized = data.copy()
        if all(col in data.columns for col in ['high', 'low']):
            normalized['range_pips'] = (data['high'] - data['low']
                ) * pip_factor
        if all(col in data.columns for col in ['open', 'close']):
            normalized['move_pips'] = (data['close'] - data['open']
                ) * pip_factor
        return normalized

    @with_analysis_resilience('calculate_volatility')
    def calculate_volatility(self, data: pd.DataFrame, symbol: str, window:
        int=14) ->pd.Series:
        """Calculate forex-specific volatility (ATR in pips)"""
        asset = self.asset_registry.get_asset(symbol)
        if (not asset or 'high' not in data.columns or 'low' not in data.
            columns):
            self.logger.warning(f'Cannot calculate volatility for {symbol}')
            return pd.Series(index=data.index)
        pip_location = asset.config.pip_location
        pip_factor = 10 ** abs(pip_location)
        high_low = (data['high'] - data['low']) * pip_factor
        high_close_prev = abs(data['high'] - data['close'].shift(1)
            ) * pip_factor
        low_close_prev = abs(data['low'] - data['close'].shift(1)) * pip_factor
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1
            ).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

    @with_resilience('get_price_levels')
    def get_price_levels(self, data: pd.DataFrame, symbol: str) ->Dict[str,
        float]:
        """Get significant price levels for forex"""
        if data.empty or 'close' not in data.columns:
            return {}
        latest = data.iloc[-1]
        levels = {'current': latest['close'], 'day_high': data['high'].iloc
            [-20:].max(), 'day_low': data['low'].iloc[-20:].min(),
            'week_high': data['high'].iloc[-100:].max() if len(data) >= 100
             else data['high'].max(), 'week_low': data['low'].iloc[-100:].
            min() if len(data) >= 100 else data['low'].min()}
        if len(data) >= 20:
            pivot = (latest['high'] + latest['low'] + latest['close']) / 3
            levels.update({'pivot': pivot, 'r1': 2 * pivot - latest['low'],
                's1': 2 * pivot - latest['high'], 'r2': pivot + (latest[
                'high'] - latest['low']), 's2': pivot - (latest['high'] -
                latest['low'])})
        return levels

    @with_resilience('get_typical_spreads')
    def get_typical_spreads(self, symbol: str) ->Dict[str, float]:
        """Get typical spread information for forex pairs"""
        forex_spreads = {'EURUSD': {'min': 0.1, 'typical': 0.2, 'high': 0.5
            }, 'GBPUSD': {'min': 0.4, 'typical': 0.7, 'high': 1.2},
            'USDJPY': {'min': 0.3, 'typical': 0.5, 'high': 0.8}, 'USDCHF':
            {'min': 0.5, 'typical': 0.8, 'high': 1.5}, 'AUDUSD': {'min': 
            0.3, 'typical': 0.5, 'high': 1.0}, 'USDCAD': {'min': 0.5,
            'typical': 0.7, 'high': 1.2}, 'NZDUSD': {'min': 0.5, 'typical':
            0.9, 'high': 1.5}, 'EURGBP': {'min': 0.5, 'typical': 0.8,
            'high': 1.4}, 'EURJPY': {'min': 0.4, 'typical': 0.8, 'high': 
            1.3}, 'GBPJPY': {'min': 0.8, 'typical': 1.2, 'high': 2.0}}
        return forex_spreads.get(symbol, {'min': 0.5, 'typical': 1.0,
            'high': 2.0})


class CryptoAssetAdapter(BaseAssetAdapter):
    """Adapter for cryptocurrency assets"""

    def normalize_data(self, data: pd.DataFrame, symbol: str) ->pd.DataFrame:
        """Normalize crypto data for consistent analysis"""
        normalized = data.copy()
        if all(col in data.columns for col in ['high', 'low']):
            normalized['range_pct'] = (data['high'] - data['low']) / data['low'
                ] * 100
        if all(col in data.columns for col in ['open', 'close']):
            normalized['move_pct'] = (data['close'] - data['open']) / data[
                'open'] * 100
        return normalized

    @with_analysis_resilience('calculate_volatility')
    def calculate_volatility(self, data: pd.DataFrame, symbol: str, window:
        int=14) ->pd.Series:
        """Calculate crypto-specific volatility (percentage based)"""
        if 'high' not in data.columns or 'low' not in data.columns:
            self.logger.warning(f'Cannot calculate volatility for {symbol}')
            return pd.Series(index=data.index)
        high_low = (data['high'] - data['low']) / data['low'] * 100
        high_close_prev = abs((data['high'] - data['close'].shift(1)) /
            data['close'].shift(1)) * 100
        low_close_prev = abs((data['low'] - data['close'].shift(1)) / data[
            'close'].shift(1)) * 100
        tr = pd.concat([high_low, high_close_prev, low_close_prev], axis=1
            ).max(axis=1)
        atr = tr.rolling(window=window).mean()
        return atr

    @with_resilience('get_price_levels')
    def get_price_levels(self, data: pd.DataFrame, symbol: str) ->Dict[str,
        float]:
        """Get significant price levels for crypto"""
        if data.empty or 'close' not in data.columns:
            return {}
        latest = data.iloc[-1]
        current = latest['close']
        magnitude = 10 ** np.floor(np.log10(current))
        round_levels = {f'round_{int(i * magnitude)}': (i * magnitude) for
            i in [0.1, 0.25, 0.5, 1, 2.5, 5, 10] if i * magnitude > current *
            0.7 and i * magnitude < current * 1.3}
        levels = {'current': current, 'day_high': data['high'].iloc[-24:].
            max() if len(data) >= 24 else data['high'].max(), 'day_low': 
            data['low'].iloc[-24:].min() if len(data) >= 24 else data['low'
            ].min(), 'week_high': data['high'].iloc[-168:].max() if len(
            data) >= 168 else data['high'].max(), 'week_low': data['low'].
            iloc[-168:].min() if len(data) >= 168 else data['low'].min(),
            **round_levels}
        return levels

    @with_resilience('get_typical_spreads')
    def get_typical_spreads(self, symbol: str) ->Dict[str, float]:
        """Get typical spread information for crypto assets (usually in basis points)"""
        crypto_spreads = {'BTCUSD': {'min': 5, 'typical': 10, 'high': 25},
            'ETHUSD': {'min': 8, 'typical': 15, 'high': 30}, 'XRPUSD': {
            'min': 10, 'typical': 20, 'high': 40}, 'LTCUSD': {'min': 10,
            'typical': 20, 'high': 35}}
        return crypto_spreads.get(symbol, {'min': 10, 'typical': 20, 'high':
            40})


class StockAssetAdapter(BaseAssetAdapter):
    """Adapter for stock assets"""

    def normalize_data(self, data: pd.DataFrame, symbol: str) ->pd.DataFrame:
        """Normalize stock data for consistent analysis"""
        normalized = data.copy()
        if all(col in data.columns for col in ['high', 'low']):
            normalized['range_pct'] = (data['high'] - data['low']) / data['low'
                ] * 100
        if all(col in data.columns for col in ['open', 'close']):
            normalized['move_pct'] = (data['close'] - data['open']) / data[
                'open'] * 100
        return normalized

    @with_analysis_resilience('calculate_volatility')
    def calculate_volatility(self, data: pd.DataFrame, symbol: str, window:
        int=14) ->pd.Series:
        """Calculate stock-specific volatility"""
        if 'close' not in data.columns:
            self.logger.warning(f'Cannot calculate volatility for {symbol}')
            return pd.Series(index=data.index)
        returns = data['close'].pct_change() * 100
        volatility = returns.rolling(window=window).std()
        return volatility

    @with_resilience('get_price_levels')
    def get_price_levels(self, data: pd.DataFrame, symbol: str) ->Dict[str,
        float]:
        """Get significant price levels for stocks"""
        if data.empty or 'close' not in data.columns:
            return {}
        latest = data.iloc[-1]
        levels = {'current': latest['close'], 'day_high': data['high'].iloc
            [-1] if 'high' in data.columns else latest['close'], 'day_low':
            data['low'].iloc[-1] if 'low' in data.columns else latest['close']}
        if len(data) >= 50:
            levels['ma20'] = data['close'].rolling(20).mean().iloc[-1]
            levels['ma50'] = data['close'].rolling(50).mean().iloc[-1]
        if len(data) >= 200:
            levels['ma200'] = data['close'].rolling(200).mean().iloc[-1]
        return levels

    @with_resilience('get_typical_spreads')
    def get_typical_spreads(self, symbol: str) ->Dict[str, float]:
        """Get typical spread information for stocks"""
        return {'min': 0.01, 'typical': 0.02, 'high': 0.05}


class AssetAdapterFactory:
    """Factory for creating appropriate asset adapters"""

    def __init__(self, asset_registry: AssetRegistry):
    """
      init  .
    
    Args:
        asset_registry: Description of asset_registry
    
    """

        self.asset_registry = asset_registry
        self.logger = logging.getLogger(__name__)

    @with_resilience('get_adapter')
    def get_adapter(self, symbol: str) ->BaseAssetAdapter:
        """Create an adapter instance for the given symbol"""
        asset = self.asset_registry.get_asset(symbol)
        if not asset:
            self.logger.warning(
                f'Asset not found in registry: {symbol}, using default forex adapter'
                )
            return ForexAssetAdapter(self.asset_registry)
        if asset.asset_class == AssetClass.FOREX:
            return ForexAssetAdapter(self.asset_registry)
        elif asset.asset_class == AssetClass.CRYPTO:
            return CryptoAssetAdapter(self.asset_registry)
        elif asset.asset_class == AssetClass.STOCKS:
            return StockAssetAdapter(self.asset_registry)
        else:
            self.logger.info(
                f'No specific adapter for {asset.asset_class}, using forex adapter'
                )
            return ForexAssetAdapter(self.asset_registry)
