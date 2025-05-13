"""
Asset Adapters Module

This module provides specialized adapters for different asset classes
to ensure consistent handling of assets across the platform.
"""
import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, time
from analysis_engine.multi_asset.asset_adapter import BaseAssetAdapter
from analysis_engine.multi_asset.asset_registry import AssetRegistry, AssetClass


from analysis_engine.resilience.utils import (
    with_resilience,
    with_analysis_resilience,
    with_database_resilience
)

class CryptoAssetAdapter(BaseAssetAdapter):
    """Adapter for cryptocurrency assets"""

    def normalize_data(self, data: pd.DataFrame, symbol: str) ->pd.DataFrame:
        """Normalize cryptocurrency data for consistent analysis"""
        asset = self.asset_registry.get_asset(symbol)
        if not asset:
            self.logger.warning(f'Crypto asset not found in registry: {symbol}'
                )
            return data
        normalized = data.copy()
        if all(col in data.columns for col in ['open', 'close']):
            normalized['percent_change'] = (data['close'] - data['open']
                ) / data['open'] * 100
        if all(col in data.columns for col in ['high', 'low', 'open']):
            normalized['range_pct'] = (data['high'] - data['low']) / data[
                'open'] * 100
        if 'close' in data.columns:
            normalized['log_return'] = np.log(data['close'] / data['close']
                .shift(1))
        if all(col in data.columns for col in ['volume', 'market_cap']):
            normalized['volume_mcap_ratio'] = data['volume'] / data[
                'market_cap']
        return normalized

    @with_analysis_resilience('calculate_volatility')
    def calculate_volatility(self, data: pd.DataFrame, symbol: str, window:
        int=14) ->pd.Series:
        """Calculate cryptocurrency-specific volatility (percentage-based)"""
        if data.empty or 'close' not in data.columns:
            self.logger.warning(f'Cannot calculate volatility for {symbol}')
            return pd.Series(index=data.index)
        pct_changes = data['close'].pct_change() * 100
        volatility = pct_changes.rolling(window=window).std()
        volatility_annualized = volatility * np.sqrt(365)
        return volatility

    @with_resilience('get_price_levels')
    def get_price_levels(self, data: pd.DataFrame, symbol: str) ->Dict[str,
        float]:
        """Get significant price levels for cryptocurrencies"""
        if data.empty or 'close' not in data.columns:
            return {}
        latest = data.iloc[-1]
        levels = {'current': latest['close'], 'day_high': data['high'].iloc
            [-24:].max() if len(data) >= 24 else data['high'].max(),
            'day_low': data['low'].iloc[-24:].min() if len(data) >= 24 else
            data['low'].min(), 'week_high': data['high'].iloc[-168:].max() if
            len(data) >= 168 else data['high'].max(), 'week_low': data[
            'low'].iloc[-168:].min() if len(data) >= 168 else data['low'].min()
            }
        levels['ath'] = data['high'].max()
        levels['atl'] = data['low'].min()
        current_price = latest['close']
        magnitude = 10 ** int(np.log10(current_price))
        levels.update({'psych_level_1': np.ceil(current_price / magnitude) *
            magnitude, 'psych_level_2': np.floor(current_price / magnitude) *
            magnitude, 'psych_level_3': np.ceil(current_price / (magnitude /
            2)) * (magnitude / 2), 'psych_level_4': np.floor(current_price /
            (magnitude / 2)) * (magnitude / 2)})
        if len(data) >= 30:
            recent_high = data['high'].iloc[-30:].max()
            recent_low = data['low'].iloc[-30:].min()
            range_size = recent_high - recent_low
            levels.update({'fib_236': recent_high - range_size * 0.236,
                'fib_382': recent_high - range_size * 0.382, 'fib_500': 
                recent_high - range_size * 0.5, 'fib_618': recent_high - 
                range_size * 0.618, 'fib_786': recent_high - range_size * 
                0.786})
        if len(data) >= 200:
            levels['ma50'] = data['close'].rolling(50).mean().iloc[-1]
            levels['ma100'] = data['close'].rolling(100).mean().iloc[-1]
            levels['ma200'] = data['close'].rolling(200).mean().iloc[-1]
        return levels

    @with_resilience('get_typical_spreads')
    def get_typical_spreads(self, symbol: str) ->Dict[str, float]:
        """Get typical spread information for crypto pairs"""
        crypto_spreads = {'BTCUSD': {'min': 0.05, 'typical': 0.1, 'high': 
            0.3}, 'ETHUSD': {'min': 0.07, 'typical': 0.15, 'high': 0.4},
            'LTCUSD': {'min': 0.1, 'typical': 0.2, 'high': 0.5}, 'XRPUSD':
            {'min': 0.15, 'typical': 0.3, 'high': 0.7}, 'BCHUSD': {'min': 
            0.1, 'typical': 0.2, 'high': 0.5}, 'EOSUSD': {'min': 0.15,
            'typical': 0.3, 'high': 0.7}, 'DEFAULT': {'min': 0.15,
            'typical': 0.3, 'high': 0.7}}
        if symbol in crypto_spreads:
            return crypto_spreads[symbol]
        return crypto_spreads['DEFAULT']

    @with_resilience('get_market_hours_filter')
    def get_market_hours_filter(self, symbol: str) ->Dict[str, Any]:
        """Get market hours filter for crypto (24/7 market)"""
        return {'is_24h_market': True, 'has_session_breaks': False,
            'weekend_trading': True, 'high_liquidity_hours': [{'start':
            time(13, 0), 'end': time(21, 0), 'description':
            'US Trading Hours'}, {'start': time(0, 0), 'end': time(8, 0),
            'description': 'Asia Trading Hours'}], 'low_liquidity_hours': [
            {'start': time(21, 0), 'end': time(23, 59), 'description':
            'Late US / Early Asia'}, {'start': time(8, 0), 'end': time(13, 
            0), 'description': 'Between Asia and US'}]}

    @with_resilience('get_volatility_normalization_factor')
    def get_volatility_normalization_factor(self, symbol: str) ->float:
        """
        Get normalization factor for volatility.
        
        Returns a multiplier to standardize volatility across asset classes.
        """
        crypto_vol_factors = {'BTCUSD': 0.5, 'ETHUSD': 0.6, 'LTCUSD': 0.7,
            'XRPUSD': 0.8, 'DEFAULT': 0.7}
        if symbol in crypto_vol_factors:
            return crypto_vol_factors[symbol]
        return crypto_vol_factors['DEFAULT']

    @with_resilience('get_bitcoin_dominance_impact')
    def get_bitcoin_dominance_impact(self, symbol: str) ->Dict[str, float]:
        """
        Calculate the impact of Bitcoin dominance on this crypto asset.
        
        Returns a dictionary with correlation and impact metrics.
        """
        if symbol == 'BTCUSD':
            return {'correlation': 1.0, 'impact_factor': 0.0}
        btc_correlations = {'ETHUSD': 0.82, 'LTCUSD': 0.78, 'XRPUSD': 0.65,
            'BCHUSD': 0.75, 'EOSUSD': 0.68, 'DEFAULT': 0.7}
        correlation = btc_correlations.get(symbol, btc_correlations['DEFAULT'])
        impact_factor = correlation * 0.8
        return {'correlation': correlation, 'impact_factor': impact_factor,
            'risk_adjustment': 1 + 0.5 * correlation}


class StockAssetAdapter(BaseAssetAdapter):
    """Adapter for stock assets"""

    def normalize_data(self, data: pd.DataFrame, symbol: str) ->pd.DataFrame:
        """Normalize stock data for consistent analysis"""
        asset = self.asset_registry.get_asset(symbol)
        if not asset:
            self.logger.warning(f'Stock asset not found in registry: {symbol}')
            return data
        normalized = data.copy()
        if 'adjusted_close' in data.columns:
            normalized['real_close'] = data['adjusted_close']
        if all(col in data.columns for col in ['open', 'close']):
            normalized['percent_change'] = (data['close'] - data['open']
                ) / data['open'] * 100
        if 'volume' in data.columns:
            normalized['volume_sma_10'] = data['volume'].rolling(10).mean()
            normalized['volume_change_pct'] = data['volume'].pct_change() * 100
            normalized['relative_volume'] = data['volume'] / data['volume'
                ].rolling(20).mean()
        if all(col in data.columns for col in ['high', 'low']):
            normalized['volatility'] = (data['high'] - data['low']) / data[
                'low'] * 100
        return normalized

    @with_analysis_resilience('calculate_volatility')
    def calculate_volatility(self, data: pd.DataFrame, symbol: str, window:
        int=14) ->pd.Series:
        """Calculate stock-specific volatility"""
        if data.empty or 'close' not in data.columns:
            self.logger.warning(f'Cannot calculate volatility for {symbol}')
            return pd.Series(index=data.index)
        returns = data['close'].pct_change() * 100
        volatility = returns.rolling(window=window).std()
        volatility_annualized = volatility * np.sqrt(252)
        return volatility

    @with_resilience('get_price_levels')
    def get_price_levels(self, data: pd.DataFrame, symbol: str) ->Dict[str,
        float]:
        """Get significant price levels for stocks"""
        if data.empty or 'close' not in data.columns:
            return {}
        latest = data.iloc[-1]
        levels = {'current': latest['close'], 'day_high': data['high'].iloc
            [-1], 'day_low': data['low'].iloc[-1], 'week_high': data['high'
            ].iloc[-5:].max() if len(data) >= 5 else data['high'].max(),
            'week_low': data['low'].iloc[-5:].min() if len(data) >= 5 else
            data['low'].min(), 'month_high': data['high'].iloc[-20:].max() if
            len(data) >= 20 else data['high'].max(), 'month_low': data[
            'low'].iloc[-20:].min() if len(data) >= 20 else data['low'].min()}
        if len(data) >= 252:
            levels['52w_high'] = data['high'].iloc[-252:].max()
            levels['52w_low'] = data['low'].iloc[-252:].min()
        if len(data) >= 200:
            levels['ma50'] = data['close'].rolling(50).mean().iloc[-1]
            levels['ma100'] = data['close'].rolling(100).mean().iloc[-1]
            levels['ma200'] = data['close'].rolling(200).mean().iloc[-1]
        if 'volume' in data.columns and len(data) >= 1:
            levels['vwap'] = (data['close'] * data['volume']).sum() / data[
                'volume'].sum()
        return levels

    @with_resilience('get_typical_spreads')
    def get_typical_spreads(self, symbol: str) ->Dict[str, float]:
        """Get typical spread information for stocks"""
        stock_spreads = {'AAPL': {'min': 0.01, 'typical': 0.01, 'high': 
            0.02}, 'MSFT': {'min': 0.01, 'typical': 0.01, 'high': 0.02},
            'AMZN': {'min': 0.01, 'typical': 0.02, 'high': 0.05}, 'MID_CAP':
            {'min': 0.02, 'typical': 0.05, 'high': 0.1}, 'SMALL_CAP': {
            'min': 0.05, 'typical': 0.1, 'high': 0.25}, 'DEFAULT': {'min': 
            0.03, 'typical': 0.07, 'high': 0.15}}
        if symbol in stock_spreads:
            return stock_spreads[symbol]
        asset = self.asset_registry.get_asset(symbol)
        if asset and asset.metadata:
            market_cap = asset.metadata.get('market_cap', 0)
            if market_cap > 100000000000.0:
                return stock_spreads['AAPL']
            elif market_cap > 10000000000.0:
                return stock_spreads['MID_CAP']
            elif market_cap > 0:
                return stock_spreads['SMALL_CAP']
        return stock_spreads['DEFAULT']

    @with_resilience('get_market_hours_filter')
    def get_market_hours_filter(self, symbol: str) ->Dict[str, Any]:
        """Get market hours filter for stocks"""
        regular_hours = {'is_24h_market': False, 'has_session_breaks': 
            False, 'weekend_trading': False, 'regular_hours': [{'start':
            time(9, 30), 'end': time(16, 0), 'description':
            'US Market Hours'}], 'extended_hours': [{'start': time(4, 0),
            'end': time(9, 30), 'description': 'Pre-Market'}, {'start':
            time(16, 0), 'end': time(20, 0), 'description': 'After-Hours'}],
            'trading_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
            'Friday']}
        asset = self.asset_registry.get_asset(symbol)
        if asset and asset.metadata:
            exchange = asset.metadata.get('exchange', '').upper()
            if exchange in ['LSE', 'FTSE']:
                return {'is_24h_market': False, 'has_session_breaks': False,
                    'weekend_trading': False, 'regular_hours': [{'start':
                    time(8, 0), 'end': time(16, 30), 'description':
                    'London Market Hours'}], 'trading_days': ['Monday',
                    'Tuesday', 'Wednesday', 'Thursday', 'Friday']}
            elif exchange in ['TSE', 'JPX']:
                return {'is_24h_market': False, 'has_session_breaks': True,
                    'weekend_trading': False, 'regular_hours': [{'start':
                    time(9, 0), 'end': time(11, 30), 'description':
                    'Tokyo AM Session'}, {'start': time(12, 30), 'end':
                    time(15, 0), 'description': 'Tokyo PM Session'}],
                    'trading_days': ['Monday', 'Tuesday', 'Wednesday',
                    'Thursday', 'Friday']}
        return regular_hours

    @with_resilience('get_sector_correlation')
    def get_sector_correlation(self, symbol: str) ->Dict[str, Any]:
        """
        Get sector correlation information for a stock.
        
        Returns correlation metrics with the stock's sector and broader market.
        """
        asset = self.asset_registry.get_asset(symbol)
        if not asset or not asset.metadata:
            return {'sector_correlation': 0.7, 'market_correlation': 0.5}
        sector = asset.metadata.get('sector', '')
        sector_correlations = {'Technology': {'sector': 0.82, 'market': 
            0.75}, 'Healthcare': {'sector': 0.75, 'market': 0.6},
            'Financial': {'sector': 0.85, 'market': 0.8}, 'Energy': {
            'sector': 0.8, 'market': 0.65}, 'Consumer': {'sector': 0.7,
            'market': 0.65}, 'Utilities': {'sector': 0.6, 'market': 0.4},
            'DEFAULT': {'sector': 0.7, 'market': 0.5}}
        return sector_correlations.get(sector, sector_correlations['DEFAULT'])


class CommodityAssetAdapter(BaseAssetAdapter):
    """Adapter for commodity assets"""

    def normalize_data(self, data: pd.DataFrame, symbol: str) ->pd.DataFrame:
        """Normalize commodity data for consistent analysis"""
        asset = self.asset_registry.get_asset(symbol)
        if not asset:
            self.logger.warning(
                f'Commodity asset not found in registry: {symbol}')
            return data
        normalized = data.copy()
        if all(col in data.columns for col in ['open', 'close']):
            normalized['percent_change'] = (data['close'] - data['open']
                ) / data['open'] * 100
        if asset.metadata and asset.metadata.get('instrument_type'
            ) == 'futures':
            if 'close' in data.columns:
                returns = data['close'].pct_change()
                std_dev = returns.std()
                rollover_threshold = 3 * std_dev
                normalized['potential_rollover'] = (returns.abs() >
                    rollover_threshold) & (returns.abs() > 0.03)
                if normalized['potential_rollover'].any():
                    self.logger.info(
                        f'Potential contract rollovers detected for {symbol}')
        if 'datetime' in data.columns or isinstance(data.index, pd.
            DatetimeIndex):
            date_series = data.index if isinstance(data.index, pd.DatetimeIndex
                ) else pd.to_datetime(data['datetime'])
            normalized['month'] = date_series.month
            normalized['quarter'] = date_series.quarter
            if symbol in ['NATGAS', 'NGAS', 'NG']:
                normalized['season_factor'] = normalized['month'].apply(lambda
                    m: 1.2 if m in [11, 12, 1, 2] else 0.9 if m in [5, 6, 7,
                    8] else 1.0)
            elif symbol in ['CORN', 'WHEAT', 'SOYBEAN']:
                normalized['season_factor'] = normalized['month'].apply(lambda
                    m: 1.1 if m in [4, 5, 6] else 1.2 if m in [9, 10] else 1.0)
        return normalized

    @with_analysis_resilience('calculate_volatility')
    def calculate_volatility(self, data: pd.DataFrame, symbol: str, window:
        int=14) ->pd.Series:
        """Calculate commodity-specific volatility"""
        if data.empty or 'close' not in data.columns:
            self.logger.warning(f'Cannot calculate volatility for {symbol}')
            return pd.Series(index=data.index)
        returns = data['close'].pct_change() * 100
        asset = self.asset_registry.get_asset(symbol)
        if asset and asset.metadata and asset.metadata.get('has_seasonality',
            False):
            if isinstance(data.index, pd.DatetimeIndex):
                months = data.index.month
            elif 'datetime' in data.columns:
                months = pd.to_datetime(data['datetime']).dt.month
            else:
                months = None
            if months is not None:
                seasonal_factors = self._get_seasonal_volatility_factors(symbol
                    )
                for month, factor in seasonal_factors.items():
                    if factor != 1.0:
                        month_mask = months == month
                        returns.loc[month_mask] = returns.loc[month_mask
                            ] / factor
        volatility = returns.rolling(window=window).std()
        volatility_annualized = volatility * np.sqrt(252)
        return volatility_annualized

    def _get_seasonal_volatility_factors(self, symbol: str) ->Dict[int, float]:
        """Get seasonal volatility adjustment factors by month"""
        seasonal_factors = {'DEFAULT': {m: (1.0) for m in range(1, 13)},
            'NATGAS': {(1): 1.3, (2): 1.3, (3): 1.2, (4): 1.0, (5): 0.9, (6
            ): 0.8, (7): 0.8, (8): 0.9, (9): 1.0, (10): 1.1, (11): 1.2, (12
            ): 1.3}, 'CORN': {(1): 0.9, (2): 0.9, (3): 1.0, (4): 1.2, (5): 
            1.3, (6): 1.2, (7): 1.0, (8): 1.1, (9): 1.3, (10): 1.2, (11): 
            1.0, (12): 0.9}}
        for key in seasonal_factors.keys():
            if key in symbol:
                return seasonal_factors[key]
        return seasonal_factors['DEFAULT']

    @with_resilience('get_price_levels')
    def get_price_levels(self, data: pd.DataFrame, symbol: str) ->Dict[str,
        float]:
        """Get significant price levels for commodities"""
        if data.empty or 'close' not in data.columns:
            return {}
        latest = data.iloc[-1]
        levels = {'current': latest['close'], 'day_high': data['high'].iloc
            [-1], 'day_low': data['low'].iloc[-1], 'week_high': data['high'
            ].iloc[-5:].max() if len(data) >= 5 else data['high'].max(),
            'week_low': data['low'].iloc[-5:].min() if len(data) >= 5 else
            data['low'].min()}
        if len(data) >= 200:
            levels['ma50'] = data['close'].rolling(50).mean().iloc[-1]
            levels['ma100'] = data['close'].rolling(100).mean().iloc[-1]
            levels['ma200'] = data['close'].rolling(200).mean().iloc[-1]
        if len(data) >= 252:
            levels['52w_high'] = data['high'].iloc[-252:].max()
            levels['52w_low'] = data['low'].iloc[-252:].min()
            asset = self.asset_registry.get_asset(symbol)
            if asset and asset.metadata and asset.metadata.get(
                'has_seasonality', False):
                if isinstance(data.index, pd.DatetimeIndex):
                    current_month = data.index[-1].month
                elif 'datetime' in data.columns:
                    current_month = pd.to_datetime(data['datetime']).iloc[-1
                        ].month
                else:
                    current_month = datetime.now().month
                prev_year_same_month_mask = (pd.DatetimeIndex(data.index if
                    isinstance(data.index, pd.DatetimeIndex) else pd.
                    to_datetime(data['datetime'])).month == current_month) & (
                    pd.DatetimeIndex(data.index if isinstance(data.index,
                    pd.DatetimeIndex) else pd.to_datetime(data['datetime'])
                    ).year == datetime.now().year - 1)
                if prev_year_same_month_mask.any():
                    prev_year_data = data.loc[prev_year_same_month_mask]
                    levels['prev_year_high'] = prev_year_data['high'].max()
                    levels['prev_year_low'] = prev_year_data['low'].min()
                    levels['prev_year_close'] = prev_year_data['close'].iloc[-1
                        ] if not prev_year_data.empty else None
        return levels

    @with_resilience('get_typical_spreads')
    def get_typical_spreads(self, symbol: str) ->Dict[str, float]:
        """Get typical spread information for commodities"""
        commodity_spreads = {'CL': {'min': 0.01, 'typical': 0.02, 'high': 
            0.05}, 'NATGAS': {'min': 0.001, 'typical': 0.002, 'high': 0.005
            }, 'GC': {'min': 0.1, 'typical': 0.3, 'high': 0.5}, 'SI': {
            'min': 0.01, 'typical': 0.02, 'high': 0.04}, 'ZC': {'min': 0.25,
            'typical': 0.5, 'high': 1.0}, 'ZW': {'min': 0.25, 'typical': 
            0.5, 'high': 1.0}, 'DEFAULT': {'min': 0.05, 'typical': 0.1,
            'high': 0.3}}
        if symbol in commodity_spreads:
            return commodity_spreads[symbol]
        for key in commodity_spreads:
            if key in symbol:
                return commodity_spreads[key]
        return commodity_spreads['DEFAULT']

    @with_resilience('get_market_hours_filter')
    def get_market_hours_filter(self, symbol: str) ->Dict[str, Any]:
        """Get market hours filter for commodities"""
        energy_hours = {'is_24h_market': False, 'has_session_breaks': False,
            'weekend_trading': False, 'regular_hours': [{'start': time(9, 0
            ), 'end': time(14, 30), 'description': 'Regular Trading'}],
            'extended_hours': [{'start': time(18, 0), 'end': time(8, 0),
            'description': 'Electronic Trading'}], 'trading_days': [
            'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']}
        metals_hours = {'is_24h_market': True, 'has_session_breaks': False,
            'weekend_trading': False, 'regular_hours': [{'start': time(0, 0
            ), 'end': time(23, 59), 'description': '24hr Trading'}],
            'trading_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
            'Friday']}
        agri_hours = {'is_24h_market': False, 'has_session_breaks': False,
            'weekend_trading': False, 'regular_hours': [{'start': time(9, 
            30), 'end': time(14, 0), 'description': 'Regular Trading'}],
            'trading_days': ['Monday', 'Tuesday', 'Wednesday', 'Thursday',
            'Friday']}
        asset = self.asset_registry.get_asset(symbol)
        if not asset or not asset.metadata:
            return energy_hours
        commodity_type = asset.metadata.get('commodity_type', '')
        if commodity_type in ['energy', 'oil', 'gas']:
            return energy_hours
        elif commodity_type in ['metal', 'gold', 'silver']:
            return metals_hours
        elif commodity_type in ['agricultural', 'grain', 'livestock']:
            return agri_hours
        return energy_hours


class IndexAssetAdapter(BaseAssetAdapter):
    """Adapter for market indices"""

    def normalize_data(self, data: pd.DataFrame, symbol: str) ->pd.DataFrame:
        """Normalize index data for consistent analysis"""
        if not isinstance(data, pd.DataFrame) or data.empty:
            return data
        normalized = data.copy()
        if all(col in data.columns for col in ['open', 'close']):
            normalized['percent_change'] = (data['close'] - data['open']
                ) / data['open'] * 100
        asset = self.asset_registry.get_asset(symbol)
        if asset and asset.metadata:
            benchmark = asset.metadata.get('benchmark')
            if benchmark:
                normalized['relative_strength'] = 0.0
        if 'close' in data.columns:
            normalized['daily_volatility'] = data['close'].pct_change().abs(
                ) * 100
        if 'advancing' in data.columns and 'declining' in data.columns:
            normalized['advance_decline_ratio'] = data['advancing'] / data[
                'declining']
        return normalized

    @with_analysis_resilience('calculate_volatility')
    def calculate_volatility(self, data: pd.DataFrame, symbol: str, window:
        int=14) ->pd.Series:
        """Calculate index-specific volatility"""
        if data.empty or 'close' not in data.columns:
            self.logger.warning(f'Cannot calculate volatility for {symbol}')
            return pd.Series(index=data.index)
        returns = data['close'].pct_change() * 100
        volatility = returns.rolling(window=window).std()
        volatility_annualized = volatility * np.sqrt(252)
        return volatility_annualized

    @with_resilience('get_price_levels')
    def get_price_levels(self, data: pd.DataFrame, symbol: str) ->Dict[str,
        float]:
        """Get significant price levels for indices"""
        if data.empty or 'close' not in data.columns:
            return {}
        latest = data.iloc[-1]
        levels = {'current': latest['close'], 'day_high': data['high'].iloc
            [-1], 'day_low': data['low'].iloc[-1], 'week_high': data['high'
            ].iloc[-5:].max() if len(data) >= 5 else data['high'].max(),
            'week_low': data['low'].iloc[-5:].min() if len(data) >= 5 else
            data['low'].min()}
        if len(data) >= 200:
            levels['ma20'] = data['close'].rolling(20).mean().iloc[-1]
            levels['ma50'] = data['close'].rolling(50).mean().iloc[-1]
            levels['ma100'] = data['close'].rolling(100).mean().iloc[-1]
            levels['ma200'] = data['close'].rolling(200).mean().iloc[-1]
        if len(data) >= 252:
            levels['52w_high'] = data['high'].iloc[-252:].max()
            levels['52w_low'] = data['low'].iloc[-252:].min()
            year_start_idx = data.index[data.index.year == data.index[-1].year
                ][0]
            year_start_close = data.loc[year_start_idx, 'close']
            levels['ytd_change_percent'] = (latest['close'] - year_start_close
                ) / year_start_close * 100
        current_price = latest['close']
        if current_price > 10000:
            round_factors = [1000, 500, 100]
        elif current_price > 1000:
            round_factors = [500, 100, 50]
        elif current_price > 100:
            round_factors = [100, 50, 10]
        else:
            round_factors = [10, 5, 1]
        for factor in round_factors:
            levels[f'psych_above_{factor}'] = np.ceil(current_price / factor
                ) * factor
            levels[f'psych_below_{factor}'] = np.floor(current_price / factor
                ) * factor
        return levels

    @with_resilience('get_typical_spreads')
    def get_typical_spreads(self, symbol: str) ->Dict[str, float]:
        """Get typical spread information for indices"""
        index_spreads = {'SPX500': {'min': 0.5, 'typical': 0.7, 'high': 1.0
            }, 'DJIA': {'min': 1.0, 'typical': 2.0, 'high': 3.0}, 'NASDAQ':
            {'min': 0.75, 'typical': 1.0, 'high': 1.5}, 'NDX': {'min': 1.0,
            'typical': 1.5, 'high': 2.5}, 'DAX': {'min': 1.0, 'typical': 
            1.5, 'high': 2.0}, 'FTSE': {'min': 0.8, 'typical': 1.0, 'high':
            1.5}, 'NIKKEI': {'min': 10, 'typical': 15, 'high': 25},
            'DEFAULT': {'min': 1.0, 'typical': 1.5, 'high': 2.5}}
        if symbol in index_spreads:
            return index_spreads[symbol]
        for key in index_spreads:
            if key in symbol:
                return index_spreads[key]
        return index_spreads['DEFAULT']

    @with_resilience('get_market_hours_filter')
    def get_market_hours_filter(self, symbol: str) ->Dict[str, Any]:
        """Get market hours filter for indices"""
        us_index_hours = {'is_24h_market': False, 'has_session_breaks': 
            False, 'weekend_trading': False, 'regular_hours': [{'start':
            time(9, 30), 'end': time(16, 0), 'description':
            'Regular Trading'}], 'trading_days': ['Monday', 'Tuesday',
            'Wednesday', 'Thursday', 'Friday']}
        eu_index_hours = {'is_24h_market': False, 'has_session_breaks': 
            False, 'weekend_trading': False, 'regular_hours': [{'start':
            time(8, 0), 'end': time(16, 30), 'description':
            'Regular Trading'}], 'trading_days': ['Monday', 'Tuesday',
            'Wednesday', 'Thursday', 'Friday']}
        asia_index_hours = {'is_24h_market': False, 'has_session_breaks': 
            True, 'weekend_trading': False, 'regular_hours': [{'start':
            time(9, 0), 'end': time(11, 30), 'description':
            'Morning Session'}, {'start': time(12, 30), 'end': time(15, 0),
            'description': 'Afternoon Session'}], 'trading_days': ['Monday',
            'Tuesday', 'Wednesday', 'Thursday', 'Friday']}
        us_indices = ['SPX', 'DJIA', 'NASDAQ', 'NDX', 'RUSSELL']
        eu_indices = ['DAX', 'FTSE', 'CAC', 'STOXX', 'IBEX']
        asia_indices = ['NIKKEI', 'HSI', 'ASX', 'KOSPI']
        for idx in us_indices:
            if idx in symbol:
                return us_index_hours
        for idx in eu_indices:
            if idx in symbol:
                return eu_index_hours
        for idx in asia_indices:
            if idx in symbol:
                return asia_index_hours
        return us_index_hours

    @with_resilience('get_vix_relationship')
    def get_vix_relationship(self, symbol: str) ->Dict[str, Any]:
        """
        Get information about this index's relationship with volatility indices
        
        For US indices, this would be relationship with VIX
        For other markets, their respective volatility indices
        """
        relationships = {'SPX': {'vix_inverse_correlation': -0.8,
            'typical_vix_range': [10, 30]}, 'DJIA': {
            'vix_inverse_correlation': -0.75, 'typical_vix_range': [10, 30]
            }, 'NASDAQ': {'vix_inverse_correlation': -0.82,
            'typical_vix_range': [15, 35]}, 'DAX': {
            'vstoxx_inverse_correlation': -0.78, 'typical_vstoxx_range': [
            12, 32]}, 'DEFAULT': {'volatility_inverse_correlation': -0.7,
            'typical_vol_range': [15, 30]}}
        for idx, data in relationships.items():
            if idx in symbol:
                return data
        return relationships['DEFAULT']


class AssetAdapterFactory:
    """Factory class for creating appropriate asset adapters"""

    @staticmethod
    def create_adapter(asset_class: AssetClass, asset_registry: AssetRegistry
        ) ->BaseAssetAdapter:
        """
        Create appropriate adapter for asset class
        
        Args:
            asset_class: Type of asset
            asset_registry: Registry of assets
            
        Returns:
            Asset adapter instance
        """
        if asset_class == AssetClass.FOREX:
            from analysis_engine.multi_asset.asset_adapter import ForexAssetAdapter
            return ForexAssetAdapter(asset_registry)
        elif asset_class == AssetClass.CRYPTO:
            return CryptoAssetAdapter(asset_registry)
        elif asset_class == AssetClass.STOCKS:
            return StockAssetAdapter(asset_registry)
        elif asset_class == AssetClass.COMMODITIES:
            return CommodityAssetAdapter(asset_registry)
        elif asset_class == AssetClass.INDICES:
            return IndexAssetAdapter(asset_registry)
        else:
            from analysis_engine.multi_asset.asset_adapter import ForexAssetAdapter
            return ForexAssetAdapter(asset_registry)
