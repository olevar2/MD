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


class CryptoAssetAdapter(BaseAssetAdapter):
    """Adapter for cryptocurrency assets"""
    
    def normalize_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize cryptocurrency data for consistent analysis"""
        asset = self.asset_registry.get_asset(symbol)
        if not asset:
            self.logger.warning(f"Crypto asset not found in registry: {symbol}")
            return data
        
        # Create a copy to avoid modifying the original
        normalized = data.copy()
        
        # Calculate daily percentage changes for crypto (more relevant than pips)
        if all(col in data.columns for col in ['open', 'close']):
            normalized['percent_change'] = ((data['close'] - data['open']) / data['open']) * 100
            
        # Calculate high-low range in percentage
        if all(col in data.columns for col in ['high', 'low', 'open']):
            normalized['range_pct'] = ((data['high'] - data['low']) / data['open']) * 100
        
        # Add log returns (commonly used for crypto analysis)
        if 'close' in data.columns:
            normalized['log_return'] = np.log(data['close'] / data['close'].shift(1))
        
        # Normalize volume by market cap if available
        if all(col in data.columns for col in ['volume', 'market_cap']):
            normalized['volume_mcap_ratio'] = data['volume'] / data['market_cap']
        
        return normalized
    
    def calculate_volatility(self, data: pd.DataFrame, symbol: str, window: int = 14) -> pd.Series:
        """Calculate cryptocurrency-specific volatility (percentage-based)"""
        if data.empty or 'close' not in data.columns:
            self.logger.warning(f"Cannot calculate volatility for {symbol}")
            return pd.Series(index=data.index)
        
        # Crypto typically uses percentage volatility rather than pips
        # Calculate percent changes
        pct_changes = data['close'].pct_change() * 100
        
        # Calculate rolling standard deviation of percent changes
        volatility = pct_changes.rolling(window=window).std()
        
        # Annualize the volatility (multiply by sqrt of trading periods in a year)
        # Assuming daily data, multiply by sqrt(365)
        volatility_annualized = volatility * np.sqrt(365)
        
        return volatility
    
    def get_price_levels(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Get significant price levels for cryptocurrencies"""
        if data.empty or 'close' not in data.columns:
            return {}
            
        # Get the latest prices
        latest = data.iloc[-1]
        
        # Initialize levels dictionary
        levels = {
            "current": latest['close'],
            "day_high": data['high'].iloc[-24:].max() if len(data) >= 24 else data['high'].max(),
            "day_low": data['low'].iloc[-24:].min() if len(data) >= 24 else data['low'].min(),
            "week_high": data['high'].iloc[-168:].max() if len(data) >= 168 else data['high'].max(),
            "week_low": data['low'].iloc[-168:].min() if len(data) >= 168 else data['low'].min(),
        }
        
        # Add ATH (All Time High) and ATL (All Time Low) if we have enough data
        levels["ath"] = data['high'].max()
        levels["atl"] = data['low'].min()
        
        # Calculate psychological levels (important for crypto)
        current_price = latest['close']
        magnitude = 10 ** int(np.log10(current_price))
        
        # Add psychological price levels
        levels.update({
            "psych_level_1": np.ceil(current_price / magnitude) * magnitude,
            "psych_level_2": np.floor(current_price / magnitude) * magnitude,
            "psych_level_3": np.ceil(current_price / (magnitude/2)) * (magnitude/2),
            "psych_level_4": np.floor(current_price / (magnitude/2)) * (magnitude/2),
        })
        
        # Add Fibonacci retracement levels using recent high and low
        if len(data) >= 30:
            recent_high = data['high'].iloc[-30:].max()
            recent_low = data['low'].iloc[-30:].min()
            range_size = recent_high - recent_low
            
            levels.update({
                "fib_236": recent_high - (range_size * 0.236),
                "fib_382": recent_high - (range_size * 0.382),
                "fib_500": recent_high - (range_size * 0.5),
                "fib_618": recent_high - (range_size * 0.618),
                "fib_786": recent_high - (range_size * 0.786)
            })
        
        # Add MAs if we have enough data
        if len(data) >= 200:
            levels["ma50"] = data['close'].rolling(50).mean().iloc[-1]
            levels["ma100"] = data['close'].rolling(100).mean().iloc[-1]
            levels["ma200"] = data['close'].rolling(200).mean().iloc[-1]
        
        return levels
    
    def get_typical_spreads(self, symbol: str) -> Dict[str, float]:
        """Get typical spread information for crypto pairs"""
        # Crypto spreads are typically calculated differently than forex
        # Often represented as a percentage of price rather than absolute pips
        crypto_spreads = {
            # Major cryptos
            "BTCUSD": {"min": 0.05, "typical": 0.1, "high": 0.3},
            "ETHUSD": {"min": 0.07, "typical": 0.15, "high": 0.4},
            "LTCUSD": {"min": 0.1, "typical": 0.2, "high": 0.5},
            "XRPUSD": {"min": 0.15, "typical": 0.3, "high": 0.7},
            "BCHUSD": {"min": 0.1, "typical": 0.2, "high": 0.5},
            "EOSUSD": {"min": 0.15, "typical": 0.3, "high": 0.7},
            # Default for others
            "DEFAULT": {"min": 0.15, "typical": 0.3, "high": 0.7}
        }
        
        if symbol in crypto_spreads:
            return crypto_spreads[symbol]
        return crypto_spreads["DEFAULT"]
        
    def get_market_hours_filter(self, symbol: str) -> Dict[str, Any]:
        """Get market hours filter for crypto (24/7 market)"""
        # Cryptocurrencies are traded 24/7
        return {
            "is_24h_market": True,
            "has_session_breaks": False,
            "weekend_trading": True,
            "high_liquidity_hours": [
                # UTC hours of typically higher liquidity
                {"start": time(13, 0), "end": time(21, 0), "description": "US Trading Hours"},
                {"start": time(0, 0), "end": time(8, 0), "description": "Asia Trading Hours"}
            ],
            "low_liquidity_hours": [
                # UTC hours of typically lower liquidity
                {"start": time(21, 0), "end": time(23, 59), "description": "Late US / Early Asia"},
                {"start": time(8, 0), "end": time(13, 0), "description": "Between Asia and US"}
            ]
        }
    
    def get_volatility_normalization_factor(self, symbol: str) -> float:
        """
        Get normalization factor for volatility.
        
        Returns a multiplier to standardize volatility across asset classes.
        """
        # Crypto typically has higher volatility than traditional assets
        crypto_vol_factors = {
            "BTCUSD": 0.5,  # Bitcoin is less volatile than many altcoins
            "ETHUSD": 0.6,
            "LTCUSD": 0.7,
            "XRPUSD": 0.8,
            # Default for other crypto
            "DEFAULT": 0.7
        }
        
        if symbol in crypto_vol_factors:
            return crypto_vol_factors[symbol]
        return crypto_vol_factors["DEFAULT"]
    
    def get_bitcoin_dominance_impact(self, symbol: str) -> Dict[str, float]:
        """
        Calculate the impact of Bitcoin dominance on this crypto asset.
        
        Returns a dictionary with correlation and impact metrics.
        """
        if symbol == "BTCUSD":
            return {"correlation": 1.0, "impact_factor": 0.0}
        
        # Example correlations with Bitcoin (these would ideally come from a database)
        btc_correlations = {
            "ETHUSD": 0.82,
            "LTCUSD": 0.78,
            "XRPUSD": 0.65,
            "BCHUSD": 0.75,
            "EOSUSD": 0.68,
            "DEFAULT": 0.7  # Default correlation for unknown cryptos
        }
        
        correlation = btc_correlations.get(symbol, btc_correlations["DEFAULT"])
        impact_factor = correlation * 0.8  # How much BTC movements affect this coin
        
        return {
            "correlation": correlation,
            "impact_factor": impact_factor,
            "risk_adjustment": 1 + (0.5 * correlation)  # Higher correlation means higher risk
        }


class StockAssetAdapter(BaseAssetAdapter):
    """Adapter for stock assets"""
    
    def normalize_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize stock data for consistent analysis"""
        asset = self.asset_registry.get_asset(symbol)
        if not asset:
            self.logger.warning(f"Stock asset not found in registry: {symbol}")
            return data
        
        # Create a copy to avoid modifying the original
        normalized = data.copy()
        
        # Ensure we use adjusted prices for stocks
        if 'adjusted_close' in data.columns:
            normalized['real_close'] = data['adjusted_close']
        
        # Calculate percentage moves (more relevant for stocks than pips)
        if all(col in data.columns for col in ['open', 'close']):
            normalized['percent_change'] = ((data['close'] - data['open']) / data['open']) * 100
        
        # Add volume indicators (important for stocks)
        if 'volume' in data.columns:
            normalized['volume_sma_10'] = data['volume'].rolling(10).mean()
            normalized['volume_change_pct'] = data['volume'].pct_change() * 100
            
            # Volume / Average Volume ratio
            normalized['relative_volume'] = data['volume'] / data['volume'].rolling(20).mean()
        
        # Add volatility 
        if all(col in data.columns for col in ['high', 'low']):
            normalized['volatility'] = (data['high'] - data['low']) / data['low'] * 100
        
        return normalized
    
    def calculate_volatility(self, data: pd.DataFrame, symbol: str, window: int = 14) -> pd.Series:
        """Calculate stock-specific volatility"""
        if data.empty or 'close' not in data.columns:
            self.logger.warning(f"Cannot calculate volatility for {symbol}")
            return pd.Series(index=data.index)
        
        # For stocks, use % volatility (similar to crypto, but with different typical ranges)
        # Calculate returns
        returns = data['close'].pct_change() * 100
        
        # Calculate rolling standard deviation of returns
        volatility = returns.rolling(window=window).std()
        
        # Annualize (assuming daily data, multiply by sqrt of trading days in a year)
        # Stocks typically use 252 trading days per year
        volatility_annualized = volatility * np.sqrt(252)
        
        return volatility
    
    def get_price_levels(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Get significant price levels for stocks"""
        if data.empty or 'close' not in data.columns:
            return {}
            
        # Get the latest prices
        latest = data.iloc[-1]
        
        # Initialize levels dictionary
        levels = {
            "current": latest['close'],
            "day_high": data['high'].iloc[-1],
            "day_low": data['low'].iloc[-1],
            "week_high": data['high'].iloc[-5:].max() if len(data) >= 5 else data['high'].max(),
            "week_low": data['low'].iloc[-5:].min() if len(data) >= 5 else data['low'].min(),
            "month_high": data['high'].iloc[-20:].max() if len(data) >= 20 else data['high'].max(),
            "month_low": data['low'].iloc[-20:].min() if len(data) >= 20 else data['low'].min(),
        }
        
        # Calculate 52-week high/low (common stock metric)
        if len(data) >= 252:  # Approximately 1 year of trading days
            levels["52w_high"] = data['high'].iloc[-252:].max()
            levels["52w_low"] = data['low'].iloc[-252:].min()
        
        # Add moving averages if we have enough data
        if len(data) >= 200:
            levels["ma50"] = data['close'].rolling(50).mean().iloc[-1]
            levels["ma100"] = data['close'].rolling(100).mean().iloc[-1]
            levels["ma200"] = data['close'].rolling(200).mean().iloc[-1]
        
        # Add VWAP (Volume Weighted Average Price) - important for stocks
        if 'volume' in data.columns and len(data) >= 1:
            # Calculate daily VWAP - assuming data is for one day
            levels["vwap"] = (data['close'] * data['volume']).sum() / data['volume'].sum()
        
        return levels
    
    def get_typical_spreads(self, symbol: str) -> Dict[str, float]:
        """Get typical spread information for stocks"""
        # Stock spreads are typically represented in cents or as a percentage
        # This is a simplified example - real implementation would use market data
        stock_spreads = {
            # Large caps generally have tighter spreads
            "AAPL": {"min": 0.01, "typical": 0.01, "high": 0.02},
            "MSFT": {"min": 0.01, "typical": 0.01, "high": 0.02},
            "AMZN": {"min": 0.01, "typical": 0.02, "high": 0.05},
            # Mid caps
            "MID_CAP": {"min": 0.02, "typical": 0.05, "high": 0.10},
            # Small caps
            "SMALL_CAP": {"min": 0.05, "typical": 0.10, "high": 0.25},
            # Default
            "DEFAULT": {"min": 0.03, "typical": 0.07, "high": 0.15}
        }
        
        # Check if symbol is directly available
        if symbol in stock_spreads:
            return stock_spreads[symbol]
        
        # If not, try to classify by price or market cap (simplified here)
        asset = self.asset_registry.get_asset(symbol)
        if asset and asset.metadata:
            market_cap = asset.metadata.get("market_cap", 0)
            if market_cap > 100e9:  # Large cap (>100B)
                return stock_spreads["AAPL"]  # Use large cap example
            elif market_cap > 10e9:  # Mid cap (>10B)
                return stock_spreads["MID_CAP"]
            elif market_cap > 0:  # Small cap
                return stock_spreads["SMALL_CAP"]
        
        # Default case
        return stock_spreads["DEFAULT"]
    
    def get_market_hours_filter(self, symbol: str) -> Dict[str, Any]:
        """Get market hours filter for stocks"""
        # Stocks have defined trading hours
        regular_hours = {
            "is_24h_market": False,
            "has_session_breaks": False,
            "weekend_trading": False,
            "regular_hours": [
                {"start": time(9, 30), "end": time(16, 0), "description": "US Market Hours"}
            ],
            "extended_hours": [
                {"start": time(4, 0), "end": time(9, 30), "description": "Pre-Market"},
                {"start": time(16, 0), "end": time(20, 0), "description": "After-Hours"}
            ],
            "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        }
        
        # Check for different market hours based on exchange
        asset = self.asset_registry.get_asset(symbol)
        if asset and asset.metadata:
            exchange = asset.metadata.get("exchange", "").upper()
            
            if exchange in ["LSE", "FTSE"]:
                # London Stock Exchange hours
                return {
                    "is_24h_market": False,
                    "has_session_breaks": False,
                    "weekend_trading": False,
                    "regular_hours": [
                        {"start": time(8, 0), "end": time(16, 30), "description": "London Market Hours"}
                    ],
                    "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                }
            elif exchange in ["TSE", "JPX"]:
                # Tokyo Stock Exchange hours
                return {
                    "is_24h_market": False,
                    "has_session_breaks": True,
                    "weekend_trading": False,
                    "regular_hours": [
                        {"start": time(9, 0), "end": time(11, 30), "description": "Tokyo AM Session"},
                        {"start": time(12, 30), "end": time(15, 0), "description": "Tokyo PM Session"}
                    ],
                    "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
                }
        
        # Default to US market hours
        return regular_hours
    
    def get_sector_correlation(self, symbol: str) -> Dict[str, Any]:
        """
        Get sector correlation information for a stock.
        
        Returns correlation metrics with the stock's sector and broader market.
        """
        # This would ideally be calculated from historical data
        # Simplified implementation with example values
        asset = self.asset_registry.get_asset(symbol)
        if not asset or not asset.metadata:
            return {
                "sector_correlation": 0.7,  # Default correlation with sector
                "market_correlation": 0.5   # Default correlation with broader market
            }
        
        sector = asset.metadata.get("sector", "")
        
        # Example sector-specific correlations
        sector_correlations = {
            "Technology": {"sector": 0.82, "market": 0.75},
            "Healthcare": {"sector": 0.75, "market": 0.6},
            "Financial": {"sector": 0.85, "market": 0.8},
            "Energy": {"sector": 0.8, "market": 0.65},
            "Consumer": {"sector": 0.7, "market": 0.65},
            "Utilities": {"sector": 0.6, "market": 0.4},
            "DEFAULT": {"sector": 0.7, "market": 0.5}
        }
        
        return sector_correlations.get(sector, sector_correlations["DEFAULT"])


class CommodityAssetAdapter(BaseAssetAdapter):
    """Adapter for commodity assets"""
    
    def normalize_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize commodity data for consistent analysis"""
        asset = self.asset_registry.get_asset(symbol)
        if not asset:
            self.logger.warning(f"Commodity asset not found in registry: {symbol}")
            return data
        
        # Create a copy to avoid modifying the original
        normalized = data.copy()
        
        # Calculate percentage moves
        if all(col in data.columns for col in ['open', 'close']):
            normalized['percent_change'] = ((data['close'] - data['open']) / data['open']) * 100
            
        # Handle rollover gaps if this is a futures-based commodity
        if asset.metadata and asset.metadata.get("instrument_type") == "futures":
            # Check for large gaps that might indicate contract rollover
            if 'close' in data.columns:
                # Calculate returns
                returns = data['close'].pct_change()
                
                # Identify potential rollover points (abnormally large returns)
                std_dev = returns.std()
                rollover_threshold = 3 * std_dev  # 3 standard deviations
                
                # Mark potential rollover dates
                normalized['potential_rollover'] = (
                    (returns.abs() > rollover_threshold) & 
                    (returns.abs() > 0.03)  # At least 3% move
                )
                
                # Adjust for rollovers in continuous analysis
                if normalized['potential_rollover'].any():
                    self.logger.info(f"Potential contract rollovers detected for {symbol}")
                    
                    # Flag these dates in the data for special handling
                    # In a real system, you might adjust the prices here to create a continuous series
        
        # Add seasonality factors for commodities
        if 'datetime' in data.columns or isinstance(data.index, pd.DatetimeIndex):
            date_series = data.index if isinstance(data.index, pd.DatetimeIndex) else pd.to_datetime(data['datetime'])
            normalized['month'] = date_series.month
            normalized['quarter'] = date_series.quarter
            
            # Add seasonal factors for certain commodities
            if symbol in ["NATGAS", "NGAS", "NG"]:  # Natural Gas
                # Natural Gas has strong seasonality (higher in winter)
                normalized['season_factor'] = normalized['month'].apply(
                    lambda m: 1.2 if m in [11, 12, 1, 2] else 
                             (0.9 if m in [5, 6, 7, 8] else 1.0)
                )
            elif symbol in ["CORN", "WHEAT", "SOYBEAN"]:  # Agricultural
                # Add growing season factor
                normalized['season_factor'] = normalized['month'].apply(
                    lambda m: 1.1 if m in [4, 5, 6] else  # Growing season
                             (1.2 if m in [9, 10] else 1.0)  # Harvest
                )
        
        return normalized
    
    def calculate_volatility(self, data: pd.DataFrame, symbol: str, window: int = 14) -> pd.Series:
        """Calculate commodity-specific volatility"""
        if data.empty or 'close' not in data.columns:
            self.logger.warning(f"Cannot calculate volatility for {symbol}")
            return pd.Series(index=data.index)
        
        # For commodities, use a volatility measure that handles seasonal factors
        # Calculate percentage returns
        returns = data['close'].pct_change() * 100
        
        # Get asset metadata for seasonal adjustments
        asset = self.asset_registry.get_asset(symbol)
        
        # Apply seasonality adjustment if appropriate
        if asset and asset.metadata and asset.metadata.get("has_seasonality", False):
            # Get month of year
            if isinstance(data.index, pd.DatetimeIndex):
                months = data.index.month
            elif 'datetime' in data.columns:
                months = pd.to_datetime(data['datetime']).dt.month
            else:
                months = None
                
            if months is not None:
                # Apply seasonal volatility adjustment
                seasonal_factors = self._get_seasonal_volatility_factors(symbol)
                
                # Adjust returns by seasonal factor
                for month, factor in seasonal_factors.items():
                    if factor != 1.0:
                        month_mask = (months == month)
                        returns.loc[month_mask] = returns.loc[month_mask] / factor
        
        # Calculate rolling standard deviation of adjusted returns
        volatility = returns.rolling(window=window).std()
        
        # Annualize (for commodities, typically use 252 trading days)
        volatility_annualized = volatility * np.sqrt(252)
        
        return volatility_annualized
    
    def _get_seasonal_volatility_factors(self, symbol: str) -> Dict[int, float]:
        """Get seasonal volatility adjustment factors by month"""
        # This would ideally come from analysis of historical volatility
        # Here providing examples for common commodities
        
        # Month -> factor mappings (higher factor = higher volatility)
        seasonal_factors = {
            # Default (no seasonal adjustment)
            "DEFAULT": {m: 1.0 for m in range(1, 13)},
            
            # Natural Gas (more volatile in winter)
            "NATGAS": {
                1: 1.3, 2: 1.3, 3: 1.2,  # Winter: higher volatility
                4: 1.0, 5: 0.9, 6: 0.8,  # Spring/summer: lower
                7: 0.8, 8: 0.9, 9: 1.0,  # Summer/fall: lower
                10: 1.1, 11: 1.2, 12: 1.3  # Fall/winter: higher volatility
            },
            
            # Agricultural commodities (more volatile around planting/harvest)
            "CORN": {
                1: 0.9, 2: 0.9, 3: 1.0,  # Winter: lower volatility
                4: 1.2, 5: 1.3, 6: 1.2,  # Planting: higher volatility
                7: 1.0, 8: 1.1, 9: 1.3,  # Growing/early harvest: higher
                10: 1.2, 11: 1.0, 12: 0.9  # Post harvest: decreasing
            }
        }
        
        # Try to match symbol to known patterns
        for key in seasonal_factors.keys():
            if key in symbol:
                return seasonal_factors[key]
        
        # Return default if no match
        return seasonal_factors["DEFAULT"]
    
    def get_price_levels(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Get significant price levels for commodities"""
        if data.empty or 'close' not in data.columns:
            return {}
            
        # Get the latest prices
        latest = data.iloc[-1]
        
        # Initialize levels dictionary with basic price levels
        levels = {
            "current": latest['close'],
            "day_high": data['high'].iloc[-1],
            "day_low": data['low'].iloc[-1],
            "week_high": data['high'].iloc[-5:].max() if len(data) >= 5 else data['high'].max(),
            "week_low": data['low'].iloc[-5:].min() if len(data) >= 5 else data['low'].min(),
        }
        
        # Add moving averages if we have enough data
        if len(data) >= 200:
            levels["ma50"] = data['close'].rolling(50).mean().iloc[-1]
            levels["ma100"] = data['close'].rolling(100).mean().iloc[-1]
            levels["ma200"] = data['close'].rolling(200).mean().iloc[-1]
        
        # Add yearly levels (important for seasonal commodities)
        if len(data) >= 252:  # Approximately 1 year of trading days
            levels["52w_high"] = data['high'].iloc[-252:].max()
            levels["52w_low"] = data['low'].iloc[-252:].min()
            
            # Add seasonal pivot levels
            # Get asset to check for seasonality
            asset = self.asset_registry.get_asset(symbol)
            if asset and asset.metadata and asset.metadata.get("has_seasonality", False):
                # If this asset has seasonality, add previous seasonal price levels
                # This is a simplified example - real implementation would be more sophisticated
                if isinstance(data.index, pd.DatetimeIndex):
                    current_month = data.index[-1].month
                elif 'datetime' in data.columns:
                    current_month = pd.to_datetime(data['datetime']).iloc[-1].month
                else:
                    current_month = datetime.now().month
                
                # Find the same month last year
                prev_year_same_month_mask = (
                    pd.DatetimeIndex(data.index if isinstance(data.index, pd.DatetimeIndex) 
                                    else pd.to_datetime(data['datetime'])).month == current_month
                ) & (
                    pd.DatetimeIndex(data.index if isinstance(data.index, pd.DatetimeIndex)
                                    else pd.to_datetime(data['datetime'])).year == 
                    (datetime.now().year - 1)
                )
                
                if prev_year_same_month_mask.any():
                    prev_year_data = data.loc[prev_year_same_month_mask]
                    levels["prev_year_high"] = prev_year_data['high'].max()
                    levels["prev_year_low"] = prev_year_data['low'].min()
                    levels["prev_year_close"] = prev_year_data['close'].iloc[-1] if not prev_year_data.empty else None
        
        return levels
    
    def get_typical_spreads(self, symbol: str) -> Dict[str, float]:
        """Get typical spread information for commodities"""
        # Commodity spreads vary by type and volume
        commodity_spreads = {
            # Energy
            "CL": {"min": 0.01, "typical": 0.02, "high": 0.05},  # Crude oil
            "NATGAS": {"min": 0.001, "typical": 0.002, "high": 0.005},  # Natural gas
            # Precious metals
            "GC": {"min": 0.1, "typical": 0.3, "high": 0.5},  # Gold
            "SI": {"min": 0.01, "typical": 0.02, "high": 0.04},  # Silver
            # Agricultural
            "ZC": {"min": 0.25, "typical": 0.5, "high": 1.0},  # Corn
            "ZW": {"min": 0.25, "typical": 0.5, "high": 1.0},  # Wheat
            # Default
            "DEFAULT": {"min": 0.05, "typical": 0.1, "high": 0.3}
        }
        
        # Try exact match first
        if symbol in commodity_spreads:
            return commodity_spreads[symbol]
        
        # Try partial match
        for key in commodity_spreads:
            if key in symbol:
                return commodity_spreads[key]
        
        # Default case
        return commodity_spreads["DEFAULT"]
    
    def get_market_hours_filter(self, symbol: str) -> Dict[str, Any]:
        """Get market hours filter for commodities"""
        # Commodity market hours vary by type
        energy_hours = {
            "is_24h_market": False,
            "has_session_breaks": False,
            "weekend_trading": False,
            "regular_hours": [
                {"start": time(9, 0), "end": time(14, 30), "description": "Regular Trading"}
            ],
            "extended_hours": [
                {"start": time(18, 0), "end": time(8, 0), "description": "Electronic Trading"}
            ],
            "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        }
        
        metals_hours = {
            "is_24h_market": True,
            "has_session_breaks": False,
            "weekend_trading": False,
            "regular_hours": [
                {"start": time(0, 0), "end": time(23, 59), "description": "24hr Trading"}
            ],
            "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        }
        
        agri_hours = {
            "is_24h_market": False,
            "has_session_breaks": False,
            "weekend_trading": False,
            "regular_hours": [
                {"start": time(9, 30), "end": time(14, 0), "description": "Regular Trading"}
            ],
            "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        }
        
        # Determine commodity type
        asset = self.asset_registry.get_asset(symbol)
        if not asset or not asset.metadata:
            return energy_hours  # Default to energy hours
        
        commodity_type = asset.metadata.get("commodity_type", "")
        
        if commodity_type in ["energy", "oil", "gas"]:
            return energy_hours
        elif commodity_type in ["metal", "gold", "silver"]:
            return metals_hours
        elif commodity_type in ["agricultural", "grain", "livestock"]:
            return agri_hours
            
        # Default
        return energy_hours


class IndexAssetAdapter(BaseAssetAdapter):
    """Adapter for market indices"""
    
    def normalize_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize index data for consistent analysis"""
        if not isinstance(data, pd.DataFrame) or data.empty:
            return data
            
        # Create a copy to avoid modifying the original
        normalized = data.copy()
        
        # Calculate percentage moves (most relevant for indices)
        if all(col in data.columns for col in ['open', 'close']):
            normalized['percent_change'] = ((data['close'] - data['open']) / data['open']) * 100
        
        # Add relative strength compared to benchmark (if available)
        asset = self.asset_registry.get_asset(symbol)
        if asset and asset.metadata:
            benchmark = asset.metadata.get("benchmark")
            if benchmark:
                # This is just a placeholder - actual implementation would compare to benchmark data
                normalized['relative_strength'] = 0.0
        
        # Add index-specific volatility measure
        if 'close' in data.columns:
            # Daily volatility (percentage)
            normalized['daily_volatility'] = data['close'].pct_change().abs() * 100
            
        # Add breadth indicators if available
        if 'advancing' in data.columns and 'declining' in data.columns:
            normalized['advance_decline_ratio'] = data['advancing'] / data['declining']
        
        return normalized
    
    def calculate_volatility(self, data: pd.DataFrame, symbol: str, window: int = 14) -> pd.Series:
        """Calculate index-specific volatility"""
        if data.empty or 'close' not in data.columns:
            self.logger.warning(f"Cannot calculate volatility for {symbol}")
            return pd.Series(index=data.index)
        
        # For indices, standard percentage volatility is appropriate
        returns = data['close'].pct_change() * 100
        
        # Calculate rolling standard deviation of returns
        volatility = returns.rolling(window=window).std()
        
        # Annualize (standard is 252 trading days for indices)
        volatility_annualized = volatility * np.sqrt(252)
        
        return volatility_annualized
    
    def get_price_levels(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Get significant price levels for indices"""
        if data.empty or 'close' not in data.columns:
            return {}
            
        # Get the latest prices
        latest = data.iloc[-1]
        
        # Initialize levels dictionary
        levels = {
            "current": latest['close'],
            "day_high": data['high'].iloc[-1],
            "day_low": data['low'].iloc[-1],
            "week_high": data['high'].iloc[-5:].max() if len(data) >= 5 else data['high'].max(),
            "week_low": data['low'].iloc[-5:].min() if len(data) >= 5 else data['low'].min(),
        }
        
        # Add moving averages - very important for indices
        if len(data) >= 200:
            levels["ma20"] = data['close'].rolling(20).mean().iloc[-1]
            levels["ma50"] = data['close'].rolling(50).mean().iloc[-1]
            levels["ma100"] = data['close'].rolling(100).mean().iloc[-1]
            levels["ma200"] = data['close'].rolling(200).mean().iloc[-1]
        
        # Add yearly levels
        if len(data) >= 252:  # Approximately 1 year of trading days
            levels["52w_high"] = data['high'].iloc[-252:].max()
            levels["52w_low"] = data['low'].iloc[-252:].min()
            
            # Add YTD performance
            year_start_idx = data.index[data.index.year == data.index[-1].year][0]
            year_start_close = data.loc[year_start_idx, 'close']
            levels["ytd_change_percent"] = (latest['close'] - year_start_close) / year_start_close * 100
            
        # Add psychological round levels (important for indices)
        current_price = latest['close']
        
        # Determine appropriate magnitude for round numbers based on index value
        if current_price > 10000:
            round_factors = [1000, 500, 100]
        elif current_price > 1000:
            round_factors = [500, 100, 50]
        elif current_price > 100:
            round_factors = [100, 50, 10]
        else:
            round_factors = [10, 5, 1]
            
        # Add key psychological levels
        for factor in round_factors:
            levels[f"psych_above_{factor}"] = np.ceil(current_price / factor) * factor
            levels[f"psych_below_{factor}"] = np.floor(current_price / factor) * factor
        
        return levels
    
    def get_typical_spreads(self, symbol: str) -> Dict[str, float]:
        """Get typical spread information for indices"""
        # Index spreads are typically represented in points or percentage
        index_spreads = {
            # Major indices
            "SPX500": {"min": 0.5, "typical": 0.7, "high": 1.0},  # S&P 500
            "DJIA": {"min": 1.0, "typical": 2.0, "high": 3.0},   # Dow Jones
            "NASDAQ": {"min": 0.75, "typical": 1.0, "high": 1.5}, # NASDAQ
            "NDX": {"min": 1.0, "typical": 1.5, "high": 2.5},    # NASDAQ-100
            "DAX": {"min": 1.0, "typical": 1.5, "high": 2.0},    # German DAX
            "FTSE": {"min": 0.8, "typical": 1.0, "high": 1.5},   # FTSE 100
            "NIKKEI": {"min": 10, "typical": 15, "high": 25},     # Nikkei 225
            # Default
            "DEFAULT": {"min": 1.0, "typical": 1.5, "high": 2.5}
        }
        
        # Try exact match first
        if symbol in index_spreads:
            return index_spreads[symbol]
        
        # Try partial match
        for key in index_spreads:
            if key in symbol:
                return index_spreads[key]
        
        # Default case
        return index_spreads["DEFAULT"]
    
    def get_market_hours_filter(self, symbol: str) -> Dict[str, Any]:
        """Get market hours filter for indices"""
        # Index hours follow their respective exchanges
        us_index_hours = {
            "is_24h_market": False,
            "has_session_breaks": False,
            "weekend_trading": False,
            "regular_hours": [
                {"start": time(9, 30), "end": time(16, 0), "description": "Regular Trading"}
            ],
            "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        }
        
        eu_index_hours = {
            "is_24h_market": False,
            "has_session_breaks": False,
            "weekend_trading": False,
            "regular_hours": [
                {"start": time(8, 0), "end": time(16, 30), "description": "Regular Trading"}
            ],
            "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        }
        
        asia_index_hours = {
            "is_24h_market": False,
            "has_session_breaks": True,
            "weekend_trading": False,
            "regular_hours": [
                {"start": time(9, 0), "end": time(11, 30), "description": "Morning Session"},
                {"start": time(12, 30), "end": time(15, 0), "description": "Afternoon Session"}
            ],
            "trading_days": ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
        }
        
        # Determine region
        us_indices = ["SPX", "DJIA", "NASDAQ", "NDX", "RUSSELL"]
        eu_indices = ["DAX", "FTSE", "CAC", "STOXX", "IBEX"]
        asia_indices = ["NIKKEI", "HSI", "ASX", "KOSPI"]
        
        for idx in us_indices:
            if idx in symbol:
                return us_index_hours
                
        for idx in eu_indices:
            if idx in symbol:
                return eu_index_hours
                
        for idx in asia_indices:
            if idx in symbol:
                return asia_index_hours
                
        # Default to US hours
        return us_index_hours
    
    def get_vix_relationship(self, symbol: str) -> Dict[str, Any]:
        """
        Get information about this index's relationship with volatility indices
        
        For US indices, this would be relationship with VIX
        For other markets, their respective volatility indices
        """
        # This would typically be computed from historical data
        # Simplified example implementation
        relationships = {
            "SPX": {"vix_inverse_correlation": -0.8, "typical_vix_range": [10, 30]},
            "DJIA": {"vix_inverse_correlation": -0.75, "typical_vix_range": [10, 30]},
            "NASDAQ": {"vix_inverse_correlation": -0.82, "typical_vix_range": [15, 35]},
            "DAX": {"vstoxx_inverse_correlation": -0.78, "typical_vstoxx_range": [12, 32]},
            "DEFAULT": {"volatility_inverse_correlation": -0.7, "typical_vol_range": [15, 30]}
        }
        
        # Look for index in supported list
        for idx, data in relationships.items():
            if idx in symbol:
                return data
        
        # Default
        return relationships["DEFAULT"]


# Create factory for asset adapters
class AssetAdapterFactory:
    """Factory class for creating appropriate asset adapters"""
    
    @staticmethod
    def create_adapter(asset_class: AssetClass, asset_registry: AssetRegistry) -> BaseAssetAdapter:
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
            return ForexAssetAdapter(asset_registry)  # Default to forex adapter
