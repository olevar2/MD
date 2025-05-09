"""
Multi-Asset Adapter Module

This module provides adapters for multi-asset functionality from analysis-engine-service,
helping to break circular dependencies between services.
"""
import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime

from common_lib.multi_asset.interfaces import (
    AssetClassEnum,
    MarketTypeEnum,
    IMultiAssetService,
    IAssetRegistry
)

logger = logging.getLogger(__name__)


class MultiAssetServiceAdapter(IMultiAssetService):
    """
    Adapter for MultiAssetService that implements the common interface.
    
    This adapter can either wrap an actual service instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, service_instance=None):
        """
        Initialize the adapter.
        
        Args:
            service_instance: Optional service instance to wrap
        """
        self.service = service_instance
        self.logger = logging.getLogger(__name__)
        
        # Default asset information for common forex pairs
        self._default_assets = {
            "EUR/USD": {
                "symbol": "EUR/USD",
                "display_name": "Euro / US Dollar",
                "asset_class": AssetClassEnum.FOREX,
                "market_type": MarketTypeEnum.SPOT,
                "base_currency": "EUR",
                "quote_currency": "USD",
                "trading_parameters": {
                    "pip_value": 0.0001,
                    "min_quantity": 0.01,
                    "quantity_precision": 2,
                    "price_precision": 5,
                    "margin_rate": 0.03,
                    "trading_fee": 0.0
                },
                "available_timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
                "metadata": {}
            },
            "GBP/USD": {
                "symbol": "GBP/USD",
                "display_name": "British Pound / US Dollar",
                "asset_class": AssetClassEnum.FOREX,
                "market_type": MarketTypeEnum.SPOT,
                "base_currency": "GBP",
                "quote_currency": "USD",
                "trading_parameters": {
                    "pip_value": 0.0001,
                    "min_quantity": 0.01,
                    "quantity_precision": 2,
                    "price_precision": 5,
                    "margin_rate": 0.03,
                    "trading_fee": 0.0
                },
                "available_timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
                "metadata": {}
            },
            "USD/JPY": {
                "symbol": "USD/JPY",
                "display_name": "US Dollar / Japanese Yen",
                "asset_class": AssetClassEnum.FOREX,
                "market_type": MarketTypeEnum.SPOT,
                "base_currency": "USD",
                "quote_currency": "JPY",
                "trading_parameters": {
                    "pip_value": 0.01,
                    "min_quantity": 0.01,
                    "quantity_precision": 2,
                    "price_precision": 3,
                    "margin_rate": 0.03,
                    "trading_fee": 0.0
                },
                "available_timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
                "metadata": {}
            }
        }
        
        # Default asset groups
        self._default_groups = {
            "major_pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"],
            "eur_crosses": ["EUR/GBP", "EUR/JPY", "EUR/CHF", "EUR/AUD", "EUR/CAD", "EUR/NZD"],
            "gbp_crosses": ["GBP/JPY", "GBP/CHF", "GBP/AUD", "GBP/CAD", "GBP/NZD"]
        }
    
    def get_asset_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about an asset."""
        if self.service:
            try:
                return self.service.get_asset_info(symbol)
            except Exception as e:
                self.logger.warning(f"Error getting asset info from service: {str(e)}")
        
        # Fallback to default assets
        return self._default_assets.get(symbol, {})
    
    def list_assets_by_class(self, asset_class: AssetClassEnum) -> List[str]:
        """List all assets of a specific class."""
        if self.service:
            try:
                return self.service.list_assets_by_class(asset_class)
            except Exception as e:
                self.logger.warning(f"Error listing assets by class from service: {str(e)}")
        
        # Fallback to default assets
        return [
            symbol for symbol, info in self._default_assets.items()
            if info.get("asset_class") == asset_class
        ]
    
    def get_asset_group(self, group_name: str) -> List[str]:
        """Get all symbols in a named group."""
        if self.service:
            try:
                return self.service.get_asset_group(group_name)
            except Exception as e:
                self.logger.warning(f"Error getting asset group from service: {str(e)}")
        
        # Fallback to default groups
        return self._default_groups.get(group_name, [])
    
    def get_trading_parameters(self, symbol: str) -> Dict[str, Any]:
        """Get trading parameters for a symbol."""
        if self.service:
            try:
                return self.service.get_trading_parameters(symbol)
            except Exception as e:
                self.logger.warning(f"Error getting trading parameters from service: {str(e)}")
        
        # Fallback to default assets
        asset_info = self._default_assets.get(symbol, {})
        return asset_info.get("trading_parameters", {})
    
    def normalize_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize data for the specified symbol."""
        if self.service:
            try:
                return self.service.normalize_data(data, symbol)
            except Exception as e:
                self.logger.warning(f"Error normalizing data from service: {str(e)}")
        
        # Fallback implementation
        normalized = data.copy()
        
        # Ensure datetime format for timestamp
        if "timestamp" in normalized.columns:
            normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
            
        # Calculate percentage changes
        if all(col in normalized.columns for col in ['open', 'close']):
            normalized['pct_change'] = (normalized['close'] - normalized['open']) / normalized['open'] * 100
        
        # Asset-specific normalization
        asset_info = self.get_asset_info(symbol)
        asset_class = asset_info.get("asset_class")
        
        if asset_class == AssetClassEnum.FOREX:
            # Get pip value
            trading_params = asset_info.get("trading_parameters", {})
            pip_value = trading_params.get("pip_value", 0.0001)
            
            # Calculate pip movements
            if all(col in normalized.columns for col in ['open', 'close']):
                normalized['pips_change'] = (normalized['close'] - normalized['open']) / pip_value
                
            if all(col in normalized.columns for col in ['high', 'low']):
                normalized['range_pips'] = (normalized['high'] - normalized['low']) / pip_value
        
        return normalized
    
    def calculate_volatility(self, data: pd.DataFrame, symbol: str, window: int = 14) -> pd.Series:
        """Calculate symbol-specific volatility."""
        if self.service:
            try:
                return self.service.calculate_volatility(data, symbol, window)
            except Exception as e:
                self.logger.warning(f"Error calculating volatility from service: {str(e)}")
        
        # Fallback implementation - use ATR (Average True Range)
        if all(col in data.columns for col in ['high', 'low', 'close']):
            high = data['high']
            low = data['low']
            close = data['close'].shift(1)
            
            tr1 = high - low
            tr2 = abs(high - close)
            tr3 = abs(low - close)
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            atr = true_range.rolling(window=window).mean()
            
            return atr
        
        # Fallback to standard deviation if OHLC not available
        if 'close' in data.columns:
            return data['close'].pct_change().rolling(window=window).std() * 100
        
        return pd.Series(index=data.index)
    
    def get_price_levels(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """Get significant price levels for the symbol."""
        if self.service:
            try:
                return self.service.get_price_levels(data, symbol)
            except Exception as e:
                self.logger.warning(f"Error getting price levels from service: {str(e)}")
        
        # Fallback implementation - calculate basic support/resistance levels
        if len(data) < 20 or 'close' not in data.columns:
            return {}
        
        # Get recent price data
        recent_data = data.tail(20)
        
        # Calculate levels
        current_price = recent_data['close'].iloc[-1]
        high = recent_data['high'].max() if 'high' in recent_data.columns else recent_data['close'].max()
        low = recent_data['low'].min() if 'low' in recent_data.columns else recent_data['close'].min()
        
        # Simple pivot points
        pivot = (high + low + current_price) / 3
        support1 = 2 * pivot - high
        resistance1 = 2 * pivot - low
        
        return {
            'current_price': current_price,
            'pivot': pivot,
            'support1': support1,
            'resistance1': resistance1,
            'recent_high': high,
            'recent_low': low
        }
    
    def get_analysis_parameters(self, symbol: str) -> Dict[str, Any]:
        """Get asset-specific analysis parameters."""
        if self.service:
            try:
                return self.service.get_analysis_parameters(symbol)
            except Exception as e:
                self.logger.warning(f"Error getting analysis parameters from service: {str(e)}")
        
        # Fallback implementation
        asset_info = self.get_asset_info(symbol)
        asset_class = asset_info.get("asset_class")
        trading_params = asset_info.get("trading_parameters", {})
        
        # Default parameters
        params = {
            "volatility_window": 14,
            "pip_value": trading_params.get("pip_value", 0.0001),
            "typical_spread": 2.0,  # Default spread in pips
            "position_sizing_factors": {
                "risk_per_trade": 0.02,  # 2% risk per trade
                "max_position_size": 0.1  # 10% of account
            }
        }
        
        # Asset-class specific adjustments
        if asset_class == AssetClassEnum.FOREX:
            params["pattern_precision"] = 5  # Decimal places for pattern recognition
            params["default_stop_atr_multiple"] = 1.5
        elif asset_class == AssetClassEnum.CRYPTO:
            params["pattern_precision"] = 2
            params["default_stop_atr_multiple"] = 2.0
            params["volatility_window"] = 20  # Longer window for crypto
        
        return params


class AssetRegistryAdapter(IAssetRegistry):
    """
    Adapter for AssetRegistry that implements the common interface.
    
    This adapter can either wrap an actual registry instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, registry_instance=None):
        """
        Initialize the adapter.
        
        Args:
            registry_instance: Optional registry instance to wrap
        """
        self.registry = registry_instance
        self.logger = logging.getLogger(__name__)
        
        # Default asset information for common forex pairs
        self._default_assets = {
            "EUR/USD": {
                "symbol": "EUR/USD",
                "display_name": "Euro / US Dollar",
                "asset_class": AssetClassEnum.FOREX,
                "market_type": MarketTypeEnum.SPOT,
                "base_currency": "EUR",
                "quote_currency": "USD",
                "trading_parameters": {
                    "pip_value": 0.0001,
                    "min_quantity": 0.01,
                    "quantity_precision": 2,
                    "price_precision": 5,
                    "margin_rate": 0.03,
                    "trading_fee": 0.0
                },
                "available_timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
                "metadata": {}
            },
            "GBP/USD": {
                "symbol": "GBP/USD",
                "display_name": "British Pound / US Dollar",
                "asset_class": AssetClassEnum.FOREX,
                "market_type": MarketTypeEnum.SPOT,
                "base_currency": "GBP",
                "quote_currency": "USD",
                "trading_parameters": {
                    "pip_value": 0.0001,
                    "min_quantity": 0.01,
                    "quantity_precision": 2,
                    "price_precision": 5,
                    "margin_rate": 0.03,
                    "trading_fee": 0.0
                },
                "available_timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
                "metadata": {}
            },
            "USD/JPY": {
                "symbol": "USD/JPY",
                "display_name": "US Dollar / Japanese Yen",
                "asset_class": AssetClassEnum.FOREX,
                "market_type": MarketTypeEnum.SPOT,
                "base_currency": "USD",
                "quote_currency": "JPY",
                "trading_parameters": {
                    "pip_value": 0.01,
                    "min_quantity": 0.01,
                    "quantity_precision": 2,
                    "price_precision": 3,
                    "margin_rate": 0.03,
                    "trading_fee": 0.0
                },
                "available_timeframes": ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1w"],
                "metadata": {}
            }
        }
        
        # Default asset groups
        self._default_groups = {
            "major_pairs": ["EUR/USD", "GBP/USD", "USD/JPY", "USD/CHF", "AUD/USD", "USD/CAD", "NZD/USD"],
            "eur_crosses": ["EUR/GBP", "EUR/JPY", "EUR/CHF", "EUR/AUD", "EUR/CAD", "EUR/NZD"],
            "gbp_crosses": ["GBP/JPY", "GBP/CHF", "GBP/AUD", "GBP/CAD", "GBP/NZD"]
        }
        
        # Default correlations
        self._default_correlations = {
            "EUR/USD_GBP/USD": 0.85,
            "EUR/USD_USD/JPY": -0.45,
            "GBP/USD_USD/JPY": -0.35
        }
    
    def get_asset(self, symbol: str) -> Optional[Any]:
        """Get asset definition by symbol."""
        if self.registry:
            try:
                return self.registry.get_asset(symbol)
            except Exception as e:
                self.logger.warning(f"Error getting asset from registry: {str(e)}")
        
        # Fallback to default assets
        return self._default_assets.get(symbol)
    
    def list_assets(self, 
                   asset_class: Optional[AssetClassEnum] = None,
                   market_type: Optional[MarketTypeEnum] = None,
                   currency: Optional[str] = None) -> List[Any]:
        """List assets filtered by various criteria."""
        if self.registry:
            try:
                return self.registry.list_assets(asset_class, market_type, currency)
            except Exception as e:
                self.logger.warning(f"Error listing assets from registry: {str(e)}")
        
        # Fallback to default assets
        result = list(self._default_assets.values())
        
        if asset_class:
            result = [a for a in result if a.get("asset_class") == asset_class]
        
        if market_type:
            result = [a for a in result if a.get("market_type") == market_type]
        
        if currency:
            result = [a for a in result if (
                a.get("base_currency") == currency or a.get("quote_currency") == currency
            )]
        
        return result
    
    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Get the most recent correlation between two assets."""
        if self.registry:
            try:
                return self.registry.get_correlation(symbol1, symbol2)
            except Exception as e:
                self.logger.warning(f"Error getting correlation from registry: {str(e)}")
        
        # Fallback to default correlations
        # Create a unique key for the correlation pair (order-independent)
        key = "_".join(sorted([symbol1, symbol2]))
        return self._default_correlations.get(key)
    
    def get_correlated_assets(self, symbol: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Get all assets correlated with the given symbol above the threshold."""
        if self.registry:
            try:
                return self.registry.get_correlated_assets(symbol, threshold)
            except Exception as e:
                self.logger.warning(f"Error getting correlated assets from registry: {str(e)}")
        
        # Fallback implementation
        result = []
        
        for key, correlation in self._default_correlations.items():
            symbols = key.split('_')
            if symbol not in symbols:
                continue
            
            if abs(correlation) >= threshold:
                other_symbol = symbols[0] if symbols[1] == symbol else symbols[1]
                asset = self.get_asset(other_symbol)
                
                if asset:
                    result.append({
                        "symbol": other_symbol,
                        "display_name": asset.get("display_name"),
                        "correlation": correlation,
                        "as_of_date": datetime.now(),
                        "asset_class": asset.get("asset_class")
                    })
        
        # Sort by absolute correlation value (descending)
        result.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return result
    
    def get_asset_group(self, group_name: str) -> List[Any]:
        """Get all assets in a named group."""
        if self.registry:
            try:
                return self.registry.get_asset_group(group_name)
            except Exception as e:
                self.logger.warning(f"Error getting asset group from registry: {str(e)}")
        
        # Fallback to default groups
        if group_name not in self._default_groups:
            return []
        
        return [self._default_assets.get(symbol) for symbol in self._default_groups[group_name] 
                if symbol in self._default_assets]
    
    def get_pip_value(self, symbol: str) -> Optional[float]:
        """Get the pip value for a symbol."""
        if self.registry:
            try:
                return self.registry.get_pip_value(symbol)
            except Exception as e:
                self.logger.warning(f"Error getting pip value from registry: {str(e)}")
        
        # Fallback to default assets
        asset = self._default_assets.get(symbol)
        if asset:
            return asset.get("trading_parameters", {}).get("pip_value")
        return None
    
    def get_trading_parameters(self, symbol: str) -> Dict[str, Any]:
        """Get trading parameters for a symbol."""
        if self.registry:
            try:
                return self.registry.get_trading_parameters(symbol)
            except Exception as e:
                self.logger.warning(f"Error getting trading parameters from registry: {str(e)}")
        
        # Fallback to default assets
        asset = self._default_assets.get(symbol)
        if not asset:
            return {}
        
        trading_params = asset.get("trading_parameters", {})
        return {
            "min_quantity": trading_params.get("min_quantity"),
            "quantity_precision": trading_params.get("quantity_precision"),
            "price_precision": trading_params.get("price_precision"),
            "pip_value": trading_params.get("pip_value"),
            "lot_size": trading_params.get("lot_size"),
            "margin_rate": trading_params.get("margin_rate"),
            "trading_fee": trading_params.get("trading_fee")
        }
