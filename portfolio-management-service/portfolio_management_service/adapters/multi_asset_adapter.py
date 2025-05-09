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
import os
import httpx

from common_lib.multi_asset.interfaces import (
    AssetClassEnum,
    MarketTypeEnum,
    IMultiAssetService
)

logger = logging.getLogger(__name__)


class MultiAssetServiceAdapter(IMultiAssetService):
    """
    Adapter for MultiAssetService that implements the common interface.
    
    This adapter can either wrap an actual service instance or provide
    standalone functionality to avoid circular dependencies.
    """
    
    def __init__(self, service_instance=None, config: Dict[str, Any] = None):
        """
        Initialize the adapter.
        
        Args:
            service_instance: Optional service instance to wrap
            config: Configuration parameters
        """
        self.service = service_instance
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Get analysis engine URL from config or environment
        analysis_engine_base_url = self.config.get(
            "analysis_engine_base_url", 
            os.environ.get("ANALYSIS_ENGINE_BASE_URL", "http://analysis-engine-service:8000")
        )
        
        # Set up the client with resolved URL
        self.client = httpx.AsyncClient(
            base_url=f"{analysis_engine_base_url.rstrip('/')}/api/v1",
            timeout=30.0
        )
        
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
        
        # Try to get from API
        try:
            response = self.client.get(f"/assets/{symbol}")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            self.logger.warning(f"Error getting asset info from API: {str(e)}")
        
        # Fallback to default assets
        return self._default_assets.get(symbol, {})
    
    def list_assets_by_class(self, asset_class: AssetClassEnum) -> List[str]:
        """List all assets of a specific class."""
        if self.service:
            try:
                return self.service.list_assets_by_class(asset_class)
            except Exception as e:
                self.logger.warning(f"Error listing assets by class from service: {str(e)}")
        
        # Try to get from API
        try:
            response = self.client.get(f"/assets", params={"asset_class": asset_class.value})
            if response.status_code == 200:
                data = response.json()
                return [asset["symbol"] for asset in data.get("assets", [])]
        except Exception as e:
            self.logger.warning(f"Error listing assets by class from API: {str(e)}")
        
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
        
        # Try to get from API
        try:
            response = self.client.get(f"/asset-groups/{group_name}")
            if response.status_code == 200:
                data = response.json()
                return data.get("symbols", [])
        except Exception as e:
            self.logger.warning(f"Error getting asset group from API: {str(e)}")
        
        # Fallback to default groups
        return self._default_groups.get(group_name, [])
    
    def get_trading_parameters(self, symbol: str) -> Dict[str, Any]:
        """Get trading parameters for a symbol."""
        if self.service:
            try:
                return self.service.get_trading_parameters(symbol)
            except Exception as e:
                self.logger.warning(f"Error getting trading parameters from service: {str(e)}")
        
        # Try to get from API
        try:
            response = self.client.get(f"/assets/{symbol}/trading-parameters")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            self.logger.warning(f"Error getting trading parameters from API: {str(e)}")
        
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
        
        # Try to get from API
        try:
            response = self.client.get(f"/assets/{symbol}/analysis-parameters")
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            self.logger.warning(f"Error getting analysis parameters from API: {str(e)}")
        
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
