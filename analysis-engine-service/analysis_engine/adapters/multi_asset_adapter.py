"""
Multi-Asset Adapter Module

This module provides adapters for multi-asset functionality,
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
from analysis_engine.multi_asset.asset_registry import AssetClass
from analysis_engine.services.multi_asset_service import MultiAssetService

logger = logging.getLogger(__name__)


class MultiAssetServiceAdapter(IMultiAssetService):
    """
    Adapter for MultiAssetService that implements the common interface.
    
    This adapter wraps the actual MultiAssetService implementation
    and adapts it to the common interface.
    """
    
    def __init__(self, service_instance=None):
        """
        Initialize the adapter.
        
        Args:
            service_instance: Optional service instance to wrap
        """
        self.service = service_instance or MultiAssetService()
        self.logger = logging.getLogger(__name__)
    
    def get_asset_info(self, symbol: str) -> Dict[str, Any]:
        """Get detailed information about an asset."""
        try:
            # Convert to common interface format
            asset_info = self.service.get_asset_info(symbol)
            if not asset_info:
                return {}
            
            # Map internal AssetClass to common AssetClassEnum
            asset_class_map = {
                AssetClass.FOREX: AssetClassEnum.FOREX,
                AssetClass.CRYPTO: AssetClassEnum.CRYPTO,
                AssetClass.STOCKS: AssetClassEnum.STOCKS,
                AssetClass.COMMODITIES: AssetClassEnum.COMMODITIES,
                AssetClass.INDICES: AssetClassEnum.INDICES,
                AssetClass.BONDS: AssetClassEnum.BONDS,
                AssetClass.ETF: AssetClassEnum.ETF
            }
            
            # Convert asset class
            if "asset_class" in asset_info:
                internal_class = asset_info["asset_class"]
                asset_info["asset_class"] = asset_class_map.get(internal_class, AssetClassEnum.FOREX)
            
            return asset_info
            
        except Exception as e:
            self.logger.error(f"Error in get_asset_info: {str(e)}")
            return {}
    
    def list_assets_by_class(self, asset_class: AssetClassEnum) -> List[str]:
        """List all assets of a specific class."""
        try:
            # Map common AssetClassEnum to internal AssetClass
            asset_class_map = {
                AssetClassEnum.FOREX: AssetClass.FOREX,
                AssetClassEnum.CRYPTO: AssetClass.CRYPTO,
                AssetClassEnum.STOCKS: AssetClass.STOCKS,
                AssetClassEnum.COMMODITIES: AssetClass.COMMODITIES,
                AssetClassEnum.INDICES: AssetClass.INDICES,
                AssetClassEnum.BONDS: AssetClass.BONDS,
                AssetClassEnum.ETF: AssetClass.ETF
            }
            
            internal_class = asset_class_map.get(asset_class, AssetClass.FOREX)
            return self.service.list_assets_by_class(internal_class)
            
        except Exception as e:
            self.logger.error(f"Error in list_assets_by_class: {str(e)}")
            return []
    
    def get_asset_group(self, group_name: str) -> List[str]:
        """Get all symbols in a named group."""
        try:
            return self.service.get_asset_group(group_name)
        except Exception as e:
            self.logger.error(f"Error in get_asset_group: {str(e)}")
            return []
    
    def get_trading_parameters(self, symbol: str) -> Dict[str, Any]:
        """Get trading parameters for a symbol."""
        try:
            return self.service.get_trading_parameters(symbol)
        except Exception as e:
            self.logger.error(f"Error in get_trading_parameters: {str(e)}")
            return {}
    
    def normalize_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Normalize data for the specified symbol."""
        try:
            return self.service.normalize_data(data, symbol)
        except Exception as e:
            self.logger.error(f"Error in normalize_data: {str(e)}")
            
            # Fallback implementation
            normalized = data.copy()
            
            # Ensure datetime format for timestamp
            if "timestamp" in normalized.columns:
                normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
                
            # Calculate percentage changes
            if all(col in normalized.columns for col in ['open', 'close']):
                normalized['pct_change'] = (normalized['close'] - normalized['open']) / normalized['open'] * 100
            
            return normalized
    
    def calculate_volatility(self, data: pd.DataFrame, symbol: str, window: int = 14) -> pd.Series:
        """Calculate symbol-specific volatility."""
        try:
            return self.service.calculate_volatility(data, symbol, window)
        except Exception as e:
            self.logger.error(f"Error in calculate_volatility: {str(e)}")
            
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
        try:
            return self.service.get_price_levels(data, symbol)
        except Exception as e:
            self.logger.error(f"Error in get_price_levels: {str(e)}")
            
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
        try:
            return self.service.get_analysis_parameters(symbol)
        except Exception as e:
            self.logger.error(f"Error in get_analysis_parameters: {str(e)}")
            
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
    
    This adapter wraps the actual AssetRegistry implementation
    and adapts it to the common interface.
    """
    
    def __init__(self, registry_instance=None):
        """
        Initialize the adapter.
        
        Args:
            registry_instance: Optional registry instance to wrap
        """
        from analysis_engine.multi_asset.asset_registry import AssetRegistry
        self.registry = registry_instance or AssetRegistry()
        self.logger = logging.getLogger(__name__)
        
        # Map internal AssetClass to common AssetClassEnum
        self.asset_class_map = {
            AssetClass.FOREX: AssetClassEnum.FOREX,
            AssetClass.CRYPTO: AssetClassEnum.CRYPTO,
            AssetClass.STOCKS: AssetClassEnum.STOCKS,
            AssetClass.COMMODITIES: AssetClassEnum.COMMODITIES,
            AssetClass.INDICES: AssetClassEnum.INDICES,
            AssetClass.BONDS: AssetClassEnum.BONDS,
            AssetClass.ETF: AssetClassEnum.ETF
        }
        
        # Map common AssetClassEnum to internal AssetClass
        self.asset_class_map_reverse = {v: k for k, v in self.asset_class_map.items()}
    
    def get_asset(self, symbol: str) -> Optional[Any]:
        """Get asset definition by symbol."""
        try:
            asset = self.registry.get_asset(symbol)
            if not asset:
                return None
            
            # Convert asset class if present
            if hasattr(asset, 'asset_class'):
                asset_class = getattr(asset, 'asset_class')
                if asset_class in self.asset_class_map:
                    asset_dict = asset.__dict__.copy()
                    asset_dict['asset_class'] = self.asset_class_map[asset_class]
                    return asset_dict
            
            # Return as dictionary
            return asset.__dict__ if hasattr(asset, '__dict__') else asset
            
        except Exception as e:
            self.logger.error(f"Error in get_asset: {str(e)}")
            return None
    
    def list_assets(self, 
                   asset_class: Optional[AssetClassEnum] = None,
                   market_type: Optional[MarketTypeEnum] = None,
                   currency: Optional[str] = None) -> List[Any]:
        """List assets filtered by various criteria."""
        try:
            # Convert asset class if provided
            internal_asset_class = None
            if asset_class:
                internal_asset_class = self.asset_class_map_reverse.get(asset_class)
            
            # Convert market type if needed (assuming similar mapping)
            internal_market_type = market_type
            
            # Get assets from registry
            assets = self.registry.list_assets(
                asset_class=internal_asset_class,
                market_type=internal_market_type,
                currency=currency
            )
            
            # Convert to common format
            result = []
            for asset in assets:
                if hasattr(asset, '__dict__'):
                    asset_dict = asset.__dict__.copy()
                    if 'asset_class' in asset_dict and asset_dict['asset_class'] in self.asset_class_map:
                        asset_dict['asset_class'] = self.asset_class_map[asset_dict['asset_class']]
                    result.append(asset_dict)
                else:
                    result.append(asset)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in list_assets: {str(e)}")
            return []
    
    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Get the most recent correlation between two assets."""
        try:
            return self.registry.get_correlation(symbol1, symbol2)
        except Exception as e:
            self.logger.error(f"Error in get_correlation: {str(e)}")
            return None
    
    def get_correlated_assets(self, symbol: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Get all assets correlated with the given symbol above the threshold."""
        try:
            correlated_assets = self.registry.get_correlated_assets(symbol, threshold)
            
            # Convert asset class in results if needed
            for asset in correlated_assets:
                if 'asset_class' in asset and asset['asset_class'] in self.asset_class_map:
                    asset['asset_class'] = self.asset_class_map[asset['asset_class']]
            
            return correlated_assets
            
        except Exception as e:
            self.logger.error(f"Error in get_correlated_assets: {str(e)}")
            return []
    
    def get_asset_group(self, group_name: str) -> List[Any]:
        """Get all assets in a named group."""
        try:
            assets = self.registry.get_asset_group(group_name)
            
            # Convert to common format
            result = []
            for asset in assets:
                if hasattr(asset, '__dict__'):
                    asset_dict = asset.__dict__.copy()
                    if 'asset_class' in asset_dict and asset_dict['asset_class'] in self.asset_class_map:
                        asset_dict['asset_class'] = self.asset_class_map[asset_dict['asset_class']]
                    result.append(asset_dict)
                else:
                    result.append(asset)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error in get_asset_group: {str(e)}")
            return []
    
    def get_pip_value(self, symbol: str) -> Optional[float]:
        """Get the pip value for a symbol."""
        try:
            return self.registry.get_pip_value(symbol)
        except Exception as e:
            self.logger.error(f"Error in get_pip_value: {str(e)}")
            return None
    
    def get_trading_parameters(self, symbol: str) -> Dict[str, Any]:
        """Get trading parameters for a symbol."""
        try:
            return self.registry.get_trading_parameters(symbol)
        except Exception as e:
            self.logger.error(f"Error in get_trading_parameters: {str(e)}")
            return {}
