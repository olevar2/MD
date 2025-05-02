"""
Asset Registry Module

This module provides a registry for different asset classes and markets,
managing their specific properties, analysis settings, and correlations.
"""

import logging
from enum import Enum
from typing import Dict, List, Optional, Any, Set
from datetime import datetime
import json

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class AssetClass(str, Enum):
    FOREX = "forex"
    CRYPTO = "crypto"
    STOCKS = "stocks"
    COMMODITIES = "commodities"
    INDICES = "indices"
    BONDS = "bonds"
    ETF = "etf"


class MarketType(str, Enum):
    SPOT = "spot"
    FUTURES = "futures"
    OPTIONS = "options"
    CFD = "cfd"
    SWAP = "swap"


class AssetConfig(BaseModel):
    """Configuration settings for an asset"""
    min_price_precision: int = Field(..., description="Minimum price precision (decimal places)")
    pip_value: float = Field(..., description="Value of 1 pip in price units")
    pip_location: int = Field(..., description="Decimal position of pip (e.g., -4 for 4 decimal places)")
    min_quantity: float = Field(..., description="Minimum tradable quantity")
    quantity_precision: int = Field(..., description="Quantity precision (decimal places)")
    margin_rate: Optional[float] = Field(None, description="Margin rate if applicable")
    trading_hours: Optional[Dict[str, Any]] = Field(None, description="Trading hours configuration")
    lot_size: Optional[float] = Field(None, description="Standard lot size if applicable")
    point_value: Optional[float] = Field(None, description="Value of 1 point in account currency")
    tick_size: Optional[float] = Field(None, description="Minimum price movement")
    trading_fee: Optional[float] = Field(None, description="Trading fee percentage")
    swap_rates: Optional[Dict[str, float]] = Field(None, description="Overnight swap rates for long/short positions")


class AssetCorrelation(BaseModel):
    """Correlation between two assets"""
    symbol1: str
    symbol2: str
    correlation: float
    as_of_date: datetime
    lookback_days: int


class AssetDefinition(BaseModel):
    """Complete definition of an asset"""
    symbol: str = Field(..., description="Unique identifier for the asset")
    display_name: str = Field(..., description="Human-readable name")
    asset_class: AssetClass
    market_type: MarketType
    base_currency: Optional[str] = Field(None, description="Base currency for forex/crypto pairs")
    quote_currency: Optional[str] = Field(None, description="Quote currency for forex/crypto pairs")
    config: AssetConfig
    enabled: bool = True
    available_timeframes: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)


class AssetRegistry:
    """
    Registry for managing different asset classes and markets.
    
    This class provides methods to:
    1. Register and retrieve asset definitions
    2. Manage asset configurations
    3. Track correlations between assets
    4. Filter assets by various criteria
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the asset registry"""
        self.assets: Dict[str, AssetDefinition] = {}
        self.correlations: Dict[str, List[AssetCorrelation]] = {}
        self._asset_groups: Dict[str, Set[str]] = {}
        self.logger = logging.getLogger(__name__)
        
        # Load configuration if provided
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str) -> bool:
        """Load asset registry configuration from JSON file"""
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Load assets
            for asset_data in config_data.get('assets', []):
                asset = AssetDefinition(**asset_data)
                self.register_asset(asset)
            
            # Load correlations
            for corr_data in config_data.get('correlations', []):
                correlation = AssetCorrelation(**corr_data)
                self.add_correlation(correlation)
            
            # Load asset groups
            for group_name, symbols in config_data.get('groups', {}).items():
                self._asset_groups[group_name] = set(symbols)
            
            self.logger.info(f"Loaded {len(self.assets)} assets and {len(self.correlations)} correlations")
            return True
        
        except Exception as e:
            self.logger.error(f"Failed to load asset registry config: {str(e)}")
            return False
    
    def register_asset(self, asset: AssetDefinition) -> None:
        """Register a new asset in the registry"""
        self.assets[asset.symbol] = asset
        self.logger.info(f"Registered asset: {asset.symbol} ({asset.display_name})")
    
    def get_asset(self, symbol: str) -> Optional[AssetDefinition]:
        """Get asset definition by symbol"""
        return self.assets.get(symbol)
    
    def list_assets(self, 
                    asset_class: Optional[AssetClass] = None,
                    market_type: Optional[MarketType] = None,
                    currency: Optional[str] = None) -> List[AssetDefinition]:
        """List assets filtered by various criteria"""
        result = list(self.assets.values())
        
        if asset_class:
            result = [a for a in result if a.asset_class == asset_class]
        
        if market_type:
            result = [a for a in result if a.market_type == market_type]
        
        if currency:
            result = [a for a in result if (
                a.base_currency == currency or a.quote_currency == currency
            )]
        
        return result
    
    def add_correlation(self, correlation: AssetCorrelation) -> None:
        """Add a correlation between two assets"""
        # Create a unique key for the correlation pair (order-independent)
        key = "_".join(sorted([correlation.symbol1, correlation.symbol2]))
        
        if key not in self.correlations:
            self.correlations[key] = []
        
        self.correlations[key].append(correlation)
    
    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """Get the most recent correlation between two assets"""
        # Create a unique key for the correlation pair (order-independent)
        key = "_".join(sorted([symbol1, symbol2]))
        
        if key not in self.correlations or not self.correlations[key]:
            return None
        
        # Return the most recent correlation value
        most_recent = max(self.correlations[key], key=lambda c: c.as_of_date)
        return most_recent.correlation
    
    def get_correlated_assets(self, symbol: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """Get all assets correlated with the given symbol above the threshold"""
        result = []
        
        for key, correlations in self.correlations.items():
            if not correlations:
                continue
                
            symbols = key.split('_')
            if symbol not in symbols:
                continue
            
            # Find the most recent correlation
            most_recent = max(correlations, key=lambda c: c.as_of_date)
            if abs(most_recent.correlation) >= threshold:
                other_symbol = symbols[0] if symbols[1] == symbol else symbols[1]
                asset = self.get_asset(other_symbol)
                
                if asset:
                    result.append({
                        "symbol": other_symbol,
                        "display_name": asset.display_name,
                        "correlation": most_recent.correlation,
                        "as_of_date": most_recent.as_of_date,
                        "asset_class": asset.asset_class
                    })
        
        # Sort by absolute correlation value (descending)
        result.sort(key=lambda x: abs(x["correlation"]), reverse=True)
        return result
    
    def create_asset_group(self, group_name: str, symbols: List[str]) -> bool:
        """Create a named group of assets"""
        if not all(symbol in self.assets for symbol in symbols):
            self.logger.error(f"Cannot create group {group_name}: not all symbols are registered")
            return False
        
        self._asset_groups[group_name] = set(symbols)
        return True
    
    def get_asset_group(self, group_name: str) -> List[AssetDefinition]:
        """Get all assets in a named group"""
        if group_name not in self._asset_groups:
            return []
        
        return [self.assets[symbol] for symbol in self._asset_groups[group_name] 
                if symbol in self.assets]
    
    def get_pip_value(self, symbol: str) -> Optional[float]:
        """Get the pip value for a symbol"""
        asset = self.get_asset(symbol)
        if asset:
            return asset.config.pip_value
        return None
    
    def get_trading_parameters(self, symbol: str) -> Dict[str, Any]:
        """Get trading parameters for a symbol"""
        asset = self.get_asset(symbol)
        if not asset:
            return {}
        
        return {
            "min_quantity": asset.config.min_quantity,
            "quantity_precision": asset.config.quantity_precision,
            "price_precision": asset.config.min_price_precision,
            "pip_value": asset.config.pip_value,
            "lot_size": asset.config.lot_size,
            "margin_rate": asset.config.margin_rate,
            "trading_fee": asset.config.trading_fee
        }
