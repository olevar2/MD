"""
Multi-Asset Interfaces Module

This module provides interfaces for multi-asset functionality used across services,
helping to break circular dependencies between services.
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, List, Any, Optional, Union
import pandas as pd


class AssetClassEnum(str, Enum):
    """Asset class enumeration for different financial instruments."""
    FOREX = "forex"
    CRYPTO = "crypto"
    STOCKS = "stocks"
    COMMODITIES = "commodities"
    INDICES = "indices"
    BONDS = "bonds"
    ETF = "etf"


class MarketTypeEnum(str, Enum):
    """Market type enumeration for different trading venues."""
    SPOT = "spot"
    FUTURES = "futures"
    OPTIONS = "options"
    CFD = "cfd"
    SWAP = "swap"


class IAssetInfo:
    """Interface for asset information."""
    
    @property
    def symbol(self) -> str:
        """Get the asset symbol."""
        pass
    
    @property
    def display_name(self) -> str:
        """Get the display name of the asset."""
        pass
    
    @property
    def asset_class(self) -> AssetClassEnum:
        """Get the asset class."""
        pass
    
    @property
    def market_type(self) -> MarketTypeEnum:
        """Get the market type."""
        pass
    
    @property
    def base_currency(self) -> Optional[str]:
        """Get the base currency (for forex/crypto pairs)."""
        pass
    
    @property
    def quote_currency(self) -> Optional[str]:
        """Get the quote currency (for forex/crypto pairs)."""
        pass
    
    @property
    def trading_parameters(self) -> Dict[str, Any]:
        """Get trading parameters for the asset."""
        pass
    
    @property
    def available_timeframes(self) -> List[str]:
        """Get available timeframes for the asset."""
        pass
    
    @property
    def metadata(self) -> Dict[str, Any]:
        """Get additional metadata for the asset."""
        pass


class IMultiAssetService(ABC):
    """Interface for multi-asset service functionality."""
    
    @abstractmethod
    def get_asset_info(self, symbol: str) -> Dict[str, Any]:
        """
        Get detailed information about an asset.
        
        Args:
            symbol: The asset symbol
            
        Returns:
            Dictionary with asset information
        """
        pass
    
    @abstractmethod
    def list_assets_by_class(self, asset_class: AssetClassEnum) -> List[str]:
        """
        List all assets of a specific class.
        
        Args:
            asset_class: The asset class to filter by
            
        Returns:
            List of asset symbols
        """
        pass
    
    @abstractmethod
    def get_asset_group(self, group_name: str) -> List[str]:
        """
        Get all symbols in a named group.
        
        Args:
            group_name: The name of the asset group
            
        Returns:
            List of asset symbols in the group
        """
        pass
    
    @abstractmethod
    def get_trading_parameters(self, symbol: str) -> Dict[str, Any]:
        """
        Get trading parameters for a symbol.
        
        Args:
            symbol: The asset symbol
            
        Returns:
            Dictionary with trading parameters
        """
        pass
    
    @abstractmethod
    def normalize_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Normalize data for the specified symbol.
        
        Args:
            data: The data to normalize
            symbol: The asset symbol
            
        Returns:
            Normalized data
        """
        pass
    
    @abstractmethod
    def calculate_volatility(self, data: pd.DataFrame, symbol: str, window: int = 14) -> pd.Series:
        """
        Calculate symbol-specific volatility.
        
        Args:
            data: The data to calculate volatility from
            symbol: The asset symbol
            window: The window size for volatility calculation
            
        Returns:
            Volatility series
        """
        pass
    
    @abstractmethod
    def get_price_levels(self, data: pd.DataFrame, symbol: str) -> Dict[str, float]:
        """
        Get significant price levels for the symbol.
        
        Args:
            data: The data to calculate price levels from
            symbol: The asset symbol
            
        Returns:
            Dictionary with price levels
        """
        pass
    
    @abstractmethod
    def get_analysis_parameters(self, symbol: str) -> Dict[str, Any]:
        """
        Get asset-specific analysis parameters.
        
        Args:
            symbol: The asset symbol
            
        Returns:
            Dictionary with analysis parameters
        """
        pass


class IAssetRegistry(ABC):
    """Interface for asset registry functionality."""
    
    @abstractmethod
    def get_asset(self, symbol: str) -> Optional[Any]:
        """
        Get asset definition by symbol.
        
        Args:
            symbol: The asset symbol
            
        Returns:
            Asset definition or None if not found
        """
        pass
    
    @abstractmethod
    def list_assets(self, 
                   asset_class: Optional[AssetClassEnum] = None,
                   market_type: Optional[MarketTypeEnum] = None,
                   currency: Optional[str] = None) -> List[Any]:
        """
        List assets filtered by various criteria.
        
        Args:
            asset_class: Filter by asset class
            market_type: Filter by market type
            currency: Filter by currency
            
        Returns:
            List of asset definitions
        """
        pass
    
    @abstractmethod
    def get_correlation(self, symbol1: str, symbol2: str) -> Optional[float]:
        """
        Get the most recent correlation between two assets.
        
        Args:
            symbol1: First asset symbol
            symbol2: Second asset symbol
            
        Returns:
            Correlation value or None if not available
        """
        pass
    
    @abstractmethod
    def get_correlated_assets(self, symbol: str, threshold: float = 0.7) -> List[Dict[str, Any]]:
        """
        Get all assets correlated with the given symbol above the threshold.
        
        Args:
            symbol: The asset symbol
            threshold: Correlation threshold
            
        Returns:
            List of correlated assets
        """
        pass
    
    @abstractmethod
    def get_asset_group(self, group_name: str) -> List[Any]:
        """
        Get all assets in a named group.
        
        Args:
            group_name: The name of the asset group
            
        Returns:
            List of asset definitions
        """
        pass
    
    @abstractmethod
    def get_pip_value(self, symbol: str) -> Optional[float]:
        """
        Get the pip value for a symbol.
        
        Args:
            symbol: The asset symbol
            
        Returns:
            Pip value or None if not available
        """
        pass
    
    @abstractmethod
    def get_trading_parameters(self, symbol: str) -> Dict[str, Any]:
        """
        Get trading parameters for a symbol.
        
        Args:
            symbol: The asset symbol
            
        Returns:
            Dictionary with trading parameters
        """
        pass
