"""
Forex Market Data Transformer

This module provides specialized transformation for forex market data.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

from ..base_transformer import BaseMarketDataTransformer
from ..operations.normalization import DataNormalizer

logger = logging.getLogger(__name__)


class ForexTransformer(BaseMarketDataTransformer):
    """
    Specialized transformer for forex market data.
    
    This transformer applies forex-specific transformations such as pip calculations,
    session identification, and other forex-specific features.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the forex transformer.
        
        Args:
            parameters: Configuration parameters for the transformer
        """
        super().__init__("forex_transformer", parameters)
        self.normalizer = DataNormalizer()
    
    def transform(self, data: pd.DataFrame, asset_info: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Transform forex market data.
        
        Args:
            data: Market data DataFrame
            asset_info: Asset information
            
        Returns:
            Transformed forex market data
        """
        # Start with generic normalization
        normalized = self.normalizer.transform(data)
        
        # Get trading parameters
        trading_params = asset_info.get("trading_parameters", {}) if asset_info else {}
        pip_value = trading_params.get("pip_value", 0.0001)
        
        # Calculate pip movements
        if all(col in normalized.columns for col in ['open', 'close']):
            normalized['pips_change'] = (normalized['close'] - normalized['open']) / pip_value
        
        if all(col in normalized.columns for col in ['high', 'low']):
            normalized['range_pips'] = (normalized['high'] - normalized['low']) / pip_value
        
        # Add session flags for forex
        if "timestamp" in normalized.columns:
            # Define forex sessions in UTC
            normalized['session'] = self._determine_forex_session(normalized['timestamp'])
        
        return normalized
    
    def transform_statistics(self, stats_dict: Dict[str, Any], 
                           asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform forex statistics.
        
        Args:
            stats_dict: Statistics dictionary
            asset_info: Asset information
            
        Returns:
            Transformed statistics
        """
        # Create a copy to avoid modifying the original
        transformed = stats_dict.copy()
        
        # Convert percentage moves to pips
        trading_params = asset_info.get("trading_parameters", {})
        pip_value = trading_params.get("pip_value", 0.0001)
        
        if "daily_range_avg" in transformed:
            transformed["daily_range_pips"] = transformed["daily_range_avg"] / pip_value
        
        if "volatility" in transformed:
            transformed["volatility_pips"] = transformed["volatility"] / pip_value
        
        return transformed
    
    def _determine_forex_session(self, timestamps: pd.Series) -> pd.Series:
        """
        Determine forex trading session for each timestamp.
        
        Args:
            timestamps: Series of timestamps
            
        Returns:
            Series with session labels
        """
        # Convert to UTC if not already
        # Note: This assumes timestamps are already in UTC
        hours = timestamps.dt.hour
        days = timestamps.dt.dayofweek
        
        # Initialize with 'none'
        sessions = pd.Series(['none'] * len(timestamps), index=timestamps.index)
        
        # Define sessions (overlapping sessions will use later overrides)
        
        # Tokyo session (approx. 23:00-08:00 UTC)
        sessions[(hours >= 23) | (hours < 8)] = 'tokyo'
        
        # London session (approx. 07:00-16:00 UTC)
        sessions[(hours >= 7) & (hours < 16)] = 'london'
        
        # New York session (approx. 12:00-21:00 UTC)
        sessions[(hours >= 12) & (hours < 21)] = 'new_york'
        
        # London/NY overlap (high liquidity)
        sessions[(hours >= 12) & (hours < 16)] = 'london_ny'
        
        # Weekend (low liquidity)
        sessions[(days > 4) | ((days == 4) & (hours >= 21)) | ((days == 0) & (hours < 23))] = 'weekend'
        
        return sessions