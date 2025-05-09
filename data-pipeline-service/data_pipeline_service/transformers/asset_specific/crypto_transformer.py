"""
Cryptocurrency Market Data Transformer

This module provides specialized transformation for cryptocurrency market data.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from ..base_transformer import BaseMarketDataTransformer
from ..operations.normalization import DataNormalizer

logger = logging.getLogger(__name__)


class CryptoTransformer(BaseMarketDataTransformer):
    """
    Specialized transformer for cryptocurrency market data.
    
    This transformer applies crypto-specific transformations such as volume analysis,
    volatility calculations, and other crypto-specific features.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the crypto transformer.
        
        Args:
            parameters: Configuration parameters for the transformer
        """
        super().__init__("crypto_transformer", parameters)
        self.normalizer = DataNormalizer()
    
    def transform(self, data: pd.DataFrame, asset_info: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Transform cryptocurrency market data.
        
        Args:
            data: Market data DataFrame
            asset_info: Asset information
            
        Returns:
            Transformed cryptocurrency market data
        """
        # Start with generic normalization
        normalized = self.normalizer.transform(data)
        
        # Calculate daily percentage changes
        if all(col in normalized.columns for col in ['open', 'close']):
            normalized['pct_change'] = (normalized['close'] - normalized['open']) / normalized['open'] * 100
        
        # Calculate volatility (standard deviation of returns)
        if 'log_return' in normalized.columns:
            normalized['rolling_volatility'] = normalized['log_return'].rolling(window=24).std() * 100
        
        # Add volume indicators (volume often more important in crypto)
        if 'volume' in normalized.columns and 'close' in normalized.columns:
            normalized['volume_quote'] = normalized['volume'] * normalized['close']
            normalized['volume_ma'] = normalized['volume'].rolling(window=24).mean()
            normalized['relative_volume'] = normalized['volume'] / normalized['volume_ma']
        
        # Add market activity indicators
        if 'timestamp' in normalized.columns:
            # Add hour of day for crypto (markets are 24/7)
            normalized['hour_of_day'] = normalized['timestamp'].dt.hour
            
            # Flag weekends (often different behavior)
            normalized['is_weekend'] = normalized['timestamp'].dt.dayofweek >= 5
        
        return normalized
    
    def transform_statistics(self, stats_dict: Dict[str, Any], 
                           asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform cryptocurrency statistics.
        
        Args:
            stats_dict: Statistics dictionary
            asset_info: Asset information
            
        Returns:
            Transformed statistics
        """
        # Create a copy to avoid modifying the original
        transformed = stats_dict.copy()
        
        # Crypto often uses percentage values
        if "daily_range_avg" in transformed:
            transformed["daily_range_pct"] = transformed["daily_range_avg"] / transformed.get("avg_price", 1) * 100
        
        # Add volume-based metrics
        if "avg_volume" in transformed and "avg_price" in transformed:
            transformed["avg_volume_quote"] = transformed["avg_volume"] * transformed["avg_price"]
        
        return transformed