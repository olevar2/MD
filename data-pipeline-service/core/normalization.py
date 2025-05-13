"""
Data Normalization Operations

This module provides operations for normalizing market data.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from ..base_transformer import BaseMarketDataTransformer

logger = logging.getLogger(__name__)


class DataNormalizer(BaseMarketDataTransformer):
    """
    Transformer for normalizing market data.
    
    This transformer applies generic normalization operations such as
    timestamp conversion, percentage change calculation, and other
    common normalization tasks.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the data normalizer.
        
        Args:
            parameters: Configuration parameters for the normalizer
        """
        super().__init__("data_normalizer", parameters)
    
    def transform(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Normalize market data.
        
        Args:
            data: Market data DataFrame
            **kwargs: Additional arguments for normalization
            
        Returns:
            Normalized market data
        """
        # Create a copy to avoid modifying the original
        normalized = data.copy()
        
        # Ensure datetime format for timestamp
        if "timestamp" in normalized.columns:
            normalized["timestamp"] = pd.to_datetime(normalized["timestamp"])
        
        # Calculate percentage changes
        if all(col in normalized.columns for col in ['open', 'close']):
            normalized['pct_change'] = (normalized['close'] - normalized['open']) / normalized['open'] * 100
        
        # Calculate true range
        if all(col in normalized.columns for col in ['high', 'low', 'close']):
            tr1 = normalized['high'] - normalized['low']
            tr2 = abs(normalized['high'] - normalized['close'].shift(1))
            tr3 = abs(normalized['low'] - normalized['close'].shift(1))
            normalized['true_range'] = pd.DataFrame({'tr1': tr1, 'tr2': tr2, 'tr3': tr3}).max(axis=1)
        
        # Calculate log returns for statistical analysis
        if 'close' in normalized.columns:
            normalized['log_return'] = np.log(normalized['close'] / normalized['close'].shift(1))
        
        # Add date parts for time-based analysis
        if "timestamp" in normalized.columns:
            normalized['date'] = normalized['timestamp'].dt.date
            normalized['hour'] = normalized['timestamp'].dt.hour
            normalized['day_of_week'] = normalized['timestamp'].dt.dayofweek
        
        return normalized
    
    def get_required_columns(self) -> List[str]:
        """
        Get the list of required columns for this transformer.
        
        Returns:
            List of required column names
        """
        # Minimum required columns for basic normalization
        return ["timestamp"]
    
    def handle_missing_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data in the input DataFrame.
        
        Args:
            data: Market data DataFrame
            
        Returns:
            DataFrame with missing data handled
        """
        # Create a copy to avoid modifying the original
        cleaned = data.copy()
        
        # Handle missing timestamps
        if "timestamp" in cleaned.columns:
            # Drop rows with missing timestamps
            cleaned = cleaned.dropna(subset=["timestamp"])
        
        # Handle missing OHLC values
        ohlc_columns = ['open', 'high', 'low', 'close']
        present_ohlc = [col for col in ohlc_columns if col in cleaned.columns]
        
        if present_ohlc:
            # Forward fill missing values
            cleaned[present_ohlc] = cleaned[present_ohlc].ffill()
        
        # Handle missing volume
        if 'volume' in cleaned.columns:
            # Fill missing volume with 0
            cleaned['volume'] = cleaned['volume'].fillna(0)
        
        return cleaned