"""
Index Market Data Transformer

This module provides specialized transformation for index market data.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd

from ..base_transformer import BaseMarketDataTransformer
from ..operations.normalization import DataNormalizer

logger = logging.getLogger(__name__)


class IndexTransformer(BaseMarketDataTransformer):
    """
    Specialized transformer for index market data.
    
    This transformer applies index-specific transformations such as
    sector analysis, breadth indicators, and other index-specific features.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the index transformer.
        
        Args:
            parameters: Configuration parameters for the transformer
        """
        super().__init__("index_transformer", parameters)
        self.normalizer = DataNormalizer()
    
    def transform(self, data: pd.DataFrame, asset_info: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Transform index market data.
        
        Args:
            data: Market data DataFrame
            asset_info: Asset information
            
        Returns:
            Transformed index market data
        """
        # Start with generic normalization
        normalized = self.normalizer.transform(data)
        
        # Add index-specific features
        if 'timestamp' in normalized.columns:
            # Add market session (pre-market, regular, after-hours)
            normalized['market_session'] = self._determine_market_session(normalized['timestamp'])
            
            # Flag trading days
            normalized['is_trading_day'] = ~normalized['timestamp'].dt.dayofweek.isin([5, 6])  # Not weekend
            
            # Flag month/quarter end effects
            normalized['is_month_end'] = normalized['timestamp'].dt.is_month_end
            normalized['is_quarter_end'] = normalized['timestamp'].dt.is_quarter_end
        
        # Add breadth indicators if available in asset_info
        if asset_info and 'breadth_data' in asset_info:
            # This would require additional data not in the main DataFrame
            # In a real implementation, this would fetch the breadth data
            pass
        
        return normalized
    
    def transform_statistics(self, stats_dict: Dict[str, Any], 
                           asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform index statistics.
        
        Args:
            stats_dict: Statistics dictionary
            asset_info: Asset information
            
        Returns:
            Transformed statistics
        """
        # Create a copy to avoid modifying the original
        transformed = stats_dict.copy()
        
        # Add index-specific metrics
        index_type = asset_info.get('index_type') if asset_info else None
        
        if index_type == 'equity':
            # Equity index specific metrics
            if 'correlation_with_vix' in transformed:
                transformed['volatility_sensitivity'] = -transformed['correlation_with_vix'] * 100
        
        elif index_type == 'bond':
            # Bond index specific metrics
            if 'correlation_with_rates' in transformed:
                transformed['rate_sensitivity'] = -transformed['correlation_with_rates'] * 100
        
        elif index_type == 'commodity':
            # Commodity index specific metrics
            if 'correlation_with_dollar' in transformed:
                transformed['dollar_sensitivity'] = -transformed['correlation_with_dollar'] * 100
        
        return transformed
    
    def _determine_market_session(self, timestamps: pd.Series) -> pd.Series:
        """
        Determine market session for each timestamp.
        
        Args:
            timestamps: Series of timestamps
            
        Returns:
            Series with session labels
        """
        # Convert to UTC if not already
        # Note: This assumes timestamps are already in UTC
        hours = timestamps.dt.hour
        minutes = timestamps.dt.minute
        days = timestamps.dt.dayofweek
        
        # Initialize with 'closed'
        sessions = pd.Series(['closed'] * len(timestamps), index=timestamps.index)
        
        # Define sessions (US market hours in UTC)
        # Pre-market: 4:00-9:30 ET (8:00-13:30 UTC)
        # Regular: 9:30-16:00 ET (13:30-20:00 UTC)
        # After-hours: 16:00-20:00 ET (20:00-24:00 UTC)
        
        # Only consider weekdays
        weekday_mask = days < 5
        
        # Pre-market
        sessions[(weekday_mask) & (
            ((hours == 8) & (minutes >= 0)) |
            ((hours > 8) & (hours < 13)) |
            ((hours == 13) & (minutes < 30))
        )] = 'pre_market'
        
        # Regular market
        sessions[(weekday_mask) & (
            ((hours == 13) & (minutes >= 30)) |
            ((hours > 13) & (hours < 20))
        )] = 'regular'
        
        # After-hours
        sessions[(weekday_mask) & (
            ((hours >= 20) & (hours < 24))
        )] = 'after_hours'
        
        return sessions