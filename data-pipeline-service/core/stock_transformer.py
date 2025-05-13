"""
Stock Market Data Transformer

This module provides specialized transformation for stock market data.
"""

import logging
from typing import Dict, List, Any, Optional
import pandas as pd
from datetime import datetime

from ..base_transformer import BaseMarketDataTransformer
from ..operations.normalization import DataNormalizer

logger = logging.getLogger(__name__)


class StockTransformer(BaseMarketDataTransformer):
    """
    Specialized transformer for stock market data.
    
    This transformer applies stock-specific transformations such as gap analysis,
    earnings event flagging, and other stock-specific features.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize the stock transformer.
        
        Args:
            parameters: Configuration parameters for the transformer
        """
        super().__init__("stock_transformer", parameters)
        self.normalizer = DataNormalizer()
    
    def transform(self, data: pd.DataFrame, asset_info: Dict[str, Any] = None) -> pd.DataFrame:
        """
        Transform stock market data.
        
        Args:
            data: Market data DataFrame
            asset_info: Asset information
            
        Returns:
            Transformed stock market data
        """
        # Start with generic normalization
        normalized = self.normalizer.transform(data)
        
        # Calculate daily percentage changes
        if all(col in normalized.columns for col in ['open', 'close']):
            normalized['pct_change'] = (normalized['close'] - normalized['open']) / normalized['open'] * 100
        
        # Add gap information (important for stocks)
        if 'open' in normalized.columns and 'close' in normalized.columns:
            normalized['gap'] = normalized['open'] - normalized['close'].shift(1)
            normalized['gap_pct'] = normalized['gap'] / normalized['close'].shift(1) * 100
        
        # Flag earnings periods if data is available
        if 'timestamp' in normalized.columns and asset_info and 'earnings_dates' in asset_info.get('metadata', {}):
            earnings_dates = asset_info['metadata']['earnings_dates']
            normalized['near_earnings'] = normalized['timestamp'].dt.date.apply(
                lambda x: self._is_near_earnings(x, earnings_dates)
            )
        
        # Add market session indicators
        if 'timestamp' in normalized.columns:
            # Add market session (pre-market, regular, after-hours)
            normalized['market_session'] = self._determine_market_session(normalized['timestamp'])
            
            # Flag trading days
            normalized['is_trading_day'] = ~normalized['timestamp'].dt.dayofweek.isin([5, 6])  # Not weekend
        
        return normalized
    
    def transform_statistics(self, stats_dict: Dict[str, Any], 
                           asset_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transform stock statistics.
        
        Args:
            stats_dict: Statistics dictionary
            asset_info: Asset information
            
        Returns:
            Transformed statistics
        """
        # Create a copy to avoid modifying the original
        transformed = stats_dict.copy()
        
        # Add stock-specific metrics
        if "avg_volume" in transformed and "avg_price" in transformed:
            transformed["avg_dollar_volume"] = transformed["avg_volume"] * transformed["avg_price"]
        
        # Add earnings-related metrics if available
        if asset_info and "earnings_dates" in asset_info.get("metadata", {}):
            transformed["next_earnings"] = min(
                (date for date in asset_info["metadata"]["earnings_dates"] 
                 if datetime.strptime(date, '%Y-%m-%d').date() > datetime.now().date()),
                default=None
            )
        
        return transformed
    
    def _is_near_earnings(self, date: datetime.date, earnings_dates: List[str]) -> bool:
        """
        Check if a date is near an earnings announcement.
        
        Args:
            date: Date to check
            earnings_dates: List of earnings announcement dates
            
        Returns:
            True if within 3 days of earnings, False otherwise
        """
        if not earnings_dates:
            return False
        
        # Convert string dates to datetime.date objects
        earnings_dates = [
            datetime.strptime(date_str, '%Y-%m-%d').date()
            if isinstance(date_str, str) else date_str
            for date_str in earnings_dates
        ]
        
        # Check if within 3 days of any earnings date
        for earnings_date in earnings_dates:
            delta = abs((date - earnings_date).days)
            if delta <= 3:
                return True
        
        return False
    
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