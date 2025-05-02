"""
Multi-Asset Market Data Normalizer

This module provides normalization logic for market data across different asset classes.
"""

import logging
from typing import Dict, List, Any, Optional, Union
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from analysis_engine.multi_asset.asset_registry import AssetClass, AssetRegistry
from analysis_engine.services.multi_asset_service import MultiAssetService

logger = logging.getLogger(__name__)


class MarketDataNormalizer:
    """
    Normalizes market data for consistent analysis across asset classes.
    
    This class ensures that raw market data from various sources is normalized
    to a consistent format for analysis, accounting for asset-specific characteristics.
    """
    
    def __init__(self, multi_asset_service: Optional[MultiAssetService] = None):
        """Initialize the market data normalizer"""
        self.multi_asset_service = multi_asset_service or MultiAssetService()
        self.logger = logging.getLogger(__name__)
        
    def normalize_data(self, data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Normalize market data for consistent analysis
        
        Args:
            data: Raw market data DataFrame
            symbol: Asset symbol
            
        Returns:
            Normalized market data
        """
        # Get asset information for asset-specific normalization
        asset_info = self.multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            self.logger.warning(f"Asset info not found for {symbol}, using generic normalization")
            return self._apply_generic_normalization(data)
            
        # Get asset class
        asset_class = asset_info.get("asset_class")
        
        # Apply asset-specific normalization
        if asset_class == AssetClass.FOREX:
            return self._normalize_forex_data(data, asset_info)
        elif asset_class == AssetClass.CRYPTO:
            return self._normalize_crypto_data(data, asset_info)
        elif asset_class == AssetClass.STOCKS:
            return self._normalize_stock_data(data, asset_info)
        elif asset_class == AssetClass.COMMODITIES:
            return self._normalize_commodity_data(data, asset_info)
        elif asset_class == AssetClass.INDICES:
            return self._normalize_index_data(data, asset_info)
        else:
            return self._apply_generic_normalization(data)
            
    def normalize_asset_statistics(self, 
                                 stats_dict: Dict[str, Any], 
                                 symbol: str) -> Dict[str, Any]:
        """
        Normalize asset statistics for consistent reporting
        
        Args:
            stats_dict: Statistics dictionary
            symbol: Asset symbol
            
        Returns:
            Normalized statistics
        """
        # Get asset information
        asset_info = self.multi_asset_service.get_asset_info(symbol)
        if not asset_info:
            return stats_dict
            
        # Get asset class and parameters
        asset_class = asset_info.get("asset_class")
        
        # Create a copy to avoid modifying the original
        normalized = stats_dict.copy()
        
        # Apply asset-specific adjustments
        if asset_class == AssetClass.FOREX:
            # Convert percentage moves to pips
            trading_params = asset_info.get("trading_parameters", {})
            pip_value = trading_params.get("pip_value", 0.0001)
            
            if "daily_range_avg" in normalized:
                normalized["daily_range_pips"] = normalized["daily_range_avg"] / pip_value
                
            if "volatility" in normalized:
                normalized["volatility_pips"] = normalized["volatility"] / pip_value
                
        elif asset_class == AssetClass.CRYPTO:
            # Crypto often uses percentage values
            if "daily_range_avg" in normalized:
                normalized["daily_range_pct"] = normalized["daily_range_avg"] / normalized.get("avg_price", 1) * 100
                
        return normalized
    
    def _apply_generic_normalization(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply generic normalization to market data"""
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
    
    def _normalize_forex_data(self, data: pd.DataFrame, asset_info: Dict[str, Any]) -> pd.DataFrame:
        """Normalize forex market data"""
        # Start with generic normalization
        normalized = self._apply_generic_normalization(data)
        
        # Get trading parameters
        trading_params = asset_info.get("trading_parameters", {})
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
    
    def _normalize_crypto_data(self, data: pd.DataFrame, asset_info: Dict[str, Any]) -> pd.DataFrame:
        """Normalize cryptocurrency market data"""
        # Start with generic normalization
        normalized = self._apply_generic_normalization(data)
        
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
            
        return normalized
    
    def _normalize_stock_data(self, data: pd.DataFrame, asset_info: Dict[str, Any]) -> pd.DataFrame:
        """Normalize stock market data"""
        # Start with generic normalization
        normalized = self._apply_generic_normalization(data)
        
        # Calculate daily percentage changes
        if all(col in normalized.columns for col in ['open', 'close']):
            normalized['pct_change'] = (normalized['close'] - normalized['open']) / normalized['open'] * 100
            
        # Add gap information (important for stocks)
        if 'open' in normalized.columns and 'close' in normalized.columns:
            normalized['gap'] = normalized['open'] - normalized['close'].shift(1)
            normalized['gap_pct'] = normalized['gap'] / normalized['close'].shift(1) * 100
            
        # Flag earnings periods if data is available
        if 'timestamp' in normalized.columns and 'earnings_dates' in asset_info.get('metadata', {}):
            earnings_dates = asset_info['metadata']['earnings_dates']
            normalized['near_earnings'] = normalized['timestamp'].dt.date.apply(
                lambda x: self._is_near_earnings(x, earnings_dates)
            )
            
        return normalized
    
    def _normalize_commodity_data(self, data: pd.DataFrame, asset_info: Dict[str, Any]) -> pd.DataFrame:
        """Normalize commodity market data"""
        # Start with generic normalization
        normalized = self._apply_generic_normalization(data)
        
        # Add commodity-specific fields
        # (Implement based on specific commodity requirements)
        
        return normalized
    
    def _normalize_index_data(self, data: pd.DataFrame, asset_info: Dict[str, Any]) -> pd.DataFrame:
        """Normalize index market data"""
        # Start with generic normalization
        normalized = self._apply_generic_normalization(data)
        
        # Add index-specific fields
        # (Implement based on specific index requirements)
        
        return normalized
    
    def _determine_forex_session(self, timestamps: pd.Series) -> pd.Series:
        """
        Determine forex trading session for each timestamp
        
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
    
    def _is_near_earnings(self, date: datetime.date, earnings_dates: List[str]) -> bool:
        """
        Check if a date is near an earnings announcement
        
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
