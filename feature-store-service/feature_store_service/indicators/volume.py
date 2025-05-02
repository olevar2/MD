"""
Volume Indicators Module.

This module provides implementations of various volume-based indicators.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from feature_store_service.indicators.base_indicator import BaseIndicator


class OnBalanceVolume(BaseIndicator):
    """
    On-Balance Volume (OBV) indicator.
    
    This volume indicator relates price changes to volume, accumulating
    volume on up days and subtracting it on down days.
    """
    
    category = "volume"
    
    def __init__(self, column: str = "close", **kwargs):
        """
        Initialize On-Balance Volume indicator.
        
        Args:
            column: Price data column to use (default: 'close')
            **kwargs: Additional parameters
        """
        self.column = column
        self.name = "obv"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate OBV for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with OBV values
        """
        required_cols = [self.column, 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate price changes
        price_change = result[self.column].diff()
        
        # Create OBV series
        obv = pd.Series(index=result.index, data=0.0)
        
        # Initial value
        obv.iloc[0] = 0
        
        # Calculate OBV
        for i in range(1, len(result)):
            if price_change.iloc[i] > 0:
                obv.iloc[i] = obv.iloc[i-1] + result['volume'].iloc[i]
            elif price_change.iloc[i] < 0:
                obv.iloc[i] = obv.iloc[i-1] - result['volume'].iloc[i]
            else:
                obv.iloc[i] = obv.iloc[i-1]
                
        # Add OBV to result
        result[self.name] = obv
        
        return result


class VolumeWeightedAveragePrice(BaseIndicator):
    """
    Volume-Weighted Average Price (VWAP) indicator.
    
    This indicator calculates the ratio of the value traded to total volume,
    giving the average price a security traded at over a specific period.
    """
    
    category = "volume"
    
    def __init__(self, reset_period: str = "daily", **kwargs):
        """
        Initialize VWAP indicator.
        
        Args:
            reset_period: When to reset the VWAP calculation ('daily', 'weekly', None)
            **kwargs: Additional parameters
        """
        self.reset_period = reset_period
        self.name = f"vwap_{reset_period}"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with VWAP values
        """
        required_cols = ['high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate typical price
        result['typical_price'] = (result['high'] + result['low'] + result['close']) / 3
        
        # Calculate price * volume
        result['pv'] = result['typical_price'] * result['volume']
        
        # Handle reset periods
        if self.reset_period == 'daily' and 'date' in result.columns:
            # Group by date and calculate cumulative sum within each day
            result['cumulative_pv'] = result.groupby(result['date'].dt.date)['pv'].cumsum()
            result['cumulative_volume'] = result.groupby(result['date'].dt.date)['volume'].cumsum()
        elif self.reset_period == 'weekly' and 'date' in result.columns:
            # Group by week and calculate cumulative sum within each week
            result['week'] = result['date'].dt.isocalendar().week
            result['cumulative_pv'] = result.groupby([result['date'].dt.year, result['week']])['pv'].cumsum()
            result['cumulative_volume'] = result.groupby([result['date'].dt.year, result['week']])['volume'].cumsum()
            result.drop(columns=['week'], inplace=True)
        else:
            # No reset, calculate running VWAP for entire period
            result['cumulative_pv'] = result['pv'].cumsum()
            result['cumulative_volume'] = result['volume'].cumsum()
        
        # Calculate VWAP
        result[self.name] = result['cumulative_pv'] / result['cumulative_volume']
        
        # Clean up temporary columns
        result.drop(columns=['typical_price', 'pv', 'cumulative_pv', 'cumulative_volume'], inplace=True)
        
        return result


class AccumulationDistributionLine(BaseIndicator):
    """
    Accumulation/Distribution Line indicator.
    
    This volume-based indicator attempts to measure underlying supply and
    demand by determining whether investors are accumulating or distributing.
    """
    
    category = "volume"
    
    def __init__(self, **kwargs):
        """
        Initialize Accumulation/Distribution Line indicator.
        
        Args:
            **kwargs: Additional parameters
        """
        self.name = "adl"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate A/D Line for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with A/D Line values
        """
        required_cols = ['high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate Money Flow Multiplier
        high_low = result['high'] - result['low']
        close_low = result['close'] - result['low']
        high_close = result['high'] - result['close']
        
        # Avoid division by zero
        high_low = np.where(high_low == 0, 0.000001, high_low)
        
        money_flow_multiplier = ((close_low - high_close) / high_low)
        
        # Calculate Money Flow Volume
        money_flow_volume = money_flow_multiplier * result['volume']
        
        # Calculate A/D Line (cumulative sum of Money Flow Volume)
        result[self.name] = money_flow_volume.cumsum()
        
        return result


class ChaikinMoneyFlow(BaseIndicator):
    """
    Chaikin Money Flow (CMF) indicator.
    
    This indicator measures the amount of Money Flow Volume over a specific period,
    indicating buying or selling pressure.
    """
    
    category = "volume"
    
    def __init__(self, window: int = 20, **kwargs):
        """
        Initialize Chaikin Money Flow indicator.
        
        Args:
            window: Period for money flow calculation
            **kwargs: Additional parameters
        """
        self.window = window
        self.name = f"cmf_{window}"
        
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate CMF for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with CMF values
        """
        required_cols = ['high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in data.columns:
                raise ValueError(f"Data must contain '{col}' column")
                
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate Money Flow Multiplier
        high_low = result['high'] - result['low']
        close_low = result['close'] - result['low']
        high_close = result['high'] - result['close']
        
        # Avoid division by zero
        high_low = np.where(high_low == 0, 0.000001, high_low)
        
        money_flow_multiplier = ((close_low - high_close) / high_low)
        
        # Calculate Money Flow Volume
        money_flow_volume = money_flow_multiplier * result['volume']
        
        # Calculate Sum of Money Flow Volume for the period
        sum_mfv = money_flow_volume.rolling(window=self.window).sum()
        
        # Calculate Sum of Volume for the period
        sum_volume = result['volume'].rolling(window=self.window).sum()
        
        # Calculate CMF
        result[self.name] = sum_mfv / sum_volume
        
        return result
"""
