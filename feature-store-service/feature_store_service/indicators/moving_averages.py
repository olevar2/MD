"""
Moving Averages Module.

This module provides implementations of various moving average indicators.
"""

from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
from feature_store_service.indicators.base_indicator import BaseIndicator
from feature_store_service.utils.profiling import log_and_time


class SimpleMovingAverage(BaseIndicator):
    """
    Simple Moving Average (SMA) indicator.
    
    This indicator calculates the arithmetic mean of a given set of prices
    over a specified period.
    """
    
    category = "moving_average"
    
    def __init__(self, window: int = 14, column: str = "close", **kwargs):
        """
        Initialize Simple Moving Average indicator.
        
        Args:
            window: Lookback period for the moving average
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.window = window
        self.column = column
        self.name = f"sma_{window}"
        
    @log_and_time
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Calculate SMA for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with SMA values
        \"\"\"
        # TODO: Add unit tests for edge cases (e.g., insufficient data, NaNs).
        # TODO: Consider caching results if this calculation is frequently repeated with the same input.
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")
            
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate SMA
        result[self.name] = result[self.column].rolling(window=self.window).mean()
        
        return result


class ExponentialMovingAverage(BaseIndicator):
    """
    Exponential Moving Average (EMA) indicator.
    
    This indicator gives more weight to recent prices while still considering
    older prices with an exponentially decreasing weight.
    """
    
    category = "moving_average"
    
    def __init__(self, window: int = 14, column: str = "close", **kwargs):
        """
        Initialize Exponential Moving Average indicator.
        
        Args:
            window: Lookback period for the moving average
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.window = window
        self.column = column
        self.name = f"ema_{window}"
        
    @log_and_time
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Calculate EMA for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with EMA values
        \"\"\"
        # TODO: Add unit tests for edge cases and compare against known values.
        # TODO: Profile performance on large datasets.
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")
            
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate EMA - using pandas built-in EMA function
        result[self.name] = result[self.column].ewm(span=self.window, adjust=False).mean()
        
        return result


class WeightedMovingAverage(BaseIndicator):
    """
    Weighted Moving Average (WMA) indicator.
    
    This indicator assigns a greater weight to more recent data points
    and less weight to data points in the distant past.
    """
    
    category = "moving_average"
    
    def __init__(self, window: int = 14, column: str = "close", **kwargs):
        """
        Initialize Weighted Moving Average indicator.
        
        Args:
            window: Lookback period for the moving average
            column: Data column to use for calculations (default: 'close')
            **kwargs: Additional parameters
        """
        self.window = window
        self.column = column
        self.name = f"wma_{window}"
        
    @log_and_time
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        \"\"\"
        Calculate WMA for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with WMA values
        \"\"\"
        # TODO: Add unit tests, especially for the custom rolling apply logic.
        # TODO: The rolling apply with a lambda can be slow; investigate potential vectorized alternatives if performance is an issue.
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")
            
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Create weights - higher weight for recent prices
        weights = np.arange(1, self.window + 1)
        sum_weights = np.sum(weights)
        
        # Calculate WMA
        result[self.name] = result[self.column].rolling(window=self.window).apply(
            lambda x: np.sum(weights * x) / sum_weights, raw=True
        )
        
        return result
