"""
Volatility Envelopes Module.

This module provides implementations of envelope-based volatility indicators.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any

from feature_store_service.indicators.base_indicator import BaseIndicator


class PriceEnvelopes(BaseIndicator):
    """
    Price Envelopes indicator with customizable percentage options.
    
    This volatility indicator creates upper and lower bands by adding/subtracting a 
    percentage of the price from a moving average.
    """
    
    category = "volatility"
    
    def __init__(
        self,
        window: int = 20,
        percent: float = 2.5,
        ma_method: str = "sma",
        column: str = "close",
        adaptive_percentage: bool = False,
        **kwargs
    ):
        """
        Initialize Price Envelopes indicator.
        
        Args:
            window: Lookback period for the moving average
            percent: Percentage for the envelopes (e.g., 2.5 for 2.5%)
            ma_method: Moving average type ('sma', 'ema', or 'wma')
            column: Data column to use for calculations
            adaptive_percentage: Whether to adapt the percentage based on volatility
            **kwargs: Additional parameters
        """
        self.window = window
        self.percent = percent
        self.ma_method = ma_method.lower()
        self.column = column
        self.adaptive_percentage = adaptive_percentage
        
        # Define output column names
        self.name_ma = f"envelope_{ma_method}_{window}"
        self.name_upper = f"envelope_upper_{window}_{percent}"
        self.name_lower = f"envelope_lower_{window}_{percent}"
        self.name_adaptive = "envelope_adaptive_percent" if adaptive_percentage else None
        
    def _calculate_ma(self, data: pd.Series, window: int, method: str) -> pd.Series:
        """
        Calculate moving average based on the specified method.
        
        Args:
            data: Price series
            window: Lookback period
            method: Moving average type ('sma', 'ema', or 'wma')
            
        Returns:
            Series with moving average values
        """
        if method == 'sma':
            return data.rolling(window=window).mean()
        elif method == 'ema':
            return data.ewm(span=window, adjust=False).mean()
        elif method == 'wma':
            weights = np.arange(1, window + 1)
            return data.rolling(window=window).apply(
                lambda x: np.sum(weights * x) / np.sum(weights), raw=True
            )
        else:
            raise ValueError(f"Invalid ma_method: {method}. Expected 'sma', 'ema', or 'wma'.")
            
    def _calculate_adaptive_percentage(self, data: pd.DataFrame) -> pd.Series:
        """
        Calculate adaptive percentage based on recent volatility.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            Series with adaptive percentage values
        """
        # Calculate price volatility using standard deviation of returns
        volatility = data[self.column].pct_change().rolling(window=self.window).std()
        
        # Scale the base percentage by the ratio of current volatility to average volatility
        mean_volatility = volatility.mean()
        if mean_volatility > 0:
            adaptive_pct = self.percent * (volatility / mean_volatility)
            # Limit the range to avoid extreme values
            return adaptive_pct.clip(lower=self.percent * 0.5, upper=self.percent * 2.0)
        else:
            return pd.Series(self.percent, index=data.index)
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Price Envelopes for the given data.
        
        Args:
            data: DataFrame with OHLCV data
            
        Returns:
            DataFrame with Price Envelopes values
        """
        if self.column not in data.columns:
            raise ValueError(f"Data must contain '{self.column}' column")
            
        # Make a copy to avoid modifying the input data
        result = data.copy()
        
        # Calculate moving average
        result[self.name_ma] = self._calculate_ma(result[self.column], self.window, self.ma_method)
        
        # Calculate adaptive percentage if enabled
        if self.adaptive_percentage:
            percent_series = self._calculate_adaptive_percentage(data)
            result[self.name_adaptive] = percent_series
        else:
            percent_series = pd.Series(self.percent, index=data.index)
        
        # Calculate upper and lower envelopes
        envelope_factor = percent_series / 100
        result[self.name_upper] = result[self.name_ma] * (1 + envelope_factor)
        result[self.name_lower] = result[self.name_ma] * (1 - envelope_factor)
        
        return result